"""
Data loader for multi-finger slap images.

This module provides loaders that segment slap images into individual fingers
and present them as separate samples for the fingerprint processing pipeline.
"""

from abc import abstractmethod
from typing import Callable, Union
import os

import cv2
import torch

from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.file_index import FileIndex
from flx.image_processing.slap_segmentation import SlapSegmenter, SlapSegmentationResult


class SlapIdentifier(Identifier):
    """
    Extended identifier for slap-segmented fingerprints.

    Adds finger_position to track which finger from the slap this sample represents,
    and slap_id to track which slap capture session it came from.
    """

    def __init__(
        self,
        subject: int,
        impression: int,
        finger_position: int,
        slap_id: int = 0,
    ):
        super().__init__(subject, impression)
        self.finger_position: int = int(finger_position)
        self.slap_id: int = int(slap_id)

    def __hash__(self):
        return hash((self.subject, self.impression, self.finger_position, self.slap_id))

    def __str__(self) -> str:
        return (
            f"SlapIdentifier(subject={self.subject}, impression={self.impression}, "
            f"finger={self.finger_position}, slap={self.slap_id})"
        )


class SlapImageLoader(DataLoader):
    """
    Base class for loading and segmenting slap images.

    Slap images contain multiple fingers captured simultaneously. This loader
    segments each slap into individual fingers and provides them as separate
    samples with proper identifiers.

    The identifier scheme encodes:
    - subject: The person (can be encoded as person * num_fingers + finger_position)
    - impression: The capture session / slap impression
    - finger_position: Which finger from the slap (0=index through 3=pinky)
    """

    def __init__(
        self,
        root_dir: str,
        num_fingers: int = 4,
        encode_finger_in_subject: bool = True,
    ):
        """
        Initialize the slap image loader.

        Args:
            root_dir: Root directory containing slap images
            num_fingers: Number of fingers expected per slap (default: 4)
            encode_finger_in_subject: If True, encodes finger position into subject ID
                (subject = person * num_fingers + finger). If False, keeps person as
                subject and tracks finger separately in SlapIdentifier.
        """
        self._num_fingers = num_fingers
        self._encode_finger_in_subject = encode_finger_in_subject
        self._segmenter = SlapSegmenter(num_fingers=num_fingers)

        # Build file index for slap images
        self._files: FileIndex = FileIndex(
            root_dir, self._extension(), self._file_to_slap_id
        )

        # Build expanded identifier set (one per finger per slap)
        self._ids, self._id_to_slap_id = self._build_expanded_ids()

        # Cache for segmentation results
        self._segmentation_cache: dict[Identifier, SlapSegmentationResult] = {}

    def _build_expanded_ids(self) -> tuple[IdentifierSet, dict[Identifier, Identifier]]:
        """
        Build expanded identifier set with one entry per finger.

        Returns:
            Tuple of (IdentifierSet with all finger IDs, mapping from finger ID to slap ID)
        """
        expanded_ids = []
        id_to_slap_id = {}

        for slap_id in self._files.ids.identifiers:
            for finger_pos in range(self._num_fingers):
                if self._encode_finger_in_subject:
                    # Encode: subject = person * num_fingers + finger_position
                    finger_subject = slap_id.subject * self._num_fingers + finger_pos
                    finger_id = Identifier(finger_subject, slap_id.impression)
                else:
                    # Use extended SlapIdentifier
                    finger_id = SlapIdentifier(
                        subject=slap_id.subject,
                        impression=slap_id.impression,
                        finger_position=finger_pos,
                        slap_id=slap_id.impression,
                    )
                expanded_ids.append(finger_id)
                id_to_slap_id[finger_id] = slap_id

        return IdentifierSet(expanded_ids), id_to_slap_id

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    @property
    def num_fingers(self) -> int:
        return self._num_fingers

    def get(self, identifier: Identifier) -> torch.Tensor:
        """
        Get a single finger image by its identifier.

        Args:
            identifier: Finger identifier (either encoded subject or SlapIdentifier)

        Returns:
            Normalized finger image tensor ready for DeepPrint
        """
        # Find the corresponding slap ID
        slap_id = self._id_to_slap_id[identifier]

        # Get or compute segmentation
        if slap_id not in self._segmentation_cache:
            slap_path = self._files.get(slap_id)
            slap_img = self._load_slap_image(slap_path)
            self._segmentation_cache[slap_id] = self._segmenter.segment(slap_img)

        segmentation = self._segmentation_cache[slap_id]

        # Determine finger position
        if isinstance(identifier, SlapIdentifier):
            finger_pos = identifier.finger_position
        else:
            # Decode from subject: finger_pos = subject % num_fingers
            finger_pos = identifier.subject % self._num_fingers

        # Return the finger image
        if finger_pos < len(segmentation.finger_images):
            return segmentation.finger_images[finger_pos]
        else:
            raise ValueError(
                f"Finger position {finger_pos} not found in segmentation "
                f"(only {len(segmentation.finger_images)} fingers detected)"
            )

    def get_slap_segmentation(self, slap_id: Identifier) -> SlapSegmentationResult:
        """
        Get the full segmentation result for a slap image.

        Useful for debugging or visualization.
        """
        if slap_id not in self._segmentation_cache:
            slap_path = self._files.get(slap_id)
            slap_img = self._load_slap_image(slap_path)
            self._segmentation_cache[slap_id] = self._segmenter.segment(slap_img)
        return self._segmentation_cache[slap_id]

    def clear_cache(self):
        """Clear the segmentation cache to free memory."""
        self._segmentation_cache.clear()

    @staticmethod
    @abstractmethod
    def _extension() -> str:
        """File extension for slap images."""
        pass

    @staticmethod
    @abstractmethod
    def _file_to_slap_id(subdir: str, filename: str) -> Identifier:
        """Convert file path to slap identifier (one ID per slap image)."""
        pass

    @staticmethod
    def _load_slap_image(filepath: str) -> torch.Tensor:
        """Load a slap image from disk."""
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load slap image: {filepath}")
        return img


class GenericSlapLoader(SlapImageLoader):
    """
    Generic slap loader with configurable file naming convention.

    File naming pattern: <person>_<impression>.<ext>
    Example: 001_01.png -> person 1, impression 1
    """

    def __init__(
        self,
        root_dir: str,
        extension: str = ".png",
        num_fingers: int = 4,
        encode_finger_in_subject: bool = True,
    ):
        self._ext = extension if extension.startswith(".") else f".{extension}"
        super().__init__(root_dir, num_fingers, encode_finger_in_subject)

    @staticmethod
    def _extension() -> str:
        # This will be overridden by instance
        return ".png"

    def _extension(self) -> str:
        return self._ext

    @staticmethod
    def _file_to_slap_id(subdir: str, filename: str) -> Identifier:
        # Pattern: <person>_<impression>
        parts = filename.split("_")
        if len(parts) >= 2:
            person = int(parts[0])
            impression = int(parts[1])
        else:
            # Single number = person, impression 0
            person = int(parts[0])
            impression = 0
        return Identifier(person, impression)


class NISTSlapLoader(SlapImageLoader):
    """
    Loader for NIST-style slap datasets (e.g., SD 29, SD 302).

    File naming pattern: <person>_<hand>_<impression>.<ext>
    Where hand is 'L' for left or 'R' for right.
    """

    @staticmethod
    def _extension() -> str:
        return ".png"

    @staticmethod
    def _file_to_slap_id(subdir: str, filename: str) -> Identifier:
        # Pattern: <person>_<hand>_<impression>
        # Example: 00001_R_01.png
        parts = filename.split("_")
        person = int(parts[0])
        hand = parts[1] if len(parts) > 1 else "R"
        impression = int(parts[2]) if len(parts) > 2 else 0

        # Encode hand in impression: even=right, odd=left
        hand_offset = 0 if hand.upper() == "R" else 1
        combined_impression = impression * 2 + hand_offset

        return Identifier(person, combined_impression)


class FVC2000SlapLoader(SlapImageLoader):
    """
    Loader for FVC-style slap datasets.

    File naming pattern: <person>_<impression>.<ext>
    """

    @staticmethod
    def _extension() -> str:
        return ".tif"

    @staticmethod
    def _file_to_slap_id(subdir: str, filename: str) -> Identifier:
        # Pattern: <person>_<impression>
        parts = filename.split("_")
        person = int(parts[0])
        impression = int(parts[1]) if len(parts) > 1 else 0
        return Identifier(person - 1, impression - 1)  # 0-indexed


def create_slap_loader(
    root_dir: str,
    file_pattern: str = "generic",
    extension: str = ".png",
    num_fingers: int = 4,
) -> SlapImageLoader:
    """
    Factory function to create appropriate slap loader.

    Args:
        root_dir: Root directory containing slap images
        file_pattern: Naming pattern - "generic", "nist", or "fvc"
        extension: File extension
        num_fingers: Expected number of fingers per slap

    Returns:
        Configured SlapImageLoader instance
    """
    if file_pattern.lower() == "nist":
        return NISTSlapLoader(root_dir, num_fingers)
    elif file_pattern.lower() == "fvc":
        return FVC2000SlapLoader(root_dir, num_fingers)
    else:
        return GenericSlapLoader(root_dir, extension, num_fingers)
