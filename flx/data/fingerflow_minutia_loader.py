"""
Custom MinutiaMapLoader for minutiae extracted with tourmii/minutiae_extractor (FingerFlow).

This loader reads .mnt files created by scripts/extract_minutiae.py and converts
them to minutia maps for DeepPrint training.

.mnt file format:
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians>
"""

import os
from pathlib import Path

import numpy as np

from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.minutia_map_loader import MinutiaMapLoader
from flx.data.file_index import FileIndex
from flx.data.image_helpers import transform_to_input_size


def read_mnt_file(filepath: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Read a .mnt file created by extract_minutiae.py.

    Args:
        filepath: Path to .mnt file

    Returns:
        (locations, orientations, (width, height))
        - locations: np.ndarray of shape (N, 2) with (x, y) coordinates
        - orientations: np.ndarray of shape (N,) with angles in radians
        - (width, height): Original image dimensions
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return np.array([]).reshape(0, 2), np.array([]), (0, 0)

    # Parse header
    num_minutiae = int(lines[0].strip())
    width, height = map(int, lines[1].strip().split())

    if num_minutiae == 0:
        return np.array([]).reshape(0, 2), np.array([]), (width, height)

    # Parse minutiae
    locs = []
    oris = []
    for line in lines[2:2 + num_minutiae]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, angle = float(parts[0]), float(parts[1]), float(parts[2])
            locs.append((x, y))
            oris.append(angle)

    return (
        np.array(locs, dtype=np.float32),
        np.array(oris, dtype=np.float32),
        (width, height)
    )


class FingerFlowMinutiaMapLoader(MinutiaMapLoader):
    """
    Loader for minutiae extracted using FingerFlow (tourmii/minutiae_extractor).

    Reads .mnt files and converts them to minutia maps for DeepPrint training.

    Expected file naming convention: {subject_id}_{impression_id}.mnt
    Modify _file_to_id_fun() if your naming convention differs.
    """

    def __init__(
        self,
        root_dir: str,
        file_pattern: str = "{subject}_{impression}.mnt",
    ):
        """
        Args:
            root_dir: Directory containing .mnt files
            file_pattern: Not used currently, kept for future extensibility
        """
        self._root_dir = root_dir

        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <subject_id>_<impression_id>.mnt
            # Modify this if your naming convention differs
            name = os.path.splitext(filename)[0]  # Remove .mnt extension
            parts = name.split("_")
            if len(parts) >= 2:
                subject_id = int(parts[0])
                impression_id = int(parts[1])
            else:
                # Fallback: try to parse as single number
                subject_id = int(name)
                impression_id = 0
            # Convert to 0-indexed for PyTorch compatibility
            return Identifier(subject_id - 1, impression_id - 1)

        self._files = FileIndex(root_dir, ".mnt", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        Load minutiae for a fingerprint sample.

        Returns:
            (locations, orientations)
            - locations: np.ndarray of shape (N, 2) with (x, y) coordinates
            - orientations: np.ndarray of shape (N,) with angles in radians
        """
        filepath = self._files.get(identifier)
        locs, oris, (width, height) = read_mnt_file(filepath)

        if len(locs) == 0:
            return locs, oris

        # Transform coordinates to DeepPrint input size
        # The transform_to_input_size function scales coordinates from
        # original image size to the model's expected input size
        transformed_locs = transform_to_input_size(
            locs,
            original_height=height,
            original_width=width,
        )

        return transformed_locs, oris


class FingerFlowMinutiaMapLoaderCustomNaming(MinutiaMapLoader):
    """
    Same as FingerFlowMinutiaMapLoader but with customizable file naming.

    Use this if your files don't follow the {subject}_{impression}.mnt pattern.
    """

    def __init__(
        self,
        root_dir: str,
        file_to_id_fun: callable,
    ):
        """
        Args:
            root_dir: Directory containing .mnt files
            file_to_id_fun: Function that takes (subdir, filename) and returns Identifier
        """
        self._root_dir = root_dir
        self._files = FileIndex(root_dir, ".mnt", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """Load minutiae for a fingerprint sample."""
        filepath = self._files.get(identifier)
        locs, oris, (width, height) = read_mnt_file(filepath)

        if len(locs) == 0:
            return locs, oris

        transformed_locs = transform_to_input_size(
            locs,
            original_height=height,
            original_width=width,
        )

        return transformed_locs, oris


def main():
    """Test the loader with sample data."""
    import tempfile
    import os

    # Create a test .mnt file
    with tempfile.TemporaryDirectory() as tmpdir:
        mnt_path = os.path.join(tmpdir, "1_1.mnt")
        with open(mnt_path, 'w') as f:
            f.write("3\n")  # 3 minutiae
            f.write("400 400\n")  # image size
            f.write("100 150 0.5\n")
            f.write("200 250 1.57\n")
            f.write("300 350 3.14\n")

        # Test reading
        locs, oris, size = read_mnt_file(mnt_path)
        print(f"Loaded {len(locs)} minutiae from test file")
        print(f"Locations:\n{locs}")
        print(f"Orientations: {oris}")
        print(f"Image size: {size}")

        # Test loader
        loader = FingerFlowMinutiaMapLoader(tmpdir)
        print(f"\nLoader IDs: {loader.ids}")

        identifier = Identifier(0, 0)  # 0-indexed (maps to 1_1.mnt)
        minu_map, weight = loader.get(identifier)
        print(f"Minutia map shape: {minu_map.shape}")
        print(f"Weight: {weight}")


if __name__ == "__main__":
    main()
