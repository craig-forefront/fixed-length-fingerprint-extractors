"""
Slap image segmentation module for isolating individual fingers from multi-finger slap captures.

This module provides methods to segment a 4-finger slap image into individual finger regions
that can be processed independently through the DeepPrint pipeline.
"""

from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as VTF

from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


@dataclass
class FingerROI:
    """Region of interest for a single finger extracted from a slap image."""

    x: int  # Left boundary
    y: int  # Top boundary
    width: int  # Width of ROI
    height: int  # Height of ROI
    finger_position: int  # 0=index, 1=middle, 2=ring, 3=pinky (left-to-right)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Returns (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    @property
    def slice(self) -> tuple[slice, slice]:
        """Returns slices for numpy array indexing: img[roi.slice]"""
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))


@dataclass
class SlapSegmentationResult:
    """Result of slap image segmentation."""

    finger_rois: list[FingerROI]  # Detected finger regions, sorted left-to-right
    finger_images: list[torch.Tensor]  # Extracted and normalized finger images
    original_shape: tuple[int, int]  # Original slap image shape (height, width)

    @property
    def num_fingers(self) -> int:
        return len(self.finger_rois)

    def get_finger_image(self, position: int) -> torch.Tensor:
        """Get normalized finger image by position (0=index, 1=middle, 2=ring, 3=pinky)."""
        for i, roi in enumerate(self.finger_rois):
            if roi.finger_position == position:
                return self.finger_images[i]
        raise ValueError(f"No finger found at position {position}")


class SlapSegmenter:
    """
    Segments multi-finger slap images into individual finger regions.

    Supports both 4-finger slaps (index through pinky) and can be configured
    for different expected finger counts.
    """

    # Minimum area ratio for a valid finger region (relative to expected finger area)
    MIN_AREA_RATIO = 0.15

    # Maximum area ratio for a valid finger region
    MAX_AREA_RATIO = 3.0

    # Minimum aspect ratio (height/width) for a valid finger
    MIN_ASPECT_RATIO = 1.2

    # Maximum aspect ratio (height/width) for a valid finger
    MAX_ASPECT_RATIO = 5.0

    def __init__(
        self,
        num_fingers: int = 4,
        margin_percent: float = 0.05,
        use_adaptive_threshold: bool = True,
    ):
        """
        Initialize the slap segmenter.

        Args:
            num_fingers: Expected number of fingers in the slap image (default: 4)
            margin_percent: Percentage of margin to add around detected finger ROIs
            use_adaptive_threshold: Use adaptive thresholding instead of Otsu's method
        """
        self.num_fingers = num_fingers
        self.margin_percent = margin_percent
        self.use_adaptive_threshold = use_adaptive_threshold

    def segment(
        self, slap_image: Union[np.ndarray, torch.Tensor]
    ) -> SlapSegmentationResult:
        """
        Segment a slap image into individual finger regions.

        Args:
            slap_image: Grayscale slap image as numpy array or torch tensor

        Returns:
            SlapSegmentationResult containing detected ROIs and normalized finger images
        """
        # Convert to numpy if needed
        if isinstance(slap_image, torch.Tensor):
            img = slap_image.squeeze().cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        else:
            img = slap_image.copy()
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

        original_shape = img.shape[:2]

        # Get binary mask of finger regions
        binary = self._create_binary_mask(img)

        # Find finger contours
        contours = self._find_finger_contours(binary, original_shape)

        # Extract ROIs from contours
        finger_rois = self._contours_to_rois(contours, original_shape)

        # Extract and normalize finger images
        finger_images = self._extract_finger_images(img, finger_rois)

        return SlapSegmentationResult(
            finger_rois=finger_rois,
            finger_images=finger_images,
            original_shape=original_shape,
        )

    def _create_binary_mask(self, img: np.ndarray) -> np.ndarray:
        """Create binary mask highlighting finger regions."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        if self.use_adaptive_threshold:
            # Adaptive thresholding works better for varying illumination
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=51,
                C=10,
            )
        else:
            # Otsu's method for uniform illumination
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Close small gaps within fingers
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove small noise regions
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return binary

    def _find_finger_contours(
        self, binary: np.ndarray, original_shape: tuple[int, int]
    ) -> list[np.ndarray]:
        """Find contours that represent individual fingers."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Calculate expected finger area
        total_area = original_shape[0] * original_shape[1]
        expected_finger_area = total_area / (self.num_fingers * 2)  # Fingers + gaps

        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Check area bounds
            min_area = expected_finger_area * self.MIN_AREA_RATIO
            max_area = expected_finger_area * self.MAX_AREA_RATIO
            if not (min_area < area < max_area):
                continue

            # Check aspect ratio (fingers are taller than wide)
            aspect_ratio = h / w if w > 0 else 0
            if not (self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO):
                continue

            valid_contours.append(contour)

        # Sort contours left-to-right by x-coordinate
        valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

        # If we have more contours than expected, take the ones with largest area
        if len(valid_contours) > self.num_fingers:
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            valid_contours = valid_contours[: self.num_fingers]
            valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

        return valid_contours

    def _contours_to_rois(
        self, contours: list[np.ndarray], original_shape: tuple[int, int]
    ) -> list[FingerROI]:
        """Convert contours to FingerROI objects with margins."""
        rois = []
        height, width = original_shape

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Add margin
            margin_x = int(w * self.margin_percent)
            margin_y = int(h * self.margin_percent)

            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(width - x, w + 2 * margin_x)
            h = min(height - y, h + 2 * margin_y)

            rois.append(
                FingerROI(x=x, y=y, width=w, height=h, finger_position=i)
            )

        return rois

    def _extract_finger_images(
        self, img: np.ndarray, rois: list[FingerROI]
    ) -> list[torch.Tensor]:
        """Extract and normalize individual finger images from ROIs."""
        finger_images = []

        for roi in rois:
            # Extract ROI from image
            finger_img = img[roi.slice]

            # Normalize to DeepPrint input size
            normalized = pad_and_resize_to_deepprint_input_size(finger_img, fill=1.0)
            finger_images.append(normalized)

        return finger_images


def segment_slap_projection(
    slap_image: Union[np.ndarray, torch.Tensor],
    num_fingers: int = 4,
    min_valley_distance: int = 50,
) -> list[tuple[int, int]]:
    """
    Alternative segmentation using vertical projection histogram.

    Finds valleys (low intensity columns) between fingers using the projection profile.
    This method is faster but may be less robust for images with touching fingers.

    Args:
        slap_image: Grayscale slap image
        num_fingers: Expected number of fingers
        min_valley_distance: Minimum pixel distance between detected valleys

    Returns:
        List of (start_x, end_x) column boundaries for each finger
    """
    # Convert to numpy if needed
    if isinstance(slap_image, torch.Tensor):
        img = slap_image.squeeze().cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        img = slap_image.copy()

    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate vertical projection (sum of each column)
    projection = np.sum(binary, axis=0).astype(np.float32)

    # Smooth the projection
    kernel_size = max(5, img.shape[1] // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    projection = cv2.GaussianBlur(projection.reshape(1, -1), (kernel_size, 1), 0).flatten()

    # Find valleys (local minima)
    # We need num_fingers - 1 valleys to separate num_fingers fingers
    valleys = []
    for i in range(min_valley_distance, len(projection) - min_valley_distance):
        # Check if this is a local minimum
        window = projection[i - min_valley_distance // 2 : i + min_valley_distance // 2]
        if projection[i] == np.min(window):
            valleys.append(i)

    # If we have more valleys than needed, keep the deepest ones
    if len(valleys) > num_fingers - 1:
        valley_depths = [(v, projection[v]) for v in valleys]
        valley_depths.sort(key=lambda x: x[1])
        valleys = sorted([v[0] for v in valley_depths[: num_fingers - 1]])

    # Convert valleys to finger boundaries
    boundaries = []
    prev_x = 0
    for valley in valleys:
        boundaries.append((prev_x, valley))
        prev_x = valley
    boundaries.append((prev_x, img.shape[1]))

    return boundaries


def visualize_segmentation(
    slap_image: Union[np.ndarray, torch.Tensor],
    result: SlapSegmentationResult,
    output_path: str = None,
) -> np.ndarray:
    """
    Visualize segmentation results by drawing bounding boxes on the slap image.

    Args:
        slap_image: Original slap image
        result: Segmentation result from SlapSegmenter
        output_path: Optional path to save the visualization

    Returns:
        BGR image with drawn bounding boxes
    """
    # Convert to numpy BGR for visualization
    if isinstance(slap_image, torch.Tensor):
        img = slap_image.squeeze().cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        img = slap_image.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # Convert to BGR for colored drawing
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Colors for each finger position
    colors = [
        (0, 255, 0),    # Green - index
        (255, 0, 0),    # Blue - middle
        (0, 0, 255),    # Red - ring
        (255, 255, 0),  # Cyan - pinky
    ]

    # Draw bounding boxes
    for roi in result.finger_rois:
        color = colors[roi.finger_position % len(colors)]
        cv2.rectangle(
            vis_img,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            color,
            2,
        )
        # Label
        label = f"F{roi.finger_position}"
        cv2.putText(
            vis_img,
            label,
            (roi.x + 5, roi.y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    if output_path:
        cv2.imwrite(output_path, vis_img)

    return vis_img
