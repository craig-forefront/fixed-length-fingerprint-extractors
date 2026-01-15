"""
Slap-aware localization network for multi-finger detection and alignment.

This module provides neural network architectures for detecting and localizing
multiple fingers within a single slap image, enabling end-to-end trainable
multi-finger fingerprint processing.
"""

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass
class SlapLocalizationOutput:
    """Output from slap localization network."""

    # Bounding boxes for each finger: (batch, num_fingers, 4) where 4 = (cx, cy, w, h)
    boxes: torch.Tensor

    # Confidence scores for each detected finger: (batch, num_fingers)
    confidence: torch.Tensor

    # Aligned finger images: (batch, num_fingers, 1, H, W)
    aligned_fingers: torch.Tensor = None

    # Rotation angles for each finger (optional): (batch, num_fingers)
    rotations: torch.Tensor = None


class SlapLocalizationNetwork(nn.Module):
    """
    Neural network for detecting and localizing multiple fingers in a slap image.

    Predicts bounding boxes and optional rotation for each finger, then extracts
    and aligns individual finger regions.
    """

    def __init__(
        self,
        num_fingers: int = 4,
        output_size: int = 299,
        predict_rotation: bool = True,
    ):
        """
        Initialize the slap localization network.

        Args:
            num_fingers: Number of fingers to detect (default: 4 for a standard slap)
            output_size: Output size for aligned finger images (default: 299 for DeepPrint)
            predict_rotation: Whether to predict rotation angle for each finger
        """
        super().__init__()
        self.num_fingers = num_fingers
        self.output_size = output_size
        self.predict_rotation = predict_rotation

        # Downsampled input size for efficiency
        self.input_size = (256, 512)  # Height x Width for typical slap aspect ratio
        self.resize = torchvision.transforms.Resize(
            size=self.input_size, antialias=True
        )

        # Feature extraction backbone
        self.features = nn.Sequential(
            # Initial conv block
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),  # Fixed output size
        )

        # Calculate feature dimension
        feature_dim = 512 * 4 * 8  # 16384

        # Box prediction head: (cx, cy, w, h) for each finger
        # Outputs normalized coordinates [0, 1]
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_fingers * 4),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_fingers),
            nn.Sigmoid(),
        )

        # Optional rotation prediction head
        if predict_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_fingers),
                nn.Tanh(),  # Output in [-1, 1], scaled to rotation range
            )
            self.max_rotation = torch.pi / 6  # ±30 degrees max rotation
        else:
            self.rotation_head = None

        # Initialize box predictions to reasonable defaults (evenly spaced fingers)
        self._init_box_weights()

    def _init_box_weights(self):
        """Initialize box prediction to reasonable default positions."""
        # Initialize bias to place fingers evenly across the image
        with torch.no_grad():
            bias = self.box_head[-2].bias
            for i in range(self.num_fingers):
                # cx: evenly spaced horizontally
                cx = (i + 0.5) / self.num_fingers
                # cy: centered vertically
                cy = 0.5
                # w: approximate finger width
                w = 0.8 / self.num_fingers
                # h: approximate finger height (taller than wide)
                h = 0.8

                idx = i * 4
                bias[idx] = cx
                bias[idx + 1] = cy
                bias[idx + 2] = w
                bias[idx + 3] = h

    def forward(
        self, x: torch.Tensor, extract_fingers: bool = True
    ) -> SlapLocalizationOutput:
        """
        Forward pass for slap localization.

        Args:
            x: Input slap image tensor (batch, 1, H, W)
            extract_fingers: Whether to extract and return aligned finger images

        Returns:
            SlapLocalizationOutput with detected boxes, confidence, and optionally aligned fingers
        """
        batch_size = x.shape[0]
        original_size = x.shape[-2:]  # (H, W)

        # Resize for feature extraction
        x_resized = self.resize(x)

        # Extract features
        features = self.features(x_resized)
        features_flat = features.view(batch_size, -1)

        # Predict boxes (normalized coordinates)
        boxes = self.box_head(features_flat)
        boxes = boxes.view(batch_size, self.num_fingers, 4)

        # Predict confidence
        confidence = self.confidence_head(features_flat)

        # Predict rotation if enabled
        rotations = None
        if self.rotation_head is not None:
            rotations = self.rotation_head(features_flat) * self.max_rotation

        # Extract aligned finger images if requested
        aligned_fingers = None
        if extract_fingers:
            aligned_fingers = self._extract_fingers(x, boxes, rotations)

        return SlapLocalizationOutput(
            boxes=boxes,
            confidence=confidence,
            aligned_fingers=aligned_fingers,
            rotations=rotations,
        )

    def _extract_fingers(
        self,
        x: torch.Tensor,
        boxes: torch.Tensor,
        rotations: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extract and align individual finger regions using spatial transformer.

        Args:
            x: Original slap image (batch, 1, H, W)
            boxes: Predicted boxes (batch, num_fingers, 4) with (cx, cy, w, h)
            rotations: Optional rotation angles (batch, num_fingers)

        Returns:
            Aligned finger images (batch, num_fingers, 1, output_size, output_size)
        """
        batch_size = x.shape[0]
        device = x.device

        aligned = []
        for finger_idx in range(self.num_fingers):
            # Get box parameters for this finger
            cx = boxes[:, finger_idx, 0]  # (batch,)
            cy = boxes[:, finger_idx, 1]
            w = boxes[:, finger_idx, 2]
            h = boxes[:, finger_idx, 3]

            # Convert from [0,1] to [-1,1] for grid_sample
            # grid_sample expects coordinates in [-1, 1]
            cx_grid = cx * 2 - 1
            cy_grid = cy * 2 - 1

            # Build affine transformation matrix
            if rotations is not None:
                theta = rotations[:, finger_idx]
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
            else:
                cos_t = torch.ones(batch_size, device=device)
                sin_t = torch.zeros(batch_size, device=device)

            # Scale factors (inverse because we're sampling from source)
            scale_x = w
            scale_y = h

            # Build 2x3 affine matrix
            # [scale_x * cos, -scale_y * sin, tx]
            # [scale_x * sin,  scale_y * cos, ty]
            affine = torch.zeros(batch_size, 2, 3, device=device)
            affine[:, 0, 0] = scale_x * cos_t
            affine[:, 0, 1] = -scale_y * sin_t
            affine[:, 0, 2] = cx_grid
            affine[:, 1, 0] = scale_x * sin_t
            affine[:, 1, 1] = scale_y * cos_t
            affine[:, 1, 2] = cy_grid

            # Generate sampling grid
            grid = F.affine_grid(
                affine, (batch_size, 1, self.output_size, self.output_size), align_corners=False
            )

            # Sample from original image
            finger_img = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
            aligned.append(finger_img)

        # Stack along new dimension: (batch, num_fingers, 1, H, W)
        return torch.stack(aligned, dim=1)

    def get_finger_rois(
        self, boxes: torch.Tensor, image_size: tuple[int, int]
    ) -> list[list[tuple[int, int, int, int]]]:
        """
        Convert normalized boxes to pixel ROIs.

        Args:
            boxes: Predicted boxes (batch, num_fingers, 4)
            image_size: Original image size (H, W)

        Returns:
            List of ROI tuples (x, y, width, height) for each image in batch
        """
        batch_size = boxes.shape[0]
        H, W = image_size

        all_rois = []
        for b in range(batch_size):
            rois = []
            for f in range(self.num_fingers):
                cx, cy, w, h = boxes[b, f].cpu().numpy()

                # Convert to pixel coordinates
                x = int((cx - w / 2) * W)
                y = int((cy - h / 2) * H)
                width = int(w * W)
                height = int(h * H)

                # Clip to image bounds
                x = max(0, min(x, W - 1))
                y = max(0, min(y, H - 1))
                width = min(width, W - x)
                height = min(height, H - y)

                rois.append((x, y, width, height))
            all_rois.append(rois)

        return all_rois


class SlapLocalizationLoss(nn.Module):
    """
    Loss function for training slap localization network.

    Combines box regression loss, confidence loss, and optional rotation loss.
    """

    def __init__(
        self,
        box_weight: float = 1.0,
        confidence_weight: float = 0.5,
        rotation_weight: float = 0.3,
        iou_loss: bool = True,
    ):
        """
        Initialize the loss function.

        Args:
            box_weight: Weight for box regression loss
            confidence_weight: Weight for confidence prediction loss
            rotation_weight: Weight for rotation prediction loss
            iou_loss: Use IoU-based loss for boxes (more stable than L1/L2)
        """
        super().__init__()
        self.box_weight = box_weight
        self.confidence_weight = confidence_weight
        self.rotation_weight = rotation_weight
        self.iou_loss = iou_loss

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_confidence: torch.Tensor,
        pred_rotations: torch.Tensor,
        target_boxes: torch.Tensor,
        target_confidence: torch.Tensor,
        target_rotations: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            pred_boxes: Predicted boxes (batch, num_fingers, 4)
            pred_confidence: Predicted confidence (batch, num_fingers)
            pred_rotations: Predicted rotations (batch, num_fingers) or None
            target_boxes: Ground truth boxes (batch, num_fingers, 4)
            target_confidence: Ground truth confidence (batch, num_fingers)
            target_rotations: Ground truth rotations or None

        Returns:
            Total loss scalar
        """
        # Box loss
        if self.iou_loss:
            box_loss = self._giou_loss(pred_boxes, target_boxes, target_confidence)
        else:
            box_loss = F.smooth_l1_loss(
                pred_boxes * target_confidence.unsqueeze(-1),
                target_boxes * target_confidence.unsqueeze(-1),
            )

        # Confidence loss (binary cross entropy)
        conf_loss = F.binary_cross_entropy(pred_confidence, target_confidence)

        # Rotation loss (if applicable)
        rot_loss = torch.tensor(0.0, device=pred_boxes.device)
        if pred_rotations is not None and target_rotations is not None:
            # Only compute loss for fingers with high confidence
            mask = target_confidence > 0.5
            if mask.sum() > 0:
                rot_loss = F.smooth_l1_loss(
                    pred_rotations[mask], target_rotations[mask]
                )

        total_loss = (
            self.box_weight * box_loss
            + self.confidence_weight * conf_loss
            + self.rotation_weight * rot_loss
        )

        return total_loss

    def _giou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized IoU loss for boxes."""
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-7)

        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

        # Loss (weighted by confidence)
        loss = (1 - giou) * confidence
        return loss.mean()


class DeepPrint_SlapLoc(nn.Module):
    """
    End-to-end model for processing slap images through localization and DeepPrint.

    Combines SlapLocalizationNetwork with DeepPrint to process multiple fingers
    from a single slap image in one forward pass.
    """

    def __init__(
        self,
        deepprint_model: nn.Module,
        num_fingers: int = 4,
        predict_rotation: bool = True,
    ):
        """
        Initialize the slap processing model.

        Args:
            deepprint_model: Pre-trained or new DeepPrint model for single fingers
            num_fingers: Number of fingers to detect per slap
            predict_rotation: Whether to predict and correct finger rotation
        """
        super().__init__()
        self.num_fingers = num_fingers
        self.localization = SlapLocalizationNetwork(
            num_fingers=num_fingers,
            output_size=299,  # DeepPrint input size
            predict_rotation=predict_rotation,
        )
        self.deepprint = deepprint_model

    def forward(self, slap_image: torch.Tensor) -> dict:
        """
        Process a slap image to extract embeddings for all fingers.

        Args:
            slap_image: Slap image tensor (batch, 1, H, W)

        Returns:
            Dictionary with:
                - 'localization': SlapLocalizationOutput
                - 'embeddings': List of DeepPrintOutput for each finger
                - 'finger_images': Aligned finger images (batch, num_fingers, 1, 299, 299)
        """
        batch_size = slap_image.shape[0]

        # Localize fingers
        loc_output = self.localization(slap_image, extract_fingers=True)

        # Process each finger through DeepPrint
        embeddings = []
        for finger_idx in range(self.num_fingers):
            # Get aligned finger image: (batch, 1, 299, 299)
            finger_img = loc_output.aligned_fingers[:, finger_idx]

            # Process through DeepPrint
            finger_output = self.deepprint(finger_img)
            embeddings.append(finger_output)

        return {
            "localization": loc_output,
            "embeddings": embeddings,
            "finger_images": loc_output.aligned_fingers,
        }

    def extract_embeddings(self, slap_image: torch.Tensor) -> torch.Tensor:
        """
        Extract and concatenate embeddings for all fingers.

        Args:
            slap_image: Slap image tensor (batch, 1, H, W)

        Returns:
            Concatenated embeddings (batch, num_fingers * embedding_dim)
        """
        output = self.forward(slap_image)

        # Collect texture embeddings (or minutia, depending on model)
        all_embeddings = []
        for finger_output in output["embeddings"]:
            if finger_output.texture_embeddings is not None:
                all_embeddings.append(finger_output.texture_embeddings)
            elif finger_output.minutia_embeddings is not None:
                all_embeddings.append(finger_output.minutia_embeddings)

        # Concatenate all finger embeddings
        return torch.cat(all_embeddings, dim=1)


def create_slap_training_targets(
    finger_rois: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    num_fingers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create training targets from ground truth finger ROIs.

    Args:
        finger_rois: List of (x, y, width, height) tuples for each finger
        image_size: Original image size (H, W)
        num_fingers: Expected number of fingers

    Returns:
        Tuple of (target_boxes, target_confidence)
    """
    H, W = image_size

    boxes = torch.zeros(num_fingers, 4)
    confidence = torch.zeros(num_fingers)

    for i, roi in enumerate(finger_rois[:num_fingers]):
        x, y, w, h = roi

        # Convert to normalized (cx, cy, w, h)
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H

        boxes[i] = torch.tensor([cx, cy, nw, nh])
        confidence[i] = 1.0

    return boxes, confidence
