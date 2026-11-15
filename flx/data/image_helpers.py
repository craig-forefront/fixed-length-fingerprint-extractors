from typing import Union

import numpy as np
import torch
import torchvision.transforms.functional as VTF

from flx.setup.config import INPUT_SIZE


def get_input_resolution() -> tuple[int, int]:
    return (INPUT_SIZE, INPUT_SIZE)


def pad_and_resize(
    img: Union[np.ndarray, torch.Tensor],
    target_size: tuple[int, int] = None,
    fill: float = 0.0,
) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = VTF.to_tensor(img)

    height = img.shape[1]
    width = img.shape[2]
    # If the difference is odd, the additional row or column goes to the bottom
    # or right, see also transform_to_input_size.
    pad_width = max(height - width, 0)
    pad_height = max(width - height, 0)
    img = VTF.pad(
        img,
        padding=(
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        ),
        fill=fill,
    )  # left, top, right, bottom

    assert img.shape[1] == img.shape[2]

    return VTF.resize(img, target_size, antialias=True)


def pad_and_resize_to_deepprint_input_size(
    img: Union[np.ndarray, torch.Tensor],
    roi: Union[None, tuple[int, int]] = None,
    fill: float = 0.0,
) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = VTF.to_tensor(img)

    if roi is not None:
        img = VTF.center_crop(img, roi)

    return pad_and_resize(img, (INPUT_SIZE, INPUT_SIZE), fill=fill)


def _center_crop_offset(size: int, crop_size: int) -> int:
    """
    Index of the first row or column that torchvision's center_crop keeps;
    negative if it pads because the crop is larger than the image.
    """
    if crop_size > size:
        return -((crop_size - size) // 2)
    # As in center_crop: for an odd difference, Python rounds the half to the even number
    return int(round((size - crop_size) / 2.0))


def transform_to_input_size(
    minutia_points: np.ndarray,
    original_height: int,
    original_width: int,
    roi: Union[None, tuple[int, int]] = None,
) -> np.ndarray:
    """
    Transforms the pixel coordinates in the same way that the pixels in the original image would be
    transformed by pad_and_resize_to_deepprint_input_size.
    """
    minutia_points = minutia_points.astype(np.float16)

    if minutia_points.shape[0] == 0:
        return minutia_points

    if roi is not None:
        minutia_points -= np.array(
            [
                _center_crop_offset(original_width, roi[1]),
                _center_crop_offset(original_height, roi[0]),
            ]
        )
        height = roi[0]
        width = roi[1]
    else:
        height = original_height
        width = original_width

    # Padding on the left and top, as in pad_and_resize
    pad_width = max(height - width, 0) // 2
    pad_height = max(width - height, 0) // 2

    minutia_points += np.array([pad_width, pad_height])
    minutia_points *= INPUT_SIZE / max(height, width)
    return minutia_points


def transform_from_input_size(
    minutia_points: np.ndarray,
    original_height: int,
    original_width: int,
) -> np.ndarray:
    """
    Inverse of transform_to_input_size (without roi).
    """
    pad_width = max(original_height - original_width, 0) // 2
    pad_height = max(original_width - original_height, 0) // 2

    # Not in the float16 of transform_to_input_size, it is too coarse for pixels of large images
    minutia_points = minutia_points.astype(float)
    minutia_points *= max(original_height, original_width) / INPUT_SIZE
    return minutia_points - np.array([pad_width, pad_height])
