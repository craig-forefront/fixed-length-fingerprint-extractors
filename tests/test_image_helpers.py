import numpy as np
import pytest

from flx.data.image_helpers import (
    pad_and_resize_to_deepprint_input_size,
    transform_from_input_size,
    transform_to_input_size,
)
from flx.setup.config import INPUT_SIZE

# (height, width); in some of them height and width differ by an odd number
SHAPES = [(392, 292), (393, 292), (425, 256), (300, 401), (544, 352), (300, 300)]


@pytest.mark.parametrize("shape", SHAPES)
def test_pad_and_resize_to_deepprint_input_size(shape):
    image = np.zeros(shape, dtype=np.uint8)
    out = pad_and_resize_to_deepprint_input_size(image, fill=1.0)
    assert tuple(out.shape) == (1, INPUT_SIZE, INPUT_SIZE)


@pytest.mark.parametrize("shape", SHAPES)
def test_transform_to_input_size_follows_the_image(shape):
    """
    A minutia must end up at the place where the image transformation
    puts the pixels around it.
    """
    height, width = shape
    points = np.array(
        [(40, 50), (width - 60, 70), (width // 2, height // 2), (55, height - 45)]
    )
    transformed = transform_to_input_size(
        points.astype(np.float32), original_height=height, original_width=width
    )
    for (x, y), (expected_x, expected_y) in zip(points, transformed):
        image = np.zeros(shape, dtype=np.uint8)
        image[y - 4 : y + 5, x - 4 : x + 5] = 255  # blob centered on pixel (x, y)
        out = pad_and_resize_to_deepprint_input_size(image)[0].numpy().astype(float)
        ys, xs = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
        # Pixel k covers [k, k + 1), its center is at k + 0.5
        center_x = (out * xs).sum() / out.sum() + 0.5
        center_y = (out * ys).sum() / out.sum() + 0.5
        scale = INPUT_SIZE / max(shape)
        assert center_x == pytest.approx(float(expected_x) + 0.5 * scale, abs=0.2)
        assert center_y == pytest.approx(float(expected_y) + 0.5 * scale, abs=0.2)


def test_transform_from_input_size_is_not_rounded_to_half_precision():
    # The minutia map loaders return the points of transform_to_input_size as
    # float16, which has a resolution of one pixel for values above 1024
    height, width = 2000, 1500
    points = np.array([(100.5, 200.25), (230.125, 40.0)], dtype=np.float16)
    original = transform_from_input_size(points, height, width)
    expected = points.astype(float) * (height / INPUT_SIZE) - np.array([250, 0])
    assert original == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("shape", SHAPES)
def test_transform_from_input_size_inverts_transform_to_input_size(shape):
    height, width = shape
    points = np.array([(40, 50), (width - 60, 70), (55, height - 45)], dtype=float)
    transformed = transform_to_input_size(points, height, width)
    # transform_to_input_size returns float16
    assert transform_from_input_size(transformed, height, width) == pytest.approx(
        points, abs=0.5
    )
