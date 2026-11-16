import math

import pytest
import torch

from flx.models.localization_network import LocalizationNetwork
from flx.setup import config


def _images(n: int = 4) -> torch.Tensor:
    # Background 0 with a bright square off the centre, like a binarized print
    images = torch.zeros(n, 1, 299, 299)
    images[:, :, 100:200, 120:220] = 1.0
    return images


def _set_raw_outputs(net: LocalizationNetwork, values: list[float]) -> None:
    """Makes the last layer return the given values for every image."""
    net.fc_loc[2].weight.data.zero_()
    net.fc_loc[2].bias.data = torch.tensor(values)


def test_default_bounds_are_those_of_the_deepprint_paper():
    net = LocalizationNetwork()
    assert config.LOCALIZATION_MAX_ROTATION_DEGREES == 60.0
    assert net.max_rotation == pytest.approx(math.radians(60))
    assert net.max_shift == 0.5  # 224 of 448 pixels
    assert net.max_scale_factor == 1.0 and not net.learns_scale
    assert net.fc_loc[2].out_features == 3


def test_new_network_is_the_identity():
    net = LocalizationNetwork(max_scale_factor=1.3).eval()
    images = torch.rand(3, 1, 299, 299)
    with torch.no_grad():
        aligned, alignment = net.align(images)
    assert (aligned - images).abs().max() < 1e-4
    assert alignment.rotation.abs().max() == 0
    assert alignment.shift.abs().max() == 0
    assert (alignment.scale == 1).all()
    assert (alignment.kept_area == 1).all()
    assert alignment.kept_content == pytest.approx(torch.ones(3), abs=1e-4)


@pytest.mark.parametrize("raw", [1e6, -1e6, 30.0, -7.5])
def test_outputs_never_leave_the_bounds(raw):
    net = LocalizationNetwork(
        max_rotation_degrees=60, max_shift=0.5, max_scale_factor=1.25
    ).eval()
    _set_raw_outputs(net, [raw] * 4)
    with torch.no_grad():
        aligned, alignment = net.align(_images())
    assert torch.isfinite(aligned).all()
    assert alignment.rotation.abs().max() <= math.radians(60) + 1e-6
    assert alignment.shift.abs().max() <= 0.5 + 1e-6
    assert alignment.scale.max() <= 1.25 + 1e-6
    assert alignment.scale.min() >= 1 / 1.25 - 1e-6
    # The matrix holds the same transform
    assert alignment.matrices[:, :, 2].abs().max() <= 1.0 + 1e-6
    assert alignment.matrices[:, :, :2].abs().max() <= 1.25 + 1e-6


def test_bounding_is_smooth_and_small_outputs_pass_almost_unchanged():
    net = LocalizationNetwork(max_rotation_degrees=60, max_shift=0.5).eval()
    _set_raw_outputs(net, [0.1, -0.1, 0.05])
    with torch.no_grad():
        _, alignment = net.align(_images(1))
    assert alignment.rotation.item() == pytest.approx(math.radians(60) * 0.1, rel=0.01)
    assert alignment.shift[0].tolist() == pytest.approx([-0.05, 0.025], rel=0.01)
    # The gradient does not vanish at a raw output of 3 (tanh, not a hard limit)
    net.fc_loc[2].bias.data = torch.tensor([3.0, 0.0, 0.0])
    _, alignment = net.align(_images(1))
    alignment.rotation.sum().backward()
    assert net.fc_loc[2].bias.grad[0].abs() > 1e-3


def test_shift_is_a_share_of_the_image_side():
    net = LocalizationNetwork(max_shift=0.5).eval()
    _set_raw_outputs(net, [0.0, math.atanh(0.5), 0.0])  # shift x = 0.25
    images = _images(1)
    with torch.no_grad():
        aligned, alignment = net.align(images)
    assert alignment.shift[0].tolist() == pytest.approx([0.25, 0.0], abs=1e-6)
    # The aligned image shows the input 0.25 * 299 pixels further right, so
    # the square appears that much further left
    columns = aligned[0, 0].sum(dim=0).nonzero().flatten()
    assert columns.min().item() == pytest.approx(120 - 0.25 * 299, abs=1.5)
    assert alignment.kept_area.item() == pytest.approx(0.75, abs=0.01)
    assert alignment.kept_content.item() == pytest.approx(1.0, abs=0.02)


def test_kept_content_reports_a_print_that_left_the_frame():
    net = LocalizationNetwork(max_shift=0.5).eval()
    # The square (columns 120 to 219) is moved out to the right
    _set_raw_outputs(net, [0.0, -10.0, 0.0])
    with torch.no_grad():
        aligned, alignment = net.align(_images(2))
    assert alignment.kept_area == pytest.approx(torch.full((2,), 0.5), abs=0.01)
    assert alignment.kept_content.max() < 0.35


def test_scale_output_only_exists_when_the_scale_may_change():
    net = LocalizationNetwork(max_scale_factor=1.25).eval()
    assert net.fc_loc[2].out_features == 4
    _set_raw_outputs(net, [0.0, 0.0, 0.0, 10.0])
    with torch.no_grad():
        aligned, alignment = net.align(_images(1))
    assert alignment.scale.item() == pytest.approx(1.25, abs=1e-4)
    # A scale above 1 shows more of the input, so the square gets smaller
    width = aligned[0, 0].sum(dim=0).gt(0.5).sum().item()
    assert width == pytest.approx(100 / 1.25, abs=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rotation_degrees": 0},
        {"max_rotation_degrees": 200},
        {"max_shift": 0},
        {"max_shift": 1.5},
        {"max_scale_factor": 0.9},
    ],
)
def test_bounds_must_make_sense(kwargs):
    with pytest.raises(ValueError):
        LocalizationNetwork(**kwargs)
