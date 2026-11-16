"""
The target minutia map follows the learned alignment (Loc* variants).
"""

import math

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as VTF

from flx.data.minutia_map import create_minutia_map
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.deep_print_loss import (
    DeepPrintLoss_Minu,
    DeepPrintLoss_TexMinu,
    LossWeights,
    align_minutia_maps,
)
from flx.models.localization_network import Alignment

MAP_SIZE = 128
INPUT_SIZE = 299


def _render(locs: np.ndarray, oris: np.ndarray) -> torch.Tensor:
    return VTF.to_tensor(
        create_minutia_map(
            locs,
            oris,
            in_resolution=(INPUT_SIZE, INPUT_SIZE),
            out_resolution=(MAP_SIZE, MAP_SIZE),
            n_layers=6,
            sigma=1.5,
        )
    ).float()


def _matrix(degrees: float, shift_x: float, shift_y: float) -> torch.Tensor:
    theta = math.radians(degrees)
    return torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 2 * shift_x],
            [math.sin(theta), math.cos(theta), 2 * shift_y],
        ]
    ).unsqueeze(0)


def _minutiae_in_aligned_frame(locs, oris, degrees, shift_x, shift_y):
    """
    Exact positions and directions of the minutiae in the aligned image: the
    point p of the aligned image shows the point R p + t of the input.
    """
    theta = math.radians(degrees)
    # Pixel to grid coordinates (-1 .. 1) of the map, and back
    grid = (locs * MAP_SIZE / INPUT_SIZE + 0.5) / (MAP_SIZE / 2) - 1
    grid = grid - 2 * np.array([shift_x, shift_y])
    back = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    grid = grid @ back.T
    moved = ((grid + 1) * (MAP_SIZE / 2) - 0.5) * INPUT_SIZE / MAP_SIZE
    return moved, oris - theta


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def _minutiae(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Around the centre, so that none leaves the frame
    locs = rng.uniform(90, 210, size=(n, 2))
    oris = rng.uniform(0, 2 * math.pi, size=n)
    return locs, oris


def _alignment(matrices: torch.Tensor) -> Alignment:
    n = matrices.shape[0]
    return Alignment(
        matrices=matrices,
        rotation=torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0]),
        shift=matrices[:, :, 2] / 2,
        scale=torch.ones(n),
        kept_area=torch.ones(n),
        kept_content=torch.ones(n),
    )


def test_identity_leaves_the_map_as_it_is():
    maps = _render(*_minutiae()).unsqueeze(0)
    assert (align_minutia_maps(maps, _matrix(0, 0, 0)) - maps).abs().max() < 1e-5


@pytest.mark.parametrize(
    "degrees, shift_x, shift_y",
    [(20, 0.0, 0.0), (-45, 0.05, -0.1), (60, 0.1, 0.1), (-60, 0.0, 0.05), (7, -0.1, 0.0)],
)
def test_moved_map_is_close_to_a_map_drawn_from_the_moved_minutiae(
    degrees, shift_x, shift_y
):
    locs, oris = _minutiae()
    moved = align_minutia_maps(
        _render(locs, oris).unsqueeze(0), _matrix(degrees, shift_x, shift_y)
    )[0]
    drawn = _render(*_minutiae_in_aligned_frame(locs, oris, degrees, shift_x, shift_y))
    assert _correlation(moved, drawn) > 0.9
    # The unmoved map is not a usable target for the aligned image
    assert _correlation(_render(locs, oris), drawn) < 0.5


def test_directions_turn_with_the_image():
    # One minutia in the centre that points to the right (channel 0). After a
    # rotation by +60 degrees its direction is -60 degrees, which is channel 5.
    maps = _render(np.array([[149.0, 149.0]]), np.array([0.0])).unsqueeze(0)
    assert maps[0].flatten(1).sum(dim=1).argmax() == 0
    moved = align_minutia_maps(maps, _matrix(60, 0, 0))
    assert moved[0].flatten(1).sum(dim=1).argmax() == 5


def test_each_sample_of_a_batch_gets_its_own_transform():
    maps = torch.stack([_render(*_minutiae(seed=s)) for s in range(3)])
    matrices = torch.cat([_matrix(30, 0.1, 0), _matrix(0, 0, 0), _matrix(-50, 0, -0.1)])
    batch = align_minutia_maps(maps, matrices)
    for i in range(3):
        single = align_minutia_maps(maps[i : i + 1], matrices[i : i + 1])[0]
        assert (batch[i] - single).abs().max() < 1e-5


def test_moved_target_carries_no_gradient():
    maps = _render(*_minutiae()).unsqueeze(0)
    matrices = _matrix(10, 0.05, 0).requires_grad_()
    assert not align_minutia_maps(maps, matrices).requires_grad


def _output(predicted_maps: torch.Tensor, alignment: Alignment | None):
    n = predicted_maps.shape[0]
    embeddings = torch.nn.functional.normalize(torch.ones(n, 4), dim=1)
    logits = torch.zeros(n, 3)
    return DeepPrintTrainingOutput(
        minutia_logits=logits,
        texture_logits=logits,
        minutia_maps=predicted_maps,
        minutia_embeddings=embeddings,
        texture_embeddings=embeddings,
        alignment=alignment,
    )


def _map_loss(loss_fun, predicted_maps, target_maps, alignment, weights=None) -> float:
    n = predicted_maps.shape[0]
    if weights is None:
        weights = torch.ones(n)
    loss_fun.reset_recorded_loss()
    loss_fun(
        _output(predicted_maps, alignment),
        torch.zeros(n, dtype=torch.long),
        target_maps,
        weights,
    )
    return loss_fun.get_recorded_loss()["minutia_map_loss"]


@pytest.mark.parametrize("loss_class", [DeepPrintLoss_Minu, DeepPrintLoss_TexMinu])
def test_map_loss_compares_with_the_target_in_the_aligned_frame(loss_class):
    args = (3, 4) if loss_class is DeepPrintLoss_Minu else (3, 4, 4)
    loss_fun = loss_class(*args, weights=LossWeights(minutia_map=1.0))
    locs, oris = _minutiae()
    target = _render(locs, oris).unsqueeze(0)
    matrices = _matrix(40, 0.1, -0.05)
    # A model that draws exactly the minutiae of the aligned image
    perfect = _render(*_minutiae_in_aligned_frame(locs, oris, 40, 0.1, -0.05)).unsqueeze(0)
    empty = torch.zeros_like(target)

    with_alignment = _map_loss(loss_fun, perfect, target, _alignment(matrices))
    without_alignment = _map_loss(loss_fun, perfect, target, None)
    empty_prediction = _map_loss(loss_fun, empty, target, _alignment(matrices))
    # The perfect model is far better than drawing nothing, which it would
    # not be against the target in the frame of the input
    assert with_alignment < 0.25 * empty_prediction
    assert without_alignment > empty_prediction


def test_variants_without_alignment_keep_their_target():
    loss_fun = DeepPrintLoss_TexMinu(3, 4, 4, weights=LossWeights(minutia_map=1.0))
    target = _render(*_minutiae()).unsqueeze(0)
    assert _map_loss(loss_fun, target.clone(), target, None) == pytest.approx(0, abs=1e-9)


def test_sample_whose_minutiae_all_left_the_frame_does_not_count():
    loss_fun = DeepPrintLoss_TexMinu(3, 4, 4, weights=LossWeights(minutia_map=1.0))
    # Minutiae in the left quarter; a shift by half the side to the right
    # shows the right half of the input and nothing of them
    locs = np.array([[20.0, 100.0], [40.0, 200.0], [30.0, 150.0]])
    maps = torch.stack([_render(locs, np.zeros(3)), _render(*_minutiae())])
    matrices = torch.cat([_matrix(0, 0.5, 0), _matrix(0, 0, 0)])
    predicted = torch.zeros_like(maps)
    predicted[1] = maps[1]  # perfect for the sample that kept its minutiae
    loss = _map_loss(loss_fun, predicted, maps, _alignment(matrices))
    assert loss == pytest.approx(0, abs=1e-6)
    assert loss_fun.n_minu_maps == 1


def test_moved_target_sends_no_gradient_to_the_alignment():
    # In a model the predicted map depends on the alignment through the
    # aligned image. The target is moved with the alignment, but must not
    # pass a gradient to it. Here the prediction does not depend on the
    # alignment, so the target would be the only path.
    loss_fun = DeepPrintLoss_TexMinu(3, 4, 4, weights=LossWeights(minutia_map=1.0))
    target = _render(*_minutiae()).unsqueeze(0)
    matrices = _matrix(10, 0.0, 0.0).requires_grad_()
    predicted = torch.zeros_like(target, requires_grad=True)
    loss = loss_fun(
        _output(predicted, _alignment(matrices)),
        torch.zeros(1, dtype=torch.long),
        target,
        torch.ones(1),
    )
    loss.backward()
    assert predicted.grad.abs().sum() > 0
    assert matrices.grad is None
