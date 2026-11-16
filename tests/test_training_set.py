import numpy as np
import torch

from flx.data.dataset import Identifier, IdentifierSet, DataLoader, Dataset
from flx.data.image_helpers import transform_to_input_size
from flx.data.minutia_map_loader import (
    MINUTIA_MAP_CHANNELS,
    MINUTIA_MAP_SIZE,
    MinutiaMapLoader,
)
from flx.models.deep_print_loss import _compute_minutia_map_loss
from flx.setup import datasets
from flx.setup.config import INPUT_SIZE

MAP_SHAPE = (MINUTIA_MAP_CHANNELS, MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE)


class _ConstantImages(DataLoader):
    def __init__(self, ids: IdentifierSet, height: int, width: int):
        self._ids = ids
        self._height = height
        self._width = width

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> torch.Tensor:
        return torch.full((1, self._height, self._width), 0.5)


class _OneMinutia(MinutiaMapLoader):
    def __init__(self, ids: IdentifierSet):
        self._ids = ids

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        locs = transform_to_input_size(np.array([[200.0, 300.0]]), 560, 416)
        return locs, np.array([1.0])


def test_mcyt_samples_have_an_empty_minutia_map_with_weight_zero(monkeypatch):
    """
    The training set has no minutia maps for MCYT. Its samples must still have the
    shape of the SFinGe samples, so that both can be in one batch, and weight 0,
    so that they do not count in the minutia map loss.
    """
    sfinge_ids = IdentifierSet(
        [Identifier(s, i) for s in range(6000) for i in range(10)]
    )
    mcyt_ids = IdentifierSet([Identifier(s, i) for s in range(2000) for i in range(12)])
    # Stand-ins for the datasets, which are not part of the repository
    monkeypatch.setattr(
        datasets, "SFingeLoader", lambda _: _ConstantImages(sfinge_ids, 560, 416)
    )
    for loader in ("MCYTOpticalLoader", "MCYTCapacitiveLoader"):
        monkeypatch.setattr(
            datasets,
            loader,
            lambda _: _ConstantImages(mcyt_ids, INPUT_SIZE, INPUT_SIZE),
        )
    monkeypatch.setattr(
        datasets, "SFingeMinutiaMapLoader", lambda _: _OneMinutia(sfinge_ids)
    )
    for transform in (
        "QUALITY_AUGMENTATION",
        "SFINGE_BINARIZATION",
        "MCYT_OPTICAL_BINARIZATION",
        "MCYT_CAPACITIVE_BINARIZATION",
    ):
        monkeypatch.setattr(datasets, transform, lambda x: x)

    images, minutia_maps, labels = datasets.get_training_set("", "", "")
    samples = Dataset.zip(images, minutia_maps, labels)
    assert len(samples) == 108000

    # First and last sample of the optical and of the capacitive MCYT part
    for index in (60000, 83999, 84000, 107999):
        _, (minutia_map, weight), _ = samples[index]
        assert tuple(minutia_map.shape) == MAP_SHAPE
        assert not minutia_map.any()
        assert weight == 0.0

    # SFinGe and MCYT in one batch
    batch = [samples[i] for i in (0, 59999, 60000, 107999)]
    _, (maps, weights), _ = torch.utils.data.default_collate(batch)
    assert tuple(maps.shape) == (4, *MAP_SHAPE)
    assert weights.tolist() == [1.0, 1.0, 0.0, 0.0]
    predicted = torch.rand(4, *MAP_SHAPE, requires_grad=True)
    loss, n_with_map = _compute_minutia_map_loss(predicted, maps, weights)
    assert n_with_map == 2
    loss.backward()
    gradient = predicted.grad.reshape(4, -1).abs().sum(dim=1)
    assert gradient[0] > 0 and gradient[1] > 0
    assert gradient[2] == 0 and gradient[3] == 0

    # A batch of MCYT samples only
    batch = [samples[i] for i in range(60000, 60064)]
    _, (maps, weights), _ = torch.utils.data.default_collate(batch)
    assert tuple(maps.shape) == (64, *MAP_SHAPE)
    loss, n_with_map = _compute_minutia_map_loss(
        torch.rand(64, *MAP_SHAPE), maps, weights
    )
    assert float(loss) == 0.0
    assert n_with_map == 0
