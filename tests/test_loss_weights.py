"""
Tests of the weights of the parts of the training loss
"""
import importlib.util
import sys
from os.path import abspath, dirname, join

import cv2
import numpy as np
import pytest
import torch

from flx.extractor import fixed_length_extractor
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.deep_print_loss import (
    DeepPrintLoss_Minu,
    DeepPrintLoss_Tex,
    DeepPrintLoss_TexMinu,
    LossWeights,
)
from flx.setup import config

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")


def _output(batch_size: int = 4, seed: int = 1) -> DeepPrintTrainingOutput:
    generator = torch.Generator().manual_seed(seed)
    return DeepPrintTrainingOutput(
        minutia_logits=torch.randn(batch_size, 6, generator=generator),
        texture_logits=torch.randn(batch_size, 6, generator=generator),
        minutia_maps=torch.rand(batch_size, 6, 8, 8, generator=generator),
        minutia_embeddings=torch.nn.functional.normalize(
            torch.randn(batch_size, 16, generator=generator), dim=1
        ),
        texture_embeddings=torch.nn.functional.normalize(
            torch.randn(batch_size, 16, generator=generator), dim=1
        ),
    )


def _texminu_loss(weights: LossWeights | None, batch_size: int = 4) -> float:
    torch.manual_seed(0)  # the class centers are random
    loss_fun = DeepPrintLoss_TexMinu(6, 16, 16, weights)
    labels = torch.arange(batch_size) % 6
    maps = torch.zeros(batch_size, 6, 8, 8)
    return float(loss_fun(_output(batch_size), labels, maps, torch.ones(batch_size)))


def test_default_weights_come_from_the_config(monkeypatch):
    weights = LossWeights()
    assert weights.cross_entropy == config.LOSS_WEIGHT_CROSS_ENTROPY
    assert weights.center == config.LOSS_WEIGHT_CENTER
    assert weights.minutia_map == config.LOSS_WEIGHT_MINUTIA_MAP
    monkeypatch.setattr(config, "LOSS_WEIGHT_CENTER", 0.5)
    assert LossWeights().center == 0.5
    assert LossWeights(minutia_map=2).center == 0.5
    assert LossWeights(minutia_map=2).minutia_map == 2.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_weights_must_be_finite_and_not_negative(value):
    with pytest.raises(ValueError):
        LossWeights(center=value)


def test_each_weight_multiplies_its_part_of_the_loss():
    only = {
        name: _texminu_loss(
            LossWeights(**{"cross_entropy": 0, "center": 0, "minutia_map": 0, name: 1})
        )
        for name in ("cross_entropy", "center", "minutia_map")
    }
    assert all(value > 0 for value in only.values())
    assert _texminu_loss(LossWeights(0, 0, 0)) == 0
    combined = _texminu_loss(LossWeights(cross_entropy=2, center=3, minutia_map=5))
    assert combined == pytest.approx(
        2 * only["cross_entropy"] + 3 * only["center"] + 5 * only["minutia_map"],
        rel=1e-5,
    )
    assert _texminu_loss(None) == pytest.approx(
        config.LOSS_WEIGHT_CROSS_ENTROPY * only["cross_entropy"]
        + config.LOSS_WEIGHT_CENTER * only["center"]
        + config.LOSS_WEIGHT_MINUTIA_MAP * only["minutia_map"],
        rel=1e-5,
    )


def test_weights_apply_to_all_loss_classes():
    weights = LossWeights(cross_entropy=0, center=0, minutia_map=0)
    output = _output()
    labels = torch.tensor([0, 1, 2, 3])
    maps = torch.zeros(4, 6, 8, 8)
    assert float(DeepPrintLoss_Minu(6, 16, weights)(output, labels, maps, torch.ones(4))) == 0
    tex = DeepPrintLoss_Tex(6, 16, weights)
    assert float(tex(output, labels, torch.zeros(4, 0), torch.zeros(4))) == 0


@pytest.mark.parametrize(
    "constructor",
    [
        "get_DeepPrint_Tex",
        "get_DeepPrint_Minu",
        "get_DeepPrint_TexMinu",
        "get_DeepPrint_LocTex",
        "get_DeepPrint_LocMinu",
        "get_DeepPrint_LocTexMinu",
    ],
)
def test_extractors_pass_the_weights_to_the_loss(constructor):
    weights = LossWeights(cross_entropy=2, center=3, minutia_map=5)
    extractor = getattr(fixed_length_extractor, constructor)(4, 8, loss_weights=weights)
    assert extractor.loss.weights == weights
    for part in ("texture_loss_fun", "minu_loss_fun"):
        if hasattr(extractor.loss, part):
            assert getattr(extractor.loss, part).weights == weights
    default = getattr(fixed_length_extractor, constructor)(4, 8)
    assert default.loss.weights == LossWeights()


@pytest.fixture(scope="module")
def train_500k():
    spec = importlib.util.spec_from_file_location(
        "train_500k", join(SCRIPTS_DIR, "train_500k.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeExtractor:
    def fit(self, **kwargs):
        pass


def _run_main(train_500k, tmp_path, monkeypatch, extra_args) -> LossWeights:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for subject in range(1, 5):
        for impression in (1, 2):
            cv2.imwrite(
                str(image_dir / f"{subject}_{impression}.png"),
                np.zeros((8, 8), dtype=np.uint8),
            )
    received = {}

    def fake_constructor(*args, **kwargs):
        received.update(kwargs)
        return _FakeExtractor()

    monkeypatch.setattr(train_500k, "get_DeepPrint_LocTex", fake_constructor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_500k.py",
            "--dataset_dir", str(image_dir),
            "--output_dir", str(tmp_path / "model"),
            "--num_subjects", "4",
            "--num_impressions", "2",
            "--validation_split", "0",
        ]
        + extra_args,
    )
    train_500k.main()
    return received["loss_weights"]


def test_train_500k_uses_the_config_weights_by_default(train_500k, tmp_path, monkeypatch):
    assert _run_main(train_500k, tmp_path, monkeypatch, []) == LossWeights()


def test_train_500k_options_set_the_weights(train_500k, tmp_path, monkeypatch):
    weights = _run_main(
        train_500k,
        tmp_path,
        monkeypatch,
        [
            "--loss_weight_cross_entropy", "2",
            "--loss_weight_center", "3",
            "--loss_weight_minutia_map", "5",
        ],
    )
    assert weights == LossWeights(cross_entropy=2, center=3, minutia_map=5)


def test_train_500k_rejects_a_negative_weight(train_500k, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(train_500k, tmp_path, monkeypatch, ["--loss_weight_center", "-1"])


def _embedding_loss(batch_size: int, repeat: int, weights: LossWeights | None = None):
    """Cross entropy part and center part of the loss of `batch_size` samples,
    each of which is `repeat` times in the batch."""
    torch.manual_seed(0)
    loss_fun = DeepPrintLoss_Tex(6, 16, weights)
    output = _output(batch_size)
    labels = torch.arange(batch_size) % 6
    loss = loss_fun(
        DeepPrintTrainingOutput(
            texture_logits=output.texture_logits.repeat(repeat, 1),
            texture_embeddings=output.texture_embeddings.repeat(repeat, 1),
        ),
        labels.repeat(repeat),
        torch.zeros(batch_size * repeat, 0),
        torch.zeros(batch_size * repeat),
    )
    return float(loss), loss_fun.get_recorded_loss()


def test_center_loss_does_not_grow_with_the_batch_size():
    # The same samples twice in the batch: every mean stays the same
    only_center = LossWeights(cross_entropy=0, center=1)
    small, small_recorded = _embedding_loss(4, 1, only_center)
    large, large_recorded = _embedding_loss(4, 2, only_center)
    assert small > 0
    assert large == pytest.approx(small, rel=1e-5)
    assert large_recorded["center_loss_sum"] == pytest.approx(small, rel=1e-5)
    assert small_recorded["center_loss_sum"] == pytest.approx(small, rel=1e-5)


def test_center_loss_is_the_mean_squared_distance_to_the_class_center():
    from flx.models.center_loss import CenterLoss

    center_loss = CenterLoss(num_classes=2, feat_dim=3)
    center_loss.centers.copy_(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    features = torch.tensor([[1.0, 2.0, 2.0], [1.0, 1.0, 3.0]])
    # Squared distances 9 and 4
    assert float(center_loss(features, torch.tensor([0, 1]))) == pytest.approx(6.5)


def test_default_center_weight_keeps_the_former_loss_for_a_batch_of_64():
    # Formerly: 0.125 times the sum over the batch
    only_center = LossWeights(cross_entropy=0)
    mean_distance, _ = _embedding_loss(64, 1, LossWeights(cross_entropy=0, center=1))
    loss, _ = _embedding_loss(64, 1, only_center)
    assert loss == pytest.approx(0.125 * 64 * mean_distance, rel=1e-5)


def _map_loss(loss_class, predicted, target, map_weights):
    """Map part of the loss and its recorded value."""
    torch.manual_seed(0)
    weights = LossWeights(cross_entropy=0, center=0, minutia_map=1)
    if loss_class is DeepPrintLoss_Minu:
        loss_fun = DeepPrintLoss_Minu(6, 16, weights)
    else:
        loss_fun = DeepPrintLoss_TexMinu(6, 16, 16, weights)
    batch_size = predicted.shape[0]
    output = _output(batch_size)
    output.minutia_maps = predicted
    loss = loss_fun(output, torch.arange(batch_size) % 6, target, map_weights)
    return float(loss), loss_fun.get_recorded_loss()["minutia_map_loss"], loss_fun


@pytest.mark.parametrize("loss_class", [DeepPrintLoss_Minu, DeepPrintLoss_TexMinu])
def test_map_loss_is_the_mean_over_the_samples_that_have_a_map(loss_class):
    generator = torch.Generator().manual_seed(3)
    predicted = torch.rand(5, 6, 8, 8, generator=generator)
    target = torch.rand(5, 6, 8, 8, generator=generator)
    with_map, recorded_with_map, _ = _map_loss(
        loss_class, predicted, target, torch.ones(5)
    )
    assert with_map > 0
    # Three more samples without a map (weight 0), as in a mixed batch
    mixed, recorded_mixed, loss_fun = _map_loss(
        loss_class,
        torch.cat([predicted, torch.rand(3, 6, 8, 8, generator=generator)]),
        torch.cat([target, torch.zeros(3, 6, 8, 8)]),
        # The default collate function makes float64 weights
        torch.tensor([1.0] * 5 + [0.0] * 3, dtype=torch.float64),
    )
    assert mixed == pytest.approx(with_map, rel=1e-5)
    assert recorded_mixed == pytest.approx(with_map, rel=1e-5)
    assert recorded_with_map == pytest.approx(with_map, rel=1e-5)
    loss_fun.reset_recorded_loss()
    assert loss_fun.get_recorded_loss()["minutia_map_loss"] == 0


@pytest.mark.parametrize("loss_class", [DeepPrintLoss_Minu, DeepPrintLoss_TexMinu])
def test_map_loss_of_a_batch_without_any_map_is_zero_and_has_no_gradient(loss_class):
    predicted = torch.rand(4, 6, 8, 8, requires_grad=True)
    torch.manual_seed(0)
    loss, recorded, _ = _map_loss(
        loss_class, predicted, torch.zeros(4, 6, 8, 8), torch.zeros(4)
    )
    assert loss == 0
    assert recorded == 0
    weights = LossWeights(cross_entropy=0, center=0, minutia_map=1)
    loss_fun = DeepPrintLoss_TexMinu(6, 16, 16, weights)
    output = _output()
    output.minutia_maps = predicted
    loss_fun(output, torch.arange(4), torch.zeros(4, 6, 8, 8), torch.zeros(4)).backward()
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().max()) == 0


def test_map_loss_does_not_turn_the_loss_into_float64():
    weights = torch.ones(4, dtype=torch.float64)
    loss_fun = DeepPrintLoss_TexMinu(6, 16, 16)
    loss = loss_fun(_output(), torch.arange(4), torch.zeros(4, 6, 8, 8), weights)
    assert loss.dtype == torch.float32
