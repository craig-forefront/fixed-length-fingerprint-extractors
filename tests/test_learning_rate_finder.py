"""
Tests of the learning rate range test (flx/models/learning_rate_finder.py and
scripts/find_learning_rate.py)
"""
import csv
import importlib.util
import math
import sys
from os.path import abspath, dirname, exists, join

import cv2
import numpy as np
import pytest
import torch

from flx.data.dataset import ConstantDataLoader, Dataset, Identifier, IdentifierSet
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.learning_rate_finder import (
    CSV_COLUMNS,
    range_test_learning_rates,
    run_range_test,
    save_range_test_csv,
    save_range_test_plot,
    smooth_losses,
    suggest_learning_rate,
)

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")


def test_rates_rise_by_a_constant_factor_from_min_to_max():
    rates = range_test_learning_rates(1e-6, 1e-2, 5)
    assert rates == pytest.approx([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    with pytest.raises(ValueError):
        range_test_learning_rates(1e-2, 1e-6, 5)
    with pytest.raises(ValueError):
        range_test_learning_rates(1e-6, 1e-2, 1)


def test_smoothing_keeps_a_constant_loss_and_follows_a_change_slowly():
    assert smooth_losses([3.0] * 5, 0.9) == pytest.approx([3.0] * 5)
    smoothed = smooth_losses([1.0] * 20 + [2.0] * 3, 0.9)
    assert 1.0 < smoothed[-1] < 1.5
    assert smooth_losses([1.0, 5.0], 0.0) == [1.0, 5.0]
    # A value that is not finite is passed on and does not spoil the average
    smoothed = smooth_losses([1.0, math.nan, 1.0], 0.5)
    assert smoothed[0] == 1.0 and math.isnan(smoothed[1]) and smoothed[2] == 1.0


def _typical_curve():
    """
    200 steps from 1e-7 to 1: the loss stays at 10 for low rates, falls most
    steeply around 1e-3, is lowest near 1e-2 and explodes above it.
    """
    rates = range_test_learning_rates(1e-7, 1.0, 200)
    losses = []
    for rate in rates:
        x = math.log10(rate)
        loss = 10 - 4 / (1 + math.exp(-3 * (x + 3)))  # falls around x = -3
        if x > -2:
            loss += 10 ** (3 * (x + 2)) - 1  # explodes above x = -2
        losses.append(loss)
    return rates, losses


def test_suggestion_is_a_tenth_of_the_rate_with_the_steepest_fall():
    rates, losses = _typical_curve()
    suggestion = suggest_learning_rate(rates, losses, smoothing=0.0)
    assert suggestion.steepest_learning_rate == pytest.approx(1e-3, rel=0.2)
    assert suggestion.lowest_loss_learning_rate == pytest.approx(1e-2, rel=0.5)
    assert suggestion.suggested_learning_rate == pytest.approx(
        suggestion.steepest_learning_rate / 10
    )
    # Smoothing delays the curve a little but must not move the result far
    smoothed = suggest_learning_rate(rates, losses, smoothing=0.9)
    assert 3e-4 < smoothed.steepest_learning_rate < 1e-2


def test_suggestion_ignores_the_explosion_and_values_that_are_not_finite():
    rates, losses = _typical_curve()
    losses[-1] = math.nan
    losses[-2] = math.inf
    suggestion = suggest_learning_rate(rates, losses, smoothing=0.0)
    assert suggestion.steepest_learning_rate == pytest.approx(1e-3, rel=0.2)


def test_no_suggestion_when_the_loss_never_falls():
    rates = range_test_learning_rates(1e-3, 1.0, 50)
    losses = [10.0 + i for i in range(50)]
    suggestion = suggest_learning_rate(rates, losses, smoothing=0.0)
    assert suggestion.suggested_learning_rate is None
    assert "--min_lr" in suggestion.explanation
    with pytest.raises(ValueError):
        suggest_learning_rate(rates[:4], losses[:4])


# --- The test run itself ------------------------------------------------------


class _TinyModel(torch.nn.Module):
    def __init__(self, num_subjects: int):
        super().__init__()
        self.linear = torch.nn.Linear(1, num_subjects)

    def forward(self, images):
        return DeepPrintTrainingOutput(texture_logits=self.linear(images.reshape(-1, 1)))


class _SquareLoss(torch.nn.Module):
    """1 + mean of the squared logits: lowest (1) when all logits are zero."""

    def __init__(self, nan_from_call: int | None = None):
        super().__init__()
        self.calls = 0
        self.nan_from_call = nan_from_call

    def forward(self, output, labels, minutia_maps, minutia_map_weights):
        self.calls += 1
        loss = 1 + (output.texture_logits**2).mean()
        if self.nan_from_call is not None and self.calls >= self.nan_from_call:
            loss = loss * math.nan
        return loss

    def reset_recorded_loss(self):
        pass


def _training_set(num_samples: int) -> Dataset:
    ids = IdentifierSet([Identifier(s, 0) for s in range(num_samples)])
    return Dataset.zip(
        Dataset(ConstantDataLoader(torch.ones(1)), ids),
        Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        Dataset(ConstantDataLoader(0), ids),
    )


def test_range_test_raises_the_rate_and_stops_when_the_loss_explodes():
    torch.manual_seed(0)
    model = _TinyModel(4)
    loss_fun = _SquareLoss()
    # 6 samples in batches of 4: the test goes through the set many times
    result = run_range_test(
        model, loss_fun, _training_set(6), min_lr=1e-4, max_lr=1e4,
        num_steps=200, batch_size=4,
    )
    n = len(result.losses)
    assert result.stop_reason == "loss exploded"
    assert 10 < n < 200
    assert loss_fun.calls == n
    assert result.learning_rates == pytest.approx(
        range_test_learning_rates(1e-4, 1e4, 200)[:n]
    )
    assert len(result.gradient_norms) == n
    # The model learned before it diverged
    assert min(result.losses) < result.losses[0]
    assert result.losses[-1] > min(result.losses)


def test_range_test_stops_at_a_loss_that_is_not_finite():
    result = run_range_test(
        _TinyModel(4), _SquareLoss(nan_from_call=4), _training_set(8),
        min_lr=1e-4, max_lr=1e-2, num_steps=10, batch_size=4,
    )
    assert result.stop_reason == "loss not finite"
    assert len(result.losses) == 4 and math.isnan(result.losses[-1])


def test_range_test_with_a_harmless_range_does_all_steps():
    result = run_range_test(
        _TinyModel(4), _SquareLoss(), _training_set(8),
        min_lr=1e-6, max_lr=1e-5, num_steps=7, batch_size=4,
    )
    assert result.stop_reason == "all steps done"
    assert len(result.losses) == 7


def test_csv_and_plot_are_written(tmp_path):
    torch.manual_seed(0)
    result = run_range_test(
        _TinyModel(4), _SquareLoss(), _training_set(8),
        min_lr=1e-4, max_lr=1e4, num_steps=100, batch_size=4,
    )
    suggestion = suggest_learning_rate(result.learning_rates, result.losses)
    csv_path = join(tmp_path, "test.csv")
    plot_path = join(tmp_path, "test.png")
    save_range_test_csv(csv_path, result)
    save_range_test_plot(plot_path, result, suggestion)
    with open(csv_path, newline="") as file:
        rows = list(csv.reader(file))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 1 + len(result.losses)
    assert [int(r[0]) for r in rows[1:]] == list(range(1, len(rows)))
    assert [float(r[1]) for r in rows[1:]] == result.learning_rates
    assert [float(r[2]) for r in rows[1:]] == result.losses
    assert cv2.imread(plot_path) is not None


# --- The script ---------------------------------------------------------------


@pytest.fixture(scope="module")
def find_learning_rate():
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        spec = importlib.util.spec_from_file_location(
            "find_learning_rate", join(SCRIPTS_DIR, "find_learning_rate.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(SCRIPTS_DIR)
    return module


def test_subject_choice_is_a_reproducible_subset(find_learning_rate):
    assert find_learning_rate.choose_subjects(5, 0, seed=0) == [0, 1, 2, 3, 4]
    assert find_learning_rate.choose_subjects(5, 9, seed=0) == [0, 1, 2, 3, 4]
    chosen = find_learning_rate.choose_subjects(1000, 50, seed=3)
    assert len(set(chosen)) == 50 and set(chosen) <= set(range(1000))
    assert chosen == find_learning_rate.choose_subjects(1000, 50, seed=3)
    assert chosen != find_learning_rate.choose_subjects(1000, 50, seed=4)


class _TinyExtractor:
    def __init__(self, num_subjects: int):
        self.model = _TinyModel(num_subjects)
        self.loss = _SquareLoss()


@pytest.mark.parametrize("with_minutiae", [False, True])
def test_script_runs_on_the_dataset_layout_of_train_500k(
    find_learning_rate, tmp_path, monkeypatch, capsys, with_minutiae
):
    # 6 subjects with 2 impressions in the layout of train_500k.py. The model
    # is replaced by a tiny one that looks at one pixel; the images and the
    # minutia maps go through the real loaders of train_500k.py.
    image_dir = tmp_path / "images"
    minutiae_dir = tmp_path / "minutiae"
    image_dir.mkdir()
    minutiae_dir.mkdir()
    rng = np.random.default_rng(0)
    for subject in range(1, 7):
        for impression in (1, 2):
            cv2.imwrite(
                str(image_dir / f"{subject}_{impression}.png"),
                rng.integers(0, 255, (80, 64), dtype=np.uint8),
            )
            with open(minutiae_dir / f"{subject}_{impression}.mnt", "w") as file:
                file.write("1\n64 80\n10 20 0.5\n")

    seen = {}

    class _OnePixelModel(_TinyModel):
        def forward(self, images):
            seen["image_shape"] = tuple(images.shape)
            return super().forward(images[:, 0, 0, 0])

    class _Loss(_SquareLoss):
        def forward(self, output, labels, minutia_maps, minutia_map_weights):
            seen["map_shape"] = tuple(minutia_maps.shape)
            seen["labels"] = seen.get("labels", set()) | set(labels.tolist())
            return super().forward(output, labels, minutia_maps, minutia_map_weights)

    def build(variant, num_subjects, embedding_dim):
        seen["variant"] = variant
        extractor = _TinyExtractor(num_subjects)
        extractor.model = _OnePixelModel(num_subjects)
        extractor.loss = _Loss()
        return extractor

    monkeypatch.setattr(find_learning_rate, "build_extractor", build)
    out_dir = tmp_path / "out"
    argv = [
        "find_learning_rate.py",
        "--dataset_dir", str(image_dir),
        "--output_dir", str(out_dir),
        "--num_subjects", "6",
        "--num_impressions", "2",
        "--max_subjects", "4",
        "--batch_size", "4",
        "--num_steps", "60",
        "--min_lr", "1e-4",
        "--max_lr", "1e4",
        "--num_workers", "0",
    ]
    if with_minutiae:
        argv += ["--minutiae_dir", str(minutiae_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    find_learning_rate.main()

    assert seen["variant"] == ("LocTexMinu" if with_minutiae else "LocTex")
    assert seen["image_shape"] == (4, 1, 299, 299)
    assert seen["map_shape"] == ((4, 6, 128, 128) if with_minutiae else (4, 0))
    assert seen["labels"] == {0, 1, 2, 3}  # 4 chosen subjects, one class each
    assert exists(out_dir / "lr_range_test.csv")
    assert exists(out_dir / "lr_range_test.png")
    assert "Suggested: --learning_rate" in capsys.readouterr().out


def test_minutia_variant_without_minutiae_is_rejected(
    find_learning_rate, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "find_learning_rate.py",
            "--dataset_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--variant", "TexMinu",
        ],
    )
    with pytest.raises(SystemExit):
        find_learning_rate.main()
