"""
Tests of the learning rate schedule, the gradient clipping and the stop of a
diverged training.
"""
import json
import math
from os.path import exists, join

import pytest
import torch

import flx.models.model_training as model_training
from flx.data.dataset import ConstantDataLoader, Dataset, Identifier, IdentifierSet
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.learning_rate import (
    DivergenceGuard,
    LearningRateConfig,
    TrainingDivergedError,
)
from flx.models.model_training import train_model
from flx.setup import config


# --- The schedule -------------------------------------------------------------


def test_defaults_come_from_the_config_module(monkeypatch):
    monkeypatch.setattr(config, "LEARNING_RATE", 0.5)
    monkeypatch.setattr(config, "LEARNING_RATE_WARMUP_STEPS", 7)
    monkeypatch.setattr(config, "LEARNING_RATE_SCHEDULE", "cosine")
    monkeypatch.setattr(config, "GRADIENT_CLIP_NORM", 3.0)
    lr = LearningRateConfig()
    assert lr.learning_rate == 0.5
    assert lr.warmup_steps == 7
    assert lr.schedule == "cosine"
    assert lr.gradient_clip_norm == 3.0
    assert LearningRateConfig(learning_rate=0.1).learning_rate == 0.1


def test_warmup_rises_in_a_straight_line_to_the_learning_rate():
    lr = LearningRateConfig(learning_rate=0.8, warmup_steps=4, schedule="constant")
    rates = [lr.learning_rate_at(step, total_steps=10) for step in range(10)]
    assert rates == pytest.approx([0.2, 0.4, 0.6, 0.8] + [0.8] * 6)


def test_without_warmup_the_first_step_has_the_full_learning_rate():
    lr = LearningRateConfig(learning_rate=0.8, warmup_steps=0, schedule="constant")
    assert lr.learning_rate_at(0, total_steps=10) == 0.8


def test_cosine_falls_from_the_learning_rate_to_the_final_factor():
    lr = LearningRateConfig(
        learning_rate=1.0, warmup_steps=5, schedule="cosine", final_factor=0.01
    )
    total = 105
    rates = [lr.learning_rate_at(step, total) for step in range(total)]
    assert rates[4] == pytest.approx(1.0)  # last warm-up step
    assert rates[5] == pytest.approx(1.0)  # first step after the warm-up
    assert rates[-1] == pytest.approx(0.01)  # last step of the training
    assert all(a >= b for a, b in zip(rates[5:], rates[6:]))
    # Half way through the decay the rate is half way between both ends
    assert lr.learning_rate_at(5 + 99 / 2, total) == pytest.approx(0.505)
    # Steps after the planned end keep the final rate
    assert lr.learning_rate_at(total + 50, total) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": math.nan},
        {"warmup_steps": -1},
        {"schedule": "step"},
        {"final_factor": 1.5},
        {"gradient_clip_norm": -1.0},
        {"loss_explosion_factor": 0.5},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LearningRateConfig(**kwargs)


# --- The check for a diverged training ----------------------------------------


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf])
def test_guard_stops_on_a_loss_that_is_not_finite(loss):
    guard = DivergenceGuard(explosion_factor=10)
    guard.check(5.0, 0.1, "step 1")
    with pytest.raises(TrainingDivergedError, match="not a finite number"):
        guard.check(loss, 0.1, "step 2")
    # Also in the very first step and with the explosion check switched off
    with pytest.raises(TrainingDivergedError):
        DivergenceGuard(explosion_factor=0).check(loss, 0.1, "step 1")


def test_guard_stops_on_a_loss_far_above_the_first_one():
    guard = DivergenceGuard(explosion_factor=10)
    guard.check(5.0, 0.1, "step 1")
    guard.check(49.0, 0.1, "step 2")  # below 10 times the first loss
    with pytest.raises(TrainingDivergedError, match="learning rate 0.1") as error:
        guard.check(51.0, 0.1, "epoch 1, step 3")
    assert "epoch 1, step 3" in str(error.value)
    assert "find_learning_rate.py" in str(error.value)


def test_guard_with_factor_zero_only_stops_on_a_loss_that_is_not_finite():
    guard = DivergenceGuard(explosion_factor=0)
    guard.check(5.0, 0.1, "step 1")
    guard.check(1e12, 0.1, "step 2")


def test_guard_of_a_resumed_training_compares_with_the_logged_loss():
    # The training resumes with a loss of 0.01; a batch with 0.5 is noise, not
    # an explosion, because the training started at 6.0
    guard = DivergenceGuard(explosion_factor=10, reference_loss=6.0)
    guard.check(0.01, 0.1, "step 1")
    guard.check(0.5, 0.1, "step 2")
    with pytest.raises(TrainingDivergedError):
        guard.check(61.0, 0.1, "step 3")


# --- The training loop --------------------------------------------------------


class _TinyModel(torch.nn.Module):
    def __init__(self, num_subjects: int = 8):
        super().__init__()
        self.linear = torch.nn.Linear(1, num_subjects)

    def forward(self, images):
        return DeepPrintTrainingOutput(texture_logits=self.linear(images))


class _ScriptedLoss(torch.nn.Module):
    """
    Loss number i is factors[i] times the sum of the logits (the last factor
    repeats), so its gradient with respect to every bias is factors[i] times
    the batch size.
    """

    def __init__(self, factors: list[float], offset: float = 0.0):
        super().__init__()
        self.factors = factors
        # Added to the sum of the logits; keeps the loss positive
        self.offset = offset
        self.calls = 0

    def forward(self, output, labels, minutia_maps, minutia_map_weights):
        factor = self.factors[min(self.calls, len(self.factors) - 1)]
        self.calls += 1
        return (self.offset + output.texture_logits.sum()) * factor

    def reset_recorded_loss(self):
        pass

    def get_recorded_loss(self):
        return {}


class _RecordingSGD(torch.optim.SGD):
    """Plain gradient descent that notes the learning rate of every step."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.rates = []

    def step(self, closure=None):
        self.rates.append(self.param_groups[0]["lr"])
        return super().step(closure)


def _training_set(num_samples: int) -> Dataset:
    ids = IdentifierSet([Identifier(s, 0) for s in range(num_samples)])
    return Dataset.zip(
        Dataset(ConstantDataLoader(torch.ones(1)), ids),
        Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        Dataset(ConstantDataLoader(0), ids),
    )


@pytest.fixture
def batches_of_two(monkeypatch):
    monkeypatch.setattr(
        model_training,
        "get_dataloader_args",
        lambda train: {"batch_size": 2, "shuffle": train},
    )


def test_training_loop_follows_the_schedule_over_the_epochs(batches_of_two):
    # 8 samples in batches of 2 are 4 steps per epoch, 12 steps in 3 epochs
    lr = LearningRateConfig(
        learning_rate=1.0, warmup_steps=3, schedule="cosine", final_factor=0.1
    )
    model = _TinyModel()
    optimizer = _RecordingSGD(model.parameters(), lr=123.0)
    for epoch in (1, 2, 3):
        model_training._train(
            model,
            _ScriptedLoss([0.0]),
            optimizer,
            _training_set(8),
            use_amp=False,
            learning_rate=lr,
            epoch=epoch,
            num_epochs=3,
        )
    assert optimizer.rates == pytest.approx(
        [lr.learning_rate_at(step, 12) for step in range(12)]
    )
    assert optimizer.rates[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert optimizer.rates[-1] == pytest.approx(0.1)


def test_resumed_epoch_continues_the_schedule_where_it_was(batches_of_two):
    # A new process that starts with epoch 3 must use the rates of epoch 3,
    # although its optimizer has never seen the first two epochs
    lr = LearningRateConfig(
        learning_rate=1.0, warmup_steps=3, schedule="cosine", final_factor=0.1
    )
    model = _TinyModel()
    optimizer = _RecordingSGD(model.parameters(), lr=123.0)
    model_training._train(
        model,
        _ScriptedLoss([0.0]),
        optimizer,
        _training_set(8),
        use_amp=False,
        learning_rate=lr,
        epoch=3,
        num_epochs=3,
    )
    assert optimizer.rates == pytest.approx(
        [lr.learning_rate_at(step, 12) for step in range(8, 12)]
    )


def test_gradient_clipping_limits_the_length_of_the_step(batches_of_two):
    # One step with a batch of 2. The gradient of each of the 2 biases is
    # 100 * 2, so the gradient norm is above 200. With learning rate 1 plain
    # gradient descent moves the parameters by the gradient, so the length of
    # the move is the (clipped) gradient norm.
    def length_of_first_step(clip_norm):
        model = _TinyModel(2)
        before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        model_training._train(
            model,
            _ScriptedLoss([100.0]),
            torch.optim.SGD(model.parameters(), lr=1.0),
            _training_set(2),
            use_amp=False,
            learning_rate=LearningRateConfig(
                learning_rate=1.0,
                warmup_steps=0,
                schedule="constant",
                gradient_clip_norm=clip_norm,
            ),
        )
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        return float(torch.linalg.norm(after - before))

    assert length_of_first_step(None) > 200
    assert length_of_first_step(0.5) == pytest.approx(0.5, rel=1e-4)


def test_loss_that_is_not_finite_stops_before_the_parameters_change(batches_of_two):
    model = _TinyModel()
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(TrainingDivergedError, match="epoch 1, step 1"):
        model_training._train(
            model,
            _ScriptedLoss([math.nan]),
            torch.optim.SGD(model.parameters(), lr=1.0),
            _training_set(4),
            use_amp=False,
            guard=DivergenceGuard(explosion_factor=10),
        )
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p, b)


def _train_tiny_model(out_dir, factors, num_epochs, learning_rate=None):
    train_set = _training_set(4)
    ids = train_set.ids
    train_model(
        fingerprints=Dataset(ConstantDataLoader(torch.ones(1)), ids),
        minutia_maps=Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        labels=Dataset(ConstantDataLoader(0), ids),
        validation_fingerprints=None,
        validation_benchmark=None,
        model=_TinyModel(4),
        loss=_ScriptedLoss(factors, offset=100.0),
        num_epochs=num_epochs,
        out_dir=str(out_dir),
        use_amp=False,
        learning_rate=learning_rate,
    )


def test_exploding_loss_stops_the_training_and_keeps_the_last_epoch(
    batches_of_two, tmp_path
):
    # Epoch 1 (2 steps) is fine. In epoch 2 the loss is 1000 times as high.
    lr = LearningRateConfig(learning_rate=1e-6, warmup_steps=0, schedule="constant")
    with pytest.raises(TrainingDivergedError, match="epoch 2, step 3"):
        _train_tiny_model(tmp_path, [1.0, 1.0, 1000.0], num_epochs=3, learning_rate=lr)
    with open(join(tmp_path, "log.json")) as file:
        entries = json.load(file)["entries"]
    assert [e["epoch"] for e in entries] == [1]
    assert exists(join(tmp_path, "model.pyt"))


def test_training_log_holds_the_learning_rate_of_each_epoch(batches_of_two, tmp_path):
    lr = LearningRateConfig(
        learning_rate=1e-3, warmup_steps=2, schedule="cosine", final_factor=0.5
    )
    _train_tiny_model(tmp_path, [1.0], num_epochs=2, learning_rate=lr)
    with open(join(tmp_path, "log.json")) as file:
        entries = json.load(file)["entries"]
    assert [e["learning_rate"] for e in entries] == pytest.approx([1e-3, 5e-4])


def test_resumed_training_uses_the_schedule_and_not_the_stored_rate(
    batches_of_two, tmp_path
):
    # The checkpoint stores the optimizer with the rate of its last step. The
    # second run must go on with the schedule: 4 steps in total, the last two
    # in the resumed run.
    lr = LearningRateConfig(
        learning_rate=1e-3, warmup_steps=0, schedule="cosine", final_factor=0.0
    )
    _train_tiny_model(tmp_path, [1.0], num_epochs=1, learning_rate=lr)
    _train_tiny_model(tmp_path, [1.0], num_epochs=2, learning_rate=lr)
    with open(join(tmp_path, "log.json")) as file:
        entries = json.load(file)["entries"]
    assert [e["epoch"] for e in entries] == [1, 2]
    # Epoch 1 alone was a training of 2 steps and ended at factor 0; epoch 2
    # is the end of a training of 4 steps and also ends at factor 0
    assert entries[1]["learning_rate"] == pytest.approx(0.0, abs=1e-12)
