"""
The check that warns or stops the training when the learned alignment
leaves the model with mostly empty images.
"""

import json
import math
from os.path import join

import pytest
import torch

import flx.models.model_training as model_training
from flx.data.dataset import ConstantDataLoader, Dataset, Identifier, IdentifierSet
from flx.models.alignment_check import AlignmentCollapsedError, AlignmentMonitor
from flx.models.deep_print_arch import (
    DeepPrint_LocTex,
    DeepPrint_Tex,
    DeepPrintTrainingOutput,
)
from flx.models.learning_rate import LearningRateConfig, TrainingDivergedError
from flx.models.localization_network import Alignment, LocalizationNetwork
from flx.setup import config


def _alignment(kept_content: list[float], kept_area: list[float] | None = None):
    n = len(kept_content)
    if kept_area is None:
        kept_area = [1.0] * n
    return Alignment(
        matrices=torch.eye(2, 3).repeat(n, 1, 1),
        rotation=torch.full((n,), math.radians(10)),
        shift=torch.tensor([[0.1, -0.2]] * n),
        scale=torch.ones(n),
        kept_area=torch.tensor(kept_area),
        kept_content=torch.tensor(kept_content),
    )


def test_default_thresholds():
    monitor = AlignmentMonitor()
    assert monitor.empty_below == config.ALIGNMENT_EMPTY_BELOW == 0.5
    assert monitor.warn_share == config.ALIGNMENT_WARN_SHARE == 0.1
    assert monitor.stop_share == config.ALIGNMENT_STOP_SHARE == 0.5


def test_images_that_kept_their_print_pass_silently(capsys):
    monitor = AlignmentMonitor()
    monitor.check(_alignment([1.0, 0.9, 0.6, 0.51]), "step 1")
    assert "Warning" not in capsys.readouterr().out
    assert monitor.summary()["share_mostly_empty"] == 0


def test_a_few_mostly_empty_images_warn_once_per_epoch(capsys):
    monitor = AlignmentMonitor()
    # 2 of 8 is above the warning share of 0.1 and below the stop share of 0.5
    batch = _alignment([0.2, 0.4, 1, 1, 1, 1, 1, 1])
    monitor.check(batch, "epoch 1, step 1")
    monitor.check(batch, "epoch 1, step 2")
    out = capsys.readouterr().out
    assert out.count("Warning") == 1
    assert "2 of the 8 aligned images" in out and "epoch 1, step 1" in out
    assert monitor.summary()["num_batches_warned"] == 2
    monitor.reset()
    monitor.check(batch, "epoch 2, step 1")
    assert capsys.readouterr().out.count("Warning") == 1


def test_a_batch_that_is_mostly_empty_stops_the_training():
    monitor = AlignmentMonitor()
    with pytest.raises(AlignmentCollapsedError) as error:
        monitor.check(_alignment([0.0, 0.1, 0.3, 1.0]), "epoch 3, step 7")
    assert isinstance(error.value, TrainingDivergedError)
    message = str(error.value)
    assert "3 of the 4" in message and "epoch 3, step 7" in message
    assert "LOCALIZATION_LEARNING_RATE_FACTOR" in message


def test_a_frame_that_mostly_lies_outside_the_input_counts_as_empty():
    # The content measure cannot see this for images with a bright background
    monitor = AlignmentMonitor()
    with pytest.raises(AlignmentCollapsedError):
        monitor.check(_alignment([1.0, 1.0], kept_area=[0.3, 0.4]), "step 1")


def test_warning_and_stop_can_be_switched_off(capsys):
    monitor = AlignmentMonitor(warn_share=0, stop_share=0)
    monitor.check(_alignment([0.0, 0.0]), "step 1")
    assert capsys.readouterr().out == ""
    assert monitor.summary()["share_mostly_empty"] == 1.0


def test_no_alignment_gives_no_summary():
    monitor = AlignmentMonitor()
    monitor.check(None, "step 1")
    assert monitor.summary() is None


def test_summary_holds_the_numbers_of_the_epoch():
    monitor = AlignmentMonitor()
    monitor.check(_alignment([1.0, 0.8]), "step 1")
    monitor.check(_alignment([0.6, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), "step 2")
    summary = monitor.summary()
    assert summary["num_images"] == 12
    assert summary["mean_rotation_degrees"] == pytest.approx(10)
    assert summary["max_rotation_degrees"] == pytest.approx(10)
    assert summary["mean_shift"] == pytest.approx(0.2)  # the larger of x and y
    assert summary["min_kept_content"] == pytest.approx(0.4)
    assert summary["mean_kept_content"] == pytest.approx(10.8 / 12)
    assert summary["share_mostly_empty"] == pytest.approx(1 / 12)
    json.dumps(summary)  # goes into the training log


def test_loc_variants_report_their_alignment_and_others_do_not():
    images = torch.rand(2, 1, 299, 299)
    loc = DeepPrint_LocTex(num_fingerprints=3, texture_embedding_dims=8)
    for mode in (loc.train, loc.eval):
        mode()
        output = loc(images)
        assert output.alignment.matrices.shape == (2, 2, 3)
        assert output.alignment.kept_content.shape == (2,)
    plain = DeepPrint_Tex(num_fingerprints=3, texture_embedding_dims=8).train()
    assert plain(images).alignment is None


class _ShiftingModel(torch.nn.Module):
    """A real localization network in front of a trivial classifier."""

    def __init__(self, raw_shift: float):
        super().__init__()
        self.localization = LocalizationNetwork()
        self.localization.fc_loc[2].bias.data = torch.tensor([0.0, raw_shift, 0.0])
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, images):
        aligned, alignment = self.localization.align(images)
        logits = self.linear(aligned.mean(dim=(1, 2, 3)).unsqueeze(1))
        return DeepPrintTrainingOutput(texture_logits=logits, alignment=alignment)


class _SumLoss(torch.nn.Module):
    def forward(self, output, labels, minutia_maps, minutia_map_weights):
        return output.texture_logits.sum() + 10.0

    def reset_recorded_loss(self):
        pass

    def get_recorded_loss(self):
        return {}


def _train(tmp_path, monkeypatch, raw_shift: float):
    monkeypatch.setattr(
        model_training,
        "get_dataloader_args",
        lambda train: {"batch_size": 2, "shuffle": train},
    )
    ids = IdentifierSet([Identifier(s, 0) for s in range(2)])
    # A print in the right half of the image, background 0
    image = torch.zeros(1, 299, 299)
    image[:, 100:200, 180:280] = 1.0
    model_training.train_model(
        fingerprints=Dataset(ConstantDataLoader(image), ids),
        minutia_maps=Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        labels=Dataset(ConstantDataLoader(0), ids),
        validation_fingerprints=None,
        validation_benchmark=None,
        model=_ShiftingModel(raw_shift),
        loss=_SumLoss(),
        num_epochs=1,
        out_dir=str(tmp_path),
        learning_rate=LearningRateConfig(warmup_steps=0),
    )


def test_training_stops_when_the_prints_left_the_frame(
    tmp_path, monkeypatch
):
    # A raw shift of 10 is the bound: half the image side to the left, which
    # moves the print out of the frame
    with pytest.raises(AlignmentCollapsedError):
        _train(tmp_path, monkeypatch, raw_shift=-10.0)
    with open(join(tmp_path, "log.json")) as file:
        assert json.load(file)["entries"] == []


def test_stopped_training_did_not_change_the_parameters(tmp_path, monkeypatch):
    model = _ShiftingModel(-10.0)
    before = model.linear.weight.detach().clone()
    monitor = AlignmentMonitor()
    ids = IdentifierSet([Identifier(s, 0) for s in range(2)])
    image = torch.zeros(1, 299, 299)
    image[:, 100:200, 180:280] = 1.0
    monkeypatch.setattr(
        model_training,
        "get_dataloader_args",
        lambda train: {"batch_size": 2, "shuffle": train},
    )
    with pytest.raises(AlignmentCollapsedError):
        model_training._train(
            model,
            _SumLoss(),
            torch.optim.SGD(model.parameters(), lr=1.0),
            Dataset.zip(
                Dataset(ConstantDataLoader(image), ids),
                Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
                Dataset(ConstantDataLoader(0), ids),
            ),
            use_amp=False,
            alignment_monitor=monitor,
        )
    assert torch.equal(before, model.linear.weight.detach())


def test_training_log_holds_the_alignment_of_the_epoch(tmp_path, monkeypatch):
    # A small shift keeps the print in the frame
    _train(tmp_path, monkeypatch, raw_shift=0.2)
    with open(join(tmp_path, "log.json")) as file:
        entry = json.load(file)["entries"][0]
    alignment = entry["alignment"]
    assert alignment["num_images"] == 2
    assert alignment["mean_shift"] == pytest.approx(0.5 * math.tanh(0.2), rel=1e-3)
    assert alignment["share_mostly_empty"] == 0
    assert alignment["min_kept_content"] > 0.9
