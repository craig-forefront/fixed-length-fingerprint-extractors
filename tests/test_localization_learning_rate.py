"""
The localization network (learned alignment) has its own, smaller learning
rate: the learning rate of the other parameters times a factor.
"""

import pytest
import torch

import flx.models.model_training as model_training
from flx.data.dataset import ConstantDataLoader, Dataset, Identifier, IdentifierSet
from flx.models.deep_print_arch import (
    DeepPrint_LocTex,
    DeepPrint_Tex,
    DeepPrintTrainingOutput,
)
from flx.models.learning_rate import (
    LearningRateConfig,
    localization_learning_rate,
    parameter_groups,
    set_learning_rate,
)
from flx.models.torch_helpers import load_model_parameters, save_model_parameters
from flx.setup import config


class _TinyLocModel(torch.nn.Module):
    """One parameter in a module called localization, one outside it."""

    def __init__(self):
        super().__init__()
        self.localization = torch.nn.Linear(1, 1, bias=False)
        self.embeddings = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.localization.weight)
        torch.nn.init.zeros_(self.embeddings.weight)

    def forward(self, images):
        logits = self.localization(images) + self.embeddings(images)
        return DeepPrintTrainingOutput(texture_logits=logits.repeat(1, 2))


class _SumLoss(torch.nn.Module):
    def forward(self, output, labels, minutia_maps, minutia_map_weights):
        return output.texture_logits.sum()

    def reset_recorded_loss(self):
        pass

    def get_recorded_loss(self):
        return {}


def _training_set(num_samples: int) -> Dataset:
    ids = IdentifierSet([Identifier(s, 0) for s in range(num_samples)])
    return Dataset.zip(
        Dataset(ConstantDataLoader(torch.ones(1)), ids),
        Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        Dataset(ConstantDataLoader(0), ids),
    )


def test_default_factor_is_the_one_of_the_deepprint_paper():
    assert config.LOCALIZATION_LEARNING_RATE_FACTOR == 0.035
    assert LearningRateConfig().localization_factor == 0.035
    assert LearningRateConfig(localization_factor=0.5).localization_factor == 0.5


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
def test_factor_must_be_a_positive_number(factor):
    with pytest.raises(ValueError):
        LearningRateConfig(localization_factor=factor)


def test_a_loc_variant_gets_a_group_for_the_localization_network():
    model = DeepPrint_LocTex(num_fingerprints=4, texture_embedding_dims=8)
    groups = parameter_groups(model, 0.035)
    assert len(groups) == 2
    localization = {id(p) for p in model.localization.parameters()}
    assert {id(p) for p in groups[1]["params"]} == localization
    assert groups[1]["lr_factor"] == 0.035
    assert "lr_factor" not in groups[0]
    # Every parameter is in exactly one group
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(
        list(model.parameters())
    )
    assert localization.isdisjoint({id(p) for p in groups[0]["params"]})


def test_a_variant_without_localization_keeps_a_single_group():
    model = DeepPrint_Tex(num_fingerprints=4, texture_embedding_dims=8)
    groups = parameter_groups(model, 0.035)
    assert len(groups) == 1 and "lr_factor" not in groups[0]
    optimizer = torch.optim.Adam(groups, lr=1.0)
    assert localization_learning_rate(optimizer) is None


def test_set_learning_rate_applies_the_factor_in_every_step():
    model = _TinyLocModel()
    optimizer = torch.optim.Adam(parameter_groups(model, 0.035), lr=1.0)
    for rate in (1e-3, 2e-4):
        set_learning_rate(optimizer, rate)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(rate)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(rate * 0.035)
        assert localization_learning_rate(optimizer) == pytest.approx(rate * 0.035)


def test_training_moves_the_localization_network_by_the_smaller_rate(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        model_training,
        "get_dataloader_args",
        lambda train: {"batch_size": 2, "shuffle": train},
    )
    model = _TinyLocModel()
    # Adam moves a parameter by the learning rate in its first step, whatever
    # the size of the gradient. One epoch of one step, without warm-up.
    model_training.train_model(
        fingerprints=Dataset(
            ConstantDataLoader(torch.ones(1)),
            IdentifierSet([Identifier(0, 0), Identifier(1, 0)]),
        ),
        minutia_maps=Dataset(
            ConstantDataLoader((torch.zeros(0), 0.0)),
            IdentifierSet([Identifier(0, 0), Identifier(1, 0)]),
        ),
        labels=Dataset(
            ConstantDataLoader(0), IdentifierSet([Identifier(0, 0), Identifier(1, 0)])
        ),
        validation_fingerprints=None,
        validation_benchmark=None,
        model=model,
        loss=_SumLoss(),
        num_epochs=1,
        out_dir=str(tmp_path),
        learning_rate=LearningRateConfig(
            learning_rate=0.01,
            warmup_steps=0,
            localization_factor=0.1,
            loss_explosion_factor=0,
        ),
    )
    assert model.embeddings.weight.item() == pytest.approx(-0.01, rel=1e-3)
    assert model.localization.weight.item() == pytest.approx(-0.001, rel=1e-3)


def test_optimizer_state_with_one_group_is_skipped_with_a_note(tmp_path, capsys):
    # A parameter file of a Loc* variant written before the second group existed
    model = _TinyLocModel()
    old_optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    path = str(tmp_path / "model.pyt")
    save_model_parameters(path, model, _SumLoss(), old_optimizer)

    new_model = _TinyLocModel()
    torch.nn.init.ones_(new_model.embeddings.weight)
    optimizer = torch.optim.Adam(parameter_groups(new_model, 0.035), lr=1.0)
    load_model_parameters(path, new_model, _SumLoss(), optimizer)
    assert new_model.embeddings.weight.item() == 0  # the model was loaded
    assert len(optimizer.param_groups) == 2
    assert "parameter group" in capsys.readouterr().out


def test_optimizer_state_with_two_groups_is_loaded(tmp_path):
    model = _TinyLocModel()
    optimizer = torch.optim.Adam(parameter_groups(model, 0.035), lr=1.0)
    model(torch.ones(2, 1)).texture_logits.sum().backward()
    optimizer.step()
    path = str(tmp_path / "model.pyt")
    save_model_parameters(path, model, _SumLoss(), optimizer)

    new_model = _TinyLocModel()
    new_optimizer = torch.optim.Adam(parameter_groups(new_model, 0.5), lr=1.0)
    load_model_parameters(path, new_model, _SumLoss(), new_optimizer)
    assert len(new_optimizer.state) == 2  # the running averages of Adam
