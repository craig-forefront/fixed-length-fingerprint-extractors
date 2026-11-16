import pytest
import torch

from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.deep_print_loss import (
    DeepPrintLoss_Tex,
    DeepPrintLoss_TexMinu,
    LossWeights,
)
from flx.setup import config
from flx.models.localization_network import LocalizationNetwork


def test_localization_network_is_identity_at_initialisation_for_a_batch():
    torch.manual_seed(0)
    net = LocalizationNetwork().eval()
    images = torch.rand(5, 1, 299, 299)
    with torch.no_grad():
        assert (net(images) - images).abs().max() < 1e-4


def test_localization_network_treats_each_sample_of_a_batch_on_its_own():
    torch.manual_seed(0)
    net = LocalizationNetwork().eval()
    # Leave the identity initialisation, so that the samples get different poses
    torch.nn.init.normal_(net.fc_loc[2].weight, std=0.01)
    net.fc_loc[2].bias.data = torch.tensor([0.3, 0.1, -0.2])
    images = torch.rand(5, 1, 299, 299)
    with torch.no_grad():
        batch = net(images)
        single = torch.cat([net(images[i : i + 1]) for i in range(5)])
    assert (batch - single).abs().max() < 1e-4


def _record_one_epoch(loss_fun: DeepPrintLoss_TexMinu) -> dict:
    loss_fun.reset_recorded_loss()
    generator = torch.Generator().manual_seed(1)
    for _ in range(3):
        embeddings = torch.nn.functional.normalize(
            torch.randn(4, 16, generator=generator), dim=1
        )
        logits = torch.randn(4, 6, generator=generator)
        output = DeepPrintTrainingOutput(
            minutia_logits=logits,
            texture_logits=logits,
            minutia_maps=torch.zeros(4, 6, 8, 8),
            minutia_embeddings=embeddings,
            texture_embeddings=embeddings,
        )
        loss_fun(output, torch.tensor([0, 1, 2, 3]), torch.ones(4, 6, 8, 8), torch.ones(4))
    return loss_fun.get_recorded_loss()


def test_recorded_loss_is_the_same_for_two_identical_epochs():
    loss_fun = DeepPrintLoss_TexMinu(6, 16, 16)
    # Keep the class centers fixed, so that both epochs see the same loss
    loss_fun.minu_loss_fun.center_loss_fun.alpha = 0
    loss_fun.texture_loss_fun.center_loss_fun.alpha = 0
    first = _record_one_epoch(loss_fun)
    assert first["minutia_map_loss"] > 0
    assert _record_one_epoch(loss_fun) == first


def test_recorded_loss_is_the_mean_per_sample():
    # The cross entropy and the minutia map loss are means over the batch. The
    # recorded value must be that mean, not the mean divided by the batch size.
    loss_fun = DeepPrintLoss_TexMinu(6, 16, 16)
    recorded = _record_one_epoch(loss_fun)

    generator = torch.Generator().manual_seed(1)
    labels = torch.tensor([0, 1, 2, 3])
    crossent = []
    for _ in range(3):
        torch.randn(4, 16, generator=generator)
        logits = torch.randn(4, 6, generator=generator)
        crossent.append(float(torch.nn.functional.cross_entropy(logits, labels)))
    expected_crossent = sum(crossent) / 3
    assert recorded["texture_loss"]["crossent_loss_sum"] == pytest.approx(
        expected_crossent
    )
    assert recorded["minutia_loss"]["crossent_loss_sum"] == pytest.approx(
        expected_crossent
    )
    # Predicted maps are all 0 and the target maps all 1: squared error 1
    assert recorded["minutia_map_loss"] == pytest.approx(config.LOSS_WEIGHT_MINUTIA_MAP)


def test_recorded_loss_does_not_depend_on_the_batch_size():
    torch.manual_seed(2)
    embeddings = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    logits = torch.randn(8, 6)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    recorded = []
    for batch_size in (8, 2):
        loss_fun = DeepPrintLoss_TexMinu(6, 16, 16)
        for part in (loss_fun.minu_loss_fun, loss_fun.texture_loss_fun):
            part.center_loss_fun.alpha = 0
            part.center_loss_fun.centers.zero_()
        for i in range(0, 8, batch_size):
            j = i + batch_size
            output = DeepPrintTrainingOutput(
                minutia_logits=logits[i:j],
                texture_logits=logits[i:j],
                minutia_maps=torch.zeros(batch_size, 6, 8, 8),
                minutia_embeddings=embeddings[i:j],
                texture_embeddings=embeddings[i:j],
            )
            loss_fun(
                output,
                labels[i:j],
                torch.ones(batch_size, 6, 8, 8),
                torch.ones(batch_size),
            )
        recorded.append(loss_fun.get_recorded_loss())
    for part in ("texture_loss", "minutia_loss"):
        for key in ("crossent_loss_sum", "center_loss_sum"):
            assert recorded[0][part][key] == pytest.approx(recorded[1][part][key])
    assert recorded[0]["minutia_map_loss"] == pytest.approx(
        recorded[1]["minutia_map_loss"]
    )


def test_texminu_loss_accepts_different_embedding_sizes():
    loss_fun = DeepPrintLoss_TexMinu(
        num_classes=6, texture_embedding_dims=16, minutia_embedding_dims=32
    )
    assert loss_fun.texture_loss_fun.center_loss_fun.centers.shape == (6, 16)
    assert loss_fun.minu_loss_fun.center_loss_fun.centers.shape == (6, 32)
    output = DeepPrintTrainingOutput(
        minutia_logits=torch.randn(4, 6),
        texture_logits=torch.randn(4, 6),
        minutia_maps=torch.zeros(4, 6, 8, 8),
        minutia_embeddings=torch.randn(4, 32),
        texture_embeddings=torch.randn(4, 16),
    )
    loss = loss_fun(
        output, torch.tensor([0, 1, 2, 3]), torch.ones(4, 6, 8, 8), torch.ones(4)
    )
    assert torch.isfinite(loss)
