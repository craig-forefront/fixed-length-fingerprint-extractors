import torch

from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.deep_print_loss import DeepPrintLoss_TexMinu
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
