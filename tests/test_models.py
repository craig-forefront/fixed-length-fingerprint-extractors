import torch

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
