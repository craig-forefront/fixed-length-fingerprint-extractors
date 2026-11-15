import numpy as np

from flx.data.minutia_map import create_minutia_map


def test_create_minutia_map():
    locations = np.array([(100, 100), (300, 300), (100, 200), (350, 50)])
    # One minutia for each of the layers 0, 3, 1 and 5
    orientations = np.array([0.0, np.pi, np.pi / 3, -np.pi / 3])
    minutia_map = create_minutia_map(
        locations,
        orientations,
        in_resolution=(400, 400),
        out_resolution=(128, 128),
        n_layers=6,
        sigma=1.5,
    )
    assert minutia_map.shape == (128, 128, 6)
    assert minutia_map.dtype == np.uint8

    for (x, y), layer in zip(locations, [0, 3, 1, 5]):
        # The minutia is the brightest point of its layer
        peak_y, peak_x = np.unravel_index(
            minutia_map[:, :, layer].argmax(), (128, 128)
        )
        assert abs(peak_x - x * 128 / 400) <= 1
        assert abs(peak_y - y * 128 / 400) <= 1
        assert minutia_map[peak_y, peak_x].argmax() == layer


def test_create_minutia_map_without_minutiae():
    minutia_map = create_minutia_map(
        np.zeros((0, 2)),
        np.zeros(0),
        in_resolution=(400, 400),
        out_resolution=(128, 128),
        n_layers=6,
        sigma=1.5,
    )
    assert minutia_map.shape == (128, 128, 6)
    assert not minutia_map.any()
