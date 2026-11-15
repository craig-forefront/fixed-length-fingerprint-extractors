from os.path import join

import numpy as np
import pytest

from flx.data.dataset import Identifier
from flx.data.fingerflow_minutia_loader import (
    FingerFlowMinutiaMapLoader,
    FingerFlowMinutiaMapLoaderCustomNaming,
    read_mnt_file,
)
from flx.setup.config import INPUT_SIZE

# x, y, angle, class, direction_confidence
MINUTIAE = [
    (100.0, 150.0, 0.0, 0, 0.93),
    (200.0, 250.0, np.pi / 3, 1, 0.0),
    (30.0, 350.0, 2 * np.pi - np.pi / 3, 5, 0.12),
]
WIDTH = 292
HEIGHT = 392


def _write_mnt_file(path: str, n_columns: int, minutiae: list = MINUTIAE) -> None:
    with open(path, "w") as file:
        file.write(f"{len(minutiae)}\n")
        file.write(f"{WIDTH} {HEIGHT}\n")
        for minutia in minutiae:
            file.write(" ".join(str(v) for v in minutia[:n_columns]) + "\n")


@pytest.mark.parametrize("n_columns", [3, 4, 5])
def test_read_mnt_file(tmp_path, n_columns: int):
    path = join(tmp_path, "1_1.mnt")
    _write_mnt_file(path, n_columns)
    locs, oris, size = read_mnt_file(path)
    assert size == (WIDTH, HEIGHT)
    assert locs == pytest.approx(np.array([m[:2] for m in MINUTIAE]))
    assert oris == pytest.approx(np.array([m[2] for m in MINUTIAE]))


def test_read_mnt_file_without_minutiae(tmp_path):
    path = join(tmp_path, "1_1.mnt")
    _write_mnt_file(path, 5, minutiae=[])
    locs, oris, size = read_mnt_file(path)
    assert size == (WIDTH, HEIGHT)
    assert locs.shape == (0, 2)
    assert oris.shape == (0,)


def _default_naming_loader(root_dir: str) -> tuple[FingerFlowMinutiaMapLoader, Identifier]:
    _write_mnt_file(join(root_dir, "1_1.mnt"), 5)
    return FingerFlowMinutiaMapLoader(root_dir), Identifier(0, 0)


def _custom_naming_loader(
    root_dir: str,
) -> tuple[FingerFlowMinutiaMapLoaderCustomNaming, Identifier]:
    _write_mnt_file(join(root_dir, "finger7.mnt"), 5)
    loader = FingerFlowMinutiaMapLoaderCustomNaming(
        root_dir, lambda _, filename: Identifier(int(filename[6:]), 0)
    )
    return loader, Identifier(7, 0)


def _assert_loads_minutiae(loader, identifier: Identifier) -> None:
    assert list(loader.ids) == [identifier]

    # The image is padded to a square and resized to the input size. All minutiae
    # are kept, also those with a low direction confidence.
    locs, oris = loader.get_minutiae(identifier)
    scale = INPUT_SIZE / HEIGHT
    expected = [((m[0] + (HEIGHT - WIDTH) / 2) * scale, m[1] * scale) for m in MINUTIAE]
    assert locs == pytest.approx(np.array(expected), abs=0.1)
    assert oris == pytest.approx(np.array([m[2] for m in MINUTIAE]))

    minutia_map, weight = loader.get(identifier)
    assert minutia_map.shape == (6, 128, 128)
    assert weight == 1.0
    # The k-th layer is the direction k * 60 degrees
    for (x, y), layer in zip(expected, [0, 1, 5]):
        peak_y, peak_x = np.unravel_index(int(minutia_map[layer].argmax()), (128, 128))
        assert abs(peak_x - x * 128 / INPUT_SIZE) <= 1
        assert abs(peak_y - y * 128 / INPUT_SIZE) <= 1
        assert minutia_map[:, peak_y, peak_x].argmax() == layer


@pytest.mark.parametrize("make_loader", [_default_naming_loader, _custom_naming_loader])
def test_fingerflow_minutia_map_loader(tmp_path, make_loader):
    loader, identifier = make_loader(str(tmp_path))
    _assert_loads_minutiae(loader, identifier)


def test_fingerflow_minutia_map_loader_without_extra_columns(tmp_path):
    _write_mnt_file(join(tmp_path, "1_1.mnt"), 3)
    _assert_loads_minutiae(FingerFlowMinutiaMapLoader(str(tmp_path)), Identifier(0, 0))
