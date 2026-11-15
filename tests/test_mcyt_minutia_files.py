from os.path import join, dirname

import pytest

from flx.data.dataset import Identifier
from flx.data.minutia_map_loader import (
    MCYTOpticalMinutiaMapLoader,
    MCYTCapacitiveMinutiaMapLoader,
)

TEST_DATA_DIR = join(dirname(__file__), "data")

# x, y, angle in radians, and the class and score columns that newer files add
MINUTIAE = [
    (137.0, 187.0, 1.623156, 1, 0.9),
    (162.25, 251.5, 5.358161, 0, 0.5),
    (30.0, 90.0, 0.5, 2, 0.1),
]

LOADERS_AND_FILES = [
    (MCYTOpticalMinutiaMapLoader, "dp_0000_0_8.mnt", Identifier(0, 8)),
    (MCYTCapacitiveMinutiaMapLoader, "pb_0000_0_0.bmp.mnt", Identifier(0, 0)),
    # scripts/extract_minutiae.py names the file without the image extension
    (MCYTCapacitiveMinutiaMapLoader, "pb_0000_0_0.mnt", Identifier(0, 0)),
]


def _write_mnt_file(path: str, n_columns: int, separator: str) -> None:
    with open(path, "w") as file:
        file.write(f"{len(MINUTIAE)}\n256 400\n")
        for minutia in MINUTIAE:
            file.write(separator.join(str(v) for v in minutia[:n_columns]) + "\n")
        file.write("\n")


@pytest.mark.parametrize("loader_type,filename,identifier", LOADERS_AND_FILES)
def test_mnt_files_with_more_columns_give_the_same_minutiae(
    tmp_path, loader_type, filename, identifier
):
    results = []
    for n_columns, separator in [(3, " "), (4, " "), (5, " "), (5, "\t"), (5, "  ")]:
        directory = tmp_path / f"{n_columns}_{len(results)}"
        directory.mkdir()
        _write_mnt_file(str(directory / filename), n_columns, separator)
        loader = loader_type(str(directory))
        assert list(loader.ids) == [identifier]
        locs, oris = loader.get_minutiae(identifier)
        assert locs.shape == (len(MINUTIAE), 2)
        assert oris.shape == (len(MINUTIAE),)
        minutia_map, weight = loader.get(identifier)
        results.append((locs, oris, minutia_map))
    for locs, oris, minutia_map in results[1:]:
        assert (locs == results[0][0]).all()
        assert (oris == results[0][1]).all()
        assert (minutia_map == results[0][2]).all()


def test_mcyt_example_files_are_read_completely():
    optical = MCYTOpticalMinutiaMapLoader(join(TEST_DATA_DIR, "MCYTOpticalExample"))
    assert len(optical.get_minutiae(Identifier(0, 8))[0]) == 57
    capacitive = MCYTCapacitiveMinutiaMapLoader(
        join(TEST_DATA_DIR, "MCYTCapacitiveExample")
    )
    assert len(capacitive.get_minutiae(Identifier(0, 0))[0]) == 44
