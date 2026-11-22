"""
Tests for flx/data/sample_table_loader.py: the loaders that serve the images
and the minutiae of a sample table to the training.

The central tests compare them with the loaders of directory mode on the same
files: CustomFingerprintLoader of scripts/train_500k.py with a
SensorNormalizer for the images, FingerFlowMinutiaMapLoader for the .mnt
files. A table whose rows hold what the SensorNormalizer reports about every
file must give the same tensors and arrays, from the loaders alone and
through the image and minutia pipeline of the training.

The files of that comparison (PRINTS) are chosen by the fault that each of
them shows, and a control assertion beside every comparison shows that the
fault would be seen:
    - a resolution that is another number in 32 bits, on a file whose
      prepared size depends on it;
    - a frame in which the locations of the minutiae come out differently
      when the sizes are given as numpy numbers;
    - prints at 500, 1000 and 385 ppi, large enough for the real binariser;
    - light ridges that the preparation has to find out from the image;
    - a print with a single minutia, which is one minutia and not none;
    - .mnt files without minutiae: one of 0 bytes, one with the header 0.

The rows are made here as scripts/build_sample_table.py makes them, from
SensorNormalizer.load and read_mnt_file; the script itself is not used. The
.mnt files are written as text, and the script of the training is imported
from its file: no test module is imported.
"""

import copy
import dataclasses
import importlib.util
import os
import pickle
import random
import tracemalloc
import weakref
from os.path import abspath, dirname, join
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as VTF
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from flx.data import sample_table_loader
from flx.data.dataset import DataLoader, Dataset, Identifier, IdentifierSet
from flx.data.fingerflow_minutia_loader import FingerFlowMinutiaMapLoader, read_mnt_file
from flx.data.image_helpers import transform_to_input_size
from flx.data.label_index import LabelIndex
from flx.data.minutia_map_loader import MinutiaMapLoader, PoseTransformedMinutiaMapLoader
from flx.data.sample_table_loader import (
    SampleTable,
    SampleTableImageLoader,
    SampleTableMinutiaMapLoader,
)
from flx.preparation.sample_table import (
    PPI_ORIGINS,
    RIDGES_ORIGINS,
    SampleRow,
    Samples,
    SampleTableError,
    samples_checksum,
    samples_from_rows,
    write_part,
)
from flx.preparation.sensor_normalization import (
    ImageReadError,
    SensorNormalizer,
    resampled_size,
)
from flx.setup.config import INPUT_SIZE

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")
BUILDER = "scripts/build_sample_table.py"

# The arrays of Samples, by the names of its fields
ARRAYS = [field.name for field in dataclasses.fields(Samples) if field.type is np.ndarray]

# The largest kernel of the binariser of the training: an image needs at
# least this many pixels per side
LARGEST_BINARISER_KERNEL = 41


@pytest.fixture(scope="module")
def train_500k():
    spec = importlib.util.spec_from_file_location("train_500k", join(SCRIPTS_DIR, "train_500k.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Files: prints, .mnt files and the rows of their table
# ---------------------------------------------------------------------------


def ridge_image(height: int, width: int, period: float = 9.5, slant: float = 0.3) -> np.ndarray:
    """Dark ridges on a white background: a print in the middle of a white frame."""
    y, x = np.mgrid[0:height, 0:width]
    ridges = 127.5 + 127.5 * np.cos(2 * np.pi * (x + slant * y) / period)
    image = np.full((height, width), 255.0)
    inside = (y > height // 6) & (y < 5 * height // 6) & (x > width // 6) & (x < 5 * width // 6)
    image[inside] = ridges[inside]
    return image.astype(np.uint8)


def save(path, image: np.ndarray, ppi=None) -> str:
    """Saves the image, with the resolution in the file metadata if ppi is given."""
    os.makedirs(dirname(str(path)), exist_ok=True)
    if ppi is None:
        Image.fromarray(image).save(str(path))
    else:
        Image.fromarray(image).save(str(path), dpi=(ppi, ppi))
    return str(path)


def write_mnt_file(path, minutiae, size: tuple, comment: str) -> None:
    """
    A .mnt file as scripts/extract_minutiae.py writes it: the number of
    minutiae, the size of the prepared image, a line per minutia (x, y,
    angle, class, direction confidence), and how the image was prepared.
    """
    lines = [str(len(minutiae)), f"{size[0]} {size[1]}"]
    lines += [f"{x:.4f} {y:.4f} {angle:.6f} 0 0.9000" for x, y, angle in minutiae]
    lines.append(f"# {comment}")
    os.makedirs(dirname(str(path)), exist_ok=True)
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def identifier_of(image_path: str) -> Identifier:
    """The identifier that directory mode reads from the name <subject>_<impression>.png."""
    subject, impression = os.path.splitext(os.path.basename(image_path))[0].split("_")
    return Identifier(int(subject) - 1, int(impression) - 1)


def rows_as_the_builder_fills_them(image_dir, minutiae_dir, normalizer) -> list:
    """
    One row per .png file below the image directory: what
    SensorNormalizer.load reports about the file, and the minutiae of the
    .mnt file under the same relative path below the minutiae directory
    (null where there is none).
    """
    rows = []
    for directory, _, names in sorted(os.walk(image_dir)):
        for name in sorted(names):
            if not name.endswith(".png"):
                continue
            path = join(directory, name)
            relative = os.path.relpath(path, image_dir).replace(os.sep, "/")
            _, info = normalizer.load(path, check_polarity=True)
            minutiae = None
            mnt_path = join(str(minutiae_dir), os.path.splitext(relative)[0] + ".mnt")
            if os.path.exists(mnt_path):
                locations, angles, _ = read_mnt_file(mnt_path)
                minutiae = np.column_stack([locations.reshape(-1, 2), angles])
            identifier = identifier_of(relative)
            rows.append(
                SampleRow(
                    subject=identifier.subject,
                    impression=identifier.impression,
                    image_path=relative,
                    file_size=os.path.getsize(path),
                    ppi=info.ppi,
                    ppi_origin=info.ppi_origin,
                    ridges=info.ridges,
                    ridges_origin=info.ridges_origin,
                    source_width=info.source_width,
                    source_height=info.source_height,
                    width=info.width,
                    height=info.height,
                    minutiae=minutiae,
                )
            )
    return rows


class Print(NamedTuple):
    """A print of the comparison with directory mode."""

    path: str  # below the image directory
    size: tuple  # (width, height) of the image file
    ppi: float  # resolution of the file
    stated: str  # where the resolution is stated: "metadata" of the file or "manifest"
    light: bool  # light ridges on a dark background
    # The .mnt file: a number of minutiae; 0 for a file with the header 0;
    # None for a file of 0 bytes
    minutiae: Optional[int]


PRINTS = [
    # At 500 ppi, used as it is. In its frame of 500x640 pixels the locations
    # of minutiae depend on whether the sizes are Python or numpy numbers.
    Print("sensor_a/1_1.png", (500, 640), 500, "metadata", False, 6),
    # At 1000 ppi: halved to 292x392
    Print("sensor_a/1_2.png", (584, 784), 1000, "metadata", False, 5),
    # At 385 ppi: enlarged to 300x400
    Print("sensor_b/2_1.png", (231, 308), 385, "metadata", False, 5),
    # 157 pixels at 251.2 ppi are 312 pixels at 500 ppi, and 313 with the
    # resolution rounded to 32 bits
    Print("sensor_b/2_2.png", (157, 201), 251.2, "manifest", False, 5),
    # Light ridges, which the preparation has to find out from the image. Its
    # .mnt file holds a single minutia: the smallest number that is not none.
    Print("3_1.png", (256, 320), 500, "metadata", True, 1),
    # A .mnt file of 0 bytes
    Print("3_2.png", (240, 300), 500, "metadata", False, None),
    # A .mnt file with the header 0, as the extraction writes it for an image
    # in which it finds no minutiae
    Print("4_1.png", (300, 220), 500, "metadata", False, 0),
    # Wider than high
    Print("4_2.png", (400, 300), 500, "metadata", False, 5),
]
FRAME_OF_MOVED_LOCATIONS, FLOAT32_PPI, LIGHT, ZERO_BYTES, HEADER_ZERO = (
    PRINTS[0],
    PRINTS[3],
    PRINTS[4],
    PRINTS[5],
    PRINTS[6],
)


def print_image(index: int) -> np.ndarray:
    """The image file of a print: ridges of the same physical width, another slant per print."""
    spec = PRINTS[index]
    width, height = spec.size
    image = ridge_image(height, width, period=9.5 * spec.ppi / 500, slant=0.1 + 0.15 * index)
    return 255 - image if spec.light else image


def minutiae_of_print(index: int, size: tuple) -> np.ndarray:
    """
    The minutiae (x, y, angle) of a print in the pixels of its prepared
    image of the given (width, height): positions with fractions, spread
    over the image.
    """
    random_numbers = np.random.default_rng(100 + index)
    count = PRINTS[index].minutiae
    minutiae = np.column_stack(
        [
            random_numbers.uniform(8, size[0] - 8, count),
            random_numbers.uniform(8, size[1] - 8, count),
            random_numbers.uniform(0, 2 * np.pi, count),
        ]
    )
    if PRINTS[index] is FRAME_OF_MOVED_LOCATIONS:
        # Positions without fractions at which the two ways of scaling differ
        minutiae[0] = (175.0, 300.0, 0.5)
        minutiae[1] = (45.0, 100.0, 4.0)
    return minutiae


class Corpus(NamedTuple):
    """The files of the comparison, their table, and the preparation of directory mode."""

    image_dir: str
    minutiae_dir: str
    table_dir: str
    normalizer: SensorNormalizer  # resolves resolution and polarity as directory mode does
    rows: list
    table: SampleTable  # read from its Parquet files

    def identifier(self, spec: Print) -> Identifier:
        return identifier_of(spec.path)

    def row(self, spec: Print) -> int:
        return self.table.row(identifier_of(spec.path))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Corpus:
    """
    The prints as image files, the .mnt files of an extraction, and the
    table of both, written as two parts and read again. No test changes
    these files.
    """
    pytest.importorskip("pyarrow")
    root = tmp_path_factory.mktemp("corpus")
    image_dir, minutiae_dir = root / "images", root / "minutiae"
    manifest = ["path,ppi"]
    for index, spec in enumerate(PRINTS):
        if spec.stated == "manifest":
            manifest.append(f"{spec.path},{spec.ppi}")
        save(
            image_dir / spec.path,
            print_image(index),
            spec.ppi if spec.stated == "metadata" else None,
        )
    (root / "resolutions.csv").write_text("\n".join(manifest) + "\n")
    normalizer = SensorNormalizer(
        root_dir=str(image_dir), manifest=str(root / "resolutions.csv"), ridges="auto"
    )

    for index, spec in enumerate(PRINTS):
        mnt_path = minutiae_dir / (os.path.splitext(spec.path)[0] + ".mnt")
        if spec.minutiae is None:
            os.makedirs(dirname(str(mnt_path)), exist_ok=True)
            open(mnt_path, "w").close()
            continue
        _, info = normalizer.load(str(image_dir / spec.path), check_polarity=True)
        size = (info.width, info.height)
        write_mnt_file(mnt_path, minutiae_of_print(index, size), size, info.describe())

    rows = rows_as_the_builder_fills_them(str(image_dir), str(minutiae_dir), normalizer)
    for part in range(2):
        write_part(str(root / "table"), part, 2, samples_from_rows(rows[part::2]), {"id": "corpus"})
    table = SampleTable.read(str(root / "table"), str(image_dir))
    return Corpus(str(image_dir), str(minutiae_dir), str(root / "table"), normalizer, rows, table)


def same_arrays(found: Samples, expected: Samples) -> bool:
    return all(
        getattr(found, name).dtype == getattr(expected, name).dtype
        and getattr(found, name).tobytes() == getattr(expected, name).tobytes()
        for name in ARRAYS
    )


# ---------------------------------------------------------------------------
# The same tensors as directory mode
# ---------------------------------------------------------------------------


def test_the_files_of_the_comparison_are_those_that_show_the_faults(corpus):
    """The controls that belong to the files themselves; the others are beside the comparisons."""
    table, samples = corpus.table, corpus.table.samples
    assert table.num_rows == len(PRINTS) == len(corpus.rows)
    # The table that was read from its files holds the rows as they were made
    assert same_arrays(samples, samples_from_rows(corpus.rows))

    # Prints at 500, 1000 and 385 ppi, by their own metadata, and one by the manifest
    stated = {
        (ppi, PPI_ORIGINS[origin])
        for ppi, origin in zip(samples.ppi.tolist(), samples.ppi_origin.tolist())
    }
    assert stated == {
        (500.0, "metadata"),
        (1000.0, "metadata"),
        (385.0, "metadata"),
        (251.2, "manifest"),
    }
    # Each is resampled to 500 ppi, and the real binariser has room in all of them
    for spec in PRINTS:
        row = corpus.row(spec)
        prepared = (int(samples.width[row]), int(samples.height[row]))
        assert prepared == resampled_size(*spec.size, spec.ppi)
        assert min(prepared) >= LARGEST_BINARISER_KERNEL
        assert (prepared == spec.size) == (spec.ppi == 500)

    # The resolution of one file is another number in 32 bits, and its
    # prepared size depends on that
    row = corpus.row(FLOAT32_PPI)
    ppi = float(samples.ppi[row])
    assert ppi == FLOAT32_PPI.ppi and float(np.float32(ppi)) != ppi
    assert resampled_size(*FLOAT32_PPI.size, ppi) == (312, 400)
    assert resampled_size(*FLOAT32_PPI.size, float(np.float32(ppi))) == (313, 400)
    assert (samples.width[row], samples.height[row]) == (312, 400)

    # The light ridges were found out from the image, as were the dark ones
    # of the other prints
    light = corpus.row(LIGHT)
    assert samples.ridges_light.tolist() == [row == light for row in range(len(PRINTS))]
    assert {RIDGES_ORIGINS[origin] for origin in samples.ridges_origin.tolist()} == {"estimated"}

    # Two .mnt files without minutiae; the rows of both hold an empty list, not null
    assert os.path.getsize(join(corpus.minutiae_dir, "3_2.mnt")) == 0
    with open(join(corpus.minutiae_dir, "4_1.mnt")) as file:
        assert file.read().startswith("0\n300 220\n")
    for spec in PRINTS:
        row = corpus.row(spec)
        assert samples.minutiae_known[row]
        assert len(samples.minutiae_of(row)) == (spec.minutiae or 0)
    assert len(table.rows_without_minutiae()) == 0
    # One print with a single minutia, beside those with several and with none
    assert sorted(spec.minutiae or 0 for spec in PRINTS) == [0, 0, 1, 5, 5, 5, 5, 6]


def test_images_of_a_table_are_the_tensors_of_directory_mode(corpus, train_500k, monkeypatch):
    directory = train_500k.CustomFingerprintLoader(corpus.image_dir, corpus.normalizer)
    loader = SampleTableImageLoader(corpus.table)
    assert set(loader.ids.identifiers) == set(directory.ids.identifiers)
    assert len(loader.ids) == len(directory.ids) == len(PRINTS)

    expected = {identifier: directory.get(identifier) for identifier in directory.ids}
    for spec in PRINTS:
        tensor = loader.get(corpus.identifier(spec))
        prepared = resampled_size(*spec.size, spec.ppi)
        assert tensor.shape == (1, prepared[1], prepared[0])
        assert tensor.dtype == expected[corpus.identifier(spec)].dtype == torch.float32
        assert torch.equal(tensor, expected[corpus.identifier(spec)]), spec.path

    # Control: the tensors are those of the prints, each another one, with
    # the ridges of the light print dark like the others
    assert torch.equal(expected[corpus.identifier(PRINTS[0])], VTF.to_tensor(print_image(0)))
    assert torch.equal(
        expected[corpus.identifier(LIGHT)], VTF.to_tensor(255 - print_image(PRINTS.index(LIGHT)))
    )

    # The loader keeps one SensorNormalizer per resolution and polarity, up
    # to a limit. With a limit of one it makes a new one for nearly every
    # file of this table, and the tensors are the same.
    monkeypatch.setattr(sample_table_loader, "_MAX_NORMALIZERS", 1)
    loader = SampleTableImageLoader(corpus.table)
    for _ in range(2):
        for identifier in directory.ids:
            assert torch.equal(loader.get(identifier), expected[identifier])


def test_the_image_loader_keeps_one_normalizer_per_resolution_and_polarity_up_to_its_limit(
    corpus, tmp_path, monkeypatch
):
    made, alive = [], weakref.WeakSet()

    class Counted(SensorNormalizer):
        """A SensorNormalizer that is counted when it is made and while it is kept."""

        def __init__(self, **settings):
            super().__init__(**settings)
            made.append((settings["ppi"], settings["ridges"]))
            alive.add(self)

    monkeypatch.setattr(sample_table_loader, "SensorNormalizer", Counted)
    loader = SampleTableImageLoader(corpus.table)
    for _ in range(2):
        for identifier in corpus.table.ids:
            loader.get(identifier)
    # One for every pair of resolution and polarity of the table, made once
    pairs = [(251.2, "dark"), (385.0, "dark"), (500.0, "dark"), (500.0, "light"), (1000.0, "dark")]
    assert sorted(made) == pairs
    assert len(alive) == len(pairs)

    # Resolutions that differ by a fraction are two pairs
    table = small_table(tmp_path)
    close = dataclasses.replace(table.samples, ppi=np.array([500.0, 500.4, 500.0]))
    made.clear()
    loader = SampleTableImageLoader(SampleTable(close, table.image_dir))
    for identifier in table.identifiers:
        assert loader.get(identifier).shape == (1, 120, 96)
    assert made == [(500.0, "dark"), (500.4, "dark")]

    # With a limit of two no more than two are kept at a time, and a pair
    # that was let go is made again when a file needs it
    del loader
    made.clear()
    monkeypatch.setattr(sample_table_loader, "_MAX_NORMALIZERS", 2)
    loader = SampleTableImageLoader(corpus.table)
    for _ in range(2):
        for identifier in corpus.table.ids:
            loader.get(identifier)
            assert len(alive) <= 2
    assert len(made) > len(pairs) and set(made) == set(pairs)


def test_a_table_with_another_polarity_or_resolution_would_show_in_the_images(corpus, train_500k):
    """
    Control of the comparison of the images: rows that hold something else
    than directory mode resolves do not give its tensors.
    """
    directory = train_500k.CustomFingerprintLoader(corpus.image_dir, corpus.normalizer)
    samples = corpus.table.samples

    # All ridges taken as dark: the light print stays as it is in the file
    dark = dataclasses.replace(samples, ridges_light=np.zeros(len(samples), dtype=np.bool_))
    loader = SampleTableImageLoader(SampleTable(dark, corpus.image_dir))
    for spec in PRINTS:
        identifier = corpus.identifier(spec)
        assert torch.equal(loader.get(identifier), directory.get(identifier)) == (spec is not LIGHT)

    # The resolution through 32 bits: such a table is refused, ...
    rounded = dataclasses.replace(samples, ppi=samples.ppi.astype(np.float32).astype(np.float64))
    with pytest.raises(SampleTableError, match="does not follow from the source size and the ppi"):
        SampleTable(rounded, corpus.image_dir)
    # ... and without that check the loader stops at the file whose prepared
    # size is another one, and only there
    loader = SampleTableImageLoader(SampleTable(rounded, corpus.image_dir, validate=False))
    for spec in PRINTS:
        identifier = corpus.identifier(spec)
        if spec is FLOAT32_PPI:
            with pytest.raises(SampleTableError) as error:
                loader.get(identifier)
            message = str(error.value)
            assert message.startswith(join(corpus.image_dir, "sensor_b", "2_2.png") + " is not")
            assert "holds 312x400 as the size in pixels (width x height) of the image" in message
            assert "prepared at 500 ppi, the file has 313x400" in message
            assert BUILDER in message
        else:
            assert torch.equal(loader.get(identifier), directory.get(identifier))


def test_minutiae_of_a_table_are_the_arrays_of_directory_mode(corpus):
    directory = FingerFlowMinutiaMapLoader(corpus.minutiae_dir)
    loader = SampleTableMinutiaMapLoader(corpus.table)
    assert set(loader.ids.identifiers) == set(directory.ids.identifiers)
    samples = corpus.table.samples

    for spec in PRINTS:
        identifier = corpus.identifier(spec)
        expected_locations, expected_angles = directory.get_minutiae(identifier)
        locations, angles = loader.get_minutiae(identifier)
        assert len(locations) == len(expected_locations) == (spec.minutiae or 0), spec.path
        assert len(angles) == len(expected_angles)
        if spec.minutiae:
            assert locations.dtype == expected_locations.dtype == np.float16
            assert angles.dtype == expected_angles.dtype == np.float32
            assert locations.shape == expected_locations.shape == (spec.minutiae, 2)
            assert np.array_equal(locations, expected_locations), spec.path
            assert np.array_equal(angles, expected_angles), spec.path
            # The frame is the header of the .mnt file: the prepared size
            frame = loader.get_frame_size(identifier)
            assert frame == directory.get_frame_size(identifier)
            assert frame == resampled_size(*spec.size, spec.ppi)
        else:
            assert locations.shape == expected_locations.shape == (0, 2)
            assert angles.shape == expected_angles.shape == (0,)
            assert locations.dtype == expected_locations.dtype
            assert angles.dtype == expected_angles.dtype
        assert [type(side) for side in loader.get_frame_size(identifier)] == [int, int]

        expected_map, expected_weight = directory.get(identifier)
        minutia_map, weight = loader.get(identifier)
        assert minutia_map.dtype == expected_map.dtype
        assert torch.equal(minutia_map, expected_map), spec.path
        assert weight == expected_weight == (1.0 if spec.minutiae else 0.0)
        assert bool(minutia_map.any()) == bool(spec.minutiae)

    # Control: in the frame of 500x640 pixels the locations depend on
    # whether the sizes are Python numbers, as the loader of .mnt files reads
    # them from the header, or numpy numbers, as the arrays of a table hold them
    identifier = corpus.identifier(FRAME_OF_MOVED_LOCATIONS)
    in_file, _, (width, height) = read_mnt_file(join(corpus.minutiae_dir, "sensor_a", "1_1.mnt"))
    assert (width, height) == (500, 640) and in_file[0].tolist() == [175.0, 300.0]
    with_python_numbers = transform_to_input_size(in_file, height, width)
    with_numpy_numbers = transform_to_input_size(in_file, np.int32(height), np.int32(width))
    assert np.array_equal(loader.get_minutiae(identifier)[0], with_python_numbers)
    row = corpus.row(FRAME_OF_MOVED_LOCATIONS)
    assert type(samples.width[row]) is np.int32
    passed_on = transform_to_input_size(
        samples.minutiae_of(row)[:, :2], samples.height[row], samples.width[row]
    )
    assert np.array_equal(passed_on, with_numpy_numbers)
    # From NumPy 2 on. NumPy 1 scales the locations in their 16 bits with
    # both kinds of numbers (run with 1.26.4), so there is nothing to show.
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        assert with_python_numbers[0, 0] == 114.5 and with_numpy_numbers[0, 0] == 114.4375
        assert not np.array_equal(with_python_numbers, with_numpy_numbers)
        # So a loader that passed the sizes on as they are in the table would
        # have shown above, and in the minutia map as well: the moved
        # location is in another cell of it
        moved = _Minutiae(passed_on, loader.get_minutiae(identifier)[1]).get(identifier)
        assert not torch.equal(moved[0], loader.get(identifier)[0])


class _Minutiae(MinutiaMapLoader):
    """A loader of the given locations and angles, whatever the identifier is."""

    def __init__(self, locations: np.ndarray, angles: np.ndarray):
        self._minutiae = (locations, angles)

    def get_minutiae(self, identifier: Identifier) -> tuple:
        return self._minutiae


def assert_same_samples(found: list, expected: list) -> None:
    """Images, minutia maps and weights are equal, sample by sample and bit by bit."""
    assert len(found) == len(expected)
    for number, (sample, other) in enumerate(zip(found, expected)):
        for tensor, other_tensor in zip(sample[:2], other[:2]):
            assert tensor.dtype == other_tensor.dtype, number
            assert torch.equal(tensor, other_tensor), number
        assert sample[2] == other[2], number


@pytest.mark.parametrize("canvas_size", [0, 512, 250])
def test_the_pipeline_of_the_training_gives_the_same_tensors_from_a_table(
    corpus, train_500k, canvas_size
):
    """
    The image and the minutia map of every print as the training gets them:
    random pose and quality, the real binariser, the canvas. The random
    numbers are seeded directly before the image of a print is fetched, and
    its map is fetched directly after it, because the map takes the pose of
    the image that was loaded last.
    """
    ids = corpus.table.ids
    # A canvas of 512 cuts the highest print and pads the others; one of 250
    # is smaller than every print in one direction at least
    longer_sides = [max(resampled_size(*spec.size, spec.ppi)) for spec in PRINTS]
    assert sorted(longer_sides)[-2] < 512 < max(longer_sides) and min(longer_sides) > 250

    def fetch(images: Dataset, maps: Dataset, seed: int = 0) -> list:
        samples = []
        for number, identifier in enumerate(ids):
            random.seed(1000 * canvas_size + 10 * number + seed)
            image = images.get(identifier)
            minutia_map, weight = maps.get(identifier)
            samples.append((image, minutia_map, weight))
        return samples

    def directory_mode() -> tuple:
        images = train_500k.create_image_dataset(
            corpus.image_dir, ids, corpus.normalizer, canvas_size
        )
        maps = train_500k.create_minutia_dataset(
            corpus.minutiae_dir, ids, images.data_loader, canvas_size
        )
        return images, maps

    def table_mode() -> tuple:
        images = train_500k.create_image_dataset(
            corpus.image_dir, ids, None, canvas_size, files=SampleTableImageLoader(corpus.table)
        )
        minutiae = PoseTransformedMinutiaMapLoader(
            SampleTableMinutiaMapLoader(corpus.table),
            images.data_loader,
            roi=train_500k.canvas_roi(canvas_size),
        )
        return images, Dataset(minutiae, ids)

    expected = fetch(*directory_mode())
    # First the pipeline of directory mode against itself: if it did not
    # repeat with the same seeds, a difference to the table would say nothing
    assert_same_samples(fetch(*directory_mode()), expected)
    # Control: the seeds are what the tensors depend on, ...
    other_poses = fetch(*directory_mode(), seed=1)
    assert not any(torch.equal(sample[0], other[0]) for sample, other in zip(expected, other_poses))
    # ... every image is the model input of another print, with ridges and
    # background, ...
    images = [sample[0] for sample in expected]
    assert all(image.shape == (1, INPUT_SIZE, INPUT_SIZE) for image in images)
    assert all(0.01 < float(image.mean()) < 0.99 for image in images)
    assert not any(torch.equal(image, other) for image, other in zip(images, images[1:]))
    # ... and the minutia maps hold minutiae, except those of the two .mnt
    # files without any
    weights = {identifier: sample[2] for identifier, sample in zip(ids, expected)}
    assert weights[corpus.identifier(ZERO_BYTES)] == weights[corpus.identifier(HEADER_ZERO)] == 0.0
    assert sum(weights.values()) >= len(PRINTS) - 3
    for _, minutia_map, weight in expected:
        assert minutia_map.shape == (6, 128, 128) and bool(minutia_map.any()) == (weight == 1.0)

    assert_same_samples(fetch(*table_mode()), expected)
    assert_same_samples(fetch(*table_mode(), seed=1), other_poses)


# ---------------------------------------------------------------------------
# A table without files: rows, identifiers, subjects
# ---------------------------------------------------------------------------


def sample_row(subject: int, impression: int, minutiae="some", **changes) -> SampleRow:
    """
    A valid row. Its values follow from the two numbers, so that the rows of
    a table differ. minutiae: "some", "none" (an empty list) or "unknown" (null).
    """
    ppi = (500.0, 1000.0, 251.2, 385.0)[(subject + impression) % 4]
    source_width, source_height = 300 + 7 * subject, 400 + 3 * impression
    width, height = resampled_size(source_width, source_height, ppi)
    count = {"some": 2 + (subject + impression) % 3, "none": 0, "unknown": None}[minutiae]
    values = dict(
        subject=subject,
        impression=impression,
        image_path=f"sensor {subject % 3}/{subject + 1}_{impression + 1}.png",
        file_size=10_000 + 17 * subject + impression,
        ppi=ppi,
        ppi_origin="manifest",
        ridges="light" if subject % 2 else "dark",
        ridges_origin="argument",
        source_width=source_width,
        source_height=source_height,
        width=width,
        height=height,
        minutiae=(
            None
            if count is None
            else np.arange(3 * count, dtype=np.float32).reshape(count, 3) / 8 + subject
        ),
    )
    values.update(changes)
    return SampleRow(**values)


# Subjects with gaps and with different numbers of impressions, in no order;
# one row without minutiae and one whose minutiae are not known
UNEVEN_ROWS = [
    sample_row(5, 1),
    sample_row(5, 4, "unknown"),
    sample_row(0, 0),
    sample_row(0, 1, "none"),
    sample_row(0, 2),
    sample_row(17, 2),
    sample_row(3, 0),
    sample_row(8, 0),
    sample_row(8, 1),
]
# The rows in the order of the table: (subject, impression)
UNEVEN_IN_ORDER = [(0, 0), (0, 1), (0, 2), (3, 0), (5, 1), (5, 4), (8, 0), (8, 1), (17, 2)]


def uneven_table(image_dir="images") -> SampleTable:
    return SampleTable(samples_from_rows(UNEVEN_ROWS), str(image_dir))


def test_every_row_is_found_by_its_identifier():
    table = uneven_table()
    samples = table.samples
    assert table.num_rows == len(samples) == len(UNEVEN_ROWS)
    identifiers = table.identifiers
    assert [(i.subject, i.impression) for i in identifiers] == UNEVEN_IN_ORDER
    for row, identifier in enumerate(identifiers):
        assert type(identifier) is Identifier
        assert (identifier.subject, identifier.impression) == (
            samples.subject[row],
            samples.impression[row],
        )
        assert type(identifier.subject) is int and type(identifier.impression) is int
        assert table.row(identifier) == row
        # By its two numbers: another object with the same numbers finds the row too
        assert table.row(Identifier(identifier.subject, identifier.impression)) == row
        assert table.image_file(row) == join(table.image_dir, *samples.image_path(row).split("/"))
    assert table.describe_row(5) == "'sensor 2/6_5.png' (subject 5, impression 4)"


def test_the_identifiers_of_a_table_are_made_once_and_shared(capsys):
    table = uneven_table()
    capsys.readouterr()
    ids = table.ids
    # One IdentifierSet, made of exactly the objects of the list, in the order of the rows
    assert isinstance(ids, IdentifierSet) and len(ids) == table.num_rows
    assert ids.num_subjects == len(table.subjects) == 5
    assert all(a is b for a, b in zip(ids.identifiers, table.identifiers))
    assert table.ids is ids and table.identifiers is table.identifiers
    # Both loaders hand out this set, also the minutia loader for the rows without minutiae
    assert SampleTableImageLoader(table).ids is ids
    assert SampleTableMinutiaMapLoader(table).ids is ids
    assert capsys.readouterr().out.count("Created IdentifierSet") == 1


def test_the_identifiers_of_a_subject_share_its_number_as_one_object():
    """
    As the identifiers of the grid of directory mode do. A table adds one
    Identifier per row and no further object where the rows of a subject
    can share one.
    """
    # Numbers above those that Python keeps as one object anyway
    rows = [sample_row(subject, i) for subject in (1000, 70_000, 2_000_000) for i in range(3)]
    table = SampleTable(samples_from_rows(rows), "images")
    identifiers = table.identifiers
    assert [i.subject for i in identifiers] == [1000] * 3 + [70_000] * 3 + [2_000_000] * 3
    for first in (0, 3, 6):
        assert identifiers[first].subject is identifiers[first + 1].subject
        assert identifiers[first].subject is identifiers[first + 2].subject
    # Control: equal numbers are not one object by themselves
    assert Identifier(int("70000"), 0).subject is not Identifier(int("70000"), 1).subject


def test_an_identifier_that_is_not_in_the_table_raises_a_key_error_that_names_it():
    table = uneven_table()
    for subject, impression in [(5, 2), (4, 0), (17, 3), (2**40, 0), (-1, 0)]:
        missing = Identifier(subject, impression)
        for use in (
            table.row,
            SampleTableImageLoader(table).get,
            SampleTableMinutiaMapLoader(table).get,
            SampleTableMinutiaMapLoader(table).get_minutiae,
            SampleTableMinutiaMapLoader(table).get_frame_size,
        ):
            with pytest.raises(KeyError) as error:
                use(missing)
            message = error.value.args[0]
            assert message.startswith(f"Identifier({subject}, {impression}) is not in the sample")
            assert f"subject {subject} with impression {impression}" in message


def test_a_loader_is_made_of_a_table_and_not_of_its_rows():
    """
    The Samples of a table are not a SampleTable. A loader over them would
    be made without an error, and a Dataset over it would skip its check of
    the identifiers, so the mistake is refused where it is made.
    """
    table = uneven_table()
    for loader in (SampleTableImageLoader, SampleTableMinutiaMapLoader):
        with pytest.raises(TypeError) as error:
            loader(table.samples)
        assert str(error.value) == f"A {loader.__name__} is made of a SampleTable, got Samples"
        assert loader(table).table is table


def test_rows_whose_identifiers_compare_as_equal_are_refused_with_both_rows_named():
    """
    An Identifier is compared by the hash of its two numbers, and two pairs
    of numbers can have the same hash. A table with two such rows passes the
    rules of a table (the numbers differ), but no set of identifiers can
    hold both. The refusal names the rows and says what to do.
    """
    first, second = (391292100, 1755844984), (1473744498, 0)
    if hash(first) != hash(second):
        pytest.skip("These two pairs of numbers have different hashes with this Python")
    assert Identifier(*first) == Identifier(*second)  # control
    rows = [
        dataclasses.replace(
            sample_row(1, 0), subject=subject, impression=impression, image_path=name
        )
        for (subject, impression), name in ((first, "a.png"), (second, "b.png"))
    ]
    table = SampleTable(samples_from_rows(rows + [sample_row(7, 0)]), "images")
    # The rows themselves are found by their numbers
    assert [(i.subject, i.impression) for i in table.identifiers] == [(7, 0), first, second]
    assert [table.row(identifier) for identifier in table.identifiers] == [0, 1, 2]
    for ids_of in (
        lambda: table.ids,
        lambda: SampleTableImageLoader(table).ids,
        lambda: SampleTableMinutiaMapLoader(table).ids,
    ):
        with pytest.raises(SampleTableError) as error:
            ids_of()
        message = str(error.value)
        assert message.startswith(
            "1 pair(s) of rows of the sample table cannot be told apart by their numbers: "
            "'a.png' (subject 391292100, impression 1755844984) and "
            "'b.png' (subject 1473744498, impression 0). "
        )
        assert "these pairs of numbers have the same hash" in message
        assert "another subject or impression" in message and BUILDER in message
    # Without the two rows the same table has its identifiers
    assert len(SampleTable(samples_from_rows(rows[:1] + [sample_row(7, 0)]), "images").ids) == 2


def test_the_subjects_of_a_table_and_where_their_rows_are():
    table = uneven_table()
    assert table.subjects.dtype == np.int32 and table.subjects.tolist() == [0, 3, 5, 8, 17]
    assert table.subject_offsets.dtype == np.int64
    assert table.subject_offsets.tolist() == [0, 3, 4, 6, 8, 9]
    assert table.impressions_per_subject.dtype == np.int64
    assert table.impressions_per_subject.tolist() == [3, 1, 2, 2, 1]
    # Each of the three is made once: asking again does not go through the rows again
    for name in ("subjects", "subject_offsets", "impressions_per_subject"):
        assert getattr(table, name) is getattr(table, name), name
    # The rows of each subject follow from the offsets
    for subject, first, end in zip(
        table.subjects, table.subject_offsets[:-1], table.subject_offsets[1:]
    ):
        assert set(table.samples.subject[first:end].tolist()) == {subject}
        assert table.rows_of([subject]).tolist() == list(range(first, end))

    # In the order in which the subjects are given, each with all its rows
    assert table.rows_of([8, 0, 17]).tolist() == [6, 7, 0, 1, 2, 8]
    assert table.rows_of(np.array([3, 5], dtype=np.int32)).tolist() == [3, 4, 5]
    assert table.rows_of(iter([5])).dtype == np.int64
    assert table.rows_of([]).tolist() == [] and table.identifiers_of([]) == []
    # A subject that is given twice gives its rows twice
    assert table.rows_of([5, 0, 5]).tolist() == [4, 5, 0, 1, 2, 4, 5]
    assert table.identifiers_of([3, 3]) == [table.identifiers[3]] * 2
    wanted = table.identifiers_of([8, 0, 17])
    assert [(i.subject, i.impression) for i in wanted] == [
        UNEVEN_IN_ORDER[row] for row in (6, 7, 0, 1, 2, 8)
    ]
    # They are the objects of the table, not new ones: a training set and a
    # validation set made of them add no object per row
    assert all(identifier is table.identifiers[table.row(identifier)] for identifier in wanted)
    assert all(a is b for a, b in zip(table.identifiers_of(table.subjects), table.identifiers))
    training = IdentifierSet(table.identifiers_of([0, 5, 17]))
    assert len(training) == 6 and training.num_subjects == 3

    # A subject that the table does not have
    for use in (table.rows_of, table.identifiers_of):
        with pytest.raises(KeyError) as error:
            use([8, 1, 2, 4, 6, 7, 100, 9, 0])
        assert error.value.args[0] == (
            "The sample table has no row of the subject(s) 1, 2, 4, 6, 7 and 2 more; it holds "
            "5 subjects"
        )
        with pytest.raises(TypeError, match="whole numbers"):
            use([8.0])
        with pytest.raises(TypeError, match=r"one dimension, got an array of .* \(1, 2\)"):
            use(np.array([[8, 0]]))


def test_subjects_are_found_in_whatever_type_of_whole_numbers_names_them():
    table = uneven_table()
    rows = [6, 7, 0, 1, 2, 8]
    given = [[8, 0, 17], (8, 0, 17), iter([8, 0, 17])]
    given += [
        np.array([8, 0, 17], dtype=dtype)
        for dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    ]
    for subjects in given:
        assert table.rows_of(subjects).tolist() == rows
    for subjects in given[:2] + given[3:]:  # the iterator is used up
        assert table.identifiers_of(subjects) == [table.identifiers[row] for row in rows]

    # A number that 32 bits cannot hold is not in the table, also where its
    # lower 32 bits are those of the subject 5
    assert table.rows_of([5]).tolist() == [4, 5]
    for number in (2**32 + 5, 5 - 2**32, 2**63 - 2**32 + 5, 2**31, -1):
        for use in (table.rows_of, table.identifiers_of):
            with pytest.raises(KeyError) as error:
                use([8, number, 0])
            assert error.value.args[0] == (
                f"The sample table has no row of the subject(s) {number}; it holds 5 subjects"
            )
    # ... and it is named with the numbers that fit but are no subjects
    unsigned = np.array([4, 2**32 + 5, 5, 6, 2**64 - 2**32 + 5], dtype=np.uint64)
    with pytest.raises(KeyError) as error:
        table.rows_of(unsigned)
    assert error.value.args[0] == (
        "The sample table has no row of the subject(s) 4, 4294967301, 6, 18446744069414584325; "
        "it holds 5 subjects"
    )


def test_a_call_for_one_subject_does_not_copy_the_subjects_of_the_table():
    """
    A split or the validation pairs can ask for the rows of one subject after
    the other, so one call must not cost a pass over all subjects, in
    whatever type the number is given. numpy searches a sorted array for
    numbers of a wider type in a copy of the whole array, and such a copy
    shows in the memory that a call takes.
    """
    count = 5_000
    rows = [sample_row(subject, 0, "unknown") for subject in range(count)]
    table = SampleTable(samples_from_rows(rows), "images")
    table.identifiers  # made at the first access, which is not what is measured
    assert table.rows_of([count - 1]).tolist() == [count - 1]

    def memory_taken(call) -> int:
        """The most bytes that are allocated at a time while the call runs, above those before."""
        # Tracing can be on already (python -X tracemalloc): it is left on then
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            call()
            return tracemalloc.get_traced_memory()[1] - before
        finally:
            if not tracing:
                tracemalloc.stop()

    # The subjects of the table take 4 bytes each
    assert table.subjects.nbytes == 4 * count
    for subject in (
        [count - 2],
        (count - 2,),
        np.array([count - 2], dtype=np.int64),
        np.array([count - 2], dtype=np.uint32),
        np.array([count - 2], dtype=np.int32),
    ):
        for use in (table.rows_of, table.identifiers_of):
            taken = memory_taken(lambda: use(subject))
            assert taken < table.subjects.nbytes, (use.__name__, subject, taken)
    # Neither does asking for the impressions per subject, which are kept
    assert memory_taken(lambda: table.impressions_per_subject) < table.subjects.nbytes
    # Control: a copy of the subjects in 64 bits would show
    assert memory_taken(lambda: table.subjects.astype(np.int64)) >= 8 * count


def test_a_table_says_which_rows_have_no_known_minutiae():
    table = uneven_table()
    # Null only: the row with an empty list is known to have no minutiae
    assert table.rows_without_minutiae().tolist() == [5]
    assert UNEVEN_IN_ORDER[5] == (5, 4) and len(table.samples.minutiae_of(1)) == 0
    assert table.samples.minutiae_known.tolist() == [row != 5 for row in range(9)]
    loader = SampleTableMinutiaMapLoader(table)
    # Both kinds are in the ids of the minutia loader and train with weight 0
    for identifier, with_minutiae in zip(table.identifiers, [1, 0, 1, 1, 1, 0, 1, 1, 1]):
        locations, angles = loader.get_minutiae(identifier)
        assert (len(locations) > 0) == (len(angles) > 0) == bool(with_minutiae)
        assert locations.shape[1:] == (2,) and angles.shape == (len(locations),)
        minutia_map, weight = loader.get(identifier)
        assert weight == float(with_minutiae) and minutia_map.shape == (6, 128, 128)
        assert bool(minutia_map.any()) == bool(with_minutiae)
        row = table.row(identifier)
        assert loader.get_frame_size(identifier) == (
            table.samples.width[row],
            table.samples.height[row],
        )


def test_a_table_hands_on_what_its_rows_say_about_the_table(tmp_path, monkeypatch):
    samples = samples_from_rows(UNEVEN_ROWS, build={"id": "b", "failed": 2})
    samples = dataclasses.replace(samples, unknown_columns=("finger", "person"))
    monkeypatch.chdir(tmp_path)
    table = SampleTable(samples, "images/../images")
    assert table.samples is samples
    # The directory is absolute from the start: a copy in a process with
    # another working directory names the same files
    assert table.image_dir == str(tmp_path / "images")
    assert table.image_file(0) == str(tmp_path / "images" / "sensor 0" / "1_1.png")
    assert table.target_ppi == 500.0
    assert table.build == {"id": "b", "failed": 2}
    assert table.unknown_columns == ("finger", "person")
    assert table.checksum == samples_checksum(samples)
    assert "9 rows" in repr(table) and str(tmp_path / "images") in repr(table)
    with pytest.raises(TypeError, match="Samples"):
        SampleTable(UNEVEN_ROWS, "images")


def test_the_checksum_of_a_table_is_computed_once_and_only_when_it_is_asked_for(monkeypatch):
    calls = []

    def checksum(samples):
        calls.append(samples)
        return samples_checksum(samples)

    monkeypatch.setattr(sample_table_loader, "samples_checksum", checksum)
    table = uneven_table()
    SampleTableMinutiaMapLoader(table).get(table.identifiers[0])
    assert calls == []
    assert table.checksum == table.checksum == samples_checksum(table.samples)
    assert calls == [table.samples]


def test_rows_that_break_a_rule_of_a_table_are_refused_when_a_table_is_made():
    twice = samples_from_rows(UNEVEN_ROWS + [sample_row(8, 1, image_path="other/9_2.png")])
    with pytest.raises(SampleTableError, match="The same subject and impression in 2 of 10 rows"):
        SampleTable(twice, "images")
    outside = samples_from_rows([sample_row(0, 0, image_path="../elsewhere/1_1.png")])
    with pytest.raises(SampleTableError, match=r"it has the component '\.\.'"):
        SampleTable(outside, "images")
    # Rows that were checked before are taken as they are
    assert SampleTable(outside, "images", validate=False).num_rows == 1


def test_a_table_without_rows_is_a_table(train_500k):
    table = SampleTable(samples_from_rows([]), "images")
    assert table.num_rows == 0 and table.identifiers == [] and len(table.ids) == 0
    assert table.subjects.tolist() == [] and table.subject_offsets.tolist() == [0]
    assert table.impressions_per_subject.tolist() == []
    assert table.rows_of([]).tolist() == [] and table.rows_without_minutiae().tolist() == []
    with pytest.raises(KeyError, match="it holds 0 subjects"):
        table.rows_of([0])
    with pytest.raises(KeyError):
        table.row(Identifier(0, 0))

    # Neither the table nor its image loader has a length, so they are not
    # false without rows: the scripts take the loader as `files or ...`
    loader = SampleTableImageLoader(table)
    for thing in (table, loader):
        assert bool(thing) and not hasattr(thing, "__len__") and not hasattr(thing, "__bool__")
    for images in (
        train_500k.create_validation_image_loader("no such directory", files=loader),
        train_500k.create_image_dataset("no such directory", table.ids, files=loader).data_loader,
    ):
        assert images.ids is table.ids


def test_reading_a_table_that_cannot_be_used_is_refused_with_its_directory(tmp_path):
    pytest.importorskip("pyarrow")
    (tmp_path / "table").mkdir()
    with pytest.raises(SampleTableError, match="holds no part of a sample table") as error:
        SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    assert str(tmp_path / "table") in str(error.value)


def test_the_rows_of_a_table_that_is_read_are_checked_once(corpus, monkeypatch):
    """By read_samples. Rows that come from elsewhere are checked when the table is made."""
    checked = []
    monkeypatch.setattr(sample_table_loader, "validate_samples", checked.append)
    table = SampleTable.read(corpus.table_dir, corpus.image_dir)
    assert checked == [] and same_arrays(table.samples, corpus.table.samples)
    SampleTable(table.samples, corpus.image_dir)
    assert checked == [table.samples]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def test_datasets_take_both_loaders_with_any_identifiers_of_the_table():
    table = uneven_table()
    images, minutiae = SampleTableImageLoader(table), SampleTableMinutiaMapLoader(table)
    for loader in (images, minutiae):
        assert isinstance(loader, DataLoader)
        assert len(Dataset(loader, table.ids)) == 9
        # Also with identifiers that were made elsewhere, and with a part of them
        elsewhere = IdentifierSet([Identifier(8, 1), Identifier(0, 0), Identifier(5, 4)])
        assert Dataset(loader, elsewhere).num_subjects == 3
        with pytest.raises(ValueError, match="subset of the identifiers in the data loader"):
            Dataset(loader, IdentifierSet([Identifier(8, 1), Identifier(8, 2)]))

    # The rows without minutiae are in the dataset of the minutia maps: they
    # count with the weight 0
    maps = Dataset(minutiae, table.ids)
    weights = [maps[index][1] for index in range(len(maps))]
    assert weights == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert maps.get(Identifier(5, 4))[0].shape == maps.get(Identifier(5, 1))[0].shape


def test_the_images_of_a_table_are_a_dataset_over_the_ids_of_their_loader(corpus, train_500k):
    """As a model is used after the training: without augmentation, over all rows."""
    image_loader = train_500k.create_validation_image_loader(
        "not used", canvas_size=512, files=SampleTableImageLoader(corpus.table)
    )
    assert image_loader.ids is corpus.table.ids
    dataset = Dataset(image_loader, image_loader.ids)
    assert len(dataset) == len(PRINTS) and dataset.num_subjects == 4
    directory = train_500k.create_validation_image_loader(corpus.image_dir, corpus.normalizer, 512)
    for index in (0, len(PRINTS) - 1):
        image = dataset[index]
        assert image.shape == (1, INPUT_SIZE, INPUT_SIZE)
        assert torch.equal(image, directory.get(image_loader.ids[index]))


# ---------------------------------------------------------------------------
# Files that are not those of the table
# ---------------------------------------------------------------------------


def small_table(tmp_path, count: int = 3) -> SampleTable:
    """Small prints of 96x120 pixels at 500 ppi in tmp_path/images, and their table."""
    image_dir = tmp_path / "images"
    for index in range(count):
        image = ridge_image(120, 96, slant=0.3 + 0.2 * index)
        save(image_dir / "sensor" / f"{index + 1}_1.png", image, ppi=500)
    rows = rows_as_the_builder_fills_them(
        str(image_dir), str(tmp_path / "minutiae"), SensorNormalizer()
    )
    return SampleTable(samples_from_rows(rows), str(image_dir))


def refusal(loader: SampleTableImageLoader, identifier: Identifier) -> str:
    """The message with which the loader refuses the file of an identifier."""
    with pytest.raises(SampleTableError) as error:
        loader.get(identifier)
    return str(error.value)


def test_a_file_of_another_size_in_bytes_is_refused_before_it_is_loaded(tmp_path, monkeypatch):
    table = small_table(tmp_path)
    loader = SampleTableImageLoader(table)
    path = table.image_file(1)
    assert path == str(tmp_path / "images" / "sensor" / "2_1.png")
    size = os.path.getsize(path)
    assert size == table.samples.file_size[1]
    # The same image again, with a text in the file: the same pixels, more bytes
    pixels = np.array(Image.open(path))
    text = PngInfo()
    text.add_text("Comment", "scanned again")
    Image.fromarray(pixels).save(path, dpi=(500, 500), pnginfo=text)
    changed = os.path.getsize(path)
    assert changed > size and np.array_equal(np.array(Image.open(path)), pixels)

    loaded = []
    load = SensorNormalizer.load
    monkeypatch.setattr(
        SensorNormalizer, "load", lambda self, path: loaded.append(path) or load(self, path)
    )
    message = refusal(loader, table.identifiers[1])
    assert loaded == []
    assert message.startswith(f"{path} is not the file that its row of the sample table describes")
    assert f"the table holds {size:,} bytes as the size of the file" in message
    assert f"the file has {changed:,} bytes" in message
    assert "'sensor/2_1.png' (subject 1, impression 0)" in message
    assert f"for the image directory {tmp_path / 'images'}." in message
    assert f"build the table again from the files as they are now ({BUILDER})" in message
    # The other files are served
    for row in (0, 2):
        assert loader.get(table.identifiers[row]).shape == (1, 120, 96)
    assert loaded == [table.image_file(0), table.image_file(2)]


def test_a_file_that_lost_bytes_is_refused_before_it_is_loaded(tmp_path, monkeypatch):
    """A copy that was cut off: fewer bytes than the row holds, here one byte less."""
    table = small_table(tmp_path)
    loader = SampleTableImageLoader(table)
    path = table.image_file(0)
    size = os.path.getsize(path)
    with open(path, "rb+") as file:
        file.truncate(size - 1)
    loaded = []
    load = SensorNormalizer.load
    monkeypatch.setattr(
        SensorNormalizer, "load", lambda self, path: loaded.append(path) or load(self, path)
    )
    message = refusal(loader, table.identifiers[0])
    assert loaded == []
    assert message.startswith(f"{path} is not the file that its row of the sample table describes")
    assert f"the table holds {size:,} bytes as the size of the file" in message
    assert f"the file has {size - 1:,} bytes" in message


def test_a_file_of_other_pixels_is_refused_also_with_the_same_number_of_bytes(tmp_path):
    table = small_table(tmp_path)
    loader = SampleTableImageLoader(table)
    path = table.image_file(0)
    size = os.path.getsize(path)
    # Another image, wider than high, as a file of exactly as many bytes: an
    # image without ridges is a smaller file, and a text fills it up. A text
    # takes 20 bytes besides its characters.
    replacement = np.full((96, 120), 200, dtype=np.uint8)
    Image.fromarray(replacement).save(path, dpi=(500, 500))
    text = PngInfo()
    text.add_text("Comment", "x" * (size - os.path.getsize(path) - 20))
    Image.fromarray(replacement).save(path, dpi=(500, 500), pnginfo=text)
    assert os.path.getsize(path) == size

    message = refusal(loader, table.identifiers[0])
    assert message.startswith(f"{path} is not the file that its row of the sample table describes")
    assert "the table holds 96x120 as its size in pixels (width x height)" in message
    assert "the file has 120x96." in message
    assert "'sensor/1_1.png' (subject 0, impression 0)" in message
    assert f"build the table again from the files as they are now ({BUILDER})" in message
    # Control: the file is an image, of the size the message says
    image, info = SensorNormalizer().load(path)
    assert (info.source_width, info.source_height) == (120, 96)


@pytest.mark.parametrize(
    "side, held",
    [
        ("source_width", "97x120"),
        ("source_height", "96x121"),
        ("width", "97x120"),
        ("height", "96x121"),
    ],
)
def test_a_row_that_differs_from_its_file_in_one_side_only_is_refused(tmp_path, side, held):
    """
    Each of the four sizes of a row is compared with the file, not one of a
    pair: the file of the test above differs from its row in width and in
    height.
    """
    table = small_table(tmp_path)
    samples = table.samples
    longer = dataclasses.replace(samples, **{side: getattr(samples, side) + 1})
    # Such rows are no table (the prepared size follows from the other
    # three), so they are taken without the check
    loader = SampleTableImageLoader(SampleTable(longer, table.image_dir, validate=False))
    message = refusal(loader, table.identifiers[0])
    what = "its size in pixels (width x height)"
    if side in ("width", "height"):
        what = "the size in pixels (width x height) of the image prepared at 500 ppi"
    assert f"the table holds {held} as {what}, the file has 96x120." in message, message
    assert f"build the table again from the files as they are now ({BUILDER})" in message
    # Control: the files are served under the rows as they were made
    assert SampleTableImageLoader(table).get(table.identifiers[0]).shape == (1, 120, 96)


def test_images_are_prepared_at_the_target_resolution_of_the_table(tmp_path, train_500k):
    """
    A table that is read from its files has the target resolution of this
    code, because another one is refused. The loader takes the resolution
    from the table all the same, and not from the code.
    """
    image_dir = tmp_path / "images"
    save(image_dir / "1_1.png", ridge_image(120, 96), ppi=500)
    normalizer = SensorNormalizer(target_ppi=250)
    rows = rows_as_the_builder_fills_them(str(image_dir), str(tmp_path / "minutiae"), normalizer)
    assert (rows[0].width, rows[0].height) == (48, 60)
    table = SampleTable(samples_from_rows(rows, target_ppi=250.0), str(image_dir))
    assert table.target_ppi == 250.0
    image = SampleTableImageLoader(table).get(Identifier(0, 0))
    assert image.shape == (1, 60, 48)
    directory = train_500k.CustomFingerprintLoader(str(image_dir), normalizer)
    assert torch.equal(image, directory.get(Identifier(0, 0)))


def test_a_missing_file_is_named_with_the_image_directory(tmp_path):
    table = small_table(tmp_path)
    loader = SampleTableImageLoader(table)
    path = table.image_file(2)
    os.remove(path)
    with pytest.raises(FileNotFoundError) as error:
        loader.get(table.identifiers[2])
    message = str(error.value)
    assert message.startswith(f"The image file {path} is missing.")
    assert "'sensor/3_1.png' (subject 2, impression 0)" in message
    assert f"{tmp_path / 'images'} was given as the directory of its images" in message
    assert f"build the table again from the files as they are now ({BUILDER})" in message
    assert loader.get(table.identifiers[0]).shape == (1, 120, 96)

    # Another directory than the one the table was built from: every file is
    # missing, also where a file is in the place of a directory
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "sensor").write_text("no directory")
    elsewhere = SampleTableImageLoader(SampleTable(table.samples, str(tmp_path / "elsewhere")))
    for identifier in table.identifiers:
        with pytest.raises(FileNotFoundError, match="elsewhere was given as the directory"):
            elsewhere.get(identifier)


def test_a_file_that_is_no_image_fails_as_in_directory_mode(tmp_path, train_500k):
    table = small_table(tmp_path)
    path = table.image_file(0)
    size = os.path.getsize(path)
    with open(path, "wb") as file:
        file.write(b"\0" * size)  # as many bytes as the image had
    directory = train_500k.CustomFingerprintLoader(str(tmp_path / "images"), SensorNormalizer())
    with pytest.raises(ImageReadError) as expected:
        directory.get(table.identifiers[0])
    with pytest.raises(ImageReadError) as error:
        SampleTableImageLoader(table).get(table.identifiers[0])
    # The file is named in front of the message of the preparation
    assert str(error.value) == str(expected.value)
    assert str(error.value) == f"{path}: Could not read the file as an image: {path}"


# ---------------------------------------------------------------------------
# The table is not changed, and copies and worker processes serve the same
# ---------------------------------------------------------------------------


def test_the_table_is_not_changed_by_what_the_loaders_hand_out(corpus):
    table, samples = corpus.table, corpus.table.samples
    before = samples_checksum(samples)
    images, minutiae = SampleTableImageLoader(table), SampleTableMinutiaMapLoader(table)

    first = {}
    for _ in range(3):
        for identifier in table.ids:
            image = images.get(identifier)
            locations, angles = minutiae.get_minutiae(identifier)
            minutia_map, _ = minutiae.get(identifier)
            first.setdefault(identifier, (image.clone(), locations.copy(), angles.copy()))
            # What the loaders hand out is not memory of the table. It can be
            # changed, like the arrays of the loaders of directory mode, ...
            for array in (locations, angles):
                assert not np.shares_memory(array, samples.minutiae)
                assert array.flags.writeable and array.flags.c_contiguous
                array[...] = -1
            image.zero_()
            minutia_map.zero_()
            minutiae.get_frame_size(identifier)
    # ... and the next call hands out what the first one did
    for identifier in table.ids:
        image, locations, angles = first[identifier]
        assert torch.equal(images.get(identifier), image)
        assert np.array_equal(minutiae.get_minutiae(identifier)[0], locations)
        assert np.array_equal(minutiae.get_minutiae(identifier)[1], angles)
    assert samples_checksum(samples) == before == table.checksum

    # What is memory of the table cannot be written to: its arrays, the
    # minutiae of a row, the subjects, their offsets and their numbers of rows
    row = corpus.row(PRINTS[0])
    of_the_table = [getattr(samples, name) for name in ARRAYS]
    of_the_table += [samples.minutiae_of(row), table.subjects, table.subject_offsets]
    of_the_table.append(table.impressions_per_subject)
    assert len(samples.minutiae_of(row)) == PRINTS[0].minutiae
    for array in of_the_table:
        assert array.size > 0
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0
    assert samples_checksum(samples) == before


def test_copies_of_the_loaders_serve_the_same_samples(corpus):
    """A worker process that is not forked gets the loaders as a copy made by pickle."""
    table = corpus.table
    images, minutiae = SampleTableImageLoader(table), SampleTableMinutiaMapLoader(table)
    images.get(table.ids[0])  # a loader that was used, as in the training
    images_copy, minutiae_copy = pickle.loads(pickle.dumps((images, minutiae)))
    table_copy = images_copy.table

    # One copy of the rows for both loaders, with the directory and every array
    assert table_copy is not table and minutiae_copy.table is table_copy
    assert table_copy.image_dir == table.image_dir
    assert table_copy.samples is not table.samples and same_arrays(
        table_copy.samples, table.samples
    )
    assert table_copy.samples.build == table.samples.build
    # The arrays of the copy are read-only like those of the original
    for array in [getattr(table_copy.samples, name) for name in ARRAYS] + [
        table_copy.subjects,
        table_copy.subject_offsets,
        table_copy.impressions_per_subject,
    ]:
        assert not array.flags.writeable and not np.shares_memory(array, table.samples.minutiae)
    assert table_copy.subjects.tolist() == table.subjects.tolist()
    assert table_copy.subject_offsets.tolist() == table.subject_offsets.tolist()
    assert table_copy.impressions_per_subject.tolist() == table.impressions_per_subject.tolist()

    # The same lookups and the same tensors, for the identifiers of the original
    for identifier in table.ids:
        assert table_copy.row(identifier) == table.row(identifier)
        assert table_copy.image_file(table_copy.row(identifier)) == table.image_file(
            table.row(identifier)
        )
        assert torch.equal(images_copy.get(identifier), images.get(identifier))
        for array, expected in zip(
            minutiae_copy.get_minutiae(identifier), minutiae.get_minutiae(identifier)
        ):
            assert array.dtype == expected.dtype and np.array_equal(array, expected)
        assert minutiae_copy.get_frame_size(identifier) == minutiae.get_frame_size(identifier)
        assert torch.equal(minutiae_copy.get(identifier)[0], minutiae.get(identifier)[0])

    # The identifiers are not copied: the copy makes its own when it is asked
    assert table_copy.__getstate__().keys() == {"samples", "image_dir"}
    assert table_copy.ids is images_copy.ids is minutiae_copy.ids
    assert table_copy.identifiers == table.identifiers
    assert not any(a is b for a, b in zip(table_copy.identifiers, table.identifiers))
    assert table_copy.checksum == table.checksum

    # The copy module makes copies in the same way
    shallow, deep = copy.copy(table), copy.deepcopy(table)
    assert shallow.samples is table.samples and deep.samples is not table.samples
    for other in (shallow, deep):
        assert other.image_dir == table.image_dir and other.checksum == table.checksum
        assert not other.samples.minutiae.flags.writeable
        assert other.identifiers == table.identifiers
        assert other.identifiers[0] is not table.identifiers[0]


def identity(image: torch.Tensor) -> torch.Tensor:
    return image


class _Process(DataLoader):
    """The number of the process that is asked, whatever the identifier is."""

    def get(self, identifier: Identifier) -> int:
        return os.getpid()


def test_the_loaders_serve_the_worker_processes_of_a_training(corpus, train_500k, monkeypatch):
    """
    Images, minutia maps and labels as the training zips them, fetched by
    forked worker processes of a torch DataLoader: every sample is the one
    that the loaders give in this process. Without augmentation, which is
    random, and without the binariser, which this comparison does not need.
    """
    monkeypatch.setattr(train_500k, "LazilyAllocatedBinarizer", lambda _: identity)
    table = corpus.table
    ids = table.ids
    before = samples_checksum(table.samples)
    images = train_500k.create_validation_image_loader(
        "not used", canvas_size=512, files=SampleTableImageLoader(table)
    )
    maps = PoseTransformedMinutiaMapLoader(
        SampleTableMinutiaMapLoader(table), images, roi=train_500k.canvas_roi(512)
    )
    samples = Dataset.zip(
        Dataset(images, ids),
        Dataset(maps, ids),
        Dataset(LabelIndex(ids), ids),
        Dataset(_Process(), ids),
    )
    expected = [samples[index] for index in range(len(samples))]
    assert {weight for _, (_, weight), _, _ in expected} == {0.0, 1.0}

    batches = torch.utils.data.DataLoader(
        samples, batch_size=3, shuffle=False, num_workers=2, multiprocessing_context="fork"
    )
    for _ in range(2):  # new worker processes each time, as in every epoch
        fetched, workers = 0, set()
        for batch_images, (batch_maps, weights), labels, processes in batches:
            workers.update(processes.tolist())
            for image, minutia_map, weight, label in zip(batch_images, batch_maps, weights, labels):
                image_here, (map_here, weight_here), label_here, _ = expected[fetched]
                assert torch.equal(image, image_here), fetched
                assert torch.equal(minutia_map, map_here), fetched
                assert float(weight) == weight_here and int(label) == label_here
                fetched += 1
        assert fetched == len(PRINTS)
        # Control: the samples were made by two other processes
        assert len(workers) == 2 and os.getpid() not in workers
    assert samples_checksum(table.samples) == before
