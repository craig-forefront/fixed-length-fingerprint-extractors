"""
scripts/train_500k.py with a sample table (--samples): the training set is
what a table says, and not what the names of the files in two directories
say.

The central test runs main() on the same files without a table (directory
mode) and with one (table mode) and compares what the model is given to
train on: the identifiers, the held-out subjects, the validation pairs, the
number of classes, and every image and minutia map bit by bit. Its files are
chosen by the fault that each of them shows, with a control assertion that
the fault would be seen. They are what both modes can read: .png files named
<subject>_<impression>, a complete grid, .mnt files whose header is the size
of the prepared image.

The other tests are about what only a table can describe (other names,
uneven numbers of impressions, gaps, missing minutiae), about every start
that is refused, about continuing a training, and about using the model.

The tables are built by scripts/build_sample_table.py through its main(), so
that the way from the builder to the training is what is tested; where a
test needs rows that the builder does not make, they are written with
samples_from_rows and write_part. The .mnt files are written as text, and
both scripts are imported from their files: no test module is imported.
"""

import dataclasses
import importlib.util
import json
import os
import random
import shutil
import sys
from os.path import abspath, dirname, exists, join
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("pyarrow")

from flx.data.dataset import Dataset, Identifier, IdentifierSet  # noqa: E402
from flx.data.fingerflow_minutia_loader import (  # noqa: E402
    FingerFlowMinutiaMapLoader,
    read_mnt_file,
)
from flx.data.image_helpers import transform_to_input_size  # noqa: E402
from flx.data.sample_table_loader import (  # noqa: E402
    SampleTable,
    SampleTableImageLoader,
    SampleTableMinutiaMapLoader,
)
from flx.models import torch_helpers  # noqa: E402
from flx.models.model_training import TrainingLog, TrainingLogEntry  # noqa: E402
from flx.preparation.sample_table import (  # noqa: E402
    PPI_ORIGINS,
    RIDGES_ORIGINS,
    SampleRow,
    Samples,
    read_samples,
    samples_from_rows,
    write_part,
)
from flx.preparation.sensor_normalization import (  # noqa: E402
    SensorNormalizer,
    resampled_size,
)
from flx.setup.config import INPUT_SIZE  # noqa: E402

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")
VARIANTS = ["LocTexMinu", "LocTex", "TexMinu", "Tex"]

# The largest kernel of the binariser of the training: an image needs at
# least this many pixels per side
LARGEST_BINARISER_KERNEL = 41


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, join(SCRIPTS_DIR, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_500k():
    return load_script("train_500k")


@pytest.fixture(scope="module")
def builder():
    """scripts/build_sample_table.py as a module. It adds a directory to sys.path when loaded."""
    path = list(sys.path)
    try:
        return load_script("build_sample_table")
    finally:
        sys.path[:] = path


@pytest.fixture(autouse=True)
def default_batch_size():
    yield
    torch_helpers.set_training_batch_size(torch_helpers.DEFAULT_TRAINING_BATCH_SIZE)


class _FakeExtractor:
    """
    Its fit writes model.pyt and one log entry per epoch, like train_model,
    and keeps what it was given. `built` holds the variant and the arguments
    of every model that a run built (see run).
    """

    def __init__(self):
        self.fits = []
        self.built = []
        self.model = "the model"

    def fit(self, out_dir, num_epochs, **kwargs):
        self.fits.append(dict(kwargs, num_epochs=num_epochs))
        log = TrainingLog(join(out_dir, "log.json"))
        for epoch in range(len(log) + 1, num_epochs + 1):
            log.add_entry(TrainingLogEntry(epoch, 5.0 - epoch, {}, 0.1, None, 1e-4))
            for name in ("model.pyt", "best_model.pyt"):
                with open(join(out_dir, name), "w") as file:
                    file.write(str(epoch))


def run(train_500k, monkeypatch, *arguments, fake=None) -> _FakeExtractor:
    """Runs main() with the arguments and a fake model. Returns the fake."""
    fake = fake or _FakeExtractor()
    for variant in VARIANTS:

        def constructor(*args, _variant=variant, **kwargs):
            assert not args
            fake.built.append((_variant, kwargs))
            return fake

        monkeypatch.setattr(train_500k, "get_DeepPrint_" + variant, constructor)
    monkeypatch.setattr(sys, "argv", ["train_500k.py"] + [str(a) for a in arguments])
    train_500k.main()
    return fake


def run_table(train_500k, monkeypatch, root, *options, table="table", output="model", fake=None):
    """Table mode: main() on the images in root/images and the table root/<table>."""
    return run(
        train_500k,
        monkeypatch,
        *("--dataset_dir", root / "images", "--samples", root / table),
        *("--output_dir", root / output, "--num_epochs", 2),
        *options,
        fake=fake,
    )


def run_directory(train_500k, monkeypatch, root, *options, output="model", fake=None):
    """
    Directory mode: main() on the images in root/images. The options say
    which grid of subjects and impressions the directory holds.
    """
    return run(
        train_500k,
        monkeypatch,
        *("--dataset_dir", root / "images", "--output_dir", root / output, "--num_epochs", 2),
        *options,
        fake=fake,
    )


def stops(runner, train_500k, monkeypatch, capsys, root, *options, **kwargs) -> str:
    """
    Runs main() through the runner of a mode. It must stop with the exit
    status 2. Returns what it wrote to standard error, on one line.
    """
    capsys.readouterr()
    fake = kwargs.setdefault("fake", _FakeExtractor())
    with pytest.raises(SystemExit) as stop:
        runner(train_500k, monkeypatch, root, *options, **kwargs)
    assert stop.value.code == 2
    # Nothing was trained, and no model was built
    assert fake.fits == [] and fake.built == []
    return " ".join(capsys.readouterr().err.split())


def build(builder, root, *options, table="table", images="images") -> int:
    """
    Builds the table root/<table> of the images in root/<images> with
    scripts/build_sample_table.py. Returns its exit status: 0, or 1 if a file
    has a line in the failures log (the table is written then as well).
    """
    arguments = ["--dataset_dir", root / images, "--samples", root / table, *options]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["build_sample_table.py"] + [str(a) for a in arguments])
        return builder.main()


def stored_settings(root, output="model") -> dict:
    with open(root / output / "training_settings.json") as file:
        return json.load(file)


def numbers(ids) -> list:
    """(subject, impression) of the identifiers of a set or a dataset, in its order."""
    return [(i.subject, i.impression) for i in ids]


def pairs(benchmark) -> list:
    """The pairs of a validation benchmark as numbers, in its order."""
    return [
        ((c.sample1.subject, c.sample1.impression), (c.sample2.subject, c.sample2.impression))
        for c in benchmark._comparisons
    ]


def printed(capsys) -> list:
    """The lines that were printed, each with single spaces."""
    return [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]


# ---------------------------------------------------------------------------
# Files: prints and .mnt files
# ---------------------------------------------------------------------------


def ridge_image(height: int, width: int, period: float = 9.5, slant: float = 0.3) -> np.ndarray:
    """Dark ridges on a white background: a print in the middle of a white frame."""
    y, x = np.mgrid[0:height, 0:width]
    ridges = 127.5 + 127.5 * np.cos(2 * np.pi * (x + slant * y) / period)
    image = np.full((height, width), 255.0)
    inside = (y > height // 6) & (y < 5 * height // 6) & (x > width // 6) & (x < 5 * width // 6)
    image[inside] = ridges[inside]
    return image.astype(np.uint8)


def save(path, image: np.ndarray, ppi=500) -> str:
    """Saves the image, with the resolution in the file metadata unless ppi is None."""
    os.makedirs(dirname(str(path)), exist_ok=True)
    if ppi is None:
        Image.fromarray(image).save(str(path))
    else:
        Image.fromarray(image).save(str(path), dpi=(ppi, ppi))
    return str(path)


def write_mnt_file(path, minutiae, size: tuple, comment: Optional[str] = None) -> None:
    """
    A .mnt file as scripts/extract_minutiae.py writes it: the number of
    minutiae, the size of the prepared image, a line per minutia (x, y,
    angle, class, direction confidence), and how the image was prepared.
    """
    lines = [str(len(minutiae)), f"{size[0]} {size[1]}"]
    lines += [f"{x:.4f} {y:.4f} {angle:.6f} 0 0.9000" for x, y, angle in minutiae]
    if comment is not None:
        lines.append(f"# {comment}")
    os.makedirs(dirname(str(path)), exist_ok=True)
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def write_small_prints(image_dir, names, ppi=500, size=(48, 64)) -> None:
    """
    Small prints (width, height) under the given names below the image
    directory, each another image. They are too small for the binariser:
    for the tests that do not fetch a sample from the datasets.
    """
    for number, name in enumerate(names):
        save(image_dir / name, ridge_image(size[1], size[0], slant=0.05 * (number + 1)), ppi)


def write_small_minutiae(minutiae_dir, names, size=(48, 64)) -> None:
    """The .mnt file of each of the small prints, where the extraction writes it."""
    for name in names:
        path = minutiae_dir / (os.path.splitext(name)[0] + ".mnt")
        write_mnt_file(path, [(10.5, 20.25, 0.5), (30.0, 40.0, 2.0)], size)


def grid(subjects: int, impressions: int, directory: str = "") -> list:
    """The file names of a complete grid as directory mode reads it, numbered from 1."""
    return [
        join(directory, f"{subject}_{impression}.png")
        for subject in range(1, subjects + 1)
        for impression in range(1, impressions + 1)
    ]


# ---------------------------------------------------------------------------
# The same training from a table as from the directories
# ---------------------------------------------------------------------------

# What the .mnt file of a print is, where it is not a number of minutiae
HEADER_ZERO = 0  # the header 0, as the extraction writes it for an image without minutiae
ZERO_BYTES = "a file of 0 bytes"
NO_MNT_FILE = "no .mnt file"


class Print(NamedTuple):
    """A print of the comparison of the two modes."""

    directory: str  # below the image directory
    size: tuple  # (width, height) of the image file
    ppi: float  # resolution of the file
    stated: str  # where the resolution is stated: "metadata" of the file or "manifest"
    light: bool  # light ridges on a dark background
    minutiae: object  # the number of minutiae of its .mnt file, or one of the three above


# The prints of the training subjects, two for each
TRAINING_PRINTS = [
    # At 500 ppi, used as it is. In its frame of 500x640 pixels the locations
    # of minutiae depend on whether the sizes are Python or numpy numbers.
    Print("sensor_a", (500, 640), 500, "metadata", False, 6),
    # At 1000 ppi: halved to 292x392
    Print("sensor_a", (584, 784), 1000, "metadata", False, 5),
    # At 385 ppi: enlarged to 300x400
    Print("sensor_b", (231, 308), 385, "metadata", False, 5),
    # 157 pixels at 251.2 ppi are 312 pixels at 500 ppi, and 313 with the
    # resolution rounded to 32 bits
    Print("sensor_b", (157, 201), 251.2, "manifest", False, 5),
    # Light ridges, which the preparation has to find out from the image
    Print("", (256, 320), 500, "metadata", True, 1),
    Print("", (240, 300), 500, "metadata", False, ZERO_BYTES),
    Print("", (300, 220), 500, "metadata", False, HEADER_ZERO),
    # Wider than high
    Print("", (400, 300), 500, "metadata", False, 5),
]
# The prints of the held-out subjects. Directory mode stops for a training
# image without .mnt file, so only a held-out subject can have such an image.
VALIDATION_PRINTS = [
    Print("sensor_a", (260, 330), 500, "metadata", False, NO_MNT_FILE),
    Print("sensor_a", (400, 520), 1000, "metadata", False, 3),
    Print("sensor_b", (270, 350), 500, "metadata", True, 2),
    Print("", (280, 280), 385, "metadata", False, 4),
]
FRAME_OF_MOVED_LOCATIONS, FLOAT32_PPI, LIGHT = (
    TRAINING_PRINTS[0],
    TRAINING_PRINTS[3],
    TRAINING_PRINTS[4],
)

SUBJECTS, IMPRESSIONS = 6, 2
VALIDATION_SPLIT = 0.34


def print_image(spec: Print, number: int) -> np.ndarray:
    """The image file of a print: ridges of the same physical width, another slant per print."""
    width, height = spec.size
    image = ridge_image(height, width, period=9.5 * spec.ppi / 500, slant=0.1 + 0.07 * number)
    return 255 - image if spec.light else image


def minutiae_of_print(spec: Print, number: int, size: tuple) -> np.ndarray:
    """
    The minutiae (x, y, angle) of a print in the pixels of its prepared
    image of the given (width, height): positions with fractions, spread
    over the image.
    """
    random_numbers = np.random.default_rng(100 + number)
    minutiae = np.column_stack(
        [
            random_numbers.uniform(8, size[0] - 8, spec.minutiae),
            random_numbers.uniform(8, size[1] - 8, spec.minutiae),
            random_numbers.uniform(0, 2 * np.pi, spec.minutiae),
        ]
    )
    if spec is FRAME_OF_MOVED_LOCATIONS:
        # Positions without fractions at which the two ways of scaling differ
        minutiae[0] = (175.0, 300.0, 0.5)
        minutiae[1] = (45.0, 100.0, 4.0)
    return minutiae


class Corpus(NamedTuple):
    """The files of the comparison of the two modes, and their table."""

    root: object  # holds images, minutiae, resolutions.csv and table
    prints: dict  # (subject, impression) as the modes number them, from 0 -> Print
    training_subjects: list
    validation_subjects: list
    table: SampleTable  # the table that the builder wrote, read again

    def path(self, number: tuple) -> str:
        """The path of the image of (subject, impression) below the image directory."""
        return join(self.prints[number].directory, f"{number[0] + 1}_{number[1] + 1}.png")

    def number_of(self, spec: Print) -> tuple:
        return next(number for number, other in self.prints.items() if other is spec)

    def directory_options(self) -> list:
        """What directory mode has to be told about these files."""
        return [
            *("--minutiae_dir", self.root / "minutiae"),
            *("--num_subjects", SUBJECTS, "--num_impressions", IMPRESSIONS),
            *("--ppi_manifest", self.root / "resolutions.csv", "--ridges", "auto"),
        ]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, train_500k, builder) -> Corpus:
    """
    Six subjects with two prints each, as files that both modes read, the
    .mnt files of an extraction, and the table that the builder makes of
    them. No test changes these files.
    """
    root = tmp_path_factory.mktemp("corpus")
    # The subjects that both modes hold out. The prints are dealt out after
    # that: those that need minutiae go to the subjects that train.
    training_subjects, validation_subjects = train_500k.split_subjects(
        SUBJECTS, VALIDATION_SPLIT, train_500k.VALIDATION_SEED
    )
    assert len(training_subjects) == 4 and len(validation_subjects) == 2
    prints = {}
    for subjects, specs in (
        (training_subjects, TRAINING_PRINTS),
        (validation_subjects, VALIDATION_PRINTS),
    ):
        for position, spec in enumerate(specs):
            prints[(subjects[position // IMPRESSIONS], position % IMPRESSIONS)] = spec
    assert len(prints) == SUBJECTS * IMPRESSIONS

    manifest = ["path,ppi"]
    for number, ((subject, impression), spec) in enumerate(sorted(prints.items())):
        path = join(spec.directory, f"{subject + 1}_{impression + 1}.png")
        if spec.stated == "manifest":
            manifest.append(f"{path},{spec.ppi}")
        save(
            root / "images" / path,
            print_image(spec, number),
            spec.ppi if spec.stated == "metadata" else None,
        )
    (root / "resolutions.csv").write_text("\n".join(manifest) + "\n")

    # The .mnt files, where the extraction writes them: at the path of the
    # image below the minutiae directory. Their header is the size of the
    # prepared image, also where the first line is 0.
    normalizer = SensorNormalizer(
        root_dir=str(root / "images"), manifest=str(root / "resolutions.csv"), ridges="auto"
    )
    for number, ((subject, impression), spec) in enumerate(sorted(prints.items())):
        name = join(spec.directory, f"{subject + 1}_{impression + 1}")
        mnt_path = root / "minutiae" / (name + ".mnt")
        if spec.minutiae == NO_MNT_FILE:
            continue
        if spec.minutiae == ZERO_BYTES:
            os.makedirs(dirname(str(mnt_path)), exist_ok=True)
            open(mnt_path, "w").close()
            continue
        _, info = normalizer.load(str(root / "images" / (name + ".png")), check_polarity=True)
        size = (info.width, info.height)
        write_mnt_file(mnt_path, minutiae_of_print(spec, number, size), size, info.describe())

    # One image has no .mnt file, so the build ends with the status 1
    status = build(
        builder,
        root,
        *("--minutiae_dir", root / "minutiae"),
        *("--ppi_manifest", root / "resolutions.csv", "--ridges", "auto"),
    )
    assert status == 1
    table = SampleTable.read(str(root / "table"), str(root / "images"))
    return Corpus(root, prints, training_subjects, validation_subjects, table)


def test_the_files_of_the_comparison_are_those_that_show_the_faults(corpus, train_500k):
    """The controls that belong to the files; the others are beside the comparisons."""
    table, samples = corpus.table, corpus.table.samples
    assert table.num_rows == len(corpus.prints) == 12
    assert numbers(table.identifiers) == sorted(corpus.prints)

    # Prints at 500, 1000 and 385 ppi by their own metadata, one by the
    # manifest; each is resampled to 500 ppi, and the real binariser has room
    # in all of them
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
    for number, spec in corpus.prints.items():
        row = samples.find(*number)
        prepared = (int(samples.width[row]), int(samples.height[row]))
        assert prepared == resampled_size(*spec.size, spec.ppi)
        assert min(prepared) >= LARGEST_BINARISER_KERNEL
        assert (prepared == spec.size) == (spec.ppi == 500)
    for ppi in (500, 1000, 385):
        assert any(
            spec.ppi == ppi and number[0] in corpus.training_subjects
            for number, spec in corpus.prints.items()
        )

    # The resolution of one training file is another number in 32 bits, and
    # its prepared size depends on that: a table that held it in 32 bits
    # would give an image of another size
    row = samples.find(*corpus.number_of(FLOAT32_PPI))
    ppi = float(samples.ppi[row])
    assert ppi == FLOAT32_PPI.ppi and float(np.float32(ppi)) != ppi
    assert resampled_size(*FLOAT32_PPI.size, ppi) == (312, 400)
    assert resampled_size(*FLOAT32_PPI.size, float(np.float32(ppi))) == (313, 400)
    assert (samples.width[row], samples.height[row]) == (312, 400)

    # The light ridges of two prints were found out from the images, as were
    # the dark ones of the others, and the table holds what was found
    light = sorted(number for number, spec in corpus.prints.items() if spec.light)
    assert len(light) == 2 and corpus.number_of(LIGHT) in light
    taken_as_light = np.flatnonzero(samples.ridges_light)
    assert [numbers(table.identifiers)[row] for row in taken_as_light] == light
    assert {RIDGES_ORIGINS[origin] for origin in samples.ridges_origin.tolist()} == {"estimated"}

    # The two .mnt files without minutiae belong to training subjects; their
    # rows hold an empty list. The image without .mnt file belongs to a
    # held-out subject; its row holds null, and it is the only such row.
    for number, spec in corpus.prints.items():
        row = samples.find(*number)
        mnt_path = corpus.root / "minutiae" / (corpus.path(number)[:-4] + ".mnt")
        if spec.minutiae == NO_MNT_FILE:
            assert number[0] in corpus.validation_subjects
            assert not exists(mnt_path) and not samples.minutiae_known[row]
            continue
        assert samples.minutiae_known[row]
        if spec.minutiae in (ZERO_BYTES, HEADER_ZERO):
            assert number[0] in corpus.training_subjects
            assert len(samples.minutiae_of(row)) == 0
            if spec.minutiae == ZERO_BYTES:
                assert os.path.getsize(mnt_path) == 0
            else:
                with open(mnt_path) as file:
                    assert file.read().startswith(f"0\n{spec.size[0]} {spec.size[1]}\n")
        else:
            assert len(samples.minutiae_of(row)) == spec.minutiae
    assert len(table.rows_without_minutiae()) == 1
    kinds = [spec.minutiae for spec in corpus.prints.values()]
    assert kinds.count(ZERO_BYTES) == kinds.count(HEADER_ZERO) == kinds.count(NO_MNT_FILE) == 1


def test_the_image_without_minutiae_can_only_belong_to_a_held_out_subject(
    corpus, train_500k, tmp_path, monkeypatch, capsys
):
    """
    Control of the comparison: when its subject trains, neither mode starts.
    Directory mode stops for the missing .mnt file, table mode for the row
    without minutiae, and each names the image.
    """
    number = next(n for n, spec in corpus.prints.items() if spec.minutiae == NO_MNT_FILE)
    name = f"{number[0] + 1}_{number[1] + 1}"
    everybody_trains = ["--validation_split", 0]
    capsys.readouterr()
    fake = _FakeExtractor()
    with pytest.raises(SystemExit) as stop:
        run_directory(
            train_500k, monkeypatch, corpus.root, *corpus.directory_options(),
            *everybody_trains, output=tmp_path / "directory", fake=fake,
        )
    message = " ".join(capsys.readouterr().err.split())
    assert stop.value.code == 2 and fake.fits == []
    assert "1 of the 12 files that are needed are missing" in message
    assert f"(--minutiae_dir). The first ones: {name}.mnt." in message

    fake = _FakeExtractor()
    with pytest.raises(SystemExit) as stop:
        run_table(
            train_500k, monkeypatch, corpus.root, *everybody_trains,
            output=tmp_path / "table", fake=fake,
        )
    message = " ".join(capsys.readouterr().err.split())
    assert stop.value.code == 2 and fake.fits == []
    assert "holds no minutiae (null) for 1 of the 12 training images" in message
    assert f"'{corpus.path(number)}' (subject {number[0]}, impression {number[1]})" in message


def fetch(fit: dict, seed: int = 0) -> tuple:
    """
    What a training would be given by the datasets of a fit: for every
    training identifier the image, its minutia map and the weight of the
    map, and every validation image. The random numbers of the augmentation
    are seeded directly before the image of an identifier is fetched, and
    its map is fetched directly after it, because the map takes the pose of
    the image that was loaded last.
    """
    images, maps = fit["fingerprints"], fit["minutia_maps"]
    state = random.getstate()
    try:
        training = {}
        for number, identifier in enumerate(images.ids):
            random.seed(1000 + 10 * number + seed)
            image = images.get(identifier)
            minutia_map, weight = maps.get(identifier)
            training[(identifier.subject, identifier.impression)] = (image, minutia_map, weight)
    finally:
        random.setstate(state)
    validation = {
        (identifier.subject, identifier.impression): fit["validation_fingerprints"].get(identifier)
        for identifier in fit["validation_fingerprints"].ids
    }
    return training, validation


def assert_same_samples(found: tuple, expected: tuple) -> None:
    """Images, minutia maps and weights are equal, sample by sample and bit by bit."""
    (training, validation), (expected_training, expected_validation) = found, expected
    assert list(training) == list(expected_training)
    for number, sample in training.items():
        for tensor, other in zip(sample[:2], expected_training[number][:2]):
            assert tensor.dtype == other.dtype, number
            assert torch.equal(tensor, other), number
        assert sample[2] == expected_training[number][2], number
    assert list(validation) == list(expected_validation)
    for number, image in validation.items():
        assert image.dtype == expected_validation[number].dtype, number
        assert torch.equal(image, expected_validation[number]), number


@pytest.mark.parametrize("canvas_size", [0, 512, 250])
def test_a_training_from_a_table_is_the_training_from_the_directories(
    corpus, train_500k, monkeypatch, canvas_size
):
    """
    main() in both modes on the same files, with a model that keeps what its
    fit is given: the same identifiers, held-out subjects, validation pairs
    and number of classes, and bit-identical images, minutia maps and
    weights, with the random pose and quality of the training, the real
    binariser and the canvas.
    """
    # A canvas of 512 cuts the highest print and pads the others; one of 250
    # is smaller than every print in one direction at least
    longer_sides = [max(resampled_size(*spec.size, spec.ppi)) for spec in corpus.prints.values()]
    assert sorted(longer_sides)[-2] < 512 < max(longer_sides) and min(longer_sides) > 250

    common = ["--validation_split", VALIDATION_SPLIT, "--canvas_size", canvas_size]
    # Every other option that both modes take, with another value than its
    # default: each must reach the training from a table as it does without
    common += ["--learning_rate", 0.002, "--warmup_steps", 7, "--lr_schedule", "cosine"]
    common += ["--gradient_clip_norm", 2.5, "--localization_lr_factor", 0.5, "--patience", 4]
    common += ["--loss_weight_cross_entropy", 2, "--loss_weight_center", 0.5]
    common += ["--loss_weight_minutia_map", 5, "--embedding_dim", 96, "--batch_size", 16]

    def directory_mode(output: str) -> _FakeExtractor:
        return run_directory(
            train_500k, monkeypatch, corpus.root, *corpus.directory_options(), *common,
            output=f"directory_{canvas_size}_{output}",
        )

    directory = directory_mode("a")
    table = run_table(train_500k, monkeypatch, corpus.root, *common, output=f"table_{canvas_size}")
    (fit,), (table_fit,) = directory.fits, table.fits

    # The two runs went through the loaders of their own mode
    inner = fit["minutia_maps"].data_loader._minutia_maps
    table_inner = table_fit["minutia_maps"].data_loader._minutia_maps
    assert type(inner) is FingerFlowMinutiaMapLoader
    assert type(table_inner) is SampleTableMinutiaMapLoader
    assert type(fit["fingerprints"].data_loader._images) is train_500k.CustomFingerprintLoader
    assert type(table_fit["fingerprints"].data_loader._images) is SampleTableImageLoader
    validation_images = table_fit["validation_fingerprints"].data_loader._images
    assert type(validation_images) is SampleTableImageLoader

    # The same identifiers and the same held-out subjects, ...
    training_numbers = sorted(n for n in corpus.prints if n[0] in corpus.training_subjects)
    validation_numbers = sorted(n for n in corpus.prints if n[0] in corpus.validation_subjects)
    for name in ("fingerprints", "minutia_maps", "labels"):
        assert numbers(table_fit[name].ids) == numbers(fit[name].ids) == training_numbers
    assert (
        numbers(table_fit["validation_fingerprints"].ids)
        == numbers(fit["validation_fingerprints"].ids)
        == validation_numbers
    )
    # ... the same number of classes and the same class for every image, ...
    assert table.built == directory.built
    (variant, arguments), = table.built
    assert variant == "LocTexMinu" and arguments["num_training_subjects"] == 4
    labels = [fit["labels"].get(identifier) for identifier in fit["labels"].ids]
    assert [table_fit["labels"].get(i) for i in table_fit["labels"].ids] == labels
    assert sorted(set(labels)) == [0, 1, 2, 3]
    # ... the same validation pairs, ...
    assert pairs(table_fit["validation_benchmark"]) == pairs(fit["validation_benchmark"])
    assert len(pairs(fit["validation_benchmark"])) == 2 + 2
    # ... and the same options of the training
    assert sorted(table_fit) == sorted(fit)
    for name in ("num_epochs", "patience", "learning_rate"):
        assert table_fit[name] == fit[name], name
    rate = table_fit["learning_rate"]
    assert (rate.learning_rate, rate.warmup_steps, rate.schedule) == (0.002, 7, "cosine")
    assert (rate.gradient_clip_norm, rate.localization_factor) == (2.5, 0.5)
    assert table_fit["patience"] == 4 and table_fit["num_epochs"] == 2
    weights = arguments["loss_weights"]
    assert (weights.cross_entropy, weights.center, weights.minutia_map) == (2, 0.5, 5)
    assert arguments["num_dims"] == 96
    # ... which are stored alike, except for what the table replaces
    stored = stored_settings(corpus.root, f"table_{canvas_size}")
    stored_directory = stored_settings(corpus.root, f"directory_{canvas_size}_a")
    assert {key for key in stored_directory if stored[key] != stored_directory[key]} == {
        "num_subjects", "num_impressions", "preparation", "ppi_manifest", "minutiae_dir",
    }
    assert stored["batch_size"] == 16 and stored["canvas_size"] == canvas_size

    # The minutiae as the loaders of the two modes read them, before any
    # pose: the same arrays for every training image
    for identifier in fit["minutia_maps"].ids:
        expected_locations, expected_angles = inner.get_minutiae(identifier)
        locations, angles = table_inner.get_minutiae(identifier)
        number = (identifier.subject, identifier.impression)
        for array, expected in ((locations, expected_locations), (angles, expected_angles)):
            assert array.dtype == expected.dtype and array.shape == expected.shape, number
            assert np.array_equal(array, expected), number
        count = corpus.prints[number].minutiae
        assert len(locations) == (count if isinstance(count, int) else 0)
    # Control: in the frame of 500x640 pixels the locations depend on whether
    # the sizes are Python numbers, as the loader of .mnt files reads them
    # from the header, or numpy numbers, as the arrays of a table hold them.
    # A table loader that passed the sizes on as they are would have shown.
    number = corpus.number_of(FRAME_OF_MOVED_LOCATIONS)
    in_file, _, (width, height) = read_mnt_file(
        str(corpus.root / "minutiae" / (corpus.path(number)[:-4] + ".mnt"))
    )
    assert (width, height) == (500, 640) and in_file[0].tolist() == [175.0, 300.0]
    with_python_numbers = transform_to_input_size(in_file, height, width)
    with_numpy_numbers = transform_to_input_size(in_file, np.int32(height), np.int32(width))
    assert np.array_equal(table_inner.get_minutiae(Identifier(*number))[0], with_python_numbers)
    row = corpus.table.samples.find(*number)
    assert type(corpus.table.samples.width[row]) is np.int32
    # From NumPy 2 on. NumPy 1 scales the locations in their 16 bits with
    # both kinds of numbers, so there is nothing to show.
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        assert with_python_numbers[0, 0] == 114.5 and with_numpy_numbers[0, 0] == 114.4375
        assert not np.array_equal(with_python_numbers, with_numpy_numbers)

    # The samples. First directory mode against itself, in a second run: if
    # it did not repeat with the same seeds, a difference to the table would
    # say nothing
    expected = fetch(fit)
    assert_same_samples(fetch(directory_mode("b").fits[0]), expected)
    # Control: the seeds are what the training samples depend on, ...
    other_poses = fetch(fit, seed=1)
    assert not any(
        torch.equal(sample[0], other[0])
        for sample, other in zip(expected[0].values(), other_poses[0].values())
    )
    # ... every image is the model input of another print, with ridges and
    # background, ...
    images = [sample[0] for sample in expected[0].values()] + list(expected[1].values())
    assert len(images) == 12
    assert all(image.shape == (1, INPUT_SIZE, INPUT_SIZE) for image in images)
    assert all(0.01 < float(image.mean()) < 0.99 for image in images)
    assert not any(
        torch.equal(image, other) for index, image in enumerate(images) for other in images[:index]
    )
    # ... and the minutia maps hold minutiae, except those of the two .mnt
    # files without any, which have the weight 0 in both modes
    map_weights = {number: sample[2] for number, sample in expected[0].items()}
    without = [
        number
        for number in training_numbers
        if corpus.prints[number].minutiae in (ZERO_BYTES, HEADER_ZERO)
    ]
    assert len(without) == 2 and all(map_weights[number] == 0.0 for number in without)
    assert sum(map_weights.values()) >= len(training_numbers) - 3
    for _, minutia_map, weight in expected[0].values():
        assert minutia_map.shape == (6, 128, 128) and bool(minutia_map.any()) == (weight == 1.0)

    assert_same_samples(fetch(table_fit), expected)
    assert_same_samples(fetch(table_fit, seed=1), other_poses)


# ---------------------------------------------------------------------------
# The held-out subjects and the validation pairs
# ---------------------------------------------------------------------------


def table_of(rows, image_dir="images") -> SampleTable:
    """
    A table with a row for each (subject, impression), made as arrays: rows
    of small images at 500 ppi whose minutiae are not known. No files.
    """
    rows = sorted(rows)
    count = len(rows)
    paths = [f"{subject}_{impression}.png".encode() for subject, impression in rows]
    offsets = np.zeros(count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(path) for path in paths])
    samples = Samples(
        subject=np.array([subject for subject, _ in rows], dtype=np.int32),
        impression=np.array([impression for _, impression in rows], dtype=np.int32),
        path_data=np.frombuffer(b"".join(paths), dtype=np.uint8),
        path_offsets=offsets,
        file_size=np.full(count, 1000, dtype=np.int64),
        ppi=np.full(count, 500.0),
        ppi_origin=np.zeros(count, dtype=np.uint8),
        ridges_origin=np.zeros(count, dtype=np.uint8),
        ridges_light=np.zeros(count, dtype=np.bool_),
        source_width=np.full(count, 48, dtype=np.int32),
        source_height=np.full(count, 64, dtype=np.int32),
        width=np.full(count, 48, dtype=np.int32),
        height=np.full(count, 64, dtype=np.int32),
        minutiae_known=np.zeros(count, dtype=np.bool_),
        minutiae_offsets=np.zeros(count + 1, dtype=np.int64),
        minutiae=np.zeros((0, 3), dtype=np.float32),
    )
    return SampleTable(samples, image_dir)


def result_or_error(function, *arguments):
    """What the function returns, or the message of the ValueError that it raises."""
    try:
        return function(*arguments)
    except ValueError as error:
        return f"ValueError: {error}"


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("limit", [0, 2, 3, 5000])
def test_the_split_of_a_grid_is_the_split_without_a_table(train_500k, seed, limit):
    """
    For subjects 0 .. n - 1 with the same number of impressions each, two or
    more: the lists of split_subjects, or its error, for every share.
    """
    cases, errors = 0, set()
    for subjects in (2, 3, 4, 5, 9, 10, 11, 20, 21, 37, 50, 64, 100, 257):
        for share in (0, 0.05, 0.1, 0.2, 0.25, 0.29, 0.34, 0.5, 0.75, 0.9, 0.99, 1.0, -0.1):
            expected = result_or_error(train_500k.split_subjects, subjects, share, seed, limit)
            for impressions in (2, 3, 10):
                found = result_or_error(
                    train_500k.split_table_subjects,
                    np.arange(subjects, dtype=np.int32),
                    np.full(subjects, impressions),
                    share,
                    seed,
                    limit,
                )
                assert found == expected, (subjects, impressions, share)
                cases += 1
            if isinstance(expected, str):
                errors.add("no share" if "below 1" in expected else expected.split(" of ")[0])
            else:
                training, validation = expected
                assert all(type(subject) is int for subject in found[0] + found[1])
                assert sorted(training + validation) == list(range(subjects))
    assert cases == 14 * 13 * 3
    # Both kinds of errors were among the cases: a share that is no share, and
    # shares that hold out fewer than two subjects
    assert "no share" in errors and "ValueError: validation_split 0.05" in errors
    assert "must be at least 0 and below 1" in result_or_error(
        train_500k.split_table_subjects, [0, 1, 2, 3], [2, 2, 2, 2], 1.0, seed, limit
    )
    assert "must not be negative" in result_or_error(
        train_500k.split_table_subjects, [0, 1, 2, 3], [2, 2, 2, 2], 0.5, seed, -1
    )
    assert result_or_error(
        train_500k.split_table_subjects, [0, 1, 2, 3], [2, 2, 2, 2], 0.5, seed, -1
    ) == result_or_error(train_500k.split_subjects, 4, 0.5, seed, -1)


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(
    "subjects, impressions", [(4, 2), (7, 1), (10, 3), (25, 2), (30, 10), (60, 4), (130, 2)]
)
def test_identifiers_and_pairs_of_a_grid_are_those_without_a_table(
    train_500k, subjects, impressions, seed
):
    """
    create_table_training_and_validation_ids and
    create_table_validation_benchmark on the table of a complete grid,
    against create_training_and_validation_ids and
    create_validation_benchmark for its two numbers.
    """
    table = table_of((s, i) for s in range(subjects) for i in range(impressions))
    of_table = {id(identifier) for identifier in table.identifiers}
    compared = 0
    for share in (0, 0.1, 0.2, 0.5, 0.9):
        for limit in (0, 3, 5000):
            arguments = (share, seed, limit)
            expected = result_or_error(
                train_500k.create_training_and_validation_ids, subjects, impressions, *arguments
            )
            found = result_or_error(
                train_500k.create_table_training_and_validation_ids, table, *arguments
            )
            if isinstance(expected, str):
                # The same cases are refused: a share that holds out fewer
                # than two subjects, and subjects with one impression
                assert isinstance(found, str), (share, limit)
                assert impressions == 1 or found == expected
                continue
            training, validation = found
            assert numbers(training) == numbers(expected[0])
            assert training.num_subjects == expected[0].num_subjects
            # The sets hold the Identifier objects of the table
            assert all(id(identifier) in of_table for identifier in training)
            if expected[1] is None:
                assert validation is None and share == 0
                continue
            assert numbers(validation) == numbers(expected[1])
            assert all(id(identifier) in of_table for identifier in validation)
            for comparisons in (10, 3):
                benchmark = train_500k.create_table_validation_benchmark(
                    validation, comparisons, seed
                )
                assert pairs(benchmark) == pairs(
                    train_500k.create_validation_benchmark(
                        expected[1], impressions, comparisons, seed
                    )
                )
                # The pairs hold the Identifier objects of the set as well
                assert all(
                    id(sample) in of_table
                    for comparison in benchmark._comparisons
                    for sample in (comparison.sample1, comparison.sample2)
                )
                mated = sum(comparison.mated for comparison in benchmark._comparisons)
                counted = train_500k.count_validation_pairs(
                    [impressions] * validation.num_subjects, comparisons
                )
                assert counted == (mated, len(benchmark._comparisons) - mated)
                compared += 1
            # Another seed gives other pairs, where pairs are drawn
            if validation.num_subjects > 2:
                other = train_500k.create_table_validation_benchmark(validation, 10, seed + 1)
                assert pairs(other) != pairs(
                    train_500k.create_table_validation_benchmark(validation, 10, seed)
                )
    assert compared > 0 or impressions == 1 or subjects < 5


def split_by_the_rule(impressions: dict, share: float, seed: int, limit: int) -> tuple:
    """
    The rule of the split of a table, written out with the standard library:
    the subjects in ascending order, one shuffle of the list, and held out
    are the first subjects of the shuffled list with two impressions, as
    many as the share of all subjects asks for, at most the limit.
    """
    subjects = sorted(impressions)
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    wanted = int(len(subjects) * share)
    if limit > 0:
        wanted = min(wanted, limit)
    held_out = [subject for subject in shuffled if impressions[subject] >= 2][:wanted]
    return [subject for subject in subjects if subject not in held_out], sorted(held_out)


def test_the_split_of_uneven_subjects_holds_out_the_first_shuffled_ones_with_two_impressions(
    train_500k,
):
    random_numbers = np.random.default_rng(5)
    splits = refused = short = 0
    for case in range(300):
        count = int(random_numbers.integers(2, 60))
        # Numbers with gaps, in no order, and one to six impressions each,
        # with many subjects of one impression in some of the sets
        subjects = random_numbers.choice(10 * count, size=count, replace=False)
        most = int(random_numbers.integers(1, 7))
        impressions = random_numbers.integers(1, most + 1, size=count)
        share = float(random_numbers.choice([0.1, 0.2, 0.34, 0.5, 0.8]))
        limit = int(random_numbers.choice([0, 0, 3, 7]))
        seed = int(random_numbers.integers(0, 100))
        by_subject = dict(zip(subjects.tolist(), impressions.tolist()))
        training, validation = split_by_the_rule(by_subject, share, seed, limit)
        found = result_or_error(
            train_500k.split_table_subjects, subjects, impressions, share, seed, limit
        )
        wanted = train_500k.wanted_validation_subjects(count, share, limit)
        assert wanted == min(int(count * share), limit or count)
        if len(validation) < 2:
            # Too small a share, or too few subjects with two impressions
            reason = "validation needs at least 2" if wanted < 2 else "with 2 or more impressions"
            assert isinstance(found, str) and reason in found, case
            refused += 1
            continue
        assert found == (training, validation), case
        # A subject is in one of the lists, and never held out with one impression
        assert sorted(training + validation) == sorted(by_subject)
        assert all(by_subject[subject] >= 2 for subject in validation)
        assert len(validation) <= wanted
        short += len(validation) < wanted
        splits += 1
    # The cases held all three: a split, a refusal, and a share that the
    # subjects with two impressions could not meet
    assert splits > 100 and refused > 20 and short > 5


def test_subjects_with_one_impression_train_and_a_share_that_cannot_be_met_holds_out_the_rest(
    train_500k,
):
    # Ten subjects, three of them with two or more impressions
    subjects = [3, 4, 9, 10, 11, 30, 31, 32, 70, 71]
    impressions = [1, 2, 1, 1, 5, 1, 1, 1, 2, 1]
    for seed in range(20):
        # Half of the subjects are asked for; the three that can be held out are
        training, validation = train_500k.split_table_subjects(subjects, impressions, 0.5, seed)
        assert validation == [4, 11, 70]
        assert training == [3, 9, 10, 30, 31, 32, 71]
        # Two are asked for: two of the three, by the seed
        training, validation = train_500k.split_table_subjects(subjects, impressions, 0.2, seed)
        assert len(validation) == 2 and set(validation) < {4, 11, 70}
        assert sorted(training + validation) == subjects
    held_out = {
        tuple(train_500k.split_table_subjects(subjects, impressions, 0.2, seed)[1])
        for seed in range(20)
    }
    assert held_out == {(4, 11), (4, 70), (11, 70)}
    # The order in which the subjects are given does not count
    assert train_500k.split_table_subjects(
        subjects[::-1], impressions[::-1], 0.2, 42
    ) == train_500k.split_table_subjects(subjects, impressions, 0.2, 42)

    # Fewer than two subjects with two impressions: no validation is possible
    for impressions in ([1] * 10, [1, 1, 1, 1, 7, 1, 1, 1, 1, 1]):
        with pytest.raises(ValueError) as error:
            train_500k.split_table_subjects(subjects, impressions, 0.5)
        message = str(error.value)
        assert "needs at least 2 subjects with 2 or more impressions" in message
        assert f"{sum(n > 1 for n in impressions)} of the 10 subjects have that many" in message
        assert "--validation_split 0" in message
    # A share that asks for fewer than two subjects is the error of split_subjects
    with pytest.raises(ValueError) as error:
        train_500k.split_table_subjects(subjects, [2] * 10, 0.1)
    assert str(error.value) == result_or_error(train_500k.split_subjects, 10, 0.1)[12:]


def test_a_share_of_zero_holds_out_nothing_and_draws_nothing(train_500k, monkeypatch):
    def no_generator(*args):
        raise AssertionError("a share of 0 needs no random numbers")

    monkeypatch.setattr(train_500k.random, "Random", no_generator)
    assert train_500k.split_table_subjects([8, 2, 5], [1, 3, 2], 0) == ([2, 5, 8], [])
    assert train_500k.split_table_subjects([8, 2, 5], [1, 3, 2], 0.0, seed=7) == ([2, 5, 8], [])
    table = table_of([(8, 0), (2, 0), (2, 4), (5, 1)])
    training, validation = train_500k.create_table_training_and_validation_ids(table)
    assert validation is None
    assert numbers(training) == [(2, 0), (2, 4), (5, 1), (8, 0)]


def test_the_split_and_the_pairs_do_not_change_the_global_random_state(train_500k):
    table = table_of((s, i) for s in range(0, 40, 2) for i in range(1 + s % 3))
    random.seed(123)
    expected = random.random()
    random.seed(123)
    _, validation = train_500k.create_table_training_and_validation_ids(table, 0.3)
    train_500k.create_table_validation_benchmark(validation)
    assert random.random() == expected
    # The same seed gives the same subjects and the same pairs
    _, again = train_500k.create_table_training_and_validation_ids(table, 0.3)
    assert numbers(again) == numbers(validation)
    assert pairs(train_500k.create_table_validation_benchmark(again)) == pairs(
        train_500k.create_table_validation_benchmark(validation)
    )


def pairs_by_the_rule(impressions: dict, comparisons_per_subject: int, seed: int) -> list:
    """
    The rule of the validation pairs of a table, written out with the calls
    of create_validation_benchmark: all pairs of the impressions of a
    subject, then for every subject the other subjects that are drawn, and
    for each of them an impression of the subject and one of the other.
    """
    rng = random.Random(seed)
    subjects = sorted(impressions)
    result = []
    for subject in subjects:
        own = impressions[subject]
        result += [((subject, a), (subject, b)) for i, a in enumerate(own) for b in own[i + 1:]]
    for subject in subjects:
        other_subjects = [s for s in subjects if s != subject]
        if len(other_subjects) >= comparisons_per_subject:
            selected = rng.sample(other_subjects, comparisons_per_subject)
        else:
            selected = other_subjects
        for other in selected:
            first = impressions[subject][rng.randint(0, len(impressions[subject]) - 1)]
            second = impressions[other][rng.randint(0, len(impressions[other]) - 1)]
            result.append(((subject, first), (other, second)))
    return result


@pytest.mark.parametrize("comparisons_per_subject", [10, 4])
def test_the_validation_pairs_of_uneven_subjects_are_made_of_the_impressions_that_exist(
    train_500k, comparisons_per_subject
):
    random_numbers = np.random.default_rng(11)
    for case in range(40):
        count = int(random_numbers.integers(2, 30))
        subjects = sorted(random_numbers.choice(1000, size=count, replace=False).tolist())
        impressions = {}
        for subject in subjects:
            # Two to six impressions, with gaps in their numbers and not from 0
            held = int(random_numbers.integers(2, 7))
            impressions[subject] = sorted(
                random_numbers.choice(20, size=held, replace=False).tolist()
            )
        validation = IdentifierSet(
            [Identifier(subject, i) for subject in subjects for i in impressions[subject]]
        )
        seed = int(random_numbers.integers(0, 100))
        benchmark = train_500k.create_table_validation_benchmark(
            validation, comparisons_per_subject, seed
        )
        found = pairs(benchmark)
        assert found == pairs_by_the_rule(impressions, comparisons_per_subject, seed), case

        # Every pair is made of images that the set holds, ...
        held = set(numbers(validation))
        assert all(first in held and second in held for first, second in found)
        # ... the mated pairs are every two impressions of every subject, once, ...
        mated = [pair for pair in found if pair[0][0] == pair[1][0]]
        assert len(set(mated)) == len(mated)
        assert all(first[1] < second[1] for first, second in mated)
        per_subject = [len(impressions[subject]) for subject in subjects]
        assert len(mated) == sum(n * (n - 1) // 2 for n in per_subject)
        # ... and every subject is compared with as many other subjects as
        # asked for, or with all if there are fewer, each another one
        non_mated = found[len(mated):]
        assert all(first[0] != second[0] for first, second in non_mated)
        others = min(comparisons_per_subject, count - 1)
        for subject in subjects:
            with_others = [second[0] for first, second in non_mated if first[0] == subject]
            assert len(with_others) == others and len(set(with_others)) == others
        assert train_500k.count_validation_pairs(per_subject, comparisons_per_subject) == (
            len(mated),
            len(non_mated),
        )
    assert train_500k.count_validation_pairs([]) == (0, 0)
    assert train_500k.count_validation_pairs([4]) == (6, 0)
    # What the limit of validation subjects stands for with 10 impressions each
    limit = train_500k.DEFAULT_MAX_VALIDATION_SUBJECTS
    assert train_500k.count_validation_pairs([10] * limit) == (225_000, 50_000)
    assert train_500k.MANY_VALIDATION_IMAGES == 10 * limit
    assert train_500k.MANY_VALIDATION_PAIRS == 225_000 + 50_000


# ---------------------------------------------------------------------------
# Small sets for the tests that do not fetch a sample
# ---------------------------------------------------------------------------

GRID = grid(4, 2)  # 1_1.png .. 4_2.png: four subjects with two impressions
NO_VALIDATION = ["--validation_split", 0]
# What directory mode has to be told about GRID
GRID_OPTIONS = ["--num_subjects", 4, "--num_impressions", 2, *NO_VALIDATION]


def write_grid(root, minutiae: bool = False) -> None:
    """The small prints of GRID in root/images, and their .mnt files in root/minutiae."""
    write_small_prints(root / "images", GRID)
    if minutiae:
        write_small_minutiae(root / "minutiae", GRID)


def snapshot(directory) -> Optional[dict]:
    """The files of a directory with their bytes; None if it is not there."""
    if not exists(directory):
        return None
    files = {}
    for path, _, names in os.walk(directory):
        for name in names:
            with open(join(path, name), "rb") as file:
                files[os.path.relpath(join(path, name), directory)] = file.read()
    return files


def refused(runner, train_500k, monkeypatch, capsys, root, *options, output="model", **kwargs):
    """
    A start that is refused: main() stops with the exit status 2, builds no
    model, and leaves --output_dir as it was (a new one is not made).
    Returns the message.
    """
    before = snapshot(root / output)
    message = stops(
        runner, train_500k, monkeypatch, capsys, root, *options, output=output, **kwargs
    )
    assert snapshot(root / output) == before
    return message


# ---------------------------------------------------------------------------
# The stored settings: which training continues
# ---------------------------------------------------------------------------


def settings_as_main_stores_them(root) -> dict:
    """
    training_settings.json as scripts/train_500k.py stored it before it knew
    sample tables (the 22 keys of its main()), for a training of two epochs
    on the grid in root/images with GRID_OPTIONS and nothing else.
    """
    from flx.setup import config

    return {
        "variant": "LocTex",
        "embedding_dim": 256,
        "num_subjects": 4,
        "num_impressions": 2,
        "num_training_subjects": 4,
        "validation_split": 0.0,
        "max_validation_subjects": 5000,
        "validation_seed": 42,
        "batch_size": 64,
        "canvas_size": 0,
        "preparation": {
            "target_ppi": 500.0,
            "ppi": None,
            "ppi_manifest_sha256": None,
            "ppi_manifest_entries": 0,
            "assume_ppi": None,
            "ridges": "dark",
        },
        "ppi_manifest": None,
        "dataset_dir": str(root / "images"),
        "minutiae_dir": None,
        "num_epochs": 2,
        "patience": 3,
        "learning_rate": config.LEARNING_RATE,
        "warmup_steps": config.LEARNING_RATE_WARMUP_STEPS,
        "lr_schedule": config.LEARNING_RATE_SCHEDULE,
        "gradient_clip_norm": config.GRADIENT_CLIP_NORM,
        "localization_lr_factor": config.LOCALIZATION_LEARNING_RATE_FACTOR,
        "loss_weights": {
            "cross_entropy": config.LOSS_WEIGHT_CROSS_ENTROPY,
            "center": config.LOSS_WEIGHT_CENTER,
            "minutia_map": None,
        },
    }


def write_training_of_main(root, output="model") -> dict:
    """
    The output directory of a training of two epochs as the script left it
    before it knew sample tables: the two settings files, written as its
    _write_json wrote them, the log and the two model files. Nothing of it is
    made by the script under test. Returns the settings.
    """
    settings = settings_as_main_stores_them(root)
    preparation = dict(settings["preparation"], canvas_size=0, ppi_manifest=None)
    os.makedirs(root / output)
    for name, content in (
        ("training_settings.json", settings),
        ("data_preparation.json", preparation),
    ):
        with open(root / output / name, "w") as file:
            json.dump(content, file, indent=2, sort_keys=True)
            file.write("\n")
    _FakeExtractor().fit(str(root / output), 2)
    return settings


def test_directory_mode_stores_the_settings_that_it_stored_before_there_were_tables(
    train_500k, tmp_path, monkeypatch, capsys
):
    """The same keys and values, byte by byte: a table adds nothing where none is used."""
    write_grid(tmp_path)
    write_training_of_main(tmp_path, output="before")
    capsys.readouterr()
    run_directory(train_500k, monkeypatch, tmp_path, *GRID_OPTIONS)
    # Nor does a run without a table say anything about tables
    output = capsys.readouterr()
    assert "Training complete! 2 epoch(s) trained in this run" in output.out
    assert "sample table" not in output.out.lower() and "--samples" not in output.out
    assert output.err == ""
    for name in ("training_settings.json", "data_preparation.json"):
        with open(tmp_path / "model" / name, "rb") as file:
            stored = file.read()
        with open(tmp_path / "before" / name, "rb") as file:
            assert stored == file.read(), name
    assert len(stored_settings(tmp_path)) == 22 and "samples" not in stored_settings(tmp_path)
    assert sorted(os.listdir(tmp_path / "model")) == sorted(os.listdir(tmp_path / "before"))


def test_a_training_stored_without_a_table_continues_without_one_and_not_with_one(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path)
    assert build(builder, tmp_path) == 0
    settings = write_training_of_main(tmp_path)
    capsys.readouterr()

    # With a table of the very same files it is another training
    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *NO_VALIDATION)
    assert "holds a training with other settings" in message
    assert "num_subjects: stored 4, now null" in message
    assert "num_impressions: stored 2, now null" in message
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    assert f'preparation: stored {json.dumps(settings["preparation"], sort_keys=True)}' in message
    assert f'now {{"samples_rows": 8, "samples_sha256": "{table.checksum}", ' in message
    assert '"samples_subjects": 4, "target_ppi": 500.0}' in message
    assert "use a new --output_dir" in message and "training_settings.json" in message

    # Without a table it continues, as it always did
    fake = run_directory(train_500k, monkeypatch, tmp_path, *GRID_OPTIONS, "--num_epochs", 3)
    assert len(fake.fits) == 1
    output = capsys.readouterr().out
    assert "Training complete! 1 epoch(s) trained in this run, 3 in the log." in output
    assert stored_settings(tmp_path) == dict(settings, num_epochs=3)


def test_a_training_before_the_settings_file_existed_is_refused_with_a_table(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    """Such a directory holds data_preparation.json only, and that is compared."""
    write_grid(tmp_path)
    assert build(builder, tmp_path) == 0
    write_training_of_main(tmp_path)
    (tmp_path / "model" / "training_settings.json").unlink()
    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *NO_VALIDATION)
    assert "preparation: stored " in message and '"samples_sha256"' in message
    assert "data_preparation.json)" in message


def test_a_training_from_a_table_continues_with_the_same_rows_and_with_nothing_else(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path, minutiae=True)
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    split = ["--validation_split", 0]
    run_table(train_500k, monkeypatch, tmp_path, *split)
    stored = stored_settings(tmp_path)
    # The table stands where the resolution settings stood, by its values
    assert stored["preparation"] == {
        "samples_sha256": table.checksum,
        "samples_rows": 8,
        "samples_subjects": 4,
        "target_ppi": 500.0,
    }
    assert stored["samples"] == str(tmp_path / "table")
    capsys.readouterr()

    # The same command continues
    fake = run_table(train_500k, monkeypatch, tmp_path, *split, "--num_epochs", 3)
    assert len(fake.fits) == 1
    output = capsys.readouterr().out
    assert "Training complete! 1 epoch(s) trained in this run, 3 in the log." in output
    assert "num_epochs 2 -> 3" in output

    # So does a table of the same rows in another directory, split into
    # other files: the rows count, not where or how they are written
    monkeypatch.setattr(builder, "FILES_PER_PART", 3)
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae", table="again") == 0
    assert len(os.listdir(tmp_path / "again")) == 3 and len(os.listdir(tmp_path / "table")) == 1
    run_table(train_500k, monkeypatch, tmp_path, *split, "--num_epochs", 4, table="again")
    assert "Training complete! 1 epoch(s) trained" in capsys.readouterr().out
    stored = stored_settings(tmp_path)
    assert stored["samples"] == str(tmp_path / "again")
    assert stored["preparation"]["samples_sha256"] == table.checksum

    # Without the table it is another training, also on the same files
    message = refused(
        run_directory, train_500k, monkeypatch, capsys, tmp_path, *GRID_OPTIONS, "--num_epochs", 5,
        *("--minutiae_dir", tmp_path / "minutiae"),
    )
    assert "num_subjects: stored null, now 4" in message
    assert "num_impressions: stored null, now 2" in message
    assert f'preparation: stored {json.dumps(stored["preparation"], sort_keys=True)}' in message
    assert "use a new --output_dir" in message

    # And so it is with another table: one that lacks a file, ...
    shutil.copytree(tmp_path / "images", tmp_path / "fewer")
    (tmp_path / "fewer" / "4_2.png").unlink()
    options = ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, tmp_path, *options, table="smaller", images="fewer") == 0
    smaller = SampleTable.read(str(tmp_path / "smaller"), str(tmp_path / "images"))
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, *split, "--num_epochs", 5,
        table="smaller",
    )
    # The message shows both checksums and the numbers of rows and subjects
    assert "variant" not in message and "num_subjects" not in message
    assert f'"samples_rows": 8, "samples_sha256": "{table.checksum}"' in message
    assert f'"samples_rows": 7, "samples_sha256": "{smaller.checksum}"' in message
    assert message.count('"samples_subjects": 4') == 2
    assert table.checksum != smaller.checksum

    # ... one that differs in a single value, of its last row: the angle of
    # one minutia of the last image
    last = tmp_path / "minutiae" / "4_2.mnt"
    kept = last.read_text()
    assert kept.splitlines()[-1].split()[2] == "2.000000"
    last.write_text(kept.replace("2.000000", "2.000001"))
    assert build(builder, tmp_path, *options, table="one value") == 0
    last.write_text(kept)
    one_value = SampleTable.read(str(tmp_path / "one value"), str(tmp_path / "images"))
    assert one_value.checksum != table.checksum and one_value.num_rows == 8
    differ = one_value.samples.minutiae != table.samples.minutiae
    assert differ.sum() == 1 and differ[-1, 2]
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, *split, "--num_epochs", 5,
        table="one value",
    )
    assert message.count(": stored ") == 1 and "preparation: stored " in message
    assert f'"samples_rows": 8, "samples_sha256": "{one_value.checksum}"' in message

    # ... and one whose rows hold other values for the same files: here the
    # table that was built without the minutiae (the variant is another, too)
    assert build(builder, tmp_path, table="without") == 0
    without = SampleTable.read(str(tmp_path / "without"), str(tmp_path / "images"))
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, *split, "--num_epochs", 5,
        table="without",
    )
    assert 'variant: stored "LocTexMinu", now "LocTex"' in message
    assert f'"samples_rows": 8, "samples_sha256": "{without.checksum}"' in message
    assert without.checksum != table.checksum

    # It can be continued all the same, at the user's own risk
    run_table(
        train_500k, monkeypatch, tmp_path, *split, "--num_epochs", 5, "--allow_changed_settings",
        table="smaller",
    )
    output = capsys.readouterr().out
    assert "Continuing although preparation changed (--allow_changed_settings)" in output
    assert stored_settings(tmp_path)["preparation"]["samples_sha256"] == smaller.checksum


def test_table_mode_stores_the_settings_of_directory_mode_with_the_table_in_their_place(
    train_500k, builder, tmp_path, monkeypatch
):
    write_grid(tmp_path, minutiae=True)
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    options = ["--validation_split", 0, "--learning_rate", 0.003, "--batch_size", 16]
    options += ["--embedding_dim", 96, "--canvas_size", 512, "--localization", "off"]
    options += ["--warmup_steps", 7, "--lr_schedule", "cosine", "--gradient_clip_norm", 2.5]
    options += ["--loss_weight_cross_entropy", 2, "--loss_weight_center", 3, "--patience", 5]
    options += ["--loss_weight_minutia_map", 5, "--validation_seed", 7]
    options += ["--max_validation_subjects", 9]
    run_directory(
        train_500k, monkeypatch, tmp_path, *GRID_OPTIONS, *options,
        *("--minutiae_dir", tmp_path / "minutiae"), output="directory",
    )
    run_table(train_500k, monkeypatch, tmp_path, *options)
    directory, table = stored_settings(tmp_path, "directory"), stored_settings(tmp_path)

    # Every key of directory mode, and the table
    assert set(table) == set(directory) | {"samples"} and "samples" not in directory
    different = {key for key in directory if table[key] != directory[key]}
    assert different == {"num_subjects", "num_impressions", "preparation", "minutiae_dir"}
    # What the table replaces is null, ...
    for key in ("num_subjects", "num_impressions", "ppi_manifest", "minutiae_dir"):
        assert table[key] is None, key
    assert (directory["num_subjects"], directory["num_impressions"]) == (4, 2)
    # ... what the options say is what they say without a table, ...
    assert table["variant"] == directory["variant"] == "TexMinu"
    assert table["num_training_subjects"] == directory["num_training_subjects"] == 4
    assert table["learning_rate"] == 0.003 and table["batch_size"] == 16
    assert table["embedding_dim"] == 96 and table["canvas_size"] == 512
    assert table["loss_weights"] == {"cross_entropy": 2.0, "center": 3.0, "minutia_map": 5.0}
    assert (table["warmup_steps"], table["lr_schedule"], table["patience"]) == (7, "cosine", 5)
    assert (table["validation_seed"], table["max_validation_subjects"]) == (7, 9)
    assert table["gradient_clip_norm"] == 2.5 and table["localization_lr_factor"] is None
    assert table["dataset_dir"] == directory["dataset_dir"] == str(tmp_path / "images")
    # ... and both the fixed and the changeable settings are all there
    assert set(train_500k.FIXED_SETTINGS + train_500k.CHANGEABLE_SETTINGS) <= set(table)
    assert "samples" not in train_500k.FIXED_SETTINGS + train_500k.CHANGEABLE_SETTINGS
    assert sorted(table["preparation"]) == [
        "samples_rows", "samples_sha256", "samples_subjects", "target_ppi",
    ]
    # data_preparation.json holds the table as well
    with open(tmp_path / "model" / "data_preparation.json") as file:
        assert json.load(file) == dict(table["preparation"], canvas_size=512, ppi_manifest=None)


# ---------------------------------------------------------------------------
# What only a table can describe
# ---------------------------------------------------------------------------

# Files with names of their own in three directories, and the finger and the
# capture that a sample list gives each: numbers with gaps, one to four
# captures per finger, one finger with a single image.
UNEVEN = {
    "plain/00002239_plain_05.png": (7, 0),
    "roll/00002239_roll_05.png": (7, 3),
    "a.png": (2, 0),
    "b.png": (2, 1),
    "c.png": (2, 5),
    "roll/only one.png": (40, 2),
    "d.png": (11, 0),
    "e.png": (11, 1),
    "plain/f.png": (300, 4),
    "plain/g.png": (300, 9),
    "h1.png": (1000, 0),
    "h2.png": (1000, 1),
    "h3.png": (1000, 2),
    "h4.png": (1000, 3),
}


def uneven_impressions() -> dict:
    """The impressions of every subject of UNEVEN, in ascending order."""
    impressions = {}
    for subject, impression in sorted(UNEVEN.values()):
        impressions.setdefault(subject, []).append(impression)
    return impressions


def write_uneven(builder, root, *options) -> SampleTable:
    """The files of UNEVEN, their .mnt files and their sample list, and the table root/table."""
    write_small_prints(root / "images", UNEVEN)
    write_small_minutiae(root / "minutiae", UNEVEN)
    lines = ["path,subject,impression"]
    lines += [f"{path},{subject},{impression}" for path, (subject, impression) in UNEVEN.items()]
    (root / "files.csv").write_text("\n".join(lines) + "\n")
    status = build(
        builder, root, "--sample_list", root / "files.csv", "--minutiae_dir", root / "minutiae",
        *options,
    )
    assert status == 0
    return SampleTable.read(str(root / "table"), str(root / "images"))


def identity(image: torch.Tensor) -> torch.Tensor:
    return image


def test_other_file_names_gaps_and_uneven_numbers_of_impressions_train(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    table = write_uneven(builder, tmp_path)
    capsys.readouterr()
    # The prints are too small for the binariser, and a random shift would
    # move their minutiae out of them: this test needs neither
    monkeypatch.setattr(train_500k, "LazilyAllocatedBinarizer", lambda _: identity)
    monkeypatch.setattr(train_500k, "POSE_AUGMENTATION", None)
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0.34)
    lines = printed(capsys)
    (fit,) = fake.fits

    # Two of the six subjects are held out. With this seed the shuffled list
    # has the subject with a single image among its first two: it trains, and
    # the next subject with two images is held out in its place.
    impressions = uneven_impressions()
    assert sorted(impressions) == [2, 7, 11, 40, 300, 1000]
    per_subject = {subject: len(held) for subject, held in impressions.items()}
    training, validation = split_by_the_rule(per_subject, 0.34, 42, 5000)
    assert len(validation) == 2 and 40 in training
    shuffled = sorted(impressions)
    random.Random(42).shuffle(shuffled)
    assert 40 in shuffled[:2] and set(validation) != set(shuffled[:2])

    # All images of the training subjects train, and none of a held-out subject
    training_numbers = sorted(n for n in UNEVEN.values() if n[0] in training)
    validation_numbers = sorted(n for n in UNEVEN.values() if n[0] in validation)
    for name in ("fingerprints", "minutia_maps", "labels"):
        assert numbers(fit[name].ids) == training_numbers
    assert numbers(fit["validation_fingerprints"].ids) == validation_numbers
    assert len(training_numbers) + len(validation_numbers) == len(UNEVEN) == table.num_rows
    # One class per training subject, numbered without the gaps of the subjects
    assert [arguments["num_training_subjects"] for _, arguments in fake.built] == [4]
    assert stored_settings(tmp_path)["num_training_subjects"] == 4
    labels = {i.subject: fit["labels"].get(i) for i in fit["labels"].ids}
    assert labels == {subject: label for label, subject in enumerate(training)}
    # The validation pairs are made of the impressions that the subjects have
    held = {subject: impressions[subject] for subject in validation}
    assert pairs(fit["validation_benchmark"]) == pairs_by_the_rule(held, 10, 42)

    # Every sample is served: the image of its file and the minutiae of its row
    files = fit["fingerprints"].data_loader._images
    for path, number in UNEVEN.items():
        image = files.get(Identifier(*number))
        on_disk = np.array(Image.open(tmp_path / "images" / path))
        assert torch.equal(image, torch.from_numpy(on_disk)[None] / 255), path
    for number in training_numbers:
        assert fit["fingerprints"].get(Identifier(*number)).shape == (1, INPUT_SIZE, INPUT_SIZE)
        minutia_map, weight = fit["minutia_maps"].get(Identifier(*number))
        assert weight == 1.0 and minutia_map.any()
    for number in validation_numbers:
        image = fit["validation_fingerprints"].get(Identifier(*number))
        assert image.shape == (1, INPUT_SIZE, INPUT_SIZE)

    # What the start prints about the set
    counts = sorted(per_subject[subject] for subject in training)
    held_counts = sorted(per_subject[subject] for subject in validation)
    mated = sum(n * (n - 1) // 2 for n in held_counts)
    assert f"Samples: {tmp_path / 'table'} (14 rows, 6 subjects)" in lines
    assert f"Dataset: {tmp_path / 'images'}" in lines
    assert "Subjects: 6 total" in lines
    assert "Training: 4 subjects" in lines
    assert "Validation: 2 subjects held out (seed 42)" in lines
    assert (
        f"Impressions: {counts[0]} to {counts[-1]} per training subject "
        f"(median {np.median(counts):g})" in lines
    )
    assert counts[0] == 1 and counts[-1] > 2
    assert (
        f"Total images: 14 ({len(training_numbers)} for training, "
        f"{len(validation_numbers)} for validation)" in lines
    )
    assert (
        f"Minutiae: from the sample table, for all {len(training_numbers)} training images"
        in lines
    )
    assert (
        f"Validation set: {len(validation_numbers)} images of 2 subjects ({held_counts[0]} to "
        f"{held_counts[1]} per subject), compared after every epoch in {mated} mated and 2 "
        "non-mated pairs." in lines
    )
    assert f"Created {len(training_numbers)} training identifiers" in lines
    assert f"Created {len(validation_numbers)} validation identifiers" in lines
    assert "Training DeepPrint with Learned Alignment (LocTexMinu (texture + minutiae))" in lines
    assert not any("WARNING" in line or "Note:" in line for line in lines)

    # Another seed holds out other subjects and draws other pairs: the seed
    # reaches both
    other = run_table(
        train_500k, monkeypatch, tmp_path, "--validation_split", 0.34, "--validation_seed", 7,
        output="another seed",
    )
    (fit,) = other.fits
    _, validation_7 = split_by_the_rule(per_subject, 0.34, 7, 5000)
    assert validation_7 != validation
    assert sorted({i.subject for i in fit["validation_fingerprints"].ids}) == validation_7
    held = {subject: impressions[subject] for subject in validation_7}
    assert pairs(fit["validation_benchmark"]) == pairs_by_the_rule(held, 10, 7)
    assert pairs(fit["validation_benchmark"]) != pairs_by_the_rule(held, 10, 42)
    assert "Validation: 2 subjects held out (seed 7)" in printed(capsys)


def test_rows_of_other_image_formats_are_loaded_as_their_rows_say(
    train_500k, builder, tmp_path, monkeypatch
):
    """Directory mode reads .png files only; a table can name a TIFF, a JPEG or a BMP file."""
    names = ["1_1.tif", "1_2.jpg", "2_1.bmp", "2_2.png"]
    write_small_prints(tmp_path / "images", names, ppi=1000, size=(96, 128))
    assert build(builder, tmp_path) == 0
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0)
    (fit,) = fake.fits
    assert numbers(fit["fingerprints"].ids) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    files = fit["fingerprints"].data_loader._images
    normalizer = SensorNormalizer()  # the resolution is in the metadata of each file
    for name, identifier in zip(names, fit["fingerprints"].ids):
        expected, info = normalizer.load(str(tmp_path / "images" / name))
        assert (info.ppi, info.ppi_origin) == (1000.0, "metadata"), name
        image = files.get(identifier)
        # Halved to 500 ppi, as the row says
        assert image.shape == (1, 64, 48)
        assert torch.equal(image, torch.from_numpy(expected)[None] / 255), name
    # Control: the four files are four different images
    images = [files.get(identifier) for identifier in fit["fingerprints"].ids]
    assert not any(torch.equal(images[0], other) for other in images[1:])


# ---------------------------------------------------------------------------
# Starts that are refused
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_set(tmp_path_factory, builder):
    """
    GRID with its .mnt files and its table, in a directory that no test
    changes: the tests that take it give an --output_dir of their own.
    """
    root = tmp_path_factory.mktemp("grid")
    write_grid(root, minutiae=True)
    assert build(builder, root, "--minutiae_dir", root / "minutiae") == 0
    return root


DEFAULT = "the value of the default of the option"


@pytest.mark.parametrize(
    "option, value, says",
    [
        ("--minutiae_dir", "minutiae", "the table holds the minutiae; --ignore_minutiae trains"),
        ("--num_subjects", 4, "the table lists the images with their subject and impression"),
        ("--num_subjects", DEFAULT, "the table lists the images with their subject and"),
        ("--num_impressions", 2, "the table lists the images with their subject and impression"),
        ("--num_impressions", DEFAULT, "the table lists the images with their subject and"),
        ("--ppi", 500, "the table states the resolution of every file"),
        ("--ppi_manifest", "resolutions.csv", "the table states the resolution and the ridge"),
        ("--assume_ppi", 500, "the table states the resolution of every file"),
        ("--ridges", "light", "the table states the ridge polarity of every file"),
        ("--ridges", DEFAULT, "the table states the ridge polarity of every file"),
    ],
)
def test_an_option_that_the_table_replaces_is_refused_also_with_the_value_of_its_default(
    train_500k, grid_set, tmp_path, monkeypatch, capsys, option, value, says
):
    if value == DEFAULT:
        value = {
            "--num_subjects": train_500k.NUM_SUBJECTS,
            "--num_impressions": train_500k.NUM_IMPRESSIONS,
            "--ridges": "dark",
        }[option]
        assert (train_500k.NUM_SUBJECTS, train_500k.NUM_IMPRESSIONS) == (50000, 10)
    elif option in ("--minutiae_dir", "--ppi_manifest"):
        value = grid_set / value
    message = refused(
        run_table, train_500k, monkeypatch, capsys, grid_set, "--validation_split", 0,
        option, value, output=tmp_path / "model",
    )
    assert f"error: --samples cannot be combined with {option} ({says}" in message
    assert "Leave this option out." in message
    assert "build the table again (scripts/build_sample_table.py" in message
    assert "docs/sample-table.md" in message
    assert not exists(tmp_path / "model")


def test_all_options_that_the_table_replaces_are_named_and_without_them_the_run_trains(
    train_500k, grid_set, tmp_path, monkeypatch, capsys
):
    message = refused(
        run_table, train_500k, monkeypatch, capsys, grid_set, "--validation_split", 0,
        "--ridges", "auto", "--num_subjects", 4, "--assume_ppi", 500,
        output=tmp_path / "model",
    )
    for option in ("--num_subjects (the table", "--assume_ppi (the table", "--ridges (the table"):
        assert option in message
    assert "Leave these options out." in message
    assert "--ppi (" not in message and "--minutiae_dir (" not in message
    # Control: the same command without them
    fake = run_table(
        train_500k, monkeypatch, grid_set, "--validation_split", 0, output=tmp_path / "model"
    )
    assert len(fake.fits) == 1 and len(fake.fits[0]["fingerprints"]) == 8
    # The table replaced them: none of them is something that table mode read
    assert sorted(train_500k.OPTIONS_REPLACED_BY_SAMPLES) == sorted(
        ["minutiae_dir", "num_subjects", "num_impressions", "ppi", "ppi_manifest"]
        + ["assume_ppi", "ridges"]
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--allow_missing_minutiae"],
        ["--ignore_minutiae"],
        ["--ignore_minutiae", "--allow_missing_minutiae"],
    ],
)
def test_the_options_of_a_table_are_refused_without_a_table(
    train_500k, grid_set, tmp_path, monkeypatch, capsys, options
):
    message = refused(
        run_directory, train_500k, monkeypatch, capsys, grid_set, *GRID_OPTIONS, *options,
        output=tmp_path / "model",
    )
    for option in options:
        assert option in message
    assert "can only be given together with --samples" in message
    assert "Give --samples as well (see docs/sample-table.md), or leave" in message
    assert not exists(tmp_path / "model")
    # Control: the same command without them
    fake = run_directory(
        train_500k, monkeypatch, grid_set, *GRID_OPTIONS, output=tmp_path / "model"
    )
    assert len(fake.fits) == 1


def test_the_help_states_the_defaults_of_the_new_options_and_of_the_old_ones(
    train_500k, monkeypatch, capsys
):
    """The three defaults that are withheld while the options are parsed are still shown."""
    monkeypatch.setattr(sys, "argv", ["train_500k.py", "--help"])
    with pytest.raises(SystemExit) as stop:
        train_500k.main()
    assert stop.value.code == 0
    lines = capsys.readouterr().out.splitlines()
    help_text = " ".join(" ".join(lines).split())
    # The entry of every option: from the line that begins with it to the next one
    entries, option = {}, None
    for line in lines:
        if line.startswith("  --"):
            option = line.split()[0]
        if option is not None:
            entries[option] = entries.get(option, "") + " " + " ".join(line.split())
    for option in ("--samples", "--allow_missing_minutiae", "--ignore_minutiae"):
        assert "(default: " in entries[option], option
    assert "--samples" in entries["--allow_missing_minutiae"]
    assert "--samples" in entries["--ignore_minutiae"]
    assert "Number of subjects (default: 50000)" in help_text
    assert "Impressions per subject (default: 10)" in help_text
    assert "dark ridges on a light background (default)" in help_text
    assert "docs/sample-table.md" in help_text


def test_directory_mode_gets_the_defaults_that_are_withheld_while_the_options_are_parsed(
    train_500k, tmp_path, monkeypatch, capsys
):
    """
    --num_subjects, --num_impressions and --ridges have defaults, and a table
    replaces all three. Their defaults are withheld while the options are
    parsed, so that table mode can tell whether one was given, and without a
    table they are put back before anything reads them.
    """
    (tmp_path / "images").mkdir()
    # No file is there, so the run stops when it has printed what it expects
    message = refused(run_directory, train_500k, monkeypatch, capsys, tmp_path)
    assert "500,000 of the 500,000 files that are needed are missing" in message
    assert "--num_subjects 50000 and --num_impressions 10 mean that every file" in message
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_directory(train_500k, monkeypatch, tmp_path)
    lines = printed(capsys)
    assert "Subjects: 50,000 total" in lines and "Impressions: 10" in lines
    assert "Total images: 500,000" in lines
    assert any(line.startswith("Images: ") and line.endswith("; ridges dark") for line in lines)

    # With one of them given, the others keep their defaults
    write_small_prints(tmp_path / "images", grid(3, 10), ppi=None)
    fake = run_directory(
        train_500k, monkeypatch, tmp_path, "--num_subjects", 3, "--validation_split", 0,
        "--assume_ppi", 500,
    )
    assert len(fake.fits[0]["fingerprints"]) == 30
    stored = stored_settings(tmp_path)
    assert (stored["num_subjects"], stored["num_impressions"]) == (3, 10)
    assert stored["preparation"]["ridges"] == "dark"


def test_the_options_of_the_two_modes_are_told_apart_for_any_parser_with_these_options(
    train_500k,
):
    """
    withhold_defaults and check_options_of_the_mode serve every script that
    takes --samples beside the options of the file layout.
    """
    import argparse

    def parse(*arguments) -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="other.py")
        parser.add_argument("--samples", default=None)
        parser.add_argument("--num_subjects", type=int, default=7)
        parser.add_argument("--minutiae_dir", default=None)
        parser.add_argument("--ridges", default="auto")
        parser.add_argument("--allow_missing_minutiae", action="store_true")
        parser.add_argument("--max_subjects", type=int, default=2000)
        withheld = train_500k.withhold_defaults(parser)
        # Only the options that a table replaces, and of them those with a default
        assert withheld == {"num_subjects": 7, "ridges": "auto"}
        args = parser.parse_args([str(argument) for argument in arguments])
        train_500k.check_options_of_the_mode(parser, args, withheld)
        return args

    args = parse()
    assert (args.num_subjects, args.ridges, args.minutiae_dir, args.max_subjects) == (
        7, "auto", None, 2000,
    )
    args = parse("--num_subjects", 3, "--ridges", "auto", "--minutiae_dir", "m")
    assert (args.num_subjects, args.ridges, args.minutiae_dir) == (3, "auto", "m")
    # With a table the replaced options stay None, and the others keep their values
    args = parse("--samples", "table", "--allow_missing_minutiae", "--max_subjects", 5)
    assert (args.num_subjects, args.ridges, args.minutiae_dir) == (None, None, None)
    assert args.allow_missing_minutiae and args.max_subjects == 5
    for arguments in (
        ["--samples", "table", "--num_subjects", 7],
        ["--samples", "table", "--ridges", "auto"],
        ["--samples", "table", "--minutiae_dir", "m"],
        ["--allow_missing_minutiae"],
    ):
        with pytest.raises(SystemExit) as stop:
            parse(*arguments)
        assert stop.value.code == 2, arguments


def test_missing_image_files_are_named_with_their_number(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    write_uneven(builder, tmp_path)
    (tmp_path / "images" / "roll" / "only one.png").unlink()
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, "--validation_split", 0
    )
    assert message.startswith(
        f"ERROR: 1 of the 14 image files that the sample table names are missing below "
        f"{tmp_path / 'images'} (--dataset_dir). The first ones: 'roll/only one.png' "
        "(subject 40, impression 2). "
    )
    assert "Give the directory that the table was built from as --dataset_dir." in message
    assert "build the table again from the files as they are now" in message
    assert "scripts/build_sample_table.py" in message

    # A whole directory that is gone, and more files than a message names
    shutil.rmtree(tmp_path / "images" / "plain")
    for name in ("a.png", "b.png", "c.png", "d.png"):
        (tmp_path / "images" / name).unlink()
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, "--validation_split", 0
    )
    assert "8 of the 14 image files that the sample table names are missing" in message
    # In the order of the rows: by subject and impression
    assert (
        "The first ones: 'a.png' (subject 2, impression 0); 'b.png' (subject 2, impression 1); "
        "'c.png' (subject 2, impression 5); 'plain/00002239_plain_05.png' (subject 7, "
        "impression 0); 'd.png' (subject 11, impression 0); and 3 more. " in message
    )

    # Another directory than the one of the table: every file is missing
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "images").mkdir(parents=True)
    message = refused(
        run_table, train_500k, monkeypatch, capsys, elsewhere, "--validation_split", 0,
        table=tmp_path / "table",
    )
    assert "14 of the 14 image files" in message and f"below {elsewhere / 'images'}" in message
    assert not exists(elsewhere / "model") and not exists(tmp_path / "model")


def test_every_directory_of_the_table_is_listed_once_and_no_file_is_asked_for(
    train_500k, builder, tmp_path, monkeypatch
):
    table = write_uneven(builder, tmp_path)
    listed, asked = [], []
    listdir, exists_ = os.listdir, os.path.exists
    monkeypatch.setattr(os, "listdir", lambda path: listed.append(path) or listdir(path))
    monkeypatch.setattr(os.path, "exists", lambda path: asked.append(path) or exists_(path))
    assert train_500k.find_missing_table_files(table).tolist() == []
    images = str(tmp_path / "images")
    assert sorted(os.path.normpath(path) for path in listed) == [
        images, join(images, "plain"), join(images, "roll"),
    ]
    assert asked == []

    # A file that is gone is found without being asked for
    os.remove(join(images, "h3.png"))
    listed.clear()
    missing = train_500k.find_missing_table_files(table)
    assert [table.describe_row(row) for row in missing] == ["'h3.png' (subject 1000, impression 2)"]
    assert missing.dtype == np.int64 and len(listed) == 3 and asked == []

    # A directory that cannot be listed can still hold its files: they are
    # asked for one by one, and only they
    def unlistable(path):
        if os.path.normpath(path) == join(images, "roll"):
            raise PermissionError(13, "Permission denied", path)
        return listdir(path)

    monkeypatch.setattr(os, "listdir", unlistable)
    assert train_500k.find_missing_table_files(table).tolist() == missing.tolist()
    assert sorted(asked) == [
        join(images, "roll", "00002239_roll_05.png"), join(images, "roll", "only one.png"),
    ]
    os.remove(join(images, "roll", "only one.png"))
    missing = train_500k.find_missing_table_files(table)
    assert [table.samples.image_path(row) for row in missing] == ["roll/only one.png", "h3.png"]


def test_the_checks_of_the_files_can_be_made_for_some_rows_of_a_table(
    train_500k, builder, tmp_path, monkeypatch
):
    """As a run needs them that uses some of the subjects of a table only."""
    from flx.preparation.sample_table import SampleTableError

    table = write_uneven(builder, tmp_path)
    images = str(tmp_path / "images")
    os.remove(join(images, "h3.png"))
    os.remove(join(images, "plain", "f.png"))
    listed = []
    listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: listed.append(path) or listdir(path))

    # The rows of two subjects whose files are in one directory: only that
    # directory is listed, and the missing file of another subject is not seen
    rows = table.rows_of([1000, 2])
    assert train_500k.find_missing_table_files(table, rows).tolist() == [rows[2]]
    assert [os.path.normpath(path) for path in listed] == [images]
    with pytest.raises(SampleTableError) as error:
        train_500k.check_table_files(table, rows)
    assert str(error.value).startswith(
        f"1 of the 7 image files that are needed of the sample table are missing below {images} "
        "(--dataset_dir). The first ones: 'h3.png' (subject 1000, impression 2). "
    )
    train_500k.check_table_files(table, table.rows_of([7, 11, 40]))
    with pytest.raises(SampleTableError, match="2 of the 14 image files that the sample table"):
        train_500k.check_table_files(table)
    # In the order in which the rows are given
    both = train_500k.find_missing_table_files(table, table.rows_of([1000, 300]))
    assert [table.samples.image_path(row) for row in both] == ["h3.png", "plain/f.png"]

    # The images that are loaded at the start, spread over the given rows
    loader = SampleTableImageLoader(table)
    asked = []
    get = SampleTableImageLoader.get
    monkeypatch.setattr(
        SampleTableImageLoader,
        "get",
        lambda self, identifier: asked.append(table.row(identifier)) or get(self, identifier),
    )
    rows = table.rows_of([2, 7, 11])
    assert train_500k.check_table_images(loader, 3, rows) == 3
    assert asked == [rows[0], rows[3], rows[6]]
    asked.clear()
    assert train_500k.check_table_images(loader, 100, rows) == 7 and asked == rows.tolist()
    with pytest.raises(SampleTableError, match="The image file .*h3.png is missing"):
        train_500k.check_table_images(loader, 100, table.rows_of([1000]))


def test_a_file_that_changed_after_the_table_was_built_stops_the_run_before_the_model_is_built(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path)
    assert build(builder, tmp_path) == 0
    split = ["--validation_split", 0]
    # The rows in their order, and one file written again as another image
    assert [name for name in GRID] == [f"{s}_{i}.png" for s in (1, 2, 3, 4) for i in (1, 2)]
    changed = tmp_path / "images" / "3_1.png"
    save(changed, ridge_image(80, 60), 500)

    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split)
    # The message is that of the loader of the table
    assert message.startswith(
        f"ERROR: {changed} is not the file that its row of the sample table describes: "
    )
    assert "'3_1.png' (subject 2, impression 0)" in message
    assert f"for the image directory {tmp_path / 'images'}" in message
    assert "build the table again from the files as they are now" in message
    assert "scripts/build_sample_table.py" in message

    # The files that are loaded at the start are spread over the rows. Of
    # three among these eight rows the changed file is none: rows 0, 3 and 7.
    monkeypatch.setattr(train_500k, "IMAGES_LOADED_AT_THE_START", 3)
    fake = run_table(train_500k, monkeypatch, tmp_path, *split, output="started")
    assert len(fake.fits) == 1
    assert "Loaded 3 image files spread over the table" in capsys.readouterr().out
    # The file at row 3 is one of them
    save(tmp_path / "images" / "2_2.png", ridge_image(80, 60), 500)
    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split)
    assert message.startswith(f"ERROR: {tmp_path / 'images' / '2_2.png'} is not the file")

    # A file that is no image any more, with as many bytes as it had
    for name in ("2_2.png", "3_1.png"):
        os.remove(tmp_path / "images" / name)
    write_small_prints(tmp_path / "images", GRID)
    last = tmp_path / "images" / "4_2.png"
    last.write_bytes(b"\0" * os.path.getsize(last))
    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split)
    assert message.startswith(f"ERROR: {last}: Could not read the file as an image: {last}. ")
    assert "the row '4_2.png' (subject 3, impression 1) of the sample table" in message
    assert "scripts/build_sample_table.py" in message
    # Control: with the files as they were when the table was built, the run starts
    write_small_prints(tmp_path / "images", GRID)
    assert len(run_table(train_500k, monkeypatch, tmp_path, *split).fits) == 1


def test_the_images_that_are_loaded_at_the_start_are_spread_over_the_table(train_500k):
    class Loader:
        """Keeps which rows it is asked for."""

        def __init__(self, rows):
            self.table = table_of((subject, 0) for subject in range(rows))
            self.asked = []

        def get(self, identifier):
            self.asked.append(self.table.row(identifier))

    loader = Loader(1000)
    assert train_500k.check_table_images(loader) == train_500k.IMAGES_LOADED_AT_THE_START == 100
    # From the first row to the last, at equal distances
    assert loader.asked[0] == 0 and loader.asked[-1] == 999 and len(loader.asked) == 100
    assert {b - a for a, b in zip(loader.asked, loader.asked[1:])} <= {10, 11}
    for rows, max_images, expected in ((8, 3, [0, 3, 7]), (5, 100, [0, 1, 2, 3, 4]), (1, 7, [0])):
        loader = Loader(rows)
        assert train_500k.check_table_images(loader, max_images) == len(expected)
        assert loader.asked == expected


def test_a_table_that_cannot_be_read_stops_the_run_with_the_reason(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    import pyarrow.parquet as pq

    write_grid(tmp_path)
    monkeypatch.setattr(builder, "FILES_PER_PART", 3)
    assert build(builder, tmp_path) == 0
    table_dir = tmp_path / "table"
    parts = sorted(os.listdir(table_dir))
    assert parts == [f"part-0000{index}-of-00003.parquet" for index in range(3)]
    split = ["--validation_split", 0]

    def message() -> str:
        return refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split)

    # A part is missing
    os.rename(table_dir / parts[1], tmp_path / parts[1])
    assert message().startswith(
        f"ERROR: {table_dir} is not a whole table: it holds 2 of its 3 parts; missing: {parts[1]}."
    )
    os.rename(tmp_path / parts[1], table_dir / parts[1])

    # A part of a schema version that this code does not know
    kept = (table_dir / parts[2]).read_bytes()
    part = pq.read_table(table_dir / parts[2])
    metadata = dict(part.schema.metadata)
    metadata[b"flx.schema_version"] = b"2"
    pq.write_table(part.replace_schema_metadata(metadata), table_dir / parts[2])
    found = message()
    assert found.startswith(f"ERROR: {table_dir / parts[2]} has the schema version '2'")
    assert "this code reads version 1" in found
    (table_dir / parts[2]).write_bytes(kept)

    # A part that is no Parquet file any more
    (table_dir / parts[0]).write_bytes(b"PAR1 no table")
    assert f"{table_dir / parts[0]} cannot be read as a Parquet file" in message()

    # A directory without a table, and one that is not there
    for name in ("empty", "no such directory"):
        if name == "empty":
            (tmp_path / name).mkdir()
        found = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split, table=name)
        assert found.startswith(f"ERROR: {tmp_path / name} ")
        assert "scripts/build_sample_table.py" in found

    # A table without rows
    write_part(str(tmp_path / "no rows"), 0, 1, samples_from_rows([]), {"id": "no rows"})
    found = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split, table="no rows")
    assert found.startswith(f"ERROR: The sample table {tmp_path / 'no rows'} holds no row")
    assert "scripts/build_sample_table.py" in found


def test_rows_that_cannot_be_told_apart_by_their_identifiers_stop_the_run_with_the_reason(
    train_500k, tmp_path, monkeypatch, capsys
):
    """
    An Identifier is compared by the hash of its two numbers, and two pairs
    of numbers can have the same hash. A table with two such rows is refused
    with both rows named, not with a traceback of the set of identifiers.
    """
    first, second = (391292100, 1755844984), (1473744498, 0)
    if hash(first) != hash(second):
        pytest.skip("These two pairs of numbers have different hashes with this Python")
    rows = [
        SampleRow(
            subject=subject, impression=impression, image_path=name, file_size=100, ppi=500.0,
            ppi_origin="metadata", ridges="dark", ridges_origin="argument", source_width=48,
            source_height=64, width=48, height=64, minutiae=None,
        )
        for (subject, impression), name in ((first, "a.png"), (second, "b.png"), ((7, 0), "c.png"))
    ]
    write_part(str(tmp_path / "table"), 0, 1, samples_from_rows(rows), {"id": "equal"})
    (tmp_path / "images").mkdir()
    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, "--validation_split", 0)
    assert message.startswith(
        "ERROR: 1 pair(s) of rows of the sample table cannot be told apart by their numbers: "
        "'a.png' (subject 391292100, impression 1755844984) and "
        "'b.png' (subject 1473744498, impression 0). "
    )
    assert "build the table again (scripts/build_sample_table.py)" in message


def test_columns_that_the_code_does_not_know_are_named_and_not_used(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    write_grid(tmp_path)
    assert build(builder, tmp_path) == 0
    run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0)
    assert "does not know" not in capsys.readouterr().out
    # Another tool adds two columns to the part
    (name,) = os.listdir(tmp_path / "table")
    part = pq.read_table(tmp_path / "table" / name)
    part = part.append_column("person", pa.array(list(range(8)), type=pa.int32()))
    part = part.append_column("finger", pa.array(["index"] * 8))
    pq.write_table(part, tmp_path / "table" / name)

    # The rows are the same rows, so the training continues with them
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0, "--num_epochs", 3)
    assert len(fake.fits) == 1
    lines = printed(capsys)
    assert (
        "Note: the sample table holds 2 column(s) that this code does not know and does not "
        "use: person, finger." in lines
    )
    assert "Training complete! 1 epoch(s) trained in this run, 3 in the log." in lines


# ---------------------------------------------------------------------------
# Minutiae: which variant is trained, and rows without minutiae
# ---------------------------------------------------------------------------


def test_training_images_without_minutiae_stop_the_run_unless_they_are_allowed(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path, minutiae=True)
    without = [(1, 0), (3, 1)]
    for name in ("2_1.mnt", "4_2.mnt"):
        (tmp_path / "minutiae" / name).unlink()
    # The builder gives the two images rows with null minutiae, and says so
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    assert numbers(table.identifiers[row] for row in table.rows_without_minutiae()) == without
    split = ["--validation_split", 0]

    message = refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *split)
    assert message.startswith(
        "ERROR: The sample table holds no minutiae (null) for 2 of the 8 training images. "
        "The first ones: '2_1.png' (subject 1, impression 0); '4_2.png' (subject 3, "
        "impression 1). "
    )
    assert "build_failures.txt" in message
    assert "build the table again (scripts/build_sample_table.py)" in message
    assert "or pass --allow_missing_minutiae: these images then train without a minutia map" in (
        message
    )
    assert "--ignore_minutiae trains the texture-only variant instead." in message

    # With the option the run trains, and exactly these two images have the
    # weight 0 and a map of zeros
    monkeypatch.setattr(train_500k, "LazilyAllocatedBinarizer", lambda _: identity)
    monkeypatch.setattr(train_500k, "POSE_AUGMENTATION", None)
    capsys.readouterr()
    fake = run_table(train_500k, monkeypatch, tmp_path, *split, "--allow_missing_minutiae")
    lines = printed(capsys)
    (fit,) = fake.fits
    assert [variant for variant, _ in fake.built] == ["LocTexMinu"]
    assert len(fit["minutia_maps"]) == 8
    for identifier in fit["minutia_maps"].ids:
        fit["fingerprints"].get(identifier)
        minutia_map, weight = fit["minutia_maps"].get(identifier)
        known = (identifier.subject, identifier.impression) not in without
        assert weight == (1.0 if known else 0.0), str(identifier)
        assert minutia_map.shape == (6, 128, 128) and bool(minutia_map.any()) == known
    assert (
        "Minutiae: from the sample table, for 6 of the 8 training images; it holds none "
        "(null) for the other 2" in lines
    )
    assert (
        "Note: training images without minutiae in the sample table: 2 of 8. They train "
        "without a minutia map (--allow_missing_minutiae), with weight 0 in the minutia map "
        "loss, and count in the other parts of the loss as every image does." in lines
    )
    # The start says what the build of the table recorded about these files
    assert (
        "Table build: 2 file(s) failed when the table was built: 0 without a row (they are "
        "not in the table), 2 with a row without minutiae; build_failures.txt in the "
        "directory of the table names them" in lines
    )
    assert stored_settings(tmp_path)["variant"] == "LocTexMinu"

    # Validation needs no minutiae: where only held-out subjects lack them,
    # nothing has to be allowed. Subjects 1 and 2 are held out of these four.
    assert train_500k.split_subjects(4, 0.5)[1] == [1, 2]
    for name in ("4_1.mnt", "4_2.mnt"):
        write_small_minutiae(tmp_path / "minutiae", [name])
    (tmp_path / "minutiae" / "3_2.mnt").unlink()
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae", table="held") == 1
    held = SampleTable.read(str(tmp_path / "held"), str(tmp_path / "images"))
    assert numbers(held.identifiers[row] for row in held.rows_without_minutiae()) == [
        (1, 0), (2, 1),
    ]
    capsys.readouterr()
    fake = run_table(
        train_500k, monkeypatch, tmp_path, "--validation_split", 0.5, table="held", output="held"
    )
    assert numbers(fake.fits[0]["minutia_maps"].ids) == [(0, 0), (0, 1), (3, 0), (3, 1)]
    lines = printed(capsys)
    assert "Minutiae: from the sample table, for all 4 training images" in lines
    assert not any("--allow_missing_minutiae" in line for line in lines)
    # Control: when these subjects train, their images stop the run
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, *split, table="held",
        output="all train",
    )
    assert "for 2 of the 8 training images" in message and "'2_1.png'" in message


def test_the_start_says_how_many_files_failed_when_the_table_was_built(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path, minutiae=True)
    # One file that is no image, which gets no row, and one image without .mnt file
    (tmp_path / "images" / "5_1.png").write_bytes(b"no image")
    (tmp_path / "minutiae" / "1_2.mnt").unlink()
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    capsys.readouterr()
    fake = run_table(
        train_500k, monkeypatch, tmp_path, "--validation_split", 0, "--allow_missing_minutiae"
    )
    assert len(fake.fits[0]["fingerprints"]) == 8
    lines = printed(capsys)
    assert f"Samples: {tmp_path / 'table'} (8 rows, 4 subjects)" in lines
    assert (
        "Table build: 2 file(s) failed when the table was built: 1 without a row (they are "
        "not in the table), 1 with a row without minutiae; build_failures.txt in the "
        "directory of the table names them" in lines
    )
    # A build without failures is not mentioned
    assert not any(line.startswith("Table build:") for line in _table_build_lines(
        train_500k, builder, tmp_path, monkeypatch, capsys
    ))
    # What another builder recorded in another form is said as far as it is known
    table = table_of([(0, 0), (0, 1)])
    for build_info, expected in (
        ({"failed": ["a.png", "b.png", "c.png"]}, "3 file(s) failed when the table was built;"),
        ({"failed": 2, "files": 3}, "2 file(s) failed when the table was built: 1 without"),
        ({"failed": 2, "files": 9}, "2 file(s) failed when the table was built;"),
        ({"failed": 0, "files": 2}, None),
        ({"failed": True}, None),
        ({}, None),
    ):
        other = SampleTable(dataclasses.replace(table.samples, build=build_info), "images")
        text = train_500k._files_that_failed_in_the_build(other)
        assert (text is None) if expected is None else text.startswith(expected), build_info


def _table_build_lines(train_500k, builder, tmp_path, monkeypatch, capsys) -> list:
    """What the start prints for a table of the grid alone, built without a failure."""
    (tmp_path / "clean").mkdir()
    write_grid(tmp_path / "clean")
    assert build(builder, tmp_path / "clean") == 0
    capsys.readouterr()
    run_table(train_500k, monkeypatch, tmp_path / "clean", "--validation_split", 0)
    return printed(capsys)


def test_an_image_without_minutiae_is_not_an_image_whose_minutiae_are_unknown(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    """
    A .mnt file that holds no minutiae gives a row with an empty list: the
    image has none, and that is known. Such rows need no permission, and a
    table of nothing else still trains the minutia branch.
    """
    write_grid(tmp_path)
    for name in GRID:
        path = tmp_path / "minutiae" / (name[:-4] + ".mnt")
        os.makedirs(tmp_path / "minutiae", exist_ok=True)
        path.write_text("0\n48 64\n")
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    assert table.samples.minutiae_known.all() and len(table.samples.minutiae) == 0
    monkeypatch.setattr(train_500k, "LazilyAllocatedBinarizer", lambda _: identity)
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0)
    assert [variant for variant, _ in fake.built] == ["LocTexMinu"]
    assert "Minutiae: from the sample table, for all 8 training images" in printed(capsys)
    (fit,) = fake.fits
    for identifier in fit["minutia_maps"].ids:
        fit["fingerprints"].get(identifier)
        assert fit["minutia_maps"].get(identifier)[1] == 0.0


def test_the_texture_only_variant_is_trained_on_request_and_from_a_table_without_minutiae(
    train_500k, builder, grid_set, tmp_path, monkeypatch, capsys
):
    split = ["--validation_split", 0]

    def texture_only(root, *options, output) -> list:
        """A run that must train the texture-only variant. Returns what it printed."""
        capsys.readouterr()
        fake = run_table(train_500k, monkeypatch, root, *split, *options, output=output)
        assert fake.built == [
            ("LocTex", {"num_training_subjects": 4, "num_texture_dims": 256,
                        "loss_weights": fake.built[0][1]["loss_weights"]})
        ]
        (fit,) = fake.fits
        # The placeholder of the minutia maps, as without --minutiae_dir
        assert len(fit["minutia_maps"]) == 8
        minutia_map, weight = fit["minutia_maps"].get(Identifier(0, 0))
        assert minutia_map.numel() == 0 and weight == 0.0
        stored = stored_settings(root, output)
        assert stored["variant"] == "LocTex" and stored["loss_weights"]["minutia_map"] is None
        lines = printed(capsys)
        assert "Training DeepPrint with Learned Alignment (LocTex (texture-only))" in lines
        assert "Embedding: 256 values per branch, 256 per print" in lines
        assert "No minutiae are used - texture-only mode" in lines
        assert not any("minutia map" in line for line in lines if line.startswith("Loss weights"))
        # What directory mode prints here is about an option that a table replaces
        assert not any("No minutiae directory provided" in line for line in lines)
        return lines

    # The table holds minutiae, and --ignore_minutiae leaves them out
    control = run_table(train_500k, monkeypatch, grid_set, *split, output=tmp_path / "with")
    assert [variant for variant, _ in control.built] == ["LocTexMinu"]
    assert stored_settings(grid_set, tmp_path / "with")["loss_weights"]["minutia_map"] is not None
    lines = texture_only(grid_set, "--ignore_minutiae", output=tmp_path / "ignored")
    assert "Minutiae: not used (--ignore_minutiae)" in lines
    # With --localization off the other texture-only variant
    capsys.readouterr()
    fake = run_table(
        train_500k, monkeypatch, grid_set, *split, "--ignore_minutiae", "--localization", "off",
        output=tmp_path / "off",
    )
    assert [variant for variant, _ in fake.built] == ["Tex"]

    # A table that was built without minutiae: the start says that it holds none
    write_grid(tmp_path)
    assert build(builder, tmp_path) == 0
    expected = "Minutiae: none: the sample table holds no minutiae for the training images"
    assert expected in texture_only(tmp_path, output="none")
    # Nothing is missing in such a training, with whatever option
    assert expected in texture_only(tmp_path, "--allow_missing_minutiae", output="allowed")
    assert expected in texture_only(tmp_path, "--ignore_minutiae", output="nothing to ignore")


# ---------------------------------------------------------------------------
# What the start prints
# ---------------------------------------------------------------------------


def described_resolutions(train_500k, table: SampleTable, rows: np.ndarray) -> list:
    """The lines of describe_table_resolutions for the rows, each with single spaces."""
    return [" ".join(line.split()) for line in train_500k.describe_table_resolutions(table, rows)]


def test_the_resolution_of_the_training_images_is_counted_from_the_table(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    """
    The block that directory mode prints from a sample of the files is an
    exact count of the training rows here: by source and resolution, and by
    ridge polarity. The rows of the held-out subjects are not counted.
    """
    images = tmp_path / "images"
    # Three sources, four files of two subjects from each: files that state
    # 500 ppi themselves; files of a sensor with 1000 ppi and light ridges,
    # which a manifest says; files that state nothing, for which the build
    # was given an assumption
    write_small_prints(images, grid(2, 2, "sensor_a"), ppi=500)
    of_the_manifest = [f"sensor_b/{subject}_{i}.png" for subject in (3, 4) for i in (1, 2)]
    write_small_prints(images, of_the_manifest, ppi=None, size=(96, 128))
    write_small_prints(images, [f"{subject}_{i}.png" for subject in (5, 6) for i in (1, 2)], None)
    (tmp_path / "resolutions.csv").write_text("path,ppi,ridges\nsensor_b,1000,light\n")
    status = build(
        builder, tmp_path, "--ppi_manifest", tmp_path / "resolutions.csv", "--assume_ppi", 500
    )
    assert status == 0
    capsys.readouterr()
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0.34)
    lines = printed(capsys)

    # Counted here from the rows of the table, for the subjects that train
    samples = read_samples(str(tmp_path / "table"))
    training_subjects = {i.subject for i in fake.fits[0]["fingerprints"].ids}
    assert training_subjects == {0, 2, 4, 5}
    counts, of_all_rows = {}, {}
    for row in range(len(samples)):
        key = (PPI_ORIGINS[samples.ppi_origin[row]], float(samples.ppi[row]))
        of_all_rows[key] = of_all_rows.get(key, 0) + 1
        if samples.subject[row] in training_subjects:
            counts[key] = counts.get(key, 0) + 1
    assert counts == {("metadata", 500.0): 2, ("manifest", 1000.0): 2, ("assumed", 500.0): 4}
    assert set(of_all_rows.values()) == {4}

    first = lines.index("Resolution of the 8 training images, as the sample table states it:")
    assert lines[first + 1 : first + 4] == [
        "4 500 ppi from --assume_ppi",
        "2 1000 ppi from manifest (resampled to 500 ppi)",
        "2 500 ppi from metadata of the file",
    ]
    # The assumption of the build did not override the metadata of two files
    assert lines[first + 4].startswith(
        "Note: 2 file(s) took the resolution from their own metadata, not from --assume_ppi 500."
    )
    assert lines[first + 5 : first + 9] == [
        "Ridge polarity of these images, as the files hold it:",
        "6 dark ridges, from --ridges",
        "2 light ridges, from the manifest",
        "These are the sources of the build of the table (scripts/build_sample_table.py): to "
        "change one, build the table again.",
    ]
    # The block comes after the files were checked and before the model is built
    assert lines[first - 1].startswith("Loaded 12 image files spread over the table")
    assert lines[first + 9 : first + 11] == ["", "Initializing DeepPrint_LocTex model..."]

    # The function, on all rows and on rows of one source
    all_rows = np.arange(12)
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    described = described_resolutions(train_500k, table, all_rows)
    assert described[:3] == [
        "4 500 ppi from --assume_ppi",
        "4 1000 ppi from manifest (resampled to 500 ppi)",
        "4 500 ppi from metadata of the file",
    ]
    assert "8 dark ridges, from --ridges" in described
    assert "4 light ridges, from the manifest" in described
    of_one_source = table.rows_of([2, 3])
    described = described_resolutions(train_500k, table, of_one_source)
    assert described[:3] == [
        "4 1000 ppi from manifest (resampled to 500 ppi)",
        "Ridge polarity of these images, as the files hold it:",
        "4 light ridges, from the manifest",
    ]


def test_many_resolutions_are_summed_up_after_the_most_frequent_ones(train_500k):
    table = table_of((subject, 0) for subject in range(60))
    samples = table.samples
    # 30 rows at 500 ppi, the others at a resolution of their own each
    ppi = np.where(np.arange(60) < 30, 500.0, 600.0 + np.arange(60))
    sizes = [resampled_size(48, 64, value) for value in ppi.tolist()]
    table = SampleTable(
        dataclasses.replace(
            samples,
            ppi=ppi,
            width=np.array([size[0] for size in sizes], dtype=np.int32),
            height=np.array([size[1] for size in sizes], dtype=np.int32),
        ),
        "images",
    )
    lines = described_resolutions(train_500k, table, np.arange(60))
    assert lines[0] == "30 500 ppi from --ppi"
    assert lines[1] == "1 630 ppi from --ppi (resampled to 500 ppi)"
    assert lines[20] == "11 with 11 further resolutions"
    assert lines[21] == "Ridge polarity of these images, as the files hold it:"
    assert lines[22] == "60 dark ridges, from --ridges"


def test_the_start_says_what_the_validation_set_holds_and_warns_when_it_is_large(
    train_500k, grid_set, tmp_path, monkeypatch, capsys
):
    def start(output: str, *options) -> list:
        """What a run with two held-out subjects prints."""
        capsys.readouterr()
        run_table(
            train_500k, monkeypatch, grid_set, "--validation_split", 0.5, *options,
            output=tmp_path / output,
        )
        return printed(capsys)

    lines = start("as it is")
    assert (
        "Validation set: 4 images of 2 subjects (2 to 2 per subject), compared after every "
        "epoch in 2 mated and 2 non-mated pairs." in lines
    )
    assert not any("WARNING" in line for line in lines)
    assert train_500k.MANY_VALIDATION_IMAGES == 50_000
    assert train_500k.MANY_VALIDATION_PAIRS == 275_000

    # A set of exactly the two limits is not above them
    monkeypatch.setattr(train_500k, "MANY_VALIDATION_IMAGES", 4)
    monkeypatch.setattr(train_500k, "MANY_VALIDATION_PAIRS", 4)
    assert not any("WARNING" in line for line in start("at the limits"))
    # One pair more than the limit, or one image more
    for name in ("MANY_VALIDATION_PAIRS", "MANY_VALIDATION_IMAGES"):
        monkeypatch.setattr(train_500k, name, 3)
        (warning,) = [line for line in start(f"above {name}") if "WARNING" in line]
        monkeypatch.setattr(train_500k, name, 4)
        assert warning.startswith("WARNING: the validation set is larger than ")
        assert "images or " in warning and "a finger with many images can dominate" in warning
        assert "--max_validation_subjects and --validation_split" in warning

    # Without validation there is nothing to say
    capsys.readouterr()
    run_table(train_500k, monkeypatch, grid_set, "--validation_split", 0, output=tmp_path / "none")
    lines = printed(capsys)
    assert not any(line.startswith("Validation set:") for line in lines)
    assert "Validation: 0 subjects held out (seed 42)" in lines
    assert "Patience: not used (no validation set)" in lines
    assert "Best model: lowest training loss (no validation set)" in lines


def test_notes_say_why_fewer_subjects_are_held_out_than_the_share_asks_for(
    train_500k, builder, grid_set, tmp_path, monkeypatch, capsys
):
    # The limit of validation subjects, as without a table
    capsys.readouterr()
    fake = run_table(
        train_500k, monkeypatch, grid_set, "--validation_split", 0.9,
        "--max_validation_subjects", 2, output=tmp_path / "limit",
    )
    assert fake.fits[0]["validation_fingerprints"].num_subjects == 2
    lines = printed(capsys)
    assert (
        "Note: --validation_split 0.9 would hold out 3 subjects; --max_validation_subjects "
        "limits this to 2. The others stay in the training set." in lines
    )
    assert not any("were to be held out" in line for line in lines)

    # Eight subjects, three of them with two images: half of the subjects
    # cannot be held out
    names = ["1_1", "1_2", "2_1", "3_1", "3_2", "4_1", "5_1", "6_1", "6_2", "7_1", "8_1"]
    write_small_prints(tmp_path / "images", [name + ".png" for name in names])
    assert build(builder, tmp_path) == 0
    capsys.readouterr()
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0.5)
    (fit,) = fake.fits
    assert numbers(fit["validation_fingerprints"].ids) == [
        (0, 0), (0, 1), (2, 0), (2, 1), (5, 0), (5, 1),
    ]
    assert numbers(fit["fingerprints"].ids) == [(1, 0), (3, 0), (4, 0), (6, 0), (7, 0)]
    assert [arguments["num_training_subjects"] for _, arguments in fake.built] == [5]
    lines = printed(capsys)
    assert (
        "Note: 4 subjects were to be held out, but only 3 subjects of the table have two or "
        "more impressions, which validation needs: it compares two impressions of a subject. "
        "These 3 are held out, and every subject with one impression trains." in lines
    )
    assert not any("--max_validation_subjects limits" in line for line in lines)
    assert "Impressions: 1 to 1 per training subject (median 1)" in lines
    # Both notes, where the limit cuts the share and the subjects cannot meet the limit
    capsys.readouterr()
    run_table(
        train_500k, monkeypatch, tmp_path, "--validation_split", 0.9,
        "--max_validation_subjects", 5, output="both",
    )
    lines = printed(capsys)
    notes = [line for line in lines if line.startswith("Note:")]
    assert len(notes) == 2
    assert "would hold out 7 subjects; --max_validation_subjects limits this to 5" in notes[0]
    assert notes[1].startswith("Note: 5 subjects were to be held out, but only 3 subjects ")
    # Neither, where the subjects with two images are as many as asked for
    capsys.readouterr()
    run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0.25, output="met")
    assert not any(line.startswith("Note:") for line in printed(capsys))


def test_a_validation_set_that_cannot_be_made_is_an_error_of_the_options(
    train_500k, builder, grid_set, tmp_path, monkeypatch, capsys
):
    # A share that holds out fewer than two subjects: the message of directory mode
    message = refused(
        run_table, train_500k, monkeypatch, capsys, grid_set, "--validation_split", 0.3,
        output=tmp_path / "model",
    )
    assert message.endswith(
        "train_500k.py: error: validation_split 0.3 of 4 subjects holds out 1 subjects, but "
        "validation needs at least 2. Use a larger share, or 0 to train without validation."
    )
    for options in (["--validation_split", 1], ["--max_validation_subjects", -1]):
        refused(
            run_table, train_500k, monkeypatch, capsys, grid_set, *options,
            output=tmp_path / "model",
        )

    # One subject with two images among four: nothing to compare it with
    names = ["1_1", "1_2", "2_1", "3_1", "4_1"]
    write_small_prints(tmp_path / "images", [name + ".png" for name in names])
    assert build(builder, tmp_path) == 0
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, "--validation_split", 0.5
    )
    assert message.endswith(
        "error: Validation compares two impressions of the same subject, so it needs at least "
        "2 subjects with 2 or more impressions, and 1 of the 4 subjects have that many. Use "
        "--validation_split 0 to train without validation."
    )
    # Without validation the same table trains
    fake = run_table(train_500k, monkeypatch, tmp_path, "--validation_split", 0)
    assert len(fake.fits[0]["fingerprints"]) == 5 and fake.fits[0]["validation_benchmark"] is None


def test_the_summary_of_the_start_is_that_of_directory_mode_with_the_table_in_it(
    train_500k, grid_set, tmp_path, monkeypatch, capsys
):
    from flx.models.deep_print_loss import LossWeights

    capsys.readouterr()
    run_table(
        train_500k, monkeypatch, grid_set, "--validation_split", 0.5, "--canvas_size", 512,
        "--batch_size", 16, "--localization", "off", output=tmp_path / "model",
    )
    lines = printed(capsys)
    weights = LossWeights()
    first = lines.index("=" * 60)
    last = lines.index("=" * 60, first + 3)
    # The last lines of the summary are about the machine
    assert lines[last - 1].startswith(("Device: ", "GPUs: ")) and lines[last + 1] == ""
    assert lines[first : first + 22] == [
        "=" * 60,
        "Training DeepPrint without Learned Alignment (TexMinu (texture + minutiae))",
        "=" * 60,
        f"Samples: {grid_set / 'table'} (8 rows, 4 subjects)",
        f"Dataset: {grid_set / 'images'}",
        "Minutiae: from the sample table, for all 4 training images",
        "Images: prepared as the sample table states for every file: resampled from its "
        "resolution to 500 ppi, dark ridges",
        "Canvas: 512x512 pixels at 500 ppi for every image",
        "Batch size: 16 images per step and GPU",
        "Subjects: 4 total",
        "Training: 2 subjects",
        "Validation: 2 subjects held out (seed 42)",
        "Impressions: 2 to 2 per training subject (median 2)",
        "Total images: 8 (4 for training, 4 for validation)",
        "Epochs: 2 (max)",
        "Patience: 3 epochs (early stopping)",
        "Best model: lowest validation equal error rate",
        "Embedding: 256 values per branch, 512 per print",
        lines[first + 18],
        "Alignment: none (--localization off)",
        f"Loss weights: cross entropy {weights.cross_entropy:g}, center {weights.center:g}, "
        f"minutia map {weights.minutia_map:g}",
        f"Output: {tmp_path / 'model'}",
    ]
    assert lines[first + 18].startswith("Optimizer: Adam, learning rate ")
    assert lines[first + 22] == ("Device: CUDA" if torch.cuda.is_available() else "Device: CPU")
    # Before it: the table is read, and the three sets of identifiers are made
    assert lines[:first] == [
        f"Reading the sample table {grid_set / 'table'}...",
        "Created IdentifierSet with 4 subjects and a total of 8 samples.",
        "Created IdentifierSet with 2 subjects and a total of 4 samples.",
        "Created IdentifierSet with 2 subjects and a total of 4 samples.",
    ]
    # After it: what the validation set holds, the checks of the files, the
    # resolutions, and then the lines of directory mode about model and datasets
    rest = [line for line in lines[last + 1 :] if line]
    assert rest[0].startswith("Validation set: 4 images of 2 subjects")
    assert rest[1:6] == [
        "Created 4 training identifiers",
        "Created 4 validation identifiers",
        f"Listing the directories of the image files below {grid_set / 'images'}...",
        "Loaded 8 image files spread over the table: each has the size in bytes and in pixels "
        "that its row holds.",
        "Resolution of the 4 training images, as the sample table states it:",
    ]
    assert rest[rest.index("Initializing DeepPrint_TexMinu model...") :][:9] == [
        "Initializing DeepPrint_TexMinu model...",
        f"Loading training image dataset from {grid_set / 'images'}...",
        "Creating minutia dataset...",
        f"Loading minutiae from the sample table {grid_set / 'table'}",
        "Creating label dataset...",
        "Creating validation dataset...",
        "Creating validation benchmark...",
        "Validation benchmark created successfully",
        "Starting training for up to 2 epochs...",
    ]
    assert "Training complete! 2 epoch(s) trained in this run, 2 in the log." in rest


def test_many_training_subjects_get_a_warning(train_500k, grid_set, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_500k, "MANY_SUBJECTS", 3)
    capsys.readouterr()
    run_table(train_500k, monkeypatch, grid_set, "--validation_split", 0, output=tmp_path / "a")
    (warning,) = [line for line in printed(capsys) if "WARNING" in line]
    assert warning.startswith("WARNING: 4 training subjects. The classifier and the centers")
    monkeypatch.setattr(train_500k, "MANY_SUBJECTS", 4)
    run_table(train_500k, monkeypatch, grid_set, "--validation_split", 0, output=tmp_path / "b")
    assert "WARNING" not in capsys.readouterr().out


def test_worker_processes_that_are_not_forked_get_a_warning(
    train_500k, grid_set, tmp_path, monkeypatch, capsys
):
    """A worker that is not forked gets a copy of the arrays of the table."""
    multiprocessing = train_500k.multiprocessing

    def warnings(output: str, fixed, default: str) -> list:
        """The warnings of a start, for a start method that is fixed (or None) and a default."""
        asked = []
        monkeypatch.setattr(
            multiprocessing,
            "get_start_method",
            lambda allow_none=False: asked.append(allow_none) or fixed,
        )
        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: [default, "other"])
        capsys.readouterr()
        run_table(
            train_500k, monkeypatch, grid_set, "--validation_split", 0, output=tmp_path / output
        )
        # The start method is asked for without fixing it for the rest of the program
        assert asked == [True]
        return [line for line in printed(capsys) if "WARNING" in line]

    assert warnings("fork", "fork", "spawn") == []
    assert warnings("default fork", None, "fork") == []
    for output, fixed, default in (("spawn", "spawn", "fork"), ("default", None, "forkserver")):
        (warning,) = warnings(output, fixed, default)
        assert warning.startswith(
            f"WARNING: worker processes are started with '{fixed or default}' here, not forked."
        )
        assert "every worker process of the training holds the table once more" in warning


def test_only_the_first_process_prints_the_start_and_every_process_checks(
    train_500k, builder, tmp_path, monkeypatch, capsys
):
    """
    With several GPUs the script runs once per GPU. Every process reads the
    table and stops by itself if files are missing; what the start prints
    comes from the first process only.
    """
    write_grid(tmp_path)
    assert build(builder, tmp_path) == 0
    options = ["--validation_split", 0.5]
    monkeypatch.setenv("RANK", "1")
    assert not train_500k._is_first_process()
    capsys.readouterr()
    fake = run_table(train_500k, monkeypatch, tmp_path, *options, output="second")
    # It trains on what the first process trains on, ...
    (fit,) = fake.fits
    assert len(fit["fingerprints"]) == 4 and len(fit["validation_fingerprints"]) == 4
    assert len(fit["validation_benchmark"]._comparisons) == 4
    # ... and prints nothing of its own before the training, which is the
    # same code in both modes, starts
    printed_lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert printed_lines[0] == "Starting training for up to 2 epochs..."
    assert all(line.startswith(("Starting training", "For multi-GPU")) for line in printed_lines)
    # Only the first process stores the settings
    assert not exists(tmp_path / "second" / "training_settings.json")

    monkeypatch.setenv("RANK", "0")
    capsys.readouterr()
    run_table(train_500k, monkeypatch, tmp_path, *options, output="first")
    assert len(printed(capsys)) > 40 and exists(tmp_path / "first" / "training_settings.json")

    # A file that changed is found by the first process, which loads files
    # at the start; the others go on to the training
    kept = (tmp_path / "images" / "1_1.png").read_bytes()
    save(tmp_path / "images" / "1_1.png", ridge_image(80, 60), 500)
    refused(run_table, train_500k, monkeypatch, capsys, tmp_path, *options, output="changed")
    monkeypatch.setenv("RANK", "1")
    assert len(run_table(train_500k, monkeypatch, tmp_path, *options, output="changed").fits) == 1
    (tmp_path / "images" / "1_1.png").write_bytes(kept)

    # A file that is missing stops every process, with the reason
    (tmp_path / "images" / "4_1.png").unlink()
    for rank in ("1", "0"):
        monkeypatch.setenv("RANK", rank)
        message = refused(
            run_table, train_500k, monkeypatch, capsys, tmp_path, *options, output="missing"
        )
        assert message.startswith("ERROR: 1 of the 8 image files that the sample table names")
        assert "'4_1.png' (subject 3, impression 0)" in message
    # So does a table that cannot be read
    monkeypatch.setenv("RANK", "1")
    message = refused(
        run_table, train_500k, monkeypatch, capsys, tmp_path, *options, table="no table",
        output="missing",
    )
    assert message.startswith(f"ERROR: {tmp_path / 'no table'} cannot be read")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Using the trained model
# ---------------------------------------------------------------------------


def test_a_trained_model_is_loaded_with_a_table_of_the_images_to_embed(
    corpus, train_500k, tmp_path, monkeypatch
):
    """
    load_trained_model with samples: every image is prepared as its row of
    the table says and then as the training prepared its images. On the
    files that both modes read, that is the loader of a directory training.
    """
    options = ["--validation_split", VALIDATION_SPLIT, "--canvas_size", 512]
    options += ["--localization", "off", "--embedding_dim", 96]
    images, table_dir = str(corpus.root / "images"), str(corpus.root / "table")
    from_table, from_directory = str(tmp_path / "from_table"), str(tmp_path / "from_directory")
    run_table(train_500k, monkeypatch, corpus.root, *options, output=from_table)
    run_directory(
        train_500k, monkeypatch, corpus.root, *corpus.directory_options(), *options,
        output=from_directory,
    )

    built, loaded = [], []
    fake = _FakeExtractor()

    def constructor(**kwargs):
        built.append(kwargs)
        return fake

    monkeypatch.setattr(train_500k, "get_DeepPrint_TexMinu", constructor)
    monkeypatch.setattr(
        train_500k.torch_helpers, "load_model_parameters", lambda *a: loaded.append(a)
    )

    # A model that was trained from a table, with a table
    extractor, image_loader = train_500k.load_trained_model(from_table, images, samples=table_dir)
    assert extractor is fake
    # One class per TRAINING subject: six subjects minus the two held out
    assert built == [{"num_training_subjects": 4, "num_dims": 96, "loss_weights": None}]
    assert loaded == [(join(from_table, "best_model.pyt"), "the model", None, None)]
    assert type(image_loader._images) is SampleTableImageLoader
    # The loader has the identifiers of all rows, as the documented use needs them
    dataset = Dataset(image_loader, image_loader.ids)
    assert numbers(image_loader.ids) == sorted(corpus.prints) and len(dataset) == 12

    # A model that was trained from the directories, without a table: as ever
    _, directory_loader = train_500k.load_trained_model(from_directory, images)
    assert type(directory_loader._images) is train_500k.CustomFingerprintLoader
    assert numbers(directory_loader.ids) == numbers(image_loader.ids)
    # ... and with a table: the table says how the files are prepared, the
    # training which canvas they get
    _, mixed_loader = train_500k.load_trained_model(from_directory, images, samples=table_dir)
    assert type(mixed_loader._images) is SampleTableImageLoader
    assert built[1:] == built[:1] * 2 and len(loaded) == 3

    # The three loaders give the same model input for every image
    embedded = {}
    for index, identifier in enumerate(image_loader.ids):
        image = dataset[index]
        assert image.shape == (1, INPUT_SIZE, INPUT_SIZE) and image.dtype == torch.float32
        assert torch.equal(image, directory_loader.get(identifier)), str(identifier)
        assert torch.equal(image, mixed_loader.get(identifier)), str(identifier)
        embedded[(identifier.subject, identifier.impression)] = image
    # Control: they are the images of twelve prints, with the canvas of the
    # training and not another one, and the light prints with dark ridges
    values = list(embedded.values())
    assert not any(torch.equal(a, b) for index, a in enumerate(values) for b in values[:index])
    without_canvas = train_500k.create_validation_image_loader(
        images, None, 0, files=SampleTableImageLoader(corpus.table)
    )
    light = corpus.number_of(LIGHT)
    assert not torch.equal(without_canvas.get(Identifier(*light)), embedded[light])
    as_dark = dataclasses.replace(
        corpus.table.samples, ridges_light=np.zeros(12, dtype=np.bool_)
    )
    taken_as_dark = train_500k.create_validation_image_loader(
        images, None, 512, files=SampleTableImageLoader(SampleTable(as_dark, images))
    )
    for number, spec in corpus.prints.items():
        same = torch.equal(taken_as_dark.get(Identifier(*number)), embedded[number])
        assert same == (not spec.light), number

    # The newest model instead of the best one
    train_500k.load_trained_model(from_table, images, best=False, samples=table_dir)
    assert loaded[-1][0] == join(from_table, "model.pyt")


def test_a_model_that_was_trained_from_a_table_is_not_loaded_without_one(
    train_500k, grid_set, tmp_path, monkeypatch
):
    from flx.preparation.sample_table import SampleTableError

    images, table_dir = str(grid_set / "images"), str(grid_set / "table")
    output = str(tmp_path / "model")
    run_table(train_500k, monkeypatch, grid_set, "--validation_split", 0, output=output)
    built = []
    monkeypatch.setattr(
        train_500k, "create_extractor", lambda *a, **k: built.append(k) or _FakeExtractor()
    )
    monkeypatch.setattr(train_500k.torch_helpers, "load_model_parameters", lambda *a: None)

    # Its settings hold no resolution sources: a message, not a KeyError
    for arguments in ({}, {"ppi_manifest": str(tmp_path / "resolutions.csv")}):
        with pytest.raises(ValueError) as error:
            train_500k.load_trained_model(output, images, **arguments)
        message = str(error.value)
        assert message.startswith(
            f"The training in {output} read its images from a sample table (--samples {table_dir})"
        )
        assert "Pass samples= with the directory of a sample table" in message
        assert "scripts/build_sample_table.py" in message and "docs/sample-table.md" in message
    # Nothing was built for it
    assert built == []
    _, image_loader = train_500k.load_trained_model(output, images, samples=table_dir)
    assert len(image_loader.ids) == 8 and len(built) == 1

    # A manifest has no meaning beside a table
    with pytest.raises(ValueError, match="ppi_manifest and samples were both given"):
        train_500k.load_trained_model(
            output, images, ppi_manifest=str(tmp_path / "resolutions.csv"), samples=table_dir
        )
    # A table that cannot be read is refused as everywhere
    with pytest.raises(SampleTableError, match="cannot be read as the directory of a sample"):
        train_500k.load_trained_model(output, images, samples=str(tmp_path / "no table"))

    # A training on images of another resolution than those of the table
    settings = stored_settings(tmp_path)
    settings["preparation"]["target_ppi"] = 250.0
    with open(tmp_path / "model" / "training_settings.json", "w") as file:
        json.dump(settings, file)
    with pytest.raises(ValueError) as error:
        train_500k.load_trained_model(output, images, samples=table_dir)
    message = str(error.value)
    assert message.startswith(
        f"The sample table {table_dir} describes images that are prepared at 500 ppi, but "
        f"the model in {output} was trained on images at 250 ppi: "
    )
    assert "Use a table whose images are prepared at the resolution of the training" in message
    assert len(built) == 1
    with pytest.raises(FileNotFoundError, match="training_settings.json"):
        train_500k.load_trained_model(str(tmp_path / "nothing"), images, samples=table_dir)
