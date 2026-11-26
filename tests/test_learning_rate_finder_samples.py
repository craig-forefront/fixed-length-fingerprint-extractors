"""
scripts/find_learning_rate.py with a sample table (--samples): the range test
takes its images from rows of a table, and not from the files that the names
in two directories give.

Three promises are tested. For the files that both modes can read (.png files
named <subject>_<impression>, a complete grid, .mnt files whose header is the
size of the prepared image), the test from a table is the test from the
directories: main() is run in both modes, and what the range test is given is
compared, every image and minutia map bit by bit. For any table, the test
follows the rules of the training (scripts/train_500k.py): which variant
--variant auto is, what stops a test before the model is built, and that all
of this is about the rows of the chosen subjects and about no other row. And
without --samples the script does what it did before it knew sample tables:
every line that such a run prints is written out here.

The tables are built by scripts/build_sample_table.py through its main(), so
that the way from the builder to the range test is what is tested. The model
is replaced by a tiny one, and in most tests the range test itself by a
function that keeps what it is given; the images and the minutia maps go
through the real loaders, the real augmentation and the real binariser. What
a test replaces of the training script, it replaces in
find_learning_rate.train_500k, which is the module that the script under test
uses. The .mnt files are written as text, and no test module is imported:
the helpers are copies.
"""

import functools
import importlib.util
import multiprocessing
import os
import pickle
import random
import re
import shutil
import sys
from os.path import abspath, dirname, exists, join
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch
import tqdm
from PIL import Image

pytest.importorskip("pyarrow")

from flx.data.dataset import ConstantDataLoader, Identifier  # noqa: E402
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
from flx.models.deep_print_arch import DeepPrintTrainingOutput  # noqa: E402
from flx.models.deep_print_loss import LossWeights  # noqa: E402
from flx.models.learning_rate_finder import (  # noqa: E402
    CSV_COLUMNS,
    RangeTestResult,
    smooth_losses,
    suggest_learning_rate,
)
from flx.models.model_training import TrainingLog, TrainingLogEntry  # noqa: E402
from flx.preparation.sample_table import (  # noqa: E402
    PPI_ORIGINS,
    RIDGES_ORIGINS,
    SampleRow,
    SampleTableError,
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


@pytest.fixture(scope="module")
def find_learning_rate():
    """
    scripts/find_learning_rate.py as a module. The script imports the
    training script by its name, train_500k, from its own directory.
    """
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        spec = importlib.util.spec_from_file_location(
            "find_learning_rate", join(SCRIPTS_DIR, "find_learning_rate.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(SCRIPTS_DIR)
    return module


@pytest.fixture(scope="module")
def builder():
    """scripts/build_sample_table.py as a module. It adds a directory to sys.path when loaded."""
    path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            "build_sample_table", join(SCRIPTS_DIR, "build_sample_table.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = path


@pytest.fixture(autouse=True)
def default_batch_size():
    """The tests that ask the training script what it does run its main(), which sets it."""
    yield
    torch_helpers.set_training_batch_size(torch_helpers.DEFAULT_TRAINING_BATCH_SIZE)


@pytest.fixture(autouse=True)
def forked_workers(monkeypatch):
    """
    A test from a table warns where worker processes are not forked, and
    whether they are is a default of the Python and of the system that run
    the tests. Every test tells the scripts that they are forked, so that
    what it says about the output of a run holds wherever it runs; the test
    of that warning tells the scripts what it needs itself.
    """
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: "fork")


# ---------------------------------------------------------------------------
# Running main() with a tiny model
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    """One layer that looks at one pixel of every image, and keeps what it was given."""

    def __init__(self, num_subjects: int, seen: dict):
        super().__init__()
        self.linear = torch.nn.Linear(1, num_subjects)
        self.seen = seen

    def forward(self, images):
        self.seen["image_shape"] = tuple(images.shape)
        return DeepPrintTrainingOutput(
            texture_logits=self.linear(images[:, 0, 0, 0].reshape(-1, 1))
        )


class _SquareLoss(torch.nn.Module):
    """1 + mean of the squared logits: lowest (1) when all logits are zero."""

    def __init__(self, seen: dict):
        super().__init__()
        self.seen = seen

    def forward(self, output, labels, minutia_maps, minutia_map_weights):
        self.seen["map_shape"] = tuple(minutia_maps.shape)
        self.seen["labels"] = self.seen.get("labels", set()) | set(labels.tolist())
        return 1 + (output.texture_logits**2).mean()

    def reset_recorded_loss(self):
        pass


class _TinyExtractor:
    def __init__(self, num_subjects: int):
        # What the model and the loss saw of the last batch, and all labels
        self.seen = {}
        self.model = _TinyModel(num_subjects, self.seen)
        self.loss = _SquareLoss(self.seen)


class Recorded:
    """What a run of main() built, and what its range test was given."""

    def __init__(self):
        # Per model: (variant, number of classes, embedding_dim, loss weights)
        self.built = []
        # The tiny extractors that stood in for these models
        self.extractors = []
        # Per range test: its keyword arguments, and its model, loss and train_set
        self.tests = []
        # The results of the range tests that were really run
        self.results = []


# What stands in for the result of a range test that is not run: two steps,
# too few for a suggestion
TWO_STEPS = ([1e-4, 1e-3], [1.5, 1.25], [0.5, 0.25], "all steps done")
# Steps with a suggestion: the loss falls, and rises at the end
FALL_AND_RISE = (
    [10.0 ** (-6 + step / 4) for step in range(24)],
    [3.0] * 6 + [3.0 - 0.15 * step for step in range(1, 13)] + [1.3, 1.5, 2.0, 3.0, 5.0, 9.0],
    [0.5] * 24,
    "loss exploded",
)


def run(
    find_learning_rate, *arguments, real: bool = False, recorded: Optional[Recorded] = None,
    steps: tuple = TWO_STEPS,
):
    """
    Runs main() with the arguments and a tiny model in the place of DeepPrint.
    Unless real is set, the range test is not run: a function in its place
    keeps what it is given and returns the steps. Returns what was recorded.
    """
    recorded = Recorded() if recorded is None else recorded
    run_range_test = find_learning_rate.run_range_test

    def build(variant, num_subjects, embedding_dim, loss_weights=None):
        recorded.built.append((variant, num_subjects, embedding_dim, loss_weights))
        recorded.extractors.append(_TinyExtractor(num_subjects))
        return recorded.extractors[-1]

    def range_test(model, loss, train_set, **options):
        recorded.tests.append(dict(options, model=model, loss=loss, train_set=train_set))
        if not real:
            return RangeTestResult(*(list(v) if isinstance(v, list) else v for v in steps))
        recorded.results.append(run_range_test(model, loss, train_set, **options))
        return recorded.results[-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(find_learning_rate, "build_extractor", build)
        patch.setattr(find_learning_rate, "run_range_test", range_test)
        patch.setattr(sys, "argv", ["find_learning_rate.py"] + [str(a) for a in arguments])
        find_learning_rate.main()
    return recorded


def run_table(find_learning_rate, root, *options, table="table", output="out", **kwargs):
    """Table mode: main() on the images in root/images and the table root/<table>."""
    return run(
        find_learning_rate,
        *("--dataset_dir", root / "images", "--samples", root / table),
        *("--output_dir", root / output),
        *options,
        **kwargs,
    )


def run_directory(find_learning_rate, root, *options, output="out", **kwargs):
    """
    Directory mode: main() on the images in root/images. The options say
    which grid of subjects and impressions the directory holds.
    """
    return run(
        find_learning_rate,
        *("--dataset_dir", root / "images", "--output_dir", root / output),
        *options,
        **kwargs,
    )


def refusal(runner, find_learning_rate, capsys, root, *options, output="refused", **kwargs):
    """
    A test that is refused: main() stops with the exit status 2, builds no
    model, runs no range test and does not make --output_dir (by default one
    that no run of the tests makes). Returns the lines that it printed, and
    what it wrote to standard error on one line.
    """
    capsys.readouterr()
    recorded = Recorded()
    with pytest.raises(SystemExit) as stop:
        runner(find_learning_rate, root, *options, output=output, recorded=recorded, **kwargs)
    assert stop.value.code == 2
    assert recorded.built == [] and recorded.tests == []
    assert not exists(root / output)
    captured = capsys.readouterr()
    return captured.out.splitlines(), " ".join(captured.err.split())


def refused(runner, find_learning_rate, capsys, root, *options, **kwargs) -> str:
    """A test that is refused (see refusal). Returns its message."""
    return refusal(runner, find_learning_rate, capsys, root, *options, **kwargs)[1]


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


class _FakeTraining:
    """In the place of the model of the training: its fit writes what train_model writes."""

    def fit(self, out_dir, num_epochs, **kwargs):
        log = TrainingLog(join(out_dir, "log.json"))
        for epoch in range(len(log) + 1, num_epochs + 1):
            log.add_entry(TrainingLogEntry(epoch, 5.0 - epoch, {}, 0.1, None, 1e-4))
            for name in ("model.pyt", "best_model.pyt"):
                with open(join(out_dir, name), "w") as file:
                    file.write(str(epoch))


def train(find_learning_rate, root, *options, table="table", output="model") -> list:
    """
    Runs main() of scripts/train_500k.py on the table, with all its subjects
    in the training set and a model that trains nothing. Returns the variants
    of the models that it built: one, unless the start is refused.
    """
    train_500k = find_learning_rate.train_500k
    built = []
    arguments = ["--dataset_dir", root / "images", "--samples", root / table]
    arguments += ["--output_dir", root / output, "--validation_split", 0, "--num_epochs", 1]
    with pytest.MonkeyPatch.context() as patch:
        for variant in VARIANTS:

            def constructor(*args, _variant=variant, **kwargs):
                built.append(_variant)
                return _FakeTraining()

            patch.setattr(train_500k, "get_DeepPrint_" + variant, constructor)
        patch.setattr(sys, "argv", ["train_500k.py"] + [str(a) for a in arguments + [*options]])
        train_500k.main()
    return built


def numbers(ids) -> list:
    """(subject, impression) of the identifiers of a set or a dataset, in its order."""
    return [(i.subject, i.impression) for i in ids]


def printed(capsys) -> list:
    """The lines that were printed, each with single spaces."""
    return [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]


def loaders(train_set) -> list:
    """The loaders of the images, of the minutia maps and of the labels of a zipped set."""
    images, maps, labels = train_set.data_loader._loaders
    return [images, maps, labels]


def options_of(test: dict) -> dict:
    """The options that a range test was given: its arguments without model, loss and set."""
    return {key: value for key, value in test.items() if key not in ("model", "loss", "train_set")}


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
    directory, each another image. They are small, so that the tests are
    quick, and they have the pixels that the binariser of the training needs
    (LARGEST_BINARISER_KERNEL per side).
    """
    for number, name in enumerate(names):
        save(image_dir / name, ridge_image(size[1], size[0], slant=0.05 * (number + 1)), ppi)


def write_small_minutiae(minutiae_dir, names, size=(48, 64)) -> None:
    """The .mnt file of each of the small prints, where the extraction writes it."""
    for name in names:
        path = minutiae_dir / (os.path.splitext(name)[0] + ".mnt")
        write_mnt_file(path, [(10.5, 20.25, 0.5), (30.0, 40.0, 2.0)], size)


def grid(subjects: int, impressions: int) -> list:
    """The file names of a complete grid as directory mode reads it, numbered from 1."""
    return [
        f"{subject}_{impression}.png"
        for subject in range(1, subjects + 1)
        for impression in range(1, impressions + 1)
    ]


def write_grid(root, subjects: int = 4, impressions: int = 2, minutiae: bool = False) -> list:
    """
    The small prints of a grid in root/images, and their .mnt files in
    root/minutiae. Returns the names of the image files.
    """
    names = grid(subjects, impressions)
    write_small_prints(root / "images", names)
    if minutiae:
        write_small_minutiae(root / "minutiae", names)
    return names


def chosen_by_the_rule(subjects, limit: int, seed: int) -> list:
    """
    The rule of the subject choice, written out with the standard library:
    the subjects in ascending order, and if the limit is above 0 and below
    their number, a sorted sample of that many of them.
    """
    subjects = sorted(subjects)
    if 0 < limit < len(subjects):
        return sorted(random.Random(seed).sample(subjects, limit))
    return subjects


def seed_that_chooses(subjects, chosen) -> int:
    """A seed with which the rule chooses exactly these of the subjects."""
    chosen = sorted(chosen)
    return next(
        seed for seed in range(1000) if chosen_by_the_rule(subjects, len(chosen), seed) == chosen
    )


# ---------------------------------------------------------------------------
# The subjects of the test
# ---------------------------------------------------------------------------


def test_the_subjects_of_a_grid_are_chosen_as_they_are_chosen_without_a_table(
    find_learning_rate,
):
    """choose_table_subjects for the subjects 0 .. n - 1 against choose_subjects for n."""
    cases = with_a_choice = 0
    for count in (1, 2, 3, 7, 10, 50, 333):
        for limit in (-1, 0, 1, 2, 5, count - 1, count, count + 1, 2000):
            for seed in (0, 1, 7, 42):
                expected = find_learning_rate.choose_subjects(count, limit, seed)
                found = find_learning_rate.choose_table_subjects(list(range(count)), limit, seed)
                assert found == expected, (count, limit, seed)
                assert all(type(subject) is int for subject in found)
                cases += 1
                with_a_choice += len(expected) < count
    # Both were among the cases: all subjects, and a choice among them
    assert cases == 7 * 9 * 4 and 50 < with_a_choice < cases - 50


def test_the_subjects_of_a_table_with_gaps_are_a_sorted_sample_of_its_sorted_subjects(
    find_learning_rate,
):
    choose = find_learning_rate.choose_table_subjects
    subjects = [300, 7, 1000, 2, 40, 11]
    ascending = [2, 7, 11, 40, 300, 1000]
    # All of them in ascending order: without a limit, and with a limit that
    # is not below their number
    for limit in (0, -3, 6, 7, 2000):
        assert choose(subjects, limit, 0) == ascending
    choices = set()
    for limit in (1, 2, 3, 4, 5):
        for seed in range(20):
            expected = sorted(random.Random(seed).sample(ascending, limit))
            assert choose(subjects, limit, seed) == expected, (limit, seed)
            # The order in which the subjects are given does not count
            assert choose(ascending[::-1], limit, seed) == expected, (limit, seed)
            choices.add(tuple(expected))
    # Control: the seed decides which subjects are chosen
    assert len(choices) > 20
    # The list that was given is not changed, and neither are the random
    # numbers of the rest of the program
    assert subjects == [300, 7, 1000, 2, 40, 11]
    state = random.getstate()
    choose(subjects, 3, 1)
    assert random.getstate() == state


# ---------------------------------------------------------------------------
# The same range test from a table as from the directories
# ---------------------------------------------------------------------------

# What the .mnt file of a print is, where it is not a number of minutiae
HEADER_ZERO = 0  # the header 0, as the extraction writes it for an image without minutiae
ZERO_BYTES = "a file of 0 bytes"


class Print(NamedTuple):
    """A print of the comparison of the two modes."""

    directory: str  # below the image directory
    size: tuple  # (width, height) of the image file
    ppi: float  # resolution of the file
    stated: str  # where the resolution is stated: "metadata" of the file or "manifest"
    light: bool  # light ridges on a dark background
    minutiae: object  # the number of minutiae of its .mnt file, or one of the two above


# The prints of four subjects, two for each. The range test needs the .mnt
# file of every image that it uses, so each of them has one.
PRINTS = [
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
FRAME_OF_MOVED_LOCATIONS, FLOAT32_PPI, LIGHT = PRINTS[0], PRINTS[3], PRINTS[4]
SUBJECTS, IMPRESSIONS = 4, 2


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
def corpus(tmp_path_factory, builder) -> Corpus:
    """
    Four subjects with two prints each, as files that both modes read, the
    .mnt files of an extraction, and the table that the builder makes of
    them. No test changes these files.
    """
    root = tmp_path_factory.mktemp("corpus")
    prints = {
        (position // IMPRESSIONS, position % IMPRESSIONS): spec
        for position, spec in enumerate(PRINTS)
    }
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
        if spec.minutiae == ZERO_BYTES:
            os.makedirs(dirname(str(mnt_path)), exist_ok=True)
            open(mnt_path, "w").close()
            continue
        _, info = normalizer.load(str(root / "images" / (name + ".png")), check_polarity=True)
        size = (info.width, info.height)
        write_mnt_file(mnt_path, minutiae_of_print(spec, number, size), size, info.describe())

    status = build(
        builder,
        root,
        *("--minutiae_dir", root / "minutiae"),
        *("--ppi_manifest", root / "resolutions.csv", "--ridges", "auto"),
    )
    assert status == 0
    table = SampleTable.read(str(root / "table"), str(root / "images"))
    return Corpus(root, prints, table)


def test_the_files_of_the_comparison_are_those_that_show_the_faults(corpus):
    """The controls that belong to the files; the others are beside the comparison."""
    table, samples = corpus.table, corpus.table.samples
    assert table.num_rows == len(corpus.prints) == 8
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

    # The resolution of one file is another number in 32 bits, and its
    # prepared size depends on that: a table, or a loader, that held it in 32
    # bits would give an image of another size
    row = samples.find(*corpus.number_of(FLOAT32_PPI))
    ppi = float(samples.ppi[row])
    assert ppi == FLOAT32_PPI.ppi and float(np.float32(ppi)) != ppi
    assert resampled_size(*FLOAT32_PPI.size, ppi) == (312, 400)
    assert resampled_size(*FLOAT32_PPI.size, float(np.float32(ppi))) == (313, 400)
    assert (samples.width[row], samples.height[row]) == (312, 400)

    # The light ridges of one print were found out from the image, as were
    # the dark ones of the others, and the table holds what was found
    taken_as_light = np.flatnonzero(samples.ridges_light)
    assert [numbers(table.identifiers)[row] for row in taken_as_light] == [
        corpus.number_of(LIGHT)
    ]
    assert {RIDGES_ORIGINS[origin] for origin in samples.ridges_origin.tolist()} == {"estimated"}

    # The two .mnt files without minutiae gave rows with an empty list, and
    # no row is without minutiae: the range test from the directories needs
    # the .mnt file of every image
    for number, spec in corpus.prints.items():
        row = samples.find(*number)
        mnt_path = corpus.root / "minutiae" / (corpus.path(number)[:-4] + ".mnt")
        assert samples.minutiae_known[row]
        if spec.minutiae == ZERO_BYTES:
            assert os.path.getsize(mnt_path) == 0 and len(samples.minutiae_of(row)) == 0
        elif spec.minutiae == HEADER_ZERO:
            with open(mnt_path) as file:
                assert file.read().startswith(f"0\n{spec.size[0]} {spec.size[1]}\n")
            assert len(samples.minutiae_of(row)) == 0
        else:
            assert len(samples.minutiae_of(row)) == spec.minutiae
    kinds = [spec.minutiae for spec in corpus.prints.values()]
    assert kinds.count(ZERO_BYTES) == kinds.count(HEADER_ZERO) == 1


def fetch(train_set, seed: int = 0) -> dict:
    """
    What a range test would be given by its training set: for every
    identifier the image, its minutia map, the weight of the map and the
    label. The random numbers of the augmentation are seeded directly before
    the sample of an identifier is fetched; the set fetches the image first
    and then the map, which takes the pose of the image that was loaded last.
    """
    state = random.getstate()
    try:
        samples = {}
        for number, identifier in enumerate(train_set.ids):
            random.seed(1000 + 10 * number + seed)
            image, (minutia_map, weight), label = train_set.get(identifier)
            samples[(identifier.subject, identifier.impression)] = (
                image,
                minutia_map,
                weight,
                label,
            )
    finally:
        random.setstate(state)
    return samples


def assert_same_samples(found: dict, expected: dict) -> None:
    """Images, minutia maps, weights and labels are equal, sample by sample and bit by bit."""
    assert list(found) == list(expected)
    for number, sample in found.items():
        for tensor, other in zip(sample[:2], expected[number][:2]):
            assert tensor.dtype == other.dtype, number
            assert torch.equal(tensor, other), number
        assert sample[2:] == expected[number][2:], number


@pytest.mark.parametrize(
    "variant, canvas_size", [("auto", 0), ("TexMinu", 512), ("LocTex", 250)]
)
def test_a_range_test_from_a_table_is_the_range_test_from_the_directories(
    corpus, find_learning_rate, tmp_path, variant, canvas_size
):
    """
    main() in both modes on the same files, with a function in the place of
    the range test that keeps what it is given: the same model, the same
    identifiers in the same order, the same options, and a training set with
    bit-identical images, minutia maps, weights and labels, with the random
    pose and quality of the training, the real binariser and the canvas.
    """
    train_500k = find_learning_rate.train_500k
    # A canvas of 512 cuts the highest print and pads the others; one of 250
    # is smaller than every print in one direction at least
    longer_sides = [max(resampled_size(*spec.size, spec.ppi)) for spec in corpus.prints.values()]
    assert sorted(longer_sides)[-2] < 512 < max(longer_sides) and min(longer_sides) > 250
    # auto is the variant of the training with minutiae: both modes are given them
    model = "LocTexMinu" if variant == "auto" else variant
    with_minutiae = model.endswith("Minu")
    assert train_500k.DEFAULT_LOCALIZATION == "on"
    common = ["--variant", variant, "--canvas_size", canvas_size]

    def directory_mode(*options) -> Recorded:
        return run_directory(
            find_learning_rate, corpus.root, *corpus.directory_options(), *common, *options,
            output=tmp_path / "directory",
        )

    def table_mode(*options) -> Recorded:
        return run_table(
            find_learning_rate, corpus.root, *common, *options, output=tmp_path / "table"
        )

    directory, table = directory_mode(), table_mode()
    (test,), (table_test,) = directory.tests, table.tests
    train_set, table_set = test["train_set"], table_test["train_set"]

    # The two runs went through the loaders of their own mode
    (images, maps, _), (table_images, table_maps, _) = loaders(train_set), loaders(table_set)
    assert type(images._images) is train_500k.CustomFingerprintLoader
    assert type(table_images._images) is SampleTableImageLoader
    if with_minutiae:
        inner, table_inner = maps._minutia_maps, table_maps._minutia_maps
        assert type(inner) is FingerFlowMinutiaMapLoader
        assert type(table_inner) is SampleTableMinutiaMapLoader
        # One table for both loaders of the set
        assert table_inner.table is table_images._images.table
    else:
        assert type(maps) is type(table_maps) is ConstantDataLoader

    # The same model with the same number of classes, ...
    assert table.built == directory.built == [(model, 4, 256, LossWeights())]
    # ... the same identifiers in the same order, ...
    assert numbers(table_set.ids) == numbers(train_set.ids) == sorted(corpus.prints)
    assert len(table_set) == len(train_set) == 8
    # ... and the same options of the range test
    assert options_of(table_test) == options_of(test)
    assert sorted(options_of(test)) == [
        "batch_size", "gradient_clip_norm", "localization_factor", "max_lr", "min_lr",
        "num_steps", "num_workers", "seed", "smoothing",
    ]

    if with_minutiae:
        # The minutiae as the loaders of the two modes read them, before any
        # pose: the same arrays for every image
        for identifier in train_set.ids:
            expected_locations, expected_angles = inner.get_minutiae(identifier)
            locations, angles = table_inner.get_minutiae(identifier)
            number = (identifier.subject, identifier.impression)
            for array, expected in ((locations, expected_locations), (angles, expected_angles)):
                assert array.dtype == expected.dtype and array.shape == expected.shape, number
                assert np.array_equal(array, expected), number
            count = corpus.prints[number].minutiae
            assert len(locations) == (count if isinstance(count, int) else 0)
        # Control: in the frame of 500x640 pixels the locations depend on
        # whether the sizes are Python numbers, as the loader of .mnt files
        # reads them from the header, or numpy numbers, as the arrays of a
        # table hold them. A table loader that passed the sizes on as they
        # are would have shown.
        number = corpus.number_of(FRAME_OF_MOVED_LOCATIONS)
        in_file, _, (width, height) = read_mnt_file(
            str(corpus.root / "minutiae" / (corpus.path(number)[:-4] + ".mnt"))
        )
        assert (width, height) == (500, 640) and in_file[0].tolist() == [175.0, 300.0]
        with_python_numbers = transform_to_input_size(in_file, height, width)
        with_numpy_numbers = transform_to_input_size(in_file, np.int32(height), np.int32(width))
        assert np.array_equal(table_inner.get_minutiae(Identifier(*number))[0], with_python_numbers)
        # From NumPy 2 on. NumPy 1 scales the locations in their 16 bits with
        # both kinds of numbers, so there is nothing to show.
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            assert not np.array_equal(with_python_numbers, with_numpy_numbers)

    # The samples. First directory mode against itself, in a second run: if
    # it did not repeat with the same seeds, a difference to the table would
    # say nothing
    expected = fetch(train_set)
    assert_same_samples(fetch(directory_mode().tests[0]["train_set"]), expected)
    # Control: the seeds are what the samples depend on, ...
    other_poses = fetch(train_set, seed=1)
    assert not any(
        torch.equal(sample[0], other[0])
        for sample, other in zip(expected.values(), other_poses.values())
    )
    # ... every image is the model input of another print, with ridges and
    # background, ...
    images = [sample[0] for sample in expected.values()]
    assert all(image.shape == (1, INPUT_SIZE, INPUT_SIZE) for image in images)
    assert all(0.01 < float(image.mean()) < 0.99 for image in images)
    assert not any(
        torch.equal(image, other) for index, image in enumerate(images) for other in images[:index]
    )
    # ... every subject is a class of its own, ...
    assert [sample[3] for sample in expected.values()] == [0, 0, 1, 1, 2, 2, 3, 3]
    weights = {number: sample[2] for number, sample in expected.items()}
    if with_minutiae:
        # ... and the minutia maps hold minutiae, except those of the two
        # .mnt files without any, which have the weight 0 in both modes
        without = [
            number for number, spec in corpus.prints.items()
            if spec.minutiae in (ZERO_BYTES, HEADER_ZERO)
        ]
        assert len(without) == 2 and all(weights[number] == 0.0 for number in without)
        assert sum(weights.values()) >= len(weights) - 3
        for _, minutia_map, weight, _ in expected.values():
            assert minutia_map.shape == (6, 128, 128)
            assert bool(minutia_map.any()) == (weight == 1.0)
    else:
        # ... and a variant without minutiae gets the placeholder of the
        # maps, also from a table that holds minutiae
        assert all(sample[1].numel() == 0 and sample[2] == 0.0 for sample in expected.values())

    assert_same_samples(fetch(table_set), expected)
    assert_same_samples(fetch(table_set, seed=1), other_poses)

    # With a choice among the subjects: the same subjects, the same number of
    # classes, and the same samples with the classes of the chosen subjects
    chosen = [0, 2, 3]
    choice = ["--max_subjects", 3, "--seed", seed_that_chooses(range(SUBJECTS), chosen)]
    directory, table = directory_mode(*choice), table_mode(*choice)
    train_set, table_set = directory.tests[0]["train_set"], table.tests[0]["train_set"]
    assert numbers(table_set.ids) == numbers(train_set.ids)
    assert numbers(train_set.ids) == [(subject, i) for subject in chosen for i in range(2)]
    assert table.built == directory.built == [(model, 3, 256, LossWeights())]
    assert options_of(table.tests[0]) == options_of(directory.tests[0])
    expected = fetch(train_set)
    assert [sample[3] for sample in expected.values()] == [0, 0, 1, 1, 2, 2]
    assert_same_samples(fetch(table_set), expected)


# ---------------------------------------------------------------------------
# Small sets for the tests that do not compare samples
# ---------------------------------------------------------------------------

GRID_SUBJECTS, GRID_IMPRESSIONS = 7, 3


@pytest.fixture(scope="module")
def grid_set(tmp_path_factory, builder):
    """
    Small prints of seven subjects with three impressions each, as a grid
    that directory mode reads, with their .mnt files and their table, in a
    directory that no test changes: the tests that take it give an
    --output_dir of their own.
    """
    root = tmp_path_factory.mktemp("grid")
    write_grid(root, GRID_SUBJECTS, GRID_IMPRESSIONS, minutiae=True)
    assert build(builder, root, "--minutiae_dir", root / "minutiae") == 0
    return root


def grid_options(root) -> list:
    """What directory mode has to be told about the files of grid_set."""
    return [
        *("--minutiae_dir", root / "minutiae"),
        *("--num_subjects", GRID_SUBJECTS, "--num_impressions", GRID_IMPRESSIONS),
    ]


def test_the_subjects_and_the_classes_of_a_grid_are_those_of_a_test_from_the_directories(
    grid_set, find_learning_rate, tmp_path, capsys
):
    """
    main() in both modes for many choices of the subjects: the same subjects
    with the same identifiers in the same order, the same number of classes,
    and the same lines about the test.
    """
    choices = set()
    for limit in (0, 1, 3, 6, 7, 9, 2000):
        for seed in (0, 1, 42):
            options = ["--max_subjects", limit, "--seed", seed]
            capsys.readouterr()
            directory = run_directory(
                find_learning_rate, grid_set, *grid_options(grid_set), *options,
                output=tmp_path / "directory",
            )
            of_directory_mode = printed(capsys)
            table = run_table(find_learning_rate, grid_set, *options, output=tmp_path / "table")
            of_table_mode = printed(capsys)

            chosen = find_learning_rate.choose_subjects(GRID_SUBJECTS, limit, seed)
            assert chosen == chosen_by_the_rule(range(GRID_SUBJECTS), limit, seed)
            expected = [(subject, i) for subject in chosen for i in range(GRID_IMPRESSIONS)]
            ids, table_ids = directory.tests[0]["train_set"].ids, table.tests[0]["train_set"].ids
            assert numbers(table_ids) == numbers(ids) == expected, (limit, seed)
            assert table.built == directory.built, (limit, seed)
            assert [built[1] for built in table.built] == [len(chosen)]
            assert options_of(table.tests[0]) == options_of(directory.tests[0])
            assert table.tests[0]["seed"] == seed
            # The line about the test, and the note of a test that sees its
            # images more than once, are the same lines in both modes
            about_the_test = [
                line for line in of_directory_mode
                if line.startswith(("Range test for ", "Note: the test needs "))
            ]
            assert about_the_test == [
                f"Range test for DeepPrint_LocTexMinu: {len(chosen)} subjects, "
                f"{len(expected)} images, batch size 64, 300 steps from 1e-07 to 1",
                f"Note: the test needs 19,200 images and sees each of the {len(expected)} "
                "images more than once.",
            ]
            assert about_the_test == [
                line for line in of_table_mode
                if line.startswith(("Range test for ", "Note: the test needs "))
            ]
            choices.add(tuple(chosen))
    # Control: all subjects, and choices among them that depend on the seed
    assert tuple(range(GRID_SUBJECTS)) in choices and len(choices) > 5


def test_the_options_reach_a_test_from_a_table_as_they_reach_one_from_the_directories(
    grid_set, find_learning_rate, tmp_path
):
    """Every option that both modes take, with its default and with another value."""

    def both_modes(*options) -> tuple:
        directory = run_directory(
            find_learning_rate, grid_set, *grid_options(grid_set), *options,
            output=tmp_path / "directory",
        )
        table = run_table(find_learning_rate, grid_set, *options, output=tmp_path / "table")
        assert table.built == directory.built
        (test,), (table_test,) = directory.tests, table.tests
        assert options_of(table_test) == options_of(test)
        for recorded, of_the_run in ((directory, test), (table, table_test)):
            # The model and the loss that are tested are those that were built
            (extractor,) = recorded.extractors
            assert of_the_run["model"] is extractor.model and of_the_run["loss"] is extractor.loss
        # The model was built from the same random numbers: both modes seed
        # torch with --seed before they build it, and draw nothing in between
        weights = [recorded.extractors[0].model.linear.weight for recorded in (directory, table)]
        assert torch.equal(*weights)
        return table.built[0], options_of(table_test), weights[0]

    built, options, initial = both_modes()
    assert built == ("LocTexMinu", 7, 256, LossWeights())
    assert options == dict(
        min_lr=1e-7, max_lr=1.0, num_steps=300, batch_size=64, gradient_clip_norm=None,
        smoothing=0.9, num_workers=4, seed=0, localization_factor=None,
    )
    assert torch.equal(both_modes()[2], initial)

    built, options, other = both_modes(
        *("--variant", "TexMinu", "--max_subjects", 5, "--embedding_dim", 96),
        *("--loss_weight_cross_entropy", 2, "--loss_weight_center", 0.5),
        *("--loss_weight_minutia_map", 5, "--min_lr", 1e-5, "--max_lr", 0.5),
        *("--num_steps", 17, "--batch_size", 3, "--gradient_clip_norm", 2.5),
        *("--smoothing", 0.8, "--num_workers", 2, "--seed", 9),
        *("--localization_lr_factor", 0.25),
    )
    assert built == (
        "TexMinu", 5, 96, LossWeights(cross_entropy=2.0, center=0.5, minutia_map=5.0)
    )
    assert options == dict(
        min_lr=1e-5, max_lr=0.5, num_steps=17, batch_size=3, gradient_clip_norm=2.5,
        smoothing=0.8, num_workers=2, seed=9, localization_factor=0.25,
    )
    # Control: another seed and another number of classes give another model
    assert other.shape == (5, 1) and initial.shape == (7, 1)
    assert not torch.equal(both_modes("--seed", 9)[2], initial)


def test_what_the_model_is_built_from_reaches_the_constructor_of_its_variant(
    find_learning_rate, monkeypatch
):
    """build_extractor, which the other tests replace: both modes build their model with it."""

    def constructor(num_training_subjects, num_dims, loss_weights=None):
        return num_training_subjects, num_dims, loss_weights

    monkeypatch.setitem(find_learning_rate.VARIANTS, "Tex", (constructor, False))
    weights = LossWeights(cross_entropy=2.0, center=0.5)
    assert find_learning_rate.build_extractor("Tex", 5, 96, weights) == (5, 96, weights)


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
UNEVEN_SUBJECTS = [2, 7, 11, 40, 300, 1000]


def uneven_impressions() -> dict:
    """The impressions of every subject of UNEVEN, in ascending order."""
    impressions = {}
    for subject, impression in sorted(UNEVEN.values()):
        impressions.setdefault(subject, []).append(impression)
    return impressions


def uneven_rows(subjects) -> list:
    """(subject, impression) of the files of UNEVEN of these subjects, as a table orders them."""
    impressions = uneven_impressions()
    return [(subject, impression) for subject in subjects for impression in impressions[subject]]


def uneven_minutiae(number: int) -> list:
    """
    The minutiae (x, y, angle) of the file with this number among those of
    UNEVEN: one to three of them, at other places in every file.
    """
    return [
        (4.0 + 2 * number + 5 * index, 6.5 + 3 * number + 4 * index, 0.25 * (number + index + 1))
        for index in range(1 + number % 3)
    ]


def write_uneven(builder, root, *options) -> None:
    """The files of UNEVEN, their .mnt files and their sample list, and the table root/table."""
    write_small_prints(root / "images", UNEVEN)
    for number, name in enumerate(UNEVEN):
        mnt_path = root / "minutiae" / (os.path.splitext(name)[0] + ".mnt")
        write_mnt_file(mnt_path, uneven_minutiae(number), (48, 64))
    lines = ["path,subject,impression"]
    lines += [f"{path},{subject},{impression}" for path, (subject, impression) in UNEVEN.items()]
    (root / "files.csv").write_text("\n".join(lines) + "\n")
    status = build(
        builder, root, "--sample_list", root / "files.csv", "--minutiae_dir", root / "minutiae",
        *options,
    )
    assert status == 0


@pytest.fixture(scope="module")
def uneven_set(tmp_path_factory, builder):
    """The files of UNEVEN with their table, in a directory that no test changes."""
    root = tmp_path_factory.mktemp("uneven")
    write_uneven(builder, root)
    return root


def test_other_file_names_gaps_and_uneven_numbers_of_impressions_are_tested(
    find_learning_rate, uneven_set, tmp_path, monkeypatch
):
    assert sorted(uneven_impressions()) == UNEVEN_SUBJECTS
    # A random shift would move the minutiae of these small prints out of
    # them, and this test does not need it
    monkeypatch.setattr(find_learning_rate.train_500k, "POSE_AUGMENTATION", None)
    recorded = run_table(find_learning_rate, uneven_set, output=tmp_path / "out")
    (test,) = recorded.tests
    train_set = test["train_set"]

    # All rows of all subjects, in the order of the table
    assert numbers(train_set.ids) == sorted(UNEVEN.values()) and len(train_set) == 14
    # One class per subject, numbered without the gaps of the subjects
    assert recorded.built == [("LocTexMinu", 6, 256, LossWeights())]
    images, maps, labels = loaders(train_set)
    assert {i.subject: labels.get(i) for i in train_set.ids} == {
        subject: label for label, subject in enumerate(UNEVEN_SUBJECTS)
    }

    # Every sample is served: the image of its own file, and the minutiae of
    # the .mnt file of that file as the loader of .mnt files would hand them
    # out
    assert type(images._images) is SampleTableImageLoader
    assert type(maps._minutia_maps) is SampleTableMinutiaMapLoader
    served = []
    for number, (path, numbers_of_file) in enumerate(UNEVEN.items()):
        identifier = Identifier(*numbers_of_file)
        on_disk = np.array(Image.open(uneven_set / "images" / path))
        assert torch.equal(
            images._images.get(identifier), torch.from_numpy(on_disk)[None] / 255
        ), path
        in_file, angles_in_file, (width, height) = read_mnt_file(
            str(uneven_set / "minutiae" / (os.path.splitext(path)[0] + ".mnt"))
        )
        assert in_file.tolist() == [[x, y] for x, y, _ in uneven_minutiae(number)], path
        expected = transform_to_input_size(in_file, original_height=height, original_width=width)
        locations, angles = maps._minutia_maps.get_minutiae(identifier)
        assert locations.dtype == expected.dtype and np.array_equal(locations, expected), path
        assert angles.dtype == angles_in_file.dtype and np.array_equal(angles, angles_in_file)
        served.append(locations.tolist())
        image, (minutia_map, weight), label = train_set.get(identifier)
        assert image.shape == (1, INPUT_SIZE, INPUT_SIZE)
        assert weight == 1.0 and minutia_map.any()
        assert label == UNEVEN_SUBJECTS.index(numbers_of_file[0])
    # Control: they are other minutiae for every file
    assert not any(a == b for index, a in enumerate(served) for b in served[:index])


def test_max_subjects_limits_the_subjects_of_a_table_and_not_its_rows(
    find_learning_rate, uneven_set, tmp_path
):
    """
    --max_subjects on a table with gaps and uneven subjects: a sorted sample
    of the sorted subjects, drawn with --seed, and all rows of each of them.
    """

    def subjects_and_classes(*options) -> tuple:
        recorded = run_table(find_learning_rate, uneven_set, *options, output=tmp_path / "out")
        ids = numbers(recorded.tests[0]["train_set"].ids)
        subjects = sorted({subject for subject, _ in ids})
        # All rows of every subject of the test, in the order of the table
        assert ids == uneven_rows(subjects)
        ((_, classes, _, _),) = recorded.built
        return subjects, classes

    choices = set()
    for limit in (1, 2, 3, 4, 5):
        for seed in (0, 1, 2, 3):
            expected = sorted(random.Random(seed).sample(UNEVEN_SUBJECTS, limit))
            found = subjects_and_classes("--max_subjects", limit, "--seed", seed)
            assert found == (expected, limit), (limit, seed)
            choices.add((limit, len(uneven_rows(expected)), tuple(expected)))
    # Control: the seed decides which subjects are chosen, and the number of
    # rows follows from the subjects, not from their number
    assert len(choices) > 10
    assert len({rows for limit, rows, _ in choices if limit == 3}) > 1
    # The subjects are limited, not the rows: a limit that is not below the
    # number of subjects (6) takes them all, also where it is below the
    # number of rows (14)
    for limit in (0, 6, 7, 13, 14, 2000):
        assert subjects_and_classes("--max_subjects", limit, "--seed", 3) == (UNEVEN_SUBJECTS, 6)
    # The default takes all subjects of a table of this size
    assert subjects_and_classes() == (UNEVEN_SUBJECTS, 6)


def test_the_identifiers_of_the_test_are_the_objects_of_the_table(
    find_learning_rate, uneven_set, tmp_path
):
    """
    The set of the test is made of the Identifier objects that the table
    holds for its rows, as the sets of a training are: a second object per
    row would be a second copy of the identifiers of every row of the test.
    """
    for options in ([], ["--max_subjects", 4, "--seed", 1]):
        recorded = run_table(find_learning_rate, uneven_set, *options, output=tmp_path / "out")
        train_set = recorded.tests[0]["train_set"]
        table = loaders(train_set)[0]._images.table
        assert len(train_set) >= 4
        for identifier in train_set.ids:
            assert identifier is table.identifiers[table.row(identifier)]


# ---------------------------------------------------------------------------
# What a test from a table prints
# ---------------------------------------------------------------------------


def test_a_test_from_a_table_prints_the_table_and_then_what_a_test_prints(
    find_learning_rate, uneven_set, tmp_path, capsys
):
    """
    Every line of a run with a table: the table is read, the line about the
    table says how much of it the test uses, and then come the lines of a
    test without a table. Nothing is printed about files or minutiae
    directories, which a table replaces.
    """
    table, out = uneven_set / "table", tmp_path / "out"
    # Four of the six subjects, which have nine of the fourteen rows
    chosen = [2, 7, 11, 300]
    assert len(uneven_rows(chosen)) == 9
    capsys.readouterr()
    recorded = run_table(
        find_learning_rate, uneven_set, "--max_subjects", 4,
        "--seed", seed_that_chooses(UNEVEN_SUBJECTS, chosen), "--batch_size", 3,
        "--num_steps", 20, "--min_lr", 1e-5, "--max_lr", 0.1, "--num_workers", 0, output=out,
    )
    output = capsys.readouterr()
    assert len(recorded.tests) == 1 and output.err == ""
    assert output.out.splitlines() == [
        f"Reading the sample table {table}...",
        "Created IdentifierSet with 6 subjects and a total of 14 samples.",
        "Created IdentifierSet with 4 subjects and a total of 9 samples.",
        f"Sample table {table}: 14 rows of 6 subjects; the test uses 9 rows of 4 subjects",
        "Range test for DeepPrint_LocTexMinu: 4 subjects, 9 images, batch size 3, 20 steps "
        "from 1e-05 to 0.1",
        "Note: the test needs 60 images and sees each of the 9 images more than once.",
        "",
        "Ended after 2 steps at learning rate 0.001: all steps done",
        f"Loss per step: {out / 'lr_range_test.csv'}",
        "No suggestion: The range test has only 2 steps with a finite loss. Start it at a "
        "lower learning rate (--min_lr).",
    ]

    # A variant without minutiae, all subjects, and a test that does not see
    # an image twice: no note, and no line about minutiae
    capsys.readouterr()
    run_table(
        find_learning_rate, uneven_set, "--ignore_minutiae", "--batch_size", 7, "--num_steps", 2,
        output=out,
    )
    output = capsys.readouterr()
    assert output.err == "" and output.out.splitlines()[:6] == [
        f"Reading the sample table {table}...",
        "Created IdentifierSet with 6 subjects and a total of 14 samples.",
        "Created IdentifierSet with 6 subjects and a total of 14 samples.",
        f"Sample table {table}: 14 rows of 6 subjects; the test uses 14 rows of 6 subjects",
        "Range test for DeepPrint_LocTex: 6 subjects, 14 images, batch size 7, 2 steps from "
        "1e-07 to 1",
        "",
    ]
    assert output.out.splitlines()[6].startswith("Ended after 2 steps")


def test_the_default_tests_2000_subjects_and_numbers_from_1000_on_are_printed_with_separators(
    find_learning_rate, tmp_path, capsys
):
    """
    A table of 2,250 rows of 2,100 subjects. The default of --max_subjects
    takes 2,000 of the subjects, and every number of rows, images and
    subjects that a test from a table prints about itself has the separator
    of thousands. The image files of 1,000 subjects are there: a test of the
    default is refused for the other files, after it said what it would be,
    and a test of these 1,000 subjects starts.
    """
    subjects, images, table = range(2100), tmp_path / "images", tmp_path / "table"
    # The 1,000 subjects of the smaller test have no minutiae in the table
    # (null), the 1,100 others have an empty list
    of_the_smaller_test = set(chosen_by_the_rule(subjects, 1000, 0))
    small_print = save(tmp_path / "print.png", ridge_image(64, 48), 500)
    rows = [
        SampleRow(
            subject=subject, impression=impression, image_path=f"{subject}_{impression}.png",
            file_size=os.path.getsize(small_print), ppi=500.0, ppi_origin="metadata",
            ridges="dark", ridges_origin="argument", source_width=48, source_height=64,
            width=48, height=64,
            minutiae=None if subject in of_the_smaller_test else np.zeros((0, 3), np.float32),
        )
        # The first 150 subjects have two rows, the others one
        for subject in subjects
        for impression in range(2 if subject < 150 else 1)
    ]
    assert len(rows) == 2250
    write_part(str(table), 0, 1, samples_from_rows(rows), {"id": "large"})
    # The images of the smaller test are there, as copies of one small print
    images.mkdir()
    for row in rows:
        if row.subject in of_the_smaller_test:
            shutil.copyfile(small_print, images / row.image_path)

    chosen = chosen_by_the_rule(subjects, 2000, 0)
    used = len(chosen) + sum(subject < 150 for subject in chosen)
    missing = sum(row.subject in chosen and row.subject not in of_the_smaller_test for row in rows)
    assert len(chosen) == 2000 and 2000 < used < 2250 and 1000 < missing < used
    lines, message = refusal(run_table, find_learning_rate, capsys, tmp_path)
    assert lines == [
        f"Reading the sample table {table}...",
        "Created IdentifierSet with 2100 subjects and a total of 2250 samples.",
        f"Created IdentifierSet with 2000 subjects and a total of {used} samples.",
        f"Sample table {table}: 2,250 rows of 2,100 subjects; the test uses {used:,} rows of "
        "2,000 subjects",
        f"Range test for DeepPrint_LocTexMinu: 2,000 subjects, {used:,} images, batch size 64, "
        "300 steps from 1e-07 to 1",
        f"Note: the test needs 19,200 images and sees each of the {used:,} images more than "
        "once.",
    ]
    assert message.startswith(
        f"ERROR: {missing:,} of the {used:,} image files that are needed of the sample table "
        "are missing below "
    )

    # The smaller test: what it says about the minutiae that the table holds
    # for the other subjects only, as a variant with minutiae and as auto. It
    # needs 800 images, which is not more than it has.
    used = 1000 + sum(subject < 150 for subject in of_the_smaller_test)
    assert 1000 < used < 1150
    smaller = ["--max_subjects", 1000, "--batch_size", 8, "--num_steps", 100]
    elsewhere = (
        f"It holds minutiae for {2250 - used:,} of its 2,250 images, which belong to other "
        "subjects"
    )
    message = refused(
        run_table, find_learning_rate, capsys, tmp_path, *smaller, "--variant", "TexMinu"
    )
    assert message.endswith(
        f"error: --variant TexMinu needs minutiae, and the sample table {table} holds none "
        f"(null) for the {used:,} images of the 1,000 subjects that the test uses. {elsewhere}: "
        "let the test use more subjects (--max_subjects, 0 for all) or other ones (another "
        "--seed)."
    )
    capsys.readouterr()
    assert len(run_table(find_learning_rate, tmp_path, *smaller).tests) == 1
    assert printed(capsys)[3:6] == [
        f"Sample table {table}: 2,250 rows of 2,100 subjects; the test uses {used:,} rows of "
        "1,000 subjects",
        f"Range test for DeepPrint_LocTex: 1,000 subjects, {used:,} images, batch size 8, 100 "
        "steps from 1e-07 to 1",
        "Note: --variant auto is DeepPrint_LocTex here, a variant without the minutia branch: "
        f"the sample table holds no minutiae (null) for the {used:,} images of the test. "
        f"{elsewhere}. A training of the table has the minutia branch if one of them is among "
        "its training images, unless it is given --ignore_minutiae. To test that variant, let "
        "the test use more subjects (--max_subjects, 0 for all) or other ones (another --seed).",
    ]


def test_worker_processes_that_are_not_forked_get_the_warning_of_the_training(
    find_learning_rate, grid_set, tmp_path, monkeypatch, capsys
):
    """
    A worker that is not forked gets a copy of the arrays of the table. The
    range test loads its images in --num_workers processes, so it warns as
    the training does, in the words of the training.
    """
    multiprocessing = find_learning_rate.train_500k.multiprocessing

    def start(fixed, default: str, *options, runner=run_table) -> tuple:
        """
        A run where the start method is fixed (or None) and has this default.
        Returns how the script asked for the method, and the lines it printed.
        """
        asked = []
        monkeypatch.setattr(
            multiprocessing,
            "get_start_method",
            lambda allow_none=False: asked.append(allow_none) or fixed,
        )
        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: [default, "other"])
        capsys.readouterr()
        recorded = runner(find_learning_rate, grid_set, *options, output=tmp_path / "out")
        assert len(recorded.tests) == 1
        return asked, printed(capsys)

    def warnings(lines: list) -> list:
        return [line for line in lines if "WARNING" in line]

    # Forked workers share the table. The start method is asked for without
    # fixing it for the rest of the program.
    for fixed, default in (("fork", "spawn"), (None, "fork")):
        asked, lines = start(fixed, default)
        assert asked == [True] and warnings(lines) == []
    for fixed, default in (("spawn", "fork"), (None, "forkserver")):
        asked, lines = start(fixed, default)
        assert asked == [True]
        assert warnings(lines) == [
            f"WARNING: worker processes are started with '{fixed or default}' here, not "
            "forked. A forked worker shares the arrays of the sample table with the process "
            "that read it; a worker that is started in another way gets a copy of them, so "
            "every worker process of the range test holds the table once more."
        ]
        # It stands between what is said about the test and its result
        place = lines.index(warnings(lines)[0])
        assert lines[place - 2].startswith("Note: the test needs ") and lines[place - 1] == ""
        assert lines[place + 1] == "" and lines[place + 2].startswith("Ended after 2 steps")
    # A test that loads its images in its own process starts no worker, and
    # so it does not ask how one would be started: the same lines without
    # the warning and the empty line before it
    without_the_warning = lines[: place - 1] + lines[place + 1 :]
    assert start("spawn", "spawn", "--num_workers", 0) == ([], without_the_warning)
    # Without a table there is no table to copy
    asked, lines = start("spawn", "spawn", *grid_options(grid_set), runner=run_directory)
    assert asked == [] and warnings(lines) == []
    # A single worker process gets its copy of the table as each of four does
    asked, lines = start("spawn", "fork", "--num_workers", 1)
    assert asked == [True] and len(warnings(lines)) == 1


def test_a_test_that_is_refused_does_not_warn_about_its_workers(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    """The warning about workers belongs to a test that starts: it comes after the checks."""
    multiprocessing = find_learning_rate.train_500k.multiprocessing
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: "spawn")
    write_grid(tmp_path, 3, 2)
    assert build(builder, tmp_path) == 0
    # Control: the test of this table warns when it starts
    capsys.readouterr()
    assert len(run_table(find_learning_rate, tmp_path).tests) == 1
    assert sum(line.startswith("WARNING: worker processes") for line in printed(capsys)) == 1
    (tmp_path / "images" / "2_1.png").unlink()
    lines, message = refusal(run_table, find_learning_rate, capsys, tmp_path)
    assert message.startswith("ERROR: 1 of the 6 image files that are needed of the sample table")
    assert not any("WARNING" in line for line in lines)
    assert lines[-1].startswith("Note: the test needs ")


def test_a_worker_that_is_not_forked_gets_the_training_set_with_one_copy_of_the_table(
    find_learning_rate, uneven_set, tmp_path
):
    """
    A worker process that is not forked gets the training set as a copy made
    by pickle. The copy of the set of a variant with minutiae holds the table
    once, for the loader of the images and for the loader of the minutiae
    together, and it serves the samples of the original.
    """
    recorded = run_table(
        find_learning_rate, uneven_set, "--canvas_size", 80, output=tmp_path / "out"
    )
    train_set = recorded.tests[0]["train_set"]
    copy = pickle.loads(pickle.dumps(train_set))
    images, maps, _ = loaders(copy)
    assert type(images._images) is SampleTableImageLoader
    assert type(maps._minutia_maps) is SampleTableMinutiaMapLoader
    assert maps._minutia_maps.table is images._images.table
    assert images._images.table is not loaders(train_set)[0]._images.table
    assert numbers(copy.ids) == numbers(train_set.ids) == sorted(UNEVEN.values())
    assert_same_samples(fetch(copy), fetch(train_set))


def test_the_help_states_the_defaults_of_the_new_options_and_of_the_old_ones(
    find_learning_rate, monkeypatch, capsys
):
    """The three defaults that are withheld while the options are parsed are still shown."""
    # argparse wraps the help to the width of the terminal, which it takes
    # from this variable first: the lines are those of 80 columns everywhere
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(sys, "argv", ["find_learning_rate.py", "--help"])
    with pytest.raises(SystemExit) as stop:
        find_learning_rate.main()
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
    for option in ("--samples", "--tolerate_missing_minutiae", "--ignore_minutiae"):
        assert "(default: " in entries[option], option
    assert "--samples" in entries["--tolerate_missing_minutiae"]
    assert "--samples" in entries["--ignore_minutiae"]
    assert "scripts/build_sample_table.py" in entries["--samples"]
    for replaced in find_learning_rate.train_500k.OPTIONS_REPLACED_BY_SAMPLES:
        assert f"--{replaced}" in entries["--samples"], replaced
    assert "Number of subjects in the dataset (default: 50000)" in help_text
    assert "Impressions per subject (default: 10)" in help_text
    assert "dark ridges on a light background (default)" in help_text
    # What auto is with a table
    assert "with the minutia branch if --minutiae_dir is given, or if the sample table of" in (
        entries["--variant"]
    )


# ---------------------------------------------------------------------------
# The variant: what --variant auto is, and which variant needs what
# ---------------------------------------------------------------------------


def test_auto_is_the_variant_that_the_training_trains_from_the_table(
    find_learning_rate, builder, grid_set, tmp_path, monkeypatch
):
    """
    The range test is given the options of the planned training. With
    --variant auto it tests the variant that scripts/train_500k.py trains by
    default from the same table with the same options: the training is asked.
    """
    trainings = []

    def variants(root, *options, table="table") -> tuple:
        """The variant of the range test, and the one that the training trains."""
        recorded = run_table(
            find_learning_rate, root, *options, table=table, output=tmp_path / "out"
        )
        ((variant, _, _, _),) = recorded.built
        trainings.append(tmp_path / f"model {len(trainings)}")
        (trained,) = train(find_learning_rate, root, *options, table=table, output=trainings[-1])
        return variant, trained

    # A table with minutiae: the minutia branch, unless they are to be ignored
    assert variants(grid_set) == ("LocTexMinu", "LocTexMinu")
    assert variants(grid_set, "--ignore_minutiae") == ("LocTex", "LocTex")
    assert variants(grid_set, "--tolerate_missing_minutiae") == ("LocTexMinu", "LocTexMinu")

    # A table that was built without minutiae (null in every row): the
    # texture-only variant, with whatever option
    write_grid(tmp_path)
    assert build(builder, tmp_path, table="without") == 0
    for options in ([], ["--ignore_minutiae"], ["--tolerate_missing_minutiae"]):
        assert variants(tmp_path, *options, table="without") == ("LocTex", "LocTex"), options

    # A table of images without minutiae (an empty list in every row): they
    # are known, so the minutia branch is tested, as it is trained
    for name in grid(4, 2):
        path = tmp_path / "minutiae" / (name[:-4] + ".mnt")
        os.makedirs(tmp_path / "minutiae", exist_ok=True)
        path.write_text("0\n48 64\n")
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae", table="empty") == 0
    assert variants(tmp_path, table="empty") == ("LocTexMinu", "LocTexMinu")
    assert variants(tmp_path, "--ignore_minutiae", table="empty") == ("LocTex", "LocTex")

    # Minutiae for some images and null for others, which are tolerated
    write_small_minutiae(tmp_path / "minutiae", ["3_1.png"])
    (tmp_path / "minutiae" / "1_2.mnt").unlink()
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae", table="some") == 1
    tolerated = ["--tolerate_missing_minutiae"]
    assert variants(tmp_path, *tolerated, table="some") == ("LocTexMinu", "LocTexMinu")
    assert variants(tmp_path, "--ignore_minutiae", table="some") == ("LocTex", "LocTex")

    # The localization network of auto is the default of the training
    assert find_learning_rate.train_500k.DEFAULT_LOCALIZATION == "on"
    monkeypatch.setattr(find_learning_rate.train_500k, "DEFAULT_LOCALIZATION", "off")
    assert variants(grid_set) == ("TexMinu", "TexMinu")
    assert variants(grid_set, "--ignore_minutiae") == ("Tex", "Tex")


def write_minutiae_of_one_subject(builder, root) -> None:
    """
    Five subjects with two small prints each, and .mnt files for the prints
    of the last subject only: the table root/table holds null for the eight
    other rows.
    """
    write_grid(root, 5, 2)
    write_small_minutiae(root / "minutiae", ["5_1.png", "5_2.png"])
    assert build(builder, root, "--minutiae_dir", root / "minutiae") == 1
    table = SampleTable.read(str(root / "table"), str(root / "images"))
    assert table.samples.minutiae_known.tolist() == [False] * 8 + [True] * 2


def test_a_variant_with_minutiae_needs_minutiae_for_the_images_of_the_test(
    find_learning_rate, builder, tmp_path, capsys
):
    """What without a table is the error "--variant ... needs --minutiae_dir"."""
    write_minutiae_of_one_subject(builder, tmp_path)
    table = tmp_path / "table"
    # Three subjects without the one that has minutiae, and three with it
    without = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 2, 3])]
    with_it = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [1, 3, 4])]

    for variant in ("LocTexMinu", "TexMinu"):
        lines, message = refusal(
            run_table, find_learning_rate, capsys, tmp_path, "--variant", variant, *without
        )
        # The table holds minutiae, for the two images of the subject that is
        # not one of the test: the message says so, and what to do about it
        assert message.endswith(
            f"find_learning_rate.py: error: --variant {variant} needs minutiae, and the sample "
            f"table {table} holds none (null) for the 6 images of the 3 subjects that the test "
            "uses. It holds minutiae for 2 of its 10 images, which belong to other subjects: let "
            "the test use more subjects (--max_subjects, 0 for all) or other ones (another "
            "--seed)."
        )
        # Nothing was said about a test that cannot be made
        assert not any(line.startswith(("Sample table ", "Range test ")) for line in lines)
    # Rows without minutiae can be tolerated beside rows with minutiae, not
    # in their place
    message = refused(
        run_table, find_learning_rate, capsys, tmp_path, "--variant", "TexMinu", *without,
        "--tolerate_missing_minutiae",
    )
    assert "error: --variant TexMinu needs minutiae" in message

    # auto is the texture-only variant for these rows, and a variant without
    # minutiae that is given by name never needs any
    for variant, tested in (("auto", "LocTex"), ("LocTex", "LocTex"), ("Tex", "Tex")):
        recorded = run_table(find_learning_rate, tmp_path, "--variant", variant, *without)
        assert recorded.built == [(tested, 3, 256, LossWeights())]
        _, (minutia_map, weight), _ = recorded.tests[0]["train_set"].get(Identifier(0, 0))
        assert minutia_map.numel() == 0 and weight == 0.0

    # As the message says: with other subjects, or with all of them, the
    # subject that has minutiae is among those of the test. A variant with
    # minutiae is what auto is then, and the rows without minutiae stop it,
    # with the message that names the option for them
    for variant in ("auto", "LocTexMinu"):
        message = refused(
            run_table, find_learning_rate, capsys, tmp_path, "--variant", variant, *with_it
        )
        assert message.startswith(
            "ERROR: The sample table holds no minutiae (null) for 4 of the 6 training images. "
        )
        message = refused(
            run_table, find_learning_rate, capsys, tmp_path, "--variant", variant,
            "--max_subjects", 0,
        )
        assert message.startswith(
            "ERROR: The sample table holds no minutiae (null) for 8 of the 10 training images. "
        )
        assert "or pass --tolerate_missing_minutiae: " in message
        recorded = run_table(
            find_learning_rate, tmp_path, "--variant", variant, "--max_subjects", 0,
            "--tolerate_missing_minutiae",
        )
        assert recorded.built == [("LocTexMinu", 5, 256, LossWeights())]
    for variant in ("LocTex", "Tex"):
        recorded = run_table(find_learning_rate, tmp_path, "--variant", variant, *with_it)
        assert [built[0] for built in recorded.built] == [variant]

    # A table that was built without the .mnt files holds minutiae for no
    # subject: there are no other subjects to test, and the message says how
    # the table gets minutiae
    assert build(builder, tmp_path, table="without") == 0
    for choice in (without, []):
        used = "6 images of the 3 subjects" if choice else "10 images of the 5 subjects"
        message = refused(
            run_table, find_learning_rate, capsys, tmp_path, "--variant", "TexMinu", *choice,
            table="without",
        )
        assert message.endswith(
            "find_learning_rate.py: error: --variant TexMinu needs minutiae, and the sample "
            f"table {tmp_path / 'without'} holds none (null) for the {used} that the test uses. "
            "Build the table with the .mnt files of the images (--minutiae_dir of "
            "scripts/build_sample_table.py), or test a variant without minutiae: --variant "
            "auto chooses the one that train_500k.py trains by default from these rows."
        )
        # As the message says, and the training of this table trains that variant
        recorded = run_table(find_learning_rate, tmp_path, *choice, table="without")
        assert [built[0] for built in recorded.built] == ["LocTex"]
    assert train(find_learning_rate, tmp_path, table="without") == ["LocTex"]


def test_auto_without_minutiae_says_if_the_table_holds_minutiae_for_other_subjects(
    find_learning_rate, builder, tmp_path, capsys
):
    """
    The training looks at all its training images, the range test at the
    images of the subjects that it chose. Where a table holds minutiae for
    some subjects only, a choice can hold none of them: --variant auto is a
    variant without minutiae then, while the training of the table has the
    minutia branch. Such a test runs, and it says what it is not a test of.
    """
    write_minutiae_of_one_subject(builder, tmp_path)
    tolerated = ["--tolerate_missing_minutiae"]
    # Three subjects without the one that has minutiae
    without = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 2, 3])]

    def variant_and_notes(*options, table="table") -> tuple:
        """The variant of a run, which must start, and what it notes about the minutiae."""
        capsys.readouterr()
        recorded = run_table(find_learning_rate, tmp_path, *options, table=table)
        lines = printed(capsys)
        notes = [
            line for line in lines
            if "WARNING" in line or (line.startswith("Note: ") and "the test needs" not in line)
        ]
        # They stand between what is said about the test and its result
        assert lines[5].startswith("Note: the test needs ") and lines[6 : 6 + len(notes)] == notes
        assert lines[6 + len(notes)] == "" and lines[7 + len(notes)].startswith("Ended after 2")
        return recorded.built[0][0], notes

    # The training of the table has the minutia branch, with the options
    # that it needs for the eight images without minutiae
    assert train(find_learning_rate, tmp_path, *tolerated) == ["LocTexMinu"]
    note = (
        "Note: --variant auto is DeepPrint_LocTex here, a variant without the minutia branch: "
        "the sample table holds no minutiae (null) for the 6 images of the test. It holds "
        "minutiae for 2 of its 10 images, which belong to other subjects. A training of the "
        "table has the minutia branch if one of them is among its training images, unless it is "
        "given --ignore_minutiae. To test that variant, let the test use more subjects "
        "(--max_subjects, 0 for all) or other ones (another --seed)."
    )
    assert variant_and_notes(*without, *tolerated) == ("LocTex", [note])
    assert variant_and_notes(*without) == ("LocTex", [note])
    # As it says: with all subjects the test is one of the variant of the training
    variant, notes = variant_and_notes("--max_subjects", 0, *tolerated)
    assert variant == "LocTexMinu" and len(notes) == 1
    assert notes[0].startswith("Note: the sample table holds no minutiae (null) for 8 of the 10 ")

    # Nothing is said where a variant without minutiae was asked for: the
    # training is to ignore the minutiae, or the variant is given by name
    assert variant_and_notes(*without, "--ignore_minutiae") == ("LocTex", [])
    assert train(find_learning_rate, tmp_path, "--ignore_minutiae", output="ignored") == ["LocTex"]
    for variant in ("LocTex", "Tex"):
        assert variant_and_notes(*without, "--variant", variant) == (variant, [])
    # ... and where the table holds minutiae for no other subject either
    assert build(builder, tmp_path, table="without") == 0
    assert variant_and_notes(*without, table="without") == ("LocTex", [])

    # A test that is refused has not said it: the note belongs to a test that starts
    (tmp_path / "images" / "1_1.png").unlink()
    lines, message = refusal(run_table, find_learning_rate, capsys, tmp_path, *without)
    assert message.startswith("ERROR: 1 of the 6 image files that are needed of the sample table")
    assert lines[-1].startswith("Note: the test needs ")


def test_a_variant_with_minutiae_cannot_be_asked_for_together_with_ignore_minutiae(
    find_learning_rate, grid_set, tmp_path, capsys
):
    """
    With --ignore_minutiae the training trains a variant without minutiae,
    so a test of a variant with minutiae would not be a test of that
    training. The two options contradict each other, whatever the table
    holds, and the test says so before it reads the table.
    """
    for variant in ("LocTexMinu", "TexMinu"):
        lines, message = refusal(
            run_table, find_learning_rate, capsys, grid_set, "--variant", variant,
            "--ignore_minutiae", output=tmp_path / "out",
        )
        assert message.endswith(
            f"find_learning_rate.py: error: --variant {variant} has the minutia branch, and "
            "--ignore_minutiae leaves the minutiae of the sample table out: give one of the "
            "two. With --ignore_minutiae train_500k.py trains a variant without minutiae, "
            "which --variant auto chooses here."
        )
        assert lines == []
    # A variant without minutiae has nothing to ignore
    for variant in ("LocTex", "Tex"):
        recorded = run_table(
            find_learning_rate, grid_set, "--variant", variant, "--ignore_minutiae",
            output=tmp_path / "out",
        )
        assert [built[0] for built in recorded.built] == [variant]


# ---------------------------------------------------------------------------
# Rows without minutiae
# ---------------------------------------------------------------------------


def test_images_without_minutiae_stop_the_test_unless_they_are_tolerated(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path, 5, 2, minutiae=True)
    without = [(1, 0), (3, 1)]
    for name in ("2_1.mnt", "4_2.mnt"):
        (tmp_path / "minutiae" / name).unlink()
    # The builder gives the two images rows with null minutiae, and says so
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    assert numbers(table.identifiers[row] for row in table.rows_without_minutiae()) == without

    # The message of the training, which the training of this table gives too
    lines, message = refusal(run_table, find_learning_rate, capsys, tmp_path)
    assert message == (
        "ERROR: The sample table holds no minutiae (null) for 2 of the 10 training images. The "
        "first ones: '2_1.png' (subject 1, impression 0); '4_2.png' (subject 3, impression 1). "
        "A row is without minutiae if the .mnt file of its image was missing or refused when "
        "the table was built (build_failures.txt in the directory of the table says which and "
        "why). Extract the minutiae of these images and build the table again "
        "(scripts/build_sample_table.py), or pass --tolerate_missing_minutiae: these images "
        "then train without a minutia map, with weight 0 in the minutia map loss. "
        "--ignore_minutiae trains the texture-only variant instead."
    )
    # The test had said what it would be, and no more
    assert lines[-2:] == [
        "Range test for DeepPrint_LocTexMinu: 5 subjects, 10 images, batch size 64, 300 steps "
        "from 1e-07 to 1",
        "Note: the test needs 19,200 images and sees each of the 10 images more than once.",
    ]
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        train(find_learning_rate, tmp_path)
    assert stop.value.code == 2 and not exists(tmp_path / "model")
    assert " ".join(capsys.readouterr().err.split()) == message
    # A variant with minutiae that is given by name is stopped as auto is
    assert refused(
        run_table, find_learning_rate, capsys, tmp_path, "--variant", "TexMinu"
    ) == message

    # As that message says: without the minutiae nothing is missing
    capsys.readouterr()
    recorded = run_table(find_learning_rate, tmp_path, "--ignore_minutiae")
    assert [built[0] for built in recorded.built] == ["LocTex"]
    assert not any(line.startswith(("Note: the sample", "WARNING")) for line in printed(capsys))

    # With the option the test runs, and exactly these two images have the
    # weight 0 and a map of zeros
    monkeypatch.setattr(find_learning_rate.train_500k, "POSE_AUGMENTATION", None)
    capsys.readouterr()
    recorded = run_table(find_learning_rate, tmp_path, "--tolerate_missing_minutiae")
    lines = printed(capsys)
    assert recorded.built == [("LocTexMinu", 5, 256, LossWeights())]
    train_set = recorded.tests[0]["train_set"]
    assert len(train_set) == 10
    for identifier in train_set.ids:
        _, (minutia_map, weight), _ = train_set.get(identifier)
        known = (identifier.subject, identifier.impression) not in without
        assert weight == (1.0 if known else 0.0), str(identifier)
        assert minutia_map.shape == (6, 128, 128) and bool(minutia_map.any()) == known
    # The test says how many they are, after the line about the test
    note = (
        "Note: the sample table holds no minutiae (null) for 2 of the 10 images of the test. "
        "They are used without a minutia map (--tolerate_missing_minutiae), with weight 0 in "
        "the minutia map loss, as in the training."
    )
    assert lines.count(note) == 1 and not any("WARNING" in line for line in lines)
    assert lines.index(note) == 6 and lines[7] == "" and lines[8].startswith("Ended after 2")
    # Where nothing is tolerated, nothing is said: a variant without minutiae
    capsys.readouterr()
    run_table(find_learning_rate, tmp_path, "--tolerate_missing_minutiae", "--variant", "Tex")
    assert not any(line.startswith(("Note: the sample", "WARNING")) for line in printed(capsys))

    # The rule is about the rows of the test. With subjects whose rows all
    # hold minutiae, nothing has to be tolerated ...
    all_known = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 2, 4])]
    capsys.readouterr()
    recorded = run_table(find_learning_rate, tmp_path, *all_known)
    assert recorded.built == [("LocTexMinu", 3, 256, LossWeights())]
    assert not any(line.startswith(("Note: the sample", "WARNING")) for line in printed(capsys))
    # ... and with others the message counts and names the rows of the test
    one_unknown = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 3, 4])]
    message = refused(run_table, find_learning_rate, capsys, tmp_path, *one_unknown)
    assert message.startswith(
        "ERROR: The sample table holds no minutiae (null) for 1 of the 6 training images. The "
        "first ones: '4_2.png' (subject 3, impression 1). "
    )
    capsys.readouterr()
    run_table(find_learning_rate, tmp_path, *one_unknown, "--tolerate_missing_minutiae")
    assert (
        "Note: the sample table holds no minutiae (null) for 1 of the 6 images of the test. "
        "They are used without a minutia map (--tolerate_missing_minutiae), with weight 0 in "
        "the minutia map loss, as in the training." in printed(capsys)
    )


def test_a_test_in_which_no_image_has_a_minutia_says_what_it_does_not_measure(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    """
    A .mnt file without minutiae gives a row with an empty list: the image
    has none, and that is known. Rows of this kind alone still make a test
    of a variant with minutiae, as they make a training of one. But no image
    has a minutia map then, so the minutia map loss is 0 in every step, and
    the test says so, as the training does.
    """
    # Three subjects with seven images: five .mnt files that hold no
    # minutiae, and two images without a .mnt file
    names = ["1_1.png", "1_2.png", "1_3.png", "2_1.png", "2_2.png", "3_1.png", "3_2.png"]
    write_small_prints(tmp_path / "images", names)
    os.makedirs(tmp_path / "minutiae")
    for name in names:
        if name not in ("1_3.png", "3_1.png"):
            (tmp_path / "minutiae" / (name[:-4] + ".mnt")).write_text("0\n48 64\n")
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    tolerated = ["--tolerate_missing_minutiae"]

    sets = []

    def warnings(*options, table="table") -> list:
        """The warnings of a run, which must start; its training set is kept in sets."""
        capsys.readouterr()
        recorded = run_table(find_learning_rate, tmp_path, *options, table=table)
        sets.append(recorded.tests[0]["train_set"])
        return [line for line in printed(capsys) if "WARNING" in line]

    for variant in ("auto", "LocTexMinu", "TexMinu"):
        model = "LocTexMinu" if variant == "auto" else variant
        assert warnings(*tolerated, "--variant", variant) == [
            f"WARNING: DeepPrint_{model} has the minutia branch, but none of the 7 images of "
            "the test has a minutia: the sample table holds an empty list for 5 of them and "
            "null for 2. Such an image has the weight 0 in the minutia map loss, so the test "
            "measures a loss without that part. If the images have minutiae, extract them and "
            "build the table again (scripts/build_sample_table.py); --ignore_minutiae with "
            "--variant auto tests the texture-only variant instead."
        ]
    # As it says: every image of this test has the weight 0
    assert len(sets[-1]) == 7
    for identifier in sets[-1].ids:
        assert sets[-1].get(identifier)[1][1] == 0.0
    # The training of the same table warns about the same images
    capsys.readouterr()
    assert train(find_learning_rate, tmp_path, *tolerated) == ["LocTexMinu"]
    (of_the_training,) = [line for line in printed(capsys) if line.startswith("WARNING")]
    assert of_the_training.startswith(
        "WARNING: the minutia branch is trained, but no training image has a minutia: of the 7 "
        "training images, the sample table holds an empty list for 5 and null for 2. "
    )
    # A variant without minutiae has no such part of the loss
    assert warnings("--ignore_minutiae") == []
    assert warnings("--variant", "Tex") == []
    # The count is that of the rows of the test: the first two subjects
    choice = ["--max_subjects", 2, "--seed", seed_that_chooses(range(3), [0, 1])]
    (warning,) = warnings(*tolerated, *choice)
    assert "none of the 5 images of the test has a minutia" in warning
    assert "holds an empty list for 4 of them and null for 1. " in warning

    # One image with minutiae among them: the loss has that part
    write_small_minutiae(tmp_path / "minutiae", ["2_2.png"])
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae", table="one") == 1
    monkeypatch.setattr(find_learning_rate.train_500k, "POSE_AUGMENTATION", None)
    assert warnings(*tolerated, table="one") == []
    weights = {numbers([i])[0]: sets[-1].get(i)[1][1] for i in sets[-1].ids}
    assert [number for number, weight in weights.items() if weight == 1.0] == [(1, 1)]
    # ... unless that image is not one of the test
    other = ["--max_subjects", 2, "--seed", seed_that_chooses(range(3), [0, 2])]
    (warning,) = warnings(*tolerated, *other, table="one")
    assert "holds an empty list for 3 of them and null for 2. " in warning


def test_what_a_test_says_about_its_minutiae_stands_before_the_warning_about_workers(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    """
    The order of what a test from a table says between the lines about the
    test and its result, where worker processes are not forked: first what
    it says about the minutiae of its images (the note about rows without
    minutiae before the warning that no image has a minutia), and then the
    warning about its workers. A warning stands behind an empty line, a note
    does not. A test that is refused has said none of this, also where it is
    the last check that refuses it. The words of these lines have tests of
    their own.
    """
    write_minutiae_of_one_subject(builder, tmp_path)
    # A second table of the same images, built after the two .mnt files were
    # written again without minutiae: an empty list in the rows of that
    # subject and null in the eight others, so that no image has a minutia
    for name in ("5_1.mnt", "5_2.mnt"):
        (tmp_path / "minutiae" / name).write_text("0\n48 64\n")
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae", table="empty") == 1
    multiprocessing = find_learning_rate.train_500k.multiprocessing
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: "spawn")
    # Three subjects without the one that has .mnt files; and all five, with
    # the rows without minutiae tolerated
    without = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 2, 3])]
    tolerated = ["--max_subjects", 0, "--tolerate_missing_minutiae"]

    def said(*options, table="table") -> list:
        """
        The lines of a run, which must start, between 'Note: the test needs
        ...' and the empty line before 'Ended after ...': each of them up to
        its first comma or full stop.
        """
        capsys.readouterr()
        assert len(run_table(find_learning_rate, tmp_path, *options, table=table).tests) == 1
        lines = printed(capsys)
        assert lines[5].startswith("Note: the test needs ")
        assert lines[-4] == "" and lines[-3].startswith("Ended after 2 steps")
        return [re.split("[,.]", line)[0] for line in lines[6:-4]]

    auto = "Note: --variant auto is DeepPrint_LocTex here"
    null_rows = "Note: the sample table holds no minutiae (null) for 8 of the 10 images of the test"
    no_minutia = "WARNING: DeepPrint_LocTexMinu has the minutia branch"
    workers = "WARNING: worker processes are started with 'spawn' here"
    # auto as a variant without minutiae, on a table with minutiae for another subject
    assert said(*without) == [auto, "", workers]
    # Rows without minutiae that are tolerated ...
    assert said(*tolerated) == [null_rows, "", workers]
    # ... in a test in which no image has a minutia
    assert said(*tolerated, table="empty") == [null_rows, "", no_minutia, "", workers]
    # Control: a test that loads its images in its own process says nothing about workers
    assert said(*without, "--num_workers", 0) == [auto]

    # A test that is refused has said none of this, also where the last of
    # its checks refuses it: an image that changed after the tables were built
    changed = save(tmp_path / "images" / "1_1.png", ridge_image(80, 60), 500)
    for options, table in ((without, "table"), (tolerated, "empty")):
        lines, message = refusal(
            run_table, find_learning_rate, capsys, tmp_path, *options, table=table
        )
        assert message.startswith(
            f"ERROR: {changed} is not the file that its row of the sample table describes: "
        )
        assert lines[-1].startswith("Note: the test needs ")


# ---------------------------------------------------------------------------
# Tests that are refused: the options
# ---------------------------------------------------------------------------

DEFAULT = "the value of the default of the option"


@pytest.mark.parametrize(
    "option, value, says",
    [
        ("--minutiae_dir", "minutiae", "the table holds the minutiae; --ignore_minutiae trains "
                                       "without them"),
        ("--num_subjects", 4, "the table lists the images with their subject and impression"),
        ("--num_subjects", DEFAULT, "the table lists the images with their subject and impression"),
        ("--num_impressions", 2, "the table lists the images with their subject and impression"),
        ("--num_impressions", DEFAULT, "the table lists the images with their subject and "
                                       "impression"),
        ("--ppi", 500, "the table states the resolution of every file"),
        ("--ppi_manifest", "resolutions.csv", "the table states the resolution and the ridge "
                                              "polarity of every file"),
        ("--assume_ppi", 500, "the table states the resolution of every file"),
        ("--ridges", "light", "the table states the ridge polarity of every file"),
        ("--ridges", DEFAULT, "the table states the ridge polarity of every file"),
    ],
)
def test_an_option_that_the_table_replaces_is_refused_also_with_the_value_of_its_default(
    find_learning_rate, grid_set, tmp_path, capsys, option, value, says
):
    """As the training refuses it, in the words of the training."""
    train_500k = find_learning_rate.train_500k
    if value == DEFAULT:
        value = {
            "--num_subjects": train_500k.NUM_SUBJECTS,
            "--num_impressions": train_500k.NUM_IMPRESSIONS,
            "--ridges": "dark",
        }[option]
        assert (train_500k.NUM_SUBJECTS, train_500k.NUM_IMPRESSIONS) == (50000, 10)
    elif option in ("--minutiae_dir", "--ppi_manifest"):
        value = grid_set / value
    lines, message = refusal(
        run_table, find_learning_rate, capsys, grid_set, option, value, output=tmp_path / "out"
    )
    assert message.endswith(
        f"find_learning_rate.py: error: --samples cannot be combined with {option} ({says}). "
        "Leave this option out. What a sample table says is decided when it is built: to change "
        "it, build the table again (scripts/build_sample_table.py, see docs/sample-table.md)."
    )
    # The table was not read for it
    assert lines == []


def test_all_options_that_the_table_replaces_are_named_and_without_them_the_test_runs(
    find_learning_rate, grid_set, tmp_path, capsys
):
    message = refused(
        run_table, find_learning_rate, capsys, grid_set, "--ridges", "auto", "--num_subjects", 4,
        "--assume_ppi", 500, output=tmp_path / "out",
    )
    for option in ("--num_subjects (the table", "--assume_ppi (the table", "--ridges (the table"):
        assert option in message
    assert "Leave these options out." in message
    assert "--ppi (" not in message and "--minutiae_dir (" not in message
    # Control: the same command without them
    recorded = run_table(find_learning_rate, grid_set, output=tmp_path / "out")
    assert len(recorded.tests) == 1 and len(recorded.tests[0]["train_set"]) == 21
    assert sorted(find_learning_rate.train_500k.OPTIONS_REPLACED_BY_SAMPLES) == sorted(
        ["minutiae_dir", "num_subjects", "num_impressions", "ppi", "ppi_manifest"]
        + ["assume_ppi", "ridges"]
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--tolerate_missing_minutiae"],
        ["--ignore_minutiae"],
        ["--ignore_minutiae", "--tolerate_missing_minutiae"],
    ],
)
def test_the_options_of_a_table_are_refused_without_a_table(
    find_learning_rate, grid_set, tmp_path, capsys, options
):
    message = refused(
        run_directory, find_learning_rate, capsys, grid_set, *grid_options(grid_set), *options,
        output=tmp_path / "out",
    )
    # In the order in which the training names them
    given = " and ".join(
        f"--{name}" for name in ("tolerate_missing_minutiae", "ignore_minutiae")
        if f"--{name}" in options
    )
    assert message.endswith(
        f"find_learning_rate.py: error: {given} can only be given together with --samples, "
        "which reads the training set and its minutiae from a sample table. Without a table "
        "the minutiae come from the .mnt files in --minutiae_dir. Give --samples as well (see "
        f"docs/sample-table.md), or leave {given} out."
    )
    # Control: the same command without them
    recorded = run_directory(
        find_learning_rate, grid_set, *grid_options(grid_set), output=tmp_path / "out"
    )
    assert len(recorded.tests) == 1


def test_an_empty_text_in_the_place_of_the_table_is_an_error_of_the_options(
    find_learning_rate, grid_set, tmp_path, capsys
):
    """As a variable of the shell gives it that is not set: --samples "$TABLE"."""

    def run_with_an_empty_text(find_learning_rate, root, *options, output, **kwargs):
        return run(
            find_learning_rate, "--dataset_dir", root / "images", "--samples", "",
            "--output_dir", output, *options, **kwargs,
        )

    for options in ([], ["--num_subjects", 7, "--tolerate_missing_minutiae"]):
        message = refused(
            run_with_an_empty_text, find_learning_rate, capsys, grid_set, *options,
            output=tmp_path / "out",
        )
        assert message.endswith(
            "find_learning_rate.py: error: --samples was given an empty text, where the "
            "directory of a sample table belongs (a variable of the shell that is not set gives "
            "that). Give the directory of the table, or leave --samples out to find the images "
            "by the names of their files."
        )


@pytest.mark.parametrize("with_a_table", [True, False])
def test_the_directory_of_the_images_is_required_in_both_modes(
    find_learning_rate, grid_set, tmp_path, capsys, with_a_table
):
    """A table names its images below --dataset_dir: it does not replace that option."""

    def without_the_image_directory(find_learning_rate, root, *options, output, **kwargs):
        mode = ["--samples", root / "table"] if with_a_table else grid_options(root)
        return run(find_learning_rate, *mode, "--output_dir", output, *options, **kwargs)

    lines, message = refusal(
        without_the_image_directory, find_learning_rate, capsys, grid_set, output=tmp_path / "out"
    )
    assert message.endswith(
        "find_learning_rate.py: error: the following arguments are required: --dataset_dir"
    )
    assert lines == []


@pytest.mark.parametrize(
    "options, says",
    [
        (["--loss_weight_center", -1], "error: The loss weight center must be a number"),
        (["--canvas_size", -1], "error: --canvas_size must not be negative"),
    ],
)
def test_an_option_that_both_modes_refuse_is_refused_before_the_table_is_read(
    find_learning_rate, grid_set, tmp_path, capsys, options, says
):
    lines, message = refusal(
        run_table, find_learning_rate, capsys, grid_set, *options, output=tmp_path / "out"
    )
    assert says in message and lines == []
    without_a_table = refused(
        run_directory, find_learning_rate, capsys, grid_set, *grid_options(grid_set), *options,
        output=tmp_path / "out",
    )
    assert message == without_a_table


# ---------------------------------------------------------------------------
# Tests that are refused: the table and its files
# ---------------------------------------------------------------------------


def test_a_missing_image_stops_the_test_if_it_is_an_image_of_the_test(
    find_learning_rate, builder, tmp_path, capsys
):
    """The message is that of the training for the rows that are needed of a table."""
    write_uneven(builder, tmp_path)
    images = tmp_path / "images"
    (images / "roll" / "only one.png").unlink()
    # Four subjects with the subject of the missing file (nine rows), and four without it
    chosen, others = [7, 40, 300, 1000], [2, 7, 11, 300]
    with_it = ["--max_subjects", 4, "--seed", seed_that_chooses(UNEVEN_SUBJECTS, chosen)]
    without_it = ["--max_subjects", 4, "--seed", seed_that_chooses(UNEVEN_SUBJECTS, others)]
    assert len(uneven_rows(chosen)) == 9

    lines, message = refusal(run_table, find_learning_rate, capsys, tmp_path)
    assert message == (
        f"ERROR: 1 of the 14 image files that are needed of the sample table are missing below "
        f"{images} (--dataset_dir). The first ones: 'roll/only one.png' (subject 40, "
        "impression 2). Give the directory that the table was built from as --dataset_dir. If "
        "files were moved, renamed or removed since then, put them back, or build the table "
        "again from the files as they are now (scripts/build_sample_table.py)."
    )
    # The test had said what it would be, and no more
    assert lines == [
        f"Reading the sample table {tmp_path / 'table'}...",
        "Created IdentifierSet with 6 subjects and a total of 14 samples.",
        "Created IdentifierSet with 6 subjects and a total of 14 samples.",
        f"Sample table {tmp_path / 'table'}: 14 rows of 6 subjects; the test uses 14 rows of 6 "
        "subjects",
        "Range test for DeepPrint_LocTexMinu: 6 subjects, 14 images, batch size 64, 300 steps "
        "from 1e-07 to 1",
        "Note: the test needs 19,200 images and sees each of the 14 images more than once.",
    ]
    # Its number is that of the rows of the test
    message = refused(run_table, find_learning_rate, capsys, tmp_path, *with_it)
    assert message.startswith(
        "ERROR: 1 of the 9 image files that are needed of the sample table are missing below "
    )
    # The file of a subject that the test does not use is not looked for
    recorded = run_table(find_learning_rate, tmp_path, *without_it)
    assert numbers(recorded.tests[0]["train_set"].ids) == uneven_rows(others)

    # A whole directory that is gone, and more files than a message names
    shutil.rmtree(images / "plain")
    for name in ("a.png", "b.png", "c.png", "d.png"):
        (images / name).unlink()
    message = refused(run_table, find_learning_rate, capsys, tmp_path)
    assert "8 of the 14 image files that are needed of the sample table are missing" in message
    assert (
        "The first ones: 'a.png' (subject 2, impression 0); 'b.png' (subject 2, impression 1); "
        "'c.png' (subject 2, impression 5); 'plain/00002239_plain_05.png' (subject 7, "
        "impression 0); 'd.png' (subject 11, impression 0); and 3 more. " in message
    )
    # Another directory than the one of the table: every file is missing
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "images").mkdir(parents=True)
    message = refused(
        run_table, find_learning_rate, capsys, elsewhere, *with_it, table=tmp_path / "table"
    )
    assert "9 of the 9 image files" in message and f"below {elsewhere / 'images'}" in message


def test_an_image_that_is_too_small_for_the_binariser_stops_the_test_if_it_is_one_of_the_test(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    """
    The binariser raises an error for an image with fewer pixels in width or
    in height than its largest filter, when the range test loads the image.
    The table holds the size of every prepared image, so the rows are named
    before the model is built.
    """
    limit = LARGEST_BINARISER_KERNEL
    write_grid(tmp_path, 5, 2)
    # A print like the others, but stated at 600 ppi: too narrow when it is
    # prepared. And one that has just the pixels that are needed.
    write_small_prints(tmp_path / "images", ["2_2.png"], ppi=600)
    save(tmp_path / "images" / "4_1.png", ridge_image(limit, limit), 500)
    assert build(builder, tmp_path) == 0
    table = SampleTable.read(str(tmp_path / "table"), str(tmp_path / "images"))
    sizes = dict(zip(numbers(table.identifiers), zip(table.samples.width, table.samples.height)))
    assert sizes[(1, 1)] == (limit - 1, 53) and sizes[(3, 0)] == (limit, limit)
    with_it = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 1, 3])]
    without_it = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 3, 4])]

    message = refused(run_table, find_learning_rate, capsys, tmp_path)
    assert message.startswith(
        "ERROR: 1 of the 10 images that are needed of the sample table are too small for the "
        f"binarisation of the training, which needs {limit} pixels in width and in height of "
        "the image prepared at 500 ppi. The first ones: '2_2.png' (subject 1, impression 1). "
        f"For an empty image of {limit - 1}x{limit - 1} pixels the binarisation stops with: "
    )
    assert "Check the resolution of these files" in message
    assert message.endswith("(scripts/build_sample_table.py).")
    # Its number is that of the rows of the test
    message = refused(run_table, find_learning_rate, capsys, tmp_path, *with_it)
    assert message.startswith("ERROR: 1 of the 6 images that are needed of the sample table are ")

    # The image of a subject that the test does not use does not stop it, and
    # an image that has just the pixels that are needed is binarised
    recorded = run_table(find_learning_rate, tmp_path, *without_it)
    train_set = recorded.tests[0]["train_set"]
    assert numbers(train_set.ids) == [(s, i) for s in (0, 3, 4) for i in (0, 1)]
    assert train_set.get(Identifier(3, 0))[0].shape == (1, INPUT_SIZE, INPUT_SIZE)

    # Control: this is what the range test would get for the small image
    monkeypatch.setattr(
        find_learning_rate.train_500k, "check_table_image_sizes", lambda table, rows=None: None
    )
    recorded = run_table(find_learning_rate, tmp_path, *with_it)
    with pytest.raises(RuntimeError):
        recorded.tests[0]["train_set"].get(Identifier(1, 1))


def test_a_file_that_changed_after_the_table_was_built_stops_the_test_if_it_is_one_of_the_test(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    """
    Images spread over the rows of the test are loaded before the model is
    built, and the loader compares every file with its row: another
    --dataset_dir, or a file that changed, stops the test at once.
    """
    train_500k = find_learning_rate.train_500k
    names = write_grid(tmp_path, 5, 2)
    assert build(builder, tmp_path) == 0
    images = tmp_path / "images"
    with_it = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [1, 2, 4])]
    without_it = ["--max_subjects", 3, "--seed", seed_that_chooses(range(5), [0, 1, 4])]

    # One file written again as another image: the print of row 4
    assert names[4] == "3_1.png"
    changed = images / "3_1.png"
    save(changed, ridge_image(80, 60), 500)
    for options in ([], with_it):
        message = refused(run_table, find_learning_rate, capsys, tmp_path, *options)
        # The message is that of the loader of the table
        assert message.startswith(
            f"ERROR: {changed} is not the file that its row of the sample table describes: "
        )
        assert "'3_1.png' (subject 2, impression 0)" in message
        assert f"for the image directory {images}" in message
        assert "build the table again from the files as they are now" in message
    # The file of a subject that the test does not use is not loaded
    recorded = run_table(find_learning_rate, tmp_path, *without_it)
    ids = recorded.tests[0]["train_set"].ids
    assert numbers(ids) == [(subject, i) for subject in (0, 1, 4) for i in (0, 1)]

    # The images that are loaded are spread over the rows of the test. Two of
    # the six rows of the subjects 1, 2 and 4: the first and the last one.
    monkeypatch.setattr(train_500k, "IMAGES_LOADED_AT_THE_START", 2)
    loaded = []
    get = SampleTableImageLoader.get
    monkeypatch.setattr(
        SampleTableImageLoader,
        "get",
        lambda self, identifier: loaded.append(numbers([identifier])[0]) or get(self, identifier),
    )
    recorded = run_table(find_learning_rate, tmp_path, *with_it)
    assert loaded == [(1, 0), (4, 1)] and len(recorded.tests) == 1
    # The changed file is not among them, and the test meets it when it loads it
    with pytest.raises(SampleTableError, match="is not the file that its row of the sample"):
        recorded.tests[0]["train_set"].get(Identifier(2, 0))
    monkeypatch.setattr(SampleTableImageLoader, "get", get)
    monkeypatch.setattr(train_500k, "IMAGES_LOADED_AT_THE_START", 100)

    # A file that is no image any more, with as many bytes as it had
    write_small_prints(images, names)
    last = images / "5_2.png"
    last.write_bytes(b"\0" * os.path.getsize(last))
    message = refused(run_table, find_learning_rate, capsys, tmp_path, *with_it)
    assert message.startswith(f"ERROR: {last}: Could not read the file as an image: {last}. ")
    assert "the row '5_2.png' (subject 4, impression 1) of the sample table" in message
    # Control: with the files as they were when the table was built, the test runs
    write_small_prints(images, names)
    assert len(run_table(find_learning_rate, tmp_path, *with_it).tests) == 1


def test_the_rows_of_the_test_are_checked_in_the_order_of_the_training(
    find_learning_rate, builder, tmp_path, capsys
):
    """
    The start of a training from a table looks for the files, then at the
    sizes of the prepared images, then at the rows without minutiae, and then
    it loads images. The range test makes these checks in this order: a test
    whose rows have faults of several kinds is refused with the message that
    the training of such rows gives first.
    """
    images, minutiae = tmp_path / "images", tmp_path / "minutiae"
    write_grid(tmp_path, 4, 2, minutiae=True)
    # Subject 1: an image without a .mnt file
    (minutiae / "2_1.mnt").unlink()
    # A table of the files as they are now, in which no image is too small
    assert build(builder, tmp_path, "--minutiae_dir", minutiae, table="earlier") == 1
    # Subject 0: an image that is too small when it is prepared (and whose
    # .mnt file does not fit it any more: it has no minutiae in the table)
    write_small_prints(images, ["1_2.png"], ppi=600)
    assert build(builder, tmp_path, "--minutiae_dir", minutiae) == 1
    # Subject 3: a file that changed after the tables were built
    changed = save(images / "4_2.png", ridge_image(80, 60), 500)

    def of_the_training(*options, table="table") -> str:
        """What the training of a table, with all its subjects, is refused with."""
        capsys.readouterr()
        with pytest.raises(SystemExit) as stop:
            train(find_learning_rate, tmp_path, *options, table=table)
        assert stop.value.code == 2 and not exists(tmp_path / "model")
        return " ".join(capsys.readouterr().err.split())

    # The training stops for a row without minutiae before it loads a file
    # that changed (control: with such rows tolerated, it stops for one), ...
    assert of_the_training(table="earlier").startswith(
        "ERROR: The sample table holds no minutiae (null) for 1 of the 8 training images. "
    )
    assert "is not the file that its row of the sample table describes: " in of_the_training(
        "--tolerate_missing_minutiae", table="earlier"
    )
    # ... for an image that is too small before both, ...
    assert of_the_training().startswith(
        "ERROR: 1 of the 8 images of the sample table are too small for the "
    )
    # (subject 2: a file that is gone)
    (images / "3_1.png").unlink()
    # ... and for a missing file before everything else
    assert of_the_training().startswith(
        "ERROR: 1 of the 8 image files that the sample table names are missing below "
    )

    def message(*subjects) -> str:
        """What a test of these subjects is refused with."""
        choice = []
        if len(subjects) < 4:
            seed = seed_that_chooses(range(4), subjects)
            choice = ["--max_subjects", len(subjects), "--seed", seed]
        return refused(run_table, find_learning_rate, capsys, tmp_path, *choice)

    # The range test, with subjects that have fewer and fewer kinds of faults
    assert message(0, 1, 2, 3).startswith(
        "ERROR: 1 of the 8 image files that are needed of the sample table are missing below "
    )
    assert message(0, 1, 3).startswith(
        "ERROR: 1 of the 6 images that are needed of the sample table are too small for the "
    )
    assert message(1, 3).startswith(
        "ERROR: The sample table holds no minutiae (null) for 1 of the 4 training images. "
    )
    assert message(3).startswith(
        f"ERROR: {changed} is not the file that its row of the sample table describes: "
    )


def test_a_table_that_cannot_be_read_stops_the_test_with_the_reason(
    find_learning_rate, builder, tmp_path, monkeypatch, capsys
):
    write_grid(tmp_path)
    monkeypatch.setattr(builder, "FILES_PER_PART", 3)
    assert build(builder, tmp_path) == 0
    table_dir = tmp_path / "table"
    parts = sorted(os.listdir(table_dir))
    assert parts == [f"part-0000{index}-of-00003.parquet" for index in range(3)]

    # A part is missing
    os.rename(table_dir / parts[1], tmp_path / parts[1])
    lines, message = refusal(run_table, find_learning_rate, capsys, tmp_path)
    assert message.startswith(
        f"ERROR: {table_dir} is not a whole table: it holds 2 of its 3 parts; missing: {parts[1]}."
    )
    assert lines == [f"Reading the sample table {table_dir}..."]
    os.rename(tmp_path / parts[1], table_dir / parts[1])
    # Control: the table as it was built is read
    assert len(run_table(find_learning_rate, tmp_path).tests) == 1

    # A part that is no Parquet file any more
    (table_dir / parts[0]).write_bytes(b"PAR1 no table")
    message = refused(run_table, find_learning_rate, capsys, tmp_path)
    assert message.startswith(f"ERROR: {table_dir / parts[0]} cannot be read as a Parquet file")

    # A directory without a table, and one that is not there
    for name in ("empty", "no such directory"):
        if name == "empty":
            (tmp_path / name).mkdir()
        message = refused(run_table, find_learning_rate, capsys, tmp_path, table=name)
        assert message.startswith(f"ERROR: {tmp_path / name} ")
        assert "scripts/build_sample_table.py" in message

    # A table without rows
    write_part(str(tmp_path / "no rows"), 0, 1, samples_from_rows([]), {"id": "no rows"})
    message = refused(run_table, find_learning_rate, capsys, tmp_path, table="no rows")
    assert message.startswith(f"ERROR: The sample table {tmp_path / 'no rows'} holds no row")


# ---------------------------------------------------------------------------
# The range test itself, from a table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_minutiae", [True, False])
def test_a_range_test_from_a_table_runs_and_writes_its_files(
    find_learning_rate, uneven_set, tmp_path, capsys, with_minutiae
):
    """
    The range test is really run, on four of the six subjects of a table with
    other file names, gaps and uneven numbers of impressions. The model is a
    tiny one that looks at one pixel; the images and the minutia maps go
    through the loaders of the table and the pipeline of the training.
    """
    out = tmp_path / "out"
    recorded = run_table(
        find_learning_rate, uneven_set, "--max_subjects", 4, "--batch_size", 4,
        "--num_steps", 60, "--min_lr", 1e-4, "--max_lr", 1e4, "--num_workers", 0,
        *([] if with_minutiae else ["--ignore_minutiae"]), output=out, real=True,
    )
    assert recorded.built == [
        ("LocTexMinu" if with_minutiae else "LocTex", 4, 256, LossWeights())
    ]
    (extractor,), (result,) = recorded.extractors, recorded.results
    assert extractor.seen["image_shape"] == (4, 1, INPUT_SIZE, INPUT_SIZE)
    assert extractor.seen["map_shape"] == ((4, 6, 128, 128) if with_minutiae else (4, 0))
    assert extractor.seen["labels"] == {0, 1, 2, 3}  # 4 chosen subjects, one class each
    assert 10 < len(result.losses) <= 60

    with open(out / "lr_range_test.csv", newline="") as file:
        rows = [line.rstrip("\r\n").split(",") for line in file]
    assert tuple(rows[0]) == CSV_COLUMNS and len(rows) == 1 + len(result.losses)
    assert [float(row[2]) for row in rows[1:]] == result.losses
    assert np.array(Image.open(out / "lr_range_test.png")).ndim == 3
    assert sorted(os.listdir(out)) == ["lr_range_test.csv", "lr_range_test.png"]
    output = capsys.readouterr().out
    assert "Suggested: --learning_rate" in output
    assert f"Plot:          {out / 'lr_range_test.png'}" in output.splitlines()


@pytest.mark.parametrize("with_a_table", [True, False])
def test_the_report_of_a_test_names_the_variant_that_was_tested_and_smooths_as_asked(
    find_learning_rate, grid_set, tmp_path, monkeypatch, capsys, with_a_table
):
    """
    What follows the range test is the same code with and without a table:
    the file of the losses, the suggestion and the plot. It is handed the
    variant that was tested, which --variant auto does not name, and it
    smooths the losses with --smoothing. The range test is not run here:
    steps that fall and rise stand for its result.
    """
    plots = []
    save_plot = find_learning_rate.save_range_test_plot
    monkeypatch.setattr(
        find_learning_rate,
        "save_range_test_plot",
        lambda *args, **kwargs: plots.append((args, kwargs)) or save_plot(*args, **kwargs),
    )
    out = tmp_path / "out"
    options = ["--smoothing", 0.5, "--batch_size", 8]
    capsys.readouterr()
    if with_a_table:
        recorded = run_table(
            find_learning_rate, grid_set, *options, output=out, steps=FALL_AND_RISE
        )
    else:
        recorded = run_directory(
            find_learning_rate, grid_set, *grid_options(grid_set), *options, output=out,
            steps=FALL_AND_RISE,
        )
    lines = capsys.readouterr().out.splitlines()
    assert [built[0] for built in recorded.built] == ["LocTexMinu"]

    rates, losses, _, _ = FALL_AND_RISE
    suggestion = suggest_learning_rate(rates, losses, 0.5)
    # Control: with the smoothing of the default, another rate would be suggested
    assert f"{suggestion.suggested_learning_rate:.2g}" == "0.0032"
    assert f"{suggest_learning_rate(rates, losses).suggested_learning_rate:.2g}" == "0.0056"
    assert len(suggestion.explanation.splitlines()) == 2
    assert lines[-9:] == [
        "",
        "Ended after 24 steps at learning rate 0.562: loss exploded",
        f"Loss per step: {out / 'lr_range_test.csv'}",
        f"Plot:          {out / 'lr_range_test.png'}",
        *suggestion.explanation.splitlines(),
        "",
        "Suggested: --learning_rate 0.0032",
        "This value holds for this data, this model variant and this batch size only. Repeat "
        "the test when one of them changes.",
    ]
    # The plot is drawn of these steps and this suggestion, and its title
    # names the model that was tested
    ((arguments, keywords),) = plots
    assert arguments[0] == str(out / "lr_range_test.png") and arguments[1].losses == losses
    assert arguments[2:] == (suggestion, 0.5)
    assert keywords == {"title": "Learning rate range test, DeepPrint_LocTexMinu, batch size 8"}
    assert np.array(Image.open(out / "lr_range_test.png")).ndim == 3
    # The file of the losses holds them smoothed as asked
    with open(out / "lr_range_test.csv", newline="") as file:
        rows = [line.rstrip("\r\n").split(",") for line in file]
    assert [float(row[2]) for row in rows[1:]] == losses
    assert [float(row[3]) for row in rows[1:]] == smooth_losses(losses, 0.5)


# ---------------------------------------------------------------------------
# The range test and the training of the same table
# ---------------------------------------------------------------------------


def test_the_range_test_stops_for_the_file_for_which_the_training_of_the_table_stops(
    find_learning_rate, builder, tmp_path, capsys
):
    """
    The files of the rows are checked by the functions of the training. With
    all subjects in the test and in the training set, a file that changed
    stops both scripts with the same message.
    """
    write_grid(tmp_path, minutiae=True)
    assert build(builder, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    # Control: both start
    assert len(run_table(find_learning_rate, tmp_path).tests) == 1
    assert train(find_learning_rate, tmp_path, output="started") == ["LocTexMinu"]

    save(tmp_path / "images" / "2_2.png", ridge_image(80, 60), 500)
    message = refused(run_table, find_learning_rate, capsys, tmp_path)
    assert message.startswith(
        f"ERROR: {tmp_path / 'images' / '2_2.png'} is not the file that its row of the sample "
        "table describes: "
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        train(find_learning_rate, tmp_path)
    assert stop.value.code == 2 and not exists(tmp_path / "model")
    assert " ".join(capsys.readouterr().err.split()) == message


# ---------------------------------------------------------------------------
# Without a table: what the script did before it knew sample tables
# ---------------------------------------------------------------------------


def test_a_test_without_a_table_prints_what_it_printed_before_there_were_tables(
    find_learning_rate, tmp_path, monkeypatch, capsys
):
    """
    Every line of a run without --samples, from its first line to its last,
    as the script printed it before it knew sample tables: a table adds no
    line where none is used, and takes none away. The range test is really
    run, with the tiny model; then a variant without minutiae, with two steps
    in the place of the range test.
    """
    images, minutiae, out = tmp_path / "images", tmp_path / "minutiae", tmp_path / "out"
    names = write_grid(tmp_path, 6, 2, minutiae=True)
    assert len(names) == 12
    grid_of_the_files = ["--num_subjects", 6, "--num_impressions", 2, "--max_subjects", 4]
    set_of_all_files = "Created IdentifierSet with 6 subjects and a total of 12 samples."
    # The range test draws a progress bar on standard error, in a form that
    # settings of the environment change (TQDM_...). It is switched off here,
    # so that everything else that a run writes there shows
    monkeypatch.setattr(tqdm, "tqdm", functools.partial(tqdm.tqdm, disable=True))
    capsys.readouterr()
    recorded = run_directory(
        find_learning_rate, tmp_path, "--minutiae_dir", minutiae, *grid_of_the_files,
        "--batch_size", 4, "--num_steps", 60, "--min_lr", 1e-4, "--max_lr", 1e4,
        "--num_workers", 0, real=True,
    )
    output = capsys.readouterr()
    (result,) = recorded.results
    suggestion = suggest_learning_rate(result.learning_rates, result.losses, 0.9)
    assert suggestion.suggested_learning_rate is not None
    assert output.out.splitlines() == [
        "Created IdentifierSet with 4 subjects and a total of 8 samples.",
        "Range test for DeepPrint_LocTexMinu: 4 subjects, 8 images, batch size 4, 60 steps "
        "from 0.0001 to 10000",
        "Note: the test needs 240 images and sees each of the 8 images more than once.",
        f"Scanning {images} for .png files...",
        set_of_all_files,
        set_of_all_files,
        f"Loading minutiae from {minutiae}",
        set_of_all_files,
        "",
        f"Ended after {len(result.losses)} steps at learning rate "
        f"{result.learning_rates[-1]:.3g}: {result.stop_reason}",
        f"Loss per step: {out / 'lr_range_test.csv'}",
        f"Plot:          {out / 'lr_range_test.png'}",
        *suggestion.explanation.splitlines(),
        "",
        f"Suggested: --learning_rate {suggestion.suggested_learning_rate:.2g}",
        "This value holds for this data, this model variant and this batch size only. Repeat "
        "the test when one of them changes.",
    ]
    assert len(suggestion.explanation.splitlines()) == 2
    # Standard error holds nothing but the progress bar, which is off here
    assert output.err == ""

    # Without --minutiae_dir, and with defaults for what the command above
    # gave: the variant without minutiae
    capsys.readouterr()
    recorded = run_directory(find_learning_rate, tmp_path, *grid_of_the_files, output="other")
    output = capsys.readouterr()
    assert len(recorded.tests) == 1 and output.err == ""
    assert output.out.splitlines() == [
        "Created IdentifierSet with 4 subjects and a total of 8 samples.",
        "Range test for DeepPrint_LocTex: 4 subjects, 8 images, batch size 64, 300 steps from "
        "1e-07 to 1",
        "Note: the test needs 19,200 images and sees each of the 8 images more than once.",
        f"Scanning {images} for .png files...",
        set_of_all_files,
        "No minutiae directory provided - using texture-only mode",
        "",
        "Ended after 2 steps at learning rate 0.001: all steps done",
        f"Loss per step: {tmp_path / 'other' / 'lr_range_test.csv'}",
        "No suggestion: The range test has only 2 steps with a finite loss. Start it at a "
        "lower learning rate (--min_lr).",
    ]


def test_a_refusal_without_a_table_prints_what_it_printed_before_there_were_tables(
    find_learning_rate, tmp_path, capsys
):
    """Every line of a run without --samples that is refused for the files it does not find."""
    images, out = tmp_path / "images", tmp_path / "out"
    write_grid(tmp_path, 6, 2)
    capsys.readouterr()
    recorded = Recorded()
    with pytest.raises(SystemExit) as stop:
        run_directory(
            find_learning_rate, tmp_path, "--num_subjects", 7, "--num_impressions", 2,
            "--max_subjects", 0, recorded=recorded,
        )
    output = capsys.readouterr()
    assert stop.value.code == 2 and recorded.built == [] and recorded.tests == []
    assert output.out.splitlines() == [
        "Created IdentifierSet with 7 subjects and a total of 14 samples.",
        "Range test for DeepPrint_LocTex: 7 subjects, 14 images, batch size 64, 300 steps from "
        "1e-07 to 1",
        "Note: the test needs 19,200 images and sees each of the 14 images more than once.",
        f"Scanning {images} for .png files...",
        "Created IdentifierSet with 6 subjects and a total of 12 samples.",
    ]
    message = (
        f"2 of the 14 files that are needed are missing below {images} (--dataset_dir). The "
        "first ones: 7_1.png, 7_2.png. --num_subjects 7 and --num_impressions 2 mean that every "
        "file <subject>_<impression>.png with subject 1 to 7 and impression 1 to 2 must exist, "
        "directly in the directory or in a subdirectory; 12 .png files of 6 subjects were found "
        "(subject numbers 1 to 6, impression numbers 1 to 2). Set the two options to what the "
        "directory holds. Subjects must be numbered without gaps and each needs all its "
        "impressions; see the docstring of scripts/train_500k.py for the layout."
    )
    # The sentence that this message of the training script gained with its table mode
    message += (
        " Files with other names, with gaps in their numbers or with uneven numbers of "
        "impressions can be described in a sample table and trained with --samples, see "
        "docs/sample-table.md."
    )
    assert output.err == f"ERROR: {message}\n"
    # Without a table the directory of the output is made before the files
    # are looked for, as it always was
    assert os.listdir(out) == []


def test_without_a_table_a_variant_with_minutiae_needs_the_directory_of_the_minutiae(
    find_learning_rate, grid_set, tmp_path, capsys
):
    """The refusal that a table has other words for is the one it was, in its words."""
    for variant in ("LocTexMinu", "TexMinu"):
        lines, message = refusal(
            run_directory, find_learning_rate, capsys, grid_set, "--variant", variant,
            "--num_subjects", 7, "--num_impressions", 3, output=tmp_path / "out",
        )
        assert message.endswith(
            f"find_learning_rate.py: error: --variant {variant} needs --minutiae_dir"
        )
        assert lines == []


@pytest.mark.parametrize(
    "given, expected",
    [
        ([], (50000, 10, "dark")),
        (["--num_subjects", 7, "--num_impressions", 3, "--ridges", "light"], (7, 3, "light")),
        (["--num_subjects", 50000, "--num_impressions", 10, "--ridges", "dark"],
         (50000, 10, "dark")),
        (["--num_impressions", 3], (50000, 3, "dark")),
        (["--ridges", "auto", "--num_subjects", 9], (9, 10, "auto")),
    ],
)
def test_directory_mode_gets_the_defaults_that_are_withheld_while_the_options_are_parsed(
    find_learning_rate, tmp_path, monkeypatch, capsys, given, expected
):
    """
    --num_subjects, --num_impressions and --ridges have defaults, and a table
    replaces all three. Their defaults are withheld while the options are
    parsed, so that table mode can tell whether one was given, and without a
    table they are put back before anything reads them: not given, given
    with another value, and given with the value of the default.
    """
    train_500k = find_learning_rate.train_500k
    assert (train_500k.NUM_SUBJECTS, train_500k.NUM_IMPRESSIONS) == (50000, 10)
    asked = []

    def find_dataset_files(
        dataset_dir, minutiae_dir, normalizer, image_ids, minutiae_ids, num_subjects,
        num_impressions,
    ):
        """In the place of the check of the files: keeps what it is asked for, and finds none."""
        ridges = normalizer.settings()["ridges"]
        asked.append((dataset_dir, minutiae_dir, num_subjects, num_impressions, ridges))
        asked.append((image_ids, minutiae_ids))
        raise train_500k.DatasetLayoutError("no file was looked for")

    monkeypatch.setattr(train_500k, "find_dataset_files", find_dataset_files)
    (tmp_path / "images").mkdir()
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        run_directory(find_learning_rate, tmp_path, "--max_subjects", 5, "--seed", 3, *given)
    output = capsys.readouterr()
    assert stop.value.code == 2 and output.err == "ERROR: no file was looked for\n"
    num_subjects, num_impressions, ridges = expected
    options, (image_ids, minutiae_ids) = asked
    assert options == (str(tmp_path / "images"), None, num_subjects, num_impressions, ridges)
    # The identifiers are those of these numbers: every impression of five
    # of the subjects
    chosen = chosen_by_the_rule(range(num_subjects), 5, 3)
    assert numbers(image_ids) == [(s, i) for s in chosen for i in range(num_impressions)]
    assert numbers(minutiae_ids) == numbers(image_ids)
    assert output.out.splitlines()[1] == (
        f"Range test for DeepPrint_LocTex: 5 subjects, {5 * num_impressions} images, batch "
        "size 64, 300 steps from 1e-07 to 1"
    )


def test_every_beginning_of_an_option_that_selected_it_before_tables_still_selects_it(
    find_learning_rate, capsys
):
    """
    argparse takes every beginning of an option that no other option begins
    with. So a new option whose name begins like an old one takes beginnings
    away from it: the option for rows without minutiae had a name that began
    with --allow_ at first, and with it --a would have been refused as
    ambiguous here, which selects --assume_ppi. Every beginning of an option
    that the script had before it knew sample tables, and that selected that
    option then, is put through the parser of main() here.
    """
    import argparse

    class Taken(Exception):
        pass

    def take(parser, *args, **kwargs):
        raise Taken(parser)

    # The parser as main() builds it, taken where main() would use it
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", take)
        with pytest.raises(Taken) as taken:
            find_learning_rate.main()
    (parser,) = taken.value.args
    actions = {
        option: action
        for option, action in parser._option_string_actions.items()
        if option.startswith("--")
    }
    of_tables = {"--samples"} | {
        f"--{name}" for name in find_learning_rate.train_500k.OPTIONS_OF_SAMPLES
    }
    assert of_tables == {"--samples", "--tolerate_missing_minutiae", "--ignore_minutiae"}
    before = sorted(set(actions) - of_tables)
    assert of_tables < set(actions) and "--assume_ppi" in before
    # An option that is added to the script later is not one of before: leave
    # it out of `before` like the options of tables, so that this test says
    # whether its name takes a beginning away from one of these
    assert len(before) == 26, f"The script had 26 options before tables, found: {before}"

    selected = {}
    for option in before:
        action = actions[option]
        for length in range(len("--") + 1, len(option) + 1):
            beginning = option[:length]
            others = [other for other in before if other.startswith(beginning) and other != option]
            if beginning in others or (others and beginning != option):
                continue  # it was another option, or ambiguous
            arguments = ["--dataset_dir", "images", "--output_dir", "out", beginning]
            if action.nargs != 0:
                # A value that is not the default of the option
                arguments.append(str(action.choices[-1]) if action.choices else "7")
            try:
                args = parser.parse_args(arguments)
            except SystemExit as stop:
                # --help ends the program with 0, an error of the options with 2
                assert option == "--help" and stop.code == 0, f"{beginning} for {option}"
            else:
                value = getattr(args, action.dest)
                assert value is not None and value != action.default, f"{beginning} for {option}"
            selected[beginning] = option
    capsys.readouterr()
    # Among them the beginnings that the first name of the new option would
    # have taken away
    for beginning in ("--a", "--as", "--ass", "--assume", "--assume_"):
        assert selected[beginning] == "--assume_ppi"
    assert len(selected) > 150 and set(before) <= set(selected)
    # Control: such a beginning is refused when two options begin with it
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["--dataset_dir", "images", "--output_dir", "out", "--num_s", "7"])
    assert stop.value.code == 2
    assert "ambiguous option: --num_s could match" in capsys.readouterr().err
