"""
Training script for DeepPrint on a large fingerprint dataset, with or without
learned alignment (LocalizationNetwork). The name is historical: the script
was first written for 500,000 images. docs/training-guide.md walks
through its use; the section "Above about one million subjects" of
docs/large-datasets.md says what it cannot do today.

One GPU (or the CPU):
    python scripts/train_500k.py \
        --dataset_dir /path/to/fingerprints \
        --minutiae_dir /path/to/minutiae \
        --num_subjects 50000 --num_impressions 10 \
        --ppi_manifest /path/to/resolutions.csv \
        --output_dir /path/to/model

Several GPUs: the same options after
    torchrun --nproc_per_node=NUM_GPUS scripts/train_500k.py ...

Without --minutiae_dir the texture-only variant is trained.

Layout of the files:
    --dataset_dir holds the images as <subject>_<impression>.png, directly
    or in any subdirectories (for example one per sensor). Subjects are
    numbered 1 .. --num_subjects without gaps, impressions
    1 .. --num_impressions, and every subject needs all its impressions:

        dataset_dir/
            sensor_a/1_1.png        subject 1, impression 1
            sensor_a/1_2.png        subject 1, impression 2
            sensor_b/2_1.png
            ...
            50000_10.png            subject 50000, impression 10

    A "subject" is one finger: two fingers of one person are two subjects.
    --minutiae_dir holds one .mnt file per image under the same name
    (1_1.mnt, ...), as scripts/extract_minutiae.py writes them.

    The script checks this before it trains and stops with a message that
    names the first missing files. Two files with the same name in different
    subdirectories are an error. Image files with a subject number above
    --num_subjects or an impression number above --num_impressions are not
    used; the script says how many. For other names, other image formats or
    an uneven number of impressions per subject, describe the files in a
    sample table (see below), rename (or link) the files to this layout, or
    change CustomFingerprintLoader and the identifier lists in this script.

Other file names, gaps and uneven sets (a sample table):
    --samples TABLE             read the training set from a sample table
                                instead of from the names of the files

    A sample table is a directory of Parquet files with one row per image
    file. A row says which finger (subject) and which capture (impression)
    the file shows, at which resolution and with which ridge polarity it is
    to be prepared, and which minutiae the prepared image has. So the files
    can have any names and be in any of the formats that the builder of the
    table reads, the subject numbers can have gaps, and a subject can have
    any number of impressions. The pixels stay in the image files:
    --dataset_dir is their directory with a table as without one. The table
    is built once, after the minutiae extraction, and then read by the range
    test and by the training:

        python scripts/build_sample_table.py \
            --dataset_dir /path/to/fingerprints \
            --minutiae_dir /path/to/minutiae \
            --ppi_manifest /path/to/resolutions.csv \
            --samples /path/to/table
        python scripts/find_learning_rate.py \
            --dataset_dir /path/to/fingerprints --samples /path/to/table \
            --output_dir /path/to/range_test
        python scripts/train_500k.py \
            --dataset_dir /path/to/fingerprints --samples /path/to/table \
            --output_dir /path/to/model

    The table says what these options say without it, so together with
    --samples they are refused: --minutiae_dir, --num_subjects,
    --num_impressions, --ppi, --ppi_manifest, --assume_ppi and --ridges.
    They are given to scripts/build_sample_table.py when the table is built.

    The minutia branch is trained if the table holds minutiae for training
    images. A training image for which it holds none (the .mnt file was
    missing or refused when the table was built) stops the run:

    --allow_missing_minutiae    train such images without a minutia map
                                (weight 0 in the minutia map loss) instead
    --ignore_minutiae           train the texture-only variant although the
                                table holds minutiae

    What else differs from a training without a table: only a subject with
    at least two impressions can be held out for validation, so a subject
    with one always trains. Images are drawn evenly, not subjects: a finger
    with many images is trained more often, and in validation all pairs of
    its images count. The script prints the number of impressions per
    training subject and what the validation set holds, and it warns when
    that is large. Every process lists the directories of the images and
    stops if files of the table are missing; the first one loads 100 images
    spread over the table before it builds the model, so that another
    --dataset_dir, or files that changed after the table was built, stop
    the run at once. With several GPUs only the first process prints what
    the start prints.

    A model that was trained from a table is loaded with
    load_trained_model(output_dir, dataset_dir, samples=TABLE), where TABLE
    describes the images to embed.

    A sample table needs the package pyarrow (pip install -e ".[table]");
    without --samples the script does not need it. docs/sample-table.md
    describes the table and how it is built and checked.

Continuing and starting again:
    The script writes model.pyt (the newest epoch), best_model.pyt, log.json
    and training_settings.json to --output_dir. If model.pyt is there, the
    same command CONTINUES that training with the next epoch. To start
    again from nothing, use a new --output_dir (or delete the old one).

    A continued training must be the same training. The script stops if the
    model variant (--localization, --minutiae_dir), --embedding_dim, the
    subjects and the validation split, --batch_size or the image preparation
    differ from training_settings.json. Learning rate options, loss weights,
    --num_epochs and --patience may change; the script lists the changes and
    they hold from the next epoch on. If all epochs are already trained, or
    the training ended by early stopping, nothing is trained and the script
    says so: a changed learning rate has no effect on a finished training.

    With --samples the sample table stands in this comparison where the
    image preparation and the numbers of subjects and impressions stand
    without it: a training continues only with a table that holds the same
    rows. Their checksum is stored; where the table lies and how its files
    are split does not count. --ignore_minutiae changes the model variant.
    A training that was started without a table does not continue with one,
    and one that was started with a table not without it.

    training_settings.json holds the settings of the last run that trained
    an epoch. A start that is refused (settings, missing files) or that
    trains nothing does not write it. A first start that passes these checks
    and then fails in its first step (for example on a file whose resolution
    is unknown) leaves the settings files and no model.pyt.

    --batch_size 64                 images per step on each GPU (default 64)
    --allow_changed_settings        continue although settings that must stay
                                    the same differ (at your own risk)

Images from different sensors (see Step 2 of docs/training-guide.md):
    Every image is loaded like scripts/extract_minutiae.py loads it: resampled
    to 500 pixels per inch (ppi) and brought to dark ridges. The minutiae in
    the .mnt files are in the pixels of that image. Give this script the same
    options as the extraction, unless it reads a sample table (--samples):
    the table states the resolution and the ridge polarity of every file,
    and the options are refused.

    --ppi 1000                  all files have this resolution
    --ppi_manifest FILE         CSV with the columns path,ppi[,ridges] for
                                files or directories, paths relative to
                                --dataset_dir
    --assume_ppi 500            for files whose resolution is not known from
                                the manifest or the metadata of the file;
                                without it such a file stops the training
    --ridges dark|light|auto    ridge polarity of the files (default: dark)

    Images that are at 500 ppi with dark ridges are used unchanged.

    --canvas_size 512           cut or pad every 500 ppi image around its
                                centre to a square of this many pixels before
                                it is scaled to the input size of the model.
                                The model then sees every sensor at the same
                                scale. Default 0: every image is padded to a
                                square of its own longer side, so a small
                                sensor image is shown larger than a large one.

    The settings are written to data_preparation.json in --output_dir. Images
    must be prepared in the same way when the model is used later.

Learned alignment (see "Learned alignment" in docs/tuning.md):
    The localization network is a small network in front of the model. It
    proposes a rotation and a shift for every image, and the image is
    resampled with them before the model sees it, so that the model sees every
    print in a similar pose. It is trained together with the model.

    --localization on               train with the localization network
                                    (DeepPrint_LocTexMinu with --minutiae_dir,
                                    DeepPrint_LocTex without); the default
    --localization off              train without it (DeepPrint_TexMinu or
                                    DeepPrint_Tex); everything else is the same
    --localization_lr_factor 0.035  the localization network uses the learning
                                    rate times this factor (default:
                                    LOCALIZATION_LEARNING_RATE_FACTOR in
                                    flx/setup/config.py)

    Whether the localization network makes matching more accurate on your
    data is not known: train both ways with the same settings and compare the
    validation equal error rates (docs/tuning.md, "How to find out whether it
    helps on your data", has the recipe). A model must be
    loaded with the variant it was trained with.

Validation (on by default):
    A share of the SUBJECTS (not of the images) is held out of the training set
    and used as validation set. All impressions of a held-out subject go to the
    validation set, so the model is validated on fingers it has never trained
    on. After every epoch the equal error rate on the validation set is
    measured (the error rate at the decision threshold where false matches and
    false non-matches are equally frequent; lower is better). The epoch with
    the lowest value is saved as best_model.pyt, and training stops after
    --patience epochs without a lower value.

    --validation_split 0.1          share of the subjects to hold out
                                    (default 0.1; 0 switches validation off)
    --max_validation_subjects 5000  upper limit for the number of held-out
                                    subjects (default 5000; 0 = no limit).
                                    Validation runs after every epoch on all
                                    impressions of the held-out subjects, and
                                    its pair list is built in a time that
                                    grows with the square of their number.
                                    Every subject above the limit stays in
                                    the training set.
    --validation_seed 42            seed of the random choice of the held-out
                                    subjects (default 42). The same seed, share,
                                    limit and number of subjects always give
                                    the same split, also when a training is
                                    resumed.

    With --validation_split 0 there is no validation set: best_model.pyt is
    then the epoch with the lowest training loss, which cannot detect
    overfitting, and early stopping is off.

    --patience 3                    stop after this many epochs without a
                                    lower validation error rate (default 3;
                                    0 = never stop early)

    These options are added to the command at the top, for example
        --validation_split 0.1 --max_validation_subjects 5000 --patience 3

Learning rate (see "Choosing the learning rate" in docs/tuning.md):
    The learning rate is the size of the change that the optimizer makes to
    the model in every step (one batch). The right value depends on the data,
    the model variant and the batch size. Find it with
    scripts/find_learning_rate.py before a long training.

    --learning_rate 0.0001          learning rate after the warm-up
    --warmup_steps 500              number of first steps over which the rate
                                    rises in a straight line to --learning_rate
                                    (0 = no warm-up)
    --lr_schedule cosine            "constant" keeps the rate after the warm-up,
                                    "cosine" lowers it smoothly to 1 % of
                                    --learning_rate in the last step of the
                                    last epoch
    --gradient_clip_norm 0          upper limit for the length of the gradient
                                    in one step (0 = no limit, the default).
                                    Take a value from your own range test:
                                    gradient norms of 10,000 and more were
                                    measured, so a small limit such as 5
                                    would shrink every step

    Options that are not given use the values in flx/setup/config.py. The
    training stops with a message when the loss is not a finite number or far
    above the loss at the start.
"""

import os
import sys
import argparse
import contextlib
import functools
import io
import json
import multiprocessing
import random

import numpy as np
import torch

from flx.data.dataset import Dataset, Identifier, IdentifierSet
from flx.benchmarks.verification import VerificationBenchmark
from flx.benchmarks.biometric_comparison import BiometricComparison
from flx.data.image_loader import ImageLoader
from flx.data.label_index import LabelIndex
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.data.fingerflow_minutia_loader import FingerFlowMinutiaMapLoader
from flx.data.minutia_map_loader import PoseTransformedMinutiaMapLoader
from flx.data.dataset import ConstantDataLoader
from flx.data.sample_table_loader import (
    SampleTable,
    SampleTableImageLoader,
    SampleTableMinutiaMapLoader,
)
from flx.extractor.fixed_length_extractor import (
    get_DeepPrint_LocTexMinu,
    get_DeepPrint_LocTex,
    get_DeepPrint_TexMinu,
    get_DeepPrint_Tex,
)
from flx.image_processing.augmentation import RandomPoseTransform, RandomQualityTransform
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.models.learning_rate import SCHEDULES, LearningRateConfig
from flx.models.model_training import TrainingLog
from flx.models import torch_helpers
from flx.setup.paths import get_best_model_file, get_newest_model_file
from flx.models.deep_print_loss import LossWeights
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.preparation import sensor_normalization
from flx.preparation.sample_table import PPI_ORIGINS, RIDGES_ORIGINS, SampleTableError
from flx.preparation.sensor_normalization import SensorNormalizer


# =============================================================================
# CONFIGURATION - Adjust these for your dataset
# =============================================================================

# Dataset structure
NUM_SUBJECTS = 50000          # Number of unique fingerprint subjects
NUM_IMPRESSIONS = 10          # Impressions per subject (50000 * 10 = 500K)

# Training hyperparameters
NUM_EPOCHS = 10               # Upper limit; how many epochs a large set needs was not measured
EMBEDDING_DIM = 256           # Values per branch (512 per print for variants with minutiae)

# Learned alignment: "on" trains with the localization network, "off" without
DEFAULT_LOCALIZATION = "on"

# Image preprocessing
# Despite its name this is not a threshold: it is the ridge width in pixels
# that the binarisation (LazilyAllocatedBinarizer, a Gabor filter) assumes
BINARIZATION_THRESHOLD = 5.0

# Data augmentation: random rotation and shift of the training images
POSE_AUGMENTATION = RandomPoseTransform(
    pad=0,
    angle_min=-30,            # Random rotation range (degrees)
    angle_max=30,
    shift_horizontal_min=-40,
    shift_horizontal_max=40,
    shift_vertical_min=-40,
    shift_vertical_max=40,
)

QUALITY_AUGMENTATION = RandomQualityTransform(
    contrast_min=1.0,
    contrast_max=1.5,
    gain_min=0.95,
    gain_max=1.05,
)


# =============================================================================
# CUSTOM DATA LOADER - Modify this for your file naming convention
# =============================================================================

class CustomFingerprintLoader(ImageLoader):
    """
    Custom loader for your fingerprint dataset.

    Modify _extension() and _file_to_id_fun() to match your file naming pattern.

    Expected directory structure:
        dataset_dir/
            1_1.png      # subject 1, impression 1
            1_2.png      # subject 1, impression 2
            ...
            50000_10.png # subject 50000, impression 10

    With a SensorNormalizer the image files are loaded like
    scripts/extract_minutiae.py loads them: resampled to 500 pixels per inch
    and brought to dark ridges (see flx/preparation/sensor_normalization.py).
    The minutiae of the .mnt files are in the pixels of that image. Without
    one the files are loaded as they are.
    """

    def __init__(self, root_dir: str, normalizer: SensorNormalizer | None = None):
        super().__init__(root_dir)
        self._normalizer = normalizer

    def get(self, identifier: Identifier) -> torch.Tensor:
        filepath = self._files.get(identifier)
        if self._normalizer is None:
            return self._load_image(filepath)
        import torchvision.transforms.functional as VTF
        try:
            img, _ = self._normalizer.load(filepath)
        except ValueError as error:
            # Name the file: the message of the preparation does not
            raise type(error)(f"{filepath}: {error}") from error
        return VTF.to_tensor(img)

    @staticmethod
    def _extension() -> str:
        # Change this to match your image format (.png, .bmp, .tif, etc.)
        return ".png"

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        # Modify this to parse your filename pattern
        # Example pattern: "<subject_id>_<impression_id>.png"
        try:
            subject_id, impression_id = filename.split("_")
            # Convert to 0-indexed for PyTorch compatibility
            identifier = Identifier(int(subject_id) - 1, int(impression_id) - 1)
        except ValueError:
            identifier = None
        if identifier is None or identifier.subject < 0 or identifier.impression < 0:
            raise ValueError(
                f"The image file {os.path.join(subdir, filename)}"
                f"{CustomFingerprintLoader._extension()} is not named "
                "<subject>_<impression> with whole numbers from 1 (for example "
                "17_3.png). Rename the files, move files that are no training "
                "images out of --dataset_dir, or change "
                "CustomFingerprintLoader in scripts/train_500k.py."
                " Files with other names, with gaps in their numbers or with "
                "uneven numbers of impressions can also be described in a "
                "sample table and trained with --samples, see "
                "docs/sample-table.md."
            )
        return identifier

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        import cv2
        import torchvision.transforms.functional as VTF
        img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        return VTF.to_tensor(img)


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

# Default seed for the choice of the validation subjects and of the pairs
# that are compared in validation
VALIDATION_SEED = 42

# The validation benchmark needs mated pairs (two impressions of one subject)
# and non-mated pairs (impressions of two subjects)
MIN_VALIDATION_SUBJECTS = 2

# Default of --max_validation_subjects. Validation embeds all impressions of
# the held-out subjects after every epoch, and create_validation_benchmark
# needs a time that grows with the square of their number.
DEFAULT_MAX_VALIDATION_SUBJECTS = 5000

# Above this number of training subjects the script prints what grows with
# the number of subjects (see "Above about one million subjects" in
# docs/large-datasets.md)
MANY_SUBJECTS = 1_000_000

# With a sample table (--samples) the script warns above this many validation
# images or validation pairs. They are what DEFAULT_MAX_VALIDATION_SUBJECTS
# stands for where every subject has 10 impressions: 50,000 images, and
# 225,000 mated and 50,000 non-mated pairs. The subjects of a table can have
# any number of images, so their number alone does not limit either.
MANY_VALIDATION_IMAGES = 50_000
MANY_VALIDATION_PAIRS = 275_000

# With a sample table this many image files, spread over the table, are
# loaded before the model is built
IMAGES_LOADED_AT_THE_START = 100

# The options that a sample table replaces, each with what the table says in
# its place. Together with --samples they are refused.
OPTIONS_REPLACED_BY_SAMPLES = {
    "minutiae_dir": "the table holds the minutiae; --ignore_minutiae trains without them",
    "num_subjects": "the table lists the images with their subject and impression",
    "num_impressions": "the table lists the images with their subject and impression",
    "ppi": "the table states the resolution of every file",
    "ppi_manifest": "the table states the resolution and the ridge polarity of every file",
    "assume_ppi": "the table states the resolution of every file",
    "ridges": "the table states the ridge polarity of every file",
}

# The options that only have a meaning with a sample table
OPTIONS_OF_SAMPLES = ("allow_missing_minutiae", "ignore_minutiae")

# How many files a message about a sample table names before it gives the
# number of the rest, and how many resolutions the start lists
_MAX_NAMED = 5
_MAX_RESOLUTIONS = 20

# Where the ridge polarity of a row of a sample table comes from (its
# ridges_origin), for the lines that count them
_RIDGES_ORIGIN_TEXT = {
    "argument": "from --ridges",
    "manifest": "from the manifest",
    "estimated": "estimated from the image",
    "default": "by default: the image allowed no estimate",
}

# File in --output_dir with the settings of the training
SETTINGS_FILE = "training_settings.json"

# Settings that must be the same when a training is continued. A change of
# the first group makes model.pyt unusable or changes which subjects the
# model has seen; a change of the others changes what is trained silently.
FIXED_SETTINGS = (
    "variant",
    "embedding_dim",
    "num_subjects",
    "num_impressions",
    "validation_split",
    "max_validation_subjects",
    "validation_seed",
    "batch_size",
    "canvas_size",
    "preparation",
)

# Settings that may change when a training is continued; changes are listed
CHANGEABLE_SETTINGS = (
    "num_epochs",
    "patience",
    "learning_rate",
    "warmup_steps",
    "lr_schedule",
    "gradient_clip_norm",
    "localization_lr_factor",
    "loss_weights",
)


def _has_training(output_dir: str) -> bool:
    """
    True if output_dir holds model.pyt, that is a training to continue.
    (get_newest_model_file is not used here: it creates the directory.)
    """
    return os.path.exists(os.path.join(output_dir, "model.pyt"))


class SettingsChangedError(RuntimeError):
    """The output directory holds a training with other settings."""


def _is_first_process() -> bool:
    """False in the processes of a multi-GPU training other than the first."""
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _write_json(path: str, obj: dict) -> None:
    # Only the first process of a training writes, so one name is enough
    temporary = f"{path}.tmp"
    with open(temporary, "w") as file:
        json.dump(obj, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def check_training_settings(
    output_dir: str, settings: dict, allow_change: bool = False
) -> list[str]:
    """
    Compares the settings of this command with those of the training that
    is already in output_dir (model.pyt exists). Writes nothing, see
    store_training_settings.

    - A difference in one of FIXED_SETTINGS raises SettingsChangedError,
      unless allow_change is set.
    - Differences in CHANGEABLE_SETTINGS are returned as lines of text.
    - A directory with model.pyt but without the settings file was written
      by an earlier version of the script: only the image preparation can be
      compared (with data_preparation.json), and a line says so.

    Returns the lines to print; none for a new training.
    """
    path = os.path.join(output_dir, SETTINGS_FILE)
    # The file that holds the stored settings, for the message of a refusal
    stored_in = path
    notes = []
    stored = None
    if _has_training(output_dir):
        if os.path.exists(path):
            with open(path) as file:
                stored = json.load(file)
        else:
            stored = {}
            notes.append(
                f"{output_dir} holds a training that was started before "
                f"{SETTINGS_FILE} existed. Only its image preparation can be "
                "compared with this command; make sure yourself that the other "
                "options are those of the first start. If that training had "
                f"more than {DEFAULT_MAX_VALIDATION_SUBJECTS:,} validation "
                "subjects, pass --max_validation_subjects 0: the limit was not "
                "the default then."
            )
            preparation_path = os.path.join(output_dir, "data_preparation.json")
            stored_in = preparation_path
            if os.path.exists(preparation_path):
                with open(preparation_path) as file:
                    old = json.load(file)
                stored["canvas_size"] = old.pop("canvas_size", None)
                old.pop("ppi_manifest", None)
                stored["preparation"] = old
    if stored:
        changed = [
            key for key in FIXED_SETTINGS
            if key in stored and stored[key] != settings[key]
        ]
        if changed and not allow_change:
            lines = [
                f"  {key}: stored {json.dumps(stored[key], sort_keys=True)}, "
                f"now {json.dumps(settings[key], sort_keys=True)}"
                for key in changed
            ]
            raise SettingsChangedError(
                f"{output_dir} holds a training with other settings, and the "
                "same --output_dir continues that training:\n"
                + "\n".join(lines)
                + "\nThese settings cannot change in a running training. To "
                "train with the new settings, use a new --output_dir. To "
                "continue the stored training, use its settings (they are in "
                f"{stored_in}). --allow_changed_settings continues anyway."
            )
        for key in changed:
            notes.append(
                f"Continuing although {key} changed (--allow_changed_settings): "
                f"stored {json.dumps(stored[key], sort_keys=True)}, now "
                f"{json.dumps(settings[key], sort_keys=True)}"
            )
        for key in CHANGEABLE_SETTINGS:
            if key in stored and stored[key] != settings[key]:
                notes.append(
                    f"Changed since the last run that trained, used from the "
                    f"next epoch on: {key} {json.dumps(stored[key], sort_keys=True)} -> "
                    f"{json.dumps(settings[key], sort_keys=True)}"
                )
        if "cosine" in (stored.get("lr_schedule"), settings["lr_schedule"]) and any(
            key in stored and stored[key] != settings[key]
            for key in ("num_epochs", "lr_schedule", "learning_rate")
        ):
            notes.append(
                "The cosine schedule is computed from the step number and the "
                "total number of steps, so with these changes the learning "
                "rate of the next epoch does not continue where the last one "
                "ended."
            )
    return notes


def store_training_settings(output_dir: str, settings: dict) -> None:
    """
    Writes the settings to training_settings.json in output_dir, and how the
    images are prepared to data_preparation.json (the same values; the file
    that earlier versions of the script wrote).

    main() calls this when a new training starts, after the file layout was
    checked, and after a run that trained at least one epoch. A start that
    is refused or that trains nothing therefore leaves the files as they
    are: they hold the settings of the last run that trained. (A process
    that is killed outright after it trained an epoch cannot write them; the
    files then still hold the settings of the run before.)

    Only the first process of a multi-GPU training writes.
    """
    if not _is_first_process():
        return
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, SETTINGS_FILE), settings)
    _write_json(
        os.path.join(output_dir, "data_preparation.json"),
        dict(
            settings["preparation"],
            canvas_size=settings["canvas_size"],
            ppi_manifest=settings["ppi_manifest"],
        ),
    )


def describe_layout_problem(
    found: IdentifierSet,
    wanted: list[Identifier],
    num_subjects: int,
    num_impressions: int,
    directory: str,
    extension: str,
    option: str,
) -> str | None:
    """
    Compares the files that were found below a directory with the identifiers
    that the training needs. Returns None if all are there, else a message
    that names the first missing files and the options that decide which
    files are needed.
    """
    available = set(found.identifiers)
    missing = [i for i in wanted if i not in available]
    if not missing:
        return None
    examples = ", ".join(
        f"{i.subject + 1}_{i.impression + 1}{extension}" for i in missing[:3]
    )
    if found.identifiers:
        subjects = [i.subject for i in found.identifiers]
        impressions = [i.impression for i in found.identifiers]
        seen = (
            f"{len(found):,} {extension} files of {found.num_subjects:,} subjects "
            f"were found (subject numbers {min(subjects) + 1} to "
            f"{max(subjects) + 1}, impression numbers {min(impressions) + 1} to "
            f"{max(impressions) + 1})"
        )
    else:
        seen = f"no {extension} file was found"
    return (
        f"{len(missing):,} of the {len(wanted):,} files that are needed "
        f"are missing below {directory} ({option}). The first ones: {examples}. "
        f"--num_subjects {num_subjects} and --num_impressions {num_impressions} "
        f"mean that every file <subject>_<impression>{extension} with subject 1 "
        f"to {num_subjects} and impression 1 to {num_impressions} must exist, "
        f"directly in the directory or in a subdirectory; {seen}. Set the two "
        "options to what the directory holds. Subjects must be numbered "
        "without gaps and each needs all its impressions; see the docstring "
        "of scripts/train_500k.py for the layout."
        " Files with other names, with gaps in their numbers or with uneven "
        "numbers of impressions can be described in a sample table and "
        "trained with --samples, see docs/sample-table.md."
    )


class DatasetLayoutError(ValueError):
    """The files of the dataset are not those that the options ask for."""


def find_dataset_files(
    dataset_dir: str,
    minutiae_dir: str | None,
    normalizer: SensorNormalizer | None,
    image_ids: list[Identifier],
    minutiae_ids: list[Identifier],
    num_subjects: int,
    num_impressions: int,
) -> CustomFingerprintLoader:
    """
    Scans dataset_dir and checks that it holds an image for every identifier
    in image_ids and, if minutiae_dir is given, that minutiae_dir holds a
    .mnt file for every identifier in minutiae_ids.

    Returns the loader of the image files. Raises DatasetLayoutError with a
    message for the user if files are missing, two files have the same
    identifier, or a file name is not <subject>_<impression>.
    """
    extension = CustomFingerprintLoader._extension()
    print(f"Scanning {dataset_dir} for {extension} files...")
    try:
        files = CustomFingerprintLoader(dataset_dir, normalizer)
    except ValueError as error:
        raise DatasetLayoutError(str(error)) from error
    problem = describe_layout_problem(
        files.ids, image_ids, num_subjects, num_impressions,
        dataset_dir, extension, "--dataset_dir",
    )
    if problem is None and minutiae_dir is not None:
        try:
            minutiae_files = FingerFlowMinutiaMapLoader(minutiae_dir).ids
        except ValueError as error:
            raise DatasetLayoutError(f"--minutiae_dir: {error}") from error
        problem = describe_layout_problem(
            minutiae_files, minutiae_ids, num_subjects, num_impressions,
            minutiae_dir, ".mnt", "--minutiae_dir",
        )
        if problem is not None:
            problem += (
                " The .mnt files are written by scripts/extract_minutiae.py; its "
                "error log names the images that have none."
            )
    if problem is not None:
        raise DatasetLayoutError(problem)
    return files


def read_sample_table(samples_dir: str, dataset_dir: str) -> SampleTable:
    """
    Reads the sample table in samples_dir for the images below dataset_dir.
    The table is checked against every rule of its format while it is read
    (flx/preparation/sample_table.py), and a note is printed about columns
    that this code does not know and does not use.

    Raises SampleTableError with a message for the user if the table cannot
    be used, holds no row, or cannot be read because pyarrow is missing.
    """
    print(f"Reading the sample table {samples_dir}...")
    table = SampleTable.read(samples_dir, dataset_dir)
    if table.num_rows == 0:
        raise SampleTableError(
            f"The sample table {samples_dir} holds no row, so there is nothing to train "
            "on. Build a table of the image files (scripts/build_sample_table.py)."
        )
    if table.unknown_columns:
        print(
            f"Note: the sample table holds {len(table.unknown_columns)} column(s) that this "
            f"code does not know and does not use: {', '.join(table.unknown_columns)}."
        )
    # The identifiers of all rows, which the loaders hand to every Dataset.
    # They are made here, because a table whose rows cannot be told apart by
    # their identifiers is refused when they are made.
    table.ids
    return table


def _named_rows(table: SampleTable, rows: np.ndarray) -> str:
    """The first of the rows as a message names them, and how many more there are."""
    named = "; ".join(table.describe_row(int(row)) for row in rows[:_MAX_NAMED])
    if len(rows) > _MAX_NAMED:
        named += f"; and {len(rows) - _MAX_NAMED:,} more"
    return named


def find_missing_table_files(table: SampleTable, rows: np.ndarray | None = None) -> np.ndarray:
    """
    The rows of a sample table whose image file is not in its directory, in
    the order of the rows. Every directory that the rows name is listed once,
    and the names of the files are looked up in what was listed: no file is
    asked for by itself, which would be a call to the file system per image.

    A file that is there is not opened here. Whether it is the file that its
    row describes is found out when it is loaded (check_table_images).

    Args:
        rows: The rows to look at; None for all rows of the table
    """
    samples = table.samples
    # Directory below the image directory -> the names in it; None for a
    # directory that is there and cannot be listed
    listed: dict[str, set | None] = {}
    missing = []
    for row in range(table.num_rows) if rows is None else rows.tolist():
        directory, _, name = samples.image_path(row).rpartition("/")
        if directory not in listed:
            try:
                listed[directory] = set(
                    os.listdir(os.path.join(table.image_dir, *directory.split("/")))
                )
            except (FileNotFoundError, NotADirectoryError):
                listed[directory] = set()
            except OSError:
                listed[directory] = None
        names = listed[directory]
        if names is None:
            # Files can be read from a directory that cannot be listed
            there = os.path.exists(table.image_file(row))
        else:
            there = name in names
        if not there:
            missing.append(row)
    return np.array(missing, dtype=np.int64)


def check_table_files(table: SampleTable, rows: np.ndarray | None = None) -> None:
    """
    Raises SampleTableError if image files that the sample table names are
    not below its image directory: with their number, the first ones by
    path, and what to do.

    Args:
        rows: The rows whose files are needed; None for all rows of the table
    """
    missing = find_missing_table_files(table, rows)
    if len(missing) == 0:
        return
    needed = "the sample table names" if rows is None else "are needed of the sample table"
    raise SampleTableError(
        f"{len(missing):,} of the {table.num_rows if rows is None else len(rows):,} image "
        f"files that {needed} are missing below {table.image_dir} (--dataset_dir). The first "
        f"ones: {_named_rows(table, missing)}. Give the directory that the table was built "
        "from as --dataset_dir. If files were moved, renamed or removed since then, put them "
        "back, or build the table again from the files as they are now "
        "(scripts/build_sample_table.py)."
    )


def table_has_minutiae(table: SampleTable, rows: np.ndarray) -> bool:
    """
    Whether a sample table holds minutiae for any of the given rows. That is
    the rule by which the minutia branch is trained on these rows: a row with
    a list of minutiae counts, also with an empty one (the image has none);
    a row with a null cell (they are not known) does not.
    """
    return bool(table.samples.minutiae_known[rows].any())


def check_table_minutiae(
    table: SampleTable, training_rows: np.ndarray, allow_missing: bool = False
) -> int:
    """
    The rule of a training with minutiae for the training rows whose minutiae
    are not known: a null cell of the table. (A row with an empty list is
    known: its image has no minutiae.) Such rows stop the run, unless
    allow_missing says that they train without a minutia map. Validation
    rows need no minutiae.

    Returns the number of such rows among training_rows. Raises
    SampleTableError with their number, the first ones by path, and what to
    do, if there are any and allow_missing is not set.
    """
    unknown = training_rows[~table.samples.minutiae_known[training_rows]]
    if len(unknown) and not allow_missing:
        raise SampleTableError(
            f"The sample table holds no minutiae (null) for {len(unknown):,} of the "
            f"{len(training_rows):,} training images. The first ones: "
            f"{_named_rows(table, unknown)}. A row is without minutiae if the .mnt file of "
            "its image was missing or refused when the table was built (build_failures.txt "
            "in the directory of the table says which and why). Extract the minutiae of "
            "these images and build the table again (scripts/build_sample_table.py), or "
            "pass --allow_missing_minutiae: these images then train without a minutia map, "
            "with weight 0 in the minutia map loss. --ignore_minutiae trains the "
            "texture-only variant instead."
        )
    return len(unknown)


def check_table_images(
    image_loader: SampleTableImageLoader,
    max_images: int | None = None,
    rows: np.ndarray | None = None,
) -> int:
    """
    Loads at most max_images images, evenly spread over the rows of the
    sample table, as the training loads them. The loader compares every file
    with its row (size in bytes and in pixels), so image files that are not
    those of the table stop the run here, before the model is built, and not
    in a worker process during the first epoch: another --dataset_dir with
    files of the same names, or files that changed after the table was
    built. A file that is not among the loaded ones is compared when the
    training loads it.

    Args:
        max_images: None for IMAGES_LOADED_AT_THE_START
        rows: The rows to spread the images over; None for all rows of the
            table

    Returns the number of images loaded. Raises SampleTableError with the
    message of the loader for the first file that cannot be used.
    """
    table = image_loader.table
    if max_images is None:
        max_images = IMAGES_LOADED_AT_THE_START
    available = table.num_rows if rows is None else len(rows)
    # From the first of the rows to the last, at equal distances
    spread = np.unique(np.linspace(0, available - 1, min(max_images, available)).astype(np.int64))
    for row in (spread if rows is None else rows[spread]).tolist():
        try:
            image_loader.get(table.identifiers[row])
        except (SampleTableError, FileNotFoundError) as error:
            # The message of the loader names the file, its row and what to do
            raise SampleTableError(str(error)) from error
        except (OSError, ValueError) as error:
            raise SampleTableError(
                f"{error}. The file belongs to the row {table.describe_row(row)} of the "
                f"sample table, for the image directory {table.image_dir}, and it cannot be "
                "loaded. Put the file back as it was when the table was built, or build the "
                "table again without it (scripts/build_sample_table.py)."
            ) from error
    return len(spread)


def sample_resolution_sources(
    normalizer: SensorNormalizer, paths: list[str], max_files: int = 1000
) -> tuple[dict, int]:
    """
    Where the resolution of the training images comes from, counted on at
    most max_files of the paths, evenly spread over the list. Only the file
    headers are read. Files whose resolution is unknown are counted under
    "unknown".

    Returns (counts per source and value, number of files looked at).
    """
    step = max(1, len(paths) // max_files)
    sample = paths[::step][:max_files]
    counts: dict[str, int] = {}
    for path in sample:
        try:
            ppi, origin = normalizer.resolve_ppi(path)
            key = sensor_normalization.resolution_source_label(ppi, origin)
        except ValueError:
            key = "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts, len(sample)


def describe_table_resolutions(table: SampleTable, rows: np.ndarray) -> list[str]:
    """
    Where the resolution and the ridge polarity of the given rows of a sample
    table come from, as lines to print: what sample_resolution_sources counts
    on a sample of the files, counted on every one of the rows, because the
    table holds it. The lines are those of
    sensor_normalization.describe_resolution_sources, then one line per ridge
    polarity and source.

    The sources are those of the build of the table: "--ppi", "--assume_ppi"
    and "--ridges" in these lines are options that
    scripts/build_sample_table.py was given, not options of this script.
    """
    samples = table.samples
    ppi, ppi_origin = samples.ppi[rows], samples.ppi_origin[rows]
    counts: dict[str, int] = {}
    for index, origin in enumerate(PPI_ORIGINS):
        values, numbers = np.unique(ppi[ppi_origin == index], return_counts=True)
        for value, number in zip(values.tolist(), numbers.tolist()):
            # Two resolutions can have the same text (it shows six digits)
            key = sensor_normalization.resolution_source_label(value, origin)
            counts[key] = counts.get(key, 0) + number
    most = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:_MAX_RESOLUTIONS]
    # The assumption that the build was given, for the note about files that
    # took the resolution of their own metadata instead
    settings = table.build.get("settings")
    assume_ppi = settings.get("assume_ppi") if isinstance(settings, dict) else None
    if isinstance(assume_ppi, bool) or not isinstance(assume_ppi, (int, float)):
        assume_ppi = None
    lines = sensor_normalization.describe_resolution_sources(
        dict(most), assume_ppi, table.target_ppi
    )
    if len(counts) > len(most):
        rest = len(rows) - sum(number for _, number in most)
        lines.append(f"  {rest:>12,}  with {len(counts) - len(most):,} further resolutions")

    lines.append("Ridge polarity of these images, as the files hold it:")
    light, ridges_origin = samples.ridges_light[rows], samples.ridges_origin[rows]
    polarities = []
    for is_light, ridges in (
        (False, sensor_normalization.RIDGES_DARK),
        (True, sensor_normalization.RIDGES_LIGHT),
    ):
        for index, origin in enumerate(RIDGES_ORIGINS):
            number = int(np.count_nonzero((light == is_light) & (ridges_origin == index)))
            if number:
                polarities.append((number, f"{ridges} ridges, {_RIDGES_ORIGIN_TEXT[origin]}"))
    polarities.sort(key=lambda polarity: -polarity[0])
    lines.extend(f"  {number:>12,}  {text}" for number, text in polarities)
    lines.append(
        "  These are the sources of the build of the table "
        "(scripts/build_sample_table.py): to change one, build the table again."
    )
    return lines


def _rows_per_subject(table: SampleTable, rows: np.ndarray) -> np.ndarray:
    """How many of the given rows of a sample table each of their subjects has."""
    return np.unique(table.samples.subject[rows], return_counts=True)[1]


def _files_that_failed_in_the_build(table: SampleTable) -> str | None:
    """
    What the build of a sample table recorded about files that failed, as a
    text for the summary of the start; None if it recorded none.
    """
    failed, files = table.build.get("failed"), table.build.get("files")
    failed = len(failed) if isinstance(failed, list) else failed
    if isinstance(failed, bool) or not isinstance(failed, int) or failed <= 0:
        return None
    text = f"{failed:,} file(s) failed when the table was built"
    # The files that were listed and have no row; the others have a row
    without_row = files - table.num_rows if type(files) is int else -1
    if 0 <= without_row <= failed:
        text += (
            f": {without_row:,} without a row (they are not in the table), "
            f"{failed - without_row:,} with a row without minutiae"
        )
    return f"{text}; build_failures.txt in the directory of the table names them"


def split_subjects(
    num_subjects: int,
    validation_split: float,
    seed: int = VALIDATION_SEED,
    max_validation_subjects: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Splits the subjects 0 .. num_subjects - 1 into training subjects and
    held-out validation subjects. A subject is never in both lists.

    Args:
        num_subjects: Total number of subjects
        validation_split: Share of the subjects to hold out for validation
            (0.0 <= share < 1.0, rounded down to whole subjects). 0 holds out
            no subject.
        seed: Seed of the random choice. The same arguments always give the
            same split.
        max_validation_subjects: Upper limit for the number of held-out
            subjects. 0 means no limit.

    Returns:
        Tuple of (training subjects, validation subjects), both sorted
    """
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(
            f"validation_split must be at least 0 and below 1, got {validation_split}"
        )
    if max_validation_subjects < 0:
        raise ValueError("max_validation_subjects must not be negative")
    all_subjects = list(range(num_subjects))
    if validation_split == 0:
        return all_subjects, []

    num_val_subjects = int(num_subjects * validation_split)
    if max_validation_subjects > 0:
        num_val_subjects = min(num_val_subjects, max_validation_subjects)
    if num_val_subjects < MIN_VALIDATION_SUBJECTS:
        raise ValueError(
            f"validation_split {validation_split} of {num_subjects} subjects "
            f"holds out {num_val_subjects} subjects, but validation needs at "
            f"least {MIN_VALIDATION_SUBJECTS}. Use a larger share, or 0 to "
            "train without validation."
        )
    # An own generator, so that the split does not depend on other uses of
    # the random module and does not change them
    random.Random(seed).shuffle(all_subjects)
    val_subjects = sorted(all_subjects[:num_val_subjects])
    train_subjects = sorted(all_subjects[num_val_subjects:])
    return train_subjects, val_subjects


def wanted_validation_subjects(
    num_subjects: int, validation_split: float, max_validation_subjects: int = 0
) -> int:
    """
    The number of subjects that share and limit ask to hold out, as
    split_subjects computes it: the share of all subjects, rounded down, and
    not more than max_validation_subjects if that is above 0.
    """
    wanted = int(num_subjects * validation_split)
    if max_validation_subjects > 0:
        wanted = min(wanted, max_validation_subjects)
    return wanted


def split_table_subjects(
    subjects,
    impressions,
    validation_split: float,
    seed: int = VALIDATION_SEED,
    max_validation_subjects: int = 0,
) -> tuple[list[int], list[int]]:
    """
    split_subjects for the subjects of a sample table, whose numbers can have
    gaps and which can have any number of impressions each. A subject is
    never in both lists.

    Validation compares two impressions of a subject, so only a subject with
    at least two impressions can be held out; a subject with one always
    trains. The subjects are shuffled as split_subjects shuffles the numbers
    0 .. num_subjects - 1: in ascending order, once, with the seed. Held out
    are the first subjects of the shuffled list that have two impressions,
    as many as wanted_validation_subjects gives for ALL subjects of the
    table. If fewer subjects have two impressions, all of these are held out
    and the validation set is smaller than the share asks for.

    For the subjects 0 .. n - 1 with the same number of impressions each (two
    or more), the result is that of split_subjects.

    Args:
        subjects: The subjects of the table, each once (SampleTable.subjects)
        impressions: The number of impressions of each of them, in the same
            order (SampleTable.impressions_per_subject)
        validation_split, seed, max_validation_subjects: As for split_subjects

    Returns:
        Tuple of (training subjects, validation subjects), both sorted
    """
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(
            f"validation_split must be at least 0 and below 1, got {validation_split}"
        )
    if max_validation_subjects < 0:
        raise ValueError("max_validation_subjects must not be negative")
    subjects, impressions = np.asarray(subjects), np.asarray(impressions)
    ascending = np.argsort(subjects, kind="stable")
    subjects, impressions = subjects[ascending], impressions[ascending]
    if validation_split == 0:
        return subjects.tolist(), []

    num_val_subjects = wanted_validation_subjects(
        len(subjects), validation_split, max_validation_subjects
    )
    if num_val_subjects < MIN_VALIDATION_SUBJECTS:
        raise ValueError(
            f"validation_split {validation_split} of {len(subjects)} subjects "
            f"holds out {num_val_subjects} subjects, but validation needs at "
            f"least {MIN_VALIDATION_SUBJECTS}. Use a larger share, or 0 to "
            "train without validation."
        )
    # The positions in the ascending list are shuffled instead of the numbers.
    # A shuffle moves the items of a list by their places, whatever they are,
    # so place k of the shuffled list holds the subject that split_subjects
    # has there when the subjects are 0 .. n - 1. An own generator, as there.
    shuffled = list(range(len(subjects)))
    random.Random(seed).shuffle(shuffled)
    shuffled = np.array(shuffled, dtype=np.int64)
    held_out = shuffled[impressions[shuffled] >= 2][:num_val_subjects]
    if len(held_out) < MIN_VALIDATION_SUBJECTS:
        raise ValueError(
            "Validation compares two impressions of the same subject, so it "
            f"needs at least {MIN_VALIDATION_SUBJECTS} subjects with 2 or more "
            f"impressions, and {len(held_out)} of the {len(subjects)} subjects "
            "have that many. Use --validation_split 0 to train without "
            "validation."
        )
    in_validation = np.zeros(len(subjects), dtype=np.bool_)
    in_validation[held_out] = True
    return subjects[~in_validation].tolist(), subjects[in_validation].tolist()


def variant_name(localization: bool, with_minutiae: bool) -> str:
    """
    Name of the DeepPrint variant that is trained: "Loc" if the localization
    network (learned alignment) is used, then "Tex" for the texture branch,
    then "Minu" if minutiae are given. The model class is DeepPrint_<name>
    and its constructor get_DeepPrint_<name> in
    flx/extractor/fixed_length_extractor.py.
    """
    return ("Loc" if localization else "") + "Tex" + ("Minu" if with_minutiae else "")


def create_extractor(
    localization: bool,
    with_minutiae: bool,
    num_training_subjects: int,
    embedding_dim: int,
    loss_weights: LossWeights | None = None,
):
    """
    The model and its loss for the chosen variant. The variants with and
    without localization network differ in nothing else.
    """
    # Looked up at call time, so that the names of this module can be replaced
    constructor = globals()["get_DeepPrint_" + variant_name(localization, with_minutiae)]
    if with_minutiae:
        return constructor(
            num_training_subjects=num_training_subjects,
            num_dims=embedding_dim,
            loss_weights=loss_weights,
        )
    return constructor(
        num_training_subjects=num_training_subjects,
        num_texture_dims=embedding_dim,
        loss_weights=loss_weights,
    )


def create_training_and_validation_ids(
    num_subjects: int,
    num_impressions: int,
    validation_split: float = 0.0,
    seed: int = VALIDATION_SEED,
    max_validation_subjects: int = 0,
) -> tuple[IdentifierSet, IdentifierSet | None]:
    """
    Create identifier sets for training and validation samples.

    Whole subjects are held out: all impressions of a validation subject are
    in the validation set and none in the training set. See split_subjects
    for the arguments.

    Returns:
        Tuple of (training_ids, validation_ids). validation_ids is None if
        validation_split is 0.
    """
    train_subjects, val_subjects = split_subjects(
        num_subjects, validation_split, seed, max_validation_subjects
    )
    if val_subjects and num_impressions < 2:
        raise ValueError(
            "Validation compares two impressions of the same subject, so it "
            "needs at least 2 impressions per subject. Use --validation_split 0 "
            "to train without validation."
        )

    training_ids = IdentifierSet([
        Identifier(s, i)
        for s in train_subjects
        for i in range(num_impressions)
    ])

    validation_ids = IdentifierSet([
        Identifier(s, i)
        for s in val_subjects
        for i in range(num_impressions)
    ]) if val_subjects else None

    return training_ids, validation_ids


def create_table_training_and_validation_ids(
    table: SampleTable,
    validation_split: float = 0.0,
    seed: int = VALIDATION_SEED,
    max_validation_subjects: int = 0,
) -> tuple[IdentifierSet, IdentifierSet | None]:
    """
    create_training_and_validation_ids for the rows of a sample table: the
    identifiers of all rows, with whole subjects held out for validation
    (see split_table_subjects for which ones and for the arguments). Every
    row of the table is in one of the two sets.

    The sets are made of the Identifier objects of the table
    (SampleTable.identifiers), not of new ones.

    Returns:
        Tuple of (training_ids, validation_ids). validation_ids is None if
        validation_split is 0.
    """
    train_subjects, val_subjects = split_table_subjects(
        table.subjects,
        table.impressions_per_subject,
        validation_split,
        seed,
        max_validation_subjects,
    )
    training_ids = IdentifierSet(table.identifiers_of(train_subjects))
    validation_ids = (
        IdentifierSet(table.identifiers_of(val_subjects)) if val_subjects else None
    )
    return training_ids, validation_ids


def create_validation_benchmark(
    validation_ids: IdentifierSet,
    num_impressions: int,
    comparisons_per_subject: int = 10,
    seed: int = VALIDATION_SEED,
) -> VerificationBenchmark:
    """
    Create a verification benchmark for validation.

    Generates mated pairs (same subject, different impressions) and
    non-mated pairs (different subjects) for EER calculation.
    The same seed always gives the same pairs.
    """
    rng = random.Random(seed)

    # Get unique subjects in validation set
    subjects = sorted(list({id.subject for id in validation_ids}))

    comparisons: list[BiometricComparison] = []

    # Create mated comparisons (same subject, different impressions)
    for subject in subjects:
        for i in range(num_impressions):
            for j in range(i + 1, num_impressions):
                comparisons.append(BiometricComparison(
                    Identifier(subject, i),
                    Identifier(subject, j),
                ))

    # Create non-mated comparisons (different subjects)
    for subject in subjects:
        other_subjects = [s for s in subjects if s != subject]
        if len(other_subjects) >= comparisons_per_subject:
            selected = rng.sample(other_subjects, comparisons_per_subject)
        else:
            selected = other_subjects

        for other in selected:
            imp1 = rng.randint(0, num_impressions - 1)
            imp2 = rng.randint(0, num_impressions - 1)
            comparisons.append(BiometricComparison(
                Identifier(subject, imp1),
                Identifier(other, imp2),
            ))

    return VerificationBenchmark(comparisons)


def count_validation_pairs(impressions, comparisons_per_subject: int = 10) -> tuple[int, int]:
    """
    The numbers of (mated pairs, non-mated pairs) that
    create_table_validation_benchmark makes for validation subjects with the
    given numbers of impressions: every two impressions of a subject, and
    comparisons_per_subject pairs with other subjects for every subject (one
    with each other subject if there are fewer).
    """
    impressions = np.asarray(impressions, dtype=np.int64)
    mated = int((impressions * (impressions - 1) // 2).sum())
    non_mated = len(impressions) * min(comparisons_per_subject, max(len(impressions) - 1, 0))
    return mated, non_mated


def create_table_validation_benchmark(
    validation_ids: IdentifierSet,
    comparisons_per_subject: int = 10,
    seed: int = VALIDATION_SEED,
) -> VerificationBenchmark:
    """
    create_validation_benchmark for validation subjects that can have any
    number of impressions each, as those of a sample table.

    The mated pairs are every two impressions that a subject has, so a
    subject with n impressions has n * (n - 1) / 2 of them and counts that
    often in the error rate. The non-mated pairs are drawn with the calls of
    create_validation_benchmark, in its order, and the two impressions of a
    pair are drawn among those that the two subjects have. So if every
    subject has the impressions 0 .. n - 1, the pairs are those of
    create_validation_benchmark for the same seed.

    The pairs hold the Identifier objects of validation_ids, not new ones.
    The same seed always gives the same pairs.
    """
    rng = random.Random(seed)

    # The impressions of every validation subject, in ascending order
    impressions: dict[int, list[Identifier]] = {}
    for identifier in validation_ids.identifiers:
        impressions.setdefault(identifier.subject, []).append(identifier)
    subjects = sorted(impressions)

    comparisons: list[BiometricComparison] = []

    # Create mated comparisons (same subject, different impressions)
    for subject in subjects:
        own = impressions[subject]
        for i in range(len(own)):
            for j in range(i + 1, len(own)):
                comparisons.append(BiometricComparison(own[i], own[j]))

    # Create non-mated comparisons (different subjects)
    for position, subject in enumerate(subjects):
        # The subjects are sorted and each is there once: all but this one
        other_subjects = subjects[:position] + subjects[position + 1:]
        if len(other_subjects) >= comparisons_per_subject:
            selected = rng.sample(other_subjects, comparisons_per_subject)
        else:
            selected = other_subjects

        own = impressions[subject]
        for other in selected:
            others = impressions[other]
            # The draw for the subject first, as create_validation_benchmark
            sample1 = own[rng.randint(0, len(own) - 1)]
            sample2 = others[rng.randint(0, len(others) - 1)]
            comparisons.append(BiometricComparison(sample1, sample2))

    return VerificationBenchmark(comparisons)


def canvas_roi(canvas_size: int) -> tuple[int, int] | None:
    """
    The (height, width) that every image is cut or padded to before it is
    scaled to the input size of the model; None without a canvas.
    """
    if canvas_size < 0:
        raise ValueError(f"canvas_size must not be negative, got {canvas_size}")
    return (canvas_size, canvas_size) if canvas_size > 0 else None


def _resize_to_input_size(canvas_size: int):
    """
    The last step of the image pipeline. Without a canvas, the image is padded
    to a square of its longer side and scaled to the input size. With one, it
    is first cut or padded around its centre to canvas_size x canvas_size
    pixels, so all images are scaled by the same factor.
    """
    roi = canvas_roi(canvas_size)
    if roi is None:
        return pad_and_resize_to_deepprint_input_size
    return functools.partial(pad_and_resize_to_deepprint_input_size, roi=roi)


def create_image_dataset(
    dataset_dir: str,
    training_ids: IdentifierSet,
    normalizer: SensorNormalizer | None = None,
    canvas_size: int = 0,
    files: CustomFingerprintLoader | None = None,
) -> Dataset:
    """
    Create image dataset with augmentation pipeline.

    Args:
        normalizer: Loads the image files at 500 pixels per inch with dark
            ridges, like scripts/extract_minutiae.py. None loads the files
            as they are.
        canvas_size: Side in pixels of the square that every image is cut or
            padded to before it is scaled to the input size. 0 pads every
            image to a square of its own longer side.
        files: The loader of the image files, if the directory was already
            scanned; None scans dataset_dir.
    """
    image_loader = TransformedImageLoader(
        images=files or CustomFingerprintLoader(dataset_dir, normalizer),
        poses=POSE_AUGMENTATION,  # Random pose for training robustness
        transforms=[
            QUALITY_AUGMENTATION,
            LazilyAllocatedBinarizer(BINARIZATION_THRESHOLD),
            _resize_to_input_size(canvas_size),
        ],
    )
    return Dataset(image_loader, training_ids)


def create_validation_image_loader(
    dataset_dir: str,
    normalizer: SensorNormalizer | None = None,
    canvas_size: int = 0,
    files: CustomFingerprintLoader | None = None,
) -> TransformedImageLoader:
    """
    The images as the model gets them outside of the training: prepared like
    the training images, but without augmentation. Use the same loader (same
    normalizer and canvas_size) when embeddings are extracted with the model.
    """
    return TransformedImageLoader(
        images=files or CustomFingerprintLoader(dataset_dir, normalizer),
        poses=None,  # No pose augmentation for validation
        transforms=[
            LazilyAllocatedBinarizer(BINARIZATION_THRESHOLD),
            _resize_to_input_size(canvas_size),
        ],
    )


def create_minutia_dataset(
    minutiae_dir: str,
    training_ids: IdentifierSet,
    images: TransformedImageLoader,
    canvas_size: int = 0,
) -> Dataset:
    """
    Create minutia map dataset.

    Args:
        minutiae_dir: Directory containing .mnt files (from extract_minutiae.py)
                      If None, uses empty tensors (model still works, texture-only)
        training_ids: Identifiers for training samples
        images: Loader of the training images. The minutia map of a sample gets
                the random pose of its image.
        canvas_size: The canvas_size of create_image_dataset, so that the
                minutiae are moved like the image.

    The size in the header of a .mnt file is compared with the size of the
    image that was loaded: a ValueError says so if extraction and training
    prepared the image differently.

    Returns:
        Dataset containing minutia maps
    """
    if minutiae_dir is not None:
        print(f"Loading minutiae from {minutiae_dir}")
        minutia_loader = PoseTransformedMinutiaMapLoader(
            FingerFlowMinutiaMapLoader(minutiae_dir), images, roi=canvas_roi(canvas_size)
        )
        return Dataset(minutia_loader, training_ids)
    else:
        # No minutiae available - use empty placeholder
        print("No minutiae directory provided - using texture-only mode")
        return Dataset(ConstantDataLoader((torch.tensor([]), 0.0)), training_ids)


def create_table_minutia_dataset(
    minutia_loader: SampleTableMinutiaMapLoader,
    training_ids: IdentifierSet,
    images: TransformedImageLoader,
    canvas_size: int = 0,
) -> Dataset:
    """
    create_minutia_dataset for the minutiae of a sample table, which are not
    read from a directory of .mnt files.

    Args:
        minutia_loader: The loader of the minutiae of the table
        training_ids: Identifiers for training samples
        images: Loader of the training images. The minutia map of a sample gets
                the random pose of its image.
        canvas_size: The canvas_size of create_image_dataset, so that the
                minutiae are moved like the image.

    A row whose minutiae are not known (null) or whose image has none gives
    a minutia map of zeros with the weight 0, like an empty .mnt file.

    Returns:
        Dataset containing minutia maps
    """
    minutia_loader = PoseTransformedMinutiaMapLoader(
        minutia_loader, images, roi=canvas_roi(canvas_size)
    )
    return Dataset(minutia_loader, training_ids)


def load_trained_model(
    output_dir: str,
    dataset_dir: str,
    ppi_manifest: str | None = None,
    best: bool = True,
    samples: str | None = None,
):
    """
    The model of a training of this script and a loader that prepares images
    as that training did (same resolution sources, ridge polarity and canvas,
    no augmentation), read from training_settings.json in output_dir.

    With samples the images are those that a sample table describes: every
    image is prepared as its row of the table says (resolution and ridge
    polarity) and then as the training prepared its images (canvas, no
    augmentation). A model that was trained from a sample table (--samples)
    can only be loaded this way, because its settings hold no resolution
    sources. The table is the one of the images to embed; it need not be the
    table of the training.

    Args:
        output_dir: The --output_dir of the training
        dataset_dir: Directory with the images to embed, in the layout that
            this script reads (<subject>_<impression>.png)
        ppi_manifest: Manifest for the images in dataset_dir, if the training
            used one. None uses the path that the training was given, which
            only fits if dataset_dir is the directory of the training.
        best: Load best_model.pyt (True) or model.pyt, the newest epoch
        samples: Directory of a sample table that describes the images below
            dataset_dir, which can then have any names. ppi_manifest is not
            used with it.

    Returns:
        (extractor, image_loader). Embeddings:
            dataset = Dataset(image_loader, image_loader.ids)
            texture, minutia = extractor.extract(dataset)

    Raises ValueError if the model was trained from a sample table and
    samples is not given, or if the table of samples holds images at another
    resolution than those of the training, and SampleTableError if the table
    cannot be read.
    """
    path = os.path.join(output_dir, SETTINGS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written by scripts/train_500k.py; a model "
            "of an earlier version must be loaded by hand with the options of "
            "its training."
        )
    with open(path) as file:
        settings = json.load(file)
    table = None
    if samples is not None:
        if ppi_manifest is not None:
            raise ValueError(
                "ppi_manifest and samples were both given. A sample table states the "
                "resolution of every image file itself: leave ppi_manifest out."
            )
        table = SampleTable.read(samples, dataset_dir)
        trained_at = settings["preparation"]["target_ppi"]
        if table.target_ppi != trained_at:
            raise ValueError(
                f"The sample table {samples} describes images that are prepared at "
                f"{table.target_ppi:g} ppi, but the model in {output_dir} was trained on "
                f"images at {trained_at:g} ppi: it would get the images of the table at a "
                "scale that it was not trained on. Use a table whose images are prepared at "
                "the resolution of the training, or, if the training read no table, load the "
                "images without samples."
            )
    elif "samples_sha256" in settings["preparation"]:
        raise ValueError(
            f"The training in {output_dir} read its images from a sample table "
            f"(--samples {settings.get('samples')}), so its settings do not say how the "
            "image files of a directory are to be prepared. Pass samples= with the "
            "directory of a sample table that describes the images below dataset_dir "
            "(scripts/build_sample_table.py builds one; see docs/sample-table.md)."
        )
    variant = settings["variant"]
    extractor = create_extractor(
        variant.startswith("Loc"),
        variant.endswith("Minu"),
        num_training_subjects=settings["num_training_subjects"],
        embedding_dim=settings["embedding_dim"],
    )
    model_file = get_best_model_file(output_dir) if best else get_newest_model_file(output_dir)
    torch_helpers.load_model_parameters(model_file, extractor.model, None, None)

    preparation = settings["preparation"]
    if table is not None:
        # Every file as its row says, then the canvas of the training
        image_loader = create_validation_image_loader(
            dataset_dir, None, settings["canvas_size"], files=SampleTableImageLoader(table)
        )
        return extractor, image_loader
    if ppi_manifest is None and preparation["ppi_manifest_sha256"] is not None:
        ppi_manifest = settings["ppi_manifest"]
    normalizer = SensorNormalizer(
        root_dir=dataset_dir,
        ppi=preparation["ppi"],
        manifest=ppi_manifest,
        assume_ppi=preparation["assume_ppi"],
        ridges=preparation["ridges"],
        target_ppi=preparation["target_ppi"],
    )
    image_loader = create_validation_image_loader(
        dataset_dir, normalizer, settings["canvas_size"]
    )
    return extractor, image_loader


def withhold_defaults(parser: argparse.ArgumentParser) -> dict:
    """
    Takes the defaults of the options that a sample table replaces out of
    the parser, so that such an option is None after parsing unless it was
    given. With its default in place, an option that was given with the
    value of its default (--ridges dark) could not be told from one that was
    left out, and table mode has to refuse the first.

    Call it directly before parser.parse_args(), and hand what it returns
    (the defaults by option) to check_options_of_the_mode directly after.
    """
    defaults = {
        name: parser.get_default(name)
        for name in OPTIONS_REPLACED_BY_SAMPLES
        if parser.get_default(name) is not None
    }
    parser.set_defaults(**dict.fromkeys(defaults))
    return defaults


def check_options_of_the_mode(
    parser: argparse.ArgumentParser, args: argparse.Namespace, withheld_defaults: dict
) -> None:
    """
    The options that only one of the two modes takes. With --samples (table
    mode) the options that the table replaces are refused, each with what the
    table says in its place; they stay None. Without --samples the options
    of table mode are refused, and the options whose defaults were withheld
    get them, so that everything after this call finds the values it always
    found.
    """
    if args.samples is None:
        given = [f"--{name}" for name in OPTIONS_OF_SAMPLES if getattr(args, name, False)]
        if given:
            parser.error(
                f"{' and '.join(given)} can only be given together with --samples, which "
                "reads the training set and its minutiae from a sample table. Without a table "
                "every training image needs its .mnt file in --minutiae_dir, and the "
                "texture-only variant is trained when --minutiae_dir is left out. Give "
                "--samples as well (see docs/sample-table.md), or leave "
                f"{' and '.join(given)} out."
            )
        for name, default in withheld_defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
        return
    given = [
        f"--{name} ({reason})"
        for name, reason in OPTIONS_REPLACED_BY_SAMPLES.items()
        if getattr(args, name, None) is not None
    ]
    if given:
        parser.error(
            f"--samples cannot be combined with {', '.join(given)}. Leave "
            f"{'these options' if len(given) > 1 else 'this option'} out. What a sample table "
            "says is decided when it is built: to change it, build the table again "
            "(scripts/build_sample_table.py, see docs/sample-table.md)."
        )


def _start_from_samples(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    loss_weights: LossWeights,
    learning_rate: LearningRateConfig,
    first_process: bool,
) -> dict:
    """
    The start of a training from a sample table, in the order of main():
    reads the table, holds out the validation subjects, compares the settings
    with a training that is already in --output_dir, prints what will be
    trained, checks the files, and builds the model and the datasets.

    Returns the arguments of _run_training that follow from it. Raises
    SampleTableError with a message for the user if the table or its files
    cannot be used.
    """
    localization = args.localization == "on"
    table = read_sample_table(args.samples, args.dataset_dir)
    samples = table.samples

    # Create identifiers for training and validation. Whole subjects are held
    # out for validation; the model gets one class per training subject.
    try:
        training_ids, validation_ids = create_table_training_and_validation_ids(
            table,
            args.validation_split,
            args.validation_seed,
            args.max_validation_subjects,
        )
    except ValueError as error:
        parser.error(str(error))
    num_train_subjects = training_ids.num_subjects
    num_val_subjects = validation_ids.num_subjects if validation_ids else 0
    # The rows of both sets: every subject is in one of them with all its rows
    validation_subjects = sorted({i.subject for i in validation_ids}) if validation_ids else []
    validation_rows = table.rows_of(validation_subjects)
    training_rows = table.rows_of(np.setdiff1d(table.subjects, validation_subjects))

    # The variant follows from the table: the minutia branch is trained if the
    # table holds minutiae for training images, unless they are to be ignored
    with_minutiae = table_has_minutiae(table, training_rows) and not args.ignore_minutiae
    known_minutiae = int(np.count_nonzero(samples.minutiae_known[training_rows]))
    unknown_minutiae = len(training_rows) - known_minutiae if with_minutiae else 0
    model_type = variant_name(localization, with_minutiae)

    # The settings of the training, with the keys of a training without a
    # table. The table stands where the resolution settings stand there, and
    # the options that it replaces are None: so a training does not continue
    # with another table, nor one that read a table without it, nor one that
    # read the file layout with a table. The rows count, not the directory.
    preparation = {
        "samples_sha256": table.checksum,
        "samples_rows": table.num_rows,
        "samples_subjects": len(table.subjects),
        "target_ppi": table.target_ppi,
    }
    settings = {
        "variant": model_type,
        "embedding_dim": args.embedding_dim,
        "num_subjects": None,
        "num_impressions": None,
        "num_training_subjects": num_train_subjects,
        "validation_split": args.validation_split,
        "max_validation_subjects": args.max_validation_subjects,
        "validation_seed": args.validation_seed,
        "batch_size": args.batch_size,
        "canvas_size": args.canvas_size,
        "preparation": preparation,
        "ppi_manifest": None,
        "dataset_dir": args.dataset_dir,
        "minutiae_dir": None,
        "samples": args.samples,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "learning_rate": learning_rate.learning_rate,
        "warmup_steps": learning_rate.warmup_steps,
        "lr_schedule": learning_rate.schedule,
        "gradient_clip_norm": learning_rate.gradient_clip_norm,
        "localization_lr_factor": learning_rate.localization_factor if localization else None,
        "loss_weights": {
            "cross_entropy": loss_weights.cross_entropy,
            "center": loss_weights.center,
            "minutia_map": loss_weights.minutia_map if with_minutiae else None,
        },
    }
    try:
        notes = check_training_settings(
            args.output_dir, settings, args.allow_changed_settings
        )
    except SettingsChangedError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    for note in notes:
        print(f"Note: {note}")
    log_path = os.path.join(args.output_dir, "log.json")
    epochs_before = len(TrainingLog(log_path)) if os.path.exists(log_path) else 0

    if not with_minutiae:
        minutiae = (
            "not used (--ignore_minutiae)" if known_minutiae
            else "none: the sample table holds no minutiae for the training images"
        )
    elif not unknown_minutiae:
        minutiae = f"from the sample table, for all {len(training_rows):,} training images"
    else:
        minutiae = (
            f"from the sample table, for {known_minutiae:,} of the {len(training_rows):,} "
            f"training images; it holds none (null) for the other {unknown_minutiae:,}"
        )
    training_impressions = _rows_per_subject(table, training_rows)
    num_val_images = len(validation_rows)
    model_name = model_type + (" (texture + minutiae)" if with_minutiae else " (texture-only)")
    print("=" * 60)
    print(f"Training DeepPrint {'with' if localization else 'without'} "
          f"Learned Alignment ({model_name})")
    print("=" * 60)
    print(f"Samples:      {args.samples} ({table.num_rows:,} rows, "
          f"{len(table.subjects):,} subjects)")
    failed_in_build = _files_that_failed_in_the_build(table)
    if failed_in_build:
        print(f"Table build:  {failed_in_build}")
    print(f"Dataset:      {args.dataset_dir}")
    print(f"Minutiae:     {minutiae}")
    print(f"Images:       prepared as the sample table states for every file: resampled "
          f"from its resolution to {table.target_ppi:g} ppi, dark ridges")
    print(f"Canvas:       "
          + (f"{args.canvas_size}x{args.canvas_size} pixels at 500 ppi for every image"
             if args.canvas_size else "square of the longer side of each image"))
    print(f"Batch size:   {args.batch_size} images per step and GPU")
    print(f"Subjects:     {len(table.subjects):,} total")
    print(f"  Training:   {num_train_subjects:,} subjects")
    print(f"  Validation: {num_val_subjects:,} subjects held out (seed {args.validation_seed})")
    print(f"Impressions:  {int(training_impressions.min()):,} to "
          f"{int(training_impressions.max()):,} per training subject "
          f"(median {float(np.median(training_impressions)):g})")
    print(f"Total images: {table.num_rows:,} ({len(training_rows):,} for training, "
          f"{num_val_images:,} for validation)")
    print(f"Epochs:       {args.num_epochs} (max)")
    if validation_ids:
        print(f"Patience:     {args.patience} epochs (early stopping)")
        print("Best model:   lowest validation equal error rate")
    else:
        print("Patience:     not used (no validation set)")
        print("Best model:   lowest training loss (no validation set)")
    print(f"Embedding:    {args.embedding_dim} values per branch, "
          f"{args.embedding_dim * (2 if with_minutiae else 1)} per print")
    print(f"Optimizer:    Adam, {learning_rate.describe()}")
    if localization:
        print(f"Alignment:    learned (--localization on), learning rate of the "
              f"localization network times {learning_rate.localization_factor:g}")
    else:
        print("Alignment:    none (--localization off)")
    print(f"Loss weights: cross entropy {loss_weights.cross_entropy:g}, "
          f"center {loss_weights.center:g}"
          + (f", minutia map {loss_weights.minutia_map:g}" if with_minutiae else ""))
    print(f"Output:       {args.output_dir}")
    print(f"Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPUs:         {torch.cuda.device_count()}")
    print("=" * 60)

    wanted_by_share = int(len(table.subjects) * args.validation_split)
    wanted_validation = wanted_validation_subjects(
        len(table.subjects), args.validation_split, args.max_validation_subjects
    )
    if 0 < wanted_validation < wanted_by_share:
        print(
            f"\nNote: --validation_split {args.validation_split:g} would hold out "
            f"{wanted_by_share:,} subjects; --max_validation_subjects limits this to "
            f"{wanted_validation:,}. The others stay in the training set."
        )
    if 0 < num_val_subjects < wanted_validation:
        print(
            f"\nNote: {wanted_validation:,} subjects were to be held out, but only "
            f"{num_val_subjects:,} subjects of the table have two or more impressions, which "
            "validation needs: it compares two impressions of a subject. These "
            f"{num_val_subjects:,} are held out, and every subject with one impression trains."
        )
    if validation_ids:
        validation_impressions = _rows_per_subject(table, validation_rows)
        mated_pairs, non_mated_pairs = count_validation_pairs(validation_impressions)
        print(
            f"\nValidation set: {num_val_images:,} images of {num_val_subjects:,} subjects "
            f"({int(validation_impressions.min()):,} to {int(validation_impressions.max()):,} "
            f"per subject), compared after every epoch in {mated_pairs:,} mated and "
            f"{non_mated_pairs:,} non-mated pairs."
        )
        if (
            num_val_images > MANY_VALIDATION_IMAGES
            or mated_pairs + non_mated_pairs > MANY_VALIDATION_PAIRS
        ):
            print(
                f"\nWARNING: the validation set is larger than {MANY_VALIDATION_IMAGES:,} "
                f"images or {MANY_VALIDATION_PAIRS:,} pairs (what "
                f"--max_validation_subjects {DEFAULT_MAX_VALIDATION_SUBJECTS} stands for where "
                "every subject has 10 impressions). All its images are embedded and all its "
                "pairs compared after every epoch. The mated pairs are all pairs of images "
                "of a subject, so a subject with n images has n * (n - 1) / 2 of them, and a "
                "finger with many images can dominate the validation. "
                "--max_validation_subjects and --validation_split say how many subjects are "
                "held out, --validation_seed which ones."
            )
    if num_train_subjects > MANY_SUBJECTS:
        print(
            f"\nWARNING: {num_train_subjects:,} training subjects. The classifier and "
            "the centers of the loss have one row per training subject, in every GPU "
            f"process, and model.pyt grows with them. This script was never run with more "
            f"than {MANY_SUBJECTS:,} training subjects."
        )

    print(f"\nCreated {len(training_ids):,} training identifiers")
    if validation_ids:
        print(f"Created {len(validation_ids):,} validation identifiers")

    # The files that are there, compared with the files that the table names.
    # Every process checks them, so that each one stops by itself.
    print(f"Listing the directories of the image files below {args.dataset_dir}...")
    check_table_files(table)
    if with_minutiae:
        check_table_minutiae(table, training_rows, args.allow_missing_minutiae)
        if unknown_minutiae:
            print(
                f"Note: training images without minutiae in the sample table: "
                f"{unknown_minutiae:,} of {len(training_rows):,}. They train without a "
                "minutia map (--allow_missing_minutiae), with weight 0 in the minutia map "
                "loss, and count in the other parts of the loss as every image does."
            )
    files = SampleTableImageLoader(table)
    if first_process:
        loaded = check_table_images(files)
        print(
            f"Loaded {loaded:,} image files spread over the table: each has the size in "
            "bytes and in pixels that its row holds."
        )
        print(
            f"Resolution of the {len(training_rows):,} training images, as the sample "
            "table states it:"
        )
        for line in describe_table_resolutions(table, training_rows):
            print(line)
        # The start method that worker processes get, asked for without
        # fixing it: the first of the list is the default of the platform
        start_method = (
            multiprocessing.get_start_method(allow_none=True)
            or multiprocessing.get_all_start_methods()[0]
        )
        if start_method != "fork":
            print(
                f"\nWARNING: worker processes are started with '{start_method}' here, not "
                "forked. A forked worker shares the arrays of the sample table with the "
                "process that read it; a worker that is started in another way gets a copy "
                "of them, so every worker process of the training holds the table once more."
            )

    # Initialize the model: with or without learned alignment
    # (LocalizationNetwork), with the minutia branch when minutiae are trained
    print(f"\nInitializing DeepPrint_{model_type} model...")
    extractor = create_extractor(
        localization,
        with_minutiae,
        num_training_subjects=num_train_subjects,
        embedding_dim=args.embedding_dim,
        loss_weights=loss_weights,
    )

    # Create training datasets: the pipelines of a training without a table,
    # over the loaders of the table
    print(f"Loading training image dataset from {args.dataset_dir}...")
    image_dataset = create_image_dataset(
        args.dataset_dir, training_ids, None, args.canvas_size, files
    )

    print("Creating minutia dataset...")
    if with_minutiae:
        print(f"Loading minutiae from the sample table {args.samples}")
        minutia_dataset = create_table_minutia_dataset(
            SampleTableMinutiaMapLoader(table),
            training_ids,
            image_dataset.data_loader,
            args.canvas_size,
        )
    else:
        print("No minutiae are used - texture-only mode")
        # The placeholder of a training without --minutiae_dir. What it prints
        # is about that option.
        with contextlib.redirect_stdout(io.StringIO()):
            minutia_dataset = create_minutia_dataset(
                None, training_ids, image_dataset.data_loader, args.canvas_size
            )

    print("Creating label dataset...")
    label_dataset = Dataset(LabelIndex(training_ids), training_ids)

    # Create validation dataset and benchmark if validation split is set
    validation_fingerprints = None
    validation_benchmark = None
    if validation_ids:
        print("\nCreating validation dataset...")
        # No augmentation for validation, as without a table
        validation_image_loader = create_validation_image_loader(
            args.dataset_dir, None, args.canvas_size, files
        )
        validation_fingerprints = Dataset(validation_image_loader, validation_ids)

        print("Creating validation benchmark...")
        validation_benchmark = create_table_validation_benchmark(
            validation_ids, seed=args.validation_seed
        )
        print("  Validation benchmark created successfully")

    return dict(
        settings=settings,
        extractor=extractor,
        image_dataset=image_dataset,
        minutia_dataset=minutia_dataset,
        label_dataset=label_dataset,
        validation_fingerprints=validation_fingerprints,
        validation_benchmark=validation_benchmark,
        model_type=model_type,
        log_path=log_path,
        epochs_before=epochs_before,
    )


def train_from_samples(
    parser: argparse.ArgumentParser, args: argparse.Namespace, first_process: bool
) -> None:
    """
    main() from the point where the two modes part, for table mode
    (--samples): the training set is what a sample table says, not what the
    names of the files in --dataset_dir and --minutiae_dir say.

    The options that both modes have are used as main() uses them, the start
    is that of main() with the table in the place of the two directories
    (_start_from_samples), and the training itself is the same code
    (_run_training).

    In a multi-GPU training every process reads the table and makes every
    check, and each one stops by itself with the exit status 2 and the reason
    if the table or its files cannot be used. What the start prints is
    printed by the first process only.
    """
    # Options that were not given are None and take the value from the config
    try:
        loss_weights = LossWeights(
            cross_entropy=args.loss_weight_cross_entropy,
            center=args.loss_weight_center,
            minutia_map=args.loss_weight_minutia_map,
        )
        learning_rate = LearningRateConfig(
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            schedule=args.lr_schedule,
            gradient_clip_norm=args.gradient_clip_norm,
            localization_factor=args.localization_lr_factor,
        )
    except ValueError as error:
        parser.error(str(error))
    quiet = contextlib.nullcontext() if first_process else contextlib.redirect_stdout(io.StringIO())
    try:
        with quiet:
            training = _start_from_samples(parser, args, loss_weights, learning_rate, first_process)
    except SampleTableError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    _run_training(args=args, learning_rate=learning_rate, first_process=first_process, **training)


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepPrint on a large fingerprint dataset",
        epilog="See the docstring at the top of this script and "
               "docs/training-guide.md; for --samples, docs/sample-table.md.",
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to fingerprint images directory")
    parser.add_argument("--samples", type=str, default=None,
                        help="Directory of a sample table (scripts/build_sample_table.py) that "
                             "describes the images below --dataset_dir: which finger and "
                             "capture each file shows, its resolution and ridge polarity, and "
                             "its minutiae. The files can then have any names and a subject "
                             "any number of impressions; --minutiae_dir, --num_subjects, "
                             "--num_impressions, --ppi, --ppi_manifest, --assume_ppi and "
                             "--ridges are refused, because the table says what they say "
                             "(default: none, the images are found by their names "
                             "<subject>_<impression>.png)")
    parser.add_argument("--allow_missing_minutiae", action="store_true",
                        help="With --samples: train although the table holds no minutiae "
                             "(null) for some training images. These images train without a "
                             "minutia map, with weight 0 in the minutia map loss (default: "
                             "such images stop the run before it trains)")
    parser.add_argument("--ignore_minutiae", action="store_true",
                        help="With --samples: train the texture-only variant although the "
                             "table holds minutiae (default: the minutia branch is trained "
                             "if the table holds minutiae for training images)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save trained model")
    parser.add_argument("--num_subjects", type=int, default=NUM_SUBJECTS,
                        help=f"Number of subjects (default: {NUM_SUBJECTS})")
    parser.add_argument("--num_impressions", type=int, default=NUM_IMPRESSIONS,
                        help=f"Impressions per subject (default: {NUM_IMPRESSIONS})")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                        help="Embedding dimensions PER BRANCH. With --minutiae_dir the model "
                             "has a texture and a minutia branch, so the embedding of a print "
                             f"has twice this many values (default: {EMBEDDING_DIM}, that is "
                             f"{2 * EMBEDDING_DIM} in total with minutiae)")
    parser.add_argument("--batch_size", type=int,
                        default=torch_helpers.DEFAULT_TRAINING_BATCH_SIZE,
                        help="Images per step on each GPU (default: "
                             f"{torch_helpers.DEFAULT_TRAINING_BATCH_SIZE}). With torchrun on "
                             "several GPUs one step sees this many images times the number "
                             "of GPUs. Validation uses twice the value")
    parser.add_argument("--allow_changed_settings", action="store_true",
                        help="Continue the training in --output_dir although settings that "
                             "must stay the same differ from training_settings.json")
    parser.add_argument("--minutiae_dir", type=str, default=None,
                        help="Path to extracted minutiae (.mnt files from extract_minutiae.py)")
    parser.add_argument("--localization", type=str, default=DEFAULT_LOCALIZATION,
                        choices=["on", "off"],
                        help="Train with (on) or without (off) the localization network, "
                             "the learned alignment of the image; everything else stays "
                             f"the same (default: {DEFAULT_LOCALIZATION})")
    parser.add_argument("--localization_lr_factor", type=float, default=None,
                        help="The localization network uses the learning rate times this "
                             "factor (default: LOCALIZATION_LEARNING_RATE_FACTOR in "
                             "flx/setup/config.py). Only used with --localization on")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Share of the subjects (not images) that is held out of the "
                             "training set for validation (default: 0.1, 0=no validation)")
    parser.add_argument("--max_validation_subjects", type=int,
                        default=DEFAULT_MAX_VALIDATION_SUBJECTS,
                        help="Upper limit for the number of validation subjects (default: "
                             f"{DEFAULT_MAX_VALIDATION_SUBJECTS}, 0 = no limit). Subjects above "
                             "the limit stay in the training set")
    parser.add_argument("--validation_seed", type=int, default=VALIDATION_SEED,
                        help="Seed of the random choice of the validation subjects and "
                             f"of the validation pairs (default: {VALIDATION_SEED})")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience - epochs without improvement (default: 3, 0=disabled)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate after the warm-up (default: LEARNING_RATE "
                             "in flx/setup/config.py). Find it with scripts/find_learning_rate.py")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of first steps over which the learning rate rises to "
                             "--learning_rate, 0=no warm-up (default: "
                             "LEARNING_RATE_WARMUP_STEPS in flx/setup/config.py)")
    parser.add_argument("--lr_schedule", type=str, default=None, choices=SCHEDULES,
                        help="Learning rate after the warm-up: constant, or cosine decay to "
                             "1%% in the last step (default: LEARNING_RATE_SCHEDULE in "
                             "flx/setup/config.py)")
    parser.add_argument("--gradient_clip_norm", type=float, default=None,
                        help="Upper limit for the gradient norm of one step, 0=no limit "
                             "(default: GRADIENT_CLIP_NORM in flx/setup/config.py)")
    parser.add_argument("--loss_weight_cross_entropy", type=float, default=None,
                        help="Factor of the cross entropy of each classifier in the training "
                             "loss (default: LOSS_WEIGHT_CROSS_ENTROPY in flx/setup/config.py)")
    parser.add_argument("--loss_weight_center", type=float, default=None,
                        help="Factor of the center loss of each embedding in the training "
                             "loss (default: LOSS_WEIGHT_CENTER in flx/setup/config.py)")
    parser.add_argument("--loss_weight_minutia_map", type=float, default=None,
                        help="Factor of the minutia map loss in the training loss, used with "
                             "--minutiae_dir (default: LOSS_WEIGHT_MINUTIA_MAP in "
                             "flx/setup/config.py)")
    sensor_normalization.add_arguments(parser)
    parser.add_argument("--canvas_size", type=int, default=0,
                        help="Cut or pad every 500 ppi image around its centre to a square "
                             "of this many pixels before it is scaled to the input size, so "
                             "that all sensors are shown at one scale (default: 0 = pad "
                             "every image to a square of its own longer side)")
    withheld_defaults = withhold_defaults(parser)
    args = parser.parse_args()
    check_options_of_the_mode(parser, args, withheld_defaults)
    if args.canvas_size < 0:
        parser.error("--canvas_size must not be negative")
    try:
        torch_helpers.set_training_batch_size(args.batch_size)
    except ValueError as error:
        parser.error(f"--{error}")
    first_process = _is_first_process()
    if args.samples is not None:
        # Table mode: from here on the two modes do different things with the
        # same options, up to the training itself
        return train_from_samples(parser, args, first_process)
    try:
        normalizer = sensor_normalization.from_arguments(args, args.dataset_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # Options that were not given are None and take the value from the config
    try:
        loss_weights = LossWeights(
            cross_entropy=args.loss_weight_cross_entropy,
            center=args.loss_weight_center,
            minutia_map=args.loss_weight_minutia_map,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        learning_rate = LearningRateConfig(
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            schedule=args.lr_schedule,
            gradient_clip_norm=args.gradient_clip_norm,
            localization_factor=args.localization_lr_factor,
        )
    except ValueError as error:
        parser.error(str(error))
    localization = args.localization == "on"
    with_minutiae = args.minutiae_dir is not None
    model_type = variant_name(localization, with_minutiae)

    # Create identifiers for training and validation. Whole subjects are held
    # out for validation; the model gets one class per training subject.
    try:
        training_ids, validation_ids = create_training_and_validation_ids(
            args.num_subjects,
            args.num_impressions,
            args.validation_split,
            args.validation_seed,
            args.max_validation_subjects,
        )
    except ValueError as error:
        parser.error(str(error))
    num_train_subjects = training_ids.num_subjects
    num_val_subjects = validation_ids.num_subjects if validation_ids else 0

    # The settings of the training. A training that is already in the output
    # directory is continued, so its settings must fit this command.
    preparation = normalizer.settings()
    settings = {
        "variant": model_type,
        "embedding_dim": args.embedding_dim,
        "num_subjects": args.num_subjects,
        "num_impressions": args.num_impressions,
        "num_training_subjects": num_train_subjects,
        "validation_split": args.validation_split,
        "max_validation_subjects": args.max_validation_subjects,
        "validation_seed": args.validation_seed,
        "batch_size": args.batch_size,
        "canvas_size": args.canvas_size,
        "preparation": preparation,
        "ppi_manifest": args.ppi_manifest,
        "dataset_dir": args.dataset_dir,
        "minutiae_dir": args.minutiae_dir,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "learning_rate": learning_rate.learning_rate,
        "warmup_steps": learning_rate.warmup_steps,
        "lr_schedule": learning_rate.schedule,
        "gradient_clip_norm": learning_rate.gradient_clip_norm,
        "localization_lr_factor": learning_rate.localization_factor if localization else None,
        "loss_weights": {
            "cross_entropy": loss_weights.cross_entropy,
            "center": loss_weights.center,
            "minutia_map": loss_weights.minutia_map if with_minutiae else None,
        },
    }
    try:
        notes = check_training_settings(
            args.output_dir, settings, args.allow_changed_settings
        )
    except SettingsChangedError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    if first_process:
        for note in notes:
            print(f"Note: {note}")
    log_path = os.path.join(args.output_dir, "log.json")
    epochs_before = len(TrainingLog(log_path)) if os.path.exists(log_path) else 0

    model_name = model_type + (" (texture + minutiae)" if with_minutiae else " (texture-only)")
    print(f"=" * 60)
    print(f"Training DeepPrint {'with' if localization else 'without'} "
          f"Learned Alignment ({model_name})")
    print(f"=" * 60)
    print(f"Dataset:      {args.dataset_dir}")
    print(f"Minutiae:     {args.minutiae_dir or 'None'}")
    print(f"Images:       {normalizer.describe()}")
    print(f"Canvas:       "
          + (f"{args.canvas_size}x{args.canvas_size} pixels at 500 ppi for every image"
             if args.canvas_size else "square of the longer side of each image"))
    print(f"Batch size:   {args.batch_size} images per step and GPU")
    print(f"Subjects:     {args.num_subjects:,} total")
    print(f"  Training:   {num_train_subjects:,} subjects")
    print(f"  Validation: {num_val_subjects:,} subjects held out (seed {args.validation_seed})")
    print(f"Impressions:  {args.num_impressions}")
    print(f"Total images: {args.num_subjects * args.num_impressions:,}")
    print(f"Epochs:       {args.num_epochs} (max)")
    if validation_ids:
        print(f"Patience:     {args.patience} epochs (early stopping)")
        print(f"Best model:   lowest validation equal error rate")
    else:
        print(f"Patience:     not used (no validation set)")
        print(f"Best model:   lowest training loss (no validation set)")
    print(f"Embedding:    {args.embedding_dim} values per branch, "
          f"{args.embedding_dim * (2 if with_minutiae else 1)} per print")
    print(f"Optimizer:    Adam, {learning_rate.describe()}")
    if localization:
        print(f"Alignment:    learned (--localization on), learning rate of the "
              f"localization network times {learning_rate.localization_factor:g}")
    else:
        print(f"Alignment:    none (--localization off)")
    print(f"Loss weights: cross entropy {loss_weights.cross_entropy:g}, "
          f"center {loss_weights.center:g}"
          + (f", minutia map {loss_weights.minutia_map:g}" if args.minutiae_dir else ""))
    print(f"Output:       {args.output_dir}")
    print(f"Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPUs:         {torch.cuda.device_count()}")
    print(f"=" * 60)

    wanted_validation = int(args.num_subjects * args.validation_split)
    if 0 < num_val_subjects < wanted_validation:
        print(
            f"\nNote: --validation_split {args.validation_split:g} would hold out "
            f"{wanted_validation:,} subjects; --max_validation_subjects limits this to "
            f"{num_val_subjects:,}. The others stay in the training set."
        )
    if num_val_subjects > DEFAULT_MAX_VALIDATION_SUBJECTS:
        print(
            f"\nWARNING: {num_val_subjects:,} validation subjects. All their "
            "impressions are embedded after every epoch, and the list of pairs is "
            "built in a time that grows with the square of this number (not "
            f"measured above {DEFAULT_MAX_VALIDATION_SUBJECTS:,}). "
            f"--max_validation_subjects {DEFAULT_MAX_VALIDATION_SUBJECTS} is the default."
        )
    if num_train_subjects > MANY_SUBJECTS:
        print(
            f"\nWARNING: {num_train_subjects:,} training subjects. The classifier and "
            "the centers of the loss have one row per subject and are not sharded: "
            "about 10 kB of GPU memory per subject with Adam (a calculation, not a "
            f"measurement), so about {num_train_subjects * 10 / 1e6:,.0f} GB here, in "
            "every GPU process, and model.pyt grows with it. This script was never "
            "run at this size. See \"Above about one million subjects\" in "
            "docs/large-datasets.md."
        )

    print(f"\nCreated {len(training_ids):,} training identifiers")
    if validation_ids:
        print(f"Created {len(validation_ids):,} validation identifiers")

    # The files that are there, compared with the files that the options ask for
    wanted = list(training_ids.identifiers)
    if validation_ids:
        wanted += list(validation_ids.identifiers)
    try:
        files = find_dataset_files(
            args.dataset_dir,
            args.minutiae_dir,
            normalizer,
            wanted,
            list(training_ids.identifiers),  # validation needs no minutiae
            args.num_subjects,
            args.num_impressions,
        )
    except DatasetLayoutError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    unused = len(files.ids) - len(wanted)
    if unused > 0 and first_process:
        print(
            f"Note: {unused:,} image files below {args.dataset_dir} are not used: their "
            "subject number is above --num_subjects or their impression number above "
            "--num_impressions."
        )
    if first_process:
        counts, looked_at = sample_resolution_sources(
            normalizer, [files._files.get(i) for i in training_ids.identifiers[:: max(1, len(training_ids) // 1000)]]
        )
        print(f"Resolution of {looked_at:,} training images spread over the set:")
        for line in sensor_normalization.describe_resolution_sources(
            counts, normalizer.assume_ppi
        ):
            print(line)

    # Initialize the model: with or without learned alignment
    # (LocalizationNetwork), with the minutia branch when minutiae are available
    print(f"\nInitializing DeepPrint_{model_type} model...")
    extractor = create_extractor(
        localization,
        with_minutiae,
        num_training_subjects=num_train_subjects,
        embedding_dim=args.embedding_dim,
        loss_weights=loss_weights,
    )

    # Create training datasets
    print(f"Loading training image dataset from {args.dataset_dir}...")
    image_dataset = create_image_dataset(
        args.dataset_dir, training_ids, normalizer, args.canvas_size, files
    )

    print(f"Creating minutia dataset...")
    minutia_dataset = create_minutia_dataset(
        args.minutiae_dir, training_ids, image_dataset.data_loader, args.canvas_size
    )

    print(f"Creating label dataset...")
    label_dataset = Dataset(LabelIndex(training_ids), training_ids)

    # Create validation dataset and benchmark if validation split is set
    validation_fingerprints = None
    validation_benchmark = None
    if validation_ids:
        print(f"\nCreating validation dataset...")
        # For validation, we don't use augmentation - just basic preprocessing
        validation_image_loader = create_validation_image_loader(
            args.dataset_dir, normalizer, args.canvas_size, files
        )
        validation_fingerprints = Dataset(validation_image_loader, validation_ids)

        print(f"Creating validation benchmark...")
        validation_benchmark = create_validation_benchmark(
            validation_ids, args.num_impressions, seed=args.validation_seed
        )
        print(f"  Validation benchmark created successfully")

    _run_training(
        args=args,
        settings=settings,
        extractor=extractor,
        image_dataset=image_dataset,
        minutia_dataset=minutia_dataset,
        label_dataset=label_dataset,
        validation_fingerprints=validation_fingerprints,
        validation_benchmark=validation_benchmark,
        learning_rate=learning_rate,
        model_type=model_type,
        log_path=log_path,
        epochs_before=epochs_before,
        first_process=first_process,
    )


def _run_training(
    args: argparse.Namespace,
    settings: dict,
    extractor,
    image_dataset: Dataset,
    minutia_dataset: Dataset,
    label_dataset: Dataset,
    validation_fingerprints: Dataset | None,
    validation_benchmark: VerificationBenchmark | None,
    learning_rate: LearningRateConfig,
    model_type: str,
    log_path: str,
    epochs_before: int,
    first_process: bool,
) -> None:
    """
    The end of main(), the same with and without a sample table: stores the
    settings of a new training, trains, stores the settings of a run that
    trained an epoch, and prints what was trained. Everything before it only
    checks and reads, so nothing is in --output_dir until this is called.

    log_path is log.json in --output_dir and epochs_before the number of
    epochs that it held when the script started.
    """
    # The settings are stored only now, after all checks: a start that was
    # refused leaves nothing behind that looks like a training. Those of a
    # training that is continued are replaced when this run trained an epoch.
    os.makedirs(args.output_dir, exist_ok=True)
    if not _has_training(args.output_dir):
        store_training_settings(args.output_dir, settings)

    # Train the model
    print(f"\nStarting training for up to {args.num_epochs} epochs...")
    if torch.cuda.device_count() > 1:
        print(f"For multi-GPU, launch with: torchrun --nproc_per_node={torch.cuda.device_count()} scripts/train_500k.py ...")
    print()

    def epochs_trained() -> int:
        if not os.path.exists(log_path):
            return 0
        return len(TrainingLog(log_path)) - epochs_before

    try:
        extractor.fit(
            fingerprints=image_dataset,
            minutia_maps=minutia_dataset,
            labels=label_dataset,
            validation_fingerprints=validation_fingerprints,
            validation_benchmark=validation_benchmark,
            num_epochs=args.num_epochs,
            out_dir=args.output_dir,
            patience=args.patience,
            learning_rate=learning_rate,
        )
    finally:
        # Also when the run is stopped or fails after it trained an epoch
        if first_process and epochs_trained() > 0:
            store_training_settings(args.output_dir, settings)

    if not first_process:
        return
    log = TrainingLog(log_path)
    trained = epochs_trained()
    print(f"\n{'=' * 60}")
    if trained == 0:
        print("Nothing was trained in this run: the training in the output directory")
        print(f"was already finished ({len(log)} epochs in its log, see the message above).")
        print("Options of this command that differ from that training had no effect")
        print(f"and were not stored in {SETTINGS_FILE}.")
        print("To train again with other settings, use a new --output_dir.")
    else:
        print(f"Training complete! {trained} epoch(s) trained in this run, "
              f"{len(log)} in the log.")
    print(f"Model type: DeepPrint_{model_type}")
    print(f"Model saved to: {args.output_dir}")
    if log.best_entry is not None:
        print(f"Best model: {get_best_model_file(args.output_dir)} "
              f"(epoch {log.best_entry.epoch})")
    print(f"Settings:   {os.path.join(args.output_dir, SETTINGS_FILE)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
