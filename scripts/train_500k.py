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
    an uneven number of impressions per subject, rename (or link) the files
    to this layout, or change CustomFingerprintLoader and the identifier
    lists in this script.

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

    training_settings.json holds the settings of the last run that trained
    an epoch. A start that is refused (settings, missing files) or that
    trains nothing does not write it.

    --batch_size 64                 images per step on each GPU (default 64)
    --allow_changed_settings        continue although settings that must stay
                                    the same differ (at your own risk)

Images from different sensors (see Step 2 of docs/training-guide.md):
    Every image is loaded like scripts/extract_minutiae.py loads it: resampled
    to 500 pixels per inch (ppi) and brought to dark ridges. The minutiae in
    the .mnt files are in the pixels of that image. Give this script the same
    options as the extraction:

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
    validation equal error rates (the guide has the recipe). A model must be
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

    torchrun --nproc_per_node=4 scripts/train_500k.py \
        --dataset_dir /path/to/fingerprints \
        --output_dir /path/to/model \
        --validation_split 0.1 \
        --max_validation_subjects 5000 \
        --patience 3

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
import functools
import json
import random

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
from flx.preparation.sensor_normalization import SensorNormalizer


# =============================================================================
# CONFIGURATION - Adjust these for your dataset
# =============================================================================

# Dataset structure
NUM_SUBJECTS = 50000          # Number of unique fingerprint subjects
NUM_IMPRESSIONS = 10          # Impressions per subject (50000 * 10 = 500K)

# Training hyperparameters
NUM_EPOCHS = 10               # Fewer epochs needed with large dataset
EMBEDDING_DIM = 256           # Dimension of the fixed-length representation

# Learned alignment: "on" trains with the localization network, "off" without
DEFAULT_LOCALIZATION = "on"

# Image preprocessing
BINARIZATION_THRESHOLD = 5.0  # Adjust based on your image quality

# Data augmentation - helps LocalizationNetwork learn robust alignment
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


def load_trained_model(
    output_dir: str,
    dataset_dir: str,
    ppi_manifest: str | None = None,
    best: bool = True,
):
    """
    The model of a training of this script and a loader that prepares images
    as that training did (same resolution sources, ridge polarity and canvas,
    no augmentation), read from training_settings.json in output_dir.

    Args:
        output_dir: The --output_dir of the training
        dataset_dir: Directory with the images to embed, in the layout that
            this script reads (<subject>_<impression>.png)
        ppi_manifest: Manifest for the images in dataset_dir, if the training
            used one. None uses the path that the training was given, which
            only fits if dataset_dir is the directory of the training.
        best: Load best_model.pyt (True) or model.pyt, the newest epoch

    Returns:
        (extractor, image_loader). Embeddings:
            dataset = Dataset(image_loader, image_loader.ids)
            texture, minutia = extractor.extract(dataset)
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


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepPrint on a large fingerprint dataset",
        epilog="See the docstring at the top of this script and "
               "docs/training-guide.md.",
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to fingerprint images directory")
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
    args = parser.parse_args()
    if args.canvas_size < 0:
        parser.error("--canvas_size must not be negative")
    try:
        torch_helpers.set_training_batch_size(args.batch_size)
    except ValueError as error:
        parser.error(f"--{error}")
    first_process = _is_first_process()
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
