"""
Learning rate range test for scripts/train_500k.py

Run this before a training on a new dataset, model variant or batch size. It
reads the same dataset layout as train_500k.py (the loader, the augmentation
and the preprocessing are imported from that script), trains a fresh model for
a few hundred steps on a subset while raising the learning rate after every
step, and reports a learning rate to start with. No model is saved. With
--samples it reads the same sample table as train_500k.py (see below).

    python scripts/find_learning_rate.py \
        --dataset_dir /path/to/fingerprints \
        --minutiae_dir /path/to/minutiae \
        --num_subjects 50000 --num_impressions 10 \
        --max_subjects 2000 \
        --batch_size 64 \
        --ppi_manifest /path/to/resolutions.csv \
        --output_dir /path/to/lr_range_test

Output in --output_dir:
    lr_range_test.csv   one row per step: step, learning_rate, loss,
                        smoothed_loss, gradient_norm
    lr_range_test.png   loss against learning rate
The suggested rate is printed at the end. Pass it to train_500k.py with
--learning_rate. See "Choosing the learning rate" in docs/tuning.md
for how to read the plot.

The images are prepared as in train_500k.py (resampled to 500 pixels per inch,
dark ridges): give the same --ppi, --ppi_manifest, --assume_ppi, --ridges and
--canvas_size options as for the training. With a sample table give
--canvas_size only: the table states the resolution and the ridge polarity of
every file.

With a sample table (other file names, gaps in the numbers, uneven numbers of
impressions; see docs/sample-table.md):

    python scripts/find_learning_rate.py \
        --dataset_dir /path/to/fingerprints \
        --samples /path/to/table \
        --max_subjects 2000 \
        --batch_size 64 \
        --output_dir /path/to/lr_range_test

    Give the table that the planned training will read
    (scripts/build_sample_table.py builds it). It says what these options say
    without it, so together with --samples they are refused: --minutiae_dir,
    --num_subjects, --num_impressions, --ppi, --ppi_manifest, --assume_ppi
    and --ridges. --dataset_dir stays the directory of the images.

    The test uses all rows of the subjects that it chooses (--max_subjects of
    the subjects of the table) and follows the rules of the training, through
    the functions of train_500k.py: --variant auto has the minutia branch if
    the table holds minutiae for these rows, and --tolerate_missing_minutiae
    and --ignore_minutiae mean what they mean there. Give them to the test if
    the training will get them. Before the model is built and --output_dir
    is made, the test checks its rows as the training checks the rows of the
    table: the image files are there, the prepared images are large enough
    for the binarisation, the minutiae of every row are known (for a variant
    with minutiae), and images spread over the rows are loaded, which
    compares each of these files with its row.

    A sample table needs the package pyarrow (pip install -e ".[table]");
    without --samples the script does not need it.

The test runs in one process on one GPU (or on the CPU). Use the batch size of
the planned training. For a multi-GPU training, the batch size that counts is
the batch size per GPU times the number of GPUs.
"""

import argparse
import contextlib
import io
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_500k  # noqa: E402

from flx.data.dataset import Dataset, Identifier, IdentifierSet  # noqa: E402
from flx.data.label_index import LabelIndex  # noqa: E402
from flx.data.sample_table_loader import (  # noqa: E402
    SampleTableImageLoader,
    SampleTableMinutiaMapLoader,
)
from flx.extractor.fixed_length_extractor import (  # noqa: E402
    get_DeepPrint_LocTex,
    get_DeepPrint_LocTexMinu,
    get_DeepPrint_Tex,
    get_DeepPrint_TexMinu,
)
from flx.models.deep_print_loss import LossWeights  # noqa: E402
from flx.preparation import sensor_normalization  # noqa: E402
from flx.preparation.sample_table import SampleTableError  # noqa: E402
from flx.models.learning_rate_finder import (  # noqa: E402
    run_range_test,
    save_range_test_csv,
    save_range_test_plot,
    suggest_learning_rate,
)

# Model variants: name -> (constructor, needs minutiae)
VARIANTS = {
    "LocTexMinu": (get_DeepPrint_LocTexMinu, True),
    "TexMinu": (get_DeepPrint_TexMinu, True),
    "LocTex": (get_DeepPrint_LocTex, False),
    "Tex": (get_DeepPrint_Tex, False),
}


def choose_subjects(num_subjects: int, max_subjects: int, seed: int) -> list[int]:
    """
    The subjects (0-indexed) that the test uses: all of them, or a random
    choice of max_subjects if that is above 0 and below num_subjects.
    """
    subjects = list(range(num_subjects))
    if 0 < max_subjects < num_subjects:
        subjects = sorted(random.Random(seed).sample(subjects, max_subjects))
    return subjects


def choose_table_subjects(subjects: list[int], max_subjects: int, seed: int) -> list[int]:
    """
    choose_subjects for the subjects of a sample table, whose numbers can
    have gaps: all of them in ascending order, or a random choice of
    max_subjects if that is above 0 and below their number. The choice is
    made among the subjects in ascending order, so for the subjects
    0 .. n - 1 it is that of choose_subjects for n.
    """
    subjects = sorted(subjects)
    if 0 < max_subjects < len(subjects):
        subjects = sorted(random.Random(seed).sample(subjects, max_subjects))
    return subjects


def build_extractor(
    variant: str,
    num_subjects: int,
    embedding_dim: int,
    loss_weights: LossWeights | None = None,
):
    constructor, _ = VARIANTS[variant]
    return constructor(num_subjects, embedding_dim, loss_weights=loss_weights)


def build_training_set(
    dataset_dir: str,
    minutiae_dir: str | None,
    ids: IdentifierSet,
    with_minutiae: bool,
    normalizer=None,
    canvas_size: int = 0,
    files=None,
) -> Dataset:
    """
    The zipped training set of train_model, built as train_500k.py builds it.
    normalizer, canvas_size and files are those of
    train_500k.create_image_dataset.
    """
    images = train_500k.create_image_dataset(
        dataset_dir, ids, normalizer, canvas_size, files
    )
    minutiae = train_500k.create_minutia_dataset(
        minutiae_dir if with_minutiae else None, ids, images.data_loader, canvas_size
    )
    labels = Dataset(LabelIndex(ids), ids)
    return Dataset.zip(images, minutiae, labels)


def build_table_training_set(
    files: SampleTableImageLoader,
    ids: IdentifierSet,
    with_minutiae: bool,
    canvas_size: int = 0,
) -> Dataset:
    """
    build_training_set for rows of a sample table, built as train_500k.py
    builds the training set of a table: the images through the loader of the
    table (files), and for a variant with minutiae the minutiae that the
    table holds for them. A variant without minutiae gets the placeholder
    that it gets without a table, also when the table holds minutiae.
    """
    table = files.table
    images = train_500k.create_image_dataset(
        table.image_dir, ids, None, canvas_size, files
    )
    if with_minutiae:
        # Over the table of the image loader: a worker process that is not
        # forked then gets one copy of the table for both loaders
        minutiae = train_500k.create_table_minutia_dataset(
            SampleTableMinutiaMapLoader(table), ids, images.data_loader, canvas_size
        )
    else:
        # What this prints is about --minutiae_dir, which a table replaces
        with contextlib.redirect_stdout(io.StringIO()):
            minutiae = train_500k.create_minutia_dataset(
                None, ids, images.data_loader, canvas_size
            )
    labels = Dataset(LabelIndex(ids), ids)
    return Dataset.zip(images, minutiae, labels)


def range_test_from_samples(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """
    main() from the point where the two modes part, for table mode
    (--samples): the images of the test are rows of a sample table, and not
    the files that the names in --dataset_dir and --minutiae_dir give.

    The test is to see what the training from the same table will see. So
    the table is read, the variant is chosen and the rows are checked by the
    rules of train_500k.py and through its functions. The table is read and
    checked as a whole. The rules about files and minutiae are applied to the
    rows that the test uses, which are all rows of the chosen subjects: a
    missing file, an image that is too small or minutiae that are not known
    do not stop the test where they belong to a row of another subject.

    A table, a row or a file that cannot be used stops the test with the exit
    status 2 and the reason, before the model is built and before
    --output_dir is made.
    """
    try:
        loss_weights = LossWeights(
            cross_entropy=args.loss_weight_cross_entropy,
            center=args.loss_weight_center,
            minutia_map=args.loss_weight_minutia_map,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.ignore_minutiae and args.variant != "auto" and VARIANTS[args.variant][1]:
        parser.error(
            f"--variant {args.variant} has the minutia branch, and --ignore_minutiae leaves "
            "the minutiae of the sample table out: give one of the two. With "
            "--ignore_minutiae train_500k.py trains a variant without minutiae, which "
            "--variant auto chooses here."
        )
    try:
        table = train_500k.read_sample_table(args.samples, args.dataset_dir)
        # The subjects of the test, and all their rows in the order of the table
        subjects = choose_table_subjects(
            table.subjects.tolist(), args.max_subjects, args.seed
        )
        rows = table.rows_of(subjects)
        ids = IdentifierSet(table.identifiers_of(subjects))

        has_minutiae = train_500k.table_has_minutiae(table, rows)
        variant = args.variant
        if variant == "auto":
            # The variant that train_500k.py trains by default: with the
            # minutia branch if the table holds minutiae for the rows that
            # are used, unless they are to be ignored
            variant = train_500k.variant_name(
                train_500k.DEFAULT_LOCALIZATION == "on",
                has_minutiae and not args.ignore_minutiae,
            )
        with_minutiae = VARIANTS[variant][1]
        if with_minutiae and not has_minutiae:
            parser.error(
                f"--variant {variant} needs minutiae, and the sample table {args.samples} "
                f"holds none (null) for the {len(rows):,} images of the {len(subjects):,} "
                "subjects that the test uses. Build the table with the .mnt files of the "
                "images (--minutiae_dir of scripts/build_sample_table.py), or test a variant "
                "without minutiae: --variant auto chooses the one that train_500k.py trains "
                "by default from these rows."
            )

        print(f"Sample table {args.samples}: {table.num_rows:,} rows of "
              f"{len(table.subjects):,} subjects; the test uses {len(rows):,} rows of "
              f"{len(subjects):,} subjects")
        print(f"Range test for DeepPrint_{variant}: {len(subjects):,} subjects, "
              f"{len(ids):,} images, batch size {args.batch_size}, "
              f"{args.num_steps} steps from {args.min_lr:g} to {args.max_lr:g}")
        if args.num_steps * args.batch_size > len(ids):
            print(f"Note: the test needs {args.num_steps * args.batch_size:,} images and "
                  f"sees each of the {len(ids):,} images more than once.")

        # The checks of the start of a training from a table, in its order,
        # for the rows of the test: the files are there, the prepared images
        # are large enough for the binarisation, a variant with minutiae has
        # the minutiae of every row unless rows without are tolerated, and
        # images spread over the rows are loaded and compared with their rows
        train_500k.check_table_files(table, rows)
        train_500k.check_table_image_sizes(table, rows)
        if with_minutiae:
            train_500k.check_table_minutiae(table, rows, args.tolerate_missing_minutiae)
        files = SampleTableImageLoader(table)
        train_500k.check_table_images(files, None, rows)
    except SampleTableError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    if with_minutiae:
        filled, empty, unknown = train_500k.count_table_minutiae(table, rows)
        if unknown:
            print(
                f"Note: the sample table holds no minutiae (null) for {unknown:,} of the "
                f"{len(rows):,} images of the test. They are used without a minutia map "
                "(--tolerate_missing_minutiae), with weight 0 in the minutia map loss, as "
                "in the training."
            )
        if not filled:
            print(
                f"\nWARNING: DeepPrint_{variant} has the minutia branch, but none of the "
                f"{len(rows):,} images of the test has a minutia: the sample table holds an "
                f"empty list for {empty:,} of them and null for {unknown:,}. Such an image "
                "has the weight 0 in the minutia map loss, so the test measures a loss "
                "without that part. If the images have minutiae, extract them and build the "
                "table again (scripts/build_sample_table.py); --ignore_minutiae with "
                "--variant auto tests the texture-only variant instead."
            )
    if args.num_workers > 0:
        # The images are loaded in worker processes, as in the training
        train_500k.warn_if_workers_are_not_forked("the range test")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    extractor = build_extractor(
        variant, len(subjects), args.embedding_dim, loss_weights
    )
    train_set = build_table_training_set(files, ids, with_minutiae, args.canvas_size)
    _run_and_report(args, variant, extractor, train_set)


def main():
    parser = argparse.ArgumentParser(
        description="Learning rate range test on the dataset layout of train_500k.py"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to fingerprint images directory")
    parser.add_argument("--samples", type=str, default=None,
                        help="Directory of the sample table that the planned training reads "
                             "(scripts/build_sample_table.py builds it). The test then uses "
                             "rows of the table: all rows of the subjects that it chooses. "
                             "The image files below --dataset_dir can have any names then; "
                             "--minutiae_dir, --num_subjects, --num_impressions, --ppi, "
                             "--ppi_manifest, --assume_ppi and --ridges are refused, because "
                             "the table says what they say (default: none, the images are "
                             "found by their names <subject>_<impression>.png)")
    parser.add_argument("--tolerate_missing_minutiae", action="store_true",
                        help="With --samples, as for train_500k.py: test a variant with "
                             "minutiae although the table holds no minutiae (null) for some "
                             "images of the test. These images are used without a minutia "
                             "map, with weight 0 in the minutia map loss (default: such "
                             "images stop the test before the model is built)")
    parser.add_argument("--ignore_minutiae", action="store_true",
                        help="With --samples, as for train_500k.py: --variant auto is a "
                             "variant without minutiae although the table holds minutiae. "
                             "Not together with a variant with minutiae that is given by "
                             "name (default: auto has the minutia branch if the table holds "
                             "minutiae for images of the test)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for lr_range_test.csv and lr_range_test.png")
    parser.add_argument("--minutiae_dir", type=str, default=None,
                        help="Path to extracted minutiae (.mnt files)")
    parser.add_argument("--num_subjects", type=int, default=train_500k.NUM_SUBJECTS,
                        help="Number of subjects in the dataset "
                             f"(default: {train_500k.NUM_SUBJECTS})")
    parser.add_argument("--num_impressions", type=int, default=train_500k.NUM_IMPRESSIONS,
                        help="Impressions per subject "
                             f"(default: {train_500k.NUM_IMPRESSIONS})")
    parser.add_argument("--max_subjects", type=int, default=2000,
                        help="Use a random choice of this many subjects "
                             "(default: 2000, 0 = all subjects)")
    parser.add_argument("--embedding_dim", type=int, default=train_500k.EMBEDDING_DIM,
                        help="Embedding dimensions PER BRANCH, as in the planned "
                             f"training (default: {train_500k.EMBEDDING_DIM})")
    parser.add_argument("--variant", type=str, default="auto",
                        choices=["auto"] + list(VARIANTS),
                        help="Model variant; use the one of the planned training. "
                             "Loc = with localization network (train_500k.py "
                             "--localization on), Minu = with minutiae. auto = what "
                             "train_500k.py trains by default, with the minutia branch "
                             "if --minutiae_dir is given, or if the sample table of "
                             "--samples holds minutiae for images of the test")
    parser.add_argument("--localization_lr_factor", type=float, default=None,
                        help="The localization network of the Loc variants uses the "
                             "learning rate times this factor; use the value of the planned "
                             "training (default: LOCALIZATION_LEARNING_RATE_FACTOR in "
                             "flx/setup/config.py)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size; use the one of the planned training (default: 64)")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help="Learning rate of the first step (default: 1e-7)")
    parser.add_argument("--max_lr", type=float, default=1.0,
                        help="Learning rate of the last step (default: 1.0)")
    parser.add_argument("--num_steps", type=int, default=300,
                        help="Number of steps (default: 300)")
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0,
                        help="Limit for the gradient norm, as in the planned "
                             "training (default: 0 = no limit)")
    parser.add_argument("--loss_weight_cross_entropy", type=float, default=None,
                        help="Loss weights as in the planned training; see the "
                             "options of the same name of train_500k.py "
                             "(default: the values in flx/setup/config.py)")
    parser.add_argument("--loss_weight_center", type=float, default=None,
                        help="Weight of the center loss, as in the planned training "
                             "(default: LOSS_WEIGHT_CENTER in flx/setup/config.py)")
    parser.add_argument("--loss_weight_minutia_map", type=float, default=None,
                        help="Weight of the minutia map loss, as in the planned training "
                             "(default: LOSS_WEIGHT_MINUTIA_MAP in flx/setup/config.py)")
    parser.add_argument("--smoothing", type=float, default=0.9,
                        help="Weight of the past in the moving average of the loss (default: 0.9)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Processes that load images (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the subject choice, the model and the batch order "
                             "(default: 0)")
    sensor_normalization.add_arguments(parser)
    parser.add_argument("--canvas_size", type=int, default=0,
                        help="As for train_500k.py: side in pixels of the square that every "
                             "500 ppi image is cut or padded to (default: 0 = off)")
    withheld_defaults = train_500k.withhold_defaults(parser)
    args = parser.parse_args()
    train_500k.check_options_of_the_mode(parser, args, withheld_defaults)
    if args.canvas_size < 0:
        parser.error("--canvas_size must not be negative")
    if args.samples is not None:
        # Table mode: from here on the two modes do different things with the
        # same options, up to the test itself
        return range_test_from_samples(parser, args)
    try:
        normalizer = sensor_normalization.from_arguments(args, args.dataset_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        loss_weights = LossWeights(
            cross_entropy=args.loss_weight_cross_entropy,
            center=args.loss_weight_center,
            minutia_map=args.loss_weight_minutia_map,
        )
    except ValueError as error:
        parser.error(str(error))

    variant = args.variant
    if variant == "auto":
        # The variant that train_500k.py trains by default
        variant = train_500k.variant_name(
            train_500k.DEFAULT_LOCALIZATION == "on", args.minutiae_dir is not None
        )
    with_minutiae = VARIANTS[variant][1]
    if with_minutiae and args.minutiae_dir is None:
        parser.error(f"--variant {variant} needs --minutiae_dir")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    subjects = choose_subjects(args.num_subjects, args.max_subjects, args.seed)
    ids = IdentifierSet(
        [Identifier(s, i) for s in subjects for i in range(args.num_impressions)]
    )
    print(f"Range test for DeepPrint_{variant}: {len(subjects):,} subjects, "
          f"{len(ids):,} images, batch size {args.batch_size}, "
          f"{args.num_steps} steps from {args.min_lr:g} to {args.max_lr:g}")
    if args.num_steps * args.batch_size > len(ids):
        print(f"Note: the test needs {args.num_steps * args.batch_size:,} images and "
              f"sees each of the {len(ids):,} images more than once.")

    # The same check of the file layout as in train_500k.py
    try:
        files = train_500k.find_dataset_files(
            args.dataset_dir,
            args.minutiae_dir if with_minutiae else None,
            normalizer,
            list(ids.identifiers),
            list(ids.identifiers),
            args.num_subjects,
            args.num_impressions,
        )
    except train_500k.DatasetLayoutError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    extractor = build_extractor(
        variant, len(subjects), args.embedding_dim, loss_weights
    )
    train_set = build_training_set(
        args.dataset_dir, args.minutiae_dir, ids, with_minutiae,
        normalizer, args.canvas_size, files,
    )
    _run_and_report(args, variant, extractor, train_set)


def _run_and_report(
    args: argparse.Namespace, variant: str, extractor, train_set: Dataset
) -> None:
    """
    The end of main(), the same with and without a sample table: runs the
    range test, writes lr_range_test.csv and lr_range_test.png to
    --output_dir and prints the suggestion.
    """
    result = run_range_test(
        extractor.model,
        extractor.loss,
        train_set,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm or None,
        smoothing=args.smoothing,
        num_workers=args.num_workers,
        seed=args.seed,
        localization_factor=args.localization_lr_factor,
    )

    csv_path = os.path.join(args.output_dir, "lr_range_test.csv")
    plot_path = os.path.join(args.output_dir, "lr_range_test.png")
    save_range_test_csv(csv_path, result, args.smoothing)
    print(f"\nEnded after {len(result.losses)} steps at learning rate "
          f"{result.learning_rates[-1]:.3g}: {result.stop_reason}")
    print(f"Loss per step: {csv_path}")
    try:
        suggestion = suggest_learning_rate(
            result.learning_rates, result.losses, args.smoothing
        )
    except ValueError as error:
        print(f"No suggestion: {error}")
        return
    save_range_test_plot(
        plot_path, result, suggestion, args.smoothing,
        title=f"Learning rate range test, DeepPrint_{variant}, batch size {args.batch_size}",
    )
    print(f"Plot:          {plot_path}")
    print(suggestion.explanation)
    if suggestion.suggested_learning_rate is not None:
        print(f"\nSuggested: --learning_rate {suggestion.suggested_learning_rate:.2g}")
        print("This value holds for this data, this model variant and this batch "
              "size only. Repeat the test when one of them changes.")


if __name__ == "__main__":
    main()
