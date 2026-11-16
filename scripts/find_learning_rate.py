"""
Learning rate range test for scripts/train_500k.py

Run this before a training on a new dataset, model variant or batch size. It
reads the same dataset layout as train_500k.py (the loader, the augmentation
and the preprocessing are imported from that script), trains a fresh model for
a few hundred steps on a subset while raising the learning rate after every
step, and reports a learning rate to start with. No model is saved.

    python scripts/find_learning_rate.py \
        --dataset_dir /path/to/fingerprints \
        --minutiae_dir /path/to/minutiae \
        --num_subjects 50000 --num_impressions 10 \
        --max_subjects 2000 \
        --batch_size 64 \
        --output_dir /path/to/lr_range_test

Output in --output_dir:
    lr_range_test.csv   one row per step: step, learning_rate, loss,
                        smoothed_loss, gradient_norm
    lr_range_test.png   loss against learning rate
The suggested rate is printed at the end. Pass it to train_500k.py with
--learning_rate. See "Choosing the learning rate" in TRAINING_PIPELINE_GUIDE.md
for how to read the plot.

The images are prepared as in train_500k.py (resampled to 500 pixels per inch,
dark ridges): give the same --ppi, --ppi_manifest, --assume_ppi, --ridges and
--canvas_size options as for the training.

The test runs in one process on one GPU (or on the CPU). Use the batch size of
the planned training. For a multi-GPU training, the batch size that counts is
the batch size per GPU times the number of GPUs.
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_500k  # noqa: E402

from flx.data.dataset import Dataset, Identifier, IdentifierSet  # noqa: E402
from flx.data.label_index import LabelIndex  # noqa: E402
from flx.extractor.fixed_length_extractor import (  # noqa: E402
    get_DeepPrint_LocTex,
    get_DeepPrint_LocTexMinu,
    get_DeepPrint_Tex,
    get_DeepPrint_TexMinu,
)
from flx.models.deep_print_loss import LossWeights  # noqa: E402
from flx.preparation import sensor_normalization  # noqa: E402
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
) -> Dataset:
    """
    The zipped training set of train_model, built as train_500k.py builds it.
    normalizer and canvas_size are those of train_500k.create_image_dataset.
    """
    images = train_500k.create_image_dataset(dataset_dir, ids, normalizer, canvas_size)
    minutiae = train_500k.create_minutia_dataset(
        minutiae_dir if with_minutiae else None, ids, images.data_loader, canvas_size
    )
    labels = Dataset(LabelIndex(ids), ids)
    return Dataset.zip(images, minutiae, labels)


def main():
    parser = argparse.ArgumentParser(
        description="Learning rate range test on the dataset layout of train_500k.py"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to fingerprint images directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for lr_range_test.csv and lr_range_test.png")
    parser.add_argument("--minutiae_dir", type=str, default=None,
                        help="Path to extracted minutiae (.mnt files)")
    parser.add_argument("--num_subjects", type=int, default=train_500k.NUM_SUBJECTS,
                        help="Number of subjects in the dataset")
    parser.add_argument("--num_impressions", type=int, default=train_500k.NUM_IMPRESSIONS,
                        help="Impressions per subject")
    parser.add_argument("--max_subjects", type=int, default=2000,
                        help="Use a random choice of this many subjects "
                             "(default: 2000, 0 = all subjects)")
    parser.add_argument("--embedding_dim", type=int, default=train_500k.EMBEDDING_DIM)
    parser.add_argument("--variant", type=str, default="auto",
                        choices=["auto"] + list(VARIANTS),
                        help="Model variant; use the one of the planned training. "
                             "Loc = with localization network (train_500k.py "
                             "--localization on), Minu = with minutiae. auto = what "
                             "train_500k.py trains by default, with the minutia branch "
                             "if --minutiae_dir is given")
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
    parser.add_argument("--loss_weight_center", type=float, default=None)
    parser.add_argument("--loss_weight_minutia_map", type=float, default=None)
    parser.add_argument("--smoothing", type=float, default=0.9,
                        help="Weight of the past in the moving average of the loss (default: 0.9)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Processes that load images (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the subject choice, the model and the batch order")
    sensor_normalization.add_arguments(parser)
    parser.add_argument("--canvas_size", type=int, default=0,
                        help="As for train_500k.py: side in pixels of the square that every "
                             "500 ppi image is cut or padded to (default: 0 = off)")
    args = parser.parse_args()
    if args.canvas_size < 0:
        parser.error("--canvas_size must not be negative")
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

    extractor = build_extractor(
        variant, len(subjects), args.embedding_dim, loss_weights
    )
    train_set = build_training_set(
        args.dataset_dir, args.minutiae_dir, ids, with_minutiae,
        normalizer, args.canvas_size,
    )
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
