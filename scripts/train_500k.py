"""
Training script for large-scale fingerprint dataset (500K images)
Uses DeepPrint with learned alignment (LocalizationNetwork)

Single GPU usage:
    python scripts/train_500k.py \
        --dataset_dir /path/to/fingerprints \
        --output_dir /path/to/model

Multi-GPU usage (recommended for large datasets):
    torchrun --nproc_per_node=NUM_GPUS scripts/train_500k.py \
        --dataset_dir /path/to/fingerprints \
        --output_dir /path/to/model

With FingerFlow-extracted minutiae:
    torchrun --nproc_per_node=4 scripts/train_500k.py \
        --dataset_dir /path/to/fingerprints \
        --minutiae_dir /path/to/minutiae \
        --output_dir /path/to/model

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
                                    subjects (default 0 = no limit). Set this
                                    for large training sets: validation runs
                                    after every epoch on all impressions of
                                    the held-out subjects, and every subject
                                    above the limit stays in the training set.
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
"""

import os
import argparse
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
from flx.extractor.fixed_length_extractor import get_DeepPrint_LocTexMinu, get_DeepPrint_LocTex
from flx.image_processing.augmentation import RandomPoseTransform, RandomQualityTransform
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


# =============================================================================
# CONFIGURATION - Adjust these for your dataset
# =============================================================================

# Dataset structure
NUM_SUBJECTS = 50000          # Number of unique fingerprint subjects
NUM_IMPRESSIONS = 10          # Impressions per subject (50000 * 10 = 500K)

# Training hyperparameters
NUM_EPOCHS = 10               # Fewer epochs needed with large dataset
EMBEDDING_DIM = 256           # Dimension of the fixed-length representation

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
    """

    @staticmethod
    def _extension() -> str:
        # Change this to match your image format (.png, .bmp, .tif, etc.)
        return ".png"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Modify this to parse your filename pattern
        # Example pattern: "<subject_id>_<impression_id>.png"
        subject_id, impression_id = filename.split("_")
        # Convert to 0-indexed for PyTorch compatibility
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

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

# Above this number of validation subjects the script recommends a limit
LARGE_VALIDATION_SET = 5000


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


def create_image_dataset(dataset_dir: str, training_ids: IdentifierSet) -> Dataset:
    """Create image dataset with augmentation pipeline."""
    image_loader = TransformedImageLoader(
        images=CustomFingerprintLoader(dataset_dir),
        poses=POSE_AUGMENTATION,  # Random pose for training robustness
        transforms=[
            QUALITY_AUGMENTATION,
            LazilyAllocatedBinarizer(BINARIZATION_THRESHOLD),
            pad_and_resize_to_deepprint_input_size,
        ],
    )
    return Dataset(image_loader, training_ids)


def create_minutia_dataset(
    minutiae_dir: str,
    training_ids: IdentifierSet,
    images: TransformedImageLoader,
) -> Dataset:
    """
    Create minutia map dataset.

    Args:
        minutiae_dir: Directory containing .mnt files (from extract_minutiae.py)
                      If None, uses empty tensors (model still works, texture-only)
        training_ids: Identifiers for training samples
        images: Loader of the training images. The minutia map of a sample gets
                the random pose of its image.

    Returns:
        Dataset containing minutia maps
    """
    if minutiae_dir is not None:
        print(f"Loading minutiae from {minutiae_dir}")
        minutia_loader = PoseTransformedMinutiaMapLoader(
            FingerFlowMinutiaMapLoader(minutiae_dir), images
        )
        return Dataset(minutia_loader, training_ids)
    else:
        # No minutiae available - use empty placeholder
        print("No minutiae directory provided - using texture-only mode")
        return Dataset(ConstantDataLoader((torch.tensor([]), 0.0)), training_ids)


def main():
    parser = argparse.ArgumentParser(description="Train DeepPrint on 500K fingerprints")
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
                        help=f"Embedding dimension (default: {EMBEDDING_DIM})")
    parser.add_argument("--minutiae_dir", type=str, default=None,
                        help="Path to extracted minutiae (.mnt files from extract_minutiae.py)")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Share of the subjects (not images) that is held out of the "
                             "training set for validation (default: 0.1, 0=no validation)")
    parser.add_argument("--max_validation_subjects", type=int, default=0,
                        help="Upper limit for the number of validation subjects "
                             "(default: 0 = no limit). Recommended for large sets, e.g. 5000")
    parser.add_argument("--validation_seed", type=int, default=VALIDATION_SEED,
                        help="Seed of the random choice of the validation subjects and "
                             f"of the validation pairs (default: {VALIDATION_SEED})")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience - epochs without improvement (default: 3, 0=disabled)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create identifiers for training and validation. Whole subjects are held
    # out for validation; the model gets one class per training subject.
    training_ids, validation_ids = create_training_and_validation_ids(
        args.num_subjects,
        args.num_impressions,
        args.validation_split,
        args.validation_seed,
        args.max_validation_subjects,
    )
    num_train_subjects = training_ids.num_subjects
    num_val_subjects = validation_ids.num_subjects if validation_ids else 0

    model_name = "LocTexMinu (texture + minutiae)" if args.minutiae_dir else "LocTex (texture-only)"
    print(f"=" * 60)
    print(f"Training DeepPrint with Learned Alignment ({model_name})")
    print(f"=" * 60)
    print(f"Dataset:      {args.dataset_dir}")
    print(f"Minutiae:     {args.minutiae_dir or 'None'}")
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
    print(f"Embedding:    {args.embedding_dim} dims")
    print(f"Output:       {args.output_dir}")
    print(f"Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPUs:         {torch.cuda.device_count()}")
    print(f"=" * 60)

    if num_val_subjects > LARGE_VALIDATION_SET and args.max_validation_subjects == 0:
        print(
            f"\nNote: {num_val_subjects:,} validation subjects are embedded and "
            f"compared after every epoch. --max_validation_subjects {LARGE_VALIDATION_SET} "
            f"keeps validation short and leaves the other subjects in the training set."
        )

    print(f"\nCreated {len(training_ids):,} training identifiers")
    if validation_ids:
        print(f"Created {len(validation_ids):,} validation identifiers")

    # Initialize model with learned alignment (LocalizationNetwork)
    # Use LocTexMinu when minutiae are available, LocTex otherwise
    if args.minutiae_dir is not None:
        print(f"\nInitializing DeepPrint_LocTexMinu model (texture + minutiae)...")
        extractor = get_DeepPrint_LocTexMinu(
            num_training_subjects=num_train_subjects,
            num_dims=args.embedding_dim,
        )
    else:
        print(f"\nInitializing DeepPrint_LocTex model (texture-only)...")
        extractor = get_DeepPrint_LocTex(
            num_training_subjects=num_train_subjects,
            num_texture_dims=args.embedding_dim,
        )

    # Create training datasets
    print(f"Loading training image dataset from {args.dataset_dir}...")
    image_dataset = create_image_dataset(args.dataset_dir, training_ids)

    print(f"Creating minutia dataset...")
    minutia_dataset = create_minutia_dataset(
        args.minutiae_dir, training_ids, image_dataset.data_loader
    )

    print(f"Creating label dataset...")
    label_dataset = Dataset(LabelIndex(training_ids), training_ids)

    # Create validation dataset and benchmark if validation split is set
    validation_fingerprints = None
    validation_benchmark = None
    if validation_ids:
        print(f"\nCreating validation dataset...")
        # For validation, we don't use augmentation - just basic preprocessing
        validation_image_loader = TransformedImageLoader(
            images=CustomFingerprintLoader(args.dataset_dir),
            poses=None,  # No pose augmentation for validation
            transforms=[
                LazilyAllocatedBinarizer(BINARIZATION_THRESHOLD),
                pad_and_resize_to_deepprint_input_size,
            ],
        )
        validation_fingerprints = Dataset(validation_image_loader, validation_ids)

        print(f"Creating validation benchmark...")
        validation_benchmark = create_validation_benchmark(
            validation_ids, args.num_impressions, seed=args.validation_seed
        )
        print(f"  Validation benchmark created successfully")

    # Train the model
    print(f"\nStarting training for up to {args.num_epochs} epochs...")
    if torch.cuda.device_count() > 1:
        print(f"For multi-GPU, launch with: torchrun --nproc_per_node={torch.cuda.device_count()} scripts/train_500k.py ...")
    print()

    extractor.fit(
        fingerprints=image_dataset,
        minutia_maps=minutia_dataset,
        labels=label_dataset,
        validation_fingerprints=validation_fingerprints,
        validation_benchmark=validation_benchmark,
        num_epochs=args.num_epochs,
        out_dir=args.output_dir,
        patience=args.patience,
    )

    model_type = "LocTexMinu" if args.minutiae_dir else "LocTex"
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Model type: DeepPrint_{model_type}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Best model: {args.output_dir}/best_model.pyt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
