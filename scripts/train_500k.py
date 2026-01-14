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

With validation and early stopping:
    torchrun --nproc_per_node=4 scripts/train_500k.py \
        --dataset_dir /path/to/fingerprints \
        --output_dir /path/to/model \
        --validation_split 0.1 \
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

def create_training_and_validation_ids(
    num_subjects: int,
    num_impressions: int,
    validation_split: float = 0.0,
) -> tuple[IdentifierSet, IdentifierSet | None]:
    """
    Create identifier sets for training and validation samples.

    Args:
        num_subjects: Total number of subjects
        num_impressions: Impressions per subject
        validation_split: Fraction of subjects to use for validation (0.0 to 1.0)

    Returns:
        Tuple of (training_ids, validation_ids)
    """
    all_subjects = list(range(num_subjects))

    if validation_split > 0:
        random.seed(42)  # Reproducible split
        random.shuffle(all_subjects)
        num_val_subjects = int(num_subjects * validation_split)
        val_subjects: set[int] = set(all_subjects[:num_val_subjects])
        train_subjects = all_subjects[num_val_subjects:]
    else:
        train_subjects = all_subjects
        val_subjects: set[int] = set()

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
) -> VerificationBenchmark:
    """
    Create a verification benchmark for validation.

    Generates mated pairs (same subject, different impressions) and
    non-mated pairs (different subjects) for EER calculation.
    """
    random.seed(42)

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
            selected = random.sample(other_subjects, comparisons_per_subject)
        else:
            selected = other_subjects

        for other in selected:
            imp1 = random.randint(0, num_impressions - 1)
            imp2 = random.randint(0, num_impressions - 1)
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
) -> Dataset:
    """
    Create minutia map dataset.

    Args:
        minutiae_dir: Directory containing .mnt files (from extract_minutiae.py)
                      If None, uses empty tensors (model still works, texture-only)
        training_ids: Identifiers for training samples

    Returns:
        Dataset containing minutia maps
    """
    if minutiae_dir is not None:
        print(f"Loading minutiae from {minutiae_dir}")
        return Dataset(FingerFlowMinutiaMapLoader(minutiae_dir), training_ids)
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
                        help="Fraction of subjects for validation (default: 0.1)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience - epochs without improvement (default: 3, 0=disabled)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate training/validation split
    num_train_subjects = int(args.num_subjects * (1 - args.validation_split))
    num_val_subjects = args.num_subjects - num_train_subjects

    model_name = "LocTexMinu (texture + minutiae)" if args.minutiae_dir else "LocTex (texture-only)"
    print(f"=" * 60)
    print(f"Training DeepPrint with Learned Alignment ({model_name})")
    print(f"=" * 60)
    print(f"Dataset:      {args.dataset_dir}")
    print(f"Minutiae:     {args.minutiae_dir or 'None'}")
    print(f"Subjects:     {args.num_subjects:,} total")
    print(f"  Training:   {num_train_subjects:,} subjects ({100*(1-args.validation_split):.0f}%)")
    print(f"  Validation: {num_val_subjects:,} subjects ({100*args.validation_split:.0f}%)")
    print(f"Impressions:  {args.num_impressions}")
    print(f"Total images: {args.num_subjects * args.num_impressions:,}")
    print(f"Epochs:       {args.num_epochs} (max)")
    print(f"Patience:     {args.patience} epochs (early stopping)")
    print(f"Embedding:    {args.embedding_dim} dims")
    print(f"Output:       {args.output_dir}")
    print(f"Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPUs:         {torch.cuda.device_count()}")
    print(f"=" * 60)

    # Create identifiers for training and validation
    training_ids, validation_ids = create_training_and_validation_ids(
        args.num_subjects, args.num_impressions, args.validation_split
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
    minutia_dataset = create_minutia_dataset(args.minutiae_dir, training_ids)

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
            validation_ids, args.num_impressions
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
