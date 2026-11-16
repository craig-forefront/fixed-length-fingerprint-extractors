# Training Pipeline Setup Guide for Air-Gapped Networks

This guide provides step-by-step instructions for setting up a complete training pipeline with your own fingerprint data in an air-gapped environment.

**IMPORTANT:** If you have a large dataset (500K+ fingerprints), also read [LARGE_SCALE_TRAINING_GUIDE.md](LARGE_SCALE_TRAINING_GUIDE.md) for specific optimizations, multi-GPU setup, and performance tuning.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Implementing Your Custom Data Loader](#implementing-your-custom-data-loader)
4. [Configuring Your Dataset](#configuring-your-dataset)
5. [Model Selection and Configuration](#model-selection-and-configuration)
6. [Training the Model](#training-the-model)
7. [Choosing the learning rate](#choosing-the-learning-rate)
8. [Loss weights](#loss-weights)
9. [Validation and Testing](#validation-and-testing)
10. [Multi-GPU Training](#multi-gpu-training)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

1. Completed the air-gapped installation from [AIRGAP_INSTALLATION.md](AIRGAP_INSTALLATION.md)
2. Your fingerprint dataset transferred to the air-gapped environment
3. Python environment activated: `source venv/bin/activate`
4. Basic understanding of your dataset structure and naming conventions

---

## Dataset Preparation

### Step 1: Organize Your Dataset

Organize your fingerprint images in a directory structure. The exact structure doesn't matter as long as you can extract subject and impression IDs from the file paths.

**Recommended structure:**

```
data/fingerprints/my-dataset/
├── subject_001/
│   ├── impression_01.png
│   ├── impression_02.png
│   ├── impression_03.png
│   └── ...
├── subject_002/
│   ├── impression_01.png
│   └── ...
└── ...
```

**Alternative flat structure:**

```
data/fingerprints/my-dataset/
├── 001_01.png
├── 001_02.png
├── 002_01.png
└── ...
```

### Step 2: Understand the Identifier System

The FLX package uses a simple identifier system:

- **Subject**: Represents a distinct finger (treated as a unique class during training)
- **Impression**: Represents a single capture/sample of that finger

**Important Notes:**
- Subject and impression IDs are **zero-indexed** (start from 0)
- Each unique finger should have a unique subject ID
- Multiple impressions of the same finger share the same subject ID

**Example:**
```python
from flx.data.dataset import Identifier

# First subject (finger), second impression (sample)
my_id = Identifier(subject=0, impression=1)
```

### Step 3: Prepare Minutiae Data (Optional)

Minutiae data is required for training models with the minutia branch (`TexMinu`, `LocTexMinu`, `Minu`, `LocMinu`).

#### Option A: Extract Minutiae Using FingerFlow

If you have the FingerFlow models (see [AIRGAP_INSTALLATION.md](AIRGAP_INSTALLATION.md)):

```bash
python scripts/extract_minutiae.py \
  --input_dir data/fingerprints/my-dataset \
  --output_dir data/poses/my-dataset \
  --coarse_net path/to/CoarseNet.h5 \
  --fine_net path/to/FineNet.h5 \
  --classify_net path/to/ClassifyNet_6_classes.h5 \
  --core_net path/to/CoreNet.weights \
  --image_dpi 500 \
  --workers 1
```

This creates `.mnt` files with minutiae coordinates and angles. The script needs the `fingerflow` package of a
minutiae_extractor version that has the direction refinement (`image_dpi` option of its `Extractor`, see
[AIRGAP_INSTALLATION.md](AIRGAP_INSTALLATION.md) for where to get it); the `fingerflow` release on PyPI does not have
it. Without it the script stops with an error at the first image. Its exit status is 1 if the extraction failed for
any image; those images are listed in `extraction_errors.txt` in the output directory.

- Feed the images as they are: 8-bit grayscale in their native resolution, not resized or enhanced. The header of
  the `.mnt` file holds the image size, so that the minutiae are scaled together with the images later.
- `--image_dpi` is only used to find the direction of the minutiae. Choose it by the ridge period of the images
  rather than by the nominal resolution: roughly `500 * period_px / 9.5`, i.e. 500 for ordinary 500 ppi prints.
  The 9.5 px depend on how the period is measured. If you have ordinary 500 ppi prints, measure them in the same way
  and use `500 * period_px / period_px_of_the_500_ppi_prints`.
  The 292x392 SFinGe example images of this repository have a ridge period of about 8 px for the typical image, which gives 400 to 450
  (single images range from 6 to 10 px).
  The setting is tolerant: on these images and on 500 ppi prints, values up to 50 below or above the suitable one
  gave the same accuracy.
- Extract all `.mnt` files of a dataset with the same setting, and extract them again when the setting or the
  extractor changes. Files from before the direction refinement have unreliable minutia directions.
- Each row is `x y angle class direction_confidence`. The angle is in radians, the direction vector is
  `(cos(angle), sin(angle))` with y pointing down. This is the convention of the minutia maps, the
  `FingerFlowMinutiaMapLoader` uses the angle as it is. Do not remove minutiae with a low `direction_confidence`
  when building minutia maps: it only rates the direction, the minutia itself is still there.

#### Option B: Use Texture-Only Models

If you don't have minutiae data, use texture-only variants (`Tex`, `LocTex`) which don't require minutiae.

---

## Implementing Your Custom Data Loader

### Step 4: Create Your ImageLoader Class

Create a new Python file for your dataset loader: `flx/data/my_dataset_loader.py`

```python
from flx.data.image_loader import ImageLoader
from flx.data.dataset import Identifier
import torch
import torchvision.transforms.functional as VTF
import cv2
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


class MyDatasetLoader(ImageLoader):
    """
    Custom loader for your fingerprint dataset.

    Adapt the methods below to match your dataset structure and naming convention.
    """

    @staticmethod
    def _extension() -> str:
        """
        Return the file extension of your images.

        Examples: ".png", ".jpg", ".bmp", ".tif"
        """
        return ".png"

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        """
        Parse the subdirectory and filename to extract subject and impression IDs.

        Args:
            subdir: Relative subdirectory path from root (e.g., "subject_001" or "")
            filename: Filename without extension (e.g., "impression_01")

        Returns:
            Identifier with zero-indexed subject and impression

        Examples:

        # Example 1: Flat structure with pattern "subjectID_impressionID.png"
        # File: "042_03.png" -> Subject 42, Impression 3
        subject_id, impression_id = filename.split("_")
        return Identifier(int(subject_id), int(impression_id))

        # Example 2: Nested structure "subject_XXX/impression_YY.png"
        # Directory: "subject_042", File: "impression_03.png"
        subject_id = int(subdir.split("_")[1])  # Extract from "subject_042"
        impression_id = int(filename.split("_")[1])  # Extract from "impression_03"
        return Identifier(subject_id, impression_id)

        # Example 3: Complex pattern "person_012_finger_04_sample_02.png"
        parts = filename.split("_")
        person_id = int(parts[1])
        finger_id = int(parts[3])
        impression_id = int(parts[5])
        # Treat each finger as separate subject (person has 10 fingers)
        subject_id = (person_id * 10) + finger_id
        return Identifier(subject_id, impression_id)
        """

        # TODO: Implement your parsing logic here
        # This example assumes flat structure: "SUBJ_IMP.png"
        subject_id, impression_id = filename.split("_")
        return Identifier(int(subject_id), int(impression_id))

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        """
        Load and preprocess a single image.

        Args:
            filepath: Absolute path to the image file

        Returns:
            torch.Tensor of shape (1, height, width) with values in [0, 1]

        Notes:
            - Images must be grayscale
            - DeepPrint expects 299x299 input size
            - Use pad_and_resize_to_deepprint_input_size for resizing
        """
        # Load as grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Only option 2 keeps the minutia maps of FingerFlowMinutiaMapLoader aligned
        # with the images: it scales the minutiae for the padded, complete image.

        # Option 1: Simple resize to 299x299 (may distort aspect ratio)
        # img = cv2.resize(img, (299, 299))
        # return VTF.to_tensor(img)

        # Option 2: Pad and resize (preserves aspect ratio, recommended)
        # The fill value determines padding color (0.0=black, 1.0=white)
        return pad_and_resize_to_deepprint_input_size(
            VTF.to_tensor(img),
            fill=1.0  # White padding
        )

        # Option 3: Crop specific region then resize (e.g., for images with borders)
        # roi = (height, width)  # Size of the center crop
        # return pad_and_resize_to_deepprint_input_size(
        #     VTF.to_tensor(img),
        #     roi=roi,
        #     fill=1.0
        # )
```

**Testing Your Loader:**

```python
# Test in Python or Jupyter notebook
from flx.data.my_dataset_loader import MyDatasetLoader

loader = MyDatasetLoader("/path/to/data/fingerprints/my-dataset")

# Check what identifiers were found
print(f"Found {loader.ids.num_subjects} subjects")
print(f"Found {len(loader.ids)} total samples")
print(f"First 5 identifiers: {list(loader.ids)[:5]}")

# Test loading an image
test_id = list(loader.ids)[0]
img = loader.get(test_id)
print(f"Image shape: {img.shape}")  # Should be (1, H, W)
print(f"Value range: [{img.min():.3f}, {img.max():.3f}]")  # Should be [0, 1]

# Visualize (optional)
import matplotlib.pyplot as plt
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Subject {test_id.subject}, Impression {test_id.impression}")
plt.show()
```

### Step 5: Create Minutiae Loader (Optional)

If you extracted minutiae using the FingerFlow script, create a minutiae map loader. The `.mnt` files are in the
same structure as your images, just in a different root directory:

```python
# In the same file: flx/data/my_dataset_loader.py

from flx.data.fingerflow_minutia_loader import FingerFlowMinutiaMapLoaderCustomNaming

class MyDatasetMinutiaMapLoader(FingerFlowMinutiaMapLoaderCustomNaming):
    """
    Loader for minutiae extracted by FingerFlow.
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, self._file_to_id_fun)

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        # Should match your image loader's logic exactly
        subject_id, impression_id = filename.split("_")
        return Identifier(int(subject_id), int(impression_id))
```

---

## Configuring Your Dataset

### Step 6: Create Dataset Configuration

Create a function in `flx/setup/datasets.py` or in your own training script:

```python
from flx.data.dataset import Dataset, IdentifierSet, Identifier
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.data.label_index import LabelIndex
from flx.data.my_dataset_loader import MyDatasetLoader, MyDatasetMinutiaMapLoader
from flx.data.minutia_map_loader import PoseTransformedMinutiaMapLoader
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.image_processing.augmentation import RandomPoseTransform, RandomQualityTransform
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


def get_my_dataset_training(root_dir: str) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load training dataset with augmentation.

    Args:
        root_dir: Root directory containing fingerprint images

    Returns:
        tuple of (images, minutia_maps, labels)
    """

    # 1. Load raw images
    raw_loader = MyDatasetLoader(root_dir)

    # 2. Configure augmentation
    # Pose augmentation: random rotation and translation during training
    pose_augmentation = RandomPoseTransform(
        pad=0,
        angle_min=-15,      # Rotate up to ±15 degrees
        angle_max=15,
        shift_horizontal_min=-25,  # Shift up to ±25 pixels
        shift_horizontal_max=25,
        shift_vertical_min=-25,
        shift_vertical_max=25,
    )

    # Quality augmentation: random contrast and gain adjustments
    quality_augmentation = RandomQualityTransform(
        contrast_min=1.3,
        contrast_max=2.0,
        gain_min=0.95,
        gain_max=1.05,
    )

    # Binarization: Gabor filtering for ridge enhancement
    # The ridge_width parameter depends on your sensor resolution
    # Typical values: 3.0-6.0 (lower for higher DPI sensors)
    binarization = LazilyAllocatedBinarizer(ridge_width=5.0)

    # 3. Create transformed image loader
    transformed_images = TransformedImageLoader(
        images=raw_loader,
        poses=pose_augmentation,  # Set to None for no augmentation
        transforms=[
            quality_augmentation,
            binarization,
            pad_and_resize_to_deepprint_input_size,
        ],
    )

    # 4. Create image dataset
    image_dataset = Dataset(transformed_images, raw_loader.ids)

    # 5. Load minutiae (if available)
    # The pose augmentation only transforms the image. PoseTransformedMinutiaMapLoader
    # applies the random pose of each image to its minutiae as well, so that the
    # minutia map still matches the image.
    try:
        minutia_loader = MyDatasetMinutiaMapLoader(root_dir.replace("fingerprints", "poses"))
        minutia_loader = PoseTransformedMinutiaMapLoader(minutia_loader, transformed_images)
        minutia_dataset = Dataset(minutia_loader, raw_loader.ids)
    except Exception as e:
        print(f"Warning: Could not load minutiae: {e}")
        print("Using empty minutiae maps (only texture-based training supported)")
        from flx.data.dataset import ConstantDataLoader
        import torch
        minutia_dataset = Dataset(
            ConstantDataLoader((torch.tensor([]), 0.0)),
            raw_loader.ids
        )

    # 6. Create labels for classification
    label_dataset = Dataset(LabelIndex(raw_loader.ids), raw_loader.ids)

    return image_dataset, minutia_dataset, label_dataset


def get_my_dataset_validation(root_dir: str) -> Dataset:
    """
    Load validation dataset WITHOUT augmentation.

    Args:
        root_dir: Root directory containing fingerprint images

    Returns:
        Dataset of validation images
    """

    raw_loader = MyDatasetLoader(root_dir)

    # No pose augmentation for validation
    transformed_images = TransformedImageLoader(
        images=raw_loader,
        poses=None,  # No augmentation
        transforms=[
            LazilyAllocatedBinarizer(ridge_width=5.0),
            pad_and_resize_to_deepprint_input_size,
        ],
    )

    return Dataset(transformed_images, raw_loader.ids)
```

---

## Model Selection and Configuration

### Step 7: Choose a Model Architecture

The FLX package provides several DeepPrint variants:

| Model Variant | Localization | Texture Branch | Minutia Branch | Minutiae Required |
|---------------|--------------|----------------|----------------|-------------------|
| `Tex` | ❌ | ✅ | ❌ | No |
| `Minu` | ❌ | ❌ | ✅ | Yes |
| `TexMinu` | ❌ | ✅ | ✅ | Yes |
| `LocTex` | ✅ | ✅ | ❌ | No |
| `LocMinu` | ✅ | ❌ | ✅ | Yes |
| `LocTexMinu` | ✅ | ✅ | ✅ | Yes |

**Recommendations:**
- **Start with `Tex`**: Simplest, fastest training, no minutiae needed
- **Use `TexMinu`**: Best performance when minutiae available
- **Use `LocTex` or `LocTexMinu`**: When training data has rotation/translation variations

**Embedding sizes:**
- Common sizes: 128, 256, 512, 1024
- Paper recommendation: **512** provides optimal performance
- Smaller sizes train faster but may have lower accuracy

### Step 8: Split Your Dataset

Split your data into training and validation sets:

```python
from flx.data.dataset import IdentifierSet, Identifier

# Example: 80% training, 20% validation
total_subjects = 1000  # Your total number of subjects
impressions_per_subject = 10  # Your impressions per subject

# Split by subject (recommended)
train_subjects = list(range(800))  # First 800 subjects
val_subjects = list(range(800, 1000))  # Last 200 subjects

# Create identifier sets
train_ids = IdentifierSet([
    Identifier(s, i)
    for s in train_subjects
    for i in range(impressions_per_subject)
])

val_ids = IdentifierSet([
    Identifier(s, i)
    for s in val_subjects
    for i in range(impressions_per_subject)
])

print(f"Training: {train_ids.num_subjects} subjects, {len(train_ids)} samples")
print(f"Validation: {val_ids.num_subjects} subjects, {len(val_ids)} samples")
```

**Important**: Filter your datasets to only include the split identifiers:

```python
# Apply splits to your datasets
train_images = train_images.ids.filter_by_identifiers(train_ids)
val_images = val_images.ids.filter_by_identifiers(val_ids)
```

---

## Training the Model

### Step 9: Create Validation Benchmark

The validation benchmark defines which fingerprint pairs to compare for computing the Equal Error Rate (EER):

```python
from flx.benchmarks.verification import VerificationBenchmark
from flx.benchmarks.biometric_comparison import BiometricComparison

def create_validation_benchmark(identifier_set: IdentifierSet) -> VerificationBenchmark:
    """
    Create a verification benchmark from an identifier set.

    Creates:
    - Mated comparisons: same subject, different impressions
    - Non-mated comparisons: different subjects
    """
    comparisons = []
    subjects = sorted(set(id.subject for id in identifier_set))

    # Mated comparisons: all pairs within same subject
    for subject in subjects:
        subject_impressions = [
            id.impression for id in identifier_set
            if id.subject == subject
        ]
        # Compare each impression with every other impression
        for i, imp1 in enumerate(subject_impressions):
            for imp2 in subject_impressions[i+1:]:
                comparisons.append(BiometricComparison(
                    Identifier(subject, imp1),
                    Identifier(subject, imp2)
                ))

    # Non-mated comparisons: different subjects
    # To keep it manageable, compare each subject with next N subjects
    N_NON_MATED_SUBJECTS = min(20, len(subjects) - 1)

    for i, subject1 in enumerate(subjects):
        for subject2 in subjects[i+1:i+1+N_NON_MATED_SUBJECTS]:
            # Use first impression from each subject
            impressions1 = [id.impression for id in identifier_set if id.subject == subject1]
            impressions2 = [id.impression for id in identifier_set if id.subject == subject2]

            if impressions1 and impressions2:
                comparisons.append(BiometricComparison(
                    Identifier(subject1, impressions1[0]),
                    Identifier(subject2, impressions2[0])
                ))

    print(f"Created benchmark with {len(comparisons)} comparisons")
    return VerificationBenchmark(comparisons)

# Create the benchmark
validation_benchmark = create_validation_benchmark(val_ids)
```

### Step 10: Create Training Script

Create `train_my_model.py`:

```python
#!/usr/bin/env python3
"""
Training script for custom dataset in air-gapped environment.
"""

import os
from flx.extractor.fixed_length_extractor import (
    get_DeepPrint_Tex,
    get_DeepPrint_TexMinu,
    get_DeepPrint_LocTexMinu,
)
from flx.data.dataset import IdentifierSet, Identifier

# Import your dataset functions
from flx.setup.datasets import (  # Or from your own module
    get_my_dataset_training,
    get_my_dataset_validation,
)

# Configuration
DATASET_ROOT = "/path/to/data/fingerprints/my-dataset"
MODEL_OUTPUT_DIR = "/path/to/models/my-model-v1"
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping after 10 epochs without improvement

# Model configuration
MODEL_TYPE = "Tex"  # Options: "Tex", "TexMinu", "LocTexMinu", etc.
EMBEDDING_SIZE = 512

# Dataset split
TOTAL_SUBJECTS = 1000
TRAIN_SUBJECTS = 800
IMPRESSIONS_PER_SUBJECT = 10

def main():
    print("="*80)
    print("Training Configuration")
    print("="*80)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Embedding size: {EMBEDDING_SIZE}")
    print(f"Training subjects: {TRAIN_SUBJECTS}")
    print(f"Validation subjects: {TOTAL_SUBJECTS - TRAIN_SUBJECTS}")
    print(f"Output directory: {MODEL_OUTPUT_DIR}")
    print("="*80)

    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # Load datasets
    print("\nLoading training dataset...")
    train_images, train_minutiae, train_labels = get_my_dataset_training(DATASET_ROOT)

    print(f"Loaded {train_images.ids.num_subjects} training subjects, {len(train_images)} samples")

    # Filter to training split
    train_ids = IdentifierSet([
        Identifier(s, i)
        for s in range(TRAIN_SUBJECTS)
        for i in range(IMPRESSIONS_PER_SUBJECT)
    ])

    train_images = train_images.filter_by_identifiers(train_ids)
    train_minutiae = train_minutiae.filter_by_identifiers(train_ids)
    train_labels = train_labels.filter_by_identifiers(train_ids)

    print(f"Filtered to {len(train_images)} training samples")

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_images = get_my_dataset_validation(DATASET_ROOT)

    val_ids = IdentifierSet([
        Identifier(s, i)
        for s in range(TRAIN_SUBJECTS, TOTAL_SUBJECTS)
        for i in range(IMPRESSIONS_PER_SUBJECT)
    ])

    val_images = val_images.filter_by_identifiers(val_ids)
    print(f"Loaded {len(val_images)} validation samples")

    # Create validation benchmark
    print("\nCreating validation benchmark...")
    from train_my_model import create_validation_benchmark  # From Step 9
    validation_benchmark = create_validation_benchmark(val_ids)

    # Initialize model
    print(f"\nInitializing {MODEL_TYPE} model...")
    if MODEL_TYPE == "Tex":
        extractor = get_DeepPrint_Tex(
            num_training_subjects=TRAIN_SUBJECTS,
            num_texture_dims=EMBEDDING_SIZE
        )
    elif MODEL_TYPE == "TexMinu":
        extractor = get_DeepPrint_TexMinu(
            num_training_subjects=TRAIN_SUBJECTS,
            num_dims=EMBEDDING_SIZE
        )
    elif MODEL_TYPE == "LocTexMinu":
        extractor = get_DeepPrint_LocTexMinu(
            num_training_subjects=TRAIN_SUBJECTS,
            num_dims=EMBEDDING_SIZE
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # Train
    print("\nStarting training...")
    print(f"Training for up to {NUM_EPOCHS} epochs (early stopping patience: {PATIENCE})")

    extractor.fit(
        fingerprints=train_images,
        minutia_maps=train_minutiae,
        labels=train_labels,
        validation_fingerprints=val_images,
        validation_benchmark=validation_benchmark,
        num_epochs=NUM_EPOCHS,
        out_dir=MODEL_OUTPUT_DIR,
        patience=PATIENCE,
    )

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print("="*80)

    # Print best validation result
    from flx.models.model_training import TrainingLog
    log = TrainingLog(os.path.join(MODEL_OUTPUT_DIR, "log.json"))
    best = log.best_entry
    print(f"\nBest epoch: {best.epoch} (lowest {log.selection_metric})")
    print(f"Validation EER of that epoch: {best.validation_equal_error_rate}")


if __name__ == "__main__":
    main()
```

### Step 11: Run Training

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python train_my_model.py
```

**Expected output:**
```
================================================================================
Training Configuration
================================================================================
Dataset: /path/to/data/fingerprints/my-dataset
Model type: Tex
Embedding size: 512
Training subjects: 800
Validation subjects: 200
Output directory: /path/to/models/my-model-v1
================================================================================

Loading training dataset...
Loaded 800 training subjects, 8000 samples

 --- Starting Epoch 1 of 100 ---

Training:
100%|██████████| 125/125 [02:15<00:00,  1.08s/it]
Average Loss: 4.234
Multiclass accuracy: 0.342

Validation:
Equal Error Rate: 0.2453

...
```

**Training time estimates (NVIDIA A100):**
- 10-15 minutes per epoch for 8,000 samples
- 50-100 epochs typical for convergence
- Total: 8-25 hours depending on complexity

---

## Choosing the learning rate

### What the learning rate is

Training repeats one small action millions of times. The model looks at a
handful of images (a *batch*), a number called the *loss* says how wrong its
answers were, and the optimizer changes every parameter of the model a little
in the direction that makes the loss smaller. One such action is a *step*.

The *learning rate* is the size of that change. It decides whether a
training works at all:

- **Too low:** every step changes almost nothing. The loss falls very slowly
  or not at all, and GPU time is wasted.
- **Too high:** every step overshoots. The loss jumps up instead of down, the
  next step is computed from a worse model and overshoots further, and within
  a few steps the loss is in the thousands or is no longer a number ("NaN").
  The training has then *diverged* and the model is useless.
- **About right:** the loss falls steadily, with some noise from batch to
  batch.

This repository uses the Adam optimizer. In its first step Adam changes
every parameter by about the learning rate itself, whatever the data look
like. In a fresh `DeepPrint_LocTexMinu` model, 93 % of the 68 million
parameters are smaller than 0.025 in size, so the former default of 0.025
changed most parameters by more than their own size in the first step. On
real fingerprint images it destroyed the training at once: the loss was 14.9 in the first step and 5,717
in the second (one run, 8 images per step). With 0.01, the value that
LARGE_SCALE_TRAINING_GUIDE.md used to recommend, the loss was 690 in the
third step.

### What warm-up does

At the start the model is random, and the direction that the optimizer
computes from the first few batches is unreliable. *Warm-up* means that the
training starts with a tiny learning rate and raises it in a straight line,
step by step, until it reaches the chosen learning rate. During these steps
the model moves carefully into a region where larger steps are safe.

The warm-up is counted in steps (default: 500). It is never longer than a
tenth of the whole training, so a short training on a small dataset still
reaches the chosen rate; the training prints the number of warm-up steps it
uses. For a large training set, 500 to 2,000 steps are a tiny part of the
first epoch.

After the warm-up there are two choices (the *schedule*):

- `constant`: the rate stays at the chosen value.
- `cosine`: the rate falls smoothly and reaches 1 % of the chosen value in the
  last step of the last epoch. Large steps at the start find a good region
  quickly, small steps at the end settle into it. This is common practice for
  long trainings; whether it gives a lower error rate than `constant` on your
  data was not measured here and is worth one comparison.

### Where the settings are

| Setting | `scripts/train_500k.py` | Default in `flx/setup/config.py` |
|---------|------------------------|----------------------------------|
| Learning rate after the warm-up | `--learning_rate` | `LEARNING_RATE = 0.0001` |
| Number of warm-up steps | `--warmup_steps` | `LEARNING_RATE_WARMUP_STEPS = 500` |
| Schedule after the warm-up | `--lr_schedule constant` or `cosine` | `LEARNING_RATE_SCHEDULE = "constant"` |
| Upper limit for the gradient (see below) | `--gradient_clip_norm` | `GRADIENT_CLIP_NORM = None` (no limit) |
| Stop when the loss is this many times the starting loss | (config only) | `LOSS_EXPLOSION_FACTOR = 10.0` |

In your own training script, pass the settings to `fit`:

```python
from flx.models.learning_rate import LearningRateConfig

extractor.fit(
    ...,
    learning_rate=LearningRateConfig(
        learning_rate=1e-4, warmup_steps=1000, schedule="cosine"
    ),
)
```

*Gradient clipping* is a safety belt. The *gradient* is the list of changes
that the optimizer would like to make in one step; its *norm* is the overall
length of that list. With a limit, a step whose gradient is longer than the
limit is scaled down to the limit, so a single bad batch cannot throw the
model far away. It is off by default, and the stability runs below did not
use it. If you use one, take it from your own range test: the column
`gradient_norm` of `lr_range_test.csv` holds the gradient norm of every step,
and a limit of a few times the typical value in the falling part of the curve
only acts on unusual batches. In the range test here with the current loss
the norm was between 10,000 and 15,200 at rates up to 0.0001 and fell to a
few hundred near 0.001, so a limit such as 1.0 from other projects would
shrink every single step.

The training follows the schedule by counting steps, so it continues
correctly when you resume a training from `model.pyt`, as long as the number
of epochs, the training set and the batch size are the same as before. If you
resume with a larger number of epochs, the `cosine` schedule is stretched to
the new length and the rate jumps back up; prefer to set the number of epochs
once. In a multi-GPU training all processes use the same rate.

### How to run the range test

The right learning rate depends on the data, the model variant and the batch
size. `scripts/find_learning_rate.py` finds a starting value. It trains a
fresh model on a subset of your data for a few hundred steps and multiplies the learning rate by a constant factor after
every step, from far too low (0.0000001) to far too high (1). It reads the
same directory layout as `scripts/train_500k.py` and saves no model.

```bash
python scripts/find_learning_rate.py \
    --dataset_dir /path/to/fingerprints \
    --minutiae_dir /path/to/minutiae \
    --num_subjects 50000 --num_impressions 10 \
    --max_subjects 2000 \
    --batch_size 64 \
    --output_dir /path/to/lr_range_test
```

- `--max_subjects 2000` uses a random 2,000 of the subjects. The test needs
  `--num_steps` times `--batch_size` images (19,200 with the defaults); a
  subset with at least that many images is enough.
- `--batch_size` must be the batch size of the planned training.
- `--variant` chooses the model (`LocTexMinu`, `TexMinu`, `LocTex`, `Tex`).
  Without it the script tests what `train_500k.py` would train.
- `--gradient_clip_norm` must be the value of the planned training.
- `--seed` changes the subset, the starting model and the order of the
  images. Run the test with two or three seeds; if the suggestions differ by
  more than a factor of 3, take the lowest.

The script writes `lr_range_test.csv` (learning rate and loss of every step),
draws `lr_range_test.png` and prints a suggested rate.

### How to read the plot

![Range test of DeepPrint_TexMinu on 440 real fingerprint images](figures/lr_range_test_example.png)

The picture is the range test of `DeepPrint_TexMinu` on 440 real images with
8 images per step, made with the former loss weights (see "What was measured
here" below). The fall is shallow
because the test is short and the data set is tiny.

The horizontal axis is the learning rate; each tick is ten times the one
before. The grey line is the loss of each batch, the blue line is its moving
average. Read it from left to right:

1. **Flat part on the left.** The rate is too low to change the model in so
   few steps. The loss stays at its starting value.
2. **Falling part.** The model learns. The further right, the faster.
3. **Lowest point and rise on the right.** The rate is too high. The test
   ends when the average loss is four times its lowest value or is no longer
   a number.

A good learning rate lies in the falling part, clearly left of the lowest
point. The lowest point itself is not a good choice: the loss is low there
because of all the steps before it, and the next steps diverge. The script
marks the rate where the loss fell fastest (orange, dashed) and suggests one
tenth of it (green). The factor of ten is a safety margin, because a real
training runs thousands of times longer than the test and an instability can
take longer than the test to show.

If the plot has no flat part on the left, start lower (`--min_lr`). If the
loss never falls, no rate is suggested: look for a problem in the data or the
labels before you look at the learning rate.

**Variants with minutiae look different.** Their loss contains the minutia
map loss (see [Loss weights](#loss-weights)), which falls from the first step
at any learning rate: a model that has not learned anything draws a map that
is far from empty, and tiny changes correct that. The curve then has no flat
part and falls evenly from the left edge to the lowest point, and the place
of the steepest fall says nothing about the learning rate. The script
notices this (the steepest fall is more than 100 times below the rate with
the lowest loss) and then suggests one tenth of the rate with the lowest
loss instead; the printed explanation says which rule was used. Starting
lower does not help in this case.

### What was measured here, and what was not

The default learning rate of 0.0001 comes from runs on the only real data
that was available when this was written: 440 images of 220 fingers (two
impressions each, rolled and slap prints from five devices of NIST Special
Database 302, with minutiae marked by examiners), on a CPU, with 8 images per
step. Each line is one run.

| Run | `DeepPrint_TexMinu` | `DeepPrint_LocTexMinu` |
|-----|--------------------|------------------------|
| Range test, 100 steps from 0.000001 to 0.1: smoothed loss above 15 (start: 14.8) from | 0.0031 | 0.0043 |
| Range test: steepest fall at | 0.00067 | 0.00067 |
| Range test: suggested rate | 0.000067 | 0.000067 |
| Training with the former default 0.025, no warm-up | stopped in step 2 (loss 14.7, then 5,119) | stopped in step 2 (loss 14.9, then 5,717) |
| Training with 0.01, no warm-up | not run | stopped in step 3 (loss 14.9, 43.8, 690) |
| Training with the new defaults (0.0001, warm-up, constant), 110 steps | no divergence; highest batch loss 14.9; epoch loss 14.70, then 14.44 | no divergence; highest batch loss 14.9; epoch loss 14.70, then 14.41 |

In the last line the warm-up took 11 steps, a tenth of the training, not the
default 500.

These runs were made before the weights of the parts of the loss changed (see
[Loss weights](#loss-weights)); with the current weights the loss is larger
(it starts near 145 instead of 14.8 for these images) and its parts are
balanced differently. One run was repeated with the current loss:
`DeepPrint_LocTexMinu` with all defaults, 110 steps of 8 images, warm-up 11
steps. It did not diverge: the highest batch loss was the first one (145.1),
and the epoch loss was 77.8, then 62.1. The recorded minutia map loss fell
from 35.9 to 22.3 and the center loss of each embedding from 15.6 to 14.4,
while the cross entropy of each classifier stayed at 5.40.

The range test of `DeepPrint_TexMinu` was also repeated with the current
loss (one run, same settings as in the table). The smoothed loss started at
133 and fell from the first step (100 at the rate 0.000016, 86 at 0.0001). It
was lowest (64) at 0.00135, rose from 0.0017 and the test ended after 69
steps at 0.0027. The steepest fall was found at 0.0000036, which is the
effect described under "Variants with minutiae look different" above; the
suggestion by the rule for that case is 0.000135. The default 0.0001 is 13
times below the rate with the lowest loss. The range test of
`DeepPrint_LocTexMinu` was not repeated, so its line in the table belongs to
the former loss.

What these runs show: 0.0001 does not diverge in the first 110 steps for
either variant, and it is 30 to 43 times below the rate at which the
loss of the range test started to rise. The default is the suggestion of the range
test rounded up to a round number.

What they do not show:

- **Whether 0.0001 is the best rate, or even a good one, for your data.** In
  110 steps on 440 images the model learned almost nothing about the
  subjects: the loss of each classifier stayed between 5.39 and 5.40 and the share
  of correctly named training subjects stayed near chance. The fall of the
  loss came from the center loss and the map loss. These runs are a check
  that the training does not break, not a measurement of how well it learns.
- **Anything about a batch of 64 images or about several GPUs.** Every run
  used 8 images per step. No run used a GPU, mixed precision, or more than
  one process.
- **Anything about accuracy.** No equal error rate was measured.
- **Whether a long training stays stable.** Some instabilities need
  thousands of steps to show.

The value must be found again on your data and with your batch size: run the
range test, start a training with the suggested rate, and watch the first
thousand steps as described below.

### How the rate changes with the batch size and the number of GPUs

`train_model` uses a batch of 64 images per GPU (`get_dataloader_args` in
`flx/models/torch_helpers.py`). With `torchrun` on several GPUs every GPU
processes its own batch of 64 and the results are averaged, so one step sees
**64 times the number of GPUs** images. That product is the *effective batch
size*, and it is the batch size that matters for the learning rate.

- A larger effective batch gives a more reliable direction per step, so it
  usually tolerates a higher rate. There is no exact rule for Adam. A
  common first guess is to multiply the rate by the square root of the factor
  by which the effective batch grew (4 GPUs instead of 1: twice the rate).
  Multiplying by the factor itself (4 GPUs: four times the rate) is the upper
  end and often too much. Neither was measured here.
- The safe way is to measure. The range test runs on one GPU. If the
  effective batch fits on one GPU, run the test with it
  (`--batch_size 256` for 4 GPUs). If it does not fit, run the test with the
  largest batch that fits and treat the suggestion as a rate that is on the
  low, safe side for the larger batch.
- More GPUs also mean fewer steps per epoch. The warm-up is counted in
  steps, so it covers more images; this is intended. Use a longer warm-up
  (2,000 steps or more) when you raise the rate for a large batch.
- All parts of the loss are means over the images of a batch, so the balance
  between them does not change with the batch size. (The center loss used to
  be a sum over the batch; see `LOSS_WEIGHT_CENTER` in `flx/setup/config.py`.)

### What to watch in the first thousand steps

The progress bar shows the loss and the learning rate of every step:

```
  3%|▎         | 212/7032 [02:41<1:26:35,  1.31it/s, loss=21.3, lr=4.24e-05]
```

- **The starting loss.** A fresh model guesses, so each of its classifiers
  starts with a loss near the natural logarithm of the number of training
  subjects (about 10.8 for 50,000 subjects). Variants with texture and
  minutiae have two classifiers. The center loss adds, per classifier, its
  weight (`LOSS_WEIGHT_CENTER`, default 8.0) times the mean squared distance
  between an embedding and the center of its subject. At the start that
  distance is close to 2, because embeddings and centers both have length 1
  and point in unrelated directions; the measured values were 1.95 and 1.98
  (one batch of 8 real images, 256-dimensional embeddings). With 220 subjects
  this gives 2 x 5.4 + 2 x 8.0 x 2 = 42.8, for every batch size. Variants
  with minutiae add the minutia map loss: its weight (default 0.095) times
  about 1,000 for a model that has not learned anything yet (measured: 1,013
  for the first batch of `DeepPrint_TexMinu`). The measured starting loss was
  138 for `DeepPrint_TexMinu` and 145 for `DeepPrint_LocTexMinu`. The map part
  falls fast in the first steps, to about a quarter within 50 steps here,
  because the model quickly learns to draw a nearly empty map; this fall is
  not yet a sign that the model learns to tell fingers apart. A starting
  loss far away from this sum
  points to a problem in the data.
- **During the warm-up** the `lr` value rises and the loss changes little.
  That is normal.
- **After the warm-up** the loss should start to fall within a few hundred
  steps, slowly and with noise. With very many subjects it falls slowly,
  because each subject is seen rarely.
- **Bad signs:** the loss doubles from one step to the next and keeps
  rising; the loss is `nan` or `inf`; the loss swings widely between
  neighbouring steps. The training stops by itself with a message when the
  loss is not a number or more than ten times its starting value, before the
  broken model is saved. The last completed epoch stays in `model.pyt`.
- **If the training stops this way:** divide the learning rate by 3, double
  the warm-up steps, and consider a limit for the gradient. Then start again.
- **If the loss does not move at all for the first thousand steps after the
  warm-up,** the rate is too low (multiply it by 3), or the labels are wrong.

After the first epoch, look at `log.json`: the training loss should be below
the starting loss, and the validation equal error rate should fall from epoch
to epoch. The log also stores the learning rate at the end of every epoch.

---

## Loss weights

### What the loss is made of

The training minimises one number, the *loss*. It is a sum of parts, and each
part is multiplied by a factor, its *weight*:

| Part | What it measures | How it is counted | Weight in `flx/setup/config.py` | `scripts/train_500k.py` |
|------|------------------|-------------------|--------------------------------|-------------------------|
| Cross entropy, once per classifier | How wrong the model is when it names the subject of a training image | Mean over the images of a batch | `LOSS_WEIGHT_CROSS_ENTROPY = 1.0` | `--loss_weight_cross_entropy` |
| Center loss, once per embedding | The squared distance between the embedding of an image and the running average of the embeddings of its subject (the *center*). It pulls the embeddings of one finger together | Mean over the images of a batch | `LOSS_WEIGHT_CENTER = 8.0` | `--loss_weight_center` |
| Minutia map loss (variants with minutiae only) | The squared difference between the minutia map that the model draws and the map made from the minutiae of the image | Sum over the 98,304 cells of a map (6 x 128 x 128), then mean over the images of the batch that have a map | `LOSS_WEIGHT_MINUTIA_MAP = 0.095` | `--loss_weight_minutia_map` |

Variants with texture and minutiae have two classifiers and two embeddings, so
their loss is

```
loss = 1.0 x (cross entropy texture + cross entropy minutiae)
     + 8.0 x (center loss texture + center loss minutiae)
     + 0.095 x minutia map loss
```

`scripts/find_learning_rate.py` has the same three options. In your own
training script, pass the weights when you create the extractor; a weight that
you leave out takes the value from the config:

```python
from flx.extractor.fixed_length_extractor import get_DeepPrint_LocTexMinu
from flx.models.deep_print_loss import LossWeights

extractor = get_DeepPrint_LocTexMinu(
    num_training_subjects, 256, loss_weights=LossWeights(minutia_map=0.01)
)
```

`loss_statistics` in `log.json` holds every part multiplied by its weight.
Divide by the weight to get the part itself.

### What changed, and how to get the former loss back

Earlier versions of this code weighted the parts differently. Four things
changed:

1. **The center loss is a mean over the batch, not a sum.** As a sum with the
   factor 0.125 its weight relative to the cross entropy grew with the batch
   size. The default 8.0 is 0.125 x 64, so with the batch size of
   `train_model` (64) the loss is the same as before. For a batch of B images
   the former balance is `--loss_weight_center` = 0.125 x B.
2. **The minutia map loss is a sum over the cells of a map with the weight
   0.095**, as in the DeepPrint paper (Engelsma et al., 2019, equation 8). It
   was the mean over the cells with the weight 0.3, which is the same as the
   sum with the weight 0.3 / 98,304 = 0.00000305. The reason is the
   measurement below.
3. **The minutia map loss is averaged over the images that have a map.** It
   was averaged over all images of the batch, so images without a map lowered
   the weight of the others.
4. **An image without any minutiae has no map.** It used to get an empty map,
   which taught the model that the image has no minutiae. A fingerprint always
   has minutiae, so an empty list means that the extraction failed. Such an
   image now counts in the cross entropy and the center loss only.

A model that was trained before is not changed by this; the loss is only used
during training. A training that is resumed from `model.pyt` continues with
the new loss unless you pass the former weights:
`--loss_weight_center` 0.125 x batch size and
`--loss_weight_minutia_map 0.00000305`. Points 3 and 4 cannot be switched
back. If you resume with the new weights, the `training_loss` of the new
epochs is much larger than that of the old ones, mainly because of the map
part, and the two cannot be compared. This matters for a training without a
validation set, where the epoch with the lowest training loss becomes
`best_model.pyt`: an old epoch would stay the best one.

### Why the weight of the minutia map loss was raised

The minutia map loss exists to make the layers that the minutia branch shares
with the rest of the model look at minutiae. Two groups of layers are shared:

- the *stem*: the first layers of the model, used by the texture branch and
  the minutia branch;
- the *minutia stem*: the layers after it that are used by the minutia
  embedding and by the map.

How strongly a part of the loss asks a group of layers to change is the length
of its *gradient* on these layers. The *share* of the map loss is the length
of its weighted gradient divided by the sum of that length and the length of
the weighted gradient of all other parts together. A share near 0 means that
the map loss has no say in what these layers learn.

Measured on this machine (no GPU) with `DeepPrint_TexMinu`, 256 dimensions, on
400 real images (200 fingers with two impressions each from NIST Special
Database 302, minutiae marked by examiners), 8 images per step, Adam with
learning rate 0.0001, center weight 8.0. Each column is one run of 40 or 50
steps with that map weight and the same random seed; the gradients were
measured every 10 steps (5 or 6 measurements per run), and the table gives the
lowest and the highest value. The column 0.095 also contains two longer runs
of 150 and 281 steps, 19 measurements in all.

| | Former loss (0.00000305) | 0.01 | **0.095** | 1.0 |
|---|---|---|---|---|
| Share of the map loss on the stem | 0.0006 % to 0.002 % | 0.2 % to 0.6 % | 1.0 % to 5.3 % | 17 % to 26 % |
| Share of the map loss on the minutia stem | 0.07 % to 1.2 % | 24 % to 83 % | 75 % to 99 % | 97 % to 100 % |
| Map gradient on the stem, relative to the gradient of the texture cross entropy | 0.00002 to 0.00008 | 0.06 to 0.18 | 0.25 to 1.5 | 4.8 to 10 |

The column "Former loss" is the loss before these changes at 8 images per
step (center loss 0.125 x sum); that run ended after 38 steps when the
machine ran out of memory, so it has 4 measurements.

What the numbers say:

- **With the former weight the map loss did not reach the shared layers.**
  Its share on the stem was about one part in 100,000.
- **Nearly all of the other gradient on the stem comes from the texture
  branch**, and most of that from the center loss: at the first step the
  lengths were 10,986 (texture center loss with weight 8.0), 364 (texture
  cross entropy), 180 (minutia center loss) and 6 (minutia cross entropy).
  The minutia embedding sends very little gradient back, so the map loss is
  what trains the minutia stem.
- The layers that only draw the map get their gradient from the map loss
  alone, and the optimizer (Adam) adapts its step to the size of the gradient.
  They therefore learn with any weight. The weight decides what the *shared*
  layers learn.

**The target and the choice.** The map loss is a helper task; the cross
entropy is the part that teaches the model to tell fingers apart. The target
was therefore: on the stem, the map loss should pull about as strongly as the
cross entropy of the texture branch (a ratio near 1, between one half and
two), and its share of the whole gradient should stay well below one half, so
that it does not decide alone what the stem learns. The weight 0.095 of the
paper meets this (ratio 0.25 to 1.5, in 16 of 19 measurements between 0.4
and 1.25; share 1.0 % to 5.3 %). With 0.01 the map
loss is a fifth of the cross entropy or less; with 1.0 it is 5 to 10 times
the cross entropy. In the longest run the ratio and the share rose slowly
(ratio 0.7 at the start, 1.5 after 250 steps), because the gradient of the
cross entropy shrank faster than that of the map loss; how they develop in a
full training was not measured. The share of 1.0 % to 5.3 % is small because the center
loss with its weight 8.0 dominates the stem; with a lower center weight the
same map weight has a larger share. On the minutia stem the map loss
dominates with 0.095 (75 % to 99 %); this is accepted, because almost no
other gradient arrives there.

### Does the map branch learn?

Two numbers describe the predicted maps. The *map error* is the minutia map
loss without its weight: the summed squared difference per image. A model
that draws an empty map has an error equal to the summed squares of the
target map, about 300 for these images; a model has to get below that to show
that it places minutiae. The *correlation* between predicted and target map
is 0 for unrelated maps and 1 for identical ones. Both were measured on 16
images of 20 fingers that were not trained on, with the same runs as above:

| After step | Former loss | 0.01 | **0.095** | 1.0 |
|---|---|---|---|---|
| 0 (start) | error 905, correlation -0.02 | the same | the same | the same |
| 10 | 440, 0.08 | 382, 0.08 | 361, 0.11 | 349, 0.11 |
| 20 | 356, 0.11 | 279, 0.15 | 252, 0.18 | 244, 0.18 |
| 30 | 318, 0.13 | 230, 0.20 | 205, 0.24 | 200, 0.25 |

An empty map has the error 190 on these 16 images.

Two more runs with the weight 0.095 went on for longer. The longer one had
281 steps, which is more than five passes over the 400 images. The last two
columns are the mean value that the model draws in the cells where the target
map is above 0.5 (minutiae; the target there is 0.60 on average) and in the
cells where the target map is below 0.01 (no minutia nearby):

| Steps | Map error on the training batches | Error of an empty map on the same batches | Correlation | Drawn on minutiae | Drawn elsewhere |
|---|---|---|---|---|---|
| 0 to 9 | 805 | 310 | 0.00 | 0.03 | 0.029 |
| 50 to 59 | 256 | 288 | 0.31 | 0.06 | 0.009 |
| 100 to 109 | 228 | 293 | 0.40 | 0.07 | 0.009 |
| 150 to 159 | 238 | 311 | 0.43 | 0.08 | 0.009 |
| 200 to 209 | 222 | 298 | 0.43 | 0.09 | 0.009 |
| 250 to 259 | 211 | 292 | 0.45 | 0.09 | 0.008 |

On the 16 images that were not trained on, the error fell from 905 to 154
after 250 steps (empty map: 190), the correlation rose to 0.36, and the model
drew 0.093 on minutiae against 0.007 elsewhere (at the start: 0.020 against
0.034). The run of 150 steps ended at an error of 158 and a correlation of
0.34 on these images.

What this shows: the map branch learns where the minutiae are. Its error
falls below that of an empty map, also for fingers it has not seen, it draws
about 13 times more on minutiae than elsewhere, and the error falls faster
when the map loss reaches the shared layers (0.095 against the former loss in
the table above).

What it does not show: maps with minutiae at full height. After 250 steps the
model drew 0.09 where the target is 0.60 on average, about a sixth of the
height. The model first learned to draw a nearly empty map and then raised it
slowly where the minutiae are. Whether the peaks reach the height of the
target in a full training was not measured.

### What was not measured

- **Accuracy.** No equal error rate was measured with any weight. A few
  hundred steps on 200 fingers show which parts of the loss reach which
  layers and that the map error falls; they cannot show which weights give
  the better model. In all runs the cross entropy stayed at its starting
  value (5.3 for 200 subjects).
- **The center weight.** 8.0 keeps what this code did at 64 images per step.
  The DeepPrint paper states 0.00125 for its center loss (whether the paper
  counts it as a sum or a mean over the batch was not checked), and on the
  stem the gradient of the center loss was 30 times that of the cross entropy
  at the start. Whether a lower value is better was not tested.
- **The variants with localization network**, a batch of 64, a GPU and mixed
  precision. The gradient shares were measured for `DeepPrint_TexMinu` only.
  One training of `DeepPrint_LocTexMinu` with all defaults ran for 110 steps
  without divergence (see the table in
  [What was measured here, and what was not](#what-was-measured-here-and-what-was-not)).

To decide on your data, train the same model with two or three settings for a
few epochs and compare the validation equal error rate, for example the map
weights 0.095, 0.01 and 1.0, and the center weights 8.0 and 0.5. Change one
weight at a time, and repeat the learning rate range test for each setting,
because the weights change the size of the loss and of its gradient.

---

## Validation and Testing

### Step 12: Monitor Training Progress

Training logs are saved to `<MODEL_OUTPUT_DIR>/log.json`:

```python
import json

# Load training log
with open("/path/to/models/my-model-v1/log.json", "r") as f:
    log = json.load(f)

# Print epoch results
for entry in log["entries"]:
    print(f"Epoch {entry['epoch']}: "
          f"Loss={entry['training_loss']:.3f}, "
          f"Acc={entry['training_accuracy']:.3f}, "
          f"EER={entry['validation_equal_error_rate']}")
```

The fields of an entry are:

- `training_loss`: the value that the training minimises, averaged over the
  epoch. Lower is better.
- `loss_statistics`: the parts of the loss, each multiplied by its weight
  (see [Loss weights](#loss-weights)) and averaged per training sample; the
  minutia map loss is averaged over the samples that have a minutia map.
- `training_accuracy`: the share of training samples whose subject the model
  named correctly. Higher is better. It says nothing about fingers that are
  not in the training set.
- `learning_rate`: the learning rate of the last step of the epoch (see
  [Choosing the learning rate](#choosing-the-learning-rate)).
- `validation_equal_error_rate` (EER): measured on the validation set. The
  equal error rate is the error rate at the decision threshold where false
  matches and false non-matches are equally frequent. Lower is better. It is
  `null` if the epoch was trained without a validation set.

#### Which epoch becomes `best_model.pyt`

After every epoch the model is saved as `model.pyt`, and copied to
`best_model.pyt` if the epoch is the best one so far:

- **With a validation set**, the best epoch is the one with the lowest
  validation EER. This is the recommended way, because it measures what the
  model is for: comparing fingers it has not seen in training.
- **Without a validation set**, no error rate exists. The best epoch is then
  the one with the lowest `training_loss`. This only protects against keeping
  an epoch after the training diverged; it cannot detect overfitting. Early
  stopping (`patience`) is off in this case.

`scripts/train_500k.py` holds out validation subjects by default: 10 % of the
subjects (`--validation_split 0.1`), chosen at random with a fixed seed
(`--validation_seed 42`), with all their impressions. Subjects are held out,
not single images, because an image of a finger that is also in the training
set would make the error rate look better than it is for new fingers. For a
large training set, `--max_validation_subjects` (for example 5000) limits the
number of held-out subjects, so that validation after every epoch stays short.
`--validation_split 0` switches validation off.

`TrainingLog.selection_metric` names the field that was used, and
`TrainingLog.best_entry` returns the entry of the best epoch:

```python
from flx.models.model_training import TrainingLog

log = TrainingLog("/path/to/models/my-model-v1/log.json")
best = log.best_entry
print(f"Best: epoch {best.epoch}, selected on {log.selection_metric}")
```

Earlier versions stored the training accuracy in `validation_equal_error_rate`
when there was no validation set and kept the epoch with the lowest value,
which was the epoch with the lowest accuracy. A `best_model.pyt` from such a
run should not be used; `model.pyt` of the same run holds the last epoch.
`TrainingLog` reads such old entries as entries without validation.

### Step 13: Extract Embeddings

After training, extract embeddings for your test set:

```python
from flx.extractor.fixed_length_extractor import get_DeepPrint_Tex
from flx.data.dataset import Dataset

# Load the trained model
extractor = get_DeepPrint_Tex(
    num_training_subjects=800,
    num_texture_dims=512
)
extractor.load_best_model("/path/to/models/my-model-v1")

# Load test dataset
test_images = get_my_dataset_validation("/path/to/test-dataset")

# Extract embeddings
print("Extracting embeddings...")
texture_embeddings, minutia_embeddings = extractor.extract(test_images)

# Combine if both exist
from flx.data.embedding_loader import EmbeddingLoader
embeddings = EmbeddingLoader.combine_if_both_exist(
    texture_embeddings, minutia_embeddings
)

# Save embeddings
embeddings.save("/path/to/embeddings/my-test-embeddings.npz")
print(f"Saved {len(embeddings.ids)} embeddings")
```

### Step 14: Run Verification Benchmark

Evaluate your model's biometric performance:

```python
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.benchmarks.verification import VerificationBenchmark
from flx.benchmarks.biometric_comparison import BiometricComparison

# Create test benchmark
test_benchmark = create_validation_benchmark(test_ids)

# Create matcher
matcher = CosineSimilarityMatcher(embeddings)

# Run benchmark
result = test_benchmark.run(matcher)

# Print results
print(f"Equal Error Rate (EER): {result.get_equal_error_rate():.4f}")
print(f"False Match Rate at FMR=0.01: {result.get_fnmr_at_fmr(0.01):.4f}")
print(f"False Match Rate at FMR=0.001: {result.get_fnmr_at_fmr(0.001):.4f}")

# Save results
result.save("/path/to/results/test-verification-results.json")
```

---

## Multi-GPU Training

### Step 15: Distributed Training Setup

For faster training with multiple GPUs, use PyTorch's distributed training:

**Modify your training script:**

```python
# No code changes needed! The training function automatically detects
# distributed mode when launched with torchrun
```

**Launch with torchrun:**

```bash
# Train on 4 GPUs
torchrun --nproc_per_node=4 train_my_model.py

# Train on 2 nodes with 4 GPUs each
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=node1.example.com \
  --master_port=29500 \
  train_my_model.py
```

**Expected speedup:**
- 2 GPUs: ~1.8x faster
- 4 GPUs: ~3.5x faster
- 8 GPUs: ~6.5x faster

**Notes:**
- Every GPU processes its own batch of 64 images, so one training step sees 64 times the number of GPUs images. The learning rate must be chosen for that number, see [Choosing the learning rate](#choosing-the-learning-rate)
- Only rank 0 process saves models and logs
- All processes participate in training

---

## Troubleshooting

### Common Issues

#### 1. FileNotFoundError: Image files not found

**Problem:** ImageLoader cannot find your image files.

**Solutions:**
- Check that `_extension()` returns correct extension (include the dot: ".png")
- Verify `_file_to_id_fun()` doesn't throw exceptions
- Check file paths and permissions
- Test with: `os.path.exists("/path/to/your/image.png")`

#### 2. RuntimeError: CUDA out of memory

**Problem:** GPU memory exhausted during training.

**Solutions:**
- Reduce batch size (edit [flx/models/torch_helpers.py](flx/models/torch_helpers.py) `BATCH_SIZE`)
- Use smaller embedding size (256 instead of 512)
- Use gradient accumulation
- Close other GPU processes

#### 3. ValueError: Identifier mismatch

**Problem:** Images and minutiae have different identifiers.

**Solutions:**
- Ensure `_file_to_id_fun()` is identical in both loaders
- Check that minutiae files match image files exactly
- Verify file naming consistency

#### 4. Training loss not decreasing

**Problem:** Model not learning.

**Solutions:**
- Check the learning rate (default: 0.0001 in [flx/setup/config.py](flx/setup/config.py)); see [Choosing the learning rate](#choosing-the-learning-rate)
- Verify data augmentation isn't too aggressive
- Ensure images are properly normalized [0, 1]
- Check that labels are correct (consecutive 0-indexed integers)

#### 5. High validation EER (>0.3)

**Problem:** Poor biometric performance.

**Solutions:**
- Train for more epochs
- Increase dataset size (minimum ~500 subjects recommended)
- Check data quality (resolution, noise, alignment)
- Try different model variant (TexMinu usually best)
- Adjust binarization ridge_width parameter

#### 6. Import errors for custom loader

**Problem:** Cannot import your custom dataset loader.

**Solutions:**
- Ensure `flx` package is installed: `pip install -e .`
- Check Python path includes project root
- Verify file is in `flx/data/` directory
- No syntax errors in your loader code

---

## Performance Benchmarks

### Expected Training Metrics

The loss values below are from an earlier version of this guide and were not
measured with the current loss, whose total also contains the weighted center
loss and minutia map loss (see [Loss weights](#loss-weights)): a
`DeepPrint_LocTexMinu` training on 220 fingers started near 145 here. Follow
the cross entropy in `loss_statistics` and the accuracy instead of the total.

**Good training progress:**
- Epoch 1: Loss ~4-5, Accuracy ~0.3-0.4
- Epoch 10: Loss ~2-3, Accuracy ~0.6-0.7
- Epoch 50: Loss ~1-2, Accuracy ~0.8-0.9
- Converged: Loss <1.0, Accuracy >0.9

**Good validation EER:**
- High-quality dataset: EER < 0.05 (5%)
- Medium-quality dataset: EER < 0.10 (10%)
- Low-quality dataset: EER < 0.20 (20%)

### Hardware Requirements

**Minimum (CPU-only):**
- Training: Very slow (~2-4 hours/epoch)
- Inference: Acceptable (~50-100 samples/sec)

**Recommended (Single GPU):**
- NVIDIA GPU with 8GB+ VRAM
- Training: ~10-15 min/epoch on A100
- Inference: Fast (~500-1000 samples/sec)

**Optimal (Multi-GPU):**
- 4x NVIDIA A100 GPUs
- Training: ~3-4 min/epoch
- Inference: Very fast (~2000+ samples/sec)

---

## Next Steps

After successful training:

1. **Evaluate on test set**: Use held-out data not seen during training
2. **Compare sensor types**: Test on different fingerprint sensors
3. **Test pose robustness**: Evaluate with rotated/translated images
4. **Optimize inference**: Convert model for production deployment
5. **Generate embeddings**: Create embedding databases for enrollment
6. **Run identification**: Test 1-to-N matching performance

For more details, see:
- [notebooks/embedding_generation_tutorial.ipynb](notebooks/embedding_generation_tutorial.ipynb)
- [Paper results](README.md) for performance comparisons

---

## Quick Reference

### Common Commands

```bash
# Activate environment
source venv/bin/activate

# Run training
python train_my_model.py

# Multi-GPU training
torchrun --nproc_per_node=4 train_my_model.py

# Extract minutiae
python scripts/extract_minutiae.py --input_dir data/fingerprints/my-dataset --output_dir data/poses/my-dataset --coarse_net path/to/CoarseNet.h5 --fine_net path/to/FineNet.h5 --classify_net path/to/ClassifyNet_6_classes.h5 --core_net path/to/CoreNet.weights --image_dpi 500

# Run tests
pytest tests/

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### File Locations

- **Training scripts**: Project root
- **Model configs**: [flx/setup/config.py](flx/setup/config.py)
- **Dataset loaders**: [flx/data/](flx/data/)
- **Model architectures**: [flx/models/deep_print_arch.py](flx/models/deep_print_arch.py)
- **Training logs**: `<MODEL_OUTPUT_DIR>/log.json`
- **Model checkpoints**: `<MODEL_OUTPUT_DIR>/model.pyt` (latest), `<MODEL_OUTPUT_DIR>/model-best.pyt` (best)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
**For questions or issues**: See [AIRGAP_INSTALLATION.md](AIRGAP_INSTALLATION.md) and project [README.md](README.md)
