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
7. [Validation and Testing](#validation-and-testing)
8. [Multi-GPU Training](#multi-gpu-training)
9. [Troubleshooting](#troubleshooting)

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
  --classify_net path/to/ClassifyNet.h5 \
  --core_net path/to/CoreNet.h5 \
  --workers 1
```

This creates `.mnt` files with minutiae coordinates and angles.

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
        # roi = (start_height, end_height)  # Crop height range
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

If you extracted minutiae using the FingerFlow script, create a minutiae map loader:

```python
# In the same file: flx/data/my_dataset_loader.py

from flx.data.minutia_map_loader import MinutiaMapLoader

class MyDatasetMinutiaMapLoader(MinutiaMapLoader):
    """
    Loader for minutiae extracted by FingerFlow.

    The .mnt files should be in the same structure as your images,
    just in a different root directory.
    """

    @staticmethod
    def _extension() -> str:
        return ".mnt"

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
    try:
        minutia_loader = MyDatasetMinutiaMapLoader(root_dir.replace("fingerprints", "poses"))
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
    print(f"\nBest validation EER: {best.validation_equal_error_rate:.4f}")
    print(f"Achieved at epoch: {best.epoch}")


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
          f"EER={entry['validation_equal_error_rate']:.4f}")

# Find best epoch
best = min(log["entries"], key=lambda e: e["validation_equal_error_rate"])
print(f"\nBest: Epoch {best['epoch']} with EER={best['validation_equal_error_rate']:.4f}")
```

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
- Batch size is automatically split across GPUs
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
- Check learning rate (default: 0.025 in [flx/setup/config.py](flx/setup/config.py))
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
python scripts/extract_minutiae.py --input_dir data/fingerprints/my-dataset --output_dir data/poses/my-dataset --coarse_net path/to/CoarseNet.h5 --fine_net path/to/FineNet.h5 --classify_net path/to/ClassifyNet.h5 --core_net path/to/CoreNet.h5

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
