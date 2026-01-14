# Large-Scale Training Guide (500K - 1.5M Fingerprints)

This guide provides specific recommendations and optimizations for training DeepPrint models on very large datasets (500,000 to 1.5 million fingerprints) in air-gapped environments.

## Table of Contents

1. [Overview and Challenges](#overview-and-challenges)
2. [Hardware Requirements](#hardware-requirements)
3. [Dataset Organization](#dataset-organization)
4. [Memory-Efficient Data Loading](#memory-efficient-data-loading)
5. [Training Configuration](#training-configuration)
6. [Multi-GPU Distributed Training](#multi-gpu-distributed-training)
7. [Storage Management](#storage-management)
8. [Training Time Estimates](#training-time-estimates)
9. [Performance Optimization](#performance-optimization)
10. [Validation Strategy](#validation-strategy)

---

## Overview and Challenges

Training on 500K-1.5M fingerprints presents unique challenges compared to small datasets:

**Scale considerations:**
- **500,000 fingerprints** = ~50,000 subjects × 10 impressions
- **1,500,000 fingerprints** = ~150,000 subjects × 10 impressions

**Key challenges:**
1. Cannot load entire dataset into RAM
2. GPU must be continuously fed data (I/O becomes bottleneck)
3. Validation on full dataset is too slow
4. Storage requirements for checkpoints and embeddings
5. Training time can be days/weeks without optimization

**Solutions:**
- Efficient on-the-fly data loading (PyTorch DataLoader handles this)
- Multi-worker data loading pipeline
- Distributed training across multiple GPUs
- Validation on representative subset
- Checkpoint rotation and compression

---

## Hardware Requirements

### Minimum Configuration (Not Recommended)

**For 500K fingerprints:**
- 1x NVIDIA GPU with 24GB VRAM (RTX 3090, A5000, or better)
- 32 CPU cores (for data loading)
- 128GB RAM
- 2TB SSD storage
- Expected training time: **5-7 days**

### Recommended Configuration

**For 500K-1M fingerprints:**
- 4x NVIDIA A100 (40GB or 80GB)
- 64 CPU cores (16 per GPU)
- 256GB RAM
- 5TB NVMe SSD storage
- 10 Gbps network (for multi-node)
- Expected training time: **1-2 days**

### Optimal Configuration

**For 1M-1.5M fingerprints:**
- 8x NVIDIA A100 (80GB)
- 128 CPU cores (16 per GPU)
- 512GB RAM
- 10TB NVMe SSD storage (RAID 0 for speed)
- 25+ Gbps network
- Expected training time: **12-24 hours**

---

## Dataset Organization

### Directory Structure

Organize your large dataset efficiently:

```
/data/fingerprints/large-dataset/
├── batch_00000/
│   ├── subject_000000/
│   │   ├── impression_00.png
│   │   ├── impression_01.png
│   │   └── ...
│   ├── subject_000001/
│   └── ...
├── batch_00001/
│   ├── subject_001000/
│   └── ...
├── batch_00002/
└── ...
```

**Benefits of batching:**
- Easier to manage large number of files
- Better filesystem performance (avoid single directories with >10K files)
- Can process/verify in chunks
- Parallel preprocessing possible

### File System Recommendations

**For air-gapped systems:**
- Use XFS or ext4 filesystem (not NTFS)
- Enable large inode tables: `mkfs.ext4 -N 2000000` for 1.5M files
- Mount with options: `noatime,nodiratime` for faster reads
- Use NVMe SSDs for data directory (not spinning disks)

```bash
# Check inode usage
df -i /data

# If running low on inodes, you may need to reformat with more inodes
```

---

## Memory-Efficient Data Loading

### Understanding PyTorch DataLoader

The good news: **PyTorch's DataLoader already handles large datasets efficiently!**

The FLX codebase uses PyTorch DataLoader, which:
- Loads images on-demand (not all at once)
- Uses multiple worker processes for parallel loading
- Prefetches batches while GPU is training
- Automatically handles batching and shuffling

### Optimizing DataLoader Workers

Edit `flx/models/torch_helpers.py` to increase data loading workers:

```python
def get_dataloader_args(train: bool = True) -> dict:
    """
    Returns arguments for torch.utils.data.DataLoader.

    Adjust these settings for large-scale training.
    """
    if train:
        return {
            "batch_size": 64,  # Adjust based on GPU memory
            "shuffle": True,
            "num_workers": 16,  # CRITICAL: Increase for large datasets
            "pin_memory": True,  # Faster GPU transfer
            "persistent_workers": True,  # Keep workers alive between epochs
            "prefetch_factor": 4,  # Prefetch 4 batches per worker
        }
    else:
        return {
            "batch_size": 128,  # Can be larger for inference
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
```

**Worker configuration guidelines:**
- **num_workers**: 2-4 workers per GPU (e.g., 16 workers for 4 GPUs)
- **Too few workers**: GPU starves waiting for data
- **Too many workers**: Memory overhead, diminishing returns
- **Rule of thumb**: `num_workers = min(CPU_cores / num_GPUs, 16)`

### Verify Data Loading is Not a Bottleneck

Add profiling to your training script:

```python
import time

# In your training loop
for epoch in range(num_epochs):
    data_time = 0
    train_time = 0

    for i, batch in enumerate(train_dataloader):
        t0 = time.time()

        # Data transfer to GPU
        images, labels = batch[0].to(device), batch[2].to(device)
        t1 = time.time()

        # Forward + backward pass
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        t2 = time.time()

        data_time += (t1 - t0)
        train_time += (t2 - t1)

        if i % 100 == 0:
            print(f"Batch {i}: Data loading: {data_time/(i+1):.3f}s, "
                  f"Training: {train_time/(i+1):.3f}s")

    # Data loading should be < 20% of total time
    # If data_time > train_time, increase num_workers
```

---

## Training Configuration

### Batch Size Optimization

**For large datasets, batch size affects:**
- GPU memory usage
- Training speed
- Model convergence
- Gradient noise

**Recommended batch sizes:**

| GPU Memory | Embedding Size | Batch Size | Gradient Accumulation |
|------------|----------------|------------|----------------------|
| 16GB | 512 | 32 | 2 steps (effective: 64) |
| 24GB | 512 | 64 | 1 step |
| 40GB | 512 | 128 | 1 step |
| 80GB | 512 | 256 | 1 step |

**Using gradient accumulation for larger effective batch sizes:**

Edit `flx/models/model_training.py`:

```python
def _train(
    model: torch.nn.Module,
    loss_fun: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_set: Dataset,
    sampler: torch.utils.data.Sampler = None,
    accumulation_steps: int = 1,  # NEW PARAMETER
) -> float:
    """
    Trains the model for one epoch with gradient accumulation.
    """
    # ... existing code ...

    for batch_idx, vals in enumerate(tqdm.tqdm(train_dataloader)):
        fp_imgs, minu_map_tpl, fp_labels = vals
        # ... move to GPU ...

        # Forward pass
        output: DeepPrintTrainingOutput = model(fp_imgs)
        loss = loss_fun.forward(...)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        # Only update weights every N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # ... rest of code ...
```

### Learning Rate Scaling

For large datasets, adjust learning rate:

Edit `flx/setup/config.py`:

```python
# Original learning rate for small datasets
# LEARNING_RATE = 0.025

# For large datasets (>100K subjects), use lower learning rate
LEARNING_RATE = 0.01  # More stable for large-scale training

# Or use learning rate warmup (recommended)
```

**Add learning rate warmup** (edit optimizer in `model_training.py`):

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

# After creating optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Warmup for first 5 epochs, then cosine decay
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=5
)
main_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs - 5,
    eta_min=LEARNING_RATE * 0.01
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[5]
)

# In training loop, after each epoch:
scheduler.step()
```

### Class Balancing

With 50K-150K subjects (classes), ensure balanced sampling:

```python
# Option 1: Use WeightedRandomSampler for class balance
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):
    """Create sampler that balances classes."""
    # Count samples per class
    class_counts = {}
    for identifier in dataset.ids:
        subject = identifier.subject
        class_counts[subject] = class_counts.get(subject, 0) + 1

    # Compute sample weights (inverse frequency)
    weights = []
    for identifier in dataset.ids:
        subject = identifier.subject
        weights.append(1.0 / class_counts[subject])

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

# In training code
balanced_sampler = create_balanced_sampler(training_set)
train_dataloader = DataLoader(
    training_set,
    batch_size=64,
    sampler=balanced_sampler,  # Instead of shuffle=True
    num_workers=16,
)
```

---

## Multi-GPU Distributed Training

For large-scale datasets, **distributed training is essential**.

### Single-Node Multi-GPU

**Training script modifications:** None needed! The existing code supports it.

**Launch command:**

```bash
# 4 GPUs on single machine
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_my_model.py

# 8 GPUs on single machine
torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_my_model.py
```

### Multi-Node Multi-GPU

For 1M+ fingerprints, use multiple machines:

**Node 0 (master):**
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train_my_model.py
```

**Node 1 (worker):**
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train_my_model.py
```

**Network requirements:**
- All nodes must be able to reach master node
- Open port 29500 (or your chosen port)
- High bandwidth recommended (10+ Gbps)
- Low latency (<1ms for best performance)

### Effective Batch Size in Distributed Training

```
Effective Batch Size = batch_size × num_GPUs × accumulation_steps

Examples:
- 4 GPUs, batch_size=64, accumulation=1 → Effective batch size = 256
- 8 GPUs, batch_size=128, accumulation=2 → Effective batch size = 2048
```

Large effective batch sizes may require learning rate adjustment:
```python
# Linear scaling rule
base_lr = 0.01
effective_batch_size = 256
base_batch_size = 64
adjusted_lr = base_lr * (effective_batch_size / base_batch_size)
# adjusted_lr = 0.04
```

---

## Storage Management

### Disk Space Requirements

**Estimate for your dataset:**

| Component | 500K Fingerprints | 1M Fingerprints | 1.5M Fingerprints |
|-----------|-------------------|-----------------|-------------------|
| Raw images (PNG) | ~200 GB | ~400 GB | ~600 GB |
| Minutiae (.mnt) | ~5 GB | ~10 GB | ~15 GB |
| Training checkpoints | ~2 GB | ~5 GB | ~10 GB |
| Embeddings (512-dim) | ~1 GB | ~2 GB | ~3 GB |
| Logs and metadata | ~1 GB | ~2 GB | ~3 GB |
| **Total** | **~210 GB** | **~420 GB** | **~630 GB** |

Add 50% margin for temporary files and multiple experiments: **~800 GB - 1 TB needed**

### Checkpoint Management

To save disk space, keep only best and latest checkpoints:

Edit `flx/models/model_training.py`:

```python
# Add checkpoint rotation
MAX_CHECKPOINTS_TO_KEEP = 3

def save_with_rotation(model_path, model, loss, optimizer, epoch):
    """Save checkpoint and remove old ones."""
    # Save current checkpoint
    checkpoint_path = model_path.replace(".pyt", f"_epoch{epoch}.pyt")
    save_model_parameters(checkpoint_path, model, loss, optimizer)

    # Keep only latest N checkpoints
    import glob
    checkpoints = sorted(glob.glob(model_path.replace(".pyt", "_epoch*.pyt")))
    if len(checkpoints) > MAX_CHECKPOINTS_TO_KEEP:
        for old_checkpoint in checkpoints[:-MAX_CHECKPOINTS_TO_KEEP]:
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")
```

### Embedding Storage Optimization

For 1M+ fingerprints, embeddings can be large. Use compression:

```python
import numpy as np

# Save embeddings with compression
embeddings_array = np.array([emb for emb in embeddings])
np.savez_compressed(
    "embeddings_compressed.npz",
    embeddings=embeddings_array,
    ids=[str(id) for id in embeddings.ids]
)

# Compression ratio: typically 2-4x smaller
```

---

## Training Time Estimates

### Samples per Second (Training)

**Baseline (1x A100, batch_size=64):**
- ~800-1000 samples/second
- Per epoch (500K samples): ~8-10 minutes
- Per epoch (1M samples): ~16-20 minutes
- Per epoch (1.5M samples): ~25-30 minutes

**4x A100 (batch_size=64 per GPU):**
- ~3000-3500 samples/second
- Per epoch (500K samples): ~2-3 minutes
- Per epoch (1M samples): ~5-6 minutes
- Per epoch (1.5M samples): ~7-9 minutes

**8x A100 (batch_size=64 per GPU):**
- ~5500-6500 samples/second
- Per epoch (500K samples): ~1.5-2 minutes
- Per epoch (1M samples): ~2.5-3 minutes
- Per epoch (1.5M samples): ~4-5 minutes

### Total Training Time

**Epochs needed:** 50-100 epochs typical for convergence

| Dataset Size | 1x A100 | 4x A100 | 8x A100 |
|--------------|---------|---------|---------|
| 500K samples | 7-17 hours | 2-5 hours | 1.5-3 hours |
| 1M samples | 13-33 hours | 4-10 hours | 2.5-5 hours |
| 1.5M samples | 20-50 hours | 6-15 hours | 3.5-7.5 hours |

**Add ~20% for validation time**

---

## Performance Optimization

### CPU-Side Optimizations

**1. Image decoding optimization:**

Use `cv2.setNumThreads()` to limit per-image decode threads:

```python
# In your ImageLoader._load_image():
import cv2
cv2.setNumThreads(1)  # Prevent thread explosion with many workers
img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
```

**2. Preprocessing caching:**

For static preprocessing (non-augmented), cache results:

```python
from functools import lru_cache

class CachedImageLoader(ImageLoader):
    def __init__(self, root_dir: str, cache_size: int = 10000):
        super().__init__(root_dir)
        self._get_cached = lru_cache(maxsize=cache_size)(self._load_uncached)

    def _load_uncached(self, filepath: str) -> torch.Tensor:
        return self._load_image(filepath)

    def get(self, identifier: Identifier) -> torch.Tensor:
        filepath = self._files.get(identifier)
        return self._get_cached(filepath)
```

**Warning:** Only cache for validation set (no augmentation). Training set should not be cached.

**3. Mixed precision training:**

Enable automatic mixed precision (AMP) for 2-3x speedup:

Edit `flx/models/model_training.py`:

```python
from torch.cuda.amp import autocast, GradScaler

def _train(model, loss_fun, optimizer, train_set, sampler=None):
    scaler = GradScaler()  # For mixed precision

    for vals in tqdm.tqdm(train_dataloader):
        fp_imgs, minu_map_tpl, fp_labels = vals
        fp_imgs = fp_imgs.to(device=get_device())
        fp_labels = fp_labels.to(device=get_device())

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            output = model(fp_imgs)
            loss = loss_fun.forward(output, fp_labels, ...)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ... rest of code ...
```

**Speedup:** 2-3x faster training, 30-40% less GPU memory

### GPU-Side Optimizations

**1. Gradient checkpointing (for memory):**

Trade computation for memory:

```python
# In model definition
from torch.utils.checkpoint import checkpoint

class DeepPrint_Tex(nn.Module):
    def forward(self, x):
        # Use checkpointing for large layers
        x = checkpoint(self.stem, x, use_reentrant=False)
        x = checkpoint(self.texture_branch, x, use_reentrant=False)
        return x
```

**2. Channels-last memory format:**

```python
# In model initialization
model = model.to(memory_format=torch.channels_last)

# In data loading
images = images.to(memory_format=torch.channels_last)
```

**Speedup:** 5-10% faster on modern GPUs

**3. Compile model (PyTorch 2.0+):**

```python
# After model creation
import torch
model = torch.compile(model, mode='max-autotune')
```

**Speedup:** 20-40% faster (one-time compilation overhead)

---

## Validation Strategy

### Problem: Full Validation is Too Slow

Validating on 200K+ validation samples every epoch is impractical.

### Solution: Stratified Validation Subset

Use a representative subset for per-epoch validation:

```python
def create_validation_subset(full_val_ids: IdentifierSet,
                            subset_size: int = 10000) -> IdentifierSet:
    """
    Create a stratified validation subset.

    Ensures all subjects are represented proportionally.
    """
    import random

    # Group by subject
    subject_impressions = {}
    for identifier in full_val_ids:
        subject = identifier.subject
        if subject not in subject_impressions:
            subject_impressions[subject] = []
        subject_impressions[subject].append(identifier.impression)

    num_subjects = len(subject_impressions)
    impressions_per_subject = subset_size // num_subjects

    # Sample from each subject
    subset = []
    for subject, impressions in subject_impressions.items():
        sampled = random.sample(
            impressions,
            min(impressions_per_subject, len(impressions))
        )
        subset.extend([Identifier(subject, imp) for imp in sampled])

    return IdentifierSet(subset)

# Usage
val_subset_ids = create_validation_subset(full_val_ids, subset_size=10000)
val_subset = val_images.filter_by_identifiers(val_subset_ids)

# Use val_subset for per-epoch validation
# Use full validation set for final evaluation only
```

### Validation Schedule

**During training:**
- Every epoch: Validate on 10K subset (~2-3 minutes)
- Every 10 epochs: Validate on full set (~20-30 minutes)
- After training: Full validation + test set

### Creating Validation Benchmark for Large Datasets

```python
def create_large_scale_benchmark(
    identifier_set: IdentifierSet,
    mated_per_subject: int = 10,
    non_mated_samples: int = 10000
) -> VerificationBenchmark:
    """
    Create validation benchmark that doesn't explode in size.

    For 50K subjects with 10 impressions each:
    - All pairs would be: 50K × 45 mated + 50K² non-mated = astronomical
    - This creates: 50K × 10 mated + 10K non-mated = manageable
    """
    import random
    comparisons = []

    subjects = sorted(set(id.subject for id in identifier_set))

    # Sample mated comparisons per subject
    for subject in subjects:
        impressions = [
            id.impression for id in identifier_set
            if id.subject == subject
        ]

        # Sample up to N mated pairs per subject
        num_pairs = min(mated_per_subject, len(impressions) * (len(impressions) - 1) // 2)
        for _ in range(num_pairs):
            imp1, imp2 = random.sample(impressions, 2)
            comparisons.append(BiometricComparison(
                Identifier(subject, imp1),
                Identifier(subject, imp2)
            ))

    # Sample non-mated comparisons
    for _ in range(non_mated_samples):
        s1, s2 = random.sample(subjects, 2)
        impressions1 = [id.impression for id in identifier_set if id.subject == s1]
        impressions2 = [id.impression for id in identifier_set if id.subject == s2]

        if impressions1 and impressions2:
            comparisons.append(BiometricComparison(
                Identifier(s1, random.choice(impressions1)),
                Identifier(s2, random.choice(impressions2))
            ))

    print(f"Created benchmark with {len(comparisons)} comparisons")
    return VerificationBenchmark(comparisons)
```

---

## Complete Large-Scale Training Script

```python
#!/usr/bin/env python3
"""
Large-scale training script for 500K-1.5M fingerprints.

Optimized for:
- Multi-GPU distributed training
- Efficient data loading
- Memory management
- Fast validation
"""

import os
import torch
from flx.extractor.fixed_length_extractor import get_DeepPrint_Tex
from flx.data.dataset import IdentifierSet, Identifier

# ============ CONFIGURATION ============
DATASET_ROOT = "/data/fingerprints/large-dataset"
MODEL_OUTPUT_DIR = "/models/large-scale-v1"

# Dataset size
TOTAL_SUBJECTS = 100000  # 100K subjects
IMPRESSIONS_PER_SUBJECT = 10
TRAIN_SUBJECTS = 80000  # 80K training
VAL_SUBJECTS = 20000    # 20K validation

# Model configuration
EMBEDDING_SIZE = 512
MODEL_TYPE = "Tex"  # Start with Tex (simplest)

# Training parameters
NUM_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 64  # Per GPU
NUM_WORKERS = 16  # Per GPU

# Validation strategy
VAL_SUBSET_SIZE = 10000  # Validate on 10K subset per epoch
FULL_VAL_EVERY_N_EPOCHS = 10

def main():
    # Distributed setup (automatic with torchrun)
    from flx.models.torch_helpers import setup_distributed, is_main_process, get_world_size

    setup_distributed()

    if is_main_process():
        print("="*80)
        print("LARGE-SCALE TRAINING")
        print("="*80)
        print(f"Total subjects: {TOTAL_SUBJECTS:,}")
        print(f"Total samples: {TOTAL_SUBJECTS * IMPRESSIONS_PER_SUBJECT:,}")
        print(f"Training subjects: {TRAIN_SUBJECTS:,}")
        print(f"Validation subjects: {VAL_SUBJECTS:,}")
        print(f"World size (GPUs): {get_world_size()}")
        print(f"Effective batch size: {BATCH_SIZE * get_world_size()}")
        print("="*80)

    # Update DataLoader configuration
    from flx.models import torch_helpers

    def get_optimized_dataloader_args(train: bool = True):
        if train:
            return {
                "batch_size": BATCH_SIZE,
                "shuffle": False,  # Will use DistributedSampler
                "num_workers": NUM_WORKERS,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        else:
            return {
                "batch_size": BATCH_SIZE * 2,
                "shuffle": False,
                "num_workers": NUM_WORKERS // 2,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
            }

    # Monkey-patch the function
    torch_helpers.get_dataloader_args = get_optimized_dataloader_args

    # Load datasets (same as before, but filters to splits)
    from flx.data.my_dataset_loader import MyDatasetLoader
    from flx.data.transformed_image_loader import TransformedImageLoader
    from flx.data.label_index import LabelIndex
    from flx.data.dataset import Dataset, ConstantDataLoader
    from flx.image_processing.binarization import LazilyAllocatedBinarizer
    from flx.image_processing.augmentation import RandomPoseTransform, RandomQualityTransform
    from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

    if is_main_process():
        print("Loading dataset...")

    # Create image loader
    raw_loader = MyDatasetLoader(DATASET_ROOT)

    # Training images (with augmentation)
    train_transformed = TransformedImageLoader(
        images=raw_loader,
        poses=RandomPoseTransform(
            pad=0, angle_min=-15, angle_max=15,
            shift_horizontal_min=-25, shift_horizontal_max=25,
            shift_vertical_min=-25, shift_vertical_max=25,
        ),
        transforms=[
            RandomQualityTransform(contrast_min=1.3, contrast_max=2.0, gain_min=0.95, gain_max=1.05),
            LazilyAllocatedBinarizer(5.0),
            pad_and_resize_to_deepprint_input_size,
        ],
    )

    # Validation images (no augmentation)
    val_transformed = TransformedImageLoader(
        images=raw_loader,
        poses=None,
        transforms=[
            LazilyAllocatedBinarizer(5.0),
            pad_and_resize_to_deepprint_input_size,
        ],
    )

    # Create splits
    train_ids = IdentifierSet([
        Identifier(s, i)
        for s in range(TRAIN_SUBJECTS)
        for i in range(IMPRESSIONS_PER_SUBJECT)
    ])

    val_ids = IdentifierSet([
        Identifier(s, i)
        for s in range(TRAIN_SUBJECTS, TRAIN_SUBJECTS + VAL_SUBJECTS)
        for i in range(IMPRESSIONS_PER_SUBJECT)
    ])

    train_images = Dataset(train_transformed, train_ids)
    train_labels = Dataset(LabelIndex(train_ids), train_ids)
    train_minutiae = Dataset(ConstantDataLoader((torch.tensor([]), 0.0)), train_ids)

    val_images_full = Dataset(val_transformed, val_ids)

    # Create validation subset for fast per-epoch validation
    val_subset_ids = create_validation_subset(val_ids, VAL_SUBSET_SIZE)
    val_images_subset = Dataset(val_transformed, val_subset_ids)

    if is_main_process():
        print(f"Training samples: {len(train_images):,}")
        print(f"Validation samples (full): {len(val_images_full):,}")
        print(f"Validation samples (subset): {len(val_images_subset):,}")

    # Create benchmarks
    if is_main_process():
        print("Creating validation benchmarks...")

    val_benchmark_subset = create_large_scale_benchmark(
        val_subset_ids,
        mated_per_subject=10,
        non_mated_samples=5000
    )

    val_benchmark_full = create_large_scale_benchmark(
        val_ids,
        mated_per_subject=20,
        non_mated_samples=20000
    )

    # Initialize model
    if is_main_process():
        print(f"Initializing {MODEL_TYPE} model...")

    extractor = get_DeepPrint_Tex(
        num_training_subjects=TRAIN_SUBJECTS,
        num_texture_dims=EMBEDDING_SIZE
    )

    # Train with custom validation schedule
    if is_main_process():
        print("Starting training...")

    # Custom training loop with alternating validation
    for epoch in range(1, NUM_EPOCHS + 1):
        # Determine which validation set to use
        if epoch % FULL_VAL_EVERY_N_EPOCHS == 0:
            val_set = val_images_full
            val_bench = val_benchmark_full
            if is_main_process():
                print(f"\n[Epoch {epoch}] Using FULL validation set")
        else:
            val_set = val_images_subset
            val_bench = val_benchmark_subset
            if is_main_process():
                print(f"\n[Epoch {epoch}] Using SUBSET validation set")

        # Train for this epoch
        extractor.fit(
            fingerprints=train_images,
            minutia_maps=train_minutiae,
            labels=train_labels,
            validation_fingerprints=val_set,
            validation_benchmark=val_bench,
            num_epochs=epoch,  # Train only current epoch
            out_dir=MODEL_OUTPUT_DIR,
            patience=PATIENCE,
        )

    if is_main_process():
        print("\n" + "="*80)
        print("Training complete!")
        print(f"Model saved to: {MODEL_OUTPUT_DIR}")
        print("="*80)


if __name__ == "__main__":
    main()
```

---

## Quick Reference

### Launch Commands

```bash
# Single GPU (not recommended for large datasets)
python train_large_scale.py

# 4 GPUs, single node
torchrun --standalone --nproc_per_node=4 train_large_scale.py

# 8 GPUs, single node
torchrun --standalone --nproc_per_node=8 train_large_scale.py

# 16 GPUs, 2 nodes (8 GPUs each)
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=node0 --master_port=29500 train_large_scale.py
# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=node0 --master_port=29500 train_large_scale.py
```

### Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor disk I/O
iostat -x 2

# Check data loading isn't bottleneck
# Look for "Data loading" time in output - should be < 20% of training time
```

### Expected Performance Metrics

**For 1M fingerprints (100K subjects):**

| Configuration | Time/Epoch | Total (80 epochs) |
|---------------|------------|-------------------|
| 1x A100 | 16-20 min | ~24 hours |
| 4x A100 | 5-6 min | ~7 hours |
| 8x A100 | 2.5-3 min | ~3.5 hours |

---

## Troubleshooting Large-Scale Training

### GPU Utilization Low (<70%)

**Symptoms:** GPUs not fully utilized, training slower than expected

**Causes:**
1. Data loading bottleneck (most common)
2. Slow disk I/O
3. CPU bottleneck

**Solutions:**
```bash
# Increase num_workers
num_workers = 16  # or higher

# Check disk I/O
iostat -x 2

# If I/O wait is high, move data to faster storage (NVMe SSD)
# Or increase OS cache:
sudo sysctl -w vm.dirty_ratio=80
sudo sysctl -w vm.dirty_background_ratio=50

# Verify CPU isn't bottleneck
htop  # Should see all cores at ~80%+ usage
```

### Out of Memory Errors

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 32  # instead of 64

# Enable gradient checkpointing
# (trade compute for memory)

# Use mixed precision training
# (30-40% memory savings)

# Reduce num_workers
num_workers = 8  # instead of 16
```

### Training Loss Unstable/NaN

**Solutions:**
```python
# Reduce learning rate
LEARNING_RATE = 0.005  # instead of 0.01

# Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for corrupted images
# Add error handling in _load_image()
```

### Slow Validation

**Solutions:**
- Use validation subset (10K samples)
- Full validation only every 10 epochs
- Cache validation preprocessed images (if enough RAM)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
**Optimized for**: 500K - 1.5M fingerprints in air-gapped environments
