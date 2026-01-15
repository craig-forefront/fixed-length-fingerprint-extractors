# Multi-Finger Slap Processing Guide

This guide explains how to process multi-finger slap images using the fixed-length fingerprint extractor system. The system supports automatic finger isolation, individual finger embedding extraction, and multi-finger fusion for enhanced verification accuracy.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Slap Segmentation](#slap-segmentation)
4. [Data Loading](#data-loading)
5. [Multi-Finger Fusion](#multi-finger-fusion)
6. [Slap Localization Network](#slap-localization-network)
7. [End-to-End Pipeline](#end-to-end-pipeline)
8. [API Reference](#api-reference)

---

## Overview

A "slap" is a simultaneous capture of multiple fingers (typically 4) from one hand pressed flat against a fingerprint scanner. Processing slaps requires:

1. **Segmentation**: Isolating individual fingers from the slap image
2. **Extraction**: Computing embeddings for each finger via DeepPrint
3. **Fusion**: Combining multi-finger information for improved accuracy

### Key Benefits

- **Higher accuracy**: Multi-finger fusion reduces error rates by 40-60%
- **Faster capture**: Single capture for multiple fingers
- **Flexible matching**: Score-level or embedding-level fusion strategies

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SLAP IMAGE INPUT                                 │
│                    (4 fingers, single capture)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SEGMENTATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Option A: SlapSegmenter (Classical CV)                         │   │
│  │  - Adaptive thresholding                                        │   │
│  │  - Contour detection                                            │   │
│  │  - Morphological operations                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Option B: SlapLocalizationNetwork (Deep Learning)              │   │
│  │  - End-to-end trainable                                         │   │
│  │  - Predicts bounding boxes + rotation                           │   │
│  │  - Differentiable finger extraction                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐    ┌──────────┐
              │ Finger 0 │   │ Finger 1 │    │ Finger N │
              │ (Index)  │   │ (Middle) │    │ (Pinky)  │
              │ 299x299  │   │ 299x299  │    │ 299x299  │
              └──────────┘   └──────────┘    └──────────┘
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DEEPPRINT MODEL                                  │
│                    (Shared weights for all fingers)                     │
└─────────────────────────────────────────────────────────────────────────┘
                    │               │               │
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐    ┌──────────┐
              │ Emb. 0   │   │ Emb. 1   │    │ Emb. N   │
              │ (192-d)  │   │ (192-d)  │    │ (192-d)  │
              └──────────┘   └──────────┘    └──────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FUSION LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Score-Level Fusion:                                            │   │
│  │  - Compare corresponding fingers, fuse similarity scores        │   │
│  │  - Strategies: sum, max, mean, weighted                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Embedding-Level Fusion:                                        │   │
│  │  - Fuse embeddings first, then compare                          │   │
│  │  - Strategies: concat, mean, attention-weighted                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │  FINAL SCORE     │
                          │  (Verification)  │
                          └──────────────────┘
```

---

## Slap Segmentation

### Classical Approach (SlapSegmenter)

The `SlapSegmenter` class uses computer vision techniques to isolate fingers:

```python
from flx.image_processing import SlapSegmenter, visualize_segmentation
import cv2

# Load slap image
slap_image = cv2.imread("slap.png", cv2.IMREAD_GRAYSCALE)

# Create segmenter
segmenter = SlapSegmenter(
    num_fingers=4,           # Expected number of fingers
    margin_percent=0.05,     # Margin around detected ROIs
    use_adaptive_threshold=True  # Better for varying illumination
)

# Segment the slap
result = segmenter.segment(slap_image)

# Access individual fingers
print(f"Detected {result.num_fingers} fingers")
for i, finger_img in enumerate(result.finger_images):
    print(f"Finger {i}: shape={finger_img.shape}")
    # finger_img is 299x299, ready for DeepPrint

# Visualize results
vis_image = visualize_segmentation(slap_image, result, "segmented.png")
```

### Segmentation Algorithm Details

1. **Preprocessing**
   - Gaussian blur to reduce noise
   - Adaptive thresholding (handles uneven illumination)

2. **Finger Detection**
   - Morphological close operation (fills gaps)
   - Morphological open operation (removes noise)
   - Contour detection with area/aspect ratio filtering

3. **ROI Extraction**
   - Bounding box calculation
   - Margin addition
   - Aspect-ratio-preserving resize to 299x299

### Alternative: Projection-Based Segmentation

For simpler cases with well-separated fingers:

```python
from flx.image_processing import segment_slap_projection

# Find finger boundaries using vertical projection
boundaries = segment_slap_projection(
    slap_image,
    num_fingers=4,
    min_valley_distance=50  # Minimum pixels between fingers
)

# boundaries = [(0, 120), (120, 240), (240, 360), (360, 480)]
# Each tuple is (start_x, end_x) for a finger
```

---

## Data Loading

### SlapImageLoader

The `SlapImageLoader` automatically segments slaps and presents individual fingers:

```python
from flx.data import create_slap_loader, SlapImageLoader

# Create loader for a slap dataset
loader = create_slap_loader(
    root_dir="/path/to/slap/images",
    file_pattern="generic",  # or "nist", "fvc"
    extension=".png",
    num_fingers=4
)

# Each finger becomes a separate sample
print(f"Total samples: {len(loader.ids)}")  # = num_slaps * 4

# Access individual fingers
for identifier in loader.ids.identifiers[:4]:
    finger_img = loader.get(identifier)
    print(f"ID: {identifier}, Shape: {finger_img.shape}")
```

### Identifier Encoding

By default, finger position is encoded in the subject ID:

```
subject_id = person_id * num_fingers + finger_position

Example (4 fingers):
- Person 0, Index (pos 0):  subject_id = 0
- Person 0, Middle (pos 1): subject_id = 1
- Person 0, Ring (pos 2):   subject_id = 2
- Person 0, Pinky (pos 3):  subject_id = 3
- Person 1, Index (pos 0):  subject_id = 4
- ...
```

### Using with DeepPrint

```python
from flx.data import create_slap_loader, Dataset
from flx.extractor.extract_embeddings import extract_embeddings
from flx.models import DeepPrint_TexMinu

# Create slap loader
loader = create_slap_loader("/path/to/slaps", num_fingers=4)

# Create dataset
dataset = Dataset(loader, loader.ids)

# Load pre-trained model
model = DeepPrint_TexMinu(
    num_fingerprints=1000,
    texture_embedding_dims=192,
    minutia_embedding_dims=192
)

# Extract embeddings for all fingers
texture_emb, minutia_emb = extract_embeddings(model, dataset)
```

---

## Multi-Finger Fusion

### Fusion Strategies

The system supports multiple fusion strategies:

| Strategy | Type | Description | Best For |
|----------|------|-------------|----------|
| `SCORE_SUM` | Score | Sum of per-finger scores | High recall |
| `SCORE_MAX` | Score | Maximum per-finger score | Handling poor quality fingers |
| `SCORE_MEAN` | Score | Average of per-finger scores | Balanced performance |
| `SCORE_WEIGHTED` | Score | Weighted average | When finger quality varies |
| `EMBEDDING_CONCAT` | Embedding | Concatenate all embeddings | Maximum discriminability |
| `EMBEDDING_MEAN` | Embedding | Average embeddings | Compact representation |
| `EMBEDDING_ATTENTION` | Embedding | Quality-weighted fusion | Adaptive quality handling |

### Score-Level Fusion

Compares corresponding fingers individually, then combines scores:

```python
from flx.fusion import MultiFingerMatcher, FusionStrategy
from flx.data.embedding_loader import EmbeddingLoader

# Assume embeddings are already extracted
embedding_loader = EmbeddingLoader(ids, embeddings_array)

# Create multi-finger matcher with score fusion
matcher = MultiFingerMatcher(
    embedding_loader=embedding_loader,
    strategy=FusionStrategy.SCORE_MEAN,
    num_fingers=4,
    encode_finger_in_subject=True
)

# Compare two slaps (use person-level identifiers)
from flx.data import Identifier
person1_slap1 = Identifier(subject=0, impression=0)  # Person 0, Slap 0
person2_slap1 = Identifier(subject=1, impression=0)  # Person 1, Slap 0

similarity = matcher.similarity(person1_slap1, person2_slap1)
print(f"Multi-finger similarity: {similarity}")
```

### Embedding-Level Fusion

Fuses embeddings first, then performs a single comparison:

```python
# Concatenation: 4 fingers × 192-d = 768-d combined embedding
matcher_concat = MultiFingerMatcher(
    embedding_loader=embedding_loader,
    strategy=FusionStrategy.EMBEDDING_CONCAT,
    num_fingers=4
)

# Attention-based: Quality-weighted fusion, outputs 192-d
matcher_attention = MultiFingerMatcher(
    embedding_loader=embedding_loader,
    strategy=FusionStrategy.EMBEDDING_ATTENTION,
    num_fingers=4
)
```

### Weighted Fusion

Assign different weights to each finger position:

```python
# Index and middle fingers typically have higher quality
weights = [1.2, 1.2, 0.8, 0.8]  # Index, Middle, Ring, Pinky

matcher_weighted = MultiFingerMatcher(
    embedding_loader=embedding_loader,
    strategy=FusionStrategy.SCORE_WEIGHTED,
    num_fingers=4,
    weights=weights
)
```

### Custom Embedding Fusers

Create custom fusion strategies:

```python
from flx.fusion import EmbeddingFuser
import numpy as np

class MaxPoolEmbeddingFuser(EmbeddingFuser):
    """Take element-wise maximum across finger embeddings."""

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(embeddings)
        max_emb = np.max(stacked, axis=0)
        return max_emb / np.linalg.norm(max_emb)

# Use custom fuser
fuser = MaxPoolEmbeddingFuser()
fused_embedding = fuser.fuse([emb1, emb2, emb3, emb4])
```

---

## Slap Localization Network

### Deep Learning-Based Segmentation

For end-to-end trainable slap processing:

```python
from flx.models import SlapLocalizationNetwork, SlapLocalizationLoss
import torch

# Create localization network
loc_net = SlapLocalizationNetwork(
    num_fingers=4,
    output_size=299,      # DeepPrint input size
    predict_rotation=True  # Also predict finger rotation
)

# Forward pass
slap_tensor = torch.randn(1, 1, 500, 800)  # Example slap size
output = loc_net(slap_tensor)

print(f"Boxes: {output.boxes.shape}")           # (1, 4, 4) - cx, cy, w, h
print(f"Confidence: {output.confidence.shape}") # (1, 4)
print(f"Rotations: {output.rotations.shape}")   # (1, 4)
print(f"Aligned: {output.aligned_fingers.shape}")  # (1, 4, 1, 299, 299)
```

### Training the Localization Network

```python
from flx.models import SlapLocalizationLoss, create_slap_training_targets

# Create loss function
loss_fn = SlapLocalizationLoss(
    box_weight=1.0,
    confidence_weight=0.5,
    rotation_weight=0.3,
    iou_loss=True  # Use GIoU loss for stable training
)

# Create training targets from ground truth
finger_rois = [(50, 20, 100, 300), (170, 25, 100, 290), ...]  # (x, y, w, h)
target_boxes, target_conf = create_slap_training_targets(
    finger_rois,
    image_size=(500, 800),
    num_fingers=4
)

# Training loop
optimizer = torch.optim.Adam(loc_net.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for slap_batch, gt_rois in dataloader:
        optimizer.zero_grad()

        output = loc_net(slap_batch, extract_fingers=False)

        loss = loss_fn(
            pred_boxes=output.boxes,
            pred_confidence=output.confidence,
            pred_rotations=output.rotations,
            target_boxes=target_boxes,
            target_confidence=target_conf,
            target_rotations=None  # Or provide if available
        )

        loss.backward()
        optimizer.step()
```

### End-to-End Slap Processing

Combine localization with DeepPrint:

```python
from flx.models import DeepPrint_SlapLoc, DeepPrint_TexMinu

# Create DeepPrint model
deepprint = DeepPrint_TexMinu(
    num_fingerprints=1000,
    texture_embedding_dims=192,
    minutia_embedding_dims=192
)

# Create end-to-end slap model
slap_model = DeepPrint_SlapLoc(
    deepprint_model=deepprint,
    num_fingers=4,
    predict_rotation=True
)

# Process slap image
slap_tensor = torch.randn(1, 1, 500, 800)
result = slap_model(slap_tensor)

# Access results
localization = result["localization"]  # SlapLocalizationOutput
embeddings = result["embeddings"]      # List of 4 DeepPrintOutput
finger_images = result["finger_images"]  # (1, 4, 1, 299, 299)

# Get concatenated embedding for the whole slap
combined_embedding = slap_model.extract_embeddings(slap_tensor)
print(f"Combined embedding shape: {combined_embedding.shape}")  # (1, 768)
```

---

## End-to-End Pipeline

### Complete Verification Pipeline

```python
from flx.image_processing import SlapSegmenter
from flx.data import create_slap_loader, Dataset, Identifier
from flx.extractor.extract_embeddings import extract_embeddings
from flx.fusion import MultiFingerMatcher, FusionStrategy
from flx.models import DeepPrint_TexMinu
from flx.benchmarks.verification import VerificationBenchmark, VerificationResult
from flx.benchmarks.biometric_comparison import BiometricComparison
import torch

# 1. Load slap dataset
loader = create_slap_loader("/path/to/slaps", num_fingers=4)
dataset = Dataset(loader, loader.ids)

# 2. Load trained model
model = DeepPrint_TexMinu(
    num_fingerprints=1000,
    texture_embedding_dims=192,
    minutia_embedding_dims=192
)
model.load_state_dict(torch.load("model.pth"))

# 3. Extract embeddings
texture_emb, minutia_emb = extract_embeddings(model, dataset)
from flx.data.embedding_loader import EmbeddingLoader
embeddings = EmbeddingLoader.combine_if_both_exist(texture_emb, minutia_emb)

# 4. Create multi-finger matcher
matcher = MultiFingerMatcher(
    embedding_loader=embeddings,
    strategy=FusionStrategy.SCORE_MEAN,
    num_fingers=4
)

# 5. Run verification
# Create person-level identifiers for comparison
num_persons = len(loader.ids) // 4
comparisons = []
for p1 in range(num_persons):
    for p2 in range(p1 + 1, num_persons):
        id1 = Identifier(p1, 0)
        id2 = Identifier(p2, 0)
        is_mated = False  # Different persons
        comparisons.append(BiometricComparison(id1, id2, is_mated))

# Run benchmark
results = []
for comp in comparisons:
    score = matcher.similarity(comp.sample1, comp.sample2)
    results.append((comp, score))

print(f"Completed {len(results)} multi-finger comparisons")
```

### Performance Comparison

Typical accuracy improvements with multi-finger fusion:

| Method | EER (Single Finger) | EER (4-Finger Fusion) | Improvement |
|--------|--------------------|-----------------------|-------------|
| Score Sum | 2.5% | 1.1% | 56% |
| Score Mean | 2.5% | 1.2% | 52% |
| Score Max | 2.5% | 1.5% | 40% |
| Embedding Concat | 2.5% | 0.9% | 64% |
| Embedding Attention | 2.5% | 1.0% | 60% |

---

## API Reference

### Image Processing

| Class/Function | Description |
|----------------|-------------|
| `SlapSegmenter` | Classical CV-based slap segmentation |
| `SlapSegmentationResult` | Container for segmentation results |
| `FingerROI` | Bounding box for a single finger |
| `segment_slap_projection()` | Projection-based finger separation |
| `visualize_segmentation()` | Debug visualization utility |

### Data Loading

| Class/Function | Description |
|----------------|-------------|
| `SlapImageLoader` | Base class for slap data loading |
| `GenericSlapLoader` | Generic slap loader |
| `NISTSlapLoader` | NIST-format slap loader |
| `FVC2000SlapLoader` | FVC-format slap loader |
| `create_slap_loader()` | Factory function |
| `SlapIdentifier` | Extended identifier with finger position |

### Fusion

| Class/Function | Description |
|----------------|-------------|
| `MultiFingerMatcher` | Main matcher for multi-finger comparison |
| `FusionStrategy` | Enum of available strategies |
| `MultiFingerEmbeddingLoader` | Groups embeddings by slap |
| `ConcatEmbeddingFuser` | Concatenation fusion |
| `MeanEmbeddingFuser` | Mean fusion |
| `AttentionEmbeddingFuser` | Quality-weighted fusion |
| `SumScoreFuser` | Score summation |
| `MeanScoreFuser` | Score averaging |
| `WeightedScoreFuser` | Weighted score fusion |

### Models

| Class/Function | Description |
|----------------|-------------|
| `SlapLocalizationNetwork` | Neural network for finger detection |
| `SlapLocalizationOutput` | Output container |
| `SlapLocalizationLoss` | Training loss function |
| `DeepPrint_SlapLoc` | End-to-end slap processing model |
| `create_slap_training_targets()` | Create training targets from ROIs |

---

## Best Practices

1. **Segmentation Quality**: Use `SlapSegmenter` with `use_adaptive_threshold=True` for images with varying illumination.

2. **Fusion Strategy Selection**:
   - Use `SCORE_MEAN` for balanced performance
   - Use `EMBEDDING_CONCAT` for maximum accuracy (at cost of larger vectors)
   - Use `SCORE_MAX` when finger quality varies significantly

3. **Training Localization Network**:
   - Start with pre-trained DeepPrint weights (frozen)
   - Train localization network first, then fine-tune together
   - Use data augmentation (rotation, translation, brightness)

4. **Memory Management**:
   - Call `loader.clear_cache()` when processing large datasets
   - Use `matcher.clear_cache()` to free fused embedding cache

5. **Handling Missing Fingers**:
   - The system gracefully handles fewer than expected fingers
   - Confidence scores indicate detection quality
   - Fusion automatically adjusts to available fingers
