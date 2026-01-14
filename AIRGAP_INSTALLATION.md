# Air-Gapped Installation Guide for Fixed-Length Fingerprint Extractors

This guide provides instructions for installing and running the fingerprint model training system in an air-gapped (offline) environment.

## Overview

Training DeepPrint models in an air-gapped environment requires pre-downloading all dependencies, Python packages, and optional pre-trained models before transferring to the isolated network.

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher (3.12 also supported)
- **Operating System**: Linux (RHEL 9, Ubuntu, or compatible)
- **GPU**: NVIDIA GPU with CUDA 12.8 support (recommended for training)
- **CPU**: Multi-core CPU (8+ cores recommended for data preprocessing)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for dependencies, plus space for datasets and models

---

## Part 1: Preparation (On Internet-Connected Machine)

### Step 1: Download Python Packages

Create a directory for all offline packages:

```bash
mkdir -p airgap-packages
cd airgap-packages
```

#### Download Core Dependencies

```bash
# Core dependencies
pip download \
  matplotlib \
  numpy \
  opencv-python \
  scipy \
  scikit-learn \
  "torch>=2.9.0" \
  "torchvision>=0.20.0" \
  torchmetrics \
  tqdm \
  setuptools \
  wheel \
  --dest ./packages
```

#### Download CUDA-Enabled PyTorch (Recommended for GPU Training)

```bash
# PyTorch with CUDA 12.8 support
pip download \
  "torch>=2.9.0" \
  "torchvision>=0.20.0" \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  --dest ./packages-cuda
```

#### Download Development Dependencies (Optional)

```bash
# Development and testing tools
pip download \
  black \
  jupyter \
  pytest \
  seaborn \
  tikzplotlib \
  --dest ./packages-dev
```

#### Download FingerFlow for Minutiae Extraction (Optional)

If you plan to use the FingerFlow minutiae extractor ([scripts/extract_minutiae.py](scripts/extract_minutiae.py)):

```bash
pip download fingerflow --dest ./packages-fingerflow
```

**Note**: FingerFlow requires pre-trained model weights (CoarseNet.h5, FineNet.h5, ClassifyNet.h5, CoreNet.h5). See Step 2 below.

### Step 2: Download Pre-trained Models (Optional)

#### DeepPrint Pre-trained Model

The authors provide a pre-trained DeepPrint model (512-dimensional embeddings):

1. Download from Google Drive: https://drive.google.com/drive/folders/1vV2skXApZMhqWTlF2j_qgXDxRYan5U1f?usp=drive_link
2. Save the model files to `pretrained-models/deepprint-512/`

#### FingerFlow Minutiae Extraction Models

If using the FingerFlow minutiae extractor, download the required model weights:

1. Visit: https://github.com/tourmii/minutiae_extractor
2. Download the following model files:
   - `CoarseNet.h5`
   - `FineNet.h5`
   - `ClassifyNet.h5`
   - `CoreNet.h5`
3. Save to `pretrained-models/fingerflow/`

### Step 3: Download Project Source Code

```bash
# Clone the repository
git clone https://github.com/DorsaRoh/fixed-length-fingerprint-extractors.git
cd fixed-length-fingerprint-extractors

# Create a tarball with the source code
cd ..
tar -czf flx-source.tar.gz fixed-length-fingerprint-extractors/
```

### Step 4: Package Everything for Transfer

```bash
# Create final package directory
mkdir flx-airgap-bundle
mv airgap-packages flx-airgap-bundle/
mv pretrained-models flx-airgap-bundle/
mv flx-source.tar.gz flx-airgap-bundle/

# Create the transfer archive
tar -czf flx-airgap-bundle.tar.gz flx-airgap-bundle/
```

Transfer `flx-airgap-bundle.tar.gz` to your air-gapped environment via approved media (USB drive, approved file transfer, etc.).

---

## Part 2: Installation (On Air-Gapped Machine)

### Step 1: Extract the Bundle

```bash
tar -xzf flx-airgap-bundle.tar.gz
cd flx-airgap-bundle
```

### Step 2: Extract Source Code

```bash
tar -xzf flx-source.tar.gz
cd fixed-length-fingerprint-extractors
```

### Step 3: Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 4: Install Dependencies

#### Install Core Dependencies

```bash
# Install from offline packages
pip install --no-index --find-links=../airgap-packages/packages \
  matplotlib \
  numpy \
  opencv-python \
  scipy \
  scikit-learn \
  torchmetrics \
  tqdm
```

#### Install PyTorch (Choose GPU or CPU)

For GPU training (CUDA 12.8):
```bash
pip install --no-index --find-links=../airgap-packages/packages-cuda \
  torch \
  torchvision \
  torchaudio
```

For CPU-only:
```bash
pip install --no-index --find-links=../airgap-packages/packages \
  torch \
  torchvision
```

#### Install Development Tools (Optional)

```bash
pip install --no-index --find-links=../airgap-packages/packages-dev \
  black \
  jupyter \
  pytest \
  seaborn \
  tikzplotlib
```

#### Install FingerFlow (Optional)

```bash
pip install --no-index --find-links=../airgap-packages/packages-fingerflow \
  fingerflow
```

### Step 5: Install the FLX Package

```bash
# Install the package in editable mode
pip install -e .
```

### Step 6: Verify Installation

```bash
# Test imports
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import flx; print('FLX package imported successfully')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

---

## Part 3: Training Models

### Prepare Your Dataset

Organize your fingerprint dataset according to the project structure:

```
data/
├── fingerprints/
│   └── your-dataset/
│       ├── subject_001/
│       │   ├── impression_01.png
│       │   ├── impression_02.png
│       │   └── ...
│       └── subject_002/
│           └── ...
└── poses/
    └── your-dataset/
        └── ... (minutiae files)
```

See [notebooks/dataset_tutorial.ipynb](notebooks/dataset_tutorial.ipynb) for detailed instructions on dataset preparation.

### Extract Minutiae (Optional)

If using the FingerFlow minutiae extractor:

```bash
python scripts/extract_minutiae.py \
  --input_dir data/fingerprints/your-dataset \
  --output_dir data/poses/your-dataset \
  --coarse_net ../pretrained-models/fingerflow/CoarseNet.h5 \
  --fine_net ../pretrained-models/fingerflow/FineNet.h5 \
  --classify_net ../pretrained-models/fingerflow/ClassifyNet.h5 \
  --core_net ../pretrained-models/fingerflow/CoreNet.h5
```

### Training Examples

Follow the training tutorial in [notebooks/model_training_tutorial.ipynb](notebooks/model_training_tutorial.ipynb).

Basic training workflow:

1. **Configure your dataset** - Edit `flx/setup/datasets.py` to add your dataset
2. **Configure the model** - Edit `flx/setup/models.py` to define model architecture
3. **Train the model** - Use the training scripts or Jupyter notebooks
4. **Generate embeddings** - See [notebooks/embedding_generation_tutorial.ipynb](notebooks/embedding_generation_tutorial.ipynb)

Example training session:
```python
from flx.data.dataset import Dataset
from flx.models.model_training import train_model
from flx.models.deep_print_arch import DeepPrint_Tex_512
from flx.models.deep_print_loss import DeepPrintLoss

# Load your dataset
train_dataset = Dataset.load("your-dataset-train")
val_dataset = Dataset.load("your-dataset-val")

# Initialize model
model = DeepPrint_Tex_512(num_classes=train_dataset.num_subjects)
loss = DeepPrintLoss(model_config=model.config)

# Train
train_model(
    fingerprints=train_dataset,
    minutia_maps=train_dataset.minutia_maps,
    labels=train_dataset.labels,
    validation_fingerprints=val_dataset,
    validation_benchmark=validation_benchmark,
    model=model,
    loss=loss,
    num_epochs=100,
    out_dir="models/your-model",
    patience=10
)
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
torchrun --nproc_per_node=4 your_training_script.py
```

---

## Important Notes

### CUDA and GPU Drivers

- CUDA drivers must be installed on the air-gapped machine separately
- CUDA 12.8 is recommended for PyTorch 2.9.0+
- Verify GPU availability: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Model Training Performance

- **Expected training time**: 10-15 minutes per epoch on NVIDIA A100
- **CPU cores**: 8+ cores recommended for efficient data preprocessing
- Training can take up to 100 epochs depending on model complexity

### Dataset Requirements

The paper used:
- **Training**: 6,000 synthetic subjects × 10 impressions (SFinGe generator)
- **Validation**: 2,000 subjects × 4 impressions
- **Testing**: MCYT330 database (optical and capacitive sensors)

You'll need to prepare your own datasets in the appropriate format.

### Storage Requirements

Estimate storage needs:
- **Python packages**: ~2-5 GB
- **Pre-trained models**: ~100-500 MB
- **Training datasets**: Varies (plan for 10-100+ GB)
- **Model checkpoints**: ~100-500 MB per checkpoint
- **Embeddings**: Depends on dataset size

---

## Troubleshooting

### ImportError: No module named 'torch'
Ensure you've activated the virtual environment: `source venv/bin/activate`

### CUDA out of memory
Reduce batch size in your training configuration or use gradient accumulation.

### Missing dependencies during installation
Ensure all wheel files were downloaded in Step 1. Re-download any missing packages on the internet-connected machine.

### FingerFlow model files not found
Verify that all four .h5 model files are present and paths are correct.

---

## Additional Resources

- **Training Tutorial**: [notebooks/model_training_tutorial.ipynb](notebooks/model_training_tutorial.ipynb)
- **Dataset Setup**: [notebooks/dataset_tutorial.ipynb](notebooks/dataset_tutorial.ipynb)
- **Embedding Extraction**: [notebooks/embedding_generation_tutorial.ipynb](notebooks/embedding_generation_tutorial.ipynb)
- **DeepPrint Paper**: https://arxiv.org/abs/1909.09901
- **BIOSIG 2023 Paper**: Benchmarking fixed-length Fingerprint Representations

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
    Rohwedder-FixedLengthFingerprintDNN-BIOSIG-2023,
    author = {T. Rohwedder and D. Osorio-Roig and C. Rathgeb and C. Busch},
    booktitle = {Intl. Conf. of the Biometrics Special Interest Group ({BIOSIG})},
    title = {Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types},
    year = {2023}
}
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
