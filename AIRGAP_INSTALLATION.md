# Air-Gapped Installation Guide for Fixed-Length Fingerprint Extractors

This guide provides instructions for installing and running the fingerprint model training system in an air-gapped (offline) environment.

## Overview

Training DeepPrint models in an air-gapped environment requires pre-downloading all dependencies, Python packages, and optional pre-trained models before transferring to the isolated network.

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher (3.12 also supported)
- **Operating System**: Linux (RHEL 9, Ubuntu, or compatible)
- **GPU**: NVIDIA GPU with CUDA 12.8 support (recommended for training)
- **CPU, RAM and storage**: no figures that hold in general. They depend on the number of images and subjects;
  see [Storage Requirements](#storage-requirements) below for what was measured and what was not. The numbers
  that were in this list before (8 cores, 16 GB of RAM, 10 GB of storage) were not measured and are far too small
  for a large corpus.

---

## Part 1: Preparation (On Internet-Connected Machine)

### Step 1: Download Python Packages

Create a directory for all offline packages. All commands of Step 1 run inside it; the last command of Step 1
leaves it again.

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
  pillow \
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
git clone https://github.com/craig-forefront/minutiae_extractor.git
cd minutiae_extractor
git checkout claude/serene-albattani-s6ehsj
cd ..
```

This is a fork of https://github.com/tourmii/minutiae_extractor. `scripts/extract_minutiae.py` needs a revision
of it whose `Extractor` has the `image_dpi` option (direction refinement, file
`src/fingerflow/extractor/direction_sense.py`) and accepts `None` for the ClassifyNet and CoreNet paths. That is
the branch `claude/serene-albattani-s6ehsj` from its commit "Make ClassifyNet and CoreNet optional in the
Extractor" on. When this was written the branch was on the server with that commit, and pull request #3 of that
repository was open to bring it into `main`; `main` itself did not have it yet. Once the pull request is merged,
`main` works and the `git checkout` line is not needed. To check a checkout:

```bash
grep -c "classify_net_path is not None" minutiae_extractor/src/fingerflow/extractor/extractor.py   # must print 1
```

If it prints 0, the checkout does not hold that commit: it is most likely `main` from before the merge, or the
`git checkout` line was left out. Check out the branch before you build the bundle; nothing in the air-gapped
environment can replace it.

The clone must stay where these commands put it, in `airgap-packages/minutiae_extractor`: Part 2 sets
`PYTHONPATH` to `airgap-packages/minutiae_extractor/src`.

An older checkout and the `fingerflow` release on PyPI do not work. With them `scripts/extract_minutiae.py` stops
at the first image with exit status 2 and a message that names this branch.

The extractor has its own dependencies (TensorFlow, pandas, scikit-image, ...), listed in its `pyproject.toml`.
Download them as its own guide describes (`AIR_GAPPED_DEPLOYMENT_GUIDE.md` in that repository, "Download All
Python Dependencies") and put them into `./packages-fingerflow`. Add Pillow and tqdm, which
`scripts/extract_minutiae.py` needs in the same environment:

```bash
pip download pillow tqdm --dest ./packages-fingerflow
```

**Note**: FingerFlow requires pre-trained model weights. `scripts/extract_minutiae.py` needs CoarseNet.h5 and FineNet.h5; ClassifyNet_6_classes.h5 is optional for it and CoreNet.weights is not used by it. See Step 2 below.

#### Leave the Package Directory

Steps 2 to 4 run in the directory that holds `airgap-packages`, not inside it:

```bash
cd ..
```

### Step 2: Download Pre-trained Models (Optional)

#### DeepPrint Pre-trained Model

The authors provide a pre-trained DeepPrint model (512-dimensional embeddings):

1. Download from Google Drive: https://drive.google.com/drive/folders/1vV2skXApZMhqWTlF2j_qgXDxRYan5U1f?usp=drive_link
2. Save the model files to `pretrained-models/deepprint-512/` (next to `airgap-packages`, not inside it; create
   `pretrained-models` also if you download no models, because Step 4 moves it)

#### FingerFlow Minutiae Extraction Models

If using the FingerFlow minutiae extractor, download the required model weights:

1. In the minutiae_extractor checkout (`airgap-packages/minutiae_extractor`), run `python scripts/install_model.py` as described in its README. It downloads
   the model files to its `models/` directory:
   - `CoarseNet.h5`
   - `FineNet.h5`
   - `ClassifyNet_6_classes.h5`
   - `CoreNet.weights`
2. Copy them to `pretrained-models/fingerflow/`

### Step 3: Download Project Source Code

```bash
# Clone the repository
git clone https://github.com/craig-forefront/fixed-length-fingerprint-extractors.git
cd fixed-length-fingerprint-extractors

# Create a tarball with the source code
cd ..
tar -czf flx-source.tar.gz fixed-length-fingerprint-extractors/
```

### Step 4: Package Everything for Transfer

```bash
# Create final package directory (in the directory that holds airgap-packages)
mkdir flx-airgap-bundle
mv airgap-packages flx-airgap-bundle/
mkdir -p pretrained-models
mv pretrained-models flx-airgap-bundle/
mv flx-source.tar.gz flx-airgap-bundle/

# Create the transfer archive
tar -czf flx-airgap-bundle.tar.gz flx-airgap-bundle/
```

The bundle must now hold `airgap-packages/packages` (and `packages-cuda`, `packages-dev`, `packages-fingerflow`,
`minutiae_extractor` if you downloaded them), `pretrained-models` and `flx-source.tar.gz`. Check it before the
transfer:

```bash
ls flx-airgap-bundle flx-airgap-bundle/airgap-packages
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
  pillow \
  scipy \
  scikit-learn \
  torchmetrics \
  tqdm \
  setuptools \
  wheel
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

The minutiae extraction runs in an environment that has the dependencies of the minutiae_extractor repository
(install them from `../airgap-packages/packages-fingerflow` as its `AIR_GAPPED_DEPLOYMENT_GUIDE.md` describes,
with `--no-index --find-links`), Pillow and tqdm, and in which its `fingerflow` package can be imported. It does
not need torch, so it can be an environment of its own. The `flx` package does not need to be installed there:
the script finds it in this repository.

```bash
pip install --no-index --find-links=../airgap-packages/packages-fingerflow pillow tqdm

# An absolute path, so that it also works from other directories
export PYTHONPATH="$(realpath ../airgap-packages/minutiae_extractor/src)"
python3 -c "from fingerflow.extractor import Extractor, direction_sense"
```

### Step 5: Install the FLX Package

```bash
# Install the package in editable mode, without looking for anything online
pip install --no-index --find-links=../airgap-packages/packages --no-build-isolation -e .
```

The dependencies were installed in Step 4; this step only makes `import flx` work. If `pip` reports a missing
dependency here, a package of Step 4 is missing from the bundle.

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
│       ├── 1_1.png     (<subject>_<impression>.png)
│       ├── 1_2.png
│       ├── 2_1.png
│       └── ...
└── poses/
    └── your-dataset/
        └── ... (minutiae files, 1_1.mnt, ...)
```

`scripts/train_500k.py` and the `FingerFlowMinutiaMapLoader` read these file names as they are. The files may be
in subdirectories (for example one per sensor); every subject needs the same number of impressions, numbered
without gaps. "Step 1" of [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md) describes the layout and what
to do when your data does not fit it.

### Extract Minutiae (Optional)

If using the FingerFlow minutiae extractor:

```bash
python scripts/extract_minutiae.py \
  --input_dir data/fingerprints/your-dataset \
  --output_dir data/poses/your-dataset \
  --coarse_net ../pretrained-models/fingerflow/CoarseNet.h5 \
  --fine_net ../pretrained-models/fingerflow/FineNet.h5 \
  --ppi_manifest data/fingerprints/your-dataset-resolutions.csv
```

The script resamples every image to 500 pixels per inch and must know the resolution of each file: from `--ppi`,
from the manifest, from the metadata of the file or from `--assume_ppi`. See "Option A: Extract Minutiae Using
FingerFlow" and "Images from different sensors" in [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md) for
these options, the ridge polarity and the extraction of very large datasets.

### Train

Follow [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md). Its training command, for 50,000 fingers with 10
images each:

```bash
python scripts/train_500k.py \
  --dataset_dir data/fingerprints/your-dataset \
  --minutiae_dir data/poses/your-dataset \
  --num_subjects 50000 --num_impressions 10 \
  --ppi_manifest data/fingerprints/your-dataset-resolutions.csv \
  --learning_rate 0.0001 --lr_schedule cosine \
  --num_epochs 10 \
  --output_dir models/your-model
```

For several GPUs, put `torchrun --nproc_per_node=4` in place of `python`. The same command again continues the
training; a new `--output_dir` starts a new one.

The notebooks [model_training_tutorial.ipynb](notebooks/model_training_tutorial.ipynb) and
[embedding_generation_tutorial.ipynb](notebooks/embedding_generation_tutorial.ipynb) show the classes underneath
on the SFinGe example data. They need the development dependencies (`jupyter`; the embedding notebook also
imports `tikzplotlib` through `flx.visualization`), so download and install "Development Dependencies" for them.
The notebooks were not run for this guide.

---

## Important Notes

### CUDA and GPU Drivers

- CUDA drivers must be installed on the air-gapped machine separately
- CUDA 12.8 is recommended for PyTorch 2.9.0+
- Verify GPU availability: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Model Training Performance

No training time was measured on a GPU for these guides. The image preparation runs on the CPU and took about
1 second per rolled print per core here, so plan CPU cores per GPU; see "What limits the speed" in
[LARGE_SCALE_TRAINING_GUIDE.md](LARGE_SCALE_TRAINING_GUIDE.md).

### Dataset Requirements

The paper used:
- **Training**: 6,000 synthetic subjects × 10 impressions (SFinGe generator)
- **Validation**: 2,000 subjects × 4 impressions
- **Testing**: MCYT330 database (optical and capacitive sensors)

You'll need to prepare your own datasets in the appropriate format.

### Storage Requirements

What was measured, and what was not:
- **Python packages and pre-trained models**: not measured; the download commands of this guide were not run.
  Look at the size of `flx-airgap-bundle` after Step 4.
- **Training images**: the size of your corpus. The 500 ppi PNG files of rolled prints that were used for the
  guides were about 200 kB each; at that size 1 million images are about 200 GB and 100 million about 20 TB. See
  the storage table in [LARGE_SCALE_TRAINING_GUIDE.md](LARGE_SCALE_TRAINING_GUIDE.md).
- **Model checkpoints**: `model.pyt` and `best_model.pyt`, 0.8 GB each plus 8.2 kB per training subject (measured).
  While one of them is written, a third file of the same size exists for a moment (`model.pyt.tmp` or
  `best_model.pyt.tmp`).
- **Embeddings and minutiae files**: grow with the number of images; see the same storage table.
- **RAM and GPU memory**: grow with the number of subjects and images; see "Above about one million subjects" in
  [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md) for what was measured and what is calculated.

---

## Troubleshooting

### ImportError: No module named 'torch'
Ensure you've activated the virtual environment: `source venv/bin/activate`

### CUDA out of memory
Lower `--batch_size` of `scripts/train_500k.py` (default 64).

### Missing dependencies during installation
Ensure all wheel files were downloaded in Step 1. Re-download any missing packages on the internet-connected machine.

### FingerFlow model files not found
`scripts/extract_minutiae.py` needs `CoarseNet.h5` and `FineNet.h5`; check the paths given to `--coarse_net` and
`--fine_net`. `ClassifyNet_6_classes.h5` is optional (`--classify_net`), and `CoreNet.weights` is not used.

### The extraction stops with "too old"
The `fingerflow` package that Python finds is not the revision named under "Download FingerFlow for Minutiae
Extraction" above. The message shows where it was imported from.

---

## Additional Resources

- **Training Tutorial**: [notebooks/model_training_tutorial.ipynb](notebooks/model_training_tutorial.ipynb)
- **Dataset Setup**: [notebooks/dataset_tutorial.ipynb](notebooks/dataset_tutorial.ipynb)
- **Embedding Extraction**: [notebooks/embedding_generation_tutorial.ipynb](notebooks/embedding_generation_tutorial.ipynb)
  (needs `tikzplotlib` from the development dependencies)
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
