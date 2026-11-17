# Installation

This document sets up what [training-guide.md](training-guide.md) needs. There are two ways: with network
access (Part A), or for a machine without network access, an "air-gapped" machine (Part B).

## The two environments

The route from images to a trained model uses two separate Python environments. An *environment* here is a
directory with its own Python packages (a "virtual environment"), so that the packages of one do not disturb
the other.

| Environment | Used for | Needs | Where this document puts it |
|-------------|----------|-------|------------------------------|
| **Training environment** | `scripts/find_learning_rate.py`, `scripts/train_500k.py`, the embeddings, the tests, the notebooks | PyTorch and the `flx` package of this repository | `venv/` in this repository |
| **Extraction environment** | `scripts/extract_minutiae.py` only | TensorFlow, the minutiae extractor (next section), Pillow, tqdm, and two model files | Part A: `.venv/` in the clone of the extractor. Part B: `venv-extractor/` next to this repository |

The two are kept apart because the minutiae extractor pins its own versions of numpy and other packages. The
extraction environment does not need PyTorch, and `flx` does not need to be installed in it: the script finds
`flx` in this repository.

If you only train the texture-only variants (without minutiae, see Step 3 of the training guide), you need
the training environment only.

Requirements for both: Python 3.11 or higher (`requires-python` in `pyproject.toml`). What was tried of this
document was tried on Linux x86_64 with Python 3.11, on a machine without a GPU; other systems were not tried. For
training on a GPU the machine needs an NVIDIA driver that fits the PyTorch build; this was not tried.

## The minutiae extractor

The *minutiae extractor* is a separate repository,
[craig-forefront/minutiae_extractor](https://github.com/craig-forefront/minutiae_extractor). Its Python package
is called `fingerflow`. Three things carry that name, and only one works here:

| What | Works with `scripts/extract_minutiae.py` |
|------|-------------------------------------------|
| The `main` branch of **craig-forefront/minutiae_extractor**, commit `420b6e5` ("Merge pull request #3") or later | **yes** |
| A clone of that repository from before that commit, or the upstream repository `tourmii/minutiae_extractor` | no |
| The package `fingerflow` on PyPI (`pip install fingerflow`) | no |

The script needs an `Extractor` that has the `image_dpi` option and accepts `None` for two of its four
networks. With the other two it stops before the first image with exit status 2 and a message that says "too
old" and shows where the package was imported from.

To check a clone, run this in the directory that holds it:

```bash
grep -c "classify_net_path is not None" minutiae_extractor/src/fingerflow/extractor/extractor.py   # must print 1
```

If it prints 0, the clone is older than that commit or is the upstream repository: update it (`git pull` on
`main` of the craig-forefront repository).

You never run a script of the extractor repository for the training route; this repository's
`scripts/extract_minutiae.py` calls it. The extractor's own documentation is needed only for what its output
means and for its options.

---

## Part A: With network access

### A1. Training environment

```bash
git clone https://github.com/craig-forefront/fixed-length-fingerprint-extractors.git
cd fixed-length-fingerprint-extractors
python3.11 -m venv venv
source venv/bin/activate
```

On a machine **without a GPU**, install the CPU build of PyTorch first; otherwise the next step downloads the
much larger build with GPU libraries:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the `flx` package with its dependencies (they are listed in `pyproject.toml`):

```bash
pip install -e .
```

Optional, for the tests and the notebooks (pytest, jupyter, black, seaborn, tikzplotlib):

```bash
pip install -e ".[dev]"
```

`pip install -r requirements.txt` does the same as the last two commands together; you do not need it after
them.

Check the installation:

```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -c "import flx; print('flx imported')"
python -m pytest -q          # only if you installed the optional tools
```

"CUDA available: False" means that PyTorch found no usable GPU. The tests take a few minutes on a CPU. Here
they ended with "357 passed, 1 skipped" in this environment; the skipped test needs the package
`scikit-image`, which is not a dependency of this repository.

### A2. Extraction environment

Only needed for the model variants with minutiae. Clone the extractor **next to** this repository, so that
both directories have the same parent directory:

```bash
cd ..
git clone https://github.com/craig-forefront/minutiae_extractor.git
grep -c "classify_net_path is not None" minutiae_extractor/src/fingerflow/extractor/extractor.py   # must print 1
cd minutiae_extractor
uv sync                                  # creates .venv and installs the dependencies
uv pip install -e .                      # makes the package importable as "fingerflow"
uv run python scripts/install_model.py   # downloads the model files into models/
cd ../fixed-length-fingerprint-extractors
```

`uv` is the package tool that the extractor uses; its README says how to get it. Use the clone line above:
the README of the extractor at commit `420b6e5` still names the upstream repository in its own clone command.
Expect a large environment (the extractor's documentation gives 6.4 GB, most of it GPU libraries that are
installed on a machine without a GPU as well) and 1.6 GB of model files. Pillow and tqdm, which
`scripts/extract_minutiae.py` needs, are part of that environment (they are in the extractor's `uv.lock`).

Of the four model files in `minutiae_extractor/models`, the extraction needs `CoarseNet.h5` (81 MB) and
`FineNet.h5` (654 MB). `ClassifyNet_6_classes.h5` (619 MB) is optional (`--classify_net`; DeepPrint does not
use its result), and `CoreNet.weights` (256 MB) is not used.

Check, from the root directory of this repository:

```bash
source ../minutiae_extractor/.venv/bin/activate
python -c "from fingerflow.extractor import Extractor, direction_sense; print('minutiae extractor found')"
ls -l ../minutiae_extractor/models
deactivate
```

Next: [training-guide.md](training-guide.md).

---

## Part B: Air-gapped

You build a bundle on a machine with network access (B1), carry it to the air-gapped machine and install from
it (B2).

**The machine that builds the bundle must have the same operating system, the same processor architecture and
the same Python version as the air-gapped machine** (the commands below use `python3.11` on both). `pip
download` fetches the package files for the machine it runs on; files for another system do not install.

### B1. On the machine with network access: build the bundle

Start in an empty working directory.

**Packages for the training environment.**

```bash
mkdir -p airgap-packages
cd airgap-packages

# Everything except PyTorch
python3.11 -m pip download \
  matplotlib numpy opencv-python pillow scipy scikit-learn torchmetrics tqdm setuptools wheel \
  --dest ./packages
```

PyTorch, in **one** of two builds. For training on an NVIDIA GPU (CUDA 12.8):

```bash
python3.11 -m pip download "torch>=2.9.0" "torchvision>=0.20.0" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --dest ./packages-torch
```

For a machine without a GPU:

```bash
python3.11 -m pip download "torch>=2.9.0" "torchvision>=0.20.0" \
  --index-url https://download.pytorch.org/whl/cpu \
  --dest ./packages-torch
```

(`./packages` also receives a copy of PyTorch, because `torchmetrics` depends on it. Part B2 installs PyTorch
from `./packages-torch` first, so that copy is not used.)

Optional, for the tests and the notebooks:

```bash
python3.11 -m pip download black jupyter pytest seaborn tikzplotlib --dest ./packages-dev
```

**The minutiae extractor and its packages** (only for the model variants with minutiae). Use these commands,
not the clone and download steps of the extractor's own guides:

```bash
git clone https://github.com/craig-forefront/minutiae_extractor.git
grep -c "classify_net_path is not None" minutiae_extractor/src/fingerflow/extractor/extractor.py   # must print 1

# The exact package versions that the extractor was tested with, as a list for pip
(cd minutiae_extractor && uv export --frozen --no-hashes --no-emit-project -o ../requirements-extractor.txt)
python3.11 -m pip download -r requirements-extractor.txt --dest ./packages-extractor
```

The `uv export` line writes the versions recorded in the extractor's `uv.lock`. The list has about 320
lines and contains TensorFlow, Pillow and tqdm. The clone must stay where these commands put it, in
`airgap-packages/minutiae_extractor`: Part B2 points Python to `airgap-packages/minutiae_extractor/src`.

**The model files of the minutiae extractor.** The extractor's script downloads them from Hugging Face into
`minutiae_extractor/models`; it needs the package `huggingface-hub`:

```bash
python3.11 -m pip install huggingface-hub
(cd minutiae_extractor && python3.11 scripts/install_model.py)

cd ..                                  # leave airgap-packages
mkdir -p pretrained-models/minutiae-extractor
mv airgap-packages/minutiae_extractor/models/CoarseNet.h5 \
   airgap-packages/minutiae_extractor/models/FineNet.h5 \
   pretrained-models/minutiae-extractor/
```

`CoarseNet.h5` (81 MB) and `FineNet.h5` (654 MB) are the two files that the extraction needs. Move
`ClassifyNet_6_classes.h5` (619 MB) too only if you want the minutia type in the `.mnt` files; DeepPrint does
not use it. `CoreNet.weights` (256 MB) is not used; you can delete it from the clone before you pack.

If you skipped the extractor, leave `airgap-packages` now with `cd ..`.

**The source code of this repository, and the bundle.** In the directory that holds `airgap-packages`:

```bash
git clone https://github.com/craig-forefront/fixed-length-fingerprint-extractors.git
tar -czf flx-source.tar.gz fixed-length-fingerprint-extractors/

mkdir flx-airgap-bundle
mkdir -p pretrained-models
mv airgap-packages pretrained-models flx-source.tar.gz flx-airgap-bundle/
tar -czf flx-airgap-bundle.tar.gz flx-airgap-bundle/
```

Check the bundle before the transfer:

```bash
ls flx-airgap-bundle flx-airgap-bundle/airgap-packages flx-airgap-bundle/pretrained-models
```

It must show `airgap-packages`, `pretrained-models` and `flx-source.tar.gz`; inside `airgap-packages` the
directories `packages` and `packages-torch`, and, if you chose them, `packages-dev`, `packages-extractor`,
`minutiae_extractor` and the file `requirements-extractor.txt`; inside `pretrained-models` the directory
`minutiae-extractor` with the two `.h5` files.

Transfer `flx-airgap-bundle.tar.gz` to the air-gapped machine on approved media.

### B2. On the air-gapped machine: install

```bash
tar -xzf flx-airgap-bundle.tar.gz
cd flx-airgap-bundle
tar -xzf flx-source.tar.gz
cd fixed-length-fingerprint-extractors
```

**Training environment.** PyTorch first, from its own directory, then the rest:

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install --no-index --find-links=../airgap-packages/packages-torch torch torchvision

pip install --no-index --find-links=../airgap-packages/packages \
  matplotlib numpy opencv-python pillow scipy scikit-learn torchmetrics tqdm setuptools wheel
```

The order matters: `torchmetrics` needs PyTorch, and pip keeps a PyTorch that is already installed instead
of taking the copy in `packages`.

Optional, for the tests and the notebooks:

```bash
pip install --no-index --find-links=../airgap-packages/packages-dev black jupyter pytest seaborn tikzplotlib
```

Then the `flx` package itself, without looking for anything online:

```bash
pip install --no-index --find-links=../airgap-packages/packages --no-build-isolation -e .
```

This step only makes `import flx` work. If `pip` reports a missing dependency here, a package of the steps
before is missing from the bundle.

Check:

```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -c "import flx; print('flx imported')"
python -m pytest -q          # only if you installed the optional tools
deactivate
```

**Extraction environment** (only for the model variants with minutiae), a second environment next to the
repository:

```bash
python3.11 -m venv ../venv-extractor
source ../venv-extractor/bin/activate
pip install --no-index --find-links=../airgap-packages/packages-extractor \
  -r ../airgap-packages/requirements-extractor.txt

# An absolute path, so that it also works from other directories
export PYTHONPATH="$(realpath ../airgap-packages/minutiae_extractor/src)"
python -c "from fingerflow.extractor import Extractor, direction_sense; print('minutiae extractor found')"
ls -l ../pretrained-models/minutiae-extractor
deactivate
```

The extractor is not installed into the environment; `PYTHONPATH` tells Python where its package is. The
variable is lost when you close the shell, so Step 4 of the training guide sets it again.

Next: [training-guide.md](training-guide.md).

---

## What was run of this document, and what was not

On one Linux x86_64 machine with 4 CPU cores, no GPU, Python 3.11:

| Part | Run? |
|------|------|
| A1, in a copy of this repository instead of a new clone: `python3.11 -m venv`, the CPU build of PyTorch, `pip install -e .`, `pip install -e ".[dev]"`, the three check lines | **Run.** PyTorch 2.15.0+cpu was kept by the later steps; the environment was 1.9 GB; 357 tests passed, 1 skipped |
| A1 without the CPU line (PyTorch with GPU libraries from PyPI), and `pip install -r requirements.txt` | Not run |
| A2: `git clone`, `uv sync`, `uv pip install -e .`, `scripts/install_model.py` | Not run for this document; the four model files were already on the machine |
| The extraction itself (Step 4 of the training guide) | **Run**, in an existing environment with TensorFlow 2.20, Pillow and tqdm, with `PYTHONPATH` set to `src` of the extractor's `main` (commit `420b6e5`) and with its `CoarseNet.h5` and `FineNet.h5`. This is the mechanism of Part B2 |
| The `grep -c` check | **Run** on the extractor's `main`: prints 1 |
| B1: `uv export ...` | **Run** on the extractor's `main`: 317 lines, including `tensorflow==2.20.0`, `pillow` and `tqdm` |
| B1: every `pip download` line, `scripts/install_model.py`, the `tar` and `mv` lines | Not run. (pip accepted `requirements-extractor.txt` and began to fetch the pinned packages for Python 3.11; that run was stopped after about 1 GB) |
| B2: every install line from the bundle | Not run. In earlier work on this repository, `pip install --no-index --find-links=... --no-build-isolation -e .` was run in a fresh environment that held only setuptools and wheel: it names the first missing dependency, as described above |
| Anything with a GPU or CUDA | Not run |

So Part B is written from the commands' documentation and from the pieces above, not from a complete run.
Do one complete trial on a spare machine before you rely on it. The size of the bundle was not measured;
expect several GB (TensorFlow and PyTorch with their GPU libraries, and 0.7 GB of model files).

---

## Notes

**GPU.** The NVIDIA driver must be installed on the machine separately; it is not part of the bundle. The
`cu128` build of PyTorch is the build for CUDA 12.8 and brings the CUDA libraries with it. Check with
`python -c "import torch; print(torch.cuda.is_available())"` in the training environment.

**Disk space, memory and time** depend on the number of images and subjects. See the storage table and the
section "Above about one million subjects" in [large-datasets.md](large-datasets.md).

**Notebooks.** They need `jupyter` from the optional tools. The last cell of
`notebooks/embedding_generation_tutorial.ipynb` (a plot) does not run with the package versions that the
optional tools install today; see the note at the top of that notebook.

**A development container.** `.devcontainer/` holds a container definition for Visual Studio Code with GPU
support. It was not built or used for these documents.

**Licence.** See [LICENSE](../LICENSE). Note that `pyproject.toml` names a different licence than that file;
which one holds is for the owner of the repository to settle.

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'` or `'flx'`.** The training environment is not active:
`source venv/bin/activate` in the root directory of the repository.

**`ModuleNotFoundError: No module named 'fingerflow'`.** The extraction environment is not active, or (Part B)
`PYTHONPATH` is not set in this shell. See "Before you run it" in
[Step 4 of the training guide](training-guide.md#step-4-extract-the-minutiae).

**The extraction stops with "too old".** The `fingerflow` package that Python finds is not the one of
[The minutiae extractor](#the-minutiae-extractor) above. The message shows where it was imported from.

**`pip` cannot find a package during the offline installation.** The file is missing from the bundle, or it was
downloaded for another Python version or system. Download it again on a machine that matches the air-gapped
one.

**Model files of the minutiae extractor not found.** `scripts/extract_minutiae.py` needs `CoarseNet.h5` and
`FineNet.h5`; check the paths given to `--coarse_net` and `--fine_net`.

**`CUDA out of memory` during training.** Lower `--batch_size` of `scripts/train_500k.py` (default 64).
