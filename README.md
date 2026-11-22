# flx: fixed-length fingerprint representations with DeepPrint

`flx` is a Python package that trains and runs [DeepPrint](https://arxiv.org/abs/1909.09901), a neural network
that turns one fingerprint image into a list of numbers of fixed length (an *embedding*). Two prints are
compared by comparing their embeddings.

The repository holds the package, three scripts that take you from a directory of your own fingerprint
images to a trained model, and the documents below. It began as the code of a BIOSIG 2023 paper
([docs/paper.md](docs/paper.md)). For the model variants that use minutiae it works together with a second
repository, the [minutiae extractor](https://github.com/craig-forefront/minutiae_extractor).

## What these documents rest on

Read this before you rely on any number in them.

- Everything was run on **one machine with 4 CPU cores and no GPU**.
- The largest training set was **440 real fingerprint images**. The synthetic SFinGe images in this
  repository are example data to try the code on, nothing more.
- **No accuracy, no GPU speed and no behaviour at large scale was measured.** Where a document gives a number,
  it says what it was measured on; where something was only calculated or read from the code, it says so.
- [docs/measurements.md](docs/measurements.md) lists every run and what was not run.

## The route, and the two repositories

| Step | What does it | Environment | Described in |
|------|--------------|-------------|--------------|
| 1. Lay out the images and say which resolution each file has | you | | [training guide](docs/training-guide.md), Steps 1 and 2 |
| 2. Bring every image to 500 ppi (pixels per inch) with dark ridges | this repository, inside the scripts (`SensorNormalizer`) | both | Step 2 |
| 3. Find the minutiae (only for the model variants with minutiae) | this repository's `scripts/extract_minutiae.py`, which calls the package `fingerflow` of the **minutiae extractor** repository and needs its two model files | extraction (TensorFlow) | Step 4 |
| 4. Find a learning rate, train | `scripts/find_learning_rate.py`, `scripts/train_500k.py` | training (PyTorch) | Steps 5 to 8 |
| 5. Compute embeddings, compare prints | `load_trained_model` in `scripts/train_500k.py`, `flx.benchmarks` | training (PyTorch) | Steps 9 and 10 |

*Minutiae* are the points where a ridge of the finger ends or splits. "Minutiae extractor" always means the
repository [craig-forefront/minutiae_extractor](https://github.com/craig-forefront/minutiae_extractor). Its
Python package is called `fingerflow`; it is **not** the `fingerflow` package on PyPI (the public Python
package index), which does not work here. You never run a script of that repository yourself; you need its
documentation only for what its output means.

The two *environments* are two separate Python installations, because the minutiae extraction needs
TensorFlow and everything else needs PyTorch.

## Installation

In short, for the training environment on a machine with network access:

```bash
git clone https://github.com/craig-forefront/fixed-length-fingerprint-extractors.git
cd fixed-length-fingerprint-extractors
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

Python 3.11 or higher is required. [docs/installation.md](docs/installation.md) has the full instructions: the
CPU-only build of PyTorch, the optional tools for tests and notebooks, the second environment for the minutiae
extraction, and the installation on a machine without network access (air-gapped).

## Shortest working example

This trains a model without minutiae for one epoch on the 100 synthetic example images of the repository (10
fingers with 10 images each). It shows that the installation works; the model it writes is useless. On the
CPU used here it took under two minutes, used up to 4.1 GB of main memory and wrote two model files of 0.5 GB
each into `models/example`.

```bash
python scripts/train_500k.py \
  --dataset_dir notebooks/example-dataset \
  --num_subjects 10 --num_impressions 10 \
  --assume_ppi 500 \
  --validation_split 0.2 \
  --num_epochs 1 \
  --batch_size 16 \
  --output_dir models/example
```

`--batch_size 16` keeps the memory small. With the default of 64 images per step the same command used up to
11.6 GB on the CPU; on a machine with less free memory the operating system then stops it with the one word
`Killed` and no Python error message.

`--assume_ppi 500` is only acceptable here because the images are examples; for your own images you say what
resolution each file really has. For your own data, follow [docs/training-guide.md](docs/training-guide.md)
from the start.

## Documents: read in this order

| Document | What it is for | When to read |
|----------|----------------|--------------|
| [docs/installation.md](docs/installation.md) | The two environments, with network access or air-gapped; how to get and check the minutiae extractor | First |
| [docs/training-guide.md](docs/training-guide.md) | The route step by step: file layout, resolution of the files, minutiae extraction, learning rate, training, continuing, embeddings, benchmark; troubleshooting; file formats | Second; follow it on a small part of your data |
| [docs/tuning.md](docs/tuning.md) | Learning rate, loss weights and learned alignment: what each setting is, what to watch, how to decide on your data | Before the first long training |
| [docs/large-datasets.md](docs/large-datasets.md) | Several GPUs, what limits the speed, storage, and what the code cannot do above about one million fingers | Before you plan more than a few hundred thousand images |
| [docs/measurements.md](docs/measurements.md) | The evidence: every measured table, what was run, what was not, and what changed against earlier versions of the code | When you want to know what a statement rests on, or have files from an earlier version |
| [docs/paper.md](docs/paper.md) | The paper this code comes from: architecture, its experiments, pre-trained model, citation | Background |
| Documentation of the [minutiae extractor](https://github.com/craig-forefront/minutiae_extractor) | What the extractor finds, its options, its measurements (also on latent prints) | When you need to know what is in the minutiae |
| [notebooks/](notebooks) | Three tutorials on the classes underneath (`Identifier`, `Dataset`, loaders, `fit`, `extract`), on the SFinGe example data | If you write your own loader or training script. They are not the route for your own data |
| [flx/visualization/det_curve_plotting/readme.md](flx/visualization/det_curve_plotting/readme.md) | Original readme of a bundled third-party plotting module | Only if you use that module |

## Scripts

| Script | Group | What it does |
|--------|-------|--------------|
| `scripts/extract_minutiae.py` | **the route** | Writes one minutiae file (`.mnt`) per image; can continue an interrupted run and work in parts on many machines |
| `scripts/find_learning_rate.py` | **the route** | Short test that suggests a learning rate before a training |
| `scripts/train_500k.py` | **the route** | Trains a DeepPrint variant; also holds `load_trained_model` for using the model |
| `scripts/run_original_minutiaenet.py`, `scripts/run_fingerflow_minutiaenet.py`, `scripts/run_minutiae_extractor.py`, `scripts/compare_minutiae_outputs.py`, `scripts/visualize_three_way.py` | comparison of minutiae extractors | Development tools that compare three minutiae extractors. They expect directories that are not part of this repository, are not needed for training and were not run for these documents |
| `flx/scripts/*.py` | the paper | Training, extraction, benchmarks, tables and plots of the paper's experiments on the paper's datasets ([docs/paper.md](docs/paper.md)). Not run for these documents |

For every script of the route, `--help` lists every option with its default; the docstring at the top of the
script explains the main groups of options.

## What is not covered

So that you do not search for it:

- **Tenprint decisions.** A "subject" in this code is one finger. The model compares single fingers; how the
  comparisons of ten fingers are combined into one decision is not part of this repository.
- **Cutting slap images into single fingers.** The scripts need one finger per image. The package has a
  `SlapSegmenter` class (`flx/image_processing/slap_segmentation.py`); it is not used by the scripts and was
  not run or checked for these documents.
- **Latent prints.** Nothing here was run on latents. The minutiae extractor's documentation has
  measurements of what the extractor finds on them.
- **Searching a gallery (identification).** The package has identification benchmarks (`flx/benchmarks`), used
  by the paper's scripts; the guides cover verification (pairs of prints) only.
- **Training on more than about one million fingers.** The code has limits there that it cannot work around
  today; [docs/large-datasets.md](docs/large-datasets.md) states them.

## Repository layout

    ├── README.md            <- This page
    ├── docs                 <- The documents listed above
    ├── LICENSE
    ├── pyproject.toml       <- Makes the package installable (`pip install -e .`); lists the dependencies
    ├── requirements.txt     <- The same dependencies plus the development tools, for `pip install -r`
    ├── scripts              <- The three scripts of the route, and five tools that compare minutiae extractors
    ├── notebooks            <- Tutorials, and `example-dataset` (100 synthetic SFinGe images with minutiae)
    ├── figures              <- Figures used by the documents
    ├── tests                <- Tests of the package and the scripts, with their data
    ├── models               <- Empty; trained models and training logs go here (not under version control)
    ├── .devcontainer        <- Container definition for Visual Studio Code (not used for these documents)
    │
    ├── data
    │   ├── benchmarks       <- Benchmark definitions of the paper (which comparisons to run)
    │   ├── fingerprints     <- Created by you: your image datasets
    │   ├── poses            <- Created by you: the output of the minutiae extraction
    │   └── embeddings       <- Created by you or the code: computed embeddings
    │
    ├── reports              <- Created by the paper's benchmark code (`flx/setup/paths.py`) for its results
    │
    └── flx                  <- The package
        ├── benchmarks       <- Biometric performance benchmarks (verification, identification)
        ├── data             <- Datasets of fingerprints, minutia maps, embeddings; the loaders
        ├── extractor        <- Embedding extraction with the DeepPrint model
        ├── models           <- DeepPrint architectures, loss functions, training loop, learning rate
        ├── image_processing <- Data augmentation, contrast enhancement, binarisation, slap segmentation
        ├── preparation      <- Images from different sensors (500 ppi, dark ridges) and the minutiae extraction
        ├── reweighting      <- Reweighting of embedding dimensions after extraction (paper)
        ├── scripts          <- Scripts of the paper's experiments
        ├── setup            <- Settings (`config.py`), predefined models and datasets of the paper
        └── visualization    <- Plots and graphical debugging

The root directory also holds ten image files and `comparison_results.csv`. They are outputs of the
extractor comparison tools; no document, test or module uses them.

## Tests

```bash
python -m pytest -q
```

358 tests, CPU only. One of them is skipped if the package `scikit-image` is not installed. `pytest` is one
of the optional tools ([docs/installation.md](docs/installation.md)).

## Credit to other authors

- Remi Cadene: Inception v4 PyTorch implementation (`flx/models/InceptionV4.py`);
  https://github.com/Cadene/pretrained-models.pytorch
- Dong Chengdong Hang Zhou: ISO 19794-2 fingerprint template encoder and decoder
  (`flx/data/iso_encoder_decoder`); https://github.com/DongChengdongHangZhou/iso-19794-2-decoder-encoder
- Andreas Nautsch: DET curve plotting (`flx/visualization/det_curve_plotting`)
- T. Rohwedder, D. Osorio-Roig, C. Rathgeb, C. Busch: the original `flx` code and the paper
  ([docs/paper.md](docs/paper.md) has the citation)

## Licence

See [LICENSE](LICENSE).
