# Training DeepPrint on your own images

This guide takes you from a directory of fingerprint images to a trained DeepPrint model and its embeddings.
DeepPrint is a neural network that turns one fingerprint image into a list of numbers of fixed length, the
*embedding*. Two prints are compared by comparing their embeddings.

The route uses three scripts of this repository, in this order:

1. `scripts/extract_minutiae.py` writes one minutiae file per image. It is only needed for the model variants
   with a minutia branch (Step 3 explains the choice). It runs in the *extraction environment*.
2. `scripts/find_learning_rate.py` suggests a learning rate. It runs in the *training environment*.
3. `scripts/train_500k.py` trains the model. It runs in the *training environment*. (The name is historical;
   the number of images is an option.)

The two environments are two Python installations: the minutiae extraction needs TensorFlow and a second
repository, everything else needs PyTorch. [installation.md](installation.md) sets up both.

**What this guide rests on.** Every command and code block in the steps below was run on a small set of images
on a machine with 4 CPU cores and no GPU. [measurements.md](measurements.md#what-was-run-for-these-documents)
lists each run. Nothing was run on a GPU or on a large dataset, and no accuracy was measured. For a large
collection read [large-datasets.md](large-datasets.md) before you plan anything.

## Contents

- [Words used in this guide](#words-used-in-this-guide)
- [Before you start](#before-you-start)
- [Step 1: Lay out the files](#step-1-lay-out-the-files)
- [Step 2: Say what resolution and ridge polarity each file has](#step-2-say-what-resolution-and-ridge-polarity-each-file-has)
- [Step 3: Decide whether to train with minutiae](#step-3-decide-whether-to-train-with-minutiae)
- [Step 4: Extract the minutiae](#step-4-extract-the-minutiae)
- [Step 5: Find a learning rate](#step-5-find-a-learning-rate)
- [Step 6: Run the training](#step-6-run-the-training)
- [Step 7: Continue a training, or start again](#step-7-continue-a-training-or-start-again)
- [Step 8: Read the training log](#step-8-read-the-training-log)
- [Step 9: Extract embeddings](#step-9-extract-embeddings)
- [Step 10: Run a verification benchmark](#step-10-run-a-verification-benchmark)
- [Several GPUs](#several-gpus)
- [If your data does not fit the layout](#if-your-data-does-not-fit-the-layout)
- [Troubleshooting](#troubleshooting)
- [Quick reference](#quick-reference)
- [Reference: the files that the scripts write](#reference-the-files-that-the-scripts-write)

---

## Words used in this guide

| Word | Meaning here |
|------|--------------|
| Subject | **One finger.** Two fingers of one person are two subjects. This differs from everyday use, where a subject is a person |
| Impression | One image of a finger. A subject has several impressions |
| ppi | Pixels per inch: the resolution at which a sensor captured the image. Most fingerprint sensors deliver 500 ppi, some 1000 ppi |
| Ridge polarity | Whether the ridges of the finger are the dark lines of the image (ink on paper, most optical sensors) or the light ones |
| Minutia (plural minutiae) | A point where a ridge ends or splits, with a position and a direction |
| Minutiae extractor | The program that finds the minutiae of an image. Here: the Python package `fingerflow` of the separate repository [minutiae_extractor](https://github.com/craig-forefront/minutiae_extractor). It is **not** the package called `fingerflow` on PyPI (the public Python package index), which does not work with this repository |
| `.mnt` file | A text file with the minutiae of one image, written by `scripts/extract_minutiae.py` |
| Minutia map | A picture made from the minutiae of an image. During training the model learns to draw it; this is a helper task and is not used when embeddings are computed |
| Embedding | The list of numbers that the model computes for one print |
| Variant | One of the four models that `scripts/train_500k.py` can train: with or without minutia branch, with or without localization network (Step 3) |
| Batch, step, epoch | The model is trained on a handful of images at a time (a *batch*); one such update is a *step*; one pass over all training images is an *epoch* |
| Loss | The number that the training makes smaller: it says how wrong the model's answers on a batch were |
| Learning rate | The size of the change that one step makes to the model ([tuning.md](tuning.md#choosing-the-learning-rate)) |
| Validation set | Subjects that are held out of the training and used to measure the model after every epoch |
| Mated, non-mated | A pair of prints of the same finger is *mated*; a pair of prints of different fingers is *non-mated* |
| Equal error rate (EER) | The error rate at the decision threshold where false matches (non-mated pairs accepted) and false non-matches (mated pairs rejected) are equally frequent. Lower is better |

---

## Before you start

You need:

1. Both environments of [installation.md](installation.md) (the part with network access or the air-gapped
   part). If you train without minutiae, the training environment is enough.
2. For each image file: which finger it shows, and at which resolution (ppi) it was captured.
3. A shell in the root directory of this repository. All commands of this guide are started there.

**Do the whole route on a small part of your data first**: a few hundred prints with prints from every kind of
sensor, through the extraction, the range test, two epochs of training and the embeddings. Mistakes in the
file layout, the resolutions or the ridge polarity show there, before a long run. Choose the size by the
minutiae extraction, which is the slow part: on the CPU used here it took 14 to 60 seconds per print
([measurements.md](measurements.md#minutiae-extraction-time-and-settings)). A few hundred prints are enough
to find such mistakes; they are not enough to judge the accuracy of the model.

---

## Step 1: Lay out the files

`scripts/train_500k.py` finds the finger and the capture of an image in its file name:

```
data/fingerprints/my-dataset/
├── optical_500/
│   ├── 1_1.png        <- subject 1, impression 1
│   ├── 2_1.png
│   └── ...
├── optical_1000/
│   ├── 1_2.png        <- subject 1, impression 2
│   └── ...
└── capacitive_light/
    ├── 1_3.png
    └── ...
```

- The name is `<subject>_<impression>.png`, both whole numbers from 1.
- A **subject** is one finger. Two fingers of one person are two subjects, and all images of one finger have the
  same subject number, whatever sensor they come from. An **impression** is one image of that finger. How you
  number the fingers is up to you, for example person 1 as subjects 1 to 10, person 2 as 11 to 20.
- Subjects are numbered 1 to `--num_subjects` **without gaps**, impressions 1 to `--num_impressions`, and **every
  subject needs all its impressions**. The script reads exactly these files.
- The files may be directly in the directory or in any subdirectories. Subdirectories per sensor, as above, make
  the resolution manifest of Step 2 short. The
  subdirectory is not part of the identity of a file: `optical_500/1_1.png` and `optical_1000/1_1.png` are the same
  subject and impression, and the script stops when it finds both.
- Files must be single fingers, 8-bit grayscale, as the sensor delivered them (not resized or enhanced).

Before it trains, the script compares the files with the two options. It stops with a message that names the first
missing files if some are not there, and it prints how many image files it does not use (subject number above
`--num_subjects` or impression number above `--num_impressions`).

If your data does not fit this layout:

- **Other names or another image format:** rename or link the files (a symbolic link per image is enough), or
  change `_extension` and `_file_to_id_fun` of `CustomFingerprintLoader` in `scripts/train_500k.py`.
- **A different number of images per finger:** the script cannot express that. Either choose `--num_impressions`
  as the smallest number that every finger has and number the images of each finger from 1 (the others are not
  used), or leave out fingers with fewer images and renumber the subjects without gaps.
- **Fingers with a single image:** with `--num_impressions 1` the script only trains with
  `--validation_split 0` (no validation set, so `best_model.pyt` is the epoch with the lowest training loss
  and there is no early stopping; see Step 8). A dataset in which some fingers have one image and others have
  more cannot be used as it is: either every finger contributes one image, or the single-image fingers are
  left out.
- **Multi-finger images** (four-finger slaps) must be cut into single fingers first. This guide has no step
  for that. The package contains a `SlapSegmenter` (`flx/image_processing/slap_segmentation.py`); it is not
  used by the three scripts and was not run or checked for these documents.
- **Ten fingers of one person (tenprint):** the model compares single fingers. How the ten comparisons of two
  tenprint cards are combined into one decision is not part of this repository.

**In the code** a sample is named by an `Identifier(subject, impression)`, and both numbers start at 0 there,
one less than in the file names: the file `17_3.png` is `Identifier(subject=16, impression=2)`. You meet this
when you read embeddings or logs (Steps 9 and 10).

---

## Step 2: Say what resolution and ridge polarity each file has

Do this step for every dataset, also if all prints come from one 500 ppi sensor with dark ridges: the scripts
do not guess the resolution of a file.

**Why it matters.** The minutiae extractor only works near 500 ppi and expects dark ridges on a light
background. It does not rescale or invert anything, and it does not fail on other images: it silently returns
few and wrong minutiae. In the one test made (7 prints), 1000 ppi files that were treated as 500 ppi files gave
5 % of the minutiae that examiners had marked, instead of 43 %; with the resolution declared, the 1000 ppi
files gave the same result as the 500 ppi files. The table is in
[measurements.md](measurements.md#images-from-different-sensors).

**What the scripts do.** `scripts/extract_minutiae.py`, `scripts/train_500k.py` and `scripts/find_learning_rate.py`
load every image file through the same function (`SensorNormalizer.load` in
`flx/preparation/sensor_normalization.py`). It resamples the image to 500 ppi and, if the file is declared to have
light ridges, inverts it. A file that is at 500 ppi with dark ridges is used unchanged, bit for bit. Because the
extraction and the training load the file in the same way, the minutiae in a `.mnt` file are in the pixels of the
image that the training loads. **Give all three scripts the same options.**

**Where the resolution of a file comes from**, in this order:

1. `--ppi 1000`: all files have this resolution. Use it only for a dataset from one kind of sensor.
2. `--ppi_manifest FILE`: a CSV file with a row per file or per directory. This is the option for mixed sensors.
3. The metadata of the image file (PNG, TIFF, JPEG and BMP files can store their resolution). A value is only used
   if it is between 200 and 2100 ppi, because image programs write 72 or 96 into files without knowing the sensor.
   The example images in `tests/data/SFingeExample` state 96 ppi, for instance.
4. `--assume_ppi 500`: for the files that are still without a value.

If none of the four gives a value, the file fails with a message that names these options (how the extraction
and the training fail is described in Steps 4 and 6). The scripts never guess a resolution. With `--assume_ppi` you take the responsibility that the files without a value really have that
resolution.

**`--assume_ppi` does not override the metadata of a file.** A file that states a value between 200 and 2100 ppi
is resampled by that value, also when `--assume_ppi` is given. Image programs do not only write 72 or 96: a
500 ppi print that was saved with a 300 or 600 ppi stamp is enlarged or shrunk, in the extraction and in the
training alike, so nothing else notices (this was tried on one file, see
[measurements.md](measurements.md#images-from-different-sensors)). Therefore:

- For a dataset of **one known resolution**, use `--ppi`. It wins over the metadata.
- For mixed sensors, use the manifest. A manifest row also wins over the metadata.
- Read the counts that the scripts print. The extraction ends with "Resolution of the files extracted in this
  run", the training starts with "Resolution of ... training images" (up to 1,000 files spread over the set): one
  line per source and value, and a note when files took their resolution from metadata although `--assume_ppi`
  was given.

The manifest has a header line and the columns `path`, `ppi` and optionally `ridges`. Paths are relative to the image
directory (`--input_dir` of the extraction, `--dataset_dir` of the training). A row for a directory applies to all
files below it; the row of a file wins over that of its directory. If the files of each sensor are in their own
directory, the manifest has one row per sensor:

```
path,ppi,ridges
optical_500,500,dark
optical_1000,1000,dark
capacitive_385,385,light
optical_500/retakes/7_3.png,1000,
```

If your sensors are not separated by directory, write one row per file. Every process reads the manifest into
memory, which took 186 bytes per row for a manifest of 1 million file rows with paths of about 35 characters. For
100 million file rows that is about 19 GB per process, so for a dataset of that size directory rows are the
practical form.

**Ridge polarity.** `--ridges dark` (the default) uses the files as they are. `--ridges light` inverts all files.
`--ridges auto` decides per image: the scripts compare the gray level of the background (blocks of the image with
little contrast) with that of the print, and take the image as light-ridged if the background is darker. A `ridges`
value in the manifest wins over the option. Prefer the manifest: a sensor delivers one polarity, so one row per
sensor is exact, while the estimate needs background in the image. On images that show background the estimate
was right every time it was tried; on crops with little or no background it often gave no decision and was
sometimes wrong ([measurements.md](measurements.md#images-from-different-sensors)). Without a decision, `auto`
takes the image as dark-ridged.

**Write the manifest now**, as `data/fingerprints/my-dataset-resolutions.csv` (next to the image directory, not
inside it), and use it in every command of Steps 4, 5, 6 and 9. If all files have one resolution and dark
ridges, you can use `--ppi 500` (or your value) in place of `--ppi_manifest ...` in all of these commands
instead.

---

## Step 3: Decide whether to train with minutiae

`scripts/train_500k.py` trains one of four DeepPrint variants. Two of its options choose it:

| Variant | `--minutiae_dir` | `--localization` | Minutiae needed |
|---------|------------------|------------------|-----------------|
| `DeepPrint_LocTexMinu` | given | `on` (the default) | yes |
| `DeepPrint_TexMinu` | given | `off` | yes |
| `DeepPrint_LocTex` | not given | `on` (the default) | no |
| `DeepPrint_Tex` | not given | `off` | no |

- `Tex` is the *texture branch*: the part of the model that computes an embedding from the ridge pattern as a
  whole. `Minu` is the *minutia branch*: a second part that computes a second embedding and, during training,
  draws the minutia map. `Loc` is the *localization network*, a learned alignment of the image.
- **With minutiae** you need a `.mnt` file for every training image (Step 4), which is the slowest part of the
  whole route. **Without minutiae** you skip Step 4 and leave `--minutiae_dir` out of the commands of Steps 5
  and 6; the script then trains a texture-only variant and says so in its first lines.
- Which variant matches better on your data was not measured here. The DeepPrint paper and the paper behind
  this repository report results for their data ([paper.md](paper.md)).
- Whether the localization network makes matching more accurate depends on the data and has to be measured.
  [Learned alignment](tuning.md#learned-alignment) in tuning.md says how it works, what keeps it stable and
  how to compare a training with and without it. In the short runs made here it was stable and did nothing
  measurable.

**Embedding size.** `--embedding_dim` is the number of values **per branch** (default 256). A variant with
minutiae has two branches, so the embedding of a print has twice that many values: 512 with the default. A
texture-only variant has 256 with the default. The paper behind this repository found 512 values best for the
texture-only variant on its data; what is best on your data was not measured here. Smaller embeddings need less
memory in the classifier (see [large-datasets.md](large-datasets.md#above-about-one-million-subjects)) and in the
gallery.

---

## Step 4: Extract the minutiae

Skip this step if you train without minutiae (Step 3).

### Before you run it

The extraction runs in the **extraction environment**, not in the training environment. In a new shell, in the
root directory of this repository, give the lines for the part of [installation.md](installation.md) that
you followed.

If you installed with network access (Part A):

```bash
source ../minutiae_extractor/.venv/bin/activate
MODELS=../minutiae_extractor/models
```

If you installed air-gapped (Part B):

```bash
source ../venv-extractor/bin/activate
export PYTHONPATH="$(realpath ../airgap-packages/minutiae_extractor/src)"
MODELS=../pretrained-models/minutiae-extractor
```

Then check that Python finds the minutiae extractor and the two model files:

```bash
python -c "from fingerflow.extractor import Extractor, direction_sense; print('minutiae extractor found')"
ls -l $MODELS/CoarseNet.h5 $MODELS/FineNet.h5
```

If the first line fails with `No module named 'fingerflow'`, you are in the wrong environment or `PYTHONPATH`
is not set in this shell. `PYTHONPATH` and `MODELS` are lost when you close the shell.

### The command

```bash
python scripts/extract_minutiae.py \
  --input_dir data/fingerprints/my-dataset \
  --output_dir data/poses/my-dataset \
  --coarse_net $MODELS/CoarseNet.h5 \
  --fine_net $MODELS/FineNet.h5 \
  --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
  --skip_existing \
  --workers 1
```

This creates one `.mnt` file per image, in the same directory structure as the images, and
`extraction_settings.json` in the output directory. At its end the script prints how many files succeeded,
failed and were skipped, and from where the files took their resolution ("Resolution of the files extracted in
this run"). Read that block.

- Feed the image files as the sensor delivered them: 8-bit grayscale, not resized or enhanced. The script
  resamples them to 500 ppi itself. The coordinates in a `.mnt` file are in the pixels of the 500 ppi image.
- `--ppi_manifest` is the file of Step 2. `--ppi`, `--assume_ppi` and `--ridges` are the other options
  described there. **Give the training (Steps 5, 6 and 9) exactly the same ones.**
- `--skip_existing` leaves out images that already have a `.mnt` file, so the same command continues a run
  that was interrupted.
- `--workers W` starts W worker processes; each loads its own copy of the networks.
- `--classify_net` is optional. ClassifyNet assigns a type (ending, bifurcation, ...) to every minutia. DeepPrint
  does not read the type, so leave the option out: the class index in the file is then -1 and the extraction is
  faster (in the one timing made, 19 instead of 33 seconds per print, with the same positions and angles).
  CoreNet, the fourth network of the extractor, is never run (`--core_net` is accepted and ignored).
- `--image_dpi` is only used to find the direction of the minutiae, and it refers to the image after the
  resampling. Leave it at 500; it is not the option for the resolution of the files (that is `--ppi`), and the
  script prints a warning when it is not 500.
- Which revision of the minutiae extractor is needed, and how to check a clone, is in
  [installation.md](installation.md#the-minutiae-extractor). With an older one, or with the `fingerflow` package
  from PyPI, the script stops before the first image with exit status 2 and a message that says "too old".

### What the extractor is run with

The script runs the minutiae extractor with its default settings. The extractor has two options that find
more minutiae, `crop_to_print` and `reweight_score`; **both are off here and cannot be set from this script**,
and `extraction_settings.json` does not record them. The extractor's documentation (section "Finding more
minutiae", in its README or, in newer revisions, in its `docs/usage.md`) describes them and what was measured
about them on minutiae. What they do to a trained DeepPrint model was not measured in either repository. All
figures in these documents were measured with the default settings. Using the options would need a change in
`make_extractor` in `flx/preparation/minutiae_extraction.py`.

All `.mnt` files of a dataset must come from the same settings and the same extractor:

- The script stores its settings in `extraction_settings.json` and refuses to add files with other settings
  to the same output directory (exit status 2). To extract again with other settings, use a new
  `--output_dir`, or give `--allow_settings_change` without `--skip_existing`, which extracts every file
  again.
- A change of the minutiae extractor itself is **not** detected, because its revision is not recorded. Note
  it yourself next to the output (for the air-gapped layout the clone has no git history; note the commit when
  you build the bundle), and use a new output directory when the extractor changes:

  ```bash
  git -C ../minutiae_extractor rev-parse HEAD > data/poses/my-dataset-extractor-revision.txt
  ```

Nothing in this repository was run on latent prints (prints lifted from a surface). What the extractor finds
on latents is measured in the extractor's documentation (section "Direction sense" of its README or its
`docs/measurements.md`); read it before you plan minutia maps for latents.

### A very large number of files

The extraction is the slow step: 14 to 60 seconds per print on the CPU used here
([measurements.md](measurements.md#minutiae-extraction-time-and-settings)). No GPU was available, so the
speed on your hardware must be measured: time one shard of a few hundred files before you choose the number of
shards. Plan the extraction as a job that runs on many machines, is interrupted and is continued.

1. **Make the file list once**, so that every process works on the same list in the same order:

   ```bash
   (cd data/fingerprints/my-dataset && find . -name '*.png' | sort > ../my-dataset-files.txt)
   ```

   The parentheses keep your shell in the repository root, where the next commands must be started.

   Without `--file_list` the script walks the image directory in sorted order itself, which gives the same list as
   long as no file is added or removed.

2. **Start one process per shard.** A shard is one of N equal parts of the list: shard I holds the files number I,
   I + N, I + 2N, ... (counted from 0). Every file is in exactly one shard.

   ```bash
   python scripts/extract_minutiae.py \
     --input_dir data/fingerprints/my-dataset \
     --file_list data/fingerprints/my-dataset-files.txt \
     --output_dir data/poses/my-dataset \
     --coarse_net $MODELS/CoarseNet.h5 --fine_net $MODELS/FineNet.h5 \
     --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
     --num_shards 64 --shard_index 0 \
     --skip_existing
   ```

   Run the same command with `--shard_index 1` to `63` on the other machines or GPUs. All shards write into the same
   output directory, so it must be on storage that all machines share. `--workers W` starts W worker processes
   inside one shard; each loads its own copy of the networks, so on a GPU keep it at 1 unless the memory allows
   more.

3. **Continue after an interruption** by running the same command again. With `--skip_existing`, images that already
   have a `.mnt` file are left out. A `.mnt` file is written under a temporary name and renamed when it is complete,
   so a file with the ending `.mnt` is always a finished one, also after a crash. Without `--skip_existing` every
   file is extracted again.

4. **Look at the failures.** One bad file never stops the run. Each failure is written at once to
   `extraction_errors.shard-00000-of-00064.txt` (`extraction_errors.txt` without shards) in the output directory:
   the path, a tab, and the reason, for example an unreadable file, an unknown resolution or an error inside the
   extractor. The exit status of the process is 1 if any file of its run failed, 0 if none did, and 2 if the run did
   not start (settings differ from those stored in the output directory, or the minutiae extractor is too old). Every run starts its log anew; with
   `--skip_existing` the log then lists exactly the files of the shard that still have no `.mnt` file. If a worker
   process dies (for example because the machine runs out of memory), the files that were being worked on are logged
   as failed and the run goes on with new processes; this happened once in the runs made here. Remove the
   cause and run the command again to extract only the failed files. If the script itself ends with a Python
   error message (a "traceback") and exit status 1 before any file, the extractor could not be set up: the
   `fingerflow` package or a model file was not found (see [Troubleshooting](#troubleshooting)).

   Also read `extraction_warnings.txt` (with shards: `extraction_warnings.shard-00000-of-00064.txt`; a run
   without warnings leaves no file). The extraction estimates the ridge polarity of every image, whatever the
   setting. An image that looks different from what it was declared to be is still extracted and is listed
   there with the reason. Many entries from one sensor mean that its polarity is declared wrongly in Step 2.

5. **Check that it is complete** before the training: run every shard once more with `--skip_existing`. Each should
   end with "Successful: 0", "Failed: 0" and exit status 0. The training needs a `.mnt` file for every image that it
   trains on, and stops before the first epoch with the names of the first missing ones. Images that can not be
   extracted must be taken out of the dataset; because the subjects must be numbered without gaps (Step 1), that
   means renumbering, or replacing the image. A print for which the extractor finds no
   minutia gets a `.mnt` file with 0 minutiae; the training leaves it out of the minutia map loss and uses it for
   the rest (see [Loss weights](tuning.md#loss-weights)). The script prints how many there were.

The image loader of `scripts/train_500k.py` reads one file extension (`.png` unless you change `_extension`,
see [If your data does not fit the layout](#if-your-data-does-not-fit-the-layout)), while the extraction reads
PNG, JPEG, BMP and TIFF files. Convert the files to one format, or adapt the loader.

**When the extraction is done, leave the extraction environment** (`deactivate`, or open a new shell) and
enter the training environment for all following steps:

```bash
source venv/bin/activate
```

---

## Step 5: Find a learning rate

The *learning rate* is the size of the change that every training step makes to the model. Too high a value
destroys a training within a few steps, too low a value wastes time, and the right value depends on the data,
the variant and the batch size. `scripts/find_learning_rate.py` finds a starting value: it trains a fresh model
on a part of your data for a few hundred steps while it raises the learning rate after every step, from far too
low (0.0000001) to far too high (1). It reads the same directory layout as the training and saves no model.

```bash
python scripts/find_learning_rate.py \
    --dataset_dir data/fingerprints/my-dataset \
    --minutiae_dir data/poses/my-dataset \
    --num_subjects 50000 --num_impressions 10 \
    --max_subjects 2000 \
    --batch_size 64 \
    --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
    --output_dir models/lr_range_test
```

- `--minutiae_dir` is the output directory of Step 4. Leave it out if you train without minutiae.
- `--num_subjects` and `--num_impressions` are what the directory holds (Step 1).
- `--ppi_manifest` (or `--ppi`, `--assume_ppi`, `--ridges`, `--canvas_size`) must be what the minutiae extraction
  and the planned training use (Step 2). Without any of them, a file that does not state its resolution stops
  the test.
- `--max_subjects 2000` uses a random 2,000 of the subjects. The test needs
  `--num_steps` times `--batch_size` images (19,200 with the defaults); a
  subset with at least that many images is enough.
- `--batch_size` must be the `--batch_size` of the planned training (both
  scripts default to 64). The script reads the same file layout as the
  training and stops with the same message when files are missing.
- `--variant` chooses the model (`LocTexMinu`, `TexMinu`, `LocTex`, `Tex`).
  Without it the script tests what `train_500k.py` trains by default. For a
  training with `--localization off`, test `TexMinu` or `Tex`.
- `--localization_lr_factor` must be the value of the planned training
  (`Loc` variants only).
- `--gradient_clip_norm` must be the value of the planned training.
- `--seed` changes the subset, the starting model and the order of the
  images. Run the test with two or three seeds, look at the plots and take
  the lowest suggestion: in the tests made here the suggestion moved by a factor of about 3 from one
  seed to the next ([measurements.md](measurements.md#how-far-to-trust-the-suggestion-of-the-range-test)).

The script writes `lr_range_test.csv` (learning rate and loss of every step),
draws `lr_range_test.png` and prints a suggested rate.

How to read the plot, what the warm-up and the schedule are, how the rate changes with the batch size and the
number of GPUs, and what to watch in the first thousand steps of the training is in
[Choosing the learning rate](tuning.md#choosing-the-learning-rate) in tuning.md. The default of the training,
0.0001, did not diverge in the short runs made here; it is not known to be the best rate for your data.

---

## Step 6: Run the training

One complete command for a dataset of 50,000 fingers with 10 images each, with minutiae and a resolution manifest:

```bash
python scripts/train_500k.py \
  --dataset_dir data/fingerprints/my-dataset \
  --minutiae_dir data/poses/my-dataset \
  --num_subjects 50000 --num_impressions 10 \
  --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
  --learning_rate 0.0001 --lr_schedule cosine \
  --num_epochs 10 \
  --output_dir models/my-model-v1
```

- `--num_subjects` and `--num_impressions` must be what the directory holds (Step 1). They have defaults
  (50,000 and 10), but always give them.
- `--minutiae_dir` is the output directory of Step 4. Without it the script trains a texture-only variant
  (Step 3).
- `--ppi_manifest` (or `--ppi`, `--assume_ppi`, `--ridges`) must be exactly what the minutiae extraction was
  given (Step 2). Without any of them, a file that does not state its resolution stops the training, but only
  after it has started: the script prints "resolution unknown: these files fail" in its block "Resolution of
  ... training images", starts the first epoch and ends in the first step with an error message that names
  the file ("Resolution unknown", exit status 1). The output directory then holds settings files and no
  `model.pyt`; the next start with the right options is a new training.
- `--learning_rate` is the value from Step 5. `--lr_schedule cosine` lowers the rate over the training. All
  learning rate options are explained under
  [Choosing the learning rate](tuning.md#choosing-the-learning-rate).
- `--batch_size` is the number of images per step on each GPU (default 64). Lower it if the GPU runs out of
  memory. The range test of Step 5 must use the batch size of the training.
- Validation is on by default: 10 % of the subjects, at most 5,000, are held out (see
  [Which epoch becomes `best_model.pyt`](#which-epoch-becomes-best_modelpyt)). `--validation_split 0` trains
  without validation.
- `--localization off` trains without the localization network (Step 3).
- The loss weights have options too (`--loss_weight_...`); see [Loss weights](tuning.md#loss-weights). Leave
  them at their defaults unless you compare settings.
- On several GPUs the same options follow `torchrun`; see [Several GPUs](#several-gpus).
- `python scripts/train_500k.py --help` lists all options; the docstring at the top of the script describes them.

**How many epochs.** `--num_epochs` is an upper limit (default 10); with validation the training stops earlier
when the validation error rate has not fallen for `--patience` epochs (default 3). How many epochs a large
dataset needs was not measured here. The trainings of the paper, on 60,000 synthetic images, ran for up to 100
epochs; with far more images per epoch fewer are needed. With `--lr_schedule cosine` the learning rate falls over
exactly `--num_epochs` epochs, so set the number once and do not change it in a running training.

**What the script prints first** (shortened; from a run on 12 images of 4 fingers):

```
============================================================
Training DeepPrint with Learned Alignment (LocTexMinu (texture + minutiae))
============================================================
Dataset:      data/fingerprints/my-dataset
Minutiae:     data/poses/my-dataset
Images:       resampled to 500 ppi; resolution from manifest (3 rows), file metadata, else the file fails; ridges dark
Canvas:       square of the longer side of each image
Batch size:   4 images per step and GPU
Subjects:     4 total
  Training:   2 subjects
  Validation: 2 subjects held out (seed 42)
Impressions:  3
...
Embedding:    256 values per branch, 512 per print
...
Scanning data/fingerprints/my-dataset for .png files...
Resolution of 6 training images spread over the set:
             4  1000 ppi from manifest (resampled to 500 ppi)
             2  500 ppi from manifest
```

Read these lines before you let a long training run: the variant, the number of training subjects, and where the
resolution of the images comes from. The last block looks at up to 1,000 training images. If it shows files that
took a resolution "from metadata of the file" which you did not expect, go back to Step 2.

**What the script writes** to `--output_dir`:

| File | Content |
|------|---------|
| `model.pyt` | The model, the loss and the optimizer after the newest epoch. The training continues from it |
| `best_model.pyt` | A copy of `model.pyt` of the best epoch so far |
| `log.json` | One entry per epoch, see [Step 8](#step-8-read-the-training-log) |
| `training_settings.json` | The settings of the training, as of the last run that trained an epoch. Needed to continue it and to load the model later |
| `data_preparation.json` | The image preparation settings alone (also part of `training_settings.json`) |

**Different image sizes: `--canvas_size`.** The sensors may deliver images of different sizes; every `.mnt` file
carries the size of its own image. By default the training pads every 500 ppi image to a square of its own
longer side and scales that to the model input of 299x299 pixels. A 256x360 image of a small sensor is then
shown 2.2 times larger than a 800x750 rolled print, although both are at 500 ppi. To show all sensors at one
scale, give the training `--canvas_size N` (and the range test of Step 5 the same value): every 500 ppi image
is cut or padded around its centre to N x N pixels before it is scaled, and the minutiae are moved with it.
Things to know before you choose N:

- The scale of the model input is 299 / N. With N = 512 a ridge period of about 9.5 pixels at 500 ppi becomes about
  5.5 pixels; with N = 800 about 3.6 pixels.
- Whatever is outside of the N x N square around the centre of the *image* (not of the print) is cut off. A rolled
  print of 800x750 pixels loses its sides with N = 512, and a print that is not in the middle of its image can lose
  more. Of the two real prints that were looked at with N = 512, the slap print (603x842 pixels at 1000 ppi) was
  complete, and the rolled print (1600x1500 pixels at 1000 ppi), which lies in the lower left of its image, lost a
  large part.
- Whether a canvas, and which N, gives better matching was not measured here. It needs a comparison on your data: two
  trainings that differ only in `--canvas_size`, compared by their validation equal error rate, in the way
  described for the learned alignment in
  [How to find out whether it helps on your data](tuning.md#how-to-find-out-whether-it-helps-on-your-data).
- The default is 0 (no canvas).

The training writes its preparation settings to `training_settings.json` (and, alone, to `data_preparation.json`)
in the output directory. When the trained model is used later, the images must be prepared in the same way;
`load_trained_model` in `scripts/train_500k.py` builds the loader from that file (Step 9). A training that
is continued with another preparation stops (Step 7).

**A check against mixed-up data.** The training compares the image size in the header of each `.mnt` file with the
size of the image that it loaded. If they do not fit (the image file was replaced, or extraction and training were
given different options), it stops with a message that names the sample and both sizes. Without `--canvas_size`
only a different shape (ratio of width to height) is an error, because image and minutiae are both scaled by the
longer side, so the same picture at another resolution still fits. With `--canvas_size` the sizes must be equal.

---

## Step 7: Continue a training, or start again

**The same command continues.** If `--output_dir` holds a `model.pyt`, the script loads it and trains the next
epoch. This is how you continue after an interruption: run the same command again. A training that was stopped
in the middle of an epoch starts that epoch again from the end of the one before; if it was stopped while an epoch
was being validated, that epoch is validated first.

**To start again from nothing, use a new `--output_dir`** (or delete the old directory). There is no option
that overwrites a training.

**A continued training must be the same training.** The script compares its options with
`training_settings.json`:

- It **stops** (exit status 2, with the list of differences) if one of these differs: the variant
  (`--localization`, `--minutiae_dir` given or not), `--embedding_dim`, `--num_subjects`, `--num_impressions`,
  the validation options, `--batch_size`, `--canvas_size`, or the image preparation (`--ppi`, the content of the
  manifest, `--assume_ppi`, `--ridges`). Train with such a change in a new `--output_dir`.
  `--allow_changed_settings` continues anyway; use it only if you know that the change is harmless (for example a
  smaller `--batch_size` after the GPU ran out of memory).
- It **continues and lists the changes** if only these differ: the learning rate options, the loss weights,
  `--num_epochs`, `--patience`. They hold from the next epoch on. `--localization_lr_factor` is one of them: a
  continued training uses the factor of the new command.
- **If nothing is left to train**, the script says "Nothing was trained in this run". That is the case when the
  log already has `--num_epochs` epochs, or when the training ended by early stopping. A changed learning rate
  has no effect on a finished training. What trains further epochs depends on how it ended, and the script's
  message says which case it is:
  - the log has `--num_epochs` epochs: raise `--num_epochs`;
  - it ended by early stopping (`--patience` epochs without a lower error rate): raise `--patience` (or pass
    `--patience 0`); a larger `--num_epochs` alone trains nothing;
  - or use a new `--output_dir` to train again from the start.

So "divide the learning rate by 3 and start again", as [tuning.md](tuning.md) sometimes advises, means: a new
`--output_dir` if you want a new training, the same one if you want to continue from the last saved epoch with
the lower rate.

The settings files are written when a new training starts, after the file check, and again after every run that
trained an epoch. A start that is refused, or that trains nothing, does not change them, and a first start
that is refused by these checks leaves the output directory empty. (A first start that passes the checks and
then fails in its first step, for example on a file of unknown resolution, leaves the settings files without
a `model.pyt`; see Step 6.) The changes that the script lists are therefore changes against the last run
that trained.

If a run was stopped while `best_model.pyt` was being written, the next start writes it again from `model.pyt`
(it prints "best_model.pyt: epoch N ... copied again"). While `model.pyt` or `best_model.pyt` is written, a file
`model.pyt.tmp` or `best_model.pyt.tmp` of the same size exists; a process that is killed outright can leave it
behind, and the next save overwrites it.

---

## Step 8: Read the training log

The training writes one entry per epoch to `log.json` in `--output_dir`:

```python
import json

# Load training log
with open("models/my-model-v1/log.json", "r") as f:
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
  (see [Loss weights](tuning.md#loss-weights)) and averaged per training sample; the
  minutia map loss is averaged over the samples that have a minutia map. The
  keys are `texture_loss` and `minutia_loss` (one per branch), each with
  `crossent_loss_sum` (cross entropy) and `center_loss_sum` (center loss),
  and `minutia_map_loss`. The names end in `_sum` for historical reasons;
  the values are means.
- `training_accuracy`: the share of training samples whose subject the model
  named correctly. Higher is better. It says nothing about fingers that are
  not in the training set.
- `learning_rate`: the learning rate of the last step of the epoch (see
  [Choosing the learning rate](tuning.md#choosing-the-learning-rate)).
- `alignment`: for the variants with localization network, what the learned
  alignment did in the epoch: mean and largest rotation and shift, the share
  of the print that stayed in the frame, and the share of aligned images
  that were mostly empty (see [Learned alignment](tuning.md#learned-alignment)). It is
  `null` for the other variants.
- `validation_equal_error_rate` (EER): measured on the validation set.
  Lower is better. It is
  `null` if the epoch was trained without a validation set, and for the
  newest epoch while its validation is still running: the entry is written
  together with `model.pyt`, before the validation.

### Which epoch becomes `best_model.pyt`

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
set would make the error rate look better than it is for new fingers.
`--max_validation_subjects` limits the number of held-out subjects (default
5,000; 0 removes the limit), so that validation after every epoch stays
short; the subjects above the limit stay in the training set. The reasons
for the limit are under
[large-datasets.md](large-datasets.md#above-about-one-million-subjects).
`--validation_split 0` switches validation off.

`TrainingLog.selection_metric` names the field that was used, and
`TrainingLog.best_entry` returns the entry of the best epoch:

```python
from flx.models.model_training import TrainingLog

log = TrainingLog("models/my-model-v1/log.json")
best = log.best_entry
print(f"Best: epoch {best.epoch}, selected on {log.selection_metric}")
```

---

## Step 9: Extract embeddings

`load_trained_model` in `scripts/train_500k.py` reads `training_settings.json` of a training and returns the model
and an image loader that prepares images exactly as the training did (resolution, ridge polarity, canvas, no
augmentation). The images to embed must be in the layout of Step 1. Run this in the training environment, from the
root directory of the repository.

```python
import sys

sys.path.insert(0, "scripts")  # the scripts directory of the repository
import train_500k
from flx.data.dataset import Dataset
from flx.data.embedding_loader import EmbeddingLoader

extractor, image_loader = train_500k.load_trained_model(
    "models/my-model-v1",              # the --output_dir of the training
    "data/fingerprints/my-test-set",   # images named <subject>_<impression>.png
    ppi_manifest="data/fingerprints/my-test-set-resolutions.csv",
)
dataset = Dataset(image_loader, image_loader.ids)
texture_embeddings, minutia_embeddings = extractor.extract(dataset)

# One vector per print: texture and minutia embedding side by side
embeddings = EmbeddingLoader.combine_if_both_exist(texture_embeddings, minutia_embeddings)
embeddings.save("data/embeddings/my-test-set")
print(f"{len(embeddings.ids)} embeddings of {embeddings.embedding_size} values")
```

- Leave `ppi_manifest` out if the training used none. If it is left out although the training used one, the path
  that the training was given is used, which only fits the training directory itself.
- `best_model.pyt` is loaded; `best=False` loads `model.pyt`, the newest epoch.
- The model has one class per **training** subject: the subjects of the dataset minus the held-out validation
  subjects (45,000 for 50,000 subjects with the default split). `load_trained_model` takes the number from
  `training_settings.json`. If you build the model by hand (`get_DeepPrint_...(num_training_subjects=...)`), it
  must be this number, the "Training:" line of the training output, or loading the parameters fails.
- A texture-only variant returns `None` as minutia embeddings; `combine_if_both_exist` then returns the texture
  embeddings.
- `save` writes `embeddings.npy` and `ids.json` into the directory; `EmbeddingLoader.load` reads them.

---

## Step 10: Run a verification benchmark

Continuing from Step 9. The benchmark compares pairs of prints of the same finger (mated) and of different
fingers (non-mated) by the cosine similarity of their embeddings:

```python
from flx.benchmarks.matchers import CosineSimilarityMatcher

# The pairs: all pairs of impressions per subject, and 10 non-mated pairs per subject
benchmark = train_500k.create_validation_benchmark(image_loader.ids, num_impressions=3)
result = benchmark.run(CosineSimilarityMatcher(embeddings))

print(f"Equal error rate: {result.get_equal_error_rate():.4f}")
threshold = result.threshold_for_fmr(0.001)
print(f"False non-match rate at a false match rate of 0.1 %: "
      f"{result.false_non_match_rate([threshold])[0]:.4f}")
```

- `num_impressions` is the number of impressions per subject of the test set.
- Use fingers that were neither in the training nor in the validation set. An error rate on the validation
  subjects is too optimistic, because the best epoch was chosen on them.
- With 10 non-mated pairs per subject, a false match rate of 0.1 % rests on very few pairs unless the test set has
  many thousands of subjects. For rates that matter in a large gallery, compare all pairs of the test set.
- Searching a gallery (identification) is not covered by this guide. `flx/benchmarks/identification.py` and the
  benchmark files under `data/benchmarks/` belong to the experiments of the paper ([paper.md](paper.md)); they
  were not run for these documents.

---

## Several GPUs

`scripts/train_500k.py` needs no change for several GPUs: it is started with `torchrun` and the same options.
The command, what the code does with several GPUs and what that means for the learning rate are in
[large-datasets.md](large-datasets.md#several-gpus). **No GPU run was made for these documents.**

---

## If your data does not fit the layout

`scripts/train_500k.py` is the route that is documented and tested. If your data cannot be brought into its
layout, the smallest change is inside that script:

- `CustomFingerprintLoader._extension` returns the file extension, and `CustomFingerprintLoader._file_to_id_fun`
  turns the subdirectory and the file name (without extension) into an `Identifier`. Subjects must still be
  numbered from 0 without gaps, because the subject number is the class of the classifier.
- `create_training_and_validation_ids` builds the identifiers as a grid of subjects and impressions. For an
  uneven set, build the `IdentifierSet`s from the files that exist instead.
- Keep `SensorNormalizer.load` in the loader. A loader that reads the files directly with OpenCV gets images in
  other pixels than the `.mnt` files for every sensor that is not at 500 ppi with dark ridges, and the check of
  the `.mnt` header against the image size is skipped when the loader resizes the image itself.
- The minutiae files are read by `FingerFlowMinutiaMapLoader` (`<subject>_<impression>.mnt`) or, for other names,
  `FingerFlowMinutiaMapLoaderCustomNaming` in `flx/data/fingerflow_minutia_loader.py`.

If you write your own image loader instead, load the files with `SensorNormalizer.load` there as well, as
`CustomFingerprintLoader` in the script does.

The notebooks [dataset_tutorial.ipynb](../notebooks/dataset_tutorial.ipynb) and
[model_training_tutorial.ipynb](../notebooks/model_training_tutorial.ipynb) show the classes underneath on the
SFinGe example data (synthetic prints that come with the repository). They are not the route for your own
data: they read minutiae that the SFinGe generator wrote, not `.mnt` files.

---

## Troubleshooting

#### 1. `ModuleNotFoundError: No module named 'fingerflow'` (extraction)

The shell is not in the extraction environment, or `PYTHONPATH` is not set in it. Do the lines of
[Before you run it](#before-you-run-it) in Step 4 again. The script ends with a Python error message and exit
status 1, and it has already written `extraction_settings.json` into the output directory; that file is
harmless and is used by the next run.

#### 2. The extraction stops with "too old" (exit status 2)

The `fingerflow` package that Python finds is not the one of the minutiae_extractor repository at the
revision that [installation.md](installation.md#the-minutiae-extractor) names: it is an older clone, the
upstream repository or the package from PyPI. The message shows where it was imported from.

#### 3. The extraction cannot find or open a model file

`scripts/extract_minutiae.py` needs `CoarseNet.h5` and `FineNet.h5`; check `ls -l $MODELS` (Step 4). The
variable `MODELS` is empty in a new shell.

#### 4. "... files that are needed are missing"

The files below `--dataset_dir` (or `--minutiae_dir`) are not what `--num_subjects` and `--num_impressions` ask
for. The message names the first missing files and what was found. See Step 1.

#### 5. "... holds a training with other settings"

`--output_dir` already holds a training, and the command differs from it. See
[Step 7](#step-7-continue-a-training-or-start-again).

#### 6. "Resolution unknown"

A file states no resolution and no option gives one (Step 2). In the extraction the file is written to the
error log and the run goes on. In the training and in the range test the run ends in its first step with a
Python error message that names the file; the training has then written `training_settings.json`,
`data_preparation.json` and `log.json` but no `model.pyt`. Give the options of Step 2 and start again.

#### 7. The minutiae do not fit the image

The size in the header of a `.mnt` file differs from the image that was loaded. The extraction and the training
were given different resolution options, or the image file was replaced after the extraction.

#### 8. `RuntimeError: CUDA out of memory`

- Lower `--batch_size` (and repeat the range test with it).
- Use a smaller `--embedding_dim`.
- With very many subjects the classifier itself is the problem, see
  [large-datasets.md](large-datasets.md#above-about-one-million-subjects).

#### 9. The loss does not fall, or the training stops with "diverged"

See [What to watch in the first thousand steps](tuning.md#what-to-watch-in-the-first-thousand-steps).

#### 10. The training stops with `AlignmentCollapsedError`

The localization network moved the prints out of the frame. See
[What to watch](tuning.md#what-to-watch) under "Learned alignment".

#### 11. The validation error rate stays high

No error rate was measured for these documents on a dataset of useful size, so there is no reference value.
Things to check, in this order: the starting loss and its parts
([tuning.md](tuning.md#what-to-watch-in-the-first-thousand-steps)), the resolution sources printed at the
start, the polarity warnings of the extraction, and whether the training accuracy rises at all.

---

## Quick reference

```bash
# --- in the extraction environment (Step 4, "Before you run it") ---

# File list for the extraction in shards
(cd data/fingerprints/my-dataset && find . -name '*.png' | sort > ../my-dataset-files.txt)

# Extract minutiae
python scripts/extract_minutiae.py --input_dir data/fingerprints/my-dataset --output_dir data/poses/my-dataset --coarse_net $MODELS/CoarseNet.h5 --fine_net $MODELS/FineNet.h5 --ppi_manifest data/fingerprints/my-dataset-resolutions.csv --skip_existing

# --- in the training environment (source venv/bin/activate) ---

# Learning rate range test
python scripts/find_learning_rate.py --dataset_dir data/fingerprints/my-dataset --minutiae_dir data/poses/my-dataset --num_subjects 50000 --num_impressions 10 --ppi_manifest data/fingerprints/my-dataset-resolutions.csv --output_dir models/lr_range_test

# Train (the same command again continues the training)
python scripts/train_500k.py --dataset_dir data/fingerprints/my-dataset --minutiae_dir data/poses/my-dataset --num_subjects 50000 --num_impressions 10 --ppi_manifest data/fingerprints/my-dataset-resolutions.csv --learning_rate 0.0001 --lr_schedule cosine --num_epochs 10 --output_dir models/my-model-v1

# All options of a script
python scripts/train_500k.py --help

# Run the tests (needs pytest, see installation.md)
python -m pytest -q
```

Where things are:

- Settings and their defaults: [flx/setup/config.py](../flx/setup/config.py)
- Training script: [scripts/train_500k.py](../scripts/train_500k.py); its docstring describes every option.
  Training loop: [flx/models/model_training.py](../flx/models/model_training.py)
- Model: [flx/models/deep_print_arch.py](../flx/models/deep_print_arch.py)
- Image preparation (500 ppi, ridge polarity): `flx/preparation/sensor_normalization.py`

---

## Reference: the files that the scripts write

### The `.mnt` file

One text file per image, written by `scripts/extract_minutiae.py`:

```
<number of minutiae>
<image width> <image height>
<x> <y> <angle> <class> <direction_confidence>
...
# <how the image was prepared>
```

- Width, height, `x` and `y` are in pixels of the image at 500 ppi. `x` is the column, `y` the row, counted
  from the top.
- `angle` is the direction of the minutia in radians, from 0 to 2 pi. The direction vector is
  `(cos(angle), sin(angle))` with y pointing down. For a ridge ending it points back along the ridge, for a
  bifurcation into the valley between the two branches (the convention of ISO 19794-2). The minutia maps use
  the angle as it is.
- `class` is the minutia type from ClassifyNet: 0 ending, 1 bifurcation, 2 fragment, 3 enclosure, 4 crossbar,
  5 other. It is -1 when `--classify_net` was not given. DeepPrint does not read it.
- `direction_confidence` (0 to 1) rates the direction only. Do not remove minutiae with a low value when
  building minutia maps: the minutia itself is still there.
- **The extractor's `score` is not in the file.** The extractor also reports how sure it is that a point is a
  minutia. The script keeps every minutia that the extractor returns with its default threshold and does not
  write the score, so the minutiae cannot be filtered by score afterwards. The fifth column is not the score.
- The last line starts with `#` and records how the image was prepared, for example
  `# source 826x1274 at 1000 ppi (manifest), ridges dark (argument), prepared 413x637`. The loaders do not read
  it.
- A file is written under a temporary name and renamed when it is complete, so a file with the ending `.mnt`
  is always a finished one.

The training reads only `x`, `y` and `angle` (`FingerFlowMinutiaMapLoader` in
`flx/data/fingerflow_minutia_loader.py`; the class is named after the `fingerflow` package of the minutiae
extractor).

### In the output directory of the extraction

| File | Content |
|------|---------|
| `<same path as the image>.mnt` | The minutiae of one image |
| `extraction_settings.json` | The settings of the extraction (resolution options, checksum of the manifest, ridge option, `--image_dpi`, whether ClassifyNet was used). A run with other settings is refused |
| `extraction_errors.txt`, `extraction_errors.shard-00000-of-00064.txt` | The files that failed in the last run of this shard: path, tab, reason |
| `extraction_warnings.txt`, `extraction_warnings.shard-...txt` | Images whose ridge polarity looks different from what was declared. No file if there were none |

### In the output directory of the training

See the table "What the script writes" in [Step 6](#step-6-run-the-training), and Step 8 for the fields of
`log.json`.

### In the output directory of the range test

`lr_range_test.csv` (one row per step: `step`, `learning_rate`, `loss`, `smoothed_loss`, `gradient_norm`) and
`lr_range_test.png`.
