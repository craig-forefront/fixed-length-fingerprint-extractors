# Large datasets and several GPUs

This document is for datasets of hundreds of thousands of images and more, up to collections of 100 million
images. It is an addition to [training-guide.md](training-guide.md), which describes the route (file layout,
minutiae extraction, learning rate, training, using the model). Follow that guide first on a small part of
your data.

**What this document rests on.** No training was run on a GPU or on a large dataset for these documents. The
machine they were written on has 4 CPU cores and no GPU; the largest real training set was 440 images. Each
figure below is marked as *measured* (on that machine, with other jobs running, so times are on the high
side), *calculated* (arithmetic on measured numbers) or *read* (from the code, not run). There are no figures
for GPU speed, training time or accuracy, because none were measured.

## Contents

1. [Before a long run](#before-a-long-run)
2. [Several GPUs](#several-gpus)
3. [Settings that matter at scale](#settings-that-matter-at-scale)
4. [What limits the speed](#what-limits-the-speed)
5. [Storage](#storage)
6. [Above about one million subjects](#above-about-one-million-subjects)

---

## Before a long run

1. **Run the whole route on a subset** with prints from all sensors: extraction, range test, two epochs of
   training, embeddings (see [Before you start](training-guide.md#before-you-start) in the training guide).
   Mistakes in the layout, the resolutions or the polarity show there.
2. **Read the first lines of the training output**: the variant, the number of training and validation subjects,
   and the block "Resolution of ... training images".
3. **Find the learning rate for the batch size and number of GPUs of the real run**, see
   [Choosing the learning rate](tuning.md#choosing-the-learning-rate). Do not copy a rate from
   a small dataset or another batch size.
4. **Time 200 steps** and compute the time of an epoch from it before you commit the hardware.
5. **Decide the number of epochs once**, in particular with `--lr_schedule cosine`.

---

## Several GPUs

`scripts/train_500k.py` needs no change for several GPUs. Start it with `torchrun`, a program that comes with
PyTorch and starts one process per GPU, and give it the same options as in
[Step 6 of the training guide](training-guide.md#step-6-run-the-training):

```bash
# 4 GPUs of one machine
torchrun --nproc_per_node=4 scripts/train_500k.py \
  --dataset_dir data/fingerprints/my-dataset \
  --minutiae_dir data/poses/my-dataset \
  --num_subjects 50000 --num_impressions 10 \
  --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
  --learning_rate 0.0001 --lr_schedule cosine \
  --num_epochs 10 \
  --output_dir models/my-model-v1
```

For several machines, `torchrun` takes `--nnodes`, `--node_rank`, `--master_addr` and `--master_port` (see the
PyTorch documentation of `torchrun`); the options after the script name stay the same on every machine.

**None of this was run: the machine these documents were written on has no GPU.** The same command with
`python` in place of `torchrun --nproc_per_node=4` was run on 12 images on the CPU. The multi-GPU path exists
in the code (`setup_distributed` and `wrap_model_ddp` in `flx/models/torch_helpers.py`, `train_model` in
`flx/models/model_training.py`) and is read as follows:

- Every GPU processes its own batch of `--batch_size` images, so one training step sees `--batch_size` times the
  number of GPUs images. The learning rate must be chosen for that number, see
  [How the rate changes with the batch size and the number of GPUs](tuning.md#how-the-rate-changes-with-the-batch-size-and-the-number-of-gpus).
- Only the first process validates and writes `model.pyt`, `best_model.pyt` and the log.
- The centers of the center loss are kept per process and are not synchronised; the copy of the first process is
  saved.
- Every process scans the dataset directory and builds the identifier lists itself, see
  [Above about one million subjects](#above-about-one-million-subjects).
- The same command again continues the training; a new `--output_dir` starts a new one
  ([Step 7](training-guide.md#step-7-continue-a-training-or-start-again)).

Time one epoch on a subset on your hardware before you plan a long run.

---

## Settings that matter at scale

| Setting | Where | Notes |
|---------|-------|-------|
| Images per step and GPU | `--batch_size` (default 64) | One step sees this many images times the number of GPUs. Lower it when the GPU runs out of memory. The range test must use the same value |
| Validation subjects | `--max_validation_subjects` (default 5,000) | All their impressions are prepared and embedded after every epoch. See the validation part of [Above about one million subjects](#above-about-one-million-subjects) |
| Learning rate, warm-up, schedule, gradient limit | `--learning_rate`, `--warmup_steps`, `--lr_schedule`, `--gradient_clip_norm` | See [Choosing the learning rate](tuning.md#choosing-the-learning-rate) |
| Epochs and early stopping | `--num_epochs` (default 10), `--patience` (default 3) | How many epochs a large set needs was not measured |
| Embedding size | `--embedding_dim` (default 256 **per branch**, 512 per print with minutiae) | The memory of the classifier grows with it |
| Learned alignment | `--localization on` or `off` | Whether it helps must be measured on your data: [Learned alignment](tuning.md#learned-alignment) |
| Data-loader worker processes (the processes that read and prepare the images) | `get_dataloader_args` in `flx/models/torch_helpers.py`: 16 per GPU process with `prefetch_factor=4` (8 and 2 if `HIGH_VRAM_GPU` is set to `False`), 4 without a GPU | Not an option of the script. Edit the function if the CPU cores do not match |
| Batch size of validation and embedding extraction | twice `--batch_size` (`get_dataloader_args`) | Not an option of the script |
| Mixed precision (computing with 16-bit numbers on the GPU) | on whenever a GPU is used (`use_amp` of `train_model` and of `fit`) | Not an option of the script; `use_amp=False` in your own call of `fit` switches it off. Not run here |
| Checkpoints | `model.pyt` after every epoch, `best_model.pyt` for the best epoch | No checkpoint inside an epoch: a crash loses up to one epoch |

There is no gradient accumulation and no checkpoint rotation in the code: `model.pyt` is overwritten after every
epoch and `best_model.pyt` holds the best one.

---

## What limits the speed

Measured on one CPU thread (60 rolled prints of 800x750 pixels, 33 slap fingers of about 285x440; other jobs were
running, so the times are on the high side):

| Stage of the image preparation in the data-loader workers | Rolled print | Slap finger |
|---|---|---|
| Read the file | 3 ms | 1 ms |
| Random pose | 11 ms | 2 to 3 ms |
| Contrast and gain | 2.5 ms | 0.5 ms |
| Binarisation | 930 to 980 ms | 144 to 158 ms |
| Resize to the model input | 2 to 3 ms | 1 ms |
| Total | about 1 second | 0.15 to 0.17 seconds |

So one CPU core prepares about 1 rolled print or 6 to 7 slap fingers per second, and 16 worker processes give a
GPU about 16 rolled prints per second. If your GPU can train faster than that (not measured here), the CPU cores
set the speed of the training, and the number of cores per GPU matters more than the GPU model. The same holds
for validation. Check this on your hardware: if the GPUs are mostly idle during training, the image preparation is
the limit.

The minutiae extraction is a separate, slow job; see
[A very large number of files](training-guide.md#a-very-large-number-of-files) in Step 4 of the training guide
and the times in [measurements.md](measurements.md#minutiae-extraction-time-and-settings).

---

## Storage

Measured on this repository's test data, per item:

| Item | Size |
|------|------|
| Rolled print, 800x750 pixels, grey PNG | about 200 kB |
| Slap finger, about 285x440 pixels, grey PNG | about 87 kB |
| `.mnt` file | about 3 kB (mean of 259 files) |
| `model.pyt` and `best_model.pyt`, each | 0.8 GB plus 8.2 kB per training subject (two branches of 256 values) |
| Embeddings | 4 bytes per value: 2 kB per print for 512 values |

While `model.pyt` or `best_model.pyt` is written, a third file of the same size exists for a moment
(`model.pyt.tmp` or `best_model.pyt.tmp`).

Images and minutiae are one file each, so a dataset needs two inodes (file entries of the file system) per
image; check `df -i` on the file system before you copy tens of millions of files. At 200 kB per rolled print,
1 million images are about 200 GB and 100 million about 20 TB (calculated).

---

## Above about one million subjects

The scripts were written for hundreds of thousands of images. This section says what is known about larger
datasets, up to the 100 million images and more of a national-scale collection. **Nothing here was run at that
size.** Each figure is marked as *measured*, *calculated* or *read*, as explained at the top of this document.

A "subject" is one finger. 100 million images are about 10 million subjects with 10 images per finger, or about
50 million with 2.

**What grows with the number of training subjects**

| What | Size | Source |
|------|------|--------|
| Classifier and center loss of a variant with two branches of 256 values | 514 trainable parameters and 512 center values per subject | measured (models with 1,000 and 2,000 subjects) |
| The same in `model.pyt` | 4,108 bytes per subject and branch, so 8.2 kB per subject | measured (10,000, 20,000 and 100,000 subjects) |
| `model.pyt` without any subject | 0.8 GB (the 67.7 million other parameters with the optimizer state) | measured (814 MB for 6 subjects) |
| GPU memory of classifier, gradient, the two running averages of Adam, and centers | about 10 kB per subject, in every GPU process | calculated |
| ... at 1 million subjects | about 10 GB per GPU; `model.pyt` about 9 GB, and as much again for `best_model.pyt` | calculated |
| ... at 10 million subjects | about 100 GB per GPU, more than an 80 GB GPU has; `model.pyt` about 83 GB | calculated |
| ... at 50 million subjects | about 500 GB per GPU | calculated |

This is where "about one million" comes from: up to there the classifier fits on a large GPU next to the rest of
the model, by calculation. Above it, it does not. **The code has no way around this today**: the classifier is one
full layer per branch with one row per subject, and the center loss keeps one center per subject. There is no
sampled or partial classifier, no margin-based loss without centers, and no way to split the classifier over the
GPUs. `scripts/train_500k.py` prints a warning with the calculated memory above one million training subjects and
then tries anyway.

**What grows with the number of images**

| What | Size | Source |
|------|------|--------|
| Start of `scripts/train_500k.py` (directory scans, identifier lists, datasets) | 22 seconds and 1.6 GB peak for 1 million images and 1 million `.mnt` files on a warm local disk | measured (one run, empty files, an earlier version that scanned the image directory twice) |
| The same at 100 million images | about 37 minutes and 110 GB, in every GPU process | calculated (linear) |
| The identifier lists alone | 1.0 GB and 5.8 seconds for 3.95 million identifiers | measured |
| A data-loader worker process that has touched every identifier | 95 MB per million images | measured |
| Resolution manifest with one row per file | 186 bytes per row in every process | measured (1 million rows) |
| Image preparation in the data-loader workers | 1.0 second per rolled 800x750 print per CPU core, 98 % of it the binarisation; 0.15 to 0.17 seconds per slap finger of about 285x440 | measured (60 rolled prints, 33 slap fingers) |
| One epoch of 100 million rolled prints | about 28,000 CPU-core-hours for the image preparation alone, whatever the GPUs can do | calculated |
| Minutiae extraction | 60 seconds per print on one CPU thread (439 prints, mostly rolled); about 190 CPU-core-years for 100 million | measured, total calculated |

**What the code cannot do today** (read from the code)

- **Sets that are not a complete grid.** Every subject needs the same number of impressions, numbered without gaps
  ([Step 1](training-guide.md#step-1-lay-out-the-files)). Fingers with different numbers of images cannot be
  expressed. A set in which every finger has one image can be trained, but only without validation
  (`--num_impressions 1 --validation_split 0`).
- **Lists that do not fit in memory.** Identifiers are Python objects, held in every GPU process and compared as
  sets when the datasets are built. Every process walks the image directory once and the minutiae directory
  twice. There is no manifest of the dataset on disk and no shard format; every sample is two small files.
- **Checkpoints inside an epoch.** `model.pyt` is written once per epoch (under a temporary name and then
  renamed, so a crash during the write keeps the file of the epoch before). A crash loses up to one epoch, which
  is days at this size. The step inside the epoch and the state of mixed precision are not saved.
- **A classifier or centers that are shared between GPUs.** See above; the centers are per process.
- **Balanced sampling.** Images are drawn uniformly, so fingers with many images would dominate if the grid
  did not force equal numbers.

**Validation**

- The held-out subjects are fingers, not people. Other fingers of a validation person can be in the training
  set. If that matters to you, number the subjects so that you can hold out people yourself.
- All impressions of the held-out subjects are prepared and embedded after every epoch, and the list of pairs is
  built in a time that grows with the square of the number of subjects: 0.8 seconds for 2,500 subjects, 1.8 for
  5,000 and 4.3 for 10,000 (measured), so about 4 hours for 1 million (calculated).
- Therefore `--max_validation_subjects` defaults to **5,000**. With 10 impressions each that is 275,000 pairs,
  scored in under a second (measured), and about 50,000 images to prepare per validation, roughly 50 minutes on
  16 CPU cores for rolled prints (calculated). `--max_validation_subjects 0` removes the limit; the script warns
  above 5,000.
- Each subject gets 10 non-mated pairs. That is enough to choose the best epoch, not to measure the low false
  match rates that matter in a gallery of millions.

**What to do with a dataset of this size, with the code as it is**

1. Train on a subset first: up to about one million subjects, chosen at random across all sensors, with a
   complete grid of impressions. Everything in the training guide applies to it.
2. On the first GPU run, measure what this section could only calculate: the memory per GPU, the time of one
   step, and how many CPU cores one GPU needs to stay busy. Time 200 steps before you start an epoch.
3. Ask whether the collection already has minutiae from its capture or matching system. That would replace the
   largest single cost. (`flx/data/iso_encoder_decoder` reads ISO 19794-2 templates; converting them to `.mnt`
   files is not written.)
4. Training on all subjects of a collection with more than a few million fingers needs changes to the model and
   the data pipeline that are not in this repository: a classifier that scores a sample of the subjects in each
   step and is split over the GPUs (with a loss that needs no centers), a manifest of the dataset on disk with
   shards instead of two files per sample, binarising the images once instead of in every epoch, and
   checkpoints inside an epoch. None of them has been built or measured here, and what they do to accuracy is
   not known.
