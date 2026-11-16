# Large-Scale Training Guide

This guide is for datasets of hundreds of thousands of images and more. It is an addition to
[TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md), which describes the route (file layout, minutiae
extraction, learning rate, training, using the model) and which you should follow first on a small part of your
data.

**What this guide rests on.** No training was run on a GPU or on a large dataset for this repository's guides.
The machine they were written on has 4 CPU cores and no GPU; the largest real training set was 440 images. An
earlier version of this guide gave hardware lists, images per second, training times and speed-up factors for
500,000 to 1.5 million images. None of them was measured, the measurements that were made contradict some of
them, and they were removed. It also described edits to the training code (gradient accumulation, checkpoint
rotation, a hand-written training script) for things that either already exist as options or do not exist at
all; those were removed as well.

## Contents

1. [The command](#the-command)
2. [Before a long run](#before-a-long-run)
3. [Settings that matter at scale](#settings-that-matter-at-scale)
4. [What limits the speed](#what-limits-the-speed)
5. [Storage](#storage)
6. [Above about one million subjects](#above-about-one-million-subjects)

---

## The command

The training command is the one of [Step 6 in TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md#step-6-run-the-training),
started with `torchrun` for several GPUs:

```bash
torchrun --nproc_per_node=4 scripts/train_500k.py \
  --dataset_dir data/fingerprints/my-dataset \
  --minutiae_dir data/poses/my-dataset \
  --num_subjects 50000 --num_impressions 10 \
  --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
  --learning_rate 0.0001 --lr_schedule cosine \
  --num_epochs 10 \
  --output_dir models/my-model-v1
```

- `--num_subjects`, `--num_impressions` and `--minutiae_dir` are needed: without the first two the script expects
  50,000 subjects with 10 impressions and stops with the list of missing files if the directory holds something
  else; without `--minutiae_dir` it trains the texture-only variant.
- The same command with `python` in place of `torchrun --nproc_per_node=4` was run on 12 images on a CPU. The
  `torchrun` form was not run (no GPU).
- The same command again continues the training; a new `--output_dir` starts a new one. See
  [Step 7](TRAINING_PIPELINE_GUIDE.md#step-7-continue-a-training-or-start-again).

## Before a long run

1. **Run the whole route on a subset** with prints from all sensors: extraction, range test, two epochs of
   training, embeddings. Mistakes in the layout, the resolutions or the polarity show there, before the long run.
   Choose the size by the extraction, which is the slow part: it took 19 to 60 seconds per print in one CPU
   process here (no GPU speed was measured), so a few hundred prints take hours in one process and a few thousand
   subjects with 10 prints each take days unless the extraction runs in many shards. A few hundred prints from
   every sensor are enough for the mistakes named above; they are not enough to judge the accuracy.
2. **Read the first lines of the training output**: the variant, the number of training and validation subjects,
   and the block "Resolution of ... training images".
3. **Find the learning rate for the batch size and number of GPUs of the real run**, see
   [Choosing the learning rate](TRAINING_PIPELINE_GUIDE.md#choosing-the-learning-rate). Do not copy a rate from
   a small dataset or another batch size.
4. **Time 200 steps** and compute the time of an epoch from it before you commit the hardware.
5. **Decide the number of epochs once**, in particular with `--lr_schedule cosine`.

## Settings that matter at scale

| Setting | Where | Notes |
|---------|-------|-------|
| Images per step and GPU | `--batch_size` (default 64) | One step sees this many images times the number of GPUs. Lower it when the GPU runs out of memory. The range test must use the same value |
| Validation subjects | `--max_validation_subjects` (default 5,000) | All their impressions are prepared and embedded after every epoch. See the validation part of [Above about one million subjects](TRAINING_PIPELINE_GUIDE.md#above-about-one-million-subjects) |
| Learning rate, warm-up, schedule, gradient limit | `--learning_rate`, `--warmup_steps`, `--lr_schedule`, `--gradient_clip_norm` | See [Choosing the learning rate](TRAINING_PIPELINE_GUIDE.md#choosing-the-learning-rate) |
| Epochs and early stopping | `--num_epochs` (default 10), `--patience` (default 3) | How many epochs a large set needs was not measured |
| Embedding size | `--embedding_dim` (default 256 **per branch**, 512 per print with minutiae) | The memory of the classifier grows with it |
| Learned alignment | `--localization on` or `off` | Whether it helps must be measured on your data: [Learned alignment](TRAINING_PIPELINE_GUIDE.md#learned-alignment) |
| Data-loader worker processes | `get_dataloader_args` in `flx/models/torch_helpers.py`: 16 per GPU process (8 if `HIGH_VRAM_GPU` is set to `False`), 4 without a GPU | Not an option of the script. Edit the function if the CPU cores do not match |
| Mixed precision | on whenever a GPU is used (`use_amp` of `fit`) | Not an option of the script. Not run here |

There is no gradient accumulation and no checkpoint rotation in the code: `model.pyt` is overwritten after every
epoch and `best_model.pyt` holds the best one.

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
[Running the extraction over a very large number of files](TRAINING_PIPELINE_GUIDE.md#running-the-extraction-over-a-very-large-number-of-files).

## Storage

Measured on this repository's test data, per item:

| Item | Size |
|------|------|
| Rolled print, 800x750 pixels, grey PNG | about 200 kB |
| Slap finger, about 285x440 pixels, grey PNG | about 87 kB |
| `.mnt` file | about 3 kB (mean of 259 files) |
| `model.pyt` and `best_model.pyt`, each | 0.8 GB plus 8.2 kB per training subject (two branches of 256 values) |
| Embeddings | 4 bytes per value: 2 kB per print for 512 values |

Images and minutiae are one file each, so a dataset needs two inodes per image; check `df -i` on the file system
before you copy tens of millions of files.

## Above about one million subjects

Read [the section of this name in TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md#above-about-one-million-subjects)
before you plan a training on a collection of that size. In short: the classifier and the center loss need about
10 kB of GPU memory per training subject in every GPU process (calculated), the identifier lists are held in
memory by every process, every subject needs the same number of impressions, and checkpoints exist only per
epoch. The code has no way around these today.
