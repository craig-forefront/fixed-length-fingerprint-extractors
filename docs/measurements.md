# Measurements and their limits

This document holds the evidence behind the other documents: every table of runs, what each number rests on,
and what was not measured. The how-to documents quote only the conclusions and link here.

**What all of it has in common**

- One machine with 4 CPU cores and **no GPU**, with other jobs running at the same time, so all times are
  rough and on the high side.
- Real fingerprints from NIST Special Database 302 (rolled and slap prints from five capture devices, with
  minutiae marked by fingerprint examiners). The largest set used was **440 images** of 220 fingers.
- The synthetic SFinGe images that come with this repository (`notebooks/example-dataset`, `tests/data`) were
  used only to run commands, never as evidence about accuracy.

**What was not measured, anywhere in these documents**

- **Accuracy.** No equal error rate was measured on a dataset of useful size, with any setting. The error
  rates that appear in the runs listed at the end come from models after a handful of training steps and mean
  nothing.
- **Anything on a GPU**: speed, memory, mixed precision, a batch of 64 images, several GPUs (`torchrun`).
- **A long training.** The longest run was 300 steps of 8 images.
- **A large dataset.** Figures for more than 440 training images are calculations, timings of file lists, or
  timings of the start of a training on one million synthetic rows with a model that does nothing (the sample
  table), and are marked as such in [large-datasets.md](large-datasets.md) and
  [sample-table.md](sample-table.md#size-time-and-memory).
- **Latent prints, and tenprint decisions.**

## Contents

- [Images from different sensors](#images-from-different-sensors)
- [Minutiae extraction: time and settings](#minutiae-extraction-time-and-settings)
- [Learning rate](#learning-rate)
- [Loss weights](#loss-weights)
- [Learned alignment](#learned-alignment)
- [Large datasets](#large-datasets)
- [What was run for these documents](#what-was-run-for-these-documents)
- [Changes against earlier versions, and what to do with old files](#changes-against-earlier-versions-and-what-to-do-with-old-files)

---

## Images from different sensors

Belongs to [Step 2 of the training guide](training-guide.md#step-2-say-what-resolution-and-ridge-polarity-each-file-has).

**Resolution and polarity.** 8 NIST SD 302 prints (5 rolled, 3 slap) with 688 minutiae marked by examiners.
A minutia counts as found when the extractor reports one within 10 pixels at 500 ppi. The extractor ran with
its default settings (`crop_to_print` and `reweight_score` off).

| Image file given to `scripts/extract_minutiae.py` | Examiner minutiae found | Reported minutiae that are examiner minutiae | Same as from the 500 ppi file |
|---|---|---|---|
| 500 ppi file | 42.7 % | 79.7 % | |
| 1000 ppi file of the same print, resolution from the file metadata | 42.6 % | 79.0 % | 98.6 % |
| 385 ppi file of the same print (made from the 1000 ppi file) | 44.8 % | 78.4 % | 88.1 % |
| 1000 ppi negative of the same print, declared as light ridges | 42.4 % | 78.9 % | 98.4 % |
| 1000 ppi file declared as 500 ppi, which is what happened before the resampling existed (7 prints) | 5.3 % | 27.2 % | 5.6 % |

Eight prints show that the preparation works; they are not an accuracy figure for your data. The 385 ppi row comes
from images that were first shrunk and then enlarged again, which loses detail; a real 385 ppi sensor was not
available.

**A wrong resolution stamp in a file.** One 300x400 file, extracted with `--assume_ppi 500`: with a stamp of
300 ppi in its metadata it was resampled to 500x667 pixels, with a stamp of 600 to 250x333, and with a stamp
of 96 it was left unchanged (96 is outside the range of 200 to 2100 ppi that the scripts accept from
metadata).

**The estimate of the ridge polarity (`--ridges auto`).** 1708 NIST SD 302 exemplars from 5 devices at 500 ppi,
each as it is and as its negative: the estimate was right for all 3416 images. On the centre third of the same
images, which shows little or no background, it gave no decision for 549, was right for 1148 and wrong for 11.

**`--canvas_size`.** Two real prints were looked at with a canvas of 512 pixels: the slap print (603x842 pixels
at 1000 ppi) was complete, and the rolled print (1600x1500 pixels at 1000 ppi), which lies in the lower left of
its image, lost a large part. Whether a canvas gives better matching was not measured.

---

## Minutiae extraction: time and settings

Belongs to [Step 4 of the training guide](training-guide.md#step-4-extract-the-minutiae).

**Time per print**, all on the CPU, model loading included:

| Prints | How it ran | ClassifyNet | Seconds per print |
|--------|-----------|-------------|-------------------|
| 8 real prints at 500 ppi, 369 minutiae | one process on the 4 cores | no | 19 (153 s in all) |
| the same 8 prints | one process on the 4 cores | yes | 33 (263 s in all) |
| 439 real prints, mostly rolled | one CPU thread | not recorded | 60 |
| 12 images made from the SFinGe example images (292x392 pixels at 500 ppi; 8 of them as such, 4 enlarged to twice the size and declared as 1000 ppi) | two runs of 6 images, one process on the 4 cores each | no | 14 (83 s and 81 s) |

The first three rows are from earlier work on this repository, the last one from the runs for this version of
the documents. The times differ because the prints differ in size and number of minutiae (FineNet is called
once per minutia) and because one thread is slower than four cores. No GPU time exists. From the 60 seconds
per print on one thread, 100 million prints would take about 190 CPU-core-years (calculated).

**With and without ClassifyNet.** The two runs on the 8 prints gave the same 369 positions, angles and
direction confidences.

**Worker processes.** In one of the runs for the sensor table above a worker process died (the machine ran out
of memory); the files it was working on were logged as failed and the run went on.

**`--image_dpi` when the declared resolution is not the true one.** `--image_dpi` is only used to find the
direction of the minutiae and refers to the image after the resampling, so it should stay at 500. If you give
the script a resolution that is not the true one (for example `--assume_ppi 500` for the 292x392 SFinGe
example images of this repository, whose ridge period of about 8 pixels corresponds to about 420 ppi), the
ridge period of the resampled image is off, and `--image_dpi` can correct the direction step for it: roughly
`500 * period_px / 9.5`. The setting is tolerant: values up to 50 below or above the suitable one gave the same
accuracy of the directions on those images and on 500 ppi prints.

**What the extractor finds** (how many of the examiner minutiae, the direction, latents, the options
`crop_to_print` and `reweight_score`) is measured in the documentation of the
[minutiae_extractor](https://github.com/craig-forefront/minutiae_extractor) repository, not here.

---

## Learning rate

Belongs to [Choosing the learning rate](tuning.md#choosing-the-learning-rate) in tuning.md.

### The default learning rate: what was measured and what was not

In a fresh `DeepPrint_LocTexMinu` model, 93 % of the 68 million parameters are smaller than 0.025 in size. The
Adam optimizer changes every parameter by about the learning rate in its first step, so a rate of 0.025 (a
former default of this code, see the last chapter) changed most parameters by more than their own size.

The default learning rate of 0.0001 comes from runs on the only real data
that was available: 440 images of 220 fingers (two
impressions each, rolled and slap prints from five devices of NIST Special
Database 302, with minutiae marked by examiners), on a CPU, with 8 images per
step. Each line is one run.

| Run | `DeepPrint_TexMinu` | `DeepPrint_LocTexMinu` |
|-----|--------------------|------------------------|
| Range test, 100 steps from 0.000001 to 0.1: smoothed loss above 15 (start: 14.8) from | 0.0031 | 0.0043 |
| Range test: lowest smoothed loss at | 0.0015 | 0.0022 |
| Range test: suggested rate (a tenth of that) | 0.00015 | 0.00022 |
| Training with the former default 0.025, no warm-up | stopped in step 2 (loss 14.7, then 5,119) | stopped in step 2 (loss 14.9, then 5,717) |
| Training with 0.01, no warm-up | not run | stopped in step 3 (loss 14.9, 43.8, 690) |
| Training with the new defaults (0.0001, warm-up, constant), 110 steps | no divergence; highest batch loss 14.9; epoch loss 14.70, then 14.44 | no divergence; highest batch loss 14.9; epoch loss 14.70, then 14.41 |

In the last line the warm-up took 11 steps, a tenth of the training, not the
default 500.

These runs were made before the weights of the parts of the loss changed (see
[the last chapter](#changes-against-earlier-versions-and-what-to-do-with-old-files));
with the current weights the loss is larger
(it starts near 145 instead of 14.8 for these images) and its parts are
balanced differently. One run was repeated with the current loss:
`DeepPrint_LocTexMinu` with all defaults, 110 steps of 8 images, warm-up 11
steps. It did not diverge: the highest batch loss was the first one (145.1),
and the epoch loss was 77.8, then 62.1. The recorded minutia map loss fell
from 35.9 to 22.3 and the center loss of each embedding from 15.6 to 14.4,
while the cross entropy of each classifier stayed at 5.40.

The range test of `DeepPrint_TexMinu` was also repeated with the current
loss (one run, same settings as in the table). The smoothed loss started at
133 and fell from the first step (100 at the rate 0.000016, 86 at 0.0001). It
was lowest (64) at 0.00135, rose from 0.0017 and the test ended after 69
steps at 0.0027. The steepest fall was found at 0.0000036, which is the
effect described under "Variants with minutiae look different" in
[tuning.md](tuning.md#how-to-read-the-plot); the
suggestion (a tenth of the rate with the lowest loss) is 0.000135. The
default 0.0001 is 13 times below the rate with the lowest loss. The range test of
`DeepPrint_LocTexMinu` was not repeated, so its line in the table belongs to
the former loss.

What these runs show: 0.0001 does not diverge in the first 110 steps for
either variant, and it is 30 to 43 times below the rate at which the
loss of the range test started to rise. The default is a round number at the
low end of the suggestions in the table of the next section (0.000095 to
0.00027).

What they do not show:

- **Whether 0.0001 is the best rate, or even a good one, for your data.** In
  110 steps on 440 images the model learned almost nothing about the
  subjects: the loss of each classifier stayed between 5.39 and 5.40 and the share
  of correctly named training subjects stayed near chance. The fall of the
  loss came from the center loss and the map loss. These runs are a check
  that the training does not break, not a measurement of how well it learns.
- **Anything about a batch of 64 images or about several GPUs.** Every run
  used 8 images per step. No run used a GPU, mixed precision, or more than
  one process.
- **Anything about accuracy.** No equal error rate was measured.
- **Whether a long training stays stable.** Some instabilities need
  thousands of steps to show.

The value must be found again on your data and with your batch size: run the
range test, start a training with the suggested rate, and watch the first
thousand steps as described in
[tuning.md](tuning.md#what-to-watch-in-the-first-thousand-steps).

**The gradient norm.** In the range test of `DeepPrint_TexMinu` with the current loss, the norm of the gradient
was between 10,000 and 15,200 at rates up to 0.0001 and fell to a few hundred near 0.001.

**The starting loss.** One batch of 8 real images, 256 values per branch, 220 subjects: the mean squared
distance between embedding and center was 1.95 and 1.98; the minutia map loss without its weight was 1,013
for the first batch of `DeepPrint_TexMinu`; the whole starting loss was 138 for `DeepPrint_TexMinu` and 145
for `DeepPrint_LocTexMinu`. The map part fell to about a quarter within 50 steps.

### How far to trust the suggestion of the range test

The suggestion is a starting point, not a measurement of the best rate. All
range tests of minutiae variants that were kept are in the table. They are short (at most 100 planned steps), use 4 or 8 images
per step and differ in more than the seed, so the spread is an upper
estimate for one setting; no test was run at the script defaults (300 steps
of 64 images), which should be less noisy.

| Test (one run each) | Steps done | Lowest smoothed loss at | Steepest fall at | Suggestion (lowest / 10) | An earlier rule based on the steepest fall gave |
|---|---|---|---|---|---|
| `LocTexMinu`, 220 fingers, batch 4, seed 1 | 67 | 0.00095 | 0.000105 | 0.000095 | 0.0000105 |
| `LocTexMinu`, 220 fingers, batch 4, seed 2 | 73 | 0.0027 | 0.0000102 | 0.00027 | 0.00027 |
| `LocTexMinu`, 60 fingers, batch 4, 30 steps planned | 23 | 0.0013 | 0.0000161 | 0.00013 | 0.0000016 |
| `TexMinu`, 220 fingers, batch 8, current loss weights | 69 | 0.0015 | 0.0000115 | 0.00015 | 0.00015 |
| `TexMinu`, 220 fingers, batch 8, former loss weights | 77 | 0.0015 | 0.0011 | 0.00015 | 0.00011 |
| `LocTexMinu`, 220 fingers, batch 8, former loss weights | 79 | 0.0022 | 0.00067 | 0.00022 | 0.000067 |

- The suggestions are between 0.000095 and 0.00027, a factor of 2.8 apart.
  The first two rows differ only in the seed and are a factor of 2.8 apart.
- The rule that followed the steepest fall gave 0.0000016 to 0.00027 for the
  same six tests, a factor of 170 apart, and a factor of 26 between the two
  seeds. That rule was replaced.
- So: expect the suggestion to move by a factor of about 3 from one seed to
  the next. Run two or three seeds, take the lowest suggestion, and let the
  plot and the first thousand steps of a real training decide. A rate that
  is 3 times too low costs training time; one that is too high can end the
  training.

The run of the range test for this version of the documents (12 images, see
[What was run](#what-was-run-for-these-documents)) only shows that the command works.

---

## Loss weights

Belongs to [Loss weights](tuning.md#loss-weights) in tuning.md, which explains the parts of the loss.

### Why the weight of the minutia map loss is 0.095

The minutia map loss exists to make the layers that the minutia branch shares
with the rest of the model look at minutiae. Two groups of layers are shared:

- the *stem*: the first layers of the model, used by the texture branch and
  the minutia branch;
- the *minutia stem*: the layers after it that are used by the minutia
  embedding and by the map.

How strongly a part of the loss asks a group of layers to change is the length
of its *gradient* on these layers. The *share* of the map loss is the length
of its weighted gradient divided by the sum of that length and the length of
the weighted gradient of all other parts together. A share near 0 means that
the map loss has no say in what these layers learn.

Measured on this machine (no GPU) with `DeepPrint_TexMinu`, 256 dimensions, on
400 real images (200 fingers with two impressions each from NIST Special
Database 302, minutiae marked by examiners), 8 images per step, Adam with
learning rate 0.0001, center weight 8.0. Each column is one run of 40 or 50
steps with that map weight and the same random seed; the gradients were
measured every 10 steps (5 or 6 measurements per run), and the table gives the
lowest and the highest value. The column 0.095 also contains two longer runs
of 150 and 281 steps, 19 measurements in all.

| | Former loss (0.00000305) | 0.01 | **0.095** | 1.0 |
|---|---|---|---|---|
| Share of the map loss on the stem | 0.0006 % to 0.002 % | 0.2 % to 0.6 % | 1.0 % to 5.3 % | 17 % to 26 % |
| Share of the map loss on the minutia stem | 0.07 % to 1.2 % | 24 % to 83 % | 75 % to 99 % | 97 % to 100 % |
| Map gradient on the stem, relative to the gradient of the texture cross entropy | 0.00002 to 0.00008 | 0.06 to 0.18 | 0.25 to 1.5 | 4.8 to 10 |

The column "Former loss" is the loss of earlier versions of this code (see
[the last chapter](#changes-against-earlier-versions-and-what-to-do-with-old-files))
at 8 images per step (center loss 0.125 x sum); that run ended after 38 steps when the
machine ran out of memory, so it has 4 measurements.

What the numbers say:

- **With the former weight the map loss did not reach the shared layers.**
  Its share on the stem was about one part in 100,000.
- **Nearly all of the other gradient on the stem comes from the texture
  branch**, and most of that from the center loss: at the first step the
  lengths were 10,986 (texture center loss with weight 8.0), 364 (texture
  cross entropy), 180 (minutia center loss) and 6 (minutia cross entropy).
  The minutia embedding sends very little gradient back, so the map loss is
  what trains the minutia stem.
- The layers that only draw the map get their gradient from the map loss
  alone, and the optimizer (Adam) adapts its step to the size of the gradient.
  They therefore learn with any weight. The weight decides what the *shared*
  layers learn.

**The target and the choice.** The map loss is a helper task; the cross
entropy is the part that teaches the model to tell fingers apart. The target
was therefore: on the stem, the map loss should pull about as strongly as the
cross entropy of the texture branch (a ratio near 1, between one half and
two), and its share of the whole gradient should stay well below one half, so
that it does not decide alone what the stem learns. The weight 0.095 of the
paper meets this (ratio 0.25 to 1.5, in 16 of 19 measurements between 0.4
and 1.25; share 1.0 % to 5.3 %). With 0.01 the map
loss is a fifth of the cross entropy or less; with 1.0 it is 5 to 10 times
the cross entropy. In the longest run the ratio and the share rose slowly
(ratio 0.7 at the start, 1.5 after 250 steps), because the gradient of the
cross entropy shrank faster than that of the map loss; how they develop in a
full training was not measured. The share of 1.0 % to 5.3 % is small because the center
loss with its weight 8.0 dominates the stem; with a lower center weight the
same map weight has a larger share. On the minutia stem the map loss
dominates with 0.095 (75 % to 99 %); this is accepted, because almost no
other gradient arrives there.

### Does the map branch learn?

Two numbers describe the predicted maps. The *map error* is the minutia map
loss without its weight: the summed squared difference per image. A model
that draws an empty map has an error equal to the summed squares of the
target map, about 300 for these images; a model has to get below that to show
that it places minutiae. The *correlation* between predicted and target map
is 0 for unrelated maps and 1 for identical ones. Both were measured on 16
images of 20 fingers that were not trained on, with the same runs as above:

| After step | Former loss | 0.01 | **0.095** | 1.0 |
|---|---|---|---|---|
| 0 (start) | error 905, correlation -0.02 | the same | the same | the same |
| 10 | 440, 0.08 | 382, 0.08 | 361, 0.11 | 349, 0.11 |
| 20 | 356, 0.11 | 279, 0.15 | 252, 0.18 | 244, 0.18 |
| 30 | 318, 0.13 | 230, 0.20 | 205, 0.24 | 200, 0.25 |

An empty map has the error 190 on these 16 images.

Two more runs with the weight 0.095 went on for longer. The longer one had
281 steps, which is more than five passes over the 400 images. The last two
columns are the mean value that the model draws in the cells where the target
map is above 0.5 (minutiae; the target there is 0.60 on average) and in the
cells where the target map is below 0.01 (no minutia nearby):

| Steps | Map error on the training batches | Error of an empty map on the same batches | Correlation | Drawn on minutiae | Drawn elsewhere |
|---|---|---|---|---|---|
| 0 to 9 | 805 | 310 | 0.00 | 0.03 | 0.029 |
| 50 to 59 | 256 | 288 | 0.31 | 0.06 | 0.009 |
| 100 to 109 | 228 | 293 | 0.40 | 0.07 | 0.009 |
| 150 to 159 | 238 | 311 | 0.43 | 0.08 | 0.009 |
| 200 to 209 | 222 | 298 | 0.43 | 0.09 | 0.009 |
| 250 to 259 | 211 | 292 | 0.45 | 0.09 | 0.008 |

On the 16 images that were not trained on, the error fell from 905 to 154
after 250 steps (empty map: 190), the correlation rose to 0.36, and the model
drew 0.093 on minutiae against 0.007 elsewhere (at the start: 0.020 against
0.034). The run of 150 steps ended at an error of 158 and a correlation of
0.34 on these images.

What this shows: the map branch learns where the minutiae are. Its error
falls below that of an empty map, also for fingers it has not seen, it draws
about 13 times more on minutiae than elsewhere, and the error falls faster
when the map loss reaches the shared layers (0.095 against the former loss in
the table above).

What it does not show: maps with minutiae at full height. After 250 steps the
model drew 0.09 where the target is 0.60 on average, about a sixth of the
height. The model first learned to draw a nearly empty map and then raised it
slowly where the minutiae are. Whether the peaks reach the height of the
target in a full training was not measured.

### What was not measured about the loss weights

- **Accuracy.** No equal error rate was measured with any weight. A few
  hundred steps on 200 fingers show which parts of the loss reach which
  layers and that the map error falls; they cannot show which weights give
  the better model. In all runs the cross entropy stayed at its starting
  value (5.3 for 200 subjects).
- **The center weight.** 8.0 keeps what this code did at 64 images per step.
  The DeepPrint paper states 0.00125 for its center loss (whether the paper
  counts it as a sum or a mean over the batch was not checked), and on the
  stem the gradient of the center loss was 30 times that of the cross entropy
  at the start. Whether a lower value is better was not tested.
- **The variants with localization network**, a batch of 64, a GPU and mixed
  precision. The gradient shares were measured for `DeepPrint_TexMinu` only.
  One training of `DeepPrint_LocTexMinu` with all defaults ran for 110 steps
  without divergence (see
  [The default learning rate](#the-default-learning-rate-what-was-measured-and-what-was-not)).

---

## Learned alignment

Belongs to [Learned alignment](tuning.md#learned-alignment) in tuning.md, which explains the localization
network, its bounds and the check for empty images.

### The resampled target map

For the variants with localization network and minutiae, the loss moves the target minutia map with the
rotation and shift that the network predicted.

The target map is resampled, not drawn again from the transformed minutiae.
Drawing again would be exact, but the data loaders deliver finished maps, and
passing minutia lists of different lengths through the batches would change
every loader. Measured against maps drawn again from the transformed
minutiae, on the examiner minutiae of 200 real prints per row with random
transforms:

| Rotation up to | Shift up to (share of the image side) | Correlation with the map drawn again: mean (lowest) |
|----------------|----------------------------------------|------------------------------------------------------|
| 10 degrees | 0.05 | 0.968 (0.950) |
| 30 degrees | 0.15 | 0.957 (0.926) |
| 60 degrees | 0.25 | 0.947 (0.905) |
| 60 degrees | 0.5 | 0.942 (0.694) |

The unmoved target, by comparison, is unrelated to what the model should
draw once the alignment is more than a few degrees.

### What was measured about the alignment, and what was not

All runs used the real data described under
[The default learning rate](#the-default-learning-rate-what-was-measured-and-what-was-not):
400 images of 200 fingers for training and 40
images of 20 further fingers that the training never saw. They ran through the
real training loop on a CPU with `DeepPrint_LocTexMinu`, 8 images per step,
the default learning rate 0.0001 with warm-up, the default loss weights and
the random rotation (up to 30 degrees) and shift (up to 40 pixels of the
original image) of `scripts/train_500k.py`. Each line is one run; only the
factor of the learning rate of the localization network differs.

| Factor | Steps | Loss: first step, then mean of the last 10 steps | Loss always finite | Largest rotation | Largest shift (share of the side) | Lowest share of the print kept | Mostly empty images | How it ended |
|--------|-------|--------------------------------------------------|--------------------|------------------|-----------------------------------|--------------------------------|---------------------|--------------|
| 0.035 (default) | 300 | 131.7, then 54.2 | yes | 0.0011 degrees | 0.00002 | 100 % | 0 of 2,400 | finished |
| 1 | 50 | 131.7, then 64.8 | yes | 0.05 degrees | 0.0002 | 100 % | 0 of 400 | stopped by hand after 50 recorded steps, to free the processor |
| 5 | 150 | 131.7, then 56.1 | yes | 1.7 degrees | 0.046 | 100 % | 0 of 1,200 | finished |
| 30 | 23 | 131.7, then 72.6 | yes | 55.7 degrees | 0.496 | 0 % | 9 of 184 | stopped by the check in step 23 |

What these runs show:

- **With the default factor the training is stable.** In 300 steps the loss
  stayed finite and fell, the predicted rotation and shift stayed far inside
  the bounds, and no aligned image lost any part of its print. The fall of
  the loss came from the minutia map loss (41.6 in the first 50 steps, 20.7
  in the last 50) and the center loss (15.7 to 12.2 per embedding); the cross
  entropy stayed at 5.30, the value of a model that guesses among 200
  fingers.
- **The bounds hold, and the check catches a collapse that the loss does
  not show.** With the factor 30 the rotation jumped from 2.3 to 18.0 to 55.7
  degrees and the shift to 0.496 of the image side within three steps, both
  still inside the bounds of 60 degrees and 0.5. The check warned in step 22
  (1 of 8 images mostly empty) and stopped the training in step 23 (8 of 8).
  The loss was falling the whole time (56.7 in the last step). Bounds as wide
  as those of the paper do not keep the print in the frame by themselves.
- **The factor matters.** 5 was stable for 150 steps, 30 collapsed after 23.
  Nothing between them was tried.

Whether the alignment brings differently placed copies of the same print
closer together was measured as follows. Each test image was given two random
poses (rotation and shift as in the training), which gives a pair of copies of
the same print. Five points of the print (the centre and four points around
it) were followed through the pose and through the alignment that the trained
localization network proposed for each copy. The number below is the mean
distance between the same point in the two copies, in pixels of the 299 pixel
input image; a working alignment makes it smaller.

| Factor | After steps | Pairs | Distance without alignment | Distance with alignment | Pairs that got closer / further |
|--------|-------------|-------|----------------------------|-------------------------|---------------------------------|
| 0.035 (default) | 300 | 40 (unseen fingers) | 24.466 | 24.466 (change -0.00003, standard deviation 0.0001) | 23 / 17 |
| 0.035 (default) | 300 | 80 (training fingers) | 24.186 | 24.186 (change -0.00003, standard deviation 0.0001) | 51 / 29 |
| 5 | 150 | 40 (unseen fingers) | 24.466 | 24.467 (change +0.002, standard deviation 0.034) | 19 / 21 |
| 5 | 150 | 80 (training fingers) | 24.186 | 24.179 (change -0.006, standard deviation 0.042) | 46 / 34 |

**The alignment did not bring the copies closer together in these runs.**
The copies were 24 pixels apart before and after. With the default factor the
localization network had not moved away from the identity after 300 steps
(largest rotation 0.001 degrees), so the model was in effect the variant
without it. With the factor 5 it rotated every image by about the same 0.5
degrees, whatever the pose of the image, which is not an alignment. A few
hundred steps on 200 fingers are too short for the model to learn to tell
fingers apart (the cross entropy did not move), and the alignment can only
learn from that.

A warning about a simpler measure: the mean pixel difference between the two
copies did fall in every pair (for the factor 5 from 0.155 to 0.137 on the 40
unseen pairs), although the points did not move closer. Resampling smooths
the thin ridges of the binarized image, and smoother images differ less from
any other image. A falling pixel difference is therefore no evidence of
alignment.

What was not measured:

- **Whether the localization network learns a useful alignment at all** with
  this code, with any factor, in a training long enough for the model to
  learn. The paper reports that it does in its setting.
- **Whether it makes matching more accurate.** No equal error rate was
  measured. Only a training on your data can answer it; the recipe is under
  [How to find out whether it helps on your data](tuning.md#how-to-find-out-whether-it-helps-on-your-data).
- **Stability over a long training.** 300 steps show that the default factor
  does not break the start of a training. The check for empty images is
  there for what may happen later.
- **A batch of 64, a GPU, mixed precision, several GPUs,** a scale factor
  above 1.0, and the variants `LocTex` and `LocMinu` in a run of this length
  (`LocTex` is covered by tests and by short runs of the script only).

`scripts/train_500k.py` keeps the localization network on by default,
because the default settings were stable in these runs and a collapse stops
the training instead of passing unnoticed. This is not a statement that it
helps.

In runs with the former learning rate of 0.025 (see the last chapter), before the bounds and the check
existed, the localization network moved the prints out of the frame within four steps, and in one of these
runs the loss fell while every image was empty.

---

## Large datasets

What was measured, calculated and only read from the code about memory, start-up time, image preparation
speed and storage is in the tables of [large-datasets.md](large-datasets.md), with each figure marked. No
training was run on more than 440 images.

**The sample table at one million rows.** Measured for the sample table on the machine of these documents,
while it was otherwise idle: 100,000 synthetic subjects with 10 rows each and 46 minutiae per row, empty image files but for
600 hard links to one example image, the files in the page cache, a model that does nothing in the place of the
real one; the builder ran on 8,000 PNG files, copies of the example images. Each figure is the range over all
runs of its kind; the reading, the starts and the builder were taken by two harnesses that were written
independently of each other (the speed of the CPU varied between series of runs by up to a third). MB = 10^6
bytes. The conditions of each figure are in [Size, time and memory](sample-table.md#size-time-and-memory) of
sample-table.md.

| What | Without a sample table | With a sample table |
|------|------------------------|---------------------|
| The minutiae on the disk | 1,000,000 `.mnt` files, 1,763.8 bytes each on average, 4,096 bytes of the disk each | 4 files, 565.5 bytes per row |
| Reading and checking the table | | 1.4 to 2.2 s, peak of the process 1,022 to 1,031 MB (a process without PyTorch) |
| Start of `scripts/train_500k.py` to the call of `fit`, the variant with minutiae (without a table with `--ppi 500`) | 21.1 to 30.4 s, peak 1,198 to 1,205 MB | 7.7 to 10.5 s, peak 1,330 to 1,339 MB |
| What one forked worker process adds, batches of 16, 384 items | 82 to 88 MB | about 51 MB |
| `scripts/build_sample_table.py`, 292x392 PNG files, files per second | | 536 to 656 with one process, 1,742 to 2,117 with four workers (8,000 files) |

Not measured: a GPU, real prints, the model built, a whole epoch, several worker processes at once, a start from
a cold disk (beyond a few runs right after the machine dropped its page cache).

---

## What was run for these documents

All on one machine with 4 CPU cores and no GPU.

### Runs for this version of the documents

Every command and code block of [training-guide.md](training-guide.md) and the runnable ones of
[tuning.md](tuning.md) were run once from the root directory of a copy of the repository, with small numbers
in place of the example numbers, and with the paths of this machine in the lines that enter the two
environments and name the model files. The images were **12 files made from the SFinGe example images
of this repository** (4 fingers with 3 images each, in three directories as in Step 1: the images as they
are, declared as 500 ppi; images enlarged to twice the size, declared as 1000 ppi; negatives, declared as
light ridges). They are synthetic and are not at a true 500 ppi, so these runs show that the commands work
and nothing about the quality of the results. The extraction ran in an environment with TensorFlow 2.20,
Pillow and tqdm, with `PYTHONPATH` set to `src` of the minutiae extractor at its `main` (commit `420b6e5`);
everything else ran in an environment made with Part A1 of [installation.md](installation.md) (Python 3.11,
PyTorch 2.15.0, CPU build).

| Command or snippet | Options in place of the example numbers | Result |
|--------------------|------------------------------------------|--------|
| File list with `find ... \| sort` (Step 4) | | 12 lines |
| Check that Python finds the extractor (Step 4, "Before you run it") | | found |
| `scripts/extract_minutiae.py` with `--file_list`, `--ppi_manifest`, `--skip_existing`, shard 0 and shard 1 (Step 4, "A very large number of files") | `--num_shards 2` | 6 + 6 `.mnt` files, exit status 0, 83 and 81 seconds |
| The command of Step 4 (no shards), afterwards, as the completeness check | | "Successful: 0", "Skipped: 12", "Failed: 0", exit status 0 |
| The same command in the training environment, where `fingerflow` cannot be imported | | Python error message `No module named 'fingerflow'`, exit status 1; `extraction_settings.json` was written |
| The extraction command with a path to a model file that does not exist (Troubleshooting 3) | | Python error message (`FileNotFoundError`), exit status 1 |
| `scripts/find_learning_rate.py`, the command of Step 5 | `--num_subjects 4 --num_impressions 3 --batch_size 4 --num_steps 30 --min_lr 1e-6 --max_lr 0.1` | ended after 23 steps, suggested 0.000085; wrote the CSV file and the plot |
| `scripts/train_500k.py`, the command of Step 6 | `--num_subjects 4 --num_impressions 3 --num_epochs 2 --validation_split 0.5 --batch_size 4` | 2 epochs, `DeepPrint_LocTexMinu`, exit status 0; `model.pyt` and `best_model.pyt` of 814 MB each |
| The same command again (Step 7) | the same | trained no further epoch, exit status 0 |
| The same command with `--localization off` (Step 7) | the same | refused: "holds a training with other settings", exit status 2 |
| The training command without any resolution option, new output directory (Step 6, Troubleshooting 6) | `--num_subjects 4 --num_impressions 3 --num_epochs 1 --validation_split 0 --batch_size 4 --localization off`, no `--minutiae_dir` | printed "12 resolution unknown: these files fail", started, ended in the first step with "Resolution unknown" and exit status 1; the output directory held `training_settings.json`, `data_preparation.json`, `log.json` and no model |
| One image per finger (Step 1) | `--num_subjects 4 --num_impressions 1 --ppi 500 --validation_split 0 --num_epochs 1 --localization off` on one of the three directories | trained 1 epoch (`DeepPrint_Tex`), exit status 0. With `--validation_split 0.5` instead: refused with exit status 2 ("needs at least 2 impressions per subject") |
| The two log snippets of Step 8 | the model of the 12 images | ran |
| Steps 9 and 10 | the same model, embedding its own 12 training and validation images | 12 embeddings of 512 values, saved; the error rates of a model after 4 training steps mean nothing |
| The snippet "To look at aligned images yourself" (tuning.md) | the same model | ran; rotations of about 0.00001 radians |
| The `LossWeights` snippet and the `LearningRateConfig(...)` of tuning.md | | the extractor and the settings object were built; `fit` was not called with them |
| The two lines that note the extractor revision (Step 4) | in a directory layout as Part B of the installation makes it, with a clone of the extractor under `../airgap-packages` | the Part B lines wrote the commit (40 characters) and printed it; the Part A lines, whose path does not exist in that layout, printed an error and wrote no file |
| The A/B recipe of tuning.md: the two range tests with `--variant`, the two trainings with `$COMMON`, the comparison snippet | 4 fingers, 1 epoch, `--batch_size 4`, one run of each kind; range tests of 4 steps | all ran; the range tests are too short for a suggestion and said so |
| The shortest working example of the README | as written (`--batch_size 16`), and once without `--batch_size` (default 64) | both: 1 epoch of `DeepPrint_LocTex` on 80 of the 100 example images, exit status 0, two model files of 0.5 GB each. With 16: 44 seconds, at most 4,069 MB of main memory. With 64: 61 seconds, at most 11,649 MB (largest resident memory of the process, read after it ended) |
| `notebooks/dataset_tutorial.ipynb`, executed with `jupyter nbconvert --execute` | as stored | all cells ran |
| `notebooks/model_training_tutorial.ipynb`, executed the same way | `num_epochs=2` in place of 20 in both training cells | first attempt: the first training cell was stopped by the operating system for lack of memory (more than 12 GB; the 50 images form one batch). Second attempt with the two commented lines of that cell active (batch of 16): both trainings ran (`TexMinu`, `LocTexMinu`) |
| `notebooks/embedding_generation_tutorial.ipynb`, executed the same way | the `TexMinu` model of the run before | all cells but the last ran ("Loaded best model", an equal error rate of 0.48 that means nothing); the last cell (DET curve) failed with `ImportError: cannot import name 'common_texification'` from `tikzplotlib` 0.10.1 with matplotlib 3.11.2 |
| `python -m pytest -q` | | 358 passed in an environment that has `scikit-image`; 357 passed and 1 skipped in the environment of Part A1, which does not (the tests as they were before the sample table) |

What was run of the installation is listed at the end of [installation.md](installation.md).

### Runs for the sample table

Made for [sample-table.md](sample-table.md) on the same machine (Python 3.11.18, PyTorch 2.15.0, CPU build,
pyarrow 26.0.1, OpenCV 5.1.0), the scripts started with `PYTHONPATH` set to the repository in the
place of the installed package, except where a row says otherwise. The images are the synthetic SFinGe example
images of this repository, copies of them and the two prints in its root directory; the trainings were a few
steps long. These runs show what the commands do, and nothing about accuracy. The last section of
sample-table.md lists every run and what was not run.

| Command or snippet | Run on | Result |
|--------------------|--------|--------|
| `scripts/build_sample_table.py`, `--check`, the range test and the training from the table | 12 files of 6 fingers in three directories (as they are, taken as 500 ppi; enlarged to 584x784, declared as 1000 ppi; negatives with light ridges), a manifest of three rows, `--ridges auto`, no `.mnt` files | exit status 0 for all; the builder wrote the same part byte for byte with 1, 2 and 4 workers |
| The same training without a table and from the table, `random`, numpy and torch seeded with 0 from outside, 2 epochs | the same 12 files | the same `model.pyt`, `best_model.pyt` and `log.json`, byte for byte (without a table twice); the range test, started without the seeding, wrote the same `lr_range_test.csv` in both modes |
| Continuing from a table: the same command again, one more epoch, after Ctrl-C, with another table, with a copy of the table, with and without a table | the same | nothing trained; one more epoch; continued with the next epoch; refused (exit status 2); accepted as the same training; refused both ways (exit status 2) |
| A set that only a table describes: a sample list, other names, PNG, BMP and TIFF files, subject numbers with gaps, fingers with 3, 3, 2, 2, 1, 1, 1 and 1 images; build, `--check`, range test, training | 14 files (12 made from the example images, the 2 prints of the root directory) | exit status 0; no finger with one image was held out with any of 1,000 validation seeds; the training without a table refused the directory |
| Refusals and failures: builds that fail or are refused, image files missing or changed after the build, damaged tables | the sets above | the messages of the troubleshooting of sample-table.md. Every byte of a part changed once on its own: 8,697 of 9,458 refused, 761 read unchanged, none read as other values |
| pyarrow without network access, and the scripts with the package installed instead of `PYTHONPATH` | a copy of the repository | see [installation.md](installation.md); the files of the runs above, byte for byte |
| A `TMPDIR` with a path of 139, 76 and 75 bytes, worker processes | the 12 files | 139 bytes: the range test and the training from the table, and the range test without a table, stood still with `OSError: AF_UNIX path too long`; the range test from the table at 76 bytes stood still, at 75 bytes it ended |
| A table, and images, at a path with letters outside ASCII in processes whose file names are ASCII or ISO-8859-1 | the 12 and the 14 files | the table: exit status 0; the images: a segmentation fault of OpenCV (ASCII) or refusals with causes that are not the cause (ISO-8859-1); with `PYTHONUTF8=1`: exit status 0 |
| A table with `.mnt` files (46 minutiae per image drawn at random, written as the extraction writes them): build, the variant with minutiae (training and range test), rows without minutiae with and without `--tolerate_missing_minutiae`, `--ignore_minutiae`, `load_trained_model(..., samples=...)` | 40 files of 20 fingers made from the example images | all as sample-table.md describes; with the default batch size of 64, the range test and the training were ended by the memory limit of the runs (14.3 GB) after their first step and their first epoch |
| The start of the training at one million rows, reading the table, a forked worker, the builder on 800 and 8,000 files, `--check` | synthetic rows, see [Large datasets](#large-datasets) | the figures there |
| `python -m pytest -q` | | 952 passed in the environment of these runs; 951 passed and 1 skipped (scikit-image) in a new environment of Part A1 with the extra `table`, 396 passed and 170 skipped without it |

### Earlier runs, on real prints

These are from earlier work on this repository and were not repeated for this version. The scripts have not
changed in behaviour since; the minutiae extractor used then is the revision that is now its `main`.

| Command or snippet | Run on | Result |
|--------------------|--------|--------|
| File list, extraction in 2 shards, completeness check | 12 real prints of 4 fingers in three sensor directories (500 ppi, 1000 ppi, 1000 ppi with light ridges) | 6 + 6 `.mnt` files, exit status 0; then "Successful: 0", "Skipped: 12" |
| `scripts/extract_minutiae.py` with a minutiae extractor from before commit `420b6e5` (too old) | 4 prints | stopped before the first image, exit status 2 |
| `scripts/find_learning_rate.py` with `--num_subjects 4 --num_impressions 3 --batch_size 4 --num_steps 30 --min_lr 1e-6 --max_lr 0.1` | the same 12 prints | ended after 23 steps, suggested 0.000085 |
| `scripts/train_500k.py` with `--num_subjects 4 --num_impressions 3 --num_epochs 2 --validation_split 0.5 --batch_size 4`, and the same command again | the same 12 prints | 2 epochs, `DeepPrint_LocTexMinu`, exit status 0; then "Nothing was trained in this run" |
| Continuing with changed options (Step 7): another learning rate; `--localization off`; `--num_epochs 3` and another learning rate | 32 prints of 8 fingers as 500, 1000 and 385 ppi files and 1000 ppi negatives | nothing trained; refused with exit status 2; one more epoch |
| Steps 8, 9 and 10 and the alignment snippet | the model of the 12 prints | ran; 12 embeddings of 512 values, saved and loaded again |
| The offline `pip install --no-index ... -e .` of the installation | a fresh environment with only setuptools and wheel from a local directory | installs with `--no-deps`; without it, it names the first missing dependency |

### Not run

- Anything on a GPU, and `torchrun`.
- The download and offline installation commands of Part B of [installation.md](installation.md), and Part A2
  (see the table there).
- Any training set larger than 440 images, any training longer than 300 steps.
- The five extractor comparison scripts in `scripts/` and the scripts of the paper in `flx/scripts/`.
- The `SlapSegmenter`, the identification benchmarks, and loading the pre-trained model of the paper.
- The last cell of `notebooks/embedding_generation_tutorial.ipynb` runs into an import error (see the table
  above).

---

## Changes against earlier versions, and what to do with old files

Read this if you have trainings, logs, model files or `.mnt` files that were made with an earlier version of
this code. If you start with today's code, you do not need it.

**Learning rate.** The default was 0.025, without warm-up. On real images it destroyed the training in the
second step (table under
[The default learning rate](#the-default-learning-rate-what-was-measured-and-what-was-not)). A rate of 0.01,
which an earlier guide recommended, did the same in the third step. The default is now 0.0001 with a warm-up.
A training that was started with one of the old rates should be started again.

**Loss.** Four things changed:

1. **The center loss is a mean over the batch, not a sum.** As a sum with the
   factor 0.125 its weight relative to the cross entropy grew with the batch
   size. The default 8.0 is 0.125 x 64, so with the batch size of
   `train_model` (64) the loss is the same as before. For a batch of B images
   the former balance is `--loss_weight_center` = 0.125 x B.
2. **The minutia map loss is a sum over the cells of a map with the weight
   0.095**, as in the DeepPrint paper (Engelsma et al., 2019, equation 8). It
   was the mean over the cells with the weight 0.3, which is the same as the
   sum with the weight 0.3 / 98,304 = 0.00000305. The reason is the
   measurement under
   [Why the weight of the minutia map loss is 0.095](#why-the-weight-of-the-minutia-map-loss-is-0095).
3. **The minutia map loss is averaged over the images that have a map.** It
   was averaged over all images of the batch, so images without a map lowered
   the weight of the others.
4. **An image without any minutiae has no map.** It used to get an empty map,
   which taught the model that the image has no minutiae. A fingerprint always
   has minutiae, so an empty list means that the extraction failed. Such an
   image now counts in the cross entropy and the center loss only.

A model that was trained before is not changed by this; the loss is only used
during training. A training that is resumed from `model.pyt` continues with
the new loss unless you pass the former weights:
`--loss_weight_center` 0.125 x batch size and
`--loss_weight_minutia_map 0.00000305`. Points 3 and 4 cannot be switched
back. If you resume with the new weights, the `training_loss` of the new
epochs is much larger than that of the old ones, mainly because of the map
part, and the two cannot be compared. This matters for a training without a
validation set, where the epoch with the lowest training loss becomes
`best_model.pyt`: an old epoch would stay the best one.

**Model files of the variants with localization network** (`Loc...`) from before the bounds on rotation and
shift existed still load, but they should not be used: they were trained while the localization network mixed
up the images of a batch, and the same raw output of the network now means a different rotation and shift.

**`best_model.pyt` of trainings without a validation set.** Earlier versions stored the training accuracy in
`validation_equal_error_rate` when there was no validation set and kept the epoch with the lowest value,
which was the epoch with the lowest accuracy. A `best_model.pyt` from such a run should not be used;
`model.pyt` of the same run holds the last epoch. `TrainingLog` reads such old entries as entries without
validation.

**`training_accuracy` in old logs** holds the mean over the subjects of the share per subject, which is not
the same number as today's share of all training samples.

**`.mnt` files that were extracted before the minutiae extractor refined the minutia directions** have unreliable
minutia directions. Extract them again.

**An output directory of a training without `training_settings.json`** is continued with a note; only its
image preparation (`data_preparation.json`) can be compared with the new command.

**`--canvas_size`** did not exist; its default 0 is the behaviour of earlier versions.

**The example picture of the range test** in tuning.md was made with the former loss weights and with an
earlier rule for the suggestion (a tenth of the rate of the steepest fall, where today it is a tenth of the
rate with the lowest loss).
