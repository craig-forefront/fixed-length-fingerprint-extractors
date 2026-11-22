# Background: the paper behind this code

This repository started as the refactored code of the paper "Benchmarking fixed-length Fingerprint
Representations across different Embedding Sizes and Sensor Types" (BIOSIG 2023, Rohwedder et al.). It
contains an implementation of [DeepPrint](https://arxiv.org/abs/1909.09901) (Engelsma, Cao and Jain, 2019), a
deep learning based extractor of fixed-length fingerprint representations, and the experiments of the paper
with this architecture.

**Everything on this page is taken from the paper and its authors. None of it was measured again for the
other documents in `docs/`**, and the code has changed since (learning rate, loss weights, localization
network; see [measurements.md](measurements.md#changes-against-earlier-versions-and-what-to-do-with-old-files)).
Do not read the figures below as results of today's code or as what to expect on your data.

## Abstract

Traditional minutiae-based fingerprint representations consist of a variable-length set of
minutiae. This necessitates a more complex comparison causing the drawback of high computational cost in one-to-many comparison. Recently, deep neural networks have been proposed to extract fixed-length embeddings from fingerprints. In this paper, we explore to what extent fingerprint
texture information contained in such embeddings can be reduced in terms of dimension, while preserving high biometric performance. This is of particular interest, since it would allow to reduce
the number of operations incurred at comparisons. We also study the impact in terms of recognition
performance of the fingerprint textural information for two sensor types, i.e. optical and capacitive.
Furthermore, the impact of rotation and translation of fingerprint images on the extraction of fingerprint embeddings is analysed. Experimental results conducted on a publicly available database reveal an optimal embedding size of 512 feature elements for the texture-based embedding part of fixed-length fingerprint representations. In addition, differences in performance between sensor types can be perceived.

## DeepPrint

![Conceptual overview of fixed-length fingerprint representation extraction](../figures/conceptual-overview-deepprint.png)

The full DeepPrint model consists of the following components:

- Localization network: proposes a rotation and a shift for the fingerprint image, with which the image is
  resampled before the other components see it (learned alignment). It is not shown in the figure. How it is
  bounded and checked in today's code is described under [Learned alignment](tuning.md#learned-alignment).
- Inception v4 stem: processes the fingerprint image and extracts a feature map.
- Texture branch: processes the feature map and extracts a fixed-length fingerprint representation.
- Minutiae branch: processes the feature map and extracts a fixed-length minutiae representation as well as a
  minutia map. The minutia map is used during training to focus the training on the minutiae regions of the
  fingerprint. It is not used to create the fixed-length fingerprint representation.

In this project, many smaller variants of the DeepPrint model were examined. In the code, the naming
convention is:

`DeepPrint_[Loc][Tex][Minu]_<NDIMS>`

`[Loc]` means that the localization network is included, `[Tex]` that the texture branch is included and
`[Minu]` that the minutiae branch is included. `<NDIMS>` is the dimensionality of the fixed-length fingerprint
representation (the number of neurons in the last linear layer). In `scripts/train_500k.py`, `--embedding_dim`
is the size per branch: a variant with texture and minutiae branch gives twice that many values per print
(512 with the default of 256).

## Databases of the paper

In the code we do not distinguish between different fingers of one person and fingers of different persons.
The term "subject" refers to one distinct finger, and "impression" to a single capture of it. This deviates
from the terminology of the paper. In these terms, the databases used in the paper are:

- Synthetic databases created with the SFinGe fingerprint generator, for training (6000 subjects with 10
  impressions each) and validation (2000 subjects with 4 impressions each).
- The MCYT330 optical and capacitive sensor databases (3300 subjects, 2 sensors, 12 impressions each). The
  first 2000 subjects were used for training and the remaining 1300 for testing.

There is also code to load and benchmark the FVC2004 DB1 and SD4 databases; these were not used in the paper.
The predefined datasets and models of the paper are in `flx/setup/`, and its experiments are run by the
scripts in `flx/scripts/` (training, embedding extraction, verification and identification benchmarks, result
tables and plots; the benchmark definitions are in `data/benchmarks/`). These scripts expect the datasets of
the paper under `data/fingerprints/` and were not run for the documents in `docs/`.

## Image processing of the paper

The input size of DeepPrint is 299x299 pixels. How the resizing and cropping is done depends on the source
dataset and can be found in `flx/data/image_loader.py`.

During training, data augmentation is applied to the fingerprint images:

- Random rotation between -15° and 15°
- Random translation of up to 25 pixels in x and y direction
- Random gain and contrast adjustment (`QUALITY_AUGMENTATION` in `flx/setup/datasets.py`)

These values belong to the predefined datasets of the paper in `flx/setup/datasets.py`.
`scripts/train_500k.py`, the script for your own images, uses a rotation between -30° and 30° and a
translation of up to 40 pixels (`POSE_AUGMENTATION` in the script).

The rotation and translation are also applied to the minutiae, so that the minutia map of a training sample
follows the augmentation of its image (`PoseTransformedMinutiaMapLoader` in
`flx/data/minutia_map_loader.py`). This does not correct a minutia map that does not match its image without
augmentation.

The following preprocessing steps are applied to the fingerprint images before they are fed into the network:

![Example of a synthetic fingerprint image generated by SFinGe with the respective preprocessing steps](../figures/preprocessing-steps.png)

## Training in the paper

Statements of the original authors, from their README of 2023, not measured here. This includes the one GPU
timing in these documents; no GPU was used for any other statement.

- The Adam optimizer was used, and the best learning rate differed between the variants: a higher one could
  be chosen for simpler variants like `DeepPrint_Tex_512` to speed up training, and a lower one was necessary
  for more complex variants like `DeepPrint_LocTexMinu_512` to avoid overfitting.
- The number of epochs was chosen by the equal error rate on a validation set; the training was stopped when
  it did not decrease any more.
- For the datasets of the paper (60,000 training images), up to 100 epochs were necessary depending on the
  learning rate and the model, at roughly 10 to 15 minutes per epoch on an A100 GPU.

How the learning rate is chosen with today's code is described in
[tuning.md](tuning.md#choosing-the-learning-rate).

## Experiments of the paper

### Embedding size

Different embedding sizes were compared for the variant with only the texture branch (which has a lower
computational cost than the full DeepPrint model):

![Results for different embedding sizes](../figures/embedding-sizes-results.png)

### Model variants and sensor types

Different model variants were compared for optical and capacitive sensor types:

![Results of different variants](../figures/different-variants-results.png)

### Pose alignment

The impact of pose alignment on the performance of the DeepPrint model was estimated by randomly changing the
image pose with increasing magnitude of shifting and rotation:

![Results for varying poses](../figures/pose-alignment-results.png)

## Pre-trained model of the authors

The authors published a model "for an embedding size of 512" on
[Google Drive](https://drive.google.com/drive/folders/1vV2skXApZMhqWTlF2j_qgXDxRYan5U1f?usp=drive_link).

**Loading it was not tried for these documents**, and the files were not looked at. What the code says: a
parameter file can only be loaded into a model of the same variant, the same embedding size and the same
number of training subjects. The models of the paper are defined in `flx/setup/experiments.py` with
2000 + 6000 = 8000 training subjects; which of them the published file belongs to (for example
`get_DeepPrint_Tex(8000, 512)` or `get_DeepPrint_TexMinu(8000, 256)`) has to be found by trying. It was
trained on the data of the paper with the code of that time, on images prepared as in the paper (not
resampled to 500 ppi by `SensorNormalizer`), so it is not a starting point for the route of
[training-guide.md](training-guide.md).

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@inproceedings{
    Rohwedder-FixedLengthFingerprintDNN-BIOSIG-2023,
    author = {T. Rohwedder and D. Osorio-Roig and C. Rathgeb and C. Busch},
    booktitle = {Intl. Conf. of the Biometrics Special Interest Group ({BIOSIG})},
    keywords = {Fingerprint recognition, computational workload reduction},
    month = {September},
    publisher = {IEEE},
    title = {Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types},
    year = {2023}
}
```
