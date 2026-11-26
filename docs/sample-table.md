# The sample table: a training set described in Parquet files

This document is for training sets that do not fit the file layout of
[Step 1 of the training guide](training-guide.md#step-1-lay-out-the-files): file names that are not
`<subject>_<impression>`, fingers with different numbers of images, subject numbers with gaps, several image
formats. It is an addition to [training-guide.md](training-guide.md), which describes the route; read that
guide first. A set that fits the layout trains as the guide describes it, without a table.

## Contents

1. [What a sample table is, and whether you need one](#what-a-sample-table-is-and-whether-you-need-one)
2. [What this document rests on](#what-this-document-rests-on)
3. [Installing pyarrow](#installing-pyarrow)
4. [The columns](#the-columns)
5. [Building a table](#building-a-table)
6. [Checking a table](#checking-a-table)
7. [The range test and the training from a table](#the-range-test-and-the-training-from-a-table)
8. [What is the same and what differs from a training without a table](#what-is-the-same-and-what-differs-from-a-training-without-a-table)
9. [Size, time and memory](#size-time-and-memory)
10. [Troubleshooting](#troubleshooting)
11. [What was run of this document, and what was not](#what-was-run-of-this-document-and-what-was-not)

---

## What a sample table is, and whether you need one

A *sample table* is a directory of Parquet files with **one row per single-finger image file**. A row says which
finger (*subject*) and which capture (*impression*) the file shows, at which resolution and with which ridge
polarity it is prepared, its size before and after the preparation, and the minutiae of the prepared image. The
**pixels stay in the image files**: a row names its file by the path below the image directory. The table is
built once, after the minutiae extraction, by `scripts/build_sample_table.py`, which loads every image as the
extraction loaded it. `scripts/find_learning_rate.py` and `scripts/train_500k.py` then read it with
`--samples`.

Without a table the training finds which finger an image shows in its file name, and the range test and the
training must be given the resolution options of the extraction again. A table helps where that does not fit:

- **File names that are not `<subject>_<impression>`.** A sample list (a CSV file) says which finger and which
  capture each file shows; the names can be anything.
- **Fingers with different numbers of images**, also fingers with one image beside fingers with many, and
  **subject numbers with gaps**. Without a table every subject needs the same number of impressions, numbered
  without gaps.
- **Several image formats.** A table built from a directory holds every image file that the extraction reads
  (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`), and a sample list can name files of other endings too; a
  training without a table reads one ending, `.png`.
- **Several sensors with different resolutions and ridge polarities.** The table records for every file what it
  was taken to have when it was built. The range test and the training read it from there and refuse the
  resolution options, so the options of the extraction are given once more, to the builder, and never again.

A set that fits the layout of the training guide trains without a table as it did before there were tables:
without `--samples` the scripts read the files by their names and do not need the package pyarrow.

**What a table does not change:**

- The images are still read from their files and prepared (resampled, augmented, binarised) in every epoch.
  The table holds no pixels.
- The classifier still has one row per training subject, with what that means above about a million fingers
  ([large-datasets.md](large-datasets.md#above-about-one-million-subjects)).
- Images are still drawn uniformly. A finger with many images is trained more often than one with few, and in
  validation all pairs of its images count. Nothing balances this.
- Every GPU process holds the whole table in memory, the minutiae of **all** rows among it (a training without
  a table reads them per image from the `.mnt` files): 635.0 MB of arrays for one million images with 46
  minutiae each (measured, [Size, time and memory](#size-time-and-memory)), 552 MB of them the minutiae
  (calculated: 46 million minutiae of 12 bytes). The identifiers of the images are still Python objects in
  every process.

What the table does not do at all: it holds no pixels, so images are not binarised once and stored; it does not
balance fingers or limit the images per finger; it has no shards and cannot be appended to (a table is built
again as a whole); and it knows fingers, not people, so validation holds out fingers and there is no column for
the person or the finger position.

---

## What this document rests on

- **One machine**: Linux x86_64 (Ubuntu 24.04.5, kernel 6.18.44), 4 CPU cores (Intel Xeon, 2.30 GHz), 16.9 GB
  of main memory (calculated from `MemTotal`, 16,473,316 KiB), no swap and **no GPU**. Python 3.11.18, PyTorch
  2.15.0 (CPU build), torchvision 0.29.2, numpy 2.4.6, **pyarrow 26.0.1**, OpenCV 5.1.0, Pillow 13.0.0, pandas
  3.1.1. Worker processes are forked (the default start method of this Python on Linux).
- **Synthetic images.** The runs on images used the SFinGe example images of this repository
  (`notebooks/example-dataset`, 292x392 pixels, files that state 96 ppi), copies of them that were enlarged,
  inverted or written as BMP and TIFF, and the two prints in the root directory of the repository. They are not
  real prints and the trainings were a few steps long, so **nothing here says anything about accuracy**.
- **Three sets of image files.** *The twelve files*: 6 fingers with 2 images each, named
  `<subject>_<impression>.png`, in three directories: 4 as they are (taken as 500 ppi), 4 enlarged to 584x784
  pixels (declared as 1000 ppi), 4 negatives (light ridges), with a resolution manifest of three rows. *The
  fourteen files*: 8 fingers with 3, 3, 2, 2, 1, 1, 1 and 1 images, subject numbers with gaps, names that are
  not `<subject>_<impression>` (one with spaces, one with a letter outside ASCII), PNG, BMP and TIFF files,
  among them the two prints of the root directory, described by a sample list. *The forty files*: 20 fingers
  with 2 images each, with a `.mnt` file per image. The measurements used one million synthetic rows
  ([Size, time and memory](#size-time-and-memory)).
- **No minutiae extractor.** The machine has none. The tables of the runs on images were built without
  `--minutiae_dir` (every row without minutiae) or with `.mnt` files whose minutiae were drawn at random and
  written as `scripts/extract_minutiae.py` writes them. So the minutiae in these runs are not those of the prints.
- **Nothing was run on a GPU**, and nothing with `torchrun`.
- **Figures are marked**: *measured* (taken in these runs: a time, a size, a count), *calculated* (arithmetic on
  measured figures) or *read* (from the code or another document, not run). **MB means 10^6 bytes and GB
  10^9 bytes in this document.** Figures that were recorded in MiB (2^20 bytes) are converted, and marked
  calculated.
- **How the commands were run.** This document gives the commands as you type them in the training
  environment with the package installed ([installation.md](installation.md)), from the root directory of the
  repository. Most runs were started with `PYTHONPATH` set to the repository in the place of the installed
  package; one series ran in a copy of the repository with the package installed and without `PYTHONPATH`, and
  wrote the same files as the others. The last section says which was which.

---

## Installing pyarrow

Only the scripts that read or write a table need the package **pyarrow**: `scripts/build_sample_table.py`
(building and checking), `scripts/find_learning_rate.py` and `scripts/train_500k.py` with `--samples`, and
`load_trained_model(..., samples=...)`. Without `--samples` nothing needs it (run: the range test of the
twelve files without a table, in an environment without pyarrow, wrote the same `lr_range_test.csv` as in one
with it).

**With network access**, in the training environment ([installation.md](installation.md), Part A1), in the
root directory of the repository:

```bash
pip install -e ".[table]"
```

Run in a new environment made as in Part A1 (the CPU build of PyTorch, `pip install -e .` and
`pip install -e ".[dev]"`): pip chose and installed pyarrow 26.0.1.

The requirement is `pyarrow>=17.0` (the extra `table` of `pyproject.toml`, and
`flx.preparation.sample_table.PYARROW_REQUIREMENT`). Not below 17.0: with pyarrow 16.1, some damaged parts end
the interpreter with a segmentation fault inside pyarrow when their rows are read, where 17.0 and later raise an
error that the reader turns into a refusal with the name of the part (read in the module; found by the
reviewers of the format, who damaged parts bit by bit).

**Without network access** (as in Part B of [installation.md](installation.md)): on a machine with network
access and with the same operating system, processor architecture and Python version as the air-gapped one,

```bash
python3.11 -m pip download "pyarrow>=17.0" --dest ./packages
```

This brought one file for Python 3.11 on Linux x86_64, `pyarrow-26.0.1-cp311-cp311-manylinux_2_28_x86_64.whl`
of 55.0 MB, and nothing else: pyarrow 26.0.1 requires no other package. On the air-gapped machine, in the
training environment, in the root directory of the repository:

```bash
pip install --no-index --find-links=./packages --no-build-isolation -e ".[table]"
```

with the directory of the file in the place of `./packages`. `--no-build-isolation` is needed when that
directory holds no `setuptools` and `wheel`: without it pip looks for them there to build the package and stops
with `No matching distribution found for setuptools>=61.0`. (With `setuptools`, `wheel` and `packaging` in the
directory the command also works without the option.) pyarrow alone, without the package of this repository:
`pip install --no-index --find-links=./packages "pyarrow>=17.0"`.

Check:

```bash
python -c "import pyarrow; print(pyarrow.__version__)"
```

**Without pyarrow**, every command that needs a table stops with exit status 2 and says what to install:

```
ERROR: A sample table is read and written as Parquet files, which needs the package pyarrow, and it cannot be imported (No module named 'pyarrow'). Install it in the root directory of this repository with: pip install -e ".[table]" or anywhere with: pip install "pyarrow>=17.0" (with the quotes).
```

---

## The columns

A table is a directory of *parts*, `part-00000-of-00004.parquet` to `part-00003-of-00004.parquet`; the builder
cuts the listed files into parts of 250,000. Beside the parts it holds at most the two logs of the build
(`build_failures.txt`, `build_warnings.txt`). Every part has these columns (the definition is the docstring of
`flx/preparation/sample_table.py`; schema version 1):

| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `subject` | int32 | no | The finger: all images of one finger have the same number. 0 or more; gaps are allowed |
| `impression` | int32 | no | The capture among the images of the finger. 0 or more and unique within the subject; a finger can have any number |
| `image_path` | string | no | The image file, relative to the image directory (`--dataset_dir`), with `/` as separator, without `.`, `..` or empty components and without `/` at the start. Unique |
| `file_size` | int64 | no | Bytes of the image file when the table was built |
| `ppi` | float64 | no | The resolution that the file was taken to have, in pixels per inch |
| `ppi_origin` | string | no | Where `ppi` comes from: `argument` (`--ppi`), `manifest`, `metadata` (the file states it) or `assumed` (`--assume_ppi`, for a file that states none or one outside of 200 to 2100 ppi) |
| `ridges` | string | no | What the file was taken to hold: `dark` or `light` ridges; never `auto` |
| `ridges_origin` | string | no | Where `ridges` comes from: `argument` (`--ridges`), `manifest`, `estimated` (from the image, `--ridges auto`) or `default` (dark, because the estimate gave no decision) |
| `source_width`, `source_height` | int32 | no | Pixels of the image file |
| `width`, `height` | int32 | no | Pixels of the prepared image: the file resampled from `ppi` to 500 ppi |
| `minutiae` | list of (`x`, `y`, `angle`), float32 each | yes | The minutiae of the prepared image: `x` the column and `y` the row in its pixels (`y` pointing down), `angle` the direction in radians, as in a `.mnt` file. **Null**: no minutiae are known (no `.mnt` file was given or accepted). **An empty list**: the image has none |

- **The numbers.** From file names the table holds each number minus 1, as a training without a table counts
  them: `17_3.png` is subject 16, impression 2. From a sample list it holds the numbers as they are written,
  so a sample list of files named that way gives the rows of a table built from their directory only if it
  gives each number minus 1.
- `ppi` has 64 bits because 32 would give another prepared size for some files: 785 pixels at 251.2 ppi are
  1562 pixels at 500 ppi, and 1563 with the value rounded to 32 bits (read in the module docstring).
- **File metadata of every part:** `flx.table` = `samples`, `flx.schema_version` = `1`, `flx.target_ppi` =
  `500.0`, and `flx.build`, a JSON object: the id of the build (the same in all its parts; it is made from the
  options and the list of files, not from what the files hold), the number of the part, the number of parts,
  its rows, the files listed and failed, the six resolution settings of the build, and whether it read a
  minutiae directory.
- **The checksum** `samples_sha256` is a SHA-256 over the values of the rows in the order of (subject,
  impression), not over the files: the same rows give the same checksum however they are split into parts.
  It leaves out `ppi_origin`, `ridges_origin` and the file metadata. The build and `--check` print it, and a
  training stores it.

**The descriptions are stored in the files.** Every column carries its meaning as the metadata `description`
of the Arrow schema, so a part can be understood without this repository. With pyarrow:

```python
import pyarrow.parquet as pq

schema = pq.read_schema("data/tables/my-dataset/part-00000-of-00001.parquet")
for field in schema:
    print(f"{field.name} ({field.type}): {field.metadata[b'description'].decode()}")
print(schema.metadata[b"flx.build"].decode())
```

On the part of the table of the twelve files this printed thirteen lines like these, and the build:

```
subject (int32): Number of the finger: all images of one finger have the same subject. 0 or more; the numbers need not be consecutive.
impression (int32): Number of the capture among the images of its finger. 0 or more and unique within the subject; a finger can have any number of impressions.
[...]
minutiae (list<element: struct<x: float not null, y: float not null, angle: float not null> not null>): The minutiae of the prepared image. Null: no minutiae are known for the image. An empty list: the image has no minutiae.
{"id": "7c531b05d53dabf16ce91e37de7d55836ab00666e8c6abce069a7551a90deab6", "settings": {"target_ppi": 500.0, "ppi": null, "ppi_manifest_sha256": "274d53857ba96acf56d1bc354e96b0853b9eb3660394383602365c71b056e314", "ppi_manifest_entries": 3, "assume_ppi": null, "ridges": "auto"}, "with_minutiae": false, "files": 12, "failed": 0, "part": 0, "parts": 1, "rows": 12}
```

The same part was also read with pyarrow alone, in an environment that held nothing of this repository: the
descriptions, the four `flx.*` keys and the rows. (A Parquet reader that is not built on Arrow sees the columns,
their types and the four `flx.*` keys, not the descriptions; read in the notes of the format, not tried.)

**The rows of a whole table** are read from the list of its parts:

```python
import glob

import pyarrow.parquet as pq

parts = sorted(glob.glob("data/tables/my-dataset/part-*.parquet"))
rows = pq.read_table(parts)
print(rows.num_rows, "rows:", ", ".join(rows.column_names))
```

With pandas, `pandas.read_parquet(parts)` reads the same rows. Give them the list, not the directory: where the
build kept a log, pandas and pyarrow take `build_failures.txt` or `build_warnings.txt` for a part and stop. Run
with pandas 3.1.1 on a table of 2 rows with both logs: the lines printed `2 rows: subject, impression,
image_path, ...` with the thirteen columns, `pandas.read_parquet(parts)` gave the same 2 rows, and
`pandas.read_parquet` and `pq.read_table` of the directory stopped with `ArrowInvalid: Error creating dataset.
Could not read schema from '<table>/build_failures.txt'. Is this a 'parquet' file?: ...`. Of a table without
logs both read the directory as well.

**What a reader refuses.** `--check`, the range test and the training read a table with the same function, and
refuse with exit status 2 and a message that names the directory or the part: a directory without parts; a
missing or doubled part, or parts of two builds; a part that is no Parquet file or is damaged (the pages of a
part carry checksums, and they are compared); a part without the `flx.*` keys, of another kind or of another
schema version; a missing column or a column of another type (nothing is converted: a float32 `ppi` is
refused); a null outside of `minutiae`; a text outside of the values of the table above; negative numbers, two
rows with the same (subject, impression) or the same `image_path`, a path that leaves the image directory, a
`ppi` that is not a number above 0, a prepared size that does not follow from the source size and `ppi`, and
minutiae that are not finite numbers. Columns that it does not know are named in a note and not used. Of the
9,458 bytes of one part, each changed once on its own (run), 8,697 changes were refused and 761 changed nothing
that the reader returns; none was read as other values.

---

## Building a table

The builder runs in the training environment, after the minutiae extraction (Step 4 of the training guide).
It loads every image once, as the extraction loaded it, reads its `.mnt` file, and writes the table.

**From a directory laid out as in Step 1 of the training guide** (names `<subject>_<impression>` with whole
numbers from 1, in any subdirectories):

```bash
python scripts/build_sample_table.py \
  --dataset_dir data/fingerprints/my-dataset \
  --minutiae_dir data/poses/my-dataset \
  --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
  --samples data/tables/my-dataset
```

Without `--sample_list` the table gets **every** image file below `--dataset_dir` with one of the six endings,
in any subdirectory, and every subject. A file whose name is not `<subject>_<impression>`, or a directory below
`--dataset_dir` that cannot be listed, refuses the build before any image is loaded.

**From a sample list**, for any other names: a CSV file with a header line and the columns `path`, `subject`
and `impression`. The list of the fourteen files began:

```
path,subject,impression
desk_a/p0007_left_index_01.png,3,1
desk_a/p0007_left_index_02.png,3,2
desk_b/p0007 left index 03.bmp,3,3
desk_a/p0008_left_index_01.png,4,1
desk_b/p0008_left_index_02.tif,4,2
```

```bash
python scripts/build_sample_table.py \
  --dataset_dir data/fingerprints/my-dataset \
  --sample_list data/fingerprints/my-dataset-samples.csv \
  --minutiae_dir data/poses/my-dataset \
  --ppi_manifest data/fingerprints/my-dataset-resolutions.csv \
  --samples data/tables/my-dataset
```

- `path` is relative to `--dataset_dir`. The builder normalises it as text (`./`, a doubled `/` and `dir/..`
  are removed: `./desk_a//p0007_left_index_01.png` went into the table as `desk_a/p0007_left_index_01.png`)
  and takes it otherwise as it is written: a space at its ends belongs to the file name, and a path with a
  comma stands in double quotes. Files of any ending can be listed.
- `subject` and `impression` are whole numbers from 0 to 2,147,483,647 and go into the table as they are
  written; nothing is subtracted. Further columns are named in a note and not used.
- Every file needs a pair of numbers of its own. The build is refused before any image is loaded if two files
  have the same pair, a file is listed twice, two files differ only in their ending (they would share one
  `.mnt` file), or a path cannot be put into a table (a backslash, a name that is not UTF-8, a path that
  leaves `--dataset_dir`). (An `Identifier` of the code compares by a hash of its two numbers, so two pairs
  whose hashes are equal could not be told apart; the start of a range test or a training refuses such a
  table and names both rows. Read in the code; no such pair occurred in these runs.)

**The resolution options**: `--ppi`, `--ppi_manifest`, `--assume_ppi` and `--ridges`, exactly as the extraction
got them ([Step 2](training-guide.md#step-2-say-what-resolution-and-ridge-polarity-each-file-has)). This is the
last time they are given: the table records for every file which resolution and polarity it was taken to have,
and the range test and the training refuse the options. With `--minutiae_dir`, the builder compares them with
`extraction_settings.json`, which the extraction writes into that directory, and refuses a build with other
settings (exit status 2); `--allow_other_extraction_settings` builds all the same, and every `.mnt` file is
then accepted or refused by its own checks. Where the directory holds no `extraction_settings.json`, the build
says so in a note and decides in the same way. A file whose resolution is not known fails, as in the
extraction.

**`--minutiae_dir`**: the output directory of the extraction. The `.mnt` file of an image is looked for where
the extraction writes it: at the path of the image below that directory, with the ending `.mnt`.

- A `.mnt` file is **accepted** if the size in its header is the size of the prepared image, and, where its
  last line is the comment that the extraction writes (`# source 292x392 at 500 ppi (manifest), ridges dark
  (argument), prepared 292x392`), if the source size and the ridge polarity named there are those of the build.
  An accepted file whose first line is 0 gives an empty list; so does a file without a header (fewer than two
  lines), which is named in the warnings log.
- A `.mnt` file that is **missing or refused** (another size, another source size or polarity in the comment, a
  file that cannot be read, a value that is not a finite number) gives the row **null minutiae** and a line in
  the failures log. The image is still in the table.
- **Without `--minutiae_dir`** every row has null minutiae, and that is no failure.

**`--workers N`**: worker processes that load the images (default 1: they are loaded in the main process). A
worker process tells OpenCV to work in its own thread before it loads anything, so the number of workers is the
number of cores that a build uses for loading. A file whose worker process dies is loaded once more; if that
worker dies as well, the file gets a line in the failures log and no row. There is no time limit per file. With
1, 2 and 4 workers the builder wrote the same part byte for byte (run on the twelve files).

**What the build prints.** On the twelve files (their directories `as_is`, `enlarged_2x` and `negative` below
`a/images`), with `--dataset_dir a/images --samples a/table --ppi_manifest a/resolutions.csv --ridges auto` and
no minutiae directory, exit status 0:

```
Images:   resampled to 500 ppi; resolution from manifest (3 rows), file metadata, else the file fails; ridges auto
Minutiae: not read (no --minutiae_dir): every row gets null minutiae
Scanning a/images for images...
Files:    12, to be written as 1 part(s)
============================================================
Sample table: a/table
  Files listed:              12
  Rows:                      12
  Files that failed:         0
  Rows with minutiae:        0
  Rows with empty minutiae:  0
  Rows with null minutiae:   12 (the table was built without --minutiae_dir)
  Subjects:                  6
  Impressions per subject:   smallest 2, median 2, largest 2
  Parts:                     1
  Checksum (samples_sha256): 85577723f185c1659dfe6547082700c9652528b18dcf224a5644edbf5cbe7f2f
  Failures log:              none
  Warnings log:              none
============================================================

Resolution of the 12 image file(s) of the table:
             8  500 ppi from manifest
             4  1000 ppi from manifest (resampled to 500 ppi)
Ridge polarity:
             8  dark ridges, estimated from the image
             4  light ridges, estimated from the image
```

With a minutiae directory, the second line names it, a line `Settings: those of the extraction
(<minutiae_dir>/extraction_settings.json)` follows (where the directory holds no such file, the note `Note:
<minutiae_dir>/extraction_settings.json is missing, so the settings of the extraction cannot be compared with
those of this build; every .mnt file is accepted or refused by its own checks.`), and the summary counts the
minutiae (`Rows with minutiae: 8,000 (368,000 minutiae)` in a build of 8,000 files of the measurements). A
progress bar goes to standard error.

**The two logs**, in the directory of the table, one line per file (path, a tab, the message, of which at most
1,000 characters), in the order of the rows. A log without lines is not kept.

- `build_failures.txt`: `no row: ...` for an image that could not be loaded (it is **not** in the table) and
  `null minutiae: ...` for an image whose `.mnt` file was missing or refused (it **is** in the table, without
  minutiae). For example, from the runs (the fourteen files without a resolution option, and the twelve
  files with a minutiae directory that held no `.mnt` file):

  ```
  desk_a/p0007_left_index_01.png	no row: UnknownResolutionError: Resolution unknown. The file states 96 ppi, which is outside of the plausible range 200 to 2100 for a fingerprint sensor and is not used, and neither an explicit resolution nor a manifest row gives one. Give the resolution of all files (--ppi), list it per file or directory in a manifest (--ppi_manifest), or state an assumption for files like this one (--assume_ppi).
  as_is/1_1.png	null minutiae: the .mnt file is missing
  ```

- `build_warnings.txt`: images that look different from the ridge polarity they were taken to have, `.mnt` files
  without a header, and `.mnt` files without the comment line of the extraction that hold fewer minutiae than
  their first line says (`the first line of the .mnt file announces 50 minutiae, and 46 could be read`). A `.mnt`
  file that ends with that comment line, as the extraction writes it, and announces more minutiae than it holds
  is refused instead: `null minutiae: the .mnt file cannot be read (ValueError: could not convert string to
  float: '#')` in `build_failures.txt`. (Both run, on two `.mnt` files whose first line was changed from 46 to
  50.)

At its end a build with failures prints the first ten of them and what the two kinds of lines mean.

**The exit status:**

| Status | Meaning |
|--------|---------|
| 0 | The table is written, and no file failed |
| 1 | The table is written, and at least one file has a line in `build_failures.txt` |
| 2 | Nothing is written: the build was refused (options that do not fit, an output directory that exists, files that cannot be told apart, other settings than those of the extraction, no image file, no image that could be loaded, pyarrow missing) or failed with an error that the script does not foresee (printed with its traceback). A refusal is printed as `ERROR: ...` on standard error, an option error as `build_sample_table.py: error: ...` after the usage text |
| 120 | The output went into a pipe that was closed before the end (`... \| head -2`): Python could not write its last output and says `Exception ignored ... BrokenPipeError` on standard error. **The table was written all the same**; look whether it exists (`--check`). With Python's output unbuffered (`python -u`, `PYTHONUNBUFFERED=1`) the same build ended with status 2, a traceback with `BrokenPipeError`, and no table |
| other | The build was interrupted: 143 for the signal of `kill`, 129 when its terminal was closed, and after Ctrl-C the status of a process that this signal ended (130 in a shell), after a `KeyboardInterrupt` traceback. Each time the build says `The build did not finish: no table was written, and <table>.tmp<process id> is removed.`, and the temporary directory is removed |

So a build that ends with 0 or 1 has written its table, and one that ends with 2 has not. If no image at all can
be loaded, nothing is written (status 2), and a build is given up after its first 1,000 files if not one of them
could be loaded. Where PyTorch, which the script imports, is missing, Python ends with its own traceback and
status 1 before a build begins.

**A table is never changed, it is built again.** `--samples` must name a directory that does not exist yet, and
it must not lie inside `--dataset_dir` or `--minutiae_dir`. The parts are written into a temporary directory
`<table>.tmp<process id>` next to it, the table is read back as a reader reads it, and only then is the directory
renamed: a table directory that exists holds a whole table. A build that is interrupted removes its temporary
directory; one that is ended without the time for that (`kill -9`, the quit key of the terminal, a machine that
goes down) leaves it behind, and it can be removed then. A build that was started with `nohup` goes on when its
terminal is closed. Nothing is written into `--dataset_dir` or `--minutiae_dir`. The same files and options give
the same checksum on every build. A table that is written although its summary could not be printed says so on
standard error (run, in a process that could not print the name of a file):

```
The table b/e4-table-built-in-ascii is written, but what it holds could not be printed (UnicodeEncodeError: 'ascii' codec can't encode character '\xdf' in position 37: ordinal not in range(128)). To see it: --check b/e4-table-built-in-ascii
```

---

## Checking a table

```bash
python scripts/build_sample_table.py --check data/tables/my-dataset
```

reads the whole table with every rule of the format and prints the summary of the build. On the table of the
twelve files above (`--check a/table`), exit status 0, nothing on standard error:

```
============================================================
Sample table: a/table
  Files listed:              12
  Rows:                      12
  Files that failed:         0
  Rows with minutiae:        0
  Rows with empty minutiae:  0
  Rows with null minutiae:   12 (the table was built without --minutiae_dir)
  Subjects:                  6
  Impressions per subject:   smallest 2, median 2, largest 2
  Parts:                     1
  Checksum (samples_sha256): 85577723f185c1659dfe6547082700c9652528b18dcf224a5644edbf5cbe7f2f
  Failures log:              none
  Warnings log:              none
============================================================

Resolution of the 12 image file(s) of the table:
             8  500 ppi from manifest
             4  1000 ppi from manifest (resampled to 500 ppi)
Ridge polarity:
             8  dark ridges, estimated from the image
             4  light ridges, estimated from the image
```

A table that cannot be read gives exit status 2 and the reason ([Troubleshooting](#troubleshooting)). `--check`
opens no image file: it checks the table, not that the images are still there. The range test and the training
check that at their start.

---

## The range test and the training from a table

**The range test** ([Step 5 of the training guide](training-guide.md#step-5-find-a-learning-rate)):

```bash
python scripts/find_learning_rate.py \
  --dataset_dir data/fingerprints/my-dataset \
  --samples data/tables/my-dataset \
  --max_subjects 2000 \
  --batch_size 64 \
  --output_dir models/lr_range_test
```

**The training** ([Step 6](training-guide.md#step-6-run-the-training)):

```bash
python scripts/train_500k.py \
  --dataset_dir data/fingerprints/my-dataset \
  --samples data/tables/my-dataset \
  --learning_rate 0.0001 --lr_schedule cosine \
  --num_epochs 10 \
  --output_dir models/my-model-v1
```

Both were run on the twelve files with `--dataset_dir a/images --samples a/table` and small numbers: the range
test with `--batch_size 4 --num_steps 12 --min_lr 1e-5 --max_lr 0.01` in place of `--max_subjects 2000
--batch_size 64` (2,000 is the default), the training with `--batch_size 4 --num_epochs 2 --validation_split 0.4`
in place of the options of the learning rate and the number of epochs. `--dataset_dir` is the directory of the
images, with a table as without one; `--samples` is the directory of the table. Every other option means what
it means without a table; where the table changes what one does (`--variant auto` of the range test, the
subjects held out for validation), this section says so below. On several GPUs the training is started with
`torchrun` in the place of `python`, as without a table ([Several GPUs](training-guide.md#several-gpus)), and
every process reads the table (read in the script; not run, the machine has no GPU).

**On a CPU, give a smaller `--batch_size`.** With the default of 64, the variant with minutiae on the forty
files (one batch of all 40 images in the range test, of 36 in the training, one thread) needed more memory than
the runs of this document had, a limit of 14.3 GB for all their processes: the kernel ended the range test after
its first step and the training in its second epoch, their largest process at 11.4 and 10.3 GB (measured;
[the last section](#what-was-run-of-this-document-and-what-was-not)). The other runs here used 4 or 8. Entry 12
of the [troubleshooting of the training guide](training-guide.md#12-the-training-on-a-cpu-ends-with-killed-and-no-error-message)
has the memory of a training without minutiae with 64 and with 16 images per step.

**The options that a table replaces are refused**, also with their default values (`--ridges dark`,
`--num_subjects 50000`): `--minutiae_dir`, `--num_subjects`, `--num_impressions`, `--ppi`, `--ppi_manifest`,
`--assume_ppi` and `--ridges`. The training of the twelve files with four of them added ended with exit status 2,
nothing written, and after the usage text:

```
train_500k.py: error: --samples cannot be combined with --num_subjects (the table lists the images with their subject and impression), --num_impressions (the table lists the images with their subject and impression), --ppi_manifest (the table states the resolution and the ridge polarity of every file), --ridges (the table states the ridge polarity of every file). Leave these options out. What a sample table says is decided when it is built: to change it, build the table again (scripts/build_sample_table.py, see docs/sample-table.md).
```

**Two options exist only with a table** (both scripts refuse them without `--samples`):

- `--tolerate_missing_minutiae`: training images whose row has null minutiae train without a minutia map, with
  the weight 0 in the minutia map loss, instead of stopping the run before it trains.
- `--ignore_minutiae`: train the texture-only variant although the table holds minutiae.

`--samples ""` (an empty text, as `--samples "$TABLE"` gives it where `TABLE` is not set) is an option error
(run):
`train_500k.py: error: --samples was given an empty text, where the directory of a sample table belongs (a
variable of the shell that is not set gives that). Give the directory of the table, or leave --samples out to
find the images by the names of their files.`

**Which variant is trained.** The minutia branch is trained if the table holds minutiae (a list, also an empty
one) for at least one training image, unless `--ignore_minutiae` is given; `--localization` chooses the
localization network as without a table. A table whose training images all have empty lists still trains the
minutia branch, and the start warns that no image then counts in the minutia map loss (read in the script).

**The range test** uses all images of the subjects that it chooses (`--max_subjects` of the subjects of the
table, drawn with `--seed`; 0 for all) and applies the same rule to them with `--variant auto`. On a table that
holds minutiae for some subjects only, it can therefore choose another variant than the training of all
subjects: where its subjects have no minutiae and other subjects of the table have, it says so in a note (read
in the script). A variant with the minutia branch given by name, together
with `--ignore_minutiae`, is an option error (run). The range test checks only the images of its subjects, and a
range test from a table that is refused makes no output directory (without a table it makes the directory before
it looks for the files).

**What the start prints.** The training from the table of the twelve files, up to the model:

```
Reading the sample table a/table...
Created IdentifierSet with 6 subjects and a total of 12 samples.
Created IdentifierSet with 4 subjects and a total of 8 samples.
Created IdentifierSet with 2 subjects and a total of 4 samples.
============================================================
Training DeepPrint with Learned Alignment (LocTex (texture-only))
============================================================
Samples:      a/table (12 rows, 6 subjects)
Dataset:      a/images
Minutiae:     none: the sample table holds no minutiae for the training images (null for all 8)
Images:       resampled to 500 ppi and brought to dark ridges, each file from the resolution and the ridge polarity that the sample table states for it
Canvas:       square of the longer side of each image
Batch size:   4 images per step and GPU
Subjects:     6 total
  Training:   4 subjects
  Validation: 2 subjects held out (seed 42)
Impressions:  2 to 2 per training subject (median 2)
Total images: 12 (8 for training, 4 for validation)
Epochs:       2 (max)
Patience:     3 epochs (early stopping)
Best model:   lowest validation equal error rate
Embedding:    256 values per branch, 256 per print
Optimizer:    Adam, learning rate 0.0001, warm-up over 500 steps, then constant, gradient norm not limited
Alignment:    learned (--localization on), learning rate of the localization network times 0.035
Loss weights: cross entropy 1, center 8
Output:       a/out/table-train
Device:       CPU
============================================================

Validation set: 4 images of 2 subjects (2 to 2 per subject), compared after every epoch in 2 mated and 2 non-mated pairs.

Created 8 training identifiers
Created 4 validation identifiers
Listing the directories of the image files below a/images...
Loaded 12 image files spread over the table: each has the size in bytes and in pixels that its row holds.
Resolution of the 8 training images, as the sample table states it:
             5  500 ppi from manifest
             3  1000 ppi from manifest (resampled to 500 ppi)
Ridge polarity of these images, as the files hold it:
             5  dark ridges, estimated from the image
             3  light ridges, estimated from the image
  These are the sources of the build of the table (scripts/build_sample_table.py): to change one, build the table again.
```

With minutiae in the table the header names the variant with the minutia branch and counts the minutiae of the
training images, and the minutia map gets its weight (a table of 40 files with `.mnt` files; one epoch, exit
status 0):

```
Training DeepPrint with Learned Alignment (LocTexMinu (texture + minutiae))
[...]
Minutiae:     from the sample table, of the 36 training images: 36 with minutiae, 0 with an empty list (an image without minutiae), 0 with null (minutiae not known)
[...]
Embedding:    256 values per branch, 512 per print
[...]
Loss weights: cross entropy 1, center 8, minutia map 0.095
```

With `--ignore_minutiae` the same table trained `LocTex (texture-only)`, and the line said `Minutiae:     not
used (--ignore_minutiae); the sample table holds them, of the 36 training images: 36 with minutiae, ...`. A
table whose build had failed files adds a line `Table build:  12 file(s) failed when the table was built: 12
without a row (they are not in the table), 0 with a row without minutiae; build_failures.txt in the directory of
the table names them` (from a run). The range test prints a line such as `Sample table b/table: 14 rows of 8
subjects; the test uses 7 rows of 4 subjects` (the fourteen files, `--max_subjects 4`) before its line `Range
test for ...`.

**How subjects are held out for validation.** The share and the limit mean what they mean without a table:
`--validation_split` of all subjects, rounded down, at most `--max_validation_subjects`. The subjects are
shuffled in ascending order with `--validation_seed`, and held out are the first of the shuffled list **that
have at least two images**, because validation compares two images of a finger. So **a finger with one image
always trains**: in the fourteen files (fingers of 3, 3, 2, 2, 1, 1, 1 and 1 images), with
`--validation_split 0.25`, no finger with one image was held out with any `--validation_seed` from 0 to 999. If
fewer fingers have two images than the share asks for, these are held out and the start says so:

```
Note: 5 subjects were to be held out, but only 4 subjects of the table have two or more impressions, which validation needs: it compares two impressions of a subject. These 4 are held out, and every subject with one impression trains.
```

Fewer than 2 such fingers is an error of the options (`--validation_split 0` trains without validation). For
a complete grid this is the split of a training without a table: on the twelve files both modes held out the
same two fingers, those of the files `2_*.png` and `4_*.png`.

**The validation set of an uneven table.** A held-out finger brings all its images, and validation compares
all pairs of them: a finger with n images gives n (n - 1) / 2 mated pairs, so the list of pairs grows with the
square of the images of one finger, and a finger with many images can dominate the validation. The start prints
the images and pairs of the validation set (above) and warns when it has more than 50,000 images or 275,000
pairs, which is what `--max_validation_subjects 5000` stands for with 10 images per finger. (Without a table the
training warns above 5,000 validation subjects instead.)

**Continuing a training.** The same command again continues the training, as without a table
([Step 7](training-guide.md#step-7-continue-a-training-or-start-again)). In place of the image preparation and
the numbers of subjects and impressions, the training stores the checksum of the rows of the table, and
continues **only with a table of the same rows**; where it lies and how its parts are split does not count (a
copy of the table in another directory continued a training of the twelve files). `--ignore_minutiae` changes the
variant, which cannot change in a running training. A table with other rows is refused with exit status 2, and
nothing in the output directory changes:

```
ERROR: The training in a/out/table-train read a sample table with other rows than the table of this command (a/table-11): 12 rows of 6 subjects then, 11 rows of 6 subjects now, and the checksums of the rows (samples_sha256) differ. A training continues only with a table that holds the same rows; where the table lies and how its files are split does not count.
a/out/table-train holds a training with other settings, and the same --output_dir continues that training:
  preparation: stored {"samples_rows": 12, "samples_sha256": "85577723f185c1659dfe6547082700c9652528b18dcf224a5644edbf5cbe7f2f", "samples_subjects": 6, "target_ppi": 500.0}, now {"samples_rows": 11, "samples_sha256": "eda77dd5f751937d55779dec52b200a3838660a8a34252925c20a63b8dc70298", "samples_subjects": 6, "target_ppi": 500.0}
These settings cannot change in a running training. To train with the new settings, use a new --output_dir. To continue the stored training, use its settings (they are in a/out/table-train/training_settings.json). --allow_changed_settings continues anyway.
```

A training without a table is not continued with one, nor the reverse; each refusal begins with a sentence that
says so (`ERROR: The training in ... read its training set from a sample table (--samples a/table), and this
command reads none: ...`, and `ERROR: The training in ... read no sample table: ...`).

**Using the model afterwards.** `load_trained_model` of `scripts/train_500k.py` takes the table of the images to
embed as `samples` ([Step 9 of the training guide](training-guide.md#step-9-extract-embeddings)):

```python
import sys

sys.path.insert(0, "scripts")  # the scripts directory of the repository
import train_500k
from flx.data.dataset import Dataset
from flx.data.embedding_loader import EmbeddingLoader

extractor, image_loader = train_500k.load_trained_model(
    "models/my-model-v1",               # the --output_dir of the training
    "data/fingerprints/my-test-set",    # the images to embed
    samples="data/tables/my-test-set",  # a sample table of these images
)
dataset = Dataset(image_loader, image_loader.ids)
texture_embeddings, minutia_embeddings = extractor.extract(dataset)
embeddings = EmbeddingLoader.combine_if_both_exist(texture_embeddings, minutia_embeddings)
print(f"{len(embeddings.ids)} embeddings of {embeddings.embedding_size} values")
```

Run with the paths of a table of 40 files with `.mnt` files, and the model of a training of one epoch from that
table, it printed `40 embeddings of 512 values`.

- Every image is prepared as its row says, then as the training prepared its images (canvas, no augmentation).
  The table need not be the table of the training: any table of the images to embed, built at 500 ppi.
- A model that was trained from a table **cannot be loaded without `samples`**: its settings hold no resolution
  options, and `load_trained_model` raises `ValueError: The training in ... read its images from a sample table
  (--samples ...), so its settings do not say how the image files of a directory are to be prepared. Pass
  samples= with the directory of a sample table that describes the images below dataset_dir ...`.
- A model that was trained without a table can be given a table too.
- **A verification benchmark** of these embeddings
  ([Step 10 of the training guide](training-guide.md#step-10-run-a-verification-benchmark)) takes its pairs from
  `train_500k.create_table_validation_benchmark(image_loader.ids)`, which pairs the impressions that each finger
  has. `create_validation_benchmark(image_loader.ids, num_impressions=...)` of Step 10 pairs the impressions 0 to
  n - 1 of every finger, so it serves only a complete grid of impressions numbered from 0. Run: on a table of 8 images of 4
  fingers with 3, 2, 2 and 1 images, numbered as a sample list gave them (the impressions 10, 20 and 30 of one
  finger, for example), Step 10 as written stopped with a `KeyError`, and with
  `create_table_validation_benchmark` it printed both error rates; on a table of a grid of 6 fingers with 2
  images each, both functions gave the same 36 pairs.

**Repeating a run.** Neither script seeds Python's `random` module, from which the augmentation draws, and the
training seeds nothing: two trainings with the same command start from other weights and give other losses,
with a table and without one. The range test seeds torch and the order of the batches with `--seed`; with
worker processes (its default) it repeated itself, without them (`--num_workers 0`) it did not. To compare the
trainings of the two modes, both were started through these lines, which seed `random`, numpy and torch before
the script runs (`python seeded.py 0 scripts/train_500k.py ...`):

```python
import os
import random
import runpy
import sys

import numpy as np
import torch

seed, script = int(sys.argv[1]), sys.argv[2]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
sys.argv = sys.argv[2:]
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")
```

---

## What is the same and what differs from a training without a table

**The same files give the same training.** For files that both modes can read (`.png` files named
`<subject>_<impression>`, a complete grid, `.mnt` files whose header is the size of the prepared image), a table
gives the same split, the same validation pairs, the same number of classes, and the same image and minutia map
tensors, bit for bit, as the directories it was built from. This was compared:

- by the tests of the work on the table (`tests/test_train_500k_samples.py` and
  `tests/test_learning_rate_finder_samples.py`, through `main()` of both scripts with a model that does
  nothing), with canvas sizes of 0, 512 and one smaller than a print, and with files chosen to show the faults
  that would make the modes differ (prints at 500, 1000 and 385 ppi, light ridges estimated from the image,
  `.mnt` files with no minutiae);
- by the authors and reviewers of that work with generators of their own (1,085 generated sets for the training,
  300 for the range test; read in their notes, not repeated for this document);
- by runs on the twelve files: with `random`, numpy and torch seeded with 0, a training without a table (run
  twice) and one from the table wrote the same `model.pyt`, `best_model.pyt` and `log.json`, byte for byte;
  the range test, started without these lines, wrote the same `lr_range_test.csv` and plot in both modes.

**"The same files" is not "the same directory".** A table built from a directory holds every image file with
one of the six endings and every subject; a training without a table reads only the `.png` files up to
`--num_subjects` and `--num_impressions`. And a training without a table finds a `.mnt` file by its name
anywhere below `--minutiae_dir`, while the builder looks only at the path of the image below it: with the
images in subdirectories and the `.mnt` files in one flat directory, every row gets null minutiae (exit status
1 and a line per file in the failures log).

**What differs:**

- **What is stored.** `training_settings.json` holds, under `preparation`, the checksum, the rows and the
  subjects of the table and the target resolution in place of the six resolution settings, `null` for
  `num_subjects`, `num_impressions`, `ppi_manifest` and `minutiae_dir`, and `samples`, the path of the table as
  it was given. From the training of the twelve files:

  ```json
    "preparation": {
      "samples_rows": 12,
      "samples_sha256": "85577723f185c1659dfe6547082700c9652528b18dcf224a5644edbf5cbe7f2f",
      "samples_subjects": 6,
      "target_ppi": 500.0
    },
    "samples": "a/table",
  ```

  `data_preparation.json` holds the same four keys of `preparation`, the canvas size and `ppi_manifest`
  (`null`).
- **The start checks.** Every process, also on several GPUs, reads and validates the whole table, lists every
  directory of the images once and stops if files of the table are missing, stops for images whose prepared
  size is below the 41 pixels per side that the binarisation needs, and, for a variant with minutiae, stops for
  training images with null minutiae unless `--tolerate_missing_minutiae` is given. Then it **loads up to 100
  images spread over the table** itself, not in worker processes, before the model is built, and compares each
  with its row (size in bytes and in pixels): another `--dataset_dir`, or files that changed since the build,
  stop the run here and not in the first epoch. A training without a table loads no image before it starts its
  worker processes; these 100 are loaded with OpenCV working in the calling thread, so that the worker processes
  are still forked from a process without the threads of OpenCV's pool, as without a table.
  Only the first process prints. The block `Resolution of ... training images` counts every training image,
  where a training without a table looks at up to 1,000.
- **What the start cannot notice.** A file that changed and kept its size in bytes and in pixels. A changed file
  that is not among the 100 images of the start is found when the training loads it, and stops the run with a
  traceback whose last line is the message of the table's image loader.
- **Messages that name the file.** Without a table, an image below 41 pixels per side, or one whose header
  OpenCV refuses with an error of its own, stops the training without a file name (left as it is). With a table
  the start names the rows of all images below 41 pixels per side (it takes their sizes from the table), and an
  image whose header OpenCV refuses if it is among the 100 images of the start; such an image elsewhere in the
  table stops the training when it is loaded, with a traceback whose last line names the file and its row. Run:
  in a table of 102 BMP files, one whose header was damaged after the build, its size in bytes kept, and that
  was not among the 100 images of the start: the start passed, and the training stopped in its first epoch with
  exit status 1 and `flx.data.sample_table_loader.UnreadableImageError: <...>/26_1.bmp: Could not read the file
  as an image: OpenCV stopped with an error of its own (...). The sample table names the file as '26_1.bmp'
  (subject 25, impression 0), ...` as the last line.
- **The memory of the table in every process.** Every GPU process holds the arrays of the whole table and an
  identifier object per row. Its forked worker processes share them; a worker that is started in another way
  (the default of Python on Linux from 3.14 on, by its documentation) gets a copy, and the start prints a
  warning then. The figures are in the next section.

---

## Size, time and memory

The figures of the table, of its reading, of the start, of the worker and of `--check` are of **one million
rows** (100,000 subjects with 10 images each, 46 minutiae per image), with the conditions beside them; one figure
of the worker is of a table of 4,000 rows made by the same functions. These tables are of **synthetic rows** and
of image files that are **empty**, but for 600 hard links to one example image of 292x392 pixels (the 100 images
that the start loads among them); a training would read real images. The figures of the builder are of 800 and
8,000 PNG files, copies of the 100 example images as they are or enlarged, each with a `.mnt` file. The files
were **in the page cache** unless a row says otherwise (this machine drops the page cache by itself a few
minutes after the last run, and the runs right after a drop are named). The speed of the CPU of this machine
**varied between series of runs by up to a third** with the same work, so a time is the range over all runs of
its kind, and where two harnesses took it, over both. torch ran with its default of 4 threads, or with 1, which
made no difference beyond that spread.

| What | Figure | Mark and conditions |
|------|--------|---------------------|
| The table, its files | 565,522,603 bytes in 4 parts, **565.5 bytes per row**; 565,538,816 bytes of the disk | measured; the same in both harnesses. With `x` and `y` as whole pixels 296.3 bytes per row (measured): pyarrow stores a column with few distinct values in fewer bytes |
| The same minutiae as `.mnt` files | 1,763,752,926 bytes, 1,763.8 per file on average (1,738 to 1,785); 4,124,110,848 bytes of the disk (4,096 per file on this file system, and 28 MB of directories) | measured; written by the function of the extraction. 3.12 and 7.29 times the table (calculated) |
| Reading and checking the table (`read_samples`) | **1.4 to 2.2 s**; the process peaked at 1,022 to 1,031 MB | measured, 12 runs in a process without PyTorch over both harnesses (2.16 s right after a drop of the page cache); the time holds the import of pyarrow |
| The same in a process with PyTorch, as at the start of a training | 1.6 to 2.2 s; peak 1,311 to 1,312 MB, 966 to 967 MB above the memory before (calculated) | measured, 6 runs, one harness |
| The arrays of the table in memory | 635.0 MB, 552 MB of them the minutiae | measured (the sum of the arrays; the minutiae calculated, 46 million of 12 bytes); an identifier object per row adds 122 to 123 MB (calculated from the memory before and after) |
| Start of `scripts/train_500k.py` to the call of `fit` (every check passed, the settings stored, the model not built) | **without a table 21.1 to 30.4 s, from the table 7.7 to 10.5 s**; without a table 2.46 to 3.18 times as long | measured, 23 and 20 runs over both harnesses; the ratio calculated pair by pair, over the 12 pairs in which the two modes ran in turns. Both modes trained the variant with minutiae: without a table on 1,000,000 image and 1,000,000 `.mnt` files, with `--ppi 500`. Without `--ppi` a start without a table reads the headers of up to 1,000 images, which was not measured |
| Peak memory of the main process at that point | **without a table 1,198 to 1,205 MB, from the table 1,330 to 1,339 MB**, 133 to 134 MB more | measured (the difference calculated); the lowest of each in starts right after a drop of the page cache |
| Up to the point where the first batch would be asked for | without a table 24.8 to 31.3 s and 1,269 MB, from the table 9.4 to 11.4 s and 1,362 to 1,363 MB | measured, 6 runs of each, one harness (the start forks its 4 worker processes at the next request) |
| What one forked worker process adds | about **51 MB** of private memory from the table (82 to 88 MB without a table) after 384 training items | measured, 3 runs of each, **with batches of 16 and one worker**. With the batches of 64 of a training it holds more: 52 to 68 MB after 128 items against 30 to 31 MB with 16, on a table of 4,000 rows (measured). Several workers at once were not measured |
| `--check` of the table | 3.9 and 4.0 s; 4.5 s for a run 13 s after a drop of the page cache | measured, 3 runs of the program; largest process 1,307 to 1,311 MB (calculated from MiB) |
| The builder, 292x392 PNG files at 500 ppi, with `.mnt` files | **536 to 656 files per second with one process, 1,742 to 2,117 with four workers** | measured on 8,000 files by two methods (the loading timed inside the process, and two runs of the whole program, of 800 and 8,000 files, against each other); one run that read its files from the disk: 471 with one process |
| The builder, 584x784 PNG files at 1000 ppi, resampled | 280.5 to 295.4 files per second with one process, 1,026 to 1,110 with four workers | measured as above |
| 1,000,000 such files with four workers | **8.4 to 9.1 minutes** of loading for files of 292x392 at 500 ppi, **15.0 to 16.0 minutes** for files of 584x784 at 1000 ppi (5,000,000: 42 to 46 and 75 to 80 minutes) | calculated from the rates of four workers on 8,000 files by the first method, the loading timed inside the process (1,824 to 1,995 and 1,042 to 1,110 per second); from the rates of both methods in the two rows above, 7.9 to 9.6 and 15.0 to 16.2 minutes (calculated). Listing, writing and reading back a table of that size were not measured. A build also has 2.1 to 3.3 s that do not grow with the files (calculated from measured times) |
| The largest process of a build of 8,000 files | 522 to 528 MB | calculated from 498 to 503.9 MiB (measured) |

**Where the time of a start goes.** Without a table the start lists the image and the `.mnt` directory and makes
an identifier object per file of each (11.2 to 13.9 s), and lists the `.mnt` directory once more for the minutia
maps (6.1 to 7.7 s). From the table it reads the table with its identifiers (2.8 to 3.7 s) and looks for every
file in the listings of its directories (1.3 to 1.4 s); the 100 images of the start took 0.08 to 0.09 s (all
measured, one harness).

**An earlier figure.** [large-datasets.md](large-datasets.md#above-about-one-million-subjects) gives 22 seconds
and 1.6 GB for the start of `scripts/train_500k.py` on 1 million images and 1 million `.mnt` files. That is the
record of another version of the script, which scanned the image directory twice, one run on empty files; it
stands as that document gives it (it does not say whether its GB are 10^9 or 2^30 bytes), and it is not a
figure of these runs.

**Not measured:** anything on a GPU; real prints and real minutiae; the model built, a real first batch, a whole
epoch; the 4 worker processes of a training at once; a start or a build from a cold disk (beyond the few runs
right after a drop of the page cache named above); more than one million rows at the start; a build of a
million files and its memory; a start method other than fork.

---

## Troubleshooting

Each entry begins with the message as the script printed it in a run. Paths that stood there as long absolute
paths are written as `<...>`.

#### 1. `ERROR: A sample table is read and written as Parquet files, which needs the package pyarrow ...`

pyarrow is not installed in this environment. See [Installing pyarrow](#installing-pyarrow).

#### 2. `ERROR: 14 of the 14 image files below b/images cannot be put into a sample table: ...: it is not named <subject>_<impression> with whole numbers from 1; ...`

Without `--sample_list` every image file below `--dataset_dir` must be named `<subject>_<impression>`. Rename
the files, move files that are no training images out of `--dataset_dir`, or say in a sample list which finger
each file shows. Nothing was loaded or written.

#### 3. `ERROR: 1 pair(s) of image files have the same subject and impression: subject 10, impression 1: line 8 (...) and line 9 (...)`

Two lines of the sample list give two files the same pair of numbers. Give every file a pair of its own.

#### 4. `ERROR: b/table exists. A table is not changed or overwritten: give a directory that does not exist yet as --samples, or remove the old one first. ...`

A table is built again into a new directory, never changed. To look at the old one: `--check`.

#### 5. `build_sample_table.py: error: --samples b/images/table lies in --dataset_dir b/images. ...`

The table must lie outside of the image directory and of the minutiae directory.

#### 6. `ERROR: <minutiae_dir>/extraction_settings.json holds other settings than this build uses: ppi is null there and 500.0 here; ...`

The build was given other resolution options than the extraction (here `--ppi 500` where the extraction had a
manifest). The message goes on: `The .mnt files of <minutiae_dir> fit images that are prepared as the extraction
prepared them: give this script the options that the extraction was given (--ppi, --ppi_manifest, --assume_ppi,
--ridges). To build all the same, pass --allow_other_extraction_settings: every .mnt file is accepted or
refused by its own checks.` With that option the same build was written, with a note in place of the
refusal.

#### 7. `ERROR: None of the 4 listed files could be loaded as an image, so no table was written: 1_1.png: no row: UnknownResolutionError: Resolution unknown. ...`

Not one image could be loaded, nearly always because no resolution is known for the files (give `--ppi`,
`--ppi_manifest` or `--assume_ppi`) or because `--dataset_dir` is not the directory of the images. A build of
more files is given up after its first 1,000 if none of them loads. Exit status 2, nothing written.

#### 8. `null minutiae: the .mnt file is missing`, and `WARNING: no row of the table holds minutiae, although the build read a minutiae directory.`

Lines of `build_failures.txt` (status 1: the table is written). The image is in the table without minutiae,
because its `.mnt` file is not where the builder looked for it, or was refused. The builder looks at the path of
the image below `--minutiae_dir` with the ending `.mnt`, where `scripts/extract_minutiae.py` writes it; with
another layout of the `.mnt` files every row gets null minutiae, and the warning says so. A `.mnt` file is
refused if its header is not the size of the prepared image, or if its comment line names another source size
or polarity than the build: then the extraction and the build were given different resolution options, or the
image file was replaced. Correct the cause and build again into a new directory; a training of rows with null
minutiae needs `--tolerate_missing_minutiae` (entry 13).

#### 9. `ERROR: <table> is not a whole table: it holds 1 of its 2 parts; missing: part-00001-of-00002.parquet. ...`

A part is missing (also `... holds no part of a sample table` for a directory without parts). A table is only
read if all parts of its build are there: copy the missing parts from where the table was built, or build it
again.

#### 10. A part that cannot be read: `... cannot be read to its end (...)`, `... does not hold the 12 rows any more that its metadata stated ...`, `... cannot be read as a Parquet file (...)`

The part was damaged or replaced after it was written (a part gets its name only when it is complete). From
runs with one byte changed in each of three copies of a part:

```
ERROR: <table>/part-00000-of-00001.parquet cannot be read to its end (OSError: could not verify page integrity, CRC checksum verification failed for page_ordinal 0); 0 of its 12 rows were read before that. A part gets its name only when it is complete, so the file was damaged later, or it changed while the table was read. Copy the part again from where the table was built, or build the table again (scripts/build_sample_table.py).
ERROR: <table>/part-00000-of-00001.parquet does not hold the 12 rows any more that its metadata stated when the table was opened: the file changed while the table was read, or it was damaged. Read the table again when nothing writes to its directory; if it is refused again, copy the part again from where the table was built, or build the table again (scripts/build_sample_table.py).
ERROR: <table>/part-00000-of-00001.parquet cannot be read as a Parquet file (ArrowInvalid: Invalid base64 input: character is not valid base64 character). A part gets its name only when it is complete, so the file was damaged or replaced later. Build the table again (scripts/build_sample_table.py).
```

The first two: copy the part again from where the table was built, or build the table again. The third: build
the table again. (In a process whose file names are not UTF-8 the third once had another cause; since the
current version that case no longer arises, see entry 17.)

#### 11. `ERROR: 1 of the 14 image files that the sample table names are missing below <image directory> (--dataset_dir).` (the range test: `... image files that are needed of the sample table are missing ...`) and `ERROR: <file> is not the file that its row of the sample table describes: ...`

The start of a training or a range test found that image files are missing, or that a file has another size in
bytes or in pixels than its row (it changed after the table was built, or `--dataset_dir` is another
directory). The range test checks only the images of the subjects that it uses, and its message says so: for a
table of 12 files of 6 fingers with one of the files removed, the range test said `ERROR: 1 of the 12 image files
that are needed of the sample table are missing below <image directory> (--dataset_dir).` and the training from
the same table `ERROR: 1 of the 12 image files that the sample table names are missing below <image directory>
(--dataset_dir).` (run; both exit status 2, no output directory). From a run in which one of the fourteen files
was replaced by another image of the same size:

```
ERROR: <image directory>/desk_a/p0007_left_index_01.png is not the file that its row of the sample table describes: the table holds 32,063 bytes as the size of the file, the file has 32,106 bytes. The row is 'desk_a/p0007_left_index_01.png' (subject 3, impression 1) for the image directory <image directory>. The file changed after the table was built, or this is not the image directory the table was built from. The minutiae of the row would not fit this file. Give the image directory of the table, or build the table again from the files as they are now (scripts/build_sample_table.py).
```

Give the image directory that the table was built from, put the files back, or build the table again from the
files as they are now. A changed file that is not among the 100 images of the start stops the run when it is
loaded, with a traceback whose last line is this message (read in the notes of the work on the table; not run
for this document).

#### 12. `ERROR: The training in <output_dir> read a sample table with other rows than the table of this command (...)`

The output directory holds a training of another table (see **Continuing a training** in
[The range test and the training from a table](#the-range-test-and-the-training-from-a-table)). Continue it
with a table of the same rows, or train into a new `--output_dir`. The same holds for the refusals `... read
its training set from a sample table ..., and this command reads none` and `... read no sample table`.

#### 13. `ERROR: The sample table holds no minutiae (null) for 3 of the 36 training images.`

The table holds minutiae for some training images and not for others, and the variant with the minutia branch
is trained. The message goes on (from a table whose build missed three `.mnt` files):

```
The first ones: 'sensor_a/1_1.png' (subject 0, impression 0); 'sensor_a/5_2.png' (subject 4, impression 1); 'sensor_a/9_1.png' (subject 8, impression 0). A row is without minutiae if the .mnt file of its image was missing or refused when the table was built (build_failures.txt in the directory of the table says which and why). Extract the minutiae of these images and build the table again (scripts/build_sample_table.py), or pass --tolerate_missing_minutiae: these images then train without a minutia map, with weight 0 in the minutia map loss. --ignore_minutiae trains the texture-only variant instead.
```

With `--tolerate_missing_minutiae` the same command trained, and its start said `Note: training images without
minutiae in the sample table: 3 of 36. They train without a minutia map (--tolerate_missing_minutiae), with
weight 0 in the minutia map loss, and count in the other parts of the loss as every image does.` The range
test checks the images that it uses with the same function and the same words (read in the script).

#### 14. `ERROR: 1 of the 5 images of the sample table are too small for the binarisation of the training, which needs 41 pixels in width and in height of the image prepared at 500 ppi.` (the range test: `... images that are needed of the sample table are too small ...`)

An image of the table is narrower or lower than 41 pixels at 500 ppi (in this run, an image of 30x40 pixels in a
table of five). The message goes on with the first rows, the error of the binarisation for such an image, and:
`Check the resolution of these files: a resolution that is stated too high makes the prepared image too small.
Then build the table again with their right resolution or without them (scripts/build_sample_table.py).` The
range test checks the images of its subjects, and from such a table it said `ERROR: 1 of the 5 images that are
needed of the sample table are too small for the binarisation of the training, ...` (run; exit status 2, no
output directory, as the training). Without a table such an image stops the training with an error of PyTorch
that names no file (read in the notes of the work on the table).

#### 15. `train_500k.py: error: --samples cannot be combined with --num_subjects (...), ...`

An option that the table replaces was given (the full message is in
[The range test and the training from a table](#the-range-test-and-the-training-from-a-table)). Leave it out:
what it said is in the table. To change it, build the table again.

#### 16. A training or a range test stands still after its start, and standard error repeats `OSError: AF_UNIX path too long`

The path of the temporary directory (`TMPDIR`) has more than 75 bytes. A worker process of a torch DataLoader
hands a batch over through a socket that Python makes below the temporary directory, and the path of a socket
may have 107 bytes on Linux; the worker drops the batch and the main process waits for it without end. Run with
a `TMPDIR` of 139 bytes: the range test from a table with two worker processes made no step in
180 seconds, its last line of output `Note: the test needs 48 images and sees each of the 12 images more than
once.` and three tracebacks of 26 lines that end with

```
OSError: AF_UNIX path too long
```

on standard error; the training from the table printed nothing after `Training:`. **It is not of the sample
table: the range test without a table stood still in the same way.** With `TMPDIR=/tmp` the three commands
ended; with a path of 75 bytes, or without worker processes (`--num_workers 0`), the range test from the table
ended; with 76 bytes it stood still. Remedy: a `TMPDIR` with a shorter path. The test suite has tests with worker processes that wait in
the same way (see the last section).

#### 17. Paths with letters outside ASCII, in a process whose file names are not UTF-8

Python takes the encoding of file names from the locale. In a process whose locale is `C` with
`PYTHONUTF8=0 PYTHONCOERCECLOCALE=0` (ASCII), or an 8-bit locale such as `de_DE.ISO-8859-1`, paths with
letters outside ASCII behave as follows (run with a directory named `größe`):

- **The table** may lie at such a path: in both kinds of process, a build into such a directory, `--check`, the
  training and the range test from the table ended with status 0 and the checksum of the same table in a UTF-8
  process, with the path given whole or relative.
- **The images** need a process whose file names are UTF-8. An image directory at such a path ends the process
  with a **segmentation fault** in OpenCV at the first image in an ASCII process (`Segmentation fault`, exit
  status 139 in a shell; a build leaves its temporary directory behind), also in a training without a table,
  whose worker processes die of it; in an ISO-8859-1 process the build and the training are refused with causes
  that are not the cause (`None of the 12 listed files could be loaded as an image`, `Could not read the file
  as an image ... Put the file back as it was when the table was built`). An image *name* with such a letter:
  the build goes on without the file (exit status 1, a line `no row: UnicodeEncodeError ...` or `no row:
  FileNotFoundError ...` in the failures log, although the file is there), and the training from a table that
  holds it is refused at its start, in an ASCII process in words:

  ```
  ERROR: The image file 'desk_b/single/p0001_zeigefinger_gro\xdf.png' (subject 20, impression 1) of the sample table cannot be looked for below <image directory>: this process writes the names of files in the encoding ascii, which cannot express the path ('ascii' codec can't encode character '\xdf' in position 21: ordinal not in range(128)). Start the script in an environment whose file names are UTF-8, for example with LC_ALL=C.UTF-8 or with PYTHONUTF8=1.
  ```

  and in an ISO-8859-1 process as a missing file.
- **Remedy:** a process whose file names are UTF-8, `PYTHONUTF8=1` or a UTF-8 locale such as
  `LC_ALL=C.UTF-8`. Run with `PYTHONUTF8=1` in the ASCII process: `--check`, the training and the range test
  from a table at such a path, and a training with the image directory at such a path, ended with status 0.
- A path whose bytes are no UTF-8 at all cannot be printed by the scripts in a strict UTF-8 locale such as
  `en_US.UTF-8`: `--check` ends with a traceback; `PYTHONUTF8=1`, or a relative path or a link without those
  bytes, helps (found by a reviewer of the work on the table; not one of the runs of this document).

---

## What was run of this document, and what was not

On the machine of [What this document rests on](#what-this-document-rests-on). The runs on the twelve and the
fourteen files, and those of entry 16 of the troubleshooting, were made on the code before its last change,
which lets the schema module open the files of a table itself, so that a table may lie at a path with letters
outside ASCII in any process. The runs of entry 17 were made with that change, and printed for the table what
the earlier runs had printed in a UTF-8 process; the measurements, the runs on the forty files and the runs
that were added for the corrections of this document (*later* in the table) were made with it as well.

| Section | What | Run? |
|---------|------|------|
| 3 | `pip install -e ".[table]"` with network access | **Run** in a new environment made as in Part A1 of installation.md: pyarrow 26.0.1 |
| 3 | `pip download "pyarrow>=17.0"`; `pip install --no-index --find-links=... --no-build-isolation -e ".[table]"` without network (`unshare -n`), and the same without `--no-build-isolation`; `pip install --no-index --find-links=... "pyarrow>=17.0"` in a new environment | **Run** on one machine, with a directory that held only pyarrow: one file of 55.0 MB; the install with the option worked, the one without it failed as described, and worked when `setuptools`, `wheel` and `packaging` were added; pyarrow 26.0.1. A second machine and the transfer were not part of it |
| 3 | The scripts without pyarrow | **Run:** `--check`, a build, the range test and the training with `--samples` ended with exit status 2 and the message; the range test without a table ran |
| 3 | pyarrow 16.1 and the damaged parts | Not run for this document (read in the module) |
| 4 | The few lines of pyarrow that print the descriptions; a part read with pyarrow alone | **Run** |
| 4 | The lines that read the rows of a whole table; pandas and pyarrow on the directory of a table with both logs and of one without | **Run** *later*, with pandas 3.1.1: the rows; the directory with logs refused as quoted, the one without read |
| 4 | What a reader refuses | **Run** for the cases of the troubleshooting (a part missing, a directory without parts, three copies of a part with one byte changed) and for every byte of one part changed once on its own. The other rules are covered by the tests of the format (`tests/test_sample_table.py`), not by runs of this document |
| 5 | Builds from a directory (the twelve files, a manifest, `--ridges auto`) and from a sample list (the fourteen files, `--assume_ppi 500`, and without a resolution option: exit status 1); `--workers 2` and `4` | **Run** |
| 5 | Builds with a minutiae directory: `.mnt` files that are accepted (the forty files from the directory and from a sample list, and 800 and 8,000 files in the measurements), three of them missing (exit status 1), other resolution options than the extraction (refused) and the same with `--allow_other_extraction_settings`, a minutiae directory without `.mnt` files; a table into a directory whose parents did not exist; the output into a pipe that was closed early, with and without `PYTHONUNBUFFERED=1`; two `.mnt` files whose first line announces 50 of their 46 minutiae, with and without the comment line of the extraction | **Run.** The `.mnt` files were written by the function of the extraction with 46 minutiae per image drawn at random. The two with 50 *later*: exit status 1, the one with the comment line refused, the other a warning |
| 5 | Refusals of the builder: names that do not fit, two files with one pair of numbers, an output directory that exists, a table inside the image directory, no image that loads | **Run** |
| 5 | A build of 3,000 files interrupted after 4 seconds by the signals of Ctrl-C (SIGINT), of `kill` (SIGTERM) and of a closed terminal (SIGHUP) | **Run** *later* (`timeout --preserve-status`): exit status 130, 143 and 129, each with the line that the build did not finish, the `KeyboardInterrupt` traceback after SIGINT only; no table and no temporary directory left |
| 5 | A `.mnt` file that is refused for another size or polarity, or that has no header; a sample list with quoted paths, further columns, or a path that leaves the image directory; JPEG files | Not run for this document. The tests of the builder cover them |
| 6 | `--check` of whole tables, of the same rows in two parts, and of damaged tables | **Run** |
| 7 | The range test and the training from a table without minutiae; the options that a table replaces; the start; holding out with fingers of one image (all 1,000 seeds from 0 to 999) and a share that cannot be met; continuing: the same command, one more epoch, after Ctrl-C, another table, a copy of the table, with and without a table; `load_trained_model` with the table of the training, with that of other images, without a table, and for a model trained without a table | **Run** on the twelve and the fourteen files, with `--batch_size 4` and 1 to 4 epochs or 12 steps |
| 7 | The variant with the minutia branch from a table (training and range test), `--tolerate_missing_minutiae`, `--ignore_minutiae`, a variant with minutiae together with `--ignore_minutiae` (refused), and the lines of `load_trained_model(..., samples=...)` of this section | **Run** on the forty files, `--batch_size 4`, one epoch or 12 steps |
| 7 | The three commands of the docstring of `scripts/train_500k.py` as they stand there (the build, the range test and the training with their default options) | **Run** on the forty files. The build ended with exit status 0. The range test (batch size 64, so one batch of all 40 images, the variant with minutiae, one thread of PyTorch) passed its checks and made its first step in 69 seconds; then the kernel ended it for lack of memory, because the runs were limited to 14.3 GB (its main process had 11.4 GB at its peak). The training passed its checks, trained and saved its first epoch, and was ended in the same way in its second (10.3 GB at its peak). With the variant with minutiae on a CPU, a batch of all 40 images (36 for the training) needed more memory than these runs had |
| 7 | The commands of this section as they are written (a batch size of 64, the learning rate options, ten epochs) | Not run as written; see the row above |
| 7 | Step 10 of the training guide on the embeddings of a table: as written, and with `create_table_validation_benchmark` | **Run** *later*: a model of one epoch from a table of a grid of 6 fingers (`--batch_size 4`), the 8 images of a table of 4 other fingers numbered from a sample list: as written a `KeyError`, with `create_table_validation_benchmark` both error rates; on the grid both functions gave the same 36 pairs |
| 7, 10 | `--samples ""`; a training from a table with an image of 30x40 pixels; the range test from such a table; the range test and the training from a table of 12 files with one of them removed | **Run:** all refused with exit status 2 and the messages quoted, nothing written (the last three *later*) |
| 8 | The same training and the same range test in both modes | **Run** on the twelve files: the trainings, seeded from outside, the same model files and log byte for byte; the range test, started without the seeding, the same CSV file and plot |
| 8 | A file whose header was damaged after the build (its size in bytes kept) and that is not among the 100 images of the start | **Run** *later*, in a table of 102 BMP files, `--batch_size 8`: the start passed, the training stopped in its first epoch with exit status 1 and the message of the loader as the last line |
| 8 | A file of another size that is not among the 100 images of the start; worker processes that are not forked | Not run |
| 9 | Every figure | **Run**, but those marked calculated or read: the measurements at one million rows were taken by one harness, and their main figures (the sizes, the reading, the starts to `fit`, the builder) were taken again by a second one written on its own; the reading in a process with PyTorch, the later stop point, the worker and `--check` were not |
| 10 | The messages | **Run**, each from a run that is named there or in the rows above |
| 10 | Entry 17 with the code of this version | **Run** with a table, and an image directory, at a path with `größe`, in processes whose file names are ASCII and ISO-8859-1; the last point (bytes that are no UTF-8) is a reviewer's finding |
| all | Anything on a GPU; `torchrun`; a start method other than fork; real prints and real minutiae; any run from which an accuracy could be read | Not run |

**How the scripts were started.** The scripts were started with `PYTHONPATH` set to the repository, in the
place of the installed package, with one exception: in a copy of the repository with `pip install -e
".[table]"` and without `PYTHONPATH`, `--check`, the range test and the training of the twelve files ran and
wrote the files of the runs with `PYTHONPATH` byte for byte (the range test's CSV file, and the model files and
log of the seeded training).

**The tests.** The repository has 952 tests. In the environment of these runs, which holds every optional
package, they ended with `952 passed, 1 warning`, nothing skipped, with one thread of PyTorch (the warning is
PyTorch's): after the measurements, and again with this document and its changes to the texts of the scripts.
In a new environment made as in Part A1 of installation.md, with the extras `dev` and `table`: `951 passed,
1 skipped` (the skipped test needs scikit-image); without the extra `table`: `396 passed, 170 skipped`, because
the tests of the table skip themselves without pyarrow (three of their files count as one skipped test each). `tests/test_sample_table_optional.py` runs in every environment; without
PyTorch, pyarrow and pandas it gave `5 passed, 4 skipped`, the four need PyTorch. Some tests skip themselves
where the system differs from this one (none did here): where the file names of the process are not UTF-8,
where its encoding has a character for every byte (ISO-8859-1), where `localedef` makes no ISO-8859-1 locale,
where the file system takes no name that is not UTF-8, where there is no `/proc`, where OpenCV has one thread
whatever it is given, where the C locale gives UTF-8 file names, where two pairs of numbers have other hashes,
and where no socket can be made below the temporary directory. **For whoever runs the suite:** a `TMPDIR` whose
path has more than 75 bytes makes tests with worker processes wait forever (three tests that were there before
the sample table do; entry 16 of the troubleshooting says why).
