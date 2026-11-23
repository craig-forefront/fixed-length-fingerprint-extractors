"""
Builds a sample table from image files and their .mnt files, and checks a
table that exists.

A sample table describes a training set with one row per single-finger image
file: which finger and which capture the file shows, how the file is prepared
(resolution, ridge polarity, sizes) and the minutiae of the prepared image.
It is a directory of Parquet files; the pixels stay in the image files.
flx/preparation/sample_table.py defines the format, and docs/sample-table.md
describes how a table is built and used.

Where this script stands: after the minutiae extraction
(scripts/extract_minutiae.py) and before the range test
(scripts/find_learning_rate.py) and the training (scripts/train_500k.py),
which read the table. It runs in the environment of the training (PyTorch),
not in that of the minutiae extractor, and it needs the package pyarrow:
    pip install -e ".[table]"

Usage:
    python scripts/build_sample_table.py \
        --dataset_dir /path/to/fingerprints \
        --minutiae_dir /path/to/minutiae \
        --samples /path/to/table \
        --ppi_manifest /path/to/resolutions.csv

    python scripts/build_sample_table.py --check /path/to/table

Which files, and which finger each one shows:
    Without --sample_list the table gets every image file below
    --dataset_dir, in any subdirectories (the files that the extraction
    lists: .png, .jpg, .jpeg, .bmp, .tif, .tiff). The name of a file without
    its ending must be <subject>_<impression> with whole numbers from 1, and
    the table holds each number minus 1, as the training reads these names:
    17_3.png is subject 16, impression 2. A "subject" is one finger. A
    directory below --dataset_dir that cannot be listed refuses the build:
    the table would lack its files without saying so.

    --sample_list FILES.csv     for files with other names: a CSV file with a
                                header line and the columns path, subject and
                                impression. path is relative to --dataset_dir
                                and is taken as it is written: white space at
                                its ends belongs to the name of the file.
                                The two numbers are whole numbers of 0 or
                                more, and they go into the table as they are
                                written (nothing is subtracted). Further
                                columns are not used.

                                    path,subject,impression
                                    sensor_a/00002239_plain_05.png,0,0
                                    sensor_b/00002239_roll_05.tif,0,1

    The numbers may have gaps, and a finger may have any number of
    impressions. Every file needs a pair of numbers of its own. The build is
    refused before any image is loaded if two files have the same pair, if a
    list names a file twice, if two files differ only in their ending (they
    share one .mnt file), if a number is above 2147483647, or if a path
    cannot be put into a table (a file name that is not UTF-8, a backslash,
    a path that leaves --dataset_dir).

Resolution and ridge polarity:
    Give this script the options that the extraction was given: --ppi,
    --ppi_manifest, --assume_ppi and --ridges (scripts/extract_minutiae.py
    describes them). Every image is loaded once, as the extraction loaded
    it, and the table records for every file which resolution and which
    ridge polarity it was taken to have. Whoever reads the table takes them
    from there.

    With --minutiae_dir the options are compared with the file
    extraction_settings.json of that directory, if it is there. The build is
    refused if they differ.

    --allow_other_extraction_settings
                                build all the same: every .mnt file is then
                                accepted or refused by its own checks

Minutiae:
    --minutiae_dir DIR          the output directory of the extraction. The
                                .mnt file of an image is looked for where the
                                extraction writes it: at the path of the
                                image below DIR, with the ending .mnt.
                                Without this option every row has null
                                minutiae, and that is no failure.

    A .mnt file is accepted if the size in its header is the size of the
    prepared image. If its last line is the comment that the extraction
    writes ("# source 826x1274 at 1000 ppi (manifest), ridges dark
    (argument), prepared 413x637"), the source size and the ridge polarity
    named there must be those of this build as well. The minutiae are stored
    as the training reads them from the file: x, y and the angle of every
    minutia, with 32 bits each.

    A .mnt file that is accepted and holds no minutiae (its first line is 0)
    gives an empty list: the image has no minutiae. So does a .mnt file
    without a header (fewer than two lines), as the training takes it; such
    a file is named in the warnings log.

    A .mnt file that is missing or refused (another size in its header,
    another source size or polarity in its comment, a file that cannot be
    read as a .mnt file, a minutia with a value that is not a finite number)
    gives a row with null minutiae and a line in the failures log. The image
    is still there for validation and for a training without minutiae.

Failures and warnings:
    One bad file does not stop the build and does not spoil the table. An
    image that cannot be loaded (not an image, resolution unknown) gets a
    line in the failures log and no row. The two logs are written into the
    table directory, one line per file: path, tab, message.

    A build in which no image at all can be loaded writes no table. It is
    given up after its first 1,000 files if not one of them could be loaded:
    that is a mistake of the options (no resolution is known for the files,
    or --dataset_dir is not the directory of the images), and a build of
    millions of files does not have to load them all to say so.

        build_failures.txt      "no row: ..." for an image that could not be
                                loaded, "null minutiae: ..." for a .mnt file
                                that is missing or was refused
        build_warnings.txt      images that look different from the ridge
                                polarity they were taken to have, .mnt files
                                without a header, and .mnt files that hold
                                fewer minutiae than their first line says

    A log without lines is not kept. The exit status is 0, or 1 if any file
    has a line in the failures log (the table is written all the same), or 2
    if the build or the check was refused: options that do not fit, an
    output directory that exists, files that cannot be told apart, other
    settings than those of the extraction, no image file, no image that
    could be loaded, pyarrow missing, a table that cannot be read. An error
    that the script does not foresee (a disk that is full while a part is
    written, for example) is printed with its traceback and ends the script
    with the status 2 as well. So a build that ends with 0 or 1 has written
    its table, and one that ends with 2 has not. A build that is interrupted
    ends with neither 0 nor 1, and so does a build that fails because its
    output cannot be written any more (the reader of a pipe went away).
    These are the statuses of a script that started: where a package that it
    imports is missing (PyTorch in the environment of the minutiae
    extractor, for example), Python ends with a traceback and its own status
    1 before a build begins.

The table:
    --samples TABLE             the directory to write. It must not exist: a
                                table is not changed, it is built again.
    --workers N                 worker processes that load the images
                                (default 1: they are loaded in this process).
                                If a worker process dies, the file that it
                                was loading is loaded once more; if that
                                worker dies as well, the file gets a line in
                                the failures log and no row. There is no time
                                limit for a file. A build that stands still
                                on a file that is never read to its end can
                                be helped along: end the worker process that
                                holds the file (kill). The file is then
                                loaded once more; when that worker is ended
                                as well, the build goes on without the file.

    The files are put into the order of (subject, impression) and cut into
    parts of 250,000 files; each part is one Parquet file. The parts are
    written into a temporary directory next to TABLE (TABLE.tmp<process
    id>). When all are written, the table is read back as a reader reads it,
    and only then is the directory renamed to TABLE: a directory TABLE that
    exists holds a whole table. If the build fails or is interrupted (from
    the keyboard, by the signal that kill sends, or by the one that a build
    gets when its terminal is closed), the temporary directory is removed. A
    build that is ended without the time for that (kill -9, the quit key of
    the terminal, a machine that goes down) leaves it behind, and it can be
    removed then. A build that was started with nohup goes on when its
    terminal is closed. Nothing is written into --dataset_dir or
    --minutiae_dir, and TABLE must not lie in them.

    At its end the script prints what the table holds: files, rows and
    failed files, the rows with, with empty and with null minutiae, the
    subjects and the impressions per subject, where the resolutions come
    from, the ridge polarities, and the checksum of the table.

Checking a table:
    --check TABLE               read the table with every rule of the format
                                and print the same summary. The exit status
                                is 0, or 2 with the reason why the table
                                cannot be read. No other option goes with it.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import multiprocessing
import multiprocessing.connection
import os
import re
import shutil
import signal
import sys
import threading
import traceback
from pathlib import PurePosixPath
from typing import Iterator, Optional

# The flx package of this repository, also when it is not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tqdm

from flx.data.fingerflow_minutia_loader import read_mnt_file
from flx.preparation import sensor_normalization
from flx.preparation.minutiae_extraction import IMAGE_EXTENSIONS, SETTINGS_FILE, iter_image_paths
from flx.preparation.sample_table import (
    PPI_ORIGINS,
    PYARROW_REQUIREMENT,
    RIDGES_ORIGINS,
    SampleRow,
    Samples,
    SampleTableError,
    normalize_image_path,
    read_samples,
    samples_checksum,
    samples_from_rows,
    validate_samples,
    write_part,
)
from flx.preparation.sensor_normalization import (
    RIDGES_DARK,
    RIDGES_LIGHT,
    PreparedImageInfo,
    SensorNormalizer,
)

# Listed files per part of a table. The files are counted, not the rows, so
# that the number of parts is known before the first image is loaded: a part
# holds fewer rows than files where images could not be loaded, and it can
# hold none.
FILES_PER_PART = 250_000

# Listed files after which a build is given up if not one of them could be
# loaded as an image. What keeps every file from loading is a mistake of the
# options, and a build of millions of files must not load them all to say so.
FILES_BEFORE_GIVING_UP = 1_000

# The two logs of a build, in the directory of the table
FAILURES_LOG = "build_failures.txt"
WARNINGS_LOG = "build_warnings.txt"

# How a line of the failures log begins: for a file that has no row in the
# table, and for a file whose row has null minutiae
NO_ROW = "no row"
NULL_MINUTIAE = "null minutiae"

# How the .mnt file of an image was accepted: as a file without a header, by
# its header and the comment line of the extraction, or by its header alone
_NO_HEADER, _COMPARED, _HEADER_ONLY = "no header", "compared", "header only"

# The largest subject or impression that a table holds (its columns have 32 bits)
_LARGEST_NUMBER = 2**31 - 1

# How many files, lines or pairs a message names before it gives the number of the rest
_MAX_NAMED = 5

# Failures that are printed at the end of a build; the log holds all of them
_MAX_PRINTED_FAILURES = 10

# Characters of the message of a line of a log. A message can quote what a
# file holds, and one file must not fill the log and the terminal.
_MAX_MESSAGE = 1000

# Resolutions that the summary lists before it gives the number of the rest
_MAX_RESOLUTIONS = 20

# How long a worker process is waited for when the build is over and the
# worker has been told so, before it is ended from outside
_SECONDS_TO_END = 10

# The signal that a build gets when its terminal is closed or the connection
# to it is lost; None on a system that does not have it
_SIGHUP = getattr(signal, "SIGHUP", None)

# The last line of a .mnt file as scripts/extract_minutiae.py writes it:
# "# " and PreparedImageInfo.describe() of the image it extracted from
_EXTRACTION_COMMENT = re.compile(
    r"# source (\d+)x(\d+) at \S+ ppi \(\w+\), ridges (\w+) \(\w+\), prepared \d+x\d+"
)

_WHOLE_NUMBER = re.compile(r"[0-9]+")

# Where the ridge polarity of a file comes from, for the summary
_RIDGES_ORIGIN_TEXT = {
    "argument": "from --ridges",
    "manifest": "from the manifest",
    "estimated": "estimated from the image",
    "default": "by default: the image allowed no estimate",
}

_NOT_GIVEN = object()


class BuildRefused(Exception):
    """A build or a check that is not done: main() prints the message and returns 2."""


def _shown(text: str) -> str:
    """
    A path as messages and logs show it: on one line. Characters that cannot
    be printed (a line break, a tab, the bytes of a file name that is not
    UTF-8) are written as escape sequences. A path that reaches a log holds
    no backslash, so the escapes cannot be mistaken for a part of it.
    """
    shown = []
    for character in text:
        if character.isprintable():
            shown.append(character)
        elif "\udc80" <= character <= "\udcff":
            # How Python holds a byte of a file name that is not UTF-8
            shown.append(f"\\x{ord(character) - 0xDC00:02x}")
        else:
            shown.append(character.encode("unicode_escape").decode("ascii"))
    return "".join(shown)


def _named(names: list, total: int) -> str:
    """The names, and how many more there are of `total`."""
    more = f"; and {total - len(names):,} more" if total > len(names) else ""
    return "; ".join(names) + more


def _cut(message: str) -> str:
    """The message, or its first _MAX_MESSAGE characters and how many more there are."""
    if len(message) <= _MAX_MESSAGE:
        return message
    return f"{message[:_MAX_MESSAGE]}... ({len(message) - _MAX_MESSAGE:,} more characters)"


# ---------------------------------------------------------------------------
# Which files, and which finger each one shows
# ---------------------------------------------------------------------------


class _Files:
    """
    The listed files: the normalised path of each, relative to the image
    directory, with its subject and impression as the table is to hold them.
    """

    def __init__(self, source: str, paths: list, subjects: list, impressions: list, lines=None):
        self.source = source  # the image directory or the sample list, for messages
        self.paths = paths
        self.subjects = np.array(subjects, dtype=np.int64)
        self.impressions = np.array(impressions, dtype=np.int64)
        # The line of the sample list that names each file; None for a directory
        self.lines = lines

    def __len__(self) -> int:
        return len(self.paths)

    def name(self, position: int) -> str:
        """A file as a message names it."""
        path = _shown(self.paths[position])
        return path if self.lines is None else f"line {self.lines[position]} ({path})"


def _path_problem(path: str) -> Optional[str]:
    """
    What keeps a normalised image path out of a table, or None. These are the
    rules of validate_samples for the text of one path, applied here so that
    a build learns of such a path before it loads an image.
    """
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return "is not valid UTF-8 (its bytes are shown escaped)"
    if path == ".":
        return "names no file"
    if "\x00" in path:
        return "holds a NUL character"
    if "\\" in path:
        return "holds a backslash, which a table does not allow in a path (its separator is /)"
    if path.startswith("/"):
        return "is absolute: it must be relative to --dataset_dir"
    if path == ".." or path.startswith("../"):
        return "leaves the image directory"
    return None


def _number_problem(name: str, number: int) -> Optional[str]:
    """What keeps a subject or an impression (0 or more) out of a table, or None."""
    if number > _LARGEST_NUMBER:
        return f"the {name} {number} is above {_LARGEST_NUMBER}, the largest number of a table"
    return None


def _numbers_of_name(path: str) -> Optional[tuple[int, int]]:
    """
    (subject, impression) of an image file that is named as the training
    expects it without a table, <subject>_<impression> with whole numbers
    from 1: each number minus 1, as CustomFingerprintLoader of
    scripts/train_500k.py reads the name. None for any other name.
    """
    name = os.path.splitext(path.rpartition("/")[2])[0]
    try:
        subject, impression = name.split("_")
        numbers = int(subject) - 1, int(impression) - 1
    except ValueError:
        return None
    return None if min(numbers) < 0 else numbers


def list_directory(dataset_dir: str) -> _Files:
    """The image files below the directory, named <subject>_<impression>."""
    # The image files are listed as the extraction lists them
    # (iter_image_paths), and that walk passes over a directory that cannot
    # be listed: its files would be missing in the table without a word. So
    # the directories are walked once before, for the errors alone.
    unlisted = []
    for _ in os.walk(dataset_dir, onerror=unlisted.append):
        pass
    if unlisted:
        named = [
            f"{_shown(str(error.filename))} ({type(error).__name__}: {error.strerror})"
            for error in unlisted[:_MAX_NAMED]
        ]
        raise BuildRefused(
            f"{dataset_dir} cannot be listed completely, so a table of it would lack image files "
            f"without saying so. What could not be listed: {_named(named, len(unlisted))}. Make "
            "these directories readable, or move them out of --dataset_dir if they hold no "
            "images of the training set, or say in a sample list which files the table is to "
            "hold: no directory is listed then (--sample_list, see docs/sample-table.md)."
        )

    paths, subjects, impressions = [], [], []
    problems, unusable = [], 0
    for listed in iter_image_paths(dataset_dir):
        path = normalize_image_path(listed)
        problem = _path_problem(path)
        if problem is not None:
            problem = f"its path {problem}"
        else:
            numbers = _numbers_of_name(path)
            if numbers is None:
                problem = "it is not named <subject>_<impression> with whole numbers from 1"
            else:
                problem = _number_problem("subject", numbers[0]) or _number_problem(
                    "impression", numbers[1]
                )
        if problem is None:
            paths.append(path)
            subjects.append(numbers[0])
            impressions.append(numbers[1])
            continue
        unusable += 1
        if len(problems) < _MAX_NAMED:
            problems.append(f"{_shown(path)}: {problem}")
    if not paths and not unusable:
        raise BuildRefused(
            f"No image file was found below {dataset_dir} (files with the endings "
            f"{', '.join(IMAGE_EXTENSIONS)}, in all subdirectories). Give the directory of the "
            "images as --dataset_dir."
        )
    if unusable:
        raise BuildRefused(
            f"{unusable:,} of the {len(paths) + unusable:,} image files below {dataset_dir} "
            f"cannot be put into a sample table: {_named(problems, unusable)}. Without "
            "--sample_list every image file must be named <subject>_<impression> with whole "
            "numbers from 1 (17_3.png becomes subject 16, impression 2 of the table). Rename the "
            "files, move files that are no training images out of --dataset_dir, or say in a "
            "sample list which finger each file shows (--sample_list, see docs/sample-table.md)."
        )
    return _Files(dataset_dir, paths, subjects, impressions)


def _number_of_list(name: str, text: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    """(the number in a cell of a sample list, None), or (None, what is wrong with the cell)."""
    text = (text or "").strip()
    if not text:
        return None, f"it names no {name}"
    if not _WHOLE_NUMBER.fullmatch(text):
        return None, f"the {name} '{_shown(text)}' is not a whole number of 0 or more"
    if len(text.lstrip("0")) > 10:  # more digits than the largest number has
        text = text if len(text) <= 20 else f"{text[:20]}... ({len(text)} digits)"
        return None, f"the {name} {text} is above {_LARGEST_NUMBER}, the largest number of a table"
    number = int(text)
    return (None, _number_problem(name, number)) if number > _LARGEST_NUMBER else (number, None)


def read_sample_list(sample_list: str) -> tuple[_Files, list[str]]:
    """
    The files of a sample list: a CSV file with a header line and the columns
    path, subject and impression. Returns them with the notes to print (the
    columns that are not used).
    """
    form = (
        "A sample list is a CSV file with a header line and the columns path, subject and "
        "impression: the path of an image file relative to --dataset_dir, and its two numbers "
        f"as the table is to hold them (whole numbers from 0 to {_LARGEST_NUMBER}). See "
        "docs/sample-table.md."
    )
    needed = ("path", "subject", "impression")
    paths, subjects, impressions, lines = [], [], [], []
    problems, unusable = [], 0
    notes = []
    try:
        # utf-8-sig: a list that a spreadsheet program wrote can begin with a byte order mark
        with open(sample_list, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns = [(name or "").strip() for name in (reader.fieldnames or [])]
            wrong = [name for name in needed if columns.count(name) != 1]
            if wrong:
                found = ", ".join(repr(name) for name in columns) if columns else "none"
                raise BuildRefused(
                    f"The first line of the sample list {sample_list} does not name the "
                    f"column(s) {', '.join(wrong)} exactly once; the columns it names: {found}. "
                    f"{form}"
                )
            reader.fieldnames = columns
            others = [repr(name) for name in columns if name not in needed]
            if others:
                notes.append(
                    f"Note: the column(s) {', '.join(others)} of {sample_list} are not used."
                )
            for row in reader:
                subject = impression = None
                # As it is written: white space at the ends of a path belongs
                # to the name of the file, and without it the path could name
                # another file, whose row would then hold these numbers
                text = row["path"] or ""
                path = normalize_image_path(text)
                if row.get(None) is not None:
                    problem = (
                        "it has more fields than the header line (a path that holds a comma "
                        "must stand in double quotes)"
                    )
                elif not text:
                    problem = "the path is empty"
                elif _path_problem(path) is not None:
                    problem = f"the path '{_shown(text)}' {_path_problem(path)}"
                else:
                    subject, problem = _number_of_list("subject", row["subject"])
                    if problem is None:
                        impression, problem = _number_of_list("impression", row["impression"])
                if problem is None:
                    paths.append(path)
                    subjects.append(subject)
                    impressions.append(impression)
                    lines.append(reader.line_num)
                    continue
                unusable += 1
                if len(problems) < _MAX_NAMED:
                    problems.append(f"line {reader.line_num}: {problem}")
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise BuildRefused(
            f"The sample list {sample_list} cannot be read as a CSV file in UTF-8 "
            f"({type(error).__name__}: {error}). {form}"
        ) from None
    if unusable:
        raise BuildRefused(
            f"{unusable:,} line(s) of the sample list {sample_list} cannot be used: "
            f"{_named(problems, unusable)}. {form}"
        )
    if not paths:
        raise BuildRefused(f"The sample list {sample_list} names no file. {form}")
    return _Files(sample_list, paths, subjects, impressions, lines), notes


def mnt_path(image_path: str) -> str:
    """
    The path of the .mnt file of an image, both relative: the ending of the
    file name replaced by .mnt, which is where scripts/extract_minutiae.py
    writes the minutiae of an image below its output directory.
    """
    return str(PurePosixPath(image_path).with_suffix(".mnt"))


def check_files_are_distinct(files: _Files, order: np.ndarray) -> None:
    """
    Refuses files that cannot be told apart: a file that is listed twice, two
    files that share a .mnt file, two files with the same subject and
    impression. `order` puts the files into the order of (subject,
    impression).
    """
    listed_twice, share_mnt = [], []
    twice = shared = 0
    first_with = {}  # .mnt path -> the first file that has it
    for position, path in enumerate(files.paths):
        first = first_with.setdefault(mnt_path(path), position)
        if first == position:
            continue
        if files.paths[first] == path:
            twice += 1
            if len(listed_twice) < _MAX_NAMED:
                listed_twice.append(
                    f"{_shown(path)} (line {files.lines[first]} and line {files.lines[position]})"
                    if files.lines is not None
                    else _shown(path)
                )
        else:
            shared += 1
            if len(share_mnt) < _MAX_NAMED:
                share_mnt.append(f"{files.name(first)} and {files.name(position)}")
    if twice:
        raise BuildRefused(
            f"{files.source} names {twice:,} file(s) more than once: "
            f"{_named(listed_twice, twice)}. An image file is one row of the table: remove the "
            "repeated lines."
        )
    if shared:
        raise BuildRefused(
            f"{shared:,} pair(s) of image files differ only in their ending: "
            f"{_named(share_mnt, shared)}. The .mnt file of an image is its path with the ending "
            ".mnt, so the two files of such a pair share one .mnt file, and the minutiae of one "
            "cannot be told from those of the other. Rename one file of each pair and extract "
            "its minutiae again, or leave it out of the table."
        )

    subjects, impressions = files.subjects[order], files.impressions[order]
    same = np.flatnonzero((subjects[1:] == subjects[:-1]) & (impressions[1:] == impressions[:-1]))
    if len(same):
        pairs = []
        for index in same[:_MAX_NAMED].tolist():
            first, second = sorted((int(order[index]), int(order[index + 1])))
            pairs.append(
                f"subject {subjects[index]}, impression {impressions[index]}: "
                f"{files.name(first)} and {files.name(second)}"
            )
        if files.lines is None:
            remedy = (
                "Files of the same name in two directories get the same numbers: rename one of "
                "them, or give the numbers of every file in a sample list (--sample_list, see "
                "docs/sample-table.md)."
            )
        else:
            remedy = "Give every line a pair of numbers of its own."
        raise BuildRefused(
            f"{len(same):,} pair(s) of image files have the same subject and impression: "
            f"{_named(pairs, len(same))}. Every image needs a pair of numbers of its own, or two "
            f"images would count as one capture. {remedy}"
        )


def build_id(files: _Files, order: np.ndarray, settings: dict, with_minutiae: bool) -> str:
    """
    The id of a build: the SHA-256 of what was asked for, that is of the
    resolution settings, of whether .mnt files are read, and of every listed
    file with its numbers. Two builds of the same list under the same options
    have the same id, however many workers and parts they use. A reader tells
    the parts of two builds apart by it.
    """
    digest = hashlib.sha256()
    digest.update(b"flx-samples-build-1\n")
    asked = {"settings": settings, "with_minutiae": with_minutiae}
    digest.update(json.dumps(asked, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    for position in order.tolist():
        path = files.paths[position].encode("utf-8")
        # The length of the path comes before it, so that no path can pass for two lines
        digest.update(
            b"%d %d %d\n" % (files.subjects[position], files.impressions[position], len(path))
        )
        digest.update(path + b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The settings of the extraction
# ---------------------------------------------------------------------------


def check_extraction_settings(minutiae_dir: str, settings: dict, allow_other: bool) -> str:
    """
    Compares the resolution settings of this build with those that the
    extraction stored in its output directory, and returns the line to print
    about it. The file is only read: nothing is written into the directory.
    Other settings raise BuildRefused, unless they are allowed.
    """
    path = os.path.join(minutiae_dir, SETTINGS_FILE)
    decide = "every .mnt file is accepted or refused by its own checks"
    allow = f"To build all the same, pass --allow_other_extraction_settings: {decide}."
    allowed = f"It was allowed (--allow_other_extraction_settings): {decide}."
    if not os.path.lexists(path):
        return (
            f"Note: {path} is missing, so the settings of the extraction cannot be compared "
            f"with those of this build; {decide}."
        )
    try:
        with open(path) as file:
            stored = json.load(file)
        if not isinstance(stored, dict):
            raise ValueError("it does not hold a JSON object")
    # RecursionError: what a file raises that opens brackets without end
    except (OSError, ValueError, RecursionError) as error:
        problem = (
            f"{path} cannot be read ({type(error).__name__}: {error}), so the settings of the "
            "extraction cannot be compared with those of this build."
        )
        if allow_other:
            return f"Note: {problem} {allowed}"
        raise BuildRefused(
            f"{problem} It is the file in which scripts/extract_minutiae.py stores its "
            f"settings: restore it, or remove it if it is damaged. {allow}"
        ) from None

    missing = object()
    # A JSON number 1 is the setting 1.0, but true is not: booleans are compared by type as well
    different = [
        key
        for key, value in settings.items()
        if stored.get(key, missing) != value
        or isinstance(stored.get(key), bool) != isinstance(value, bool)
    ]
    if not different:
        return f"Settings: those of the extraction ({path})"
    problem = f"{path} holds other settings than this build uses: " + "; ".join(
        f"{key} is {json.dumps(stored[key]) if key in stored else 'not stored'} there and "
        f"{json.dumps(settings[key])} here"
        for key in different
    )
    if allow_other:
        return f"Note: {problem}. {allowed}"
    raise BuildRefused(
        f"{problem}. The .mnt files of {minutiae_dir} fit images that are prepared as the "
        "extraction prepared them: give this script the options that the extraction was given "
        f"(--ppi, --ppi_manifest, --assume_ppi, --ridges). {allow}"
    )


# ---------------------------------------------------------------------------
# One file: the image, and its .mnt file
# ---------------------------------------------------------------------------


def read_minutiae(path: str, info: PreparedImageInfo) -> tuple:
    """
    Reads the .mnt file of an image that was loaded with the given result.

    Returns (minutiae, why not, warning, accepted):
        minutiae: float32 array (k, 3) of x, y and angle as read_mnt_file
            returns them, k == 0 for a file without minutiae; None if the
            file is missing or refused
        why not: the reason where minutiae is None
        warning: what is to be said about a file that was accepted, or None
        accepted: how the file was accepted: _NO_HEADER, _COMPARED (it ends
            with the comment of the extraction, so its source size and its
            polarity were compared as well) or _HEADER_ONLY; else None
    """
    try:
        # A value that is too large for 32 bits becomes infinite, which is
        # refused below: numpy need not warn about it as well
        with np.errstate(over="ignore"):
            locations, angles, (width, height) = read_mnt_file(path)
        with open(path) as file:
            lines = file.readlines()
    except FileNotFoundError:
        return None, "the .mnt file is missing", None, None
    except Exception as error:  # whatever a file that is no .mnt file raises
        reason = f"the .mnt file cannot be read ({type(error).__name__}: {error})"
        return None, reason, None, None

    if len(lines) < 2:
        # Without a header there is no size to compare. The training takes
        # such a file as an image without minutiae, and so does the table.
        warning = (
            f"the .mnt file holds no header ({len(lines)} line(s)); it was taken as a file "
            "without minutiae"
        )
        return np.zeros((0, 3), dtype=np.float32), None, warning, _NO_HEADER

    if (width, height) != (info.width, info.height):
        reason = (
            f"the header of the .mnt file states an image of {width}x{height} pixels, and the "
            f"prepared image has {info.width}x{info.height}: the minutiae were extracted from "
            "an image that was prepared in another way, or from another image"
        )
        return None, reason, None, None

    last_line = next((line.strip() for line in reversed(lines) if line.strip()), "")
    comment = _EXTRACTION_COMMENT.fullmatch(last_line)
    if comment is not None:
        different = []
        source = int(comment[1]), int(comment[2])
        if source != (info.source_width, info.source_height):
            different.append(
                f"a source image of {source[0]}x{source[1]} pixels, and the image file has "
                f"{info.source_width}x{info.source_height}"
            )
        if comment[3] != info.ridges:
            different.append(
                f"{comment[3]} ridges, and this build took the image to have {info.ridges} "
                f"ridges ({info.ridges_origin})"
            )
        if different:
            reason = f"the last line of the .mnt file names {'; and '.join(different)}"
            return None, reason, None, None

    # read_mnt_file returns the shape (0,) where no row could be read
    locations = locations.reshape(-1, 2)
    minutiae = np.empty((len(locations), 3), dtype=np.float32)
    minutiae[:, :2] = locations
    minutiae[:, 2] = angles
    finite = np.isfinite(minutiae).all(axis=1)
    if not finite.all():
        first = int(np.argmin(finite))
        values = " ".join(repr(float(value)) for value in minutiae[first])
        reason = (
            f"{int(np.count_nonzero(~finite))} of the {len(minutiae)} minutiae of the .mnt "
            "file hold a value that is not a finite number; the first of them is minutia "
            f"{first + 1} ({values})"
        )
        return None, reason, None, None

    warning = None
    announced = int(lines[0].strip())
    if announced != len(minutiae):
        warning = (
            f"the first line of the .mnt file announces {announced} minutiae, and "
            f"{len(minutiae)} could be read"
        )
    return minutiae, None, warning, _HEADER_ONLY if comment is None else _COMPARED


def _load_sample(source: tuple, subject: int, impression: int, image_path: str) -> tuple:
    """load_sample, but an image that cannot be loaded raises its error."""
    dataset_dir, minutiae_dir, normalizer = source
    path = os.path.join(dataset_dir, image_path)
    file_size = os.stat(path).st_size
    _, info = normalizer.load(path, check_polarity=True)

    minutiae, failure, accepted = None, None, None
    notes = [info.polarity_warning] if info.polarity_warning else []
    if minutiae_dir is not None:
        minutiae, reason, warning, accepted = read_minutiae(
            os.path.join(minutiae_dir, mnt_path(image_path)), info
        )
        if minutiae is None:
            failure = f"{NULL_MINUTIAE}: {reason}"
        if warning:
            notes.append(warning)

    def row(minutiae) -> SampleRow:
        return SampleRow(
            subject=subject,
            impression=impression,
            image_path=image_path,
            file_size=file_size,
            ppi=info.ppi,
            ppi_origin=info.ppi_origin,
            ridges=info.ridges,
            ridges_origin=info.ridges_origin,
            source_width=info.source_width,
            source_height=info.source_height,
            width=info.width,
            height=info.height,
            minutiae=minutiae,
        )

    # The rules of the table, for this file alone: what they refuse costs
    # this file its minutiae or its row, here and now, and not the whole
    # table when it is read at the end of the build
    try:
        validate_samples(samples_from_rows([row(minutiae)]))
    except SampleTableError as error:
        if minutiae is None:
            raise
        validate_samples(samples_from_rows([row(None)]))
        minutiae, failure, accepted = None, f"{NULL_MINUTIAE}: {error}", None
    return row(minutiae), failure, "; ".join(notes) or None, accepted


def load_sample(source: tuple, subject: int, impression: int, image_path: str) -> tuple:
    """
    Loads one listed file: the image as the preparation loads it, and its
    .mnt file where minutiae are read.

    source: (image directory, minutiae directory or None, SensorNormalizer)

    Returns (row, failure, warning, accepted), all of plain values and numpy
    arrays, so that a worker process can send them:
        row: the SampleRow of the file, or None if the image could not be
            loaded
        failure: the line for the failures log, or None: why there is no
            row, or why the row has null minutiae although .mnt files are read
        warning: the line for the warnings log, or None
        accepted: how the .mnt file was accepted, or None (see read_minutiae)

    An error of the file is returned, not raised: one bad file must not stop
    the build.
    """
    try:
        return _load_sample(source, subject, impression, image_path)
    except Exception as error:
        return None, f"{NO_ROW}: {type(error).__name__}: {error}", None, None


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------


def _serve(connection, inherited: tuple, source: tuple) -> None:
    """
    What a worker process does: it loads the files that are sent to it
    through its pipe, one at a time, and sends back the result of each, until
    the pipe is closed.
    """
    # An interruption from the keyboard reaches all processes of the build,
    # and so does the signal of a terminal that is closed. The main process
    # ends the workers; a worker that raised an error of its own in the
    # middle of a file would only add its traceback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if _SIGHUP is not None:
        signal.signal(_SIGHUP, signal.SIG_IGN)
    # The main process turns SIGTERM into an exception; a worker is to end at once
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # A forked worker holds the ends of the pipes that the main process held
    # when it was started. They are closed here: a pipe counts as closed only
    # when no process holds its end any more, and by that a worker notices
    # that the main process is gone.
    for other in inherited:
        other.close()
    while True:
        try:
            job = connection.recv()
        except (EOFError, OSError):
            break
        if job is None:  # the build is over
            break
        result = load_sample(source, *job)
        try:
            connection.send(result)
        except OSError:
            break


class _Workers:
    """
    Worker processes that load files, each connected to this process by a
    pipe of its own. A worker has at most one file at a time, so if a worker
    dies, the file it was loading is known, and the other workers are not
    touched: they share no lock and no queue with it.

    Used in a with statement: the workers are ended when it is left, at once
    if it is left by an exception.
    """

    def __init__(self, count: int, source: tuple):
        # A forked worker has the functions and the settings of this script
        # without their being sent to it, also where the script was loaded
        # under a name that cannot be imported (as tests load it)
        method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        self._context = multiprocessing.get_context(method)
        self._source = source
        self.count = count
        self._processes = {}  # the pipe to a worker -> its process
        self._idle = []  # the pipes of the workers that have no file
        self._loading = {}  # the pipe of a worker that has a file -> the key of the file
        try:
            for _ in range(count):
                self._start()
        except BaseException:
            self.stop(at_once=True)
            raise

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        self.stop(at_once=error_type is not None)

    def _start(self) -> None:
        ours, theirs = self._context.Pipe()
        forked = self._context.get_start_method() == "fork"
        inherited = (*self._processes, ours) if forked else ()
        process = self._context.Process(
            target=_serve, args=(theirs, inherited, self._source), daemon=True
        )
        process.start()
        theirs.close()
        self._processes[ours] = process
        self._idle.append(ours)

    def _replace(self, connection) -> None:
        """Puts a new worker in the place of one that died."""
        process = self._processes.pop(connection)
        connection.close()
        process.kill()  # it is dead already, unless it only closed its pipe
        process.join()
        self._start()

    @property
    def idle(self) -> int:
        """The number of workers that have no file."""
        return len(self._idle)

    def load(self, key, job: tuple) -> None:
        """Hands a job (the arguments of load_sample after the source) to an idle worker."""
        for _ in range(3):
            connection = self._idle.pop()
            try:
                connection.send(job)
            except OSError:
                self._replace(connection)  # the worker died while it had no file
                continue
            self._loading[connection] = key
            return
        raise RuntimeError("The worker processes of the build end as soon as they are started")

    def finished(self) -> list:
        """
        Waits until a worker has finished its file, and returns (key, result)
        for every worker that has. The result is None if the worker died.
        """
        results = []
        for connection in multiprocessing.connection.wait(list(self._loading)):
            key = self._loading.pop(connection)
            try:
                result = connection.recv()
            except (EOFError, OSError):
                self._replace(connection)
                result = None
            else:
                self._idle.append(connection)
            results.append((key, result))
        return results

    def stop(self, at_once: bool) -> None:
        """
        Ends the workers. A worker without a file ends when it is told to, or
        when its pipe is closed. With at_once no worker is left the time to
        finish its file; without it, one that does not end by itself in a
        while is ended as well.
        """
        for connection, process in self._processes.items():
            if at_once:
                process.terminate()
            else:
                with contextlib.suppress(OSError):
                    connection.send(None)
            connection.close()
        for process in self._processes.values():
            process.join(None if at_once else _SECONDS_TO_END)
            if process.is_alive():
                process.terminate()
                process.join()
        self._processes, self._idle, self._loading = {}, [], {}


def loaded_files(
    pool: Optional[_Workers], source: tuple, files: _Files, order: np.ndarray
) -> Iterator[tuple]:
    """
    Yields what load_sample returns for every listed file, in the given
    order, whatever order the workers finish in: the rows of a part and the
    lines of the logs do not depend on the number of workers. Without a pool
    the files are loaded in this process.
    """

    def job(index: int) -> tuple:
        position = int(order[index])
        return (
            int(files.subjects[position]),
            int(files.impressions[position]),
            files.paths[position],
        )

    total = len(files)
    if pool is None:
        for index in range(total):
            yield load_sample(source, *job(index))
        return

    # Files that are handed out before the first of them is yielded: more
    # than there are workers, so that one slow file does not leave the other
    # workers without work, and limited, so that few results wait in memory
    window = 16 * pool.count
    waiting = {}  # index -> the result of a file that finished before one in front of it
    again = []  # files whose worker died: each is tried once more
    tried_again = set()
    handed_out = yielded = 0
    while yielded < total:
        while pool.idle and (again or (handed_out < total and handed_out - yielded < window)):
            if again:
                index = again.pop()
            else:
                index, handed_out = handed_out, handed_out + 1
            pool.load(index, job(index))
        for index, result in pool.finished():
            if result is None and index not in tried_again:
                # The worker may have died of something else than this file
                # (it was killed, or the memory was short for a moment)
                tried_again.add(index)
                again.append(index)
                continue
            if result is None:
                reason = (
                    f"{NO_ROW}: the worker process died while it loaded this file, also at "
                    "the second attempt (not enough memory, or a file that the image library "
                    "does not survive)"
                )
                result = (None, reason, None, None)
            waiting[index] = result
        while yielded in waiting:
            yield waiting.pop(yielded)
            yielded += 1


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class _Log:
    """A log with one line per file (path, tab, message) that is removed if it stays empty."""

    def __init__(self, path: str):
        self.path = path
        # A message can quote what a file holds: nothing in it may stop the log
        self._file = open(path, "w", encoding="utf-8", errors="backslashreplace")
        self.lines = 0

    def add(self, image_path: str, message: str) -> str:
        """Writes the line of a file. Returns its message as it was written: on one line, cut."""
        message = _cut(" ".join(str(message).split()))
        self._file.write(f"{_shown(image_path)}\t{message}\n")
        self._file.flush()
        self.lines += 1
        return message

    def close(self) -> None:
        self._file.close()
        if self.lines == 0:
            os.remove(self.path)


def _count_lines(path: str) -> Optional[int]:
    """The number of lines of a log, or None if it is not there."""
    try:
        with open(path, "rb") as file:
            return sum(block.count(b"\n") for block in iter(lambda: file.read(1 << 20), b""))
    except OSError:
        return None


def _number(value) -> Optional[int]:
    """A count of the build information, or None where a table does not state it as a number."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def describe_table(directory: str, samples: Samples) -> list[str]:
    """
    The lines that say what a table holds. The end of a build and --check
    print them. Everything comes from the rows and the build information
    that read_samples returned, and from the two logs beside the parts.
    """
    build = samples.build
    rows = len(samples)
    files, failed = _number(build.get("files")), _number(build.get("failed"))
    with_minutiae = build.get("with_minutiae")
    filled = int(np.count_nonzero(np.diff(samples.minutiae_offsets)))
    known = int(np.count_nonzero(samples.minutiae_known))
    subjects, impressions = np.unique(samples.subject, return_counts=True)

    def line(label: str, value, note: str = "") -> str:
        value = f"{value:,}" if isinstance(value, int) else value
        return f"  {label + ':':<27}{value}" + (f" ({note})" if note else "")

    lines = ["=" * 60, f"Sample table: {directory}"]
    lines.append(line("Files listed", "not recorded" if files is None else files))
    lines.append(line("Rows", rows))
    note = ""
    if failed and files is not None and 0 <= files - rows <= failed:
        note = f"{files - rows:,} without a row, {failed - (files - rows):,} with null minutiae"
    lines.append(line("Files that failed", "not recorded" if failed is None else failed, note))
    note = f"{len(samples.minutiae):,} minutiae" if filled else ""
    lines.append(line("Rows with minutiae", filled, note))
    lines.append(line("Rows with empty minutiae", known - filled))
    note = ""
    if rows - known and with_minutiae is False:
        note = "the table was built without --minutiae_dir"
    elif rows - known and with_minutiae is True:
        note = "the .mnt file was missing or refused"
    lines.append(line("Rows with null minutiae", rows - known, note))
    lines.append(line("Subjects", len(subjects)))
    if len(subjects):
        spread = (
            f"smallest {int(impressions.min()):,}, median {float(np.median(impressions)):g}, "
            f"largest {int(impressions.max()):,}"
        )
        lines.append(line("Impressions per subject", spread))
    lines.append(line("Parts", _number(build.get("parts")) or "not recorded"))
    lines.append(line("Checksum (samples_sha256)", samples_checksum(samples)))
    for label, name in (("Failures log", FAILURES_LOG), ("Warnings log", WARNINGS_LOG)):
        path = os.path.join(directory, name)
        count = _count_lines(path)
        if count is not None:
            lines.append(line(label, f"{path} ({count:,} line(s))"))
        elif name == FAILURES_LOG and failed:
            lines.append(line(label, f"missing, although {failed:,} file(s) failed ({path})"))
        else:
            lines.append(line(label, "none"))
    lines.append("=" * 60)
    if with_minutiae is True and rows and not known:
        lines.append(
            "WARNING: no row of the table holds minutiae, although the build read a minutiae "
            "directory. Was it the directory that scripts/extract_minutiae.py wrote for these "
            "images, with the same subdirectories?"
        )
    if not rows:
        return lines

    # Per source and resolution. Two resolutions can have the same text (it
    # shows six digits), so the files are added up by the text.
    resolutions = {}
    for index, origin in enumerate(PPI_ORIGINS):
        values, counts = np.unique(samples.ppi[samples.ppi_origin == index], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label = sensor_normalization.resolution_source_label(value, origin)
            resolutions[label] = resolutions.get(label, 0) + count
    most = sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))[:_MAX_RESOLUTIONS]
    settings = build.get("settings")
    assume_ppi = settings.get("assume_ppi") if isinstance(settings, dict) else None
    if isinstance(assume_ppi, bool) or not isinstance(assume_ppi, (int, float)):
        assume_ppi = None
    lines.append("")
    lines.append(f"Resolution of the {rows:,} image file(s) of the table:")
    # One line per resolution in the order of `most`, and behind them the note
    # if there is one. The lines are made of all resolutions and cut then: the
    # note counts the files of those that are not listed as well.
    described = sensor_normalization.describe_resolution_sources(
        resolutions, assume_ppi, samples.target_ppi
    )
    lines.extend(described[: len(most)])
    if len(resolutions) > len(most):
        rest = rows - sum(count for _, count in most)
        lines.append(f"  {rest:>12,}  with {len(resolutions) - len(most):,} further resolutions")
    lines.extend(described[len(resolutions) :])
    lines.append("Ridge polarity:")
    polarities = []
    for light, ridges in ((False, RIDGES_DARK), (True, RIDGES_LIGHT)):
        of_polarity = samples.ridges_light == light
        for index, origin in enumerate(RIDGES_ORIGINS):
            count = int(np.count_nonzero(of_polarity & (samples.ridges_origin == index)))
            if count:
                polarities.append((count, f"{ridges} ridges, {_RIDGES_ORIGIN_TEXT[origin]}"))
    polarities.sort(key=lambda polarity: -polarity[0])
    lines.extend(f"  {count:>12,}  {text}" for count, text in polarities)
    return lines


def check_table(directory: str) -> int:
    """--check: reads a table with every rule of the format and prints what it holds."""
    try:
        samples = read_samples(directory)
    except SampleTableError as error:
        raise BuildRefused(str(error)) from None
    print("\n".join(describe_table(directory, samples)))
    if samples.unknown_columns:
        print(
            f"\nNote: the table holds {len(samples.unknown_columns)} column(s) that this code "
            f"does not know and does not use: {', '.join(samples.unknown_columns)}."
        )
    return 0


def _check_pyarrow(directory: str) -> None:
    """
    Writes a part without rows into the directory and reads it back, before
    an image is loaded. A build must not load images for hours and then learn
    that pyarrow is missing or too old, or that the directory cannot be
    written to.
    """
    trial = os.path.join(directory, "trial")
    try:
        write_part(trial, 0, 1, samples_from_rows([]), {"id": "trial"})
        read_samples(trial)
    except SampleTableError as error:
        raise BuildRefused(str(error)) from None
    except Exception as error:
        raise BuildRefused(
            f"A part of a table could not be written to {directory} and read back "
            f"({type(error).__name__}: {error}). If the directory can be written to, the "
            "installed pyarrow is not one that a sample table can use: it needs "
            f'{PYARROW_REQUIREMENT} (pip install -e ".[table]").'
        ) from None
    finally:
        shutil.rmtree(trial, ignore_errors=True)


@contextlib.contextmanager
def _ended_by_signals():
    """
    While a build runs, SIGTERM (what kill and a job scheduler send) and
    SIGHUP (what a build gets when its terminal is closed) end it as an
    interruption from the keyboard does: with an exception, so that the
    workers are ended and the temporary directory is removed.

    SIGHUP is left alone where the process ignores it: that is how nohup
    starts a program that is to go on when its terminal is closed.
    """
    if threading.current_thread() is not threading.main_thread():
        yield  # a signal handler can only be set in the main thread
        return

    def end(number, frame):
        raise SystemExit(128 + number)

    previous = {signal.SIGTERM: signal.signal(signal.SIGTERM, end)}
    if _SIGHUP is not None and signal.getsignal(_SIGHUP) is not signal.SIG_IGN:
        previous[_SIGHUP] = signal.signal(_SIGHUP, end)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _print_summary(
    samples_dir: str, samples: Samples, failed: int, first_failures: list, accepted: dict
) -> None:
    """Prints what a build wrote: the table, the notes about its .mnt files, the first failures."""
    print("\n".join(describe_table(samples_dir, samples)))
    if accepted[_NO_HEADER]:
        print(
            f"\nNote: {accepted[_NO_HEADER]:,} .mnt file(s) hold no header and were taken as "
            "files without minutiae. They are named in "
            f"{os.path.join(samples_dir, WARNINGS_LOG)}."
        )
    if accepted[_HEADER_ONLY]:
        print(
            f"\nNote: {accepted[_HEADER_ONLY]:,} .mnt file(s) do not end with the comment line "
            "of scripts/extract_minutiae.py: the size in their header was compared, their "
            "source size and ridge polarity could not be."
        )
    if failed:
        print(
            f"\n{failed:,} file(s) failed. The first {len(first_failures)} of them (all are in "
            f"{os.path.join(samples_dir, FAILURES_LOG)}):"
        )
        print("\n".join(f"  {failure}" for failure in first_failures))
        print(
            f'A line "{NO_ROW}: ..." names an image that could not be loaded; the table does '
            f'not hold it. A line "{NULL_MINUTIAE}: ..." names an image that is in the table '
            "without minutiae, because its .mnt file is missing or was refused. Remove the "
            "causes and build the table again, into a new directory."
        )


def build_table(args, normalizer: SensorNormalizer) -> int:
    """Builds the table of --samples. Returns the exit status 0 or 1; BuildRefused stands for 2."""
    samples_dir = os.path.normpath(args.samples)
    if os.path.lexists(samples_dir):
        raise BuildRefused(
            f"{samples_dir} exists. A table is not changed or overwritten: give a directory "
            "that does not exist yet as --samples, or remove the old one first. (To check a "
            f"table that exists: --check {samples_dir})"
        )
    temporary = f"{samples_dir}.tmp{os.getpid()}"
    # Directories above the table that are missing are made, and removed
    # again if no table is written
    parent = os.path.dirname(os.path.abspath(samples_dir))
    made, directory = [], parent
    while not os.path.lexists(directory):
        made.append(directory)
        directory = os.path.dirname(directory)
    published = made_temporary = False
    try:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as error:
            raise BuildRefused(
                f"The directory {parent}, in which the table is to be written, cannot be made "
                f"({type(error).__name__}: {error}). Give as --samples a path in a directory "
                "that exists or can be made."
            ) from None
        try:
            os.mkdir(temporary)
        except FileExistsError:
            raise BuildRefused(
                f"The temporary directory {temporary}, in which the table would be built, "
                f"exists already: another build into {samples_dir} is running, or one was "
                "killed and left it. It was not touched. If no build is running, remove it and "
                "start this build again."
            ) from None
        except OSError as error:
            raise BuildRefused(
                f"The temporary directory {temporary}, in which the table is built, cannot be "
                f"made ({type(error).__name__}: {error}). The table is built next to its place: "
                f"{parent} must be a directory that can be written to."
            ) from None
        made_temporary = True
        with _ended_by_signals():
            samples, failed, first_failures, accepted = _build(args, normalizer, temporary)
            if os.path.lexists(samples_dir):
                raise BuildRefused(
                    f"{samples_dir} was made by something else while the table was built. The "
                    "table was not kept: build it again, into a directory that nothing else "
                    "writes to."
                )
            os.rename(temporary, samples_dir)
            published = True
    except BaseException as error:
        if not isinstance(error, BuildRefused):
            print(
                f"\nThe build did not finish: no table was written, and {temporary} is removed.",
                file=sys.stderr,
            )
        raise
    finally:
        if not published:
            # Only the temporary directory that this build made: one that
            # was there is the work of another build
            if made_temporary:
                shutil.rmtree(temporary, ignore_errors=True)
            for directory in made:
                with contextlib.suppress(OSError):
                    os.rmdir(directory)

    # The table has its name: nothing that is printed from here on can cost
    # it. The exit status says that it is written, also where what it holds
    # could not be printed (the reader of the output went away, as the reader
    # of a pipe can), and where not even that could be said (a build that
    # outlived its terminal can write nothing to it).
    try:
        _print_summary(samples_dir, samples, failed, first_failures, accepted)
    except Exception as error:
        with contextlib.suppress(Exception):
            print(
                f"\nThe table {samples_dir} is written, but what it holds could not be printed "
                f"({type(error).__name__}: {error}). To see it: --check {samples_dir}",
                file=sys.stderr,
            )
    return 1 if failed else 0


def _build(args, normalizer: SensorNormalizer, temporary: str) -> tuple:
    """
    Lists the files, loads them, writes the parts and the logs into the
    temporary directory and reads the table back. Returns (the table as it
    was read, the number of files in the failures log, the first lines of
    that log, the numbers of .mnt files by how they were accepted).
    """
    _check_pyarrow(temporary)

    settings = normalizer.settings()
    with_minutiae = args.minutiae_dir is not None
    print(f"Images:   {normalizer.describe()}")
    if with_minutiae:
        print(f"Minutiae: the .mnt files below {args.minutiae_dir}")
        print(
            check_extraction_settings(
                args.minutiae_dir, settings, args.allow_other_extraction_settings
            )
        )
    else:
        print("Minutiae: not read (no --minutiae_dir): every row gets null minutiae")

    if args.sample_list is not None:
        print(f"Reading the sample list {args.sample_list}")
        files, notes = read_sample_list(args.sample_list)
    else:
        print(f"Scanning {args.dataset_dir} for images...")
        files, notes = list_directory(args.dataset_dir), []
    for note in notes:
        print(note)
    # The parts follow each other in the order of the rows: whoever reads
    # such a table has its rows in order without moving them
    order = np.lexsort((files.impressions, files.subjects))
    check_files_are_distinct(files, order)
    total = len(files)
    parts = -(-total // FILES_PER_PART)
    print(f"Files:    {total:,}, to be written as {parts:,} part(s)")
    build = {
        "id": build_id(files, order, settings, with_minutiae),
        "settings": settings,
        "with_minutiae": with_minutiae,
    }

    source = (args.dataset_dir, args.minutiae_dir, normalizer)
    failures = _Log(os.path.join(temporary, FAILURES_LOG))
    warnings = _Log(os.path.join(temporary, WARNINGS_LOG))
    first_failures = []
    accepted = {_NO_HEADER: 0, _COMPARED: 0, _HEADER_ONLY: 0}
    written = 0
    rows, failed_of_part = [], 0
    any_row = False
    # What to do where no file could be loaded as an image
    no_image = (
        "If the resolution of the files is not known, give --ppi, --ppi_manifest or "
        "--assume_ppi; if the files were not found, check --dataset_dir."
    )
    try:
        with contextlib.ExitStack() as stack:
            pool = None
            if args.workers > 1:
                pool = stack.enter_context(_Workers(min(args.workers, total), source))
            progress = stack.enter_context(
                tqdm.tqdm(total=total, desc="Loading images", unit="img")
            )
            results = loaded_files(pool, source, files, order)
            for index, (row, failure, warning, how_accepted) in enumerate(results):
                path = files.paths[int(order[index])]
                if failure is not None:
                    failure = failures.add(path, failure)
                    failed_of_part += 1
                    if len(first_failures) < _MAX_PRINTED_FAILURES:
                        first_failures.append(f"{_shown(path)}: {failure}")
                if warning is not None:
                    warnings.add(path, warning)
                if row is not None:
                    rows.append(row)
                    any_row = True
                if how_accepted is not None:
                    accepted[how_accepted] += 1
                if not any_row and index + 1 == FILES_BEFORE_GIVING_UP and index + 1 < total:
                    raise BuildRefused(
                        f"None of the first {index + 1:,} of the {total:,} listed files could "
                        "be loaded as an image, so the build was given up and no table was "
                        f"written: {_named(first_failures[:2], index + 1)}. {no_image}"
                    )
                if (index + 1) % FILES_PER_PART == 0 or index + 1 == total:
                    # The rows of one part are all the rows that are held at a time
                    write_part(
                        temporary,
                        index // FILES_PER_PART,
                        parts,
                        samples_from_rows(rows),
                        dict(build, files=index % FILES_PER_PART + 1, failed=failed_of_part),
                    )
                    written += len(rows)
                    rows, failed_of_part = [], 0
                progress.update(1)
    except SampleTableError as error:
        raise BuildRefused(f"The table could not be written: {error}") from None
    finally:
        failures.close()
        warnings.close()

    if not written:
        # The log goes with the temporary directory, so the message says why
        # the first two files failed (the reason of one file can be long)
        named = _named(first_failures[:2], total)
        raise BuildRefused(
            f"None of the {total:,} listed files could be loaded as an image, so no table was "
            f"written: {named}. {no_image}"
        )

    # The table as a reader gets it, with every rule of the format. Only a
    # table that passes gets its name.
    try:
        samples = read_samples(temporary)
    except SampleTableError as error:
        # The message of the reader says what to do with a table that is
        # kept. Of this one nothing is left, so the refusal says what to do
        # about the build.
        raise BuildRefused(
            f"The table that was built does not pass its own check, and it was not kept: {error} "
            "That is the message of the reader, for a table that it is given. Of this build "
            "nothing is left to copy or to read again: its temporary directory is removed. Build "
            "the table again. If it ends like this once more, something changes the parts "
            "between their writing and their reading: another program that writes to the "
            "directory of the table, or a fault of the disk or of the memory of this machine."
        ) from None
    if len(samples) != written or samples.build.get("files") != total:
        raise BuildRefused(
            f"The table that was built holds {len(samples):,} rows of "
            f"{samples.build.get('files')} files where {written:,} rows of {total:,} files "
            "were written: its temporary directory was changed while it was built. It was not "
            "kept."
        )
    return samples, failures.lines, first_failures, accepted


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _is_inside(path: str, directory: str) -> bool:
    """Whether the path is the directory or lies below it, by what both resolve to."""
    path, directory = os.path.realpath(path), os.path.realpath(directory)
    return os.path.commonpath([path, directory]) == directory


def main() -> int:
    """
    Returns the exit status: 0, 1 if a file has a line in the failures log,
    2 if the build or the check was refused. An error that is no refusal is
    raised; the script, run as a program, ends with the status 2 for it too.
    """
    parser = argparse.ArgumentParser(
        description="Build a sample table from image files and their .mnt files, or check one",
        usage=(
            "%(prog)s --dataset_dir IMAGES --samples TABLE [--minutiae_dir MNT] [options]\n"
            "       %(prog)s --check TABLE"
        ),
        epilog="See the docstring at the top of this script and docs/sample-table.md.",
    )
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Directory of the image files; the image paths of the table are "
                             "relative to it. Needed for a build (default: none)")
    parser.add_argument("--samples", type=str, default=None,
                        help="Directory of the table to write. It must not exist yet. Needed "
                             "for a build (default: none)")
    parser.add_argument("--minutiae_dir", type=str, default=None,
                        help="Output directory of scripts/extract_minutiae.py: the .mnt file of "
                             "an image is looked for at the path of the image below it, with "
                             "the ending .mnt (default: none, every row then has null minutiae)")
    parser.add_argument("--sample_list", type=str, default=None,
                        help="CSV file with a header line and the columns path,subject,"
                             "impression: the image files of the table, relative to "
                             "--dataset_dir, with their numbers as the table is to hold them, "
                             "0 or more (default: none, the table then gets all image files "
                             "below --dataset_dir, which must be named <subject>_<impression> "
                             "with numbers from 1)")
    sensor_normalization.add_arguments(parser)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes that load the images (default: 1, "
                             "the images are loaded in this process)")
    parser.add_argument("--allow_other_extraction_settings", action="store_true",
                        help="Build although extraction_settings.json in --minutiae_dir holds "
                             "other resolution settings than this build uses; every .mnt file "
                             "is then accepted or refused by its own checks (default: such a "
                             "build is refused)")
    parser.add_argument("--check", type=str, default=None, metavar="TABLE",
                        help="Read the table in this directory with every rule of the format "
                             "and print what it holds, instead of building one. No other "
                             "option goes with it (default: none, a table is built)")

    # Which options were given, also where one was given the value of its
    # default: the defaults are withheld while the command line is parsed
    defaults = vars(parser.parse_args([]))
    parser.set_defaults(**dict.fromkeys(defaults, _NOT_GIVEN))
    args = parser.parse_args()
    given = [name for name in defaults if getattr(args, name) is not _NOT_GIVEN]
    for name in defaults:
        if name not in given:
            setattr(args, name, defaults[name])

    try:
        if "check" in given:
            if len(given) > 1:
                others = ", ".join(f"--{name}" for name in given if name != "check")
                parser.error(f"--check goes with no other option (given: {others})")
            return check_table(args.check)

        missing = [name for name in ("dataset_dir", "samples") if name not in given]
        if missing:
            parser.error(
                "a build needs --dataset_dir and --samples (missing: "
                f"{', '.join('--' + name for name in missing)}); to check a table that exists, "
                "give --check alone"
            )
        if args.workers < 1:
            parser.error("--workers must be at least 1")
        if not os.path.isdir(args.dataset_dir):
            parser.error(f"--dataset_dir {args.dataset_dir} is not a directory")
        if args.minutiae_dir is not None and not os.path.isdir(args.minutiae_dir):
            parser.error(f"--minutiae_dir {args.minutiae_dir} is not a directory")
        if args.sample_list is not None and not os.path.isfile(args.sample_list):
            parser.error(f"--sample_list {args.sample_list} is not a file")
        if args.allow_other_extraction_settings and args.minutiae_dir is None:
            parser.error(
                "--allow_other_extraction_settings has no meaning without --minutiae_dir: the "
                "settings of the extraction are compared only where .mnt files are read"
            )
        for name in ("dataset_dir", "minutiae_dir"):
            directory = getattr(args, name)
            if directory is not None and _is_inside(args.samples, directory):
                parser.error(
                    f"--samples {args.samples} lies in --{name} {directory}. Nothing is written "
                    "into the directories of the images and of the minutiae: give a directory "
                    "outside of them"
                )
        try:
            normalizer = sensor_normalization.from_arguments(args, args.dataset_dir)
        except OSError as error:
            parser.error(f"--ppi_manifest {args.ppi_manifest} cannot be read ({error})")
        except ValueError as error:
            parser.error(str(error))
        return build_table(args, normalizer)
    except BuildRefused as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        status = main()
    except Exception:
        # An error that is no refusal. Without this the interpreter would
        # end with the status 1, which stands for a table that was written.
        # It would as well if the traceback raised an error of its own,
        # where standard error cannot be written to any more (its reader
        # went away, or its terminal was closed).
        with contextlib.suppress(Exception):
            traceback.print_exc()
        status = 2
    sys.exit(status)
