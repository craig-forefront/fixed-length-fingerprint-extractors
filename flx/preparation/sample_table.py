"""
The sample table: a training set described as a table with one row per
single-finger image file. A row says which finger and which capture the file
shows, how the file is prepared (resolution, ridge polarity, sizes) and which
minutiae the prepared image has. The table is a directory of Parquet files.
The pixels stay in the image files.

Why: without a table these facts come from three places. The identity of an
image is read from its file name, which only fits sets named
<subject>_<impression> with the same number of impressions for every subject.
Resolution and polarity come from options that have to be given again exactly
as the minutiae extraction got them. The minutiae are one .mnt file per
image. A table states all of it once, row by row, and can describe any file
names, any number of impressions per finger and numbers with gaps.

This module defines the table (schema version 1), writes its parts, reads a
table into flat numpy arrays, validates it and computes its checksum. Like the
other modules of flx/preparation it needs no torch. pyarrow is imported only
inside the functions that read or write Parquet files (read_samples and
write_part); the import of the module and everything else work without it.
How a table is built and used is described in docs/sample-table.md.

A table on disk is a directory with the parts of one build:
    part-00000-of-00003.parquet
    part-00001-of-00003.parquet
    part-00002-of-00003.parquet
A part is written under a temporary name and renamed when it is complete, and
a table is only read if every part of its build is there, so a table that can
be read is whole. Neither the split of the rows into parts nor their order
inside a part means anything.

The pages of a part, which hold its rows, are written with their checksums
(CRC-32, a field of the Parquet format), and the checksums are compared when
the rows are read. A part whose rows were damaged after it was written is
refused this way; without the checksums most such damage would be read as
other values. A page that carries no checksum, as in a part that another tool
wrote, is read without that comparison.

Columns (Arrow type; only minutiae can be null):
    subject        int32    Number of the finger: all images of one finger
                            have it. 0 or more; gaps are allowed.
    impression     int32    Number of the capture among the images of the
                            finger. 0 or more and unique within the subject;
                            a finger can have any number of impressions.
    image_path     string   Path of the image file relative to the root
                            directory of the images, with / as separator and
                            normalised: no ".", ".." or empty component and
                            no / at the start. Unique in the table.
    file_size      int64    Size of the image file in bytes when the table
                            was built.
    ppi            float64  Resolution of the file in pixels per inch, as
                            the preparation took it. 64 bits, because the
                            value rounded to 32 bits gives another size of
                            the prepared image for some files (785 pixels at
                            251.2 ppi become 1562 pixels, and 1563 with the
                            rounded value).
    ppi_origin     string   Where ppi comes from: "argument", "manifest",
                            "metadata" or "assumed".
    ridges         string   What the file was taken to hold: "dark" or
                            "light" ridges. Never "auto".
    ridges_origin  string   Where ridges comes from: "argument", "manifest",
                            "estimated" or "default".
    source_width, source_height
                   int32    Size of the image file in pixels.
    width, height  int32    Size in pixels of the prepared image: the file
                            resampled to the target resolution. It is always
                            resampled_size(source_width, source_height, ppi,
                            target resolution) of sensor_normalization.py.
    minutiae       list<struct<x: float32, y: float32, angle: float32>>
                            The minutiae of the prepared image. x is the
                            column and y the row in its pixels (y pointing
                            down), angle the direction in radians: the
                            direction vector is (cos(angle), sin(angle)) in
                            these axes. Null: no minutiae are known for the
                            image. An empty list: the image has none.
subject and impression are the two numbers of flx.data.dataset.Identifier.
ppi, ridges and the two origins are what PreparedImageInfo of
sensor_normalization.py holds for the file. Meaning and unit of every column
are also stored in the parts themselves, as the field metadata "description"
of the Arrow schema.

File metadata of every part (key-value metadata of the Parquet file):
    flx.table           "samples"
    flx.schema_version  "1"
    flx.target_ppi      Resolution of the prepared images, "500.0"
    flx.build           A JSON object about the build that wrote the part:
                        "id" (the same text in all parts of one build),
                        "part" (index of the part, from 0), "parts" (number
                        of parts), "rows" (rows of the part), and whatever
                        the builder adds (see write_part).

read_samples refuses, with a SampleTableError that names the directory or
the part:
    - a directory without parts; a part that cannot be read as a Parquet
      file (its footer, or its rows when they are read), lacks this metadata
      or is another kind of table; a part of another schema version;
    - a part in which a column of the list above is missing, is there twice
      or has another type. Nothing is cast: a float32 ppi is refused. Two
      things are not compared: the name of the item field of the minutiae
      list, because Parquet writers name it differently, and whether a
      field is declared as never null;
    - a null in a column other than minutiae, or inside a list of minutiae;
    - a text other than the values above in ppi_origin, ridges or
      ridges_origin;
    - parts that are not the parts 0 to parts - 1 of one build, each exactly
      once (a part is missing or there twice, parts of two builds, parts
      that state different numbers of parts), a part whose file name says
      something else than its metadata, and a part that holds another
      number of rows than its metadata state;
    - parts whose target resolutions or whose "settings" of the build
      differ, and a target resolution other than TARGET_PPI of
      sensor_normalization.py;
    - a table that validate_samples refuses.
Columns that are not in the list above are not read. They are no error; their
names are returned as Samples.unknown_columns.

validate_samples refuses a table in which
    - a subject or an impression is negative;
    - two rows have the same (subject, impression), or the same image_path;
    - an image_path is not valid UTF-8, is empty, holds a backslash or a NUL
      character, starts with /, or has a component that is ".", ".." or
      empty (so it is equal to normalize_image_path of itself and cannot
      leave the root directory of the images);
    - a ppi is not finite or not above 0, or a file_size is negative;
    - one of the four sizes is not above 0, or (width, height) is not
      resampled_size(source_width, source_height, ppi, target resolution);
    - ppi_origin or ridges_origin is none of the values above;
    - a minutia holds a value that is not finite;
and a table whose target resolution is not a finite number above 0. The
message names the first rows concerned by image_path, subject and impression,
and says how many there are.

The checksum (samples_checksum) is the SHA-256, as hexadecimal text, of these
bytes, with the rows in the order of (subject, impression):
    1. the line "flx-samples-1\\n" (ASCII);
    2. the number of rows as int64 and the target resolution as float64;
    3. the columns one after the other, each as one block with a value per
       row: subject and impression as int32, file_size as int64, ppi as
       float64, ridges as one byte (0 dark, 1 light), source_width,
       source_height, width and height as int32;
    4. the length in bytes of every image_path as int32, then the UTF-8
       bytes of all paths, row after row;
    5. one byte per row that is 1 where minutiae are known and 0 where the
       cell is null, the number of minutiae of every row as int32, then all
       minutiae as float32: x, y and angle of the first minutia, x, y and
       angle of the second, and so on.
All numbers are little-endian. Not included: ppi_origin, ridges_origin,
unknown columns, the file metadata, and how the rows are split into parts or
ordered in them. The checksum is computed from the values and not from the
files, so it does not change when the same table is written again.
"""

import dataclasses
import errno
import hashlib
import itertools
import json
import operator
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np

from flx.preparation.sensor_normalization import RIDGES_DARK, RIDGES_LIGHT, TARGET_PPI

SCHEMA_VERSION = 1

# Value of the metadata key flx.table: what tells a part of a sample table
# from any other Parquet file
TABLE_NAME = "samples"

# The values of the columns ppi_origin and ridges_origin, as PreparedImageInfo
# holds them. Samples holds the index of the value instead of the text.
PPI_ORIGINS = ("argument", "manifest", "metadata", "assumed")
RIDGES_ORIGINS = ("argument", "manifest", "estimated", "default")

# The columns of the table, in the order of the schema
COLUMNS = (
    "subject",
    "impression",
    "image_path",
    "file_size",
    "ppi",
    "ppi_origin",
    "ridges",
    "ridges_origin",
    "source_width",
    "source_height",
    "width",
    "height",
    "minutiae",
)

# The requirement as the extra "table" of pyproject.toml and requirements.txt
# state it. It is repeated here for the message that says how to install it.
# Not below 17.0: with pyarrow 16.1 some damaged parts end the interpreter
# with a segmentation fault inside of pyarrow when their rows are read, where
# 17.0 raises an error that read_samples turns into a refusal.
PYARROW_REQUIREMENT = "pyarrow>=17.0"

_KEY_TABLE = "flx.table"
_KEY_VERSION = "flx.schema_version"
_KEY_TARGET_PPI = "flx.target_ppi"
_KEY_BUILD = "flx.build"

# Meaning and unit of every field, stored as its metadata "description", so
# that a part can be understood without this repository
_DESCRIPTIONS = {
    "subject": (
        "Number of the finger: all images of one finger have the same subject. 0 or more; "
        "the numbers need not be consecutive."
    ),
    "impression": (
        "Number of the capture among the images of its finger. 0 or more and unique within "
        "the subject; a finger can have any number of impressions."
    ),
    "image_path": (
        "Path of the image file relative to the root directory of the images, with / as "
        "separator and without '.', '..' or empty components. Unique in the table."
    ),
    "file_size": "Size of the image file in bytes when the table was built.",
    "ppi": (
        "Resolution of the image file in pixels per inch, as it was taken when the image "
        "was prepared."
    ),
    "ppi_origin": (
        "Where ppi comes from: 'argument' (one value given for all files), 'manifest' (a "
        "list of resolutions per file or directory), 'metadata' (stated by the image file "
        "itself) or 'assumed' (an assumption for files that state no resolution, or one that "
        "is not plausible for a fingerprint sensor)."
    ),
    "ridges": (
        "What the image file was taken to hold: 'dark' (dark ridges on a light background) "
        "or 'light' (light ridges on a dark background; such an image is inverted when it "
        "is prepared)."
    ),
    "ridges_origin": (
        "Where ridges comes from: 'argument' (one value given for all files), 'manifest' (a "
        "list per file or directory), 'estimated' (from the image) or 'default' (dark, "
        "because the estimate gave no decision)."
    ),
    "source_width": "Width of the image file in pixels.",
    "source_height": "Height of the image file in pixels.",
    "width": (
        "Width in pixels of the prepared image: the image file resampled from ppi to the "
        "target resolution (file metadata flx.target_ppi, in pixels per inch)."
    ),
    "height": (
        "Height in pixels of the prepared image: the image file resampled from ppi to the "
        "target resolution (file metadata flx.target_ppi, in pixels per inch)."
    ),
    "minutiae": (
        "The minutiae of the prepared image. Null: no minutiae are known for the image. An "
        "empty list: the image has no minutiae."
    ),
    "x": "Column of the minutia in pixels of the prepared image, counted from its left edge.",
    "y": (
        "Row of the minutia in pixels of the prepared image, counted from its top edge (y "
        "points down)."
    ),
    "angle": (
        "Direction of the minutia in radians: the direction vector is (cos(angle), "
        "sin(angle)) in the axes of x and y."
    ),
}

# The script that writes tables, for the messages that say what to do
_BUILDER = "scripts/build_sample_table.py"

_PART_NAME = re.compile(r"part-(\d{5,})-of-(\d{5,})\.parquet")

_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1

# How many rows, parts or texts a message names before it gives the number of the rest
_MAX_NAMED = 5

# Rows that are taken from a Parquet file at one time and copied into the
# numpy arrays before the next ones are asked for
_BATCH_ROWS = 16384

# Bytes of image paths, values of minutiae and items of rows that are looked
# at or moved in one piece, so that temporary arrays stay small beside the table
_BLOCK_BYTES = 1 << 20
_BLOCK_VALUES = 1 << 22
_BLOCK_ITEMS = 1 << 20

# Base of the hash that finds equal image paths (an odd number, so that it
# has an inverse modulo 2**64)
_HASH_BASE = 0x9E3779B97F4A7C15

_SLASH, _BACKSLASH, _DOT = ord("/"), ord("\\"), ord(".")


class SampleTableError(Exception):
    """A sample table cannot be used, or pyarrow is missing where a table is read or written."""


@dataclass(frozen=True, eq=False)
class SampleRow:
    """
    One row of a sample table as Python values: what the builder of a table
    knows about one image file. See the module docstring for the columns.
    """

    subject: int
    impression: int
    image_path: str
    file_size: int
    ppi: float
    ppi_origin: str  # one of PPI_ORIGINS
    ridges: str  # "dark" or "light"
    ridges_origin: str  # one of RIDGES_ORIGINS
    source_width: int
    source_height: int
    width: int
    height: int
    # Array (k, 3) of x, y, angle. k can be 0: the image has no minutiae.
    # None: no minutiae are known for the image (a null cell).
    minutiae: Optional[np.ndarray]


# Type of every array of Samples
_ARRAY_TYPES = {
    "subject": np.int32,
    "impression": np.int32,
    "path_data": np.uint8,
    "path_offsets": np.int64,
    "file_size": np.int64,
    "ppi": np.float64,
    "ppi_origin": np.uint8,
    "ridges_origin": np.uint8,
    "ridges_light": np.bool_,
    "source_width": np.int32,
    "source_height": np.int32,
    "width": np.int32,
    "height": np.int32,
    "minutiae_known": np.bool_,
    "minutiae_offsets": np.int64,
    "minutiae": np.float32,
}
# The arrays that hold something else than one value per row
_NOT_PER_ROW = ("path_data", "path_offsets", "minutiae_offsets", "minutiae")


@dataclass(eq=False, repr=False)
class Samples:
    """
    The rows of a sample table as flat numpy arrays, always in the order of
    (subject, impression), whatever the order in the files was. n is the
    number of rows and m the number of all minutiae.

    Texts and minutiae are not held as one object per row: the image paths
    are one buffer of bytes and the minutiae one array, each with the offsets
    of the rows. A table of millions of rows is a few large arrays this way
    and not millions of Python objects.

    Every array is read-only: it is a view whose writeable flag is off, so
    that a table is not changed through it by accident after it was
    validated. That also holds for a copy of the table (pickle, copy.copy,
    copy.deepcopy), which is made like every other Samples. It is a guard
    against accidents and no more: numpy lets whoever means to set the flag
    of such a view back, where the memory below it can be written.

    Making a Samples checks the types, the lengths and the order of the rows
    (ValueError); the rules of the table are checked by validate_samples.
    """

    subject: np.ndarray  # int32 (n,)
    impression: np.ndarray  # int32 (n,)
    path_data: np.ndarray  # uint8: the UTF-8 bytes of all image paths, row after row
    path_offsets: np.ndarray  # int64 (n + 1,): row i is path_data[offsets[i]:offsets[i + 1]]
    file_size: np.ndarray  # int64 (n,)
    ppi: np.ndarray  # float64 (n,)
    ppi_origin: np.ndarray  # uint8 (n,): index into PPI_ORIGINS
    ridges_origin: np.ndarray  # uint8 (n,): index into RIDGES_ORIGINS
    ridges_light: np.ndarray  # bool (n,): True for light ridges, False for dark
    source_width: np.ndarray  # int32 (n,)
    source_height: np.ndarray  # int32 (n,)
    width: np.ndarray  # int32 (n,)
    height: np.ndarray  # int32 (n,)
    minutiae_known: np.ndarray  # bool (n,): False where the cell is null
    minutiae_offsets: np.ndarray  # int64 (n + 1,): row i is minutiae[offsets[i]:offsets[i + 1]]
    minutiae: np.ndarray  # float32 (m, 3): x, y, angle
    target_ppi: float = TARGET_PPI  # resolution of the prepared images
    # What the builder recorded (flx.build), merged over the parts by read_samples
    build: dict = field(default_factory=dict)
    # Names of the columns of the files that this code does not know
    unknown_columns: tuple = ()

    def __post_init__(self):
        for name, dtype in _ARRAY_TYPES.items():
            array = getattr(self, name)
            dimensions = 2 if name == "minutiae" else 1
            if not isinstance(array, np.ndarray) or array.dtype != dtype:
                found = array.dtype if isinstance(array, np.ndarray) else type(array).__name__
                raise ValueError(
                    f"Samples.{name} must be an array of {np.dtype(dtype)}, got {found}"
                )
            if array.ndim != dimensions or (dimensions == 2 and array.shape[1] != 3):
                expected = "the shape (m, 3)" if dimensions == 2 else "one dimension"
                raise ValueError(f"Samples.{name} must have {expected}, got {array.shape}")
            # A view of its own, so that the array of the caller keeps its flags
            array = np.ascontiguousarray(array).view()
            array.flags.writeable = False
            setattr(self, name, array)
        rows = len(self.subject)
        for name in _ARRAY_TYPES:
            if name not in _NOT_PER_ROW and len(getattr(self, name)) != rows:
                raise ValueError(
                    f"Samples.{name} holds {len(getattr(self, name))} values for {rows} rows"
                )
        _check_offsets("path_offsets", self.path_offsets, rows, len(self.path_data))
        _check_offsets("minutiae_offsets", self.minutiae_offsets, rows, len(self.minutiae))
        if np.any(np.diff(self.minutiae_offsets)[~self.minutiae_known]):
            raise ValueError("A row whose minutiae are not known (a null cell) holds minutiae")
        if _sorting_order(self.subject, self.impression) is not None:
            raise ValueError(
                "The rows of Samples must be in the order of (subject, impression); "
                "samples_from_rows puts them into it"
            )
        self.target_ppi = float(self.target_ppi)
        self.build = dict(self.build)
        self.unknown_columns = tuple(self.unknown_columns)
        self._keys = _lookup_keys(self.subject, self.impression)

    def __reduce__(self):
        # How pickle and the copy module make a copy: from the fields, like
        # every other Samples. numpy does not keep the writeable flag of an
        # array in a copy, so a copy made of the attributes as they are, which
        # is what Python does by itself, would have arrays that can be written to.
        return type(self), tuple(getattr(self, each.name) for each in dataclasses.fields(self))

    def __len__(self) -> int:
        return len(self.subject)

    def __repr__(self) -> str:
        return (
            f"Samples({len(self):,} rows, {len(self.minutiae):,} minutiae, "
            f"target_ppi={self.target_ppi:g})"
        )

    def _span(self, offsets: np.ndarray, row: int) -> tuple[int, int]:
        row = operator.index(row)
        if not 0 <= row < len(self):
            raise IndexError(f"row {row} of a table of {len(self)} rows")
        return int(offsets[row]), int(offsets[row + 1])

    def image_path(self, row: int) -> str:
        """The image path of a row: relative to the root directory of the images, with /."""
        start, end = self._span(self.path_offsets, row)
        return self.path_data[start:end].tobytes().decode("utf-8")

    def minutiae_of(self, row: int) -> np.ndarray:
        """
        The minutiae of a row as a float32 array (k, 3) of x, y, angle: a
        read-only view, not a copy. k is 0 where the image has no minutiae
        and where none are known; minutiae_known tells the two apart.
        """
        start, end = self._span(self.minutiae_offsets, row)
        return self.minutiae[start:end]

    def find(self, subject: int, impression: int) -> int:
        """The row of (subject, impression), or -1 if the table has no such row."""
        subject, impression = operator.index(subject), operator.index(impression)
        if not (_INT32_MIN <= subject <= _INT32_MAX and _INT32_MIN <= impression <= _INT32_MAX):
            return -1
        key = (subject << 32) + 2**31 + impression
        position = int(np.searchsorted(self._keys, key))
        if position < len(self._keys) and int(self._keys[position]) == key:
            return position
        return -1


def _check_offsets(name: str, offsets: np.ndarray, rows: int, items: int) -> None:
    if len(offsets) != rows + 1:
        raise ValueError(
            f"Samples.{name} holds {len(offsets)} values for {rows} rows; it needs one more "
            "than there are rows"
        )
    if offsets[0] != 0 or offsets[-1] != items or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(
            f"Samples.{name} must go from 0 to the number of items ({items}) and never fall"
        )


def _sorting_order(subject: np.ndarray, impression: np.ndarray) -> Optional[np.ndarray]:
    """
    None if the rows are in the order of (subject, impression), else the
    order that sorts them. Rows with the same two numbers keep their order.
    """
    ascending = subject[1:] > subject[:-1]
    ascending |= (subject[1:] == subject[:-1]) & (impression[1:] >= impression[:-1])
    if ascending.all():
        return None
    return np.lexsort((impression, subject))


def _lookup_keys(subject: np.ndarray, impression: np.ndarray) -> np.ndarray:
    """
    One 64-bit number per row that orders the rows as (subject, impression)
    does: the subject in the upper half and the impression in the lower. The
    impression is moved up by 2**31, so that every pair of 32-bit numbers
    gets a key of its own and the order also holds for negative numbers
    (which a table must not hold, but which a table can be asked for).
    """
    keys = subject.astype(np.int64)
    keys <<= 32
    keys += 2**31
    keys += impression
    keys.flags.writeable = False
    return keys


def normalize_image_path(text: str) -> str:
    """
    The form in which a table holds the path of an image: os.path.normpath
    with / as separator, so "./a//b/../1_1.png" becomes "a/1_1.png". The
    result is relative to the same directory as the argument; whether it
    stays inside of it is checked by validate_samples.
    """
    return os.path.normpath(os.fspath(text)).replace(os.sep, "/")


def _row_text(row: SampleRow) -> str:
    return f"Row {row.image_path!r} (subject {row.subject}, impression {row.impression})"


def _whole_number(row: SampleRow, name: str, low: int, high: int) -> int:
    value = getattr(row, name)
    try:
        number = operator.index(value)
    except TypeError:
        raise SampleTableError(
            f"{_row_text(row)}: {name} must be a whole number, got {value!r}"
        ) from None
    if not low <= number <= high:
        raise SampleTableError(
            f"{_row_text(row)}: {name} is {number}; the column holds numbers from {low} to {high}"
        )
    return number


def _index_of(row: SampleRow, name: str, values: tuple) -> int:
    value = getattr(row, name)
    if value not in values:
        raise SampleTableError(
            f"{_row_text(row)}: {name} is {value!r}; it must be one of "
            + ", ".join(repr(v) for v in values)
        )
    return values.index(value)


def samples_from_rows(
    rows: Iterable[SampleRow], target_ppi: float = TARGET_PPI, build: Optional[dict] = None
) -> Samples:
    """
    The rows as Samples, in the order of (subject, impression) whatever
    order they are given in.

    The rules of a table are not checked here (see validate_samples). Only
    what the arrays cannot hold raises SampleTableError, with the row named:
    a number outside of the integer type of its column, a text that is none
    of the values of its column, an image path that cannot be written as
    UTF-8, minutiae that are not an array (k, 3).

    Args:
        rows: The rows, in any order
        target_ppi: Resolution of the prepared images that width and height
            of the rows refer to
        build: What to keep as Samples.build (write_part takes the build
            information of a part as an argument of its own)
    """
    rows = list(rows)
    count = len(rows)
    integers = {
        name: np.empty(count, dtype=_ARRAY_TYPES[name])
        for name in (
            "subject",
            "impression",
            "file_size",
            "source_width",
            "source_height",
            "width",
            "height",
        )
    }
    limits = {
        name: (int(np.iinfo(array.dtype).min), int(np.iinfo(array.dtype).max))
        for name, array in integers.items()
    }
    ppi = np.empty(count, dtype=np.float64)
    ppi_origin = np.empty(count, dtype=np.uint8)
    ridges_origin = np.empty(count, dtype=np.uint8)
    ridges_light = np.empty(count, dtype=np.bool_)
    minutiae_known = np.empty(count, dtype=np.bool_)
    paths, minutiae = [], []
    for position, row in enumerate(rows):
        for name, array in integers.items():
            array[position] = _whole_number(row, name, *limits[name])
        try:
            ppi[position] = float(row.ppi)
        except (TypeError, ValueError):
            raise SampleTableError(
                f"{_row_text(row)}: ppi must be a number, got {row.ppi!r}"
            ) from None
        ppi_origin[position] = _index_of(row, "ppi_origin", PPI_ORIGINS)
        ridges_origin[position] = _index_of(row, "ridges_origin", RIDGES_ORIGINS)
        # Never "auto": a row holds what the image was taken to hold
        ridges_light[position] = _index_of(row, "ridges", (RIDGES_DARK, RIDGES_LIGHT)) == 1
        try:
            paths.append(row.image_path.encode("utf-8"))
        except (AttributeError, UnicodeEncodeError):
            raise SampleTableError(
                f"{_row_text(row)}: image_path must be text that can be written as UTF-8. "
                "A file name that is not valid UTF-8 cannot be put into a table: rename the file."
            ) from None
        minutiae_known[position] = row.minutiae is not None
        if row.minutiae is None:
            minutiae.append(np.zeros((0, 3), dtype=np.float32))
            continue
        try:
            values = np.asarray(row.minutiae, dtype=np.float32)
        except (TypeError, ValueError):
            values = None
        if values is None or values.ndim != 2 or values.shape[1] != 3:
            found = "no array of numbers" if values is None else f"an array {values.shape}"
            raise SampleTableError(
                f"{_row_text(row)}: minutiae must be an array (k, 3) of x, y and angle, or None "
                f"where no minutiae are known; got {found}"
            )
        minutiae.append(values)

    order = np.lexsort((integers["impression"], integers["subject"]))
    path_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.array([len(paths[i]) for i in order], dtype=np.int64), out=path_offsets[1:])
    minutiae_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.array([len(minutiae[i]) for i in order], dtype=np.int64), out=minutiae_offsets[1:])
    return Samples(
        subject=integers["subject"][order],
        impression=integers["impression"][order],
        path_data=np.frombuffer(b"".join(paths[i] for i in order), dtype=np.uint8),
        path_offsets=path_offsets,
        file_size=integers["file_size"][order],
        ppi=ppi[order],
        ppi_origin=ppi_origin[order],
        ridges_origin=ridges_origin[order],
        ridges_light=ridges_light[order],
        source_width=integers["source_width"][order],
        source_height=integers["source_height"][order],
        width=integers["width"][order],
        height=integers["height"][order],
        minutiae_known=minutiae_known[order],
        minutiae_offsets=minutiae_offsets,
        minutiae=np.concatenate(
            [np.zeros((0, 3), dtype=np.float32)] + [minutiae[i] for i in order]
        ),
        target_ppi=target_ppi,
        build={} if build is None else build,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _row_name(samples: Samples, row: int) -> str:
    """A row as a message names it: image path, subject and impression."""
    start, end = int(samples.path_offsets[row]), int(samples.path_offsets[row + 1])
    # A path that is not valid UTF-8 is shown with its bytes escaped
    path = samples.path_data[start:end].tobytes().decode("utf-8", errors="backslashreplace")
    return f"{path!r} (subject {samples.subject[row]}, impression {samples.impression[row]})"


def _refuse(samples: Samples, concerned: np.ndarray, problem: str, remedy: str, detail=None):
    """
    Raises SampleTableError if a row is concerned (one flag per row). The
    message names the first of these rows and says how many there are;
    detail(row) adds what is wrong with a named row.
    """
    rows = np.flatnonzero(concerned)
    if len(rows) == 0:
        return
    named = []
    for row in rows[:_MAX_NAMED].tolist():
        named.append(_row_name(samples, row) + ("" if detail is None else f": {detail(row)}"))
    if len(rows) > _MAX_NAMED:
        named.append(f"and {len(rows) - _MAX_NAMED:,} more")
    raise SampleTableError(
        f"{problem} in {len(rows):,} of {len(samples):,} rows: {'; '.join(named)}. {remedy}"
    )


def _blocks(offsets: np.ndarray, size: int) -> Iterator[tuple[int, int]]:
    """
    Yields (first row, row after the last) of blocks of rows that follow
    each other and together hold all rows. By the offsets a block holds at
    most `size` items, or one row that alone holds more.
    """
    rows = len(offsets) - 1
    first = 0
    while first < rows:
        end = int(np.searchsorted(offsets, offsets[first] + size, side="right")) - 1
        end = min(max(end, first + 1), rows)
        yield first, end
        first = end


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _path_faults(samples: Samples) -> np.ndarray:
    """
    One flag per row: True where the image path breaks a rule for its text
    (see _path_fault for the rules). The rules are applied to the bytes of
    all paths at once. That is sound for UTF-8, in which the bytes of /,
    backslash, "." and NUL never occur inside of another character.
    """
    offsets, data = samples.path_offsets, samples.path_data
    lengths = np.diff(offsets)
    faulty = lengths == 0
    for first, end in _blocks(offsets, _BLOCK_BYTES):
        start, stop = int(offsets[first]), int(offsets[end])
        if start == stop:
            continue
        text = data[start:stop]
        filled = lengths[first:end] > 0
        starts = (offsets[first:end] - start)[filled]  # first byte of every path of the block
        ends = starts + lengths[first:end][filled] - 1  # and its last byte
        is_start = np.zeros(stop - start, dtype=np.bool_)
        is_start[starts] = True
        is_end = np.zeros(stop - start, dtype=np.bool_)
        is_end[ends] = True
        slash, dot = text == _SLASH, text == _DOT
        # A component begins at the start of its path or after a slash, and
        # it ends at the end of its path or before a slash. A byte that is
        # not the first of its path has its neighbour to the left in the
        # same path, and likewise to the right.
        begins = is_start.copy()
        begins[1:] |= slash[:-1]
        finishes = is_end.copy()
        finishes[:-1] |= slash[1:]
        fault = (text == _BACKSLASH) | (text == 0)
        # A slash where a component begins or as the last byte: a slash at
        # the start of the path or an empty component
        fault |= slash & (begins | is_end)
        fault |= dot & begins & finishes  # the component "."
        # The component "..": a dot that begins a component and a dot behind
        # it that ends one. (If the two dots were in two paths, the first
        # would end its path and be the component "." of the line above.)
        fault[:-1] |= dot[:-1] & begins[:-1] & dot[1:] & finishes[1:]
        rows_of_block = np.flatnonzero(filled)
        positions = np.flatnonzero(fault)
        faulty[first + rows_of_block[np.searchsorted(starts, positions, side="right") - 1]] = True
        # All paths of the block are valid UTF-8 exactly if the bytes of the
        # block are and no path starts inside of a character: there a
        # character would be cut in two, and its halves are valid together
        # but not in their paths. Only then is each path looked at alone.
        if np.any(is_start & ((text & 0xC0) == 0x80)) or not _is_utf8(text.tobytes()):
            for row in range(first, end):
                if not _is_utf8(data[offsets[row] : offsets[row + 1]].tobytes()):
                    faulty[row] = True
    return faulty


def _path_fault(samples: Samples, row: int) -> str:
    """What is wrong with the image path of a row that _path_faults found."""
    data = samples.path_data[samples.path_offsets[row] : samples.path_offsets[row + 1]].tobytes()
    if not data:
        return "it is empty"
    if not _is_utf8(data):
        return "it is not valid UTF-8"
    if b"\x00" in data:
        return "it holds a NUL character"
    if b"\\" in data:
        return "it holds a backslash (the separator is /)"
    if data.startswith(b"/"):
        return "it starts with / (it must be relative)"
    components = data.split(b"/")
    if b".." in components:
        return "it has the component '..'"
    if b"." in components:
        return "it has the component '.'"
    return "it has an empty component (// or a / at the end)"


def _powers(base: int, count: int) -> np.ndarray:
    """base**0, base**1, ... (count values) modulo 2**64."""
    powers = np.full(count, base, dtype=np.uint64)
    powers[:1] = 1
    return np.cumprod(powers, out=powers)


def _path_hashes(samples: Samples) -> np.ndarray:
    """
    A 64-bit number per row that is the same for rows with the same image
    path: the sum of (byte + 1) * base**(position in the path), modulo
    2**64. Different paths can have the same number, so rows with the same
    number are compared byte by byte (_path_duplicates).
    """
    offsets, data = samples.path_offsets, samples.path_data
    lengths = np.diff(offsets)
    hashes = np.zeros(len(samples), dtype=np.uint64)
    powers = inverse_powers = np.zeros(0, dtype=np.uint64)
    for first, end in _blocks(offsets, _BLOCK_BYTES):
        start, stop = int(offsets[first]), int(offsets[end])
        if start == stop:
            continue
        if stop - start > len(powers):
            powers = _powers(_HASH_BASE, stop - start)
            inverse_powers = _powers(pow(_HASH_BASE, -1, 2**64), stop - start)
        filled = lengths[first:end] > 0
        starts = (offsets[first:end] - start)[filled]
        # Every byte is weighted by the power of its position in the block,
        # and the sum of a path is brought back to the power of its position
        # in the path by the inverse power of where the path starts
        weighted = data[start:stop].astype(np.uint64)
        weighted += 1
        weighted *= powers[: stop - start]
        sums = np.add.reduceat(weighted, starts)
        sums *= inverse_powers[starts]
        hashes[first + np.flatnonzero(filled)] = sums
    return hashes


def _same_paths(samples: Samples, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """One flag per pair of rows (first[i], second[i]): True where the image paths are equal."""
    offsets, data = samples.path_offsets, samples.path_data
    lengths = offsets[first + 1] - offsets[first]
    same = lengths == offsets[second + 1] - offsets[second]
    pairs = np.flatnonzero(same & (lengths > 0))
    # Offsets of the pairs to compare, as if their paths stood one after the other
    pair_offsets = np.zeros(len(pairs) + 1, dtype=np.int64)
    np.cumsum(lengths[pairs], out=pair_offsets[1:])
    for begin, end in _blocks(pair_offsets, _BLOCK_BYTES):
        block = pairs[begin:end]
        counts = lengths[block]
        starts = pair_offsets[begin:end] - pair_offsets[begin]
        # Position of every byte of the block in its path
        within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(starts, counts)
        differs = (
            data[np.repeat(offsets[first[block]], counts) + within]
            != data[np.repeat(offsets[second[block]], counts) + within]
        )
        same[block] = np.add.reduceat(differs.astype(np.int64), starts) == 0
    return same


def _path_duplicates(samples: Samples) -> np.ndarray:
    """One flag per row: True where another row has the same image path."""
    rows = len(samples)
    duplicate = np.zeros(rows, dtype=np.bool_)
    hashes = _path_hashes(samples)
    in_order = np.sort(hashes)
    if not np.any(in_order[1:] == in_order[:-1]):
        return duplicate  # all hashes differ, so all paths do
    # Rows with the same hash follow each other in `order`. Each of them is
    # compared with the first row of its run.
    order = np.argsort(hashes, kind="stable")
    in_order = hashes[order]
    repeats = np.flatnonzero(in_order[1:] == in_order[:-1]) + 1
    run_start = np.zeros(rows, dtype=np.int64)
    is_first = np.ones(rows, dtype=np.bool_)
    is_first[repeats] = False
    run_start[is_first] = np.flatnonzero(is_first)
    run_start = np.maximum.accumulate(run_start)
    leaders, followers = order[run_start[repeats]], order[repeats]
    same = _same_paths(samples, leaders, followers)
    duplicate[leaders[same]] = True
    duplicate[followers[same]] = True
    # What is left has the hash of the first row of its run but another
    # path. Among each other these rows are compared as Python values; there
    # are none unless different paths have the same hash.
    seen = {}
    for row in followers[~same].tolist():
        path = samples.path_data[samples.path_offsets[row] : samples.path_offsets[row + 1]]
        seen.setdefault(path.tobytes(), []).append(row)
    for group in seen.values():
        if len(group) > 1:
            duplicate[group] = True
    return duplicate


def _not_finite(samples: Samples) -> np.ndarray:
    """One flag per row: True where a minutia of the row holds a value that is not finite."""
    concerned = np.zeros(len(samples), dtype=np.bool_)
    values = samples.minutiae
    step = max(1, _BLOCK_VALUES // 3)
    for start in range(0, len(values), step):
        finite = np.isfinite(values[start : start + step])
        if finite.all():
            continue
        minutia = start + np.flatnonzero(~finite.all(axis=1))
        concerned[np.searchsorted(samples.minutiae_offsets, minutia, side="right") - 1] = True
    return concerned


def validate_samples(samples: Samples) -> None:
    """
    Checks the rules of a table (see the module docstring) and raises
    SampleTableError for the first rule that rows break. The message names
    the first of these rows by image path, subject and impression, says how
    many there are and what to do. read_samples calls this, so a table that
    was read is valid.
    """
    target_ppi = samples.target_ppi
    if not (np.isfinite(target_ppi) and target_ppi > 0):
        raise SampleTableError(
            f"The target resolution of the table is {target_ppi!r} ppi; it must be a positive "
            "number. It is the resolution of the prepared images that width and height refer to."
        )
    rebuild = f"Correct the rows where the table is made and build it again ({_BUILDER})."

    _refuse(
        samples,
        (samples.subject < 0) | (samples.impression < 0),
        "A negative subject or impression",
        f"Both numbers must be 0 or more: count fingers and captures from 0. {rebuild}",
    )

    # The rows are in the order of (subject, impression): equal ones are neighbours
    same = (samples.subject[1:] == samples.subject[:-1]) & (
        samples.impression[1:] == samples.impression[:-1]
    )
    repeated = np.zeros(len(samples), dtype=np.bool_)
    repeated[1:] |= same
    repeated[:-1] |= same
    _refuse(
        samples,
        repeated,
        "The same subject and impression",
        "Every image needs a pair of numbers of its own: give the captures of one finger "
        f"different impressions. {rebuild}",
    )

    _refuse(
        samples,
        _path_faults(samples),
        "An image_path that is not a normalised relative path",
        "An image_path is relative to the root directory of the images, has / as separator "
        "and no '.', '..' or empty component; normalize_image_path of this module gives that "
        f"form for a path below the root directory. {rebuild}",
        detail=lambda row: _path_fault(samples, row),
    )
    _refuse(
        samples,
        _path_duplicates(samples),
        "The same image_path",
        "Every row must name a file of its own: an image under two pairs of numbers would "
        f"count as two captures. {rebuild}",
    )

    with np.errstate(invalid="ignore"):
        bad_ppi = ~(np.isfinite(samples.ppi) & (samples.ppi > 0))
    _refuse(
        samples,
        bad_ppi,
        "A ppi that is not a positive number",
        f"The resolution of every file must be known and above 0. {rebuild}",
        detail=lambda row: f"ppi {float(samples.ppi[row])!r}",
    )
    _refuse(
        samples,
        samples.file_size < 0,
        "A negative file_size",
        f"file_size is the size of the image file in bytes. {rebuild}",
        detail=lambda row: f"file_size {samples.file_size[row]}",
    )

    def sizes(row: int) -> str:
        return (
            f"source {samples.source_width[row]}x{samples.source_height[row]}, "
            f"prepared {samples.width[row]}x{samples.height[row]}"
        )

    _refuse(
        samples,
        (samples.source_width <= 0)
        | (samples.source_height <= 0)
        | (samples.width <= 0)
        | (samples.height <= 0),
        "A size that is not above 0",
        f"Widths and heights are counted in pixels and an image has at least one. {rebuild}",
        detail=sizes,
    )

    expected_width, expected_height = _prepared_sizes(
        samples.source_width, samples.source_height, samples.ppi, target_ppi
    )

    def expected_sizes(row: int) -> str:
        return (
            f"source {samples.source_width[row]}x{samples.source_height[row]} at "
            f"{float(samples.ppi[row])!r} ppi gives {expected_width[row]:.0f}x"
            f"{expected_height[row]:.0f} at {target_ppi:g} ppi, the row holds "
            f"{samples.width[row]}x{samples.height[row]}"
        )

    _refuse(
        samples,
        (samples.width != expected_width) | (samples.height != expected_height),
        "A prepared size that does not follow from the source size and the ppi",
        "width and height must be what the preparation gives for the file (resampled_size "
        "of flx/preparation/sensor_normalization.py). A ppi that was rounded on the way into "
        f"the table, for example to 32 bits, has this effect. {rebuild}",
        detail=expected_sizes,
    )

    _refuse(
        samples,
        (samples.ppi_origin >= len(PPI_ORIGINS)) | (samples.ridges_origin >= len(RIDGES_ORIGINS)),
        "A ppi_origin or ridges_origin that is none of the values of its column",
        f"ppi_origin is an index into {PPI_ORIGINS} and ridges_origin one into "
        f"{RIDGES_ORIGINS}. {rebuild}",
        detail=lambda row: (
            f"ppi_origin {samples.ppi_origin[row]}, ridges_origin {samples.ridges_origin[row]}"
        ),
    )

    _refuse(
        samples,
        _not_finite(samples),
        "Minutiae with an x, y or angle that is not a finite number",
        f"Leave such a minutia out, or store null if the minutiae of the image are not "
        f"known. {rebuild}",
    )


def _prepared_sizes(
    source_width: np.ndarray, source_height: np.ndarray, ppi: np.ndarray, target_ppi: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    resampled_size of sensor_normalization.py for all rows at once, as two
    float64 arrays (width, height). The arithmetic is the same: the scale is
    target_ppi / ppi in 64 bits, a side times the scale is rounded with
    halves going to the even number (np.rint as round), and the result is at
    least 1. The sizes stay floating-point numbers, so that a row with a
    senseless ppi gives a size that no row holds instead of an error.
    """
    with np.errstate(all="ignore"):
        scale = float(target_ppi) / ppi
        return (
            np.maximum(1.0, np.rint(source_width * scale)),
            np.maximum(1.0, np.rint(source_height * scale)),
        )


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def _fits_int32(array: np.ndarray, what: str) -> np.ndarray:
    if len(array) and int(array.max()) > _INT32_MAX:
        raise SampleTableError(
            f"A row holds {int(array.max()):,} {what}; the checksum counts them in 32 bits"
        )
    return array


def samples_checksum(samples: Samples) -> str:
    """
    The SHA-256 of the values of the table as hexadecimal text; the module
    docstring defines the bytes. Two tables have the same checksum if they
    hold the same rows, however these are split into parts or ordered in
    the files. ppi_origin, ridges_origin, unknown columns and the build
    information do not count.
    """
    digest = hashlib.sha256()

    def add(array: np.ndarray, dtype: str) -> None:
        # The bytes of the array in this type. It is not copied where it has
        # the type already, as the large arrays have on a little-endian machine.
        digest.update(np.ascontiguousarray(array, dtype=np.dtype(dtype)))

    digest.update(b"flx-samples-1\n")
    digest.update(struct.pack("<qd", len(samples), samples.target_ppi))
    add(samples.subject, "<i4")
    add(samples.impression, "<i4")
    add(samples.file_size, "<i8")
    add(samples.ppi, "<f8")
    add(samples.ridges_light, "u1")
    for name in ("source_width", "source_height", "width", "height"):
        add(getattr(samples, name), "<i4")
    add(_fits_int32(np.diff(samples.path_offsets), "bytes of image_path"), "<i4")
    add(samples.path_data, "u1")
    add(samples.minutiae_known, "u1")
    add(_fits_int32(np.diff(samples.minutiae_offsets), "minutiae"), "<i4")
    add(samples.minutiae, "<f4")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Parquet files
# ---------------------------------------------------------------------------


def _import_pyarrow():
    """pyarrow and pyarrow.parquet, or SampleTableError with the way to install them."""
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as error:
        raise SampleTableError(
            "A sample table is read and written as Parquet files, which needs the package "
            f"pyarrow, and it cannot be imported ({error}). Install it in the root directory "
            'of this repository with: pip install -e ".[table]" or anywhere with: '
            f'pip install "{PYARROW_REQUIREMENT}" (with the quotes).'
        ) from error
    return pyarrow, pyarrow.parquet


def _open_file(pa, path: str, mode: str):
    """
    Opens the file at the path with pyarrow, to read it ("rb") or to write it
    anew ("wb"), and returns the open file. pq.ParquetFile and pq.write_table
    get it in the place of the path, and whoever opened it closes it: pyarrow
    does not close a file that it was handed. A directory at the path raises
    IsADirectoryError, as Python's open does.
    """
    # pyarrow takes the text of a path as UTF-8, whatever the encoding of the
    # process for the names of files is, and as a URI where it can pass for
    # one. Python turns the text back into the bytes that the name has on disk
    # (os.fsencode); that is how os.listdir and open find a file. With these
    # bytes pyarrow opens the same file. With the text it looks for another
    # one, or raises an error, where the path holds characters outside ASCII
    # in a process whose file names are not UTF-8, or bytes of a name that are
    # no UTF-8 in any process.
    if os.path.isdir(path):
        # Python's words for a directory, with the path as text. pyarrow would
        # refuse it before it opens anything, and show the bytes of the path.
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    return pa.OSFile(os.fsencode(path), mode)


def _no_file_name(directory: str, done: str, error: UnicodeEncodeError) -> SampleTableError:
    """
    The refusal for the directory of a table that the process cannot hand to
    the file system: the encoding in which it writes the names of files has
    no bytes for this text. `done` says what was to be done with the table.
    """
    encoding = sys.getfilesystemencoding()
    # The path with Python's escapes where the encoding has no bytes for it.
    # Standard output has the same encoding by default, and could not print
    # the message with the path as it is.
    shown = directory.encode(encoding, "backslashreplace").decode(encoding)
    if encoding == "utf-8":
        # UTF-8 has bytes for every character: what it cannot express is none
        remedy = (
            "The path holds half of a surrogate pair, which is no character and has no bytes "
            "in the name of a file: correct the path where it is made."
        )
    else:
        remedy = (
            "Start the script in an environment whose file names are UTF-8, for example with "
            "LC_ALL=C.UTF-8 or with PYTHONUTF8=1."
        )
    return SampleTableError(
        f"The sample table {shown} cannot be {done}: this process writes the names of "
        f"files in the encoding {encoding}, which cannot express the path ({error}). {remedy}"
    )


def part_name(index: int, count: int) -> str:
    """The file name of part `index` (from 0) of the `count` parts of a table."""
    return f"part-{index:05d}-of-{count:05d}.parquet"


def _arrow_schema(pa, target_ppi: float, build_text: str):
    """The Arrow schema of a part, with the description of every field and the file metadata."""

    def described(name: str, arrow_type, nullable: bool = False):
        return pa.field(
            name, arrow_type, nullable=nullable, metadata={"description": _DESCRIPTIONS[name]}
        )

    minutia = pa.struct([described(name, pa.float32()) for name in ("x", "y", "angle")])
    types = {
        "subject": pa.int32(),
        "impression": pa.int32(),
        "image_path": pa.string(),
        "file_size": pa.int64(),
        "ppi": pa.float64(),
        "ppi_origin": pa.string(),
        "ridges": pa.string(),
        "ridges_origin": pa.string(),
        "source_width": pa.int32(),
        "source_height": pa.int32(),
        "width": pa.int32(),
        "height": pa.int32(),
        # "element" is the name that the Parquet format gives the item of a list
        "minutiae": pa.list_(pa.field("element", minutia, nullable=False)),
    }
    return pa.schema(
        [described(name, types[name], nullable=name == "minutiae") for name in COLUMNS],
        metadata={
            _KEY_TABLE: TABLE_NAME,
            _KEY_VERSION: str(SCHEMA_VERSION),
            # repr gives the shortest text that reads back as the same number
            _KEY_TARGET_PPI: repr(float(target_ppi)),
            _KEY_BUILD: build_text,
        },
    )


def _texts(pa, indices: np.ndarray, values: tuple, name: str):
    """A string column from the index of the value of every row."""
    if len(indices) and int(indices.max()) >= len(values):
        raise SampleTableError(
            f"{name} of {int(np.count_nonzero(indices >= len(values))):,} rows is not an index "
            f"into {values}; such a table cannot be written"
        )
    return pa.array(np.array(values, dtype=object)[indices.astype(np.intp)], type=pa.string())


def _offsets32(pa, offsets: np.ndarray, what: str):
    """Offsets as the 32-bit buffer that Arrow strings and lists have."""
    if offsets[-1] > _INT32_MAX:
        raise SampleTableError(
            f"The rows hold {int(offsets[-1]):,} {what}, more than one part can hold "
            f"({_INT32_MAX:,}). Write the table in more parts."
        )
    return pa.py_buffer(offsets.astype(np.int32))


def _arrow_table(pa, samples: Samples, build_text: str):
    """The rows as an Arrow table with the schema of a part."""
    schema = _arrow_schema(pa, samples.target_ppi, build_text)
    rows = len(samples)
    minutiae_type = schema.field("minutiae").type
    minutia_values = pa.StructArray.from_arrays(
        # A column of the (m, 3) array is not contiguous: Arrow needs a copy of it
        [pa.array(np.ascontiguousarray(samples.minutiae[:, column])) for column in range(3)],
        fields=list(minutiae_type.value_type),
    )
    known = samples.minutiae_known
    # Bit i of the validity bitmap of Arrow is bit i % 8 of byte i // 8
    validity = None if known.all() else pa.py_buffer(np.packbits(known, bitorder="little"))
    columns = {
        "subject": pa.array(samples.subject, type=pa.int32()),
        "impression": pa.array(samples.impression, type=pa.int32()),
        "image_path": pa.StringArray.from_buffers(
            rows,
            _offsets32(pa, samples.path_offsets, "bytes of image paths"),
            pa.py_buffer(samples.path_data),
        ),
        "file_size": pa.array(samples.file_size, type=pa.int64()),
        "ppi": pa.array(samples.ppi, type=pa.float64()),
        "ppi_origin": _texts(pa, samples.ppi_origin, PPI_ORIGINS, "ppi_origin"),
        "ridges": _texts(
            pa, samples.ridges_light.astype(np.uint8), (RIDGES_DARK, RIDGES_LIGHT), "ridges"
        ),
        "ridges_origin": _texts(pa, samples.ridges_origin, RIDGES_ORIGINS, "ridges_origin"),
        "source_width": pa.array(samples.source_width, type=pa.int32()),
        "source_height": pa.array(samples.source_height, type=pa.int32()),
        "width": pa.array(samples.width, type=pa.int32()),
        "height": pa.array(samples.height, type=pa.int32()),
        "minutiae": pa.Array.from_buffers(
            minutiae_type,
            rows,
            [validity, _offsets32(pa, samples.minutiae_offsets, "minutiae")],
            null_count=int(rows - np.count_nonzero(known)),
            children=[minutia_values],
        ),
    }
    return pa.Table.from_arrays([columns[name] for name in COLUMNS], schema=schema)


def write_part(directory: str, index: int, count: int, samples: Samples, build: dict) -> str:
    """
    Writes the rows as part `index` (from 0) of the `count` parts of a table
    in the directory, which is created if it is missing, and returns the path
    of the file. The file appears under its name only when it is complete: it
    is written under a temporary name in the same directory and renamed, and
    a write that fails leaves neither of the two. An existing part of that
    name is replaced.

    The rows are written as they are; validate_samples is not called.

    Args:
        directory: Directory of the table
        index: Index of the part, from 0 to count - 1
        count: Number of parts of the table. A table is read only if all of
            them are there.
        samples: The rows of this part
        build: What to record about the build, as a dictionary of plain JSON
            values. "id" is needed: a text that is the same in all parts of
            one build and different in every other build, by which a reader
            tells the parts of two builds apart. "part", "parts" and "rows"
            are set here. Everything else is the builder's. When a table is
            read, "files" and "failed" (numbers, or lists) are added up over
            the parts, "settings" must be equal in all parts, and any other
            key is taken from part 0.

    Raises SampleTableError if pyarrow is not installed, if the rows cannot
    be written as one part, or if the encoding in which this process writes
    the names of files cannot express the directory. Arguments that make no
    sense raise ValueError.
    """
    pa, pq = _import_pyarrow()
    index, count = operator.index(index), operator.index(count)
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"A part needs an index from 0 to count - 1, got {index} of {count}")
    if not isinstance(build, dict) or not isinstance(build.get("id"), str) or not build["id"]:
        raise ValueError(
            'The build information must be a dictionary with a text under "id" that is the '
            f"same in all parts of the build, got {build!r}"
        )
    information = dict(build, part=index, parts=count, rows=len(samples))
    try:
        build_text = json.dumps(information, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"The build information must hold plain JSON values: {error}") from None
    table = _arrow_table(pa, samples, build_text)

    directory = os.fspath(directory)
    try:
        os.makedirs(directory, exist_ok=True)
    except UnicodeEncodeError as error:
        raise _no_file_name(directory, "written", error) from None
    path = os.path.join(directory, part_name(index, count))
    temporary = f"{path}.tmp{os.getpid()}"
    try:
        # With the checksum of every page, which _read_rows compares. pyarrow
        # gets the open file and not its path: see _open_file.
        with _open_file(pa, temporary, "wb") as sink:
            pq.write_table(table, sink, write_page_checksum=True)
        os.replace(temporary, path)
    except BaseException:
        # Nothing may be left of a part that was not written completely
        try:
            os.remove(temporary)
        except OSError:
            pass  # it was not created, or it cannot be removed either
        raise
    return path


@dataclass
class _Part:
    """What the footer of one part says."""

    path: str
    rows: int
    target_ppi: float
    build: dict
    unknown_columns: tuple


def _one_line(error: Exception) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())}"


def _named(names: list, total: Optional[int] = None) -> str:
    """
    The first names of a list and how many more there are: more in the list,
    or more of `total` if the list holds the first names only.
    """
    total = len(names) if total is None else total
    text = ", ".join(str(name) for name in names[:_MAX_NAMED])
    if total > _MAX_NAMED:
        text += f" and {total - _MAX_NAMED:,} more"
    return text


def _is_minutiae_type(pa, found) -> bool:
    """
    Whether a type is list<struct<x: float32, y: float32, angle: float32>>.
    The name of the item field of the list is not compared (Arrow calls it
    "item", the Parquet format "element"), and neither is which fields are
    declared as never null: a null that is really there is refused when the
    rows are read.
    """
    if not pa.types.is_list(found) or not pa.types.is_struct(found.value_type):
        return False
    fields = [found.value_type.field(i) for i in range(found.value_type.num_fields)]
    return [f.name for f in fields] == ["x", "y", "angle"] and all(
        f.type == pa.float32() for f in fields
    )


def _read_footer(pa, pq, path: str) -> _Part:
    """Reads the metadata and the schema of a part and applies the rules for a single part."""
    rebuild = f"Build the table again ({_BUILDER})."
    try:
        # pyarrow gets the open file and not its path: see _open_file
        with _open_file(pa, path, "rb") as source, pq.ParquetFile(source) as file:
            metadata = dict(file.metadata.metadata or {})
            rows = file.metadata.num_rows
            schema = file.schema_arrow
    # UnicodeDecodeError: what pyarrow raises for a column name that is not
    # UTF-8, as a damaged footer can hold one. It is no error class of Arrow.
    except (pa.ArrowException, OSError, UnicodeDecodeError) as error:
        raise SampleTableError(
            f"{path} cannot be read as a Parquet file ({_one_line(error)}). A part gets its "
            f"name only when it is complete, so the file was damaged or replaced later. {rebuild}"
        ) from error

    def text(key: str) -> Optional[str]:
        value = metadata.get(key.encode())
        try:
            return None if value is None else value.decode("utf-8")
        except UnicodeDecodeError:
            return repr(value)

    table, version = text(_KEY_TABLE), text(_KEY_VERSION)
    if table is None or version is None:
        missing = _KEY_TABLE if table is None else _KEY_VERSION
        raise SampleTableError(
            f"{path} is a Parquet file, but not a part of a sample table: its file metadata "
            f"lack the key {missing}. A sample table is written by {_BUILDER}; a Parquet file "
            "from another tool is not one, whatever its columns are."
        )
    if table != TABLE_NAME:
        raise SampleTableError(
            f"{path} holds a table of the kind {table!r} ({_KEY_TABLE}), not {TABLE_NAME!r}: "
            f"it is not a part of a sample table. Give the directory that {_BUILDER} wrote."
        )
    if version != str(SCHEMA_VERSION):
        raise SampleTableError(
            f"{path} has the schema version {version!r}; this code reads version "
            f"{SCHEMA_VERSION} only. Read the table with the version of this repository that "
            "wrote it, or build it again with this one."
        )

    try:
        target_ppi = float(text(_KEY_TARGET_PPI))
        build = json.loads(text(_KEY_BUILD))
        if not isinstance(build, dict) or not isinstance(build.get("id"), str):
            raise ValueError('no JSON object with a text under "id"')
        for key in ("part", "parts", "rows"):
            if type(build.get(key)) is not int:
                raise ValueError(f'"{key}" is not a whole number')
        if not 0 <= build["part"] < build["parts"]:
            raise ValueError(f'part {build["part"]} of {build["parts"]} parts')
    except (TypeError, ValueError) as error:
        raise SampleTableError(
            f"{path} has file metadata that this module did not write: {_KEY_TARGET_PPI} must "
            f"be a number and {_KEY_BUILD} a JSON object with the build of the part "
            f"({_one_line(error)}). {rebuild}"
        ) from None
    if build["rows"] != rows:
        raise SampleTableError(
            f"{path} holds {rows:,} rows, but its metadata state {build['rows']:,}: the file "
            f"is not as it was written. {rebuild}"
        )

    names = schema.names
    missing = [name for name in COLUMNS if name not in names]
    if missing:
        raise SampleTableError(
            f"{path} lacks the column(s) {_named(missing)}; it has {_named(names)}. A sample "
            f"table of schema version {SCHEMA_VERSION} has the columns {', '.join(COLUMNS)}. "
            f"{rebuild}"
        )
    expected_schema = _arrow_schema(pa, target_ppi, "")
    for name in COLUMNS:
        if names.count(name) > 1:
            raise SampleTableError(
                f"{path} has the column {name} {names.count(name)} times. {rebuild}"
            )
        found, expected = schema.field(name).type, expected_schema.field(name).type
        if name == "minutiae":
            same = _is_minutiae_type(pa, found)
            expected = "list<struct<x: float, y: float, angle: float>>"
        else:
            same = found == expected
        if not same:
            raise SampleTableError(
                f"{path}: the column {name} has the type {found}, where a sample table has "
                f"{expected} (in the names of Arrow: float has 32 bits and double 64). Values "
                "are not converted when a table is read, because a converted value can be "
                "another one: a ppi that went through 32 bits gives another size of the "
                "prepared image for some files. Write the column with the type of the schema, "
                f"or build the table again ({_BUILDER})."
            )
    return _Part(
        path=path,
        rows=rows,
        target_ppi=target_ppi,
        build=build,
        unknown_columns=tuple(dict.fromkeys(name for name in names if name not in COLUMNS)),
    )


def _check_parts(directory: str, parts: list) -> None:
    """Applies the rules for the parts of a table together: one whole build, one target."""
    rebuild = f"Build the table again, into a new directory ({_BUILDER})."
    names = [os.path.basename(part.path) for part in parts]

    builds = {}
    for part, name in zip(parts, names):
        builds.setdefault(part.build["id"], []).append(name)
    if len(builds) > 1:
        listed = "; ".join(f"build {key!r}: {_named(value)}" for key, value in builds.items())
        raise SampleTableError(
            f"{directory} holds parts of {len(builds)} builds ({listed}). A table is the parts "
            f"of one build; parts of two builds cannot be read as one table. {rebuild}"
        )
    counts = sorted({part.build["parts"] for part in parts})
    if len(counts) > 1:
        raise SampleTableError(
            f"The parts in {directory} state different numbers of parts of their table "
            f"({_named(counts)}), so they are not the parts of one table. {rebuild}"
        )
    count = counts[0]
    by_index = {}
    for part, name in zip(parts, names):
        by_index.setdefault(part.build["part"], []).append(name)
    doubled = {index: found for index, found in by_index.items() if len(found) > 1}
    if doubled:
        listed = "; ".join(f"part {index}: {_named(found)}" for index, found in doubled.items())
        raise SampleTableError(
            f"{directory} holds a part of the table more than once ({listed}); a copy of a "
            f"part under another name is not another part. {rebuild}"
        )
    if len(parts) < count:
        # Only the first names are made: the number of parts is what a file states
        missing = (part_name(index, count) for index in range(count) if index not in by_index)
        first_missing = list(itertools.islice(missing, _MAX_NAMED))
        raise SampleTableError(
            f"{directory} is not a whole table: it holds {len(parts):,} of its {count:,} parts; "
            f"missing: {_named(first_missing, count - len(parts))}. A table is only read if "
            "all parts of its build are there. Copy the missing parts from where the table "
            f"was built, or build it again ({_BUILDER})."
        )
    for part, name in zip(parts, names):
        expected = part_name(part.build["part"], count)
        if name != expected:
            raise SampleTableError(
                f"{part.path} is part {part.build['part']} of {count} by its metadata, so its "
                f"name would be {expected}: the file was renamed or copied. {rebuild}"
            )

    targets = sorted({part.target_ppi for part in parts})
    if len(targets) > 1:
        raise SampleTableError(
            f"The parts in {directory} were written for different target resolutions "
            f"({_named(targets)} ppi). {rebuild}"
        )
    if targets[0] != TARGET_PPI:
        raise SampleTableError(
            f"The table in {directory} was built for images prepared at {targets[0]!r} ppi; "
            f"this code prepares images at {TARGET_PPI!r} ppi, and width, height and minutiae "
            f"of the rows would not fit them. {rebuild}"
        )
    for part in parts:
        if part.build.get("settings") != parts[0].build.get("settings"):
            raise SampleTableError(
                f"{part.path} was built with other settings than {parts[0].path} (the "
                f'"settings" in {_KEY_BUILD} differ), so the files of the two parts were not '
                f"prepared in the same way. {rebuild}"
            )


def _merged_build(parts: list) -> dict:
    """The build information of a table from that of its parts (in the order of their index)."""
    merged = {key: value for key, value in parts[0].build.items() if key != "part"}
    for key in ("rows", "files", "failed"):
        values = [(part, part.build[key]) for part in parts if key in part.build]
        if not values:
            continue
        total = values[0][1]
        for part, value in values[1:]:
            numbers = all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in (total, value)
            )
            if not numbers and not (isinstance(total, list) and isinstance(value, list)):
                raise SampleTableError(
                    f'{part.path}: "{key}" in {_KEY_BUILD} cannot be added to that of the parts '
                    f"before it ({value!r} to {total!r}); it must be a number in all parts or a "
                    f"list in all parts. Build the table again ({_BUILDER})."
                )
            total = total + value
        merged[key] = total
    return merged


class _Columns:
    """
    The columns of a table while its parts are read: an array per column for
    all rows, filled batch after batch in the order of the files, and the
    bytes of the image paths and of the minutiae of all rows read so far.

    The number of rows is known from the footers of the parts before a row is
    read; the number of path bytes and of minutiae is not. These two are
    therefore collected in a bytearray each, which grows at its end. A numpy
    array cannot grow, and joining pieces at the end needs the memory for the
    pieces and for the joined array at the same time.
    """

    def __init__(self, rows: int):
        self.rows = rows
        self.filled = 0
        self.arrays = {
            name: np.empty(rows, dtype=dtype)
            for name, dtype in _ARRAY_TYPES.items()
            if name not in _NOT_PER_ROW
        }
        self.path_lengths = np.empty(rows, dtype=np.int64)
        self.minutiae_counts = np.empty(rows, dtype=np.int64)
        self.path_bytes = bytearray()
        self.minutiae_bytes = bytearray()  # float32: x, y, angle of one minutia after the other


def _string_bytes(array) -> tuple[np.ndarray, np.ndarray]:
    """
    (length in bytes of every value, the bytes of all values) of an Arrow
    string array. The bytes are a view into the memory of the array.
    """
    count = len(array)
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint8)
    _, offsets, data = array.buffers()
    # The array can be a slice: its values begin at array.offset in the buffers
    offsets = np.frombuffer(offsets, dtype=np.int32, count=count + 1, offset=array.offset * 4)
    start, end = int(offsets[0]), int(offsets[-1])
    if end == start:
        text = np.zeros(0, dtype=np.uint8)
    else:
        text = np.frombuffer(data, dtype=np.uint8, count=end - start, offset=start)
    return np.diff(offsets).astype(np.int64), text


def _text_indices(array, values: tuple) -> tuple[np.ndarray, list]:
    """
    (index into `values` of every text of an Arrow string array, and
    len(values) where the text is none of them; the first such texts, as a
    message can show them).
    """
    encoded = array.dictionary_encode()
    # Only as many distinct texts become Python values as a message can name.
    # They are taken as bytes: a Parquet reader does not check that a text is
    # UTF-8, and a text that is not cannot become a Python text.
    first = encoded.dictionary.slice(0, len(values) + _MAX_NAMED).cast("binary").to_pylist()
    known = [value.encode("utf-8") for value in values]
    lookup = np.full(len(encoded.dictionary), len(values), dtype=np.uint8)
    for position, data in enumerate(first):
        if data in known:
            lookup[position] = known.index(data)
    indices = lookup[encoded.indices.to_numpy(zero_copy_only=True)]
    # For the message, bytes that are not UTF-8 are shown escaped
    return indices, [
        data.decode("utf-8", errors="backslashreplace") for data in first if data not in known
    ]


def _add_batch(columns: _Columns, batch, path: str, first_row: int) -> None:
    """
    Copies the rows of an Arrow record batch into the columns. The batch
    holds the rows of the part `path` from its row `first_row` on.
    """
    count = batch.num_rows
    rows = slice(columns.filled, columns.filled + count)

    def column(name: str):
        return batch.column(batch.schema.get_field_index(name))

    def where(row: int) -> str:
        """A row of the batch for a message: its number in the part, and its image path."""
        text = f"row {first_row + row:,} of the part (counted from 0)"
        try:
            image_path = column("image_path")[row].as_py()
        except UnicodeDecodeError:
            image_path = None
        return text if image_path is None else f"{text}, image_path {image_path!r}"

    def without_null(name: str):
        array = column(name)
        if array.null_count:
            first = int(np.flatnonzero(array.is_null().to_numpy(zero_copy_only=False))[0])
            raise SampleTableError(
                f"{path}: the column {name} holds {array.null_count:,} null value(s), the "
                f"first in {where(first)}. Only minutiae can be null. Build the table again "
                f"({_BUILDER})."
            )
        return array

    def indices(name: str, values: tuple) -> np.ndarray:
        found, unknown = _text_indices(without_null(name), values)
        if unknown:
            first = int(np.argmax(found == len(values)))
            raise SampleTableError(
                f"{path}: the column {name} holds {_named([repr(text) for text in unknown])}, "
                f"first in {where(first)}. It can hold "
                f"{', '.join(repr(value) for value in values)} and nothing else. Build the "
                f"table again ({_BUILDER})."
            )
        return found

    for name in (
        "subject",
        "impression",
        "file_size",
        "ppi",
        "source_width",
        "source_height",
        "width",
        "height",
    ):
        columns.arrays[name][rows] = without_null(name).to_numpy(zero_copy_only=True)
    columns.arrays["ppi_origin"][rows] = indices("ppi_origin", PPI_ORIGINS)
    columns.arrays["ridges_origin"][rows] = indices("ridges_origin", RIDGES_ORIGINS)
    columns.arrays["ridges_light"][rows] = indices("ridges", (RIDGES_DARK, RIDGES_LIGHT)) == 1
    lengths, text = _string_bytes(without_null("image_path"))
    columns.path_lengths[rows] = lengths
    columns.path_bytes += memoryview(text)

    minutiae = column("minutiae")
    known = ~minutiae.is_null().to_numpy(zero_copy_only=False)
    counts = np.diff(minutiae.offsets.to_numpy(zero_copy_only=True)).astype(np.int64)
    # The Arrow format allows values behind a null list; flatten leaves them out
    counts[~known] = 0
    items = minutiae.flatten()
    fields = [items.field(position) for position in range(3)]
    if items.null_count or any(values.null_count for values in fields):
        raise SampleTableError(
            f"{path}: the column minutiae holds a null minutia or a null x, y or angle in the "
            f"rows {first_row:,} to {first_row + count - 1:,} of the part. Only the whole "
            f"list of an image can be null. Build the table again ({_BUILDER})."
        )
    if len(items):
        # x, y and angle are three arrays in Arrow and one array (k, 3) here
        values = np.empty((len(items), 3), dtype=np.float32)
        for position in range(3):
            values[:, position] = fields[position].to_numpy(zero_copy_only=True)
        columns.minutiae_bytes += memoryview(values).cast("B")
    columns.arrays["minutiae_known"][rows] = known
    columns.minutiae_counts[rows] = counts
    columns.filled += count


def _read_rows(pa, pq, part: _Part, columns: _Columns) -> None:
    """
    Copies the rows of a part into the columns, a batch of rows at a time.
    When the function returns, nothing refers to the Arrow memory of the part
    any more.
    """
    # pyarrow hands out another number of rows than the footer stated where the
    # file was written again after its footer was read, and where the file is
    # damaged in a way for which pyarrow raises no error: the footer still
    # states the rows of the part, and none of them are handed out, or the
    # first ones only. Reading the table again does not help in the second
    # case, so the refusal names both and says what to do in each.
    other_rows = SampleTableError(
        f"{part.path} does not hold the {part.rows:,} rows any more that its metadata stated "
        "when the table was opened: the file changed while the table was read, or it was "
        "damaged. Read the table again when nothing writes to its directory; if it is refused "
        "again, copy the part again from where the table was built, or build the table again "
        f"({_BUILDER})."
    )
    first_row = 0
    try:
        # Without threads and without reading ahead: the batches are wanted one
        # after the other, and each is copied before the next is asked for.
        # The checksums of the pages are compared: a page that does not have
        # the bytes it was written with makes pyarrow raise an error. pyarrow
        # gets the open file and not its path: see _open_file.
        with (
            _open_file(pa, part.path, "rb") as source,
            pq.ParquetFile(source, pre_buffer=False, page_checksum_verification=True) as file,
        ):
            batches = file.iter_batches(
                batch_size=_BATCH_ROWS, columns=list(COLUMNS), use_threads=False
            )
            for batch in batches:
                # The arrays have room for the rows that the footer stated, no more
                if first_row + batch.num_rows > part.rows:
                    raise other_rows
                _add_batch(columns, batch, part.path, first_row)
                first_row += batch.num_rows
    # The footer of the part was read before and fits a table. What pyarrow
    # raises here is about the rows, which cannot be decoded, or about a file
    # that is no longer the one whose footer was read.
    except (pa.ArrowException, OSError) as error:
        raise SampleTableError(
            f"{part.path} cannot be read to its end ({_one_line(error)}); {first_row:,} of its "
            f"{part.rows:,} rows were read before that. A part gets its name only when it is "
            "complete, so the file was damaged later, or it changed while the table was read. "
            "Copy the part again from where the table was built, or build the table again "
            f"({_BUILDER})."
        ) from error
    if first_row != part.rows:
        raise other_rows


def _in_order(
    items: np.ndarray, counts: np.ndarray, order: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    The items of all rows (bytes of the image paths, or minutiae) with the
    rows in the wanted order, as (items, offsets): row i holds
    items[offsets[i]:offsets[i + 1]].

    items: The items of the rows one row after the other, in the order in
        which the rows were read
    counts: Items of every row, in that order
    order: None if the rows stay in that order: the items are returned as
        they are. Else the row that was read at position order[i] becomes
        row i, and the items are copied into a new array. Both arrays exist
        until the caller lets go of the old one.
    """
    rows = len(counts)
    offsets = np.zeros(rows + 1, dtype=np.int64)
    if order is None:
        np.cumsum(counts, out=offsets[1:])
        return items, offsets
    np.cumsum(counts[order], out=offsets[1:])
    starts = np.cumsum(counts) - counts  # where the items of every row are now
    moved = np.empty_like(items)
    for first, end in _blocks(offsets, _BLOCK_ITEMS):
        start, stop = int(offsets[first]), int(offsets[end])
        block = order[first:end]
        # An item is taken from where its row is now plus its position in
        # the row: its new position moved by the same distance for all items
        # of a row
        index = np.repeat(starts[block] - offsets[first:end], counts[block])
        index += np.arange(start, stop, dtype=np.int64)
        moved[start:stop] = items[index]
    return moved, offsets


def read_samples(directory: str) -> Samples:
    """
    Reads the table in a directory and returns its rows as Samples, in the
    order of (subject, impression). The table is checked completely: the
    rules for the files (see the module docstring) and validate_samples. A
    table that breaks one raises SampleTableError with the part or the rows
    and what to do.

    The parts are read one after the other, a batch of rows at a time, and
    each batch is copied into the numpy arrays before the next is read.
    Nothing refers to the Arrow memory of a part when the next part is read.

    Samples.build is the build information of the parts as one dictionary:
    "id", "parts" and "settings" as in every part, "rows" (and "files" and
    "failed" where the builder recorded them) added up over the parts,
    other keys from part 0, and no "part".
    """
    pa, pq = _import_pyarrow()
    directory = os.fspath(directory)
    try:
        names = sorted(os.listdir(directory))
    except OSError as error:
        raise SampleTableError(
            f"{directory} cannot be read as the directory of a sample table "
            f"({_one_line(error)}). A sample table is a directory of files named like "
            f"{part_name(0, 3)}, written by {_BUILDER}; give the path of that directory."
        ) from None
    except UnicodeEncodeError as error:
        raise _no_file_name(directory, "read", error) from None
    part_names = [name for name in names if _PART_NAME.fullmatch(name)]
    if not part_names:
        raise SampleTableError(
            f"{directory} holds no part of a sample table (a file named like "
            f"{part_name(0, 3)}); it holds {_named(names) if names else 'no files'}. Give the "
            f"directory that {_BUILDER} wrote, or build the table."
        )
    parts = [_read_footer(pa, pq, os.path.join(directory, name)) for name in part_names]
    _check_parts(directory, parts)
    parts.sort(key=lambda part: part.build["part"])

    columns = _Columns(sum(part.rows for part in parts))
    for part in parts:
        _read_rows(pa, pq, part, columns)
    # Arrow keeps the memory of the batches to use it again. Nothing uses it
    # again once the table is read, so it is given back.
    pa.default_memory_pool().release_unused()

    arrays = columns.arrays
    order = _sorting_order(arrays["subject"], arrays["impression"])
    if order is not None:
        for name in arrays:
            arrays[name] = arrays[name][order]
    path_data, path_offsets = _in_order(
        np.frombuffer(columns.path_bytes, dtype=np.uint8), columns.path_lengths, order
    )
    minutiae, minutiae_offsets = _in_order(
        np.frombuffer(columns.minutiae_bytes, dtype=np.float32).reshape(-1, 3),
        columns.minutiae_counts,
        order,
    )
    del columns  # with it the bytes as they were read, if the rows were put into another order
    samples = Samples(
        path_data=path_data,
        path_offsets=path_offsets,
        minutiae_offsets=minutiae_offsets,
        minutiae=minutiae,
        target_ppi=parts[0].target_ppi,
        build=_merged_build(parts),
        unknown_columns=tuple(
            dict.fromkeys(name for part in parts for name in part.unknown_columns)
        ),
        **arrays,
    )
    try:
        validate_samples(samples)
    except SampleTableError as error:
        raise SampleTableError(f"{directory}: {error}") from None
    return samples
