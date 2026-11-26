"""
The loaders of a sample table: they serve the images and the minutiae of the
rows of a table (flx/preparation/sample_table.py) to the training.

Why: without a table the training finds its samples by the names of the files
in two directories. CustomFingerprintLoader of scripts/train_500k.py takes
subject and impression from the name of an image file and lets a
SensorNormalizer find out its resolution and ridge polarity;
FingerFlowMinutiaMapLoader reads the minutiae from the .mnt file of the same
name. A table states all of this once for every image file. The loaders of
this module do with the rows what those loaders do with the files:

    SampleTableImageLoader.get returns the tensor that CustomFingerprintLoader
    returns for the same file when its SensorNormalizer resolves the
    resolution and the polarity that the row holds.

    SampleTableMinutiaMapLoader.get_minutiae returns what the loader of .mnt
    files returns for the .mnt file whose minutiae the row holds.

So the training gets the same tensors from a table as from the directories
the table was built from, for the files that both can serve: image files
named <subject>_<impression>.png and .mnt files whose header is the size of
the prepared image. Both loaders fit where the loaders of the directories
are used: the image loader as the `files` of create_image_dataset and
create_validation_image_loader of scripts/train_500k.py, the minutia loader
inside a PoseTransformedMinutiaMapLoader.

A SampleTable holds what both loaders and the scripts need of one table: the
rows as the flat arrays of Samples, the root directory of the images, and one
Identifier object per row. The loaders of the directories each keep a
dictionary from Identifier to file; here a row is found by a search in the
sorted numbers of the rows (Samples.find), and the identifiers of the table,
of the training set and of the validation set can be the same objects.

This module needs torch, like the other loaders of flx/data. pyarrow is only
needed where a table is read from its files (SampleTable.read), inside
flx/preparation/sample_table.py; the module is imported without it. How a
table is built and used is described in docs/sample-table.md.
"""

import os
from typing import Iterable, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as VTF

from flx.data.dataset import DataLoader, Identifier, IdentifierSet
from flx.data.image_helpers import transform_to_input_size
from flx.data.minutia_map_loader import MinutiaMapLoader
from flx.preparation.sample_table import (
    Samples,
    SampleTableError,
    read_samples,
    samples_checksum,
    validate_samples,
)
from flx.preparation.sensor_normalization import (
    RIDGES_DARK,
    RIDGES_LIGHT,
    ImageReadError,
    SensorNormalizer,
)

# The script that writes tables, for the messages that say what to do
_BUILDER = "scripts/build_sample_table.py"

# How many SensorNormalizer objects an image loader keeps, one per pair of
# resolution and ridge polarity. The files of one sensor share a pair. The
# limit is there for a table whose files state many different resolutions.
_MAX_NORMALIZERS = 64

# How many subjects a message names before it gives the number of the rest
_MAX_NAMED = 5


class UnreadableImageError(SampleTableError, ImageReadError):
    """
    An image file of a table for which OpenCV raised an error of its own. It
    is an ImageReadError, the error of the preparation for a file that cannot
    be read as an image, and a SampleTableError like the other errors of
    SampleTableImageLoader.get whose message names the file, its row and what
    to do.
    """


class SampleTable:
    """
    The rows of a sample table together with the root directory of its
    images.

    The rows are the Samples object as it is (`samples`): flat read-only
    arrays in the order of (subject, impression). Nothing is added per row
    except one list of Identifier objects (`identifiers`, identifiers[row])
    and the IdentifierSet of exactly these objects (`ids`). Both are made
    when they are asked for the first time. Build the training set and the
    validation set from these objects (identifiers_of), so that a process
    holds one object per row and not one per row and set.

    A row is found by the two numbers of an identifier (row), not by the
    object: any Identifier with the subject and the impression of a row
    finds it, also one that another process made.

    A copy (pickle, copy.copy, copy.deepcopy) is made of the rows and the
    directory only. A worker process that is not forked gets such a copy.
    The identifiers are not part of it: a loader looks rows up by the
    numbers of the identifiers it is given and does not need those of the
    table. A copy that is asked for them makes them again, as it makes the
    subjects and the checksum again. The arrays of a copy are read-only
    like those of the original.

    The class has no length on purpose (see `num_rows`): a table without
    rows would be false where a script tests whether a table was given.
    """

    def __init__(self, samples: Samples, image_dir: str, validate: bool = True):
        """
        Args:
            samples: The rows of the table
            image_dir: Root directory of the images. The image_path of a row
                is relative to it.
            validate: Check the rows with validate_samples, which raises
                SampleTableError for rows that break a rule of a table. Only
                rows that were checked before need no check: read_samples
                has checked the rows it returns.
        """
        if not isinstance(samples, Samples):
            raise TypeError(
                f"A SampleTable is made of the Samples of a table, got {type(samples).__name__}"
            )
        if validate:
            validate_samples(samples)
        self._samples = samples
        # Absolute, so that a copy in a process with another working
        # directory names the same files
        self._image_dir = os.path.normpath(os.path.abspath(os.fspath(image_dir)))
        self._forget_derived()

    @classmethod
    def read(cls, table_dir: str, image_dir: str) -> "SampleTable":
        """
        Reads the table in the directory table_dir (read_samples, which
        needs pyarrow and checks the table completely) for the images below
        image_dir. Raises SampleTableError with the part or the rows and
        what to do if the table cannot be used.
        """
        return cls(read_samples(table_dir), image_dir, validate=False)

    def _forget_derived(self) -> None:
        # What follows from the rows and is made when it is first asked for
        self._identifiers = None
        self._ids = None
        self._subjects = None
        self._subject_offsets = None
        self._impressions_per_subject = None
        self._checksum = None

    def __getstate__(self) -> dict:
        # What a copy is made of; see the docstring of the class
        return {"samples": self._samples, "image_dir": self._image_dir}

    def __setstate__(self, state: dict) -> None:
        self._samples = state["samples"]
        self._image_dir = state["image_dir"]
        self._forget_derived()

    def __repr__(self) -> str:
        return f"SampleTable({self.num_rows:,} rows, images in {self._image_dir})"

    @property
    def samples(self) -> Samples:
        """The rows: read-only arrays in the order of (subject, impression)."""
        return self._samples

    @property
    def image_dir(self) -> str:
        """The root directory of the images as an absolute path."""
        return self._image_dir

    @property
    def num_rows(self) -> int:
        return len(self._samples)

    @property
    def target_ppi(self) -> float:
        """Resolution of the prepared images: width, height and minutiae of the rows refer to it."""
        return self._samples.target_ppi

    @property
    def build(self) -> dict:
        """What the builder of the table recorded about its run (Samples.build)."""
        return self._samples.build

    @property
    def unknown_columns(self) -> tuple:
        """Names of the columns of the files that this code does not know and did not read."""
        return self._samples.unknown_columns

    @property
    def checksum(self) -> str:
        """
        samples_checksum of the rows: the SHA-256 of their values, by which
        two tables with the same rows are known to be the same. It is
        computed when it is asked for the first time.
        """
        if self._checksum is None:
            self._checksum = samples_checksum(self._samples)
        return self._checksum

    # --- Subjects ----------------------------------------------------------

    def _group_by_subject(self) -> None:
        subject = self._samples.subject
        # The rows are in the order of (subject, impression), so the rows of
        # a subject follow each other: a subject begins where the number changes
        first_rows = np.flatnonzero(subject[1:] != subject[:-1]) + 1
        if len(subject):
            first_rows = np.concatenate(([0], first_rows))
        offsets = np.append(first_rows, len(subject)).astype(np.int64)
        subjects = subject[first_rows]
        impressions = np.diff(offsets)
        for array in (subjects, offsets, impressions):
            array.flags.writeable = False
        self._subjects, self._subject_offsets = subjects, offsets
        self._impressions_per_subject = impressions

    @property
    def subjects(self) -> np.ndarray:
        """The subjects of the table: every number once, in ascending order (int32, read-only)."""
        if self._subjects is None:
            self._group_by_subject()
        return self._subjects

    @property
    def subject_offsets(self) -> np.ndarray:
        """
        Where the rows of every subject are: subjects[k] has the rows
        subject_offsets[k] to subject_offsets[k + 1] - 1, in the order of its
        impressions. One value more than there are subjects (int64, read-only).
        """
        if self._subject_offsets is None:
            self._group_by_subject()
        return self._subject_offsets

    @property
    def impressions_per_subject(self) -> np.ndarray:
        """The number of rows of every subject, in the order of `subjects` (int64, read-only)."""
        if self._impressions_per_subject is None:
            self._group_by_subject()
        return self._impressions_per_subject

    def _positions(self, subjects: Union[Iterable[int], np.ndarray]) -> np.ndarray:
        """Where the given subjects are in self.subjects. KeyError for those the table lacks."""
        wanted = np.asarray(subjects if isinstance(subjects, np.ndarray) else list(subjects))
        if wanted.size == 0:
            return np.zeros(0, dtype=np.int64)
        if wanted.ndim != 1 or wanted.dtype.kind not in "iu":
            raise TypeError(
                "The subjects must be whole numbers, as a list or an array of one dimension, "
                f"got an array of {wanted.dtype} with the shape {wanted.shape}"
            )
        known = self.subjects
        if len(known) == 0:
            found = np.zeros(len(wanted), dtype=np.bool_)
            positions = np.zeros(len(wanted), dtype=np.int64)
        else:
            # The numbers are searched for in the type of the subjects. For
            # numbers of a wider type, as a list of Python numbers gives
            # them, numpy copies all subjects of the table into that type at
            # every call, so that a call per subject would cost a pass over
            # the table each. A number that 32 bits cannot hold becomes
            # another number by this, and is told by that: it is not in the
            # table.
            searched = wanted.astype(known.dtype, copy=False)
            # A number above all subjects of the table is searched for
            # behind the last one: it is compared with the last one instead
            positions = np.minimum(np.searchsorted(known, searched), len(known) - 1)
            found = (known[positions] == searched) & (searched == wanted)
        if not found.all():
            missing = wanted[~found]
            named = ", ".join(str(number) for number in missing[:_MAX_NAMED].tolist())
            if len(missing) > _MAX_NAMED:
                named += f" and {len(missing) - _MAX_NAMED:,} more"
            raise KeyError(
                f"The sample table has no row of the subject(s) {named}; it holds "
                f"{len(known):,} subjects"
            )
        return positions.astype(np.int64, copy=False)

    def rows_of(self, subjects: Union[Iterable[int], np.ndarray]) -> np.ndarray:
        """
        The rows of the given subjects as one int64 array: all rows of the
        first subject in the order of its impressions, then those of the
        second, and so on (a subject that is given twice gives its rows
        twice). The rows are taken from subject_offsets: no subject needs a
        look at the rows of the others. Raises KeyError with the subjects
        that the table does not have.
        """
        positions = self._positions(subjects)
        offsets = self.subject_offsets
        first_rows = offsets[positions]
        counts = offsets[positions + 1] - first_rows
        ends = np.cumsum(counts)
        rows = np.arange(int(ends[-1]) if len(ends) else 0, dtype=np.int64)
        # Item k of the result belongs to a subject whose rows begin at
        # first_rows and whose items begin at ends - counts
        rows += np.repeat(first_rows - (ends - counts), counts)
        return rows

    def rows_without_minutiae(self) -> np.ndarray:
        """
        The rows whose minutiae are not known (a null cell of the table), as
        an int64 array in ascending order. A row whose image has no minutiae
        (an empty list) is not among them. Both kinds give a minutia map of
        zeros with the weight 0.
        """
        return np.flatnonzero(~self._samples.minutiae_known)

    # --- Identifiers -------------------------------------------------------

    @property
    def identifiers(self) -> list[Identifier]:
        """
        The Identifier of every row: identifiers[row]. It is the one list
        of these objects, not a copy; do not change it.
        """
        if self._identifiers is None:
            impressions = self._samples.impression.tolist()
            offsets = self.subject_offsets.tolist()
            identifiers = []
            # Subject by subject, so that the identifiers of a subject share
            # the int object of its number instead of holding one each, as
            # those of create_training_and_validation_ids of
            # scripts/train_500k.py do
            for subject, first, end in zip(self.subjects.tolist(), offsets, offsets[1:]):
                identifiers.extend(
                    [Identifier(subject, impression) for impression in impressions[first:end]]
                )
            self._identifiers = identifiers
        return self._identifiers

    @property
    def ids(self) -> IdentifierSet:
        """
        The IdentifierSet of all rows, made of the objects of `identifiers`.
        It is what the loaders of the table return as their `ids`.
        """
        if self._ids is None:
            try:
                self._ids = IdentifierSet(self.identifiers)
            except RuntimeError as error:
                # IdentifierSet refuses identifiers that are equal. The rows
                # of a table have different numbers, but an Identifier is
                # compared by the hash of its two numbers, and two pairs of
                # numbers can have the same hash.
                raise SampleTableError(self._equal_identifiers()) from error
        return self._ids

    def _equal_identifiers(self) -> str:
        """The refusal for a table with rows whose identifiers compare as equal."""
        first_with, pairs = {}, []
        for row, identifier in enumerate(self.identifiers):
            first = first_with.setdefault(hash(identifier), row)
            if first != row:
                pairs.append(f"{self.describe_row(first)} and {self.describe_row(row)}")
        named = "; ".join(pairs[:_MAX_NAMED])
        if len(pairs) > _MAX_NAMED:
            named += f"; and {len(pairs) - _MAX_NAMED:,} more"
        return (
            f"{len(pairs):,} pair(s) of rows of the sample table cannot be told apart by "
            f"their numbers: {named}. The numbers of the two rows differ, but an image is "
            "found by the hash of its subject and impression, and these pairs of numbers have "
            "the same hash. That takes very large numbers. Give one row of each pair another "
            f"subject or impression and build the table again ({_BUILDER})."
        )

    def identifiers_of(self, subjects: Union[Iterable[int], np.ndarray]) -> list[Identifier]:
        """
        The identifiers of all rows of the given subjects, in the order of
        rows_of: the objects of `identifiers`, not new ones. Raises KeyError
        with the subjects that the table does not have.
        """
        positions = self._positions(subjects)
        identifiers = self.identifiers
        offsets = self.subject_offsets
        result = []
        for first, end in zip(offsets[positions].tolist(), offsets[positions + 1].tolist()):
            result.extend(identifiers[first:end])
        return result

    # --- Rows --------------------------------------------------------------

    def row(self, identifier: Identifier) -> int:
        """
        The row of an identifier. Raises KeyError with the identifier if the
        table has no row of its subject and impression.
        """
        row = self._samples.find(identifier.subject, identifier.impression)
        if row < 0:
            raise KeyError(
                f"{identifier} is not in the sample table: it has no row of subject "
                f"{identifier.subject} with impression {identifier.impression}"
            )
        return row

    def image_file(self, row: int) -> str:
        """The absolute path of the image file of a row."""
        return os.path.join(self._image_dir, *self._samples.image_path(row).split("/"))

    def describe_row(self, row: int) -> str:
        """A row as the messages name it: image path, subject and impression."""
        samples = self._samples
        return (
            f"{samples.image_path(row)!r} (subject {samples.subject[row]}, "
            f"impression {samples.impression[row]})"
        )


def _table_of_a_loader(table, loader) -> "SampleTable":
    """
    The table a loader is given. The Samples of a table are not one: a loader
    over them would be made without an error, and a Dataset over it would
    skip its check of the identifiers, because it takes the error of `ids`
    for a loader without identifiers.
    """
    if not isinstance(table, SampleTable):
        raise TypeError(
            f"A {type(loader).__name__} is made of a SampleTable, got {type(table).__name__}"
        )
    return table


class SampleTableImageLoader(DataLoader):
    """
    Loads the image of a row as CustomFingerprintLoader of
    scripts/train_500k.py loads an image file with its SensorNormalizer:
    resampled to the target resolution of the table and with dark ridges, as
    a float tensor (1, height, width) with values from 0 to 1. Resolution and
    ridge polarity are those of the row. Nothing is read from the metadata of
    the file or estimated from its pixels, so the file is prepared as it was
    when the table was built, whatever the options of the scripts are.

    The minutiae of a row are in the pixels of that prepared image, so the
    file must be the one the table was built from. get compares the size of
    the file in bytes with the row before it loads the file, and the size of
    the file in pixels and that of the prepared image after it. A file that
    differs raises SampleTableError with the file, what differs and what to
    do; a file that is missing raises FileNotFoundError. A file that changed
    and kept all three sizes is not noticed.

    The class has no length on purpose: scripts/train_500k.py takes the
    loader as `files or ...`, and a loader of a table without rows would
    count as not given.
    """

    def __init__(self, table: SampleTable):
        self._table = _table_of_a_loader(table, self)
        # SensorNormalizer by (ppi, light ridges)
        self._normalizers: dict[tuple[float, bool], SensorNormalizer] = {}

    @property
    def table(self) -> SampleTable:
        return self._table

    @property
    def ids(self) -> IdentifierSet:
        """The identifiers of all rows of the table (SampleTable.ids)."""
        return self._table.ids

    def _normalizer(self, ppi: float, light: bool) -> SensorNormalizer:
        """
        The SensorNormalizer for files of this resolution and polarity. Both
        are given to it as the values for all files, which win over the
        metadata of a file and are never estimated.
        """
        normalizer = self._normalizers.get((ppi, light))
        if normalizer is None:
            if len(self._normalizers) >= _MAX_NORMALIZERS:
                self._normalizers.clear()
            normalizer = SensorNormalizer(
                ppi=ppi,
                ridges=RIDGES_LIGHT if light else RIDGES_DARK,
                target_ppi=self._table.target_ppi,
            )
            self._normalizers[(ppi, light)] = normalizer
        return normalizer

    def _differs(self, row: int, path: str, what: str, expected: str, found: str):
        table = self._table
        return SampleTableError(
            f"{path} is not the file that its row of the sample table describes: the table "
            f"holds {expected} as {what}, the file has {found}. The row is "
            f"{table.describe_row(row)} for the image directory {table.image_dir}. The file "
            "changed after the table was built, or this is not the image directory the table "
            "was built from. The minutiae of the row would not fit this file. Give the image "
            "directory of the table, or build the table again from the files as they are now "
            f"({_BUILDER})."
        )

    def get(self, identifier: Identifier) -> torch.Tensor:
        """
        The prepared image of the identifier's row. Raises KeyError if the
        table has no such row, FileNotFoundError if the file is missing,
        SampleTableError if the file is not the one the row describes, and
        the ValueError of the preparation, with the file named, if the file
        cannot be read as an image. Where OpenCV raises an error of its own
        for such a file, that is an UnreadableImageError, which names the
        row as well.
        """
        table = self._table
        samples = table.samples
        row = table.row(identifier)
        path = table.image_file(row)
        try:
            file_size = os.stat(path).st_size
        except (FileNotFoundError, NotADirectoryError):
            raise FileNotFoundError(
                f"The image file {path} is missing. The sample table names it as "
                f"{table.describe_row(row)}, and {table.image_dir} was given as the directory "
                "of its images. Give the image directory the table was built from; if the "
                "file is gone from there, put it back or build the table again from the "
                f"files as they are now ({_BUILDER})."
            ) from None
        if file_size != int(samples.file_size[row]):
            raise self._differs(
                row,
                path,
                "the size of the file",
                f"{int(samples.file_size[row]):,} bytes",
                f"{file_size:,} bytes",
            )

        normalizer = self._normalizer(float(samples.ppi[row]), bool(samples.ridges_light[row]))
        try:
            image, info = normalizer.load(path)
        except ValueError as error:
            # Name the file: the message of the preparation does not
            raise type(error)(f"{path}: {error}") from error
        except cv2.error as error:
            # For most files that are no image OpenCV returns nothing, and the
            # preparation raises its ImageReadError. For some it raises an
            # error of its own, which is no ValueError and names no file: a
            # BMP file whose header states a width that OpenCV refuses is one.
            raise UnreadableImageError(
                f"{path}: Could not read the file as an image: OpenCV stopped with an error "
                f"of its own ({' '.join(str(error).split())}). The sample table names the "
                f"file as {table.describe_row(row)}, and {table.image_dir} was given as the "
                "directory of its images. Put the file back as it was when the table was "
                f"built, or build the table again from the files as they are now ({_BUILDER})."
            ) from error

        source = (int(samples.source_width[row]), int(samples.source_height[row]))
        if (info.source_width, info.source_height) != source:
            raise self._differs(
                row,
                path,
                "its size in pixels (width x height)",
                f"{source[0]}x{source[1]}",
                f"{info.source_width}x{info.source_height}",
            )
        prepared = (int(samples.width[row]), int(samples.height[row]))
        if (info.width, info.height) != prepared:
            raise self._differs(
                row,
                path,
                f"the size in pixels (width x height) of the image prepared at "
                f"{table.target_ppi:g} ppi",
                f"{prepared[0]}x{prepared[1]}",
                f"{info.width}x{info.height}",
            )
        return VTF.to_tensor(image)


class SampleTableMinutiaMapLoader(MinutiaMapLoader):
    """
    Serves the minutiae of the rows as the loaders of
    flx/data/fingerflow_minutia_loader.py serve those of .mnt files:
    get_minutiae returns the locations in the pixels of the model input and
    the angles, and get (of MinutiaMapLoader) makes the minutia map of them.

    A row without minutiae gives a map of zeros with the weight 0, so the
    sample does not count in the minutia map loss. That holds for a row
    whose minutiae are not known (a null cell) and for one whose image has
    none (an empty list), as it holds for an empty .mnt file. `ids` are all
    rows of the table, also these: whether a training accepts rows without
    minutiae is decided by the script, not here.
    """

    def __init__(self, table: SampleTable):
        self._table = _table_of_a_loader(table, self)

    @property
    def table(self) -> SampleTable:
        return self._table

    @property
    def ids(self) -> IdentifierSet:
        """The identifiers of all rows of the table (SampleTable.ids)."""
        return self._table.ids

    def get_frame_size(self, identifier: Identifier) -> tuple[int, int]:
        """
        @returns (width, height) of the prepared image of the sample, in
            whose pixels its minutiae are: what the header of a .mnt file
            states. PoseTransformedMinutiaMapLoader compares it with the
            image that was loaded.
        """
        samples = self._table.samples
        row = self._table.row(identifier)
        return int(samples.width[row]), int(samples.height[row])

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each
            row is one coordinate (x, y) of a minutia in the pixels of the model input. The
            second array is a 1-D array containing the orientation in radians. Both are new
            arrays, not memory of the table. A row without minutiae gives arrays of the
            shapes (0, 2) and (0,).
        """
        samples = self._table.samples
        row = self._table.row(identifier)
        minutiae = samples.minutiae_of(row)
        if len(minutiae) == 0:
            # As read_mnt_file returns them for a file without minutiae
            return np.zeros((0, 2)), np.zeros(0)
        locations = transform_to_input_size(
            minutiae[:, :2],
            # As Python numbers, like the sizes that are read from the header
            # of a .mnt file. With the numpy numbers of the arrays the scale
            # inside of transform_to_input_size is a numpy number of 64 bits,
            # and NumPy 2 then scales the locations in 64 bits instead of
            # their own 16. Some of them come out as other numbers that way:
            # x = 175 in a frame of 500x640 pixels as 114.4375 instead of 114.5.
            original_height=int(samples.height[row]),
            original_width=int(samples.width[row]),
        )
        return locations, minutiae[:, 2].copy()
