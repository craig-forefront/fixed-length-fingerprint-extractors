"""
Tests for flx/preparation/sample_table.py: the Parquet files of a sample table
and their metadata, what a reader refuses, the rules for the rows and the
checksum. These tests need pyarrow. What the module does without pyarrow is in
tests/test_sample_table_optional.py, which runs in every environment.
"""

import copy
import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from flx.preparation import sample_table
from flx.preparation.sample_table import (
    COLUMNS,
    PPI_ORIGINS,
    RIDGES_ORIGINS,
    SampleRow,
    Samples,
    SampleTableError,
    normalize_image_path,
    part_name,
    read_samples,
    samples_checksum,
    samples_from_rows,
    validate_samples,
    write_part,
)
from flx.preparation.sensor_normalization import TARGET_PPI, resampled_size

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

BUILDER = "scripts/build_sample_table.py"
REPOSITORY = os.path.dirname(os.path.dirname(__file__))

# The arrays of Samples, by the names of its fields
ARRAYS = [field.name for field in dataclasses.fields(Samples) if field.type is np.ndarray]


def sample_row(subject: int, impression: int, **changes) -> SampleRow:
    """
    A valid row. Its values follow from the two numbers, so that the rows of
    a table differ in every column: four resolutions, both polarities, all
    origins, paths that are not ASCII, and null, empty and filled minutiae
    next to each other.
    """
    ppi = (500.0, 1000.0, 251.2, 385.0)[(subject + impression) % 4]
    source_width, source_height = 300 + 7 * subject, 400 + 3 * impression
    width, height = resampled_size(source_width, source_height, ppi)
    kind = (subject // 2 + impression) % 3
    if kind == 0:
        minutiae = None
    else:
        count = 0 if kind == 1 else 1 + (subject + 2 * impression) % 4
        minutiae = np.arange(3 * count, dtype=np.float32).reshape(count, 3) / 8 + subject
        minutiae[:, 2] = 0.25 * impression + np.arange(count)
    values = dict(
        subject=subject,
        impression=impression,
        image_path=f"sensor {subject % 3}/größe_{subject}_{impression}.png",
        file_size=10_000 + 17 * subject + impression,
        ppi=ppi,
        ppi_origin=PPI_ORIGINS[(subject + impression) % 4],
        ridges="light" if subject % 2 else "dark",
        ridges_origin=RIDGES_ORIGINS[impression % 4],
        source_width=source_width,
        source_height=source_height,
        width=width,
        height=height,
        minutiae=minutiae,
    )
    values.update(changes)
    return SampleRow(**values)


# Subjects with gaps, given in no order
ROWS = [sample_row(subject, impression) for subject in (5, 0, 17, 3, 8) for impression in (2, 0, 1)]


def write_table(directory, rows, parts: int = 2, build: dict = None, target_ppi=TARGET_PPI):
    """
    Writes the rows as a table of `parts` parts, row k into part k % parts,
    and returns the paths of the parts.
    """
    build = {"id": "build-1"} if build is None else build
    return [
        write_part(
            str(directory), index, parts, samples_from_rows(rows[index::parts], target_ppi), build
        )
        for index in range(parts)
    ]


def change_part(path, change=None, metadata=None, build=None, **options) -> None:
    """
    Writes a part again with pyarrow alone, as another tool or a damaged
    build would have written it. `change` gets the Arrow table and returns
    the one to write, `metadata` replaces file metadata (None removes a key),
    `build` replaces keys of flx.build, and the options go to pq.write_table.
    """
    table = pq.read_table(path)
    file_metadata = {key.decode(): value.decode() for key, value in table.schema.metadata.items()}
    if change is not None:
        table = change(table)
    if build is not None:
        stored = dict(json.loads(file_metadata["flx.build"]), **build)
        file_metadata["flx.build"] = json.dumps(stored)
    for key, value in (metadata or {}).items():
        if value is None:
            del file_metadata[key]
        else:
            file_metadata[key] = value
    pq.write_table(table.replace_schema_metadata(file_metadata), path, **options)


def with_column(name: str, values):
    """A change for change_part: one column is replaced, or added at the end."""

    def change(table):
        column = values(table) if callable(values) else values
        index = table.schema.get_field_index(name)
        if index < 0:
            return table.append_column(name, column)
        return table.set_column(index, name, column)

    return change


def footer_position(content: bytes) -> int:
    """
    Where the footer begins in the bytes of a Parquet file. The rows are in
    the pages between the first four bytes and the footer; the footer says
    where they are and holds the schema and the file metadata; the last eight
    bytes are the length of the footer and the letters PAR1.
    """
    assert content[:4] == content[-4:] == b"PAR1"
    return len(content) - 8 - int.from_bytes(content[-8:-4], "little")


def refusal(directory) -> str:
    """The message with which read_samples refuses the table in a directory."""
    with pytest.raises(SampleTableError) as error:
        read_samples(str(directory))
    return str(error.value)


def invalid(rows) -> str:
    """The message with which validate_samples refuses the rows."""
    samples = rows if isinstance(rows, Samples) else samples_from_rows(rows)
    with pytest.raises(SampleTableError) as error:
        validate_samples(samples)
    return str(error.value)


def row_name(row: SampleRow) -> str:
    """A row as the messages name it."""
    return f"{row.image_path!r} (subject {row.subject}, impression {row.impression})"


def assert_same_rows(found: Samples, expected: Samples) -> None:
    """Every array has the same type and the same bytes."""
    for name in ARRAYS:
        left, right = getattr(found, name), getattr(expected, name)
        assert left.dtype == right.dtype, name
        assert left.shape == right.shape, name
        assert left.tobytes() == right.tobytes(), name
    assert found.target_ppi == expected.target_ppi


# ---------------------------------------------------------------------------
# A table is written and read
# ---------------------------------------------------------------------------


def test_a_table_of_two_parts_comes_back_as_its_rows_were_given(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    assert [os.path.basename(path) for path in paths] == [
        "part-00000-of-00002.parquet",
        "part-00001-of-00002.parquet",
    ]
    assert sorted(os.listdir(tmp_path / "table")) == [os.path.basename(path) for path in paths]
    # Control: both parts hold null, empty and filled minutiae
    for part in (ROWS[0::2], ROWS[1::2]):
        kinds = {None if row.minutiae is None else len(row.minutiae) > 0 for row in part}
        assert kinds == {None, False, True}

    # Control: a ppi among the rows is not a number of 32 bits
    assert any(float(np.float32(row.ppi)) != row.ppi for row in ROWS)

    samples = read_samples(tmp_path / "table")
    assert len(samples) == len(ROWS)
    assert_same_rows(samples, samples_from_rows(ROWS))
    # In the order of (subject, impression), which is neither the order in
    # which the rows were given nor the order of the files
    in_order = sorted(ROWS, key=lambda row: (row.subject, row.impression))
    assert in_order != ROWS
    for position, row in enumerate(in_order):
        assert (samples.subject[position], samples.impression[position]) == (
            row.subject,
            row.impression,
        )
        assert samples.image_path(position) == row.image_path
        assert samples.find(row.subject, row.impression) == position
        assert samples.file_size[position] == row.file_size
        assert samples.ppi[position] == row.ppi
        assert PPI_ORIGINS[samples.ppi_origin[position]] == row.ppi_origin
        assert RIDGES_ORIGINS[samples.ridges_origin[position]] == row.ridges_origin
        assert samples.ridges_light[position] == (row.ridges == "light")
        assert (samples.source_width[position], samples.source_height[position]) == (
            row.source_width,
            row.source_height,
        )
        assert (samples.width[position], samples.height[position]) == (row.width, row.height)
        minutiae = samples.minutiae_of(position)
        assert minutiae.dtype == np.float32 and minutiae.shape[1:] == (3,)
        assert samples.minutiae_known[position] == (row.minutiae is not None)
        if row.minutiae is None:
            assert len(minutiae) == 0
        else:
            assert minutiae.tobytes() == row.minutiae.tobytes()
    assert samples.unknown_columns == ()
    assert "15 rows" in repr(samples)


def test_a_part_holds_the_values_of_its_rows_as_any_parquet_reader_sees_them(tmp_path):
    """
    Read with pyarrow alone. A writer and a reader of this module that agreed
    on another layout (y under the name x, 'light' for dark ridges) would
    give back every table as it was given, and the files would not hold what
    the descriptions stored in them say.
    """
    # Control: no row looks the same with two of its values exchanged
    assert all(row.width != row.height for row in ROWS)
    assert all(row.source_width != row.source_height for row in ROWS)
    assert {row.ridges for row in ROWS} == {"dark", "light"}
    filled = [row.minutiae for row in ROWS if row.minutiae is not None and len(row.minutiae)]
    assert filled and all(x != y for x, y, _ in np.concatenate(filled))

    (path,) = write_table(tmp_path / "table", ROWS, parts=1)
    stored = {(row["subject"], row["impression"]): row for row in pq.read_table(path).to_pylist()}
    assert len(stored) == len(ROWS)
    for row in ROWS:
        minutiae = None
        if row.minutiae is not None:
            minutiae = [
                {"x": float(x), "y": float(y), "angle": float(angle)}
                for x, y, angle in row.minutiae
            ]
        assert stored[(row.subject, row.impression)] == {
            "subject": row.subject,
            "impression": row.impression,
            "image_path": row.image_path,
            "file_size": row.file_size,
            "ppi": row.ppi,
            "ppi_origin": row.ppi_origin,
            "ridges": row.ridges,
            "ridges_origin": row.ridges_origin,
            "source_width": row.source_width,
            "source_height": row.source_height,
            "width": row.width,
            "height": row.height,
            "minutiae": minutiae,
        }


def test_null_and_empty_minutiae_are_told_apart(tmp_path):
    rows = [
        sample_row(1, 0, minutiae=None),
        sample_row(1, 1, minutiae=np.zeros((0, 3), dtype=np.float32)),
        sample_row(1, 2, minutiae=np.array([[3.5, 4.25, 0.5]], dtype=np.float32)),
        sample_row(2, 0, minutiae=None),
    ]
    write_table(tmp_path / "table", rows, parts=1)
    samples = read_samples(str(tmp_path / "table"))
    assert samples.minutiae_known.tolist() == [False, True, True, False]
    assert [len(samples.minutiae_of(row)) for row in range(4)] == [0, 0, 1, 0]
    assert samples.minutiae_offsets.tolist() == [0, 0, 0, 1, 1]
    assert samples.minutiae_of(2).tolist() == [[3.5, 4.25, 0.5]]


def test_the_order_of_the_rows_in_a_file_and_its_row_groups_mean_nothing(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    rng = np.random.default_rng(3)
    for path in paths:
        change_part(
            path, lambda table: table.take(rng.permutation(table.num_rows)), row_group_size=3
        )
        assert pq.read_metadata(path).num_row_groups == 3  # control: several row groups
    # Control: the files do not hold the rows in the order of (subject, impression)
    subjects = pq.read_table(paths[0]).column("subject").to_pylist()
    assert subjects != sorted(subjects)
    assert_same_rows(read_samples(str(tmp_path / "table")), samples_from_rows(ROWS))


def stored_numbers(path) -> list:
    """(subject, impression) of the rows of a part, in the order of the file."""
    table = pq.read_table(path, columns=["subject", "impression"])
    return list(zip(table.column("subject").to_pylist(), table.column("impression").to_pylist()))


def test_rows_of_one_subject_that_are_not_in_the_order_of_their_impressions_are_sorted(tmp_path):
    """
    The subjects in order and the impressions of a subject not: what files
    give that are listed by their names (5_1.png, 5_10.png, 5_2.png).
    """
    rows = [sample_row(subject, impression) for subject in (2, 4) for impression in (0, 1, 2)]
    (path,) = write_table(tmp_path / "table", rows, parts=1)
    change_part(path, lambda table: table.take([2, 0, 1, 5, 3, 4]))
    # Control: no subject is smaller than the one before it
    assert stored_numbers(path) == [(2, 2), (2, 0), (2, 1), (4, 2), (4, 0), (4, 1)]
    samples = read_samples(str(tmp_path / "table"))
    assert list(zip(samples.subject.tolist(), samples.impression.tolist())) == [
        (row.subject, row.impression) for row in rows
    ]
    assert_same_rows(samples, samples_from_rows(rows))
    assert [samples.find(row.subject, row.impression) for row in rows] == list(range(6))
    assert samples_checksum(samples) == expected_checksum(rows)


def test_rows_behind_a_first_row_with_the_smallest_subject_are_sorted(tmp_path):
    """Rows out of order that would pass for sorted if each were compared with the first one."""
    rows = [sample_row(subject, 0) for subject in (0, 3, 5, 8)]
    (path,) = write_table(tmp_path / "table", rows, parts=1)
    change_part(path, lambda table: table.take([0, 3, 1, 2]))
    assert stored_numbers(path) == [(0, 0), (8, 0), (3, 0), (5, 0)]  # control
    samples = read_samples(str(tmp_path / "table"))
    assert samples.subject.tolist() == [0, 3, 5, 8]
    assert_same_rows(samples, samples_from_rows(rows))
    assert [samples.find(subject, 0) for subject in (0, 3, 5, 8)] == [0, 1, 2, 3]
    assert samples_checksum(samples) == expected_checksum(rows)


def test_the_same_numbers_in_rows_that_are_no_neighbours_in_the_file_are_refused(tmp_path):
    """
    The rule that the two numbers are unique compares rows that follow each
    other, so it rests on the order: two rows with the same numbers and a row
    of the same subject between them, in a file whose subjects are in order.
    """
    rows = [sample_row(1, impression) for impression in (0, 1, 2)] + [sample_row(2, 0)]
    rows.append(sample_row(1, 0, image_path="again.png"))
    (path,) = write_table(tmp_path / "table", rows, parts=1)
    assert stored_numbers(path) == [(1, 0), (1, 0), (1, 1), (1, 2), (2, 0)]
    change_part(path, lambda table: table.take([0, 2, 1, 3, 4]))
    assert stored_numbers(path) == [(1, 0), (1, 1), (1, 0), (1, 2), (2, 0)]  # control
    message = refusal(tmp_path / "table")
    assert "The same subject and impression in 2 of 5 rows" in message
    assert row_name(rows[0]) in message and "'again.png' (subject 1, impression 0)" in message


def test_batches_that_are_slices_of_larger_arrays_are_read(tmp_path, monkeypatch):
    """
    How pyarrow cuts the rows of a file into batches is its own matter. Here
    every batch is a slice of the arrays of the whole part, so its values
    begin somewhere inside of their buffers.
    """
    write_table(tmp_path / "table", ROWS)
    offsets = set()

    def slices(self, batch_size=None, columns=None, **options):
        (whole,) = self.read(columns=columns).combine_chunks().to_batches()
        for start in range(0, whole.num_rows, 3):
            batch = whole.slice(start, 3)
            offsets.update(column.offset for column in batch.columns)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", slices)
    samples = read_samples(str(tmp_path / "table"))
    assert offsets == {0, 3, 6}  # control: the batches were slices
    assert_same_rows(samples, samples_from_rows(ROWS))


def many_rows(count: int) -> list:
    """Rows with long paths and up to 60 minutiae each: more than one batch and one block."""
    rng = np.random.default_rng(11)
    counts = rng.integers(0, 61, size=count)
    rows = []
    for number in range(count):
        minutiae = None
        if number % 20 != 7:
            minutiae = rng.random((int(counts[number]), 3), dtype=np.float32) * 500
        rows.append(
            SampleRow(
                subject=2 * (number // 3),
                impression=number % 3,
                image_path=f"sensor_{number % 7}/session_{number % 3}/finger_{number:07d}.png",
                file_size=100_000 + number,
                ppi=500.0,
                ppi_origin=PPI_ORIGINS[number % 4],
                ridges="light" if number % 5 == 0 else "dark",
                ridges_origin=RIDGES_ORIGINS[number % 4],
                source_width=400 + number % 100,
                source_height=500 + number % 90,
                width=400 + number % 100,
                height=500 + number % 90,
                minutiae=minutiae,
            )
        )
    return rows


@pytest.fixture(scope="module")
def large_rows():
    return many_rows(50_000)


@pytest.fixture(scope="module")
def large_samples(large_rows):
    """
    More rows than one batch of the reader, and more path bytes and minutiae
    than one block of the validation: the pieces a table is read and checked in.
    """
    samples = samples_from_rows(large_rows)
    assert len(samples) // 3 > sample_table._BATCH_ROWS
    assert len(samples.path_data) > sample_table._BLOCK_BYTES
    assert len(samples.minutiae) > sample_table._BLOCK_ITEMS
    assert samples.minutiae.size > sample_table._BLOCK_VALUES
    return samples


def test_a_table_that_is_larger_than_the_pieces_it_is_read_in(tmp_path, large_rows, large_samples):
    """In files that are in the order of the rows, and in files that are not."""
    rows, expected = large_rows, large_samples
    # Every part holds rows from all over the table
    write_table(tmp_path / "mixed", rows, parts=3)
    mixed = read_samples(str(tmp_path / "mixed"))
    assert_same_rows(mixed, expected)
    # The parts follow each other in the order of the rows
    third = len(rows) // 3
    for index in range(3):
        part = samples_from_rows(rows[index * third : (index + 1) * third if index < 2 else None])
        write_part(str(tmp_path / "ordered"), index, 3, part, {"id": "ordered"})
    ordered = read_samples(str(tmp_path / "ordered"))
    assert_same_rows(ordered, expected)
    assert samples_checksum(mixed) == samples_checksum(ordered) == samples_checksum(expected)
    # Rows from both ends and the middle
    for position in (0, 1, 16_383, 16_384, 25_000, 49_999):
        row = rows[position]
        assert mixed.image_path(position) == row.image_path
        assert mixed.find(row.subject, row.impression) == position


def test_a_table_without_rows_and_a_part_without_rows(tmp_path):
    write_table(tmp_path / "empty", [], parts=2)
    empty = read_samples(str(tmp_path / "empty"))
    assert len(empty) == 0 and empty.minutiae.shape == (0, 3)
    assert empty.path_offsets.tolist() == [0] and empty.find(0, 0) == -1
    assert samples_checksum(empty) == samples_checksum(samples_from_rows([]))

    write_part(str(tmp_path / "table"), 0, 2, samples_from_rows(ROWS), {"id": "a"})
    write_part(str(tmp_path / "table"), 1, 2, samples_from_rows([]), {"id": "a"})
    assert_same_rows(read_samples(str(tmp_path / "table")), samples_from_rows(ROWS))


def test_the_arrays_of_a_table_cannot_be_written_to(tmp_path):
    write_table(tmp_path / "table", ROWS)
    for samples in (read_samples(str(tmp_path / "table")), samples_from_rows(ROWS)):
        for name in ARRAYS:
            array = getattr(samples, name)
            assert not array.flags.writeable, name
            with pytest.raises(ValueError, match="read-only"):
                array[:1] = 0
        with pytest.raises(ValueError, match="read-only"):
            samples.minutiae_of(len(samples) - 1)[:] = 0.0


def test_a_copy_of_a_table_cannot_be_written_to_either(tmp_path):
    """
    numpy does not keep an array read-only when it is pickled or copied. A
    pickled copy is what a process gets that is started with the table as an
    argument and is not a fork of the process that read it.
    """
    paths = write_table(tmp_path / "table", ROWS, build={"id": "a", "files": 9})
    change_part(
        paths[0], lambda table: table.append_column("split", pa.array(["train"] * table.num_rows))
    )
    made = {"read": read_samples(str(tmp_path / "table")), "from rows": samples_from_rows(ROWS)}
    # Control: the table that was read has more to copy than its arrays
    assert made["read"].unknown_columns == ("split",)
    assert made["read"].build == {"id": "a", "parts": 2, "rows": 15, "files": 18}
    for origin, samples in made.items():
        with_minutiae = int(np.flatnonzero(np.diff(samples.minutiae_offsets))[0])
        copies = {
            "pickle": pickle.loads(pickle.dumps(samples)),
            "copy.deepcopy": copy.deepcopy(samples),
            "copy.copy": copy.copy(samples),
        }
        for how, copied in copies.items():
            assert type(copied) is Samples and copied is not samples
            for name in ARRAYS:
                array = getattr(copied, name)
                assert not array.flags.writeable, (origin, how, name)
                with pytest.raises(ValueError, match="read-only"):
                    array[:1] = 0
            with pytest.raises(ValueError, match="read-only"):
                copied.minutiae_of(with_minutiae)[:] = 0.0
            # And it is the same table
            assert_same_rows(copied, samples)
            assert copied.build == samples.build
            assert copied.unknown_columns == samples.unknown_columns
            assert samples_checksum(copied) == samples_checksum(samples)
            for position, row in enumerate(sorted(ROWS, key=lambda r: (r.subject, r.impression))):
                assert copied.find(row.subject, row.impression) == position
                assert copied.image_path(position) == row.image_path
        # A deep copy and an unpickled table have memory of their own
        for how in ("pickle", "copy.deepcopy"):
            assert not np.shares_memory(copies[how].minutiae, samples.minutiae), (origin, how)


def test_find_gives_the_row_or_minus_one():
    samples = samples_from_rows(ROWS)
    assert samples.find(0, 0) == 0
    assert samples.find(17, 2) == len(ROWS) - 1
    assert samples.find(np.int32(3), np.int64(1)) == 4
    for missing in (
        (1, 0),
        (0, 3),
        (18, 0),
        (-1, 0),
        (0, -1),
        (2**31, 0),
        (0, 2**40),
        (-(2**70), 0),
    ):
        assert samples.find(*missing) == -1
    with pytest.raises(TypeError):
        samples.find(0.0, 0)
    with pytest.raises(IndexError):
        samples.image_path(len(ROWS))
    with pytest.raises(IndexError):
        samples.minutiae_of(-1)

    # The largest and the smallest numbers the columns hold. A table must not
    # hold negative ones, but a table that does is not searched wrongly.
    largest, smallest = 2**31 - 1, -(2**31)
    pairs = [(smallest, smallest), (smallest, largest), (-1, -1), (-1, 0), (0, 0)]
    pairs += [(0, largest), (1, 0), (largest, 0), (largest, largest)]
    rows = [
        dataclasses.replace(sample_row(4, 0), subject=pair[0], impression=pair[1])
        for pair in pairs[::-1]
    ]
    samples = samples_from_rows(rows)
    assert list(zip(samples.subject.tolist(), samples.impression.tolist())) == pairs
    assert [samples.find(*pair) for pair in pairs] == list(range(len(pairs)))
    assert samples.find(0, 1) == samples.find(largest, 1) == samples.find(smallest, 0) == -1


def run_alone(code: str, *arguments: str) -> subprocess.CompletedProcess:
    """Runs the code in a new interpreter, in the repository."""
    return subprocess.run(
        [sys.executable, "-c", code, *arguments], cwd=REPOSITORY, capture_output=True, text=True
    )


def test_a_table_is_written_and_read_by_processes_that_import_nothing_of_pyarrow(tmp_path):
    """
    This file imports pyarrow.parquet itself. Its other tests would therefore
    pass for a module that imports pyarrow alone and then uses
    pyarrow.parquet, which is only there once somebody has imported it: not
    in a script that reads a table and imports nothing but this module.
    """
    start = (
        "import sys\n"
        "from flx.preparation.sample_table import SampleRow, read_samples, samples_from_rows\n"
        "from flx.preparation.sample_table import write_part\n"
        # Control: the module did not import pyarrow when it was imported
        "assert not [name for name in sys.modules if name.split('.')[0] == 'pyarrow']\n"
    )
    write = (
        "row = SampleRow(4, 1, 'a/5_2.png', 10, 500.0, 'argument', 'dark', 'default', 8, 9, 8, 9,\n"
        "                None)\n"
        "print(write_part(sys.argv[1], 0, 1, samples_from_rows([row]), {'id': 'a'}))\n"
    )
    read = (
        "samples = read_samples(sys.argv[1])\n"
        "print(len(samples), samples.image_path(0), samples.find(4, 1))\n"
    )
    directory = str(tmp_path / "table")
    result = run_alone(start + write, directory)
    assert result.returncode == 0, result.stderr
    # The whole line: the path of a temporary directory can hold a space
    assert result.stdout == os.path.join(directory, part_name(0, 1)) + "\n"
    # Another process: one that reads a table has usually not written one before
    result = run_alone(start + read, directory)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1", "a/5_2.png", "0"]


# ---------------------------------------------------------------------------
# What a part says about itself
# ---------------------------------------------------------------------------


def test_every_part_carries_the_metadata_of_the_table(tmp_path):
    settings = {"target_ppi": 500.0, "ppi": None, "ridges": "auto"}
    build = {"id": "2026-11-22 a", "files": 5, "failed": 0, "settings": settings}
    paths = write_table(tmp_path / "table", ROWS, parts=3, build=build)
    assert [os.path.basename(path) for path in paths] == [part_name(i, 3) for i in range(3)]
    for index, path in enumerate(paths):
        metadata = pq.read_metadata(path).metadata
        assert metadata[b"flx.table"] == b"samples"
        assert metadata[b"flx.schema_version"] == b"1"
        assert metadata[b"flx.target_ppi"] == b"500.0"
        assert json.loads(metadata[b"flx.build"]) == dict(
            build, part=index, parts=3, rows=len(ROWS[index::3])
        )
    assert build == {"id": "2026-11-22 a", "files": 5, "failed": 0, "settings": settings}


def test_every_field_of_a_part_has_its_type_and_a_description(tmp_path):
    (path,) = write_table(tmp_path / "table", ROWS, parts=1)
    schema = pq.read_schema(path)
    assert schema.names == list(COLUMNS)
    expected = {
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
    }
    for name, arrow_type in expected.items():
        assert schema.field(name).type == arrow_type, name
        assert not schema.field(name).nullable, name
    minutiae = schema.field("minutiae")
    assert minutiae.nullable and pa.types.is_list(minutiae.type)
    minutia = minutiae.type.value_type
    assert [(field.name, field.type) for field in minutia] == [
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("angle", pa.float32()),
    ]

    # Unit and meaning of every field can be read from the file alone
    fields = list(schema) + list(minutia)
    descriptions = {field.name: field.metadata[b"description"].decode() for field in fields}
    assert set(descriptions) == set(COLUMNS) | {"x", "y", "angle"}
    for name, text in descriptions.items():
        assert len(text) > 30 and text.endswith("."), name
    for name, unit in (
        ("file_size", "bytes"),
        ("ppi", "pixels per inch"),
        ("source_width", "pixels"),
        ("source_height", "pixels"),
        ("width", "pixels"),
        ("height", "pixels"),
        ("x", "pixels"),
        ("y", "pixels"),
        ("angle", "radians"),
    ):
        assert unit in descriptions[name], name
    for value in PPI_ORIGINS:
        assert repr(value) in descriptions["ppi_origin"]
    for value in RIDGES_ORIGINS:
        assert repr(value) in descriptions["ridges_origin"]
    assert "'dark'" in descriptions["ridges"] and "'light'" in descriptions["ridges"]
    assert "Null" in descriptions["minutiae"] and "empty list" in descriptions["minutiae"]
    # A resolution is assumed for two kinds of files, and who reads a part without this
    # repository must not take the rows for one kind only: files that state no resolution,
    # and files that state one which the preparation does not believe (the example images
    # of this repository state 96 ppi)
    about_assumed = descriptions["ppi_origin"].partition("'assumed'")[2]
    assert "state no resolution" in about_assumed and "not plausible" in about_assumed


def test_the_build_information_of_the_parts_is_merged(tmp_path):
    directory = str(tmp_path / "table")
    settings = {"ppi": None, "assume_ppi": 500.0}
    for index, (files, failed) in enumerate(((7, 2), (8, 0), (5, 1))):
        build = {"id": "b", "files": files, "failed": failed, "settings": settings}
        build["note"] = f"part {index}"
        write_part(directory, index, 3, samples_from_rows(ROWS[index::3]), build)
    assert read_samples(directory).build == {
        "id": "b",
        "parts": 3,
        "rows": len(ROWS),
        "files": 20,
        "failed": 3,
        "settings": settings,
        "note": "part 0",  # any other key is that of part 0
    }

    # Lists are joined in the order of the parts; a part can lack a key
    directory = str(tmp_path / "lists")
    for index, failed in enumerate((["a.png"], None, ["b.png", "c.png"])):
        build = {"id": "b"} if failed is None else {"id": "b", "failed": failed}
        write_part(directory, index, 3, samples_from_rows(ROWS[index::3]), build)
    assert read_samples(directory).build == {
        "id": "b",
        "parts": 3,
        "rows": len(ROWS),
        "failed": ["a.png", "b.png", "c.png"],
    }

    # A number in one part and a list in another cannot be added
    write_part(directory, 1, 3, samples_from_rows(ROWS[1::3]), {"id": "b", "failed": 4})
    message = refusal(directory)
    assert part_name(1, 3) in message and '"failed"' in message and BUILDER in message


# ---------------------------------------------------------------------------
# What a reader refuses: files that are no table
# ---------------------------------------------------------------------------


def test_a_directory_without_parts_is_refused(tmp_path):
    message = refusal(tmp_path / "missing")
    assert str(tmp_path / "missing") in message and BUILDER in message
    assert "part-00000-of-00003.parquet" in message  # how the files of a table are named

    (tmp_path / "file").write_text("no directory")
    assert str(tmp_path / "file") in refusal(tmp_path / "file")

    (tmp_path / "empty").mkdir()
    message = refusal(tmp_path / "empty")
    assert "holds no part" in message and "no files" in message and BUILDER in message

    # Parquet files that are not named as parts are not taken for a table
    (path,) = write_table(tmp_path / "other", ROWS, parts=1)
    os.rename(path, tmp_path / "other" / "samples.parquet")
    message = refusal(tmp_path / "other")
    assert "holds no part" in message and "samples.parquet" in message


def test_a_part_that_is_no_parquet_file_is_refused(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    with open(paths[1], "wb") as file:
        file.write(b"not a Parquet file")
    message = refusal(tmp_path / "table")
    assert paths[1] in message and "cannot be read as a Parquet file" in message
    assert BUILDER in message

    # Half a part, as a copy that broke off leaves it
    paths = write_table(tmp_path / "half", ROWS)
    with open(paths[0], "rb") as file:
        content = file.read()
    with open(paths[0], "wb") as file:
        file.write(content[: len(content) // 2])
    message = refusal(tmp_path / "half")
    assert paths[0] in message and "cannot be read as a Parquet file" in message


def test_a_part_whose_footer_cannot_be_decoded_is_refused(tmp_path):
    """
    The footer of the part is overwritten. What follows it is as before: the
    length of the footer and the letters PAR1, so the file still ends as a
    Parquet file does. For a file that is no Parquet file pyarrow raises an
    error class of its own; for a footer that it cannot decode it raises an
    OSError, which is none. Both must become the same refusal.
    """
    paths = write_table(tmp_path / "table", ROWS)
    with open(paths[0], "rb") as file:
        content = file.read()
    footer = footer_position(content)
    with open(paths[0], "wb") as file:
        file.write(content[:footer] + b"\xff" * (len(content) - 8 - footer) + content[-8:])
    with pytest.raises(OSError) as error:  # control
        pq.ParquetFile(paths[0])
    assert not isinstance(error.value, pa.ArrowException)
    message = refusal(tmp_path / "table")
    assert message.startswith(f"{paths[0]} cannot be read as a Parquet file (OSError: ")
    assert "damaged or replaced later" in message and BUILDER in message


def test_a_part_with_a_column_name_that_is_not_utf8_is_refused(tmp_path):
    """
    One changed byte in the footer of a part: where the schema names the
    first column. What pyarrow raises for such a file is no error class of
    its own, and it must still become a refusal that names the part.
    """
    paths = write_table(tmp_path / "table", ROWS)
    with open(paths[0], "rb") as file:
        content = file.read()
    footer = footer_position(content)
    assert b"subject" in content[footer:]
    with open(paths[0], "wb") as file:
        file.write(content[:footer] + content[footer:].replace(b"subject", b"\xf3ubject", 1))
    with pytest.raises(UnicodeDecodeError):  # control
        pq.ParquetFile(paths[0])
    message = refusal(tmp_path / "table")
    assert paths[0] in message and "cannot be read as a Parquet file" in message
    assert "UnicodeDecodeError" in message and BUILDER in message


def refusal_for_its_rows(directory, path) -> str:
    """The message with which a table is refused because the rows of the part cannot be read."""
    message = refusal(directory)
    assert message.startswith(f"{path} cannot be read to its end (")
    assert "damaged later, or it changed while the table was read" in message
    assert "Copy the part again from where the table was built" in message and BUILDER in message
    return message


def test_a_part_whose_rows_are_damaged_is_refused(tmp_path):
    """
    The footer of the part is whole and says what the part of a table says;
    the pages with the rows before it are overwritten. pyarrow notices when
    the rows are read, with an error of its own that must become a refusal
    which names the part.
    """
    paths = write_table(tmp_path / "table", ROWS)
    with open(paths[1], "rb") as file:
        content = file.read()
    footer = footer_position(content)
    with open(paths[1], "wb") as file:
        file.write(content[:4] + b"\xff" * (footer - 4) + content[footer:])
    # Control: the metadata of the file are read as before, its rows are not
    assert pq.read_metadata(paths[1]).num_rows == 7
    with pytest.raises(OSError):
        pq.read_table(paths[1])
    message = refusal_for_its_rows(tmp_path / "table", paths[1])
    assert "(OSError: " in message and "; 0 of its 7 rows were read before that. " in message


def test_damage_that_pyarrow_alone_reads_as_other_values_is_refused(tmp_path):
    """
    Most damage in the pages of a Parquet file is not noticed when the file
    is read: the rows come back with other values. The parts are therefore
    written with the checksums of their pages, which are compared when the
    rows are read. Here one bit is inverted in the pages of every column in
    turn, each time a bit with which pyarrow alone reads the part without an
    error and with other values.
    """
    directory = tmp_path / "table"
    (path,) = write_table(directory, ROWS, parts=1)
    with open(path, "rb") as file:
        content = file.read()
    intact = pq.read_table(path)
    checksum = samples_checksum(read_samples(str(directory)))
    row_group = pq.read_metadata(path).row_group(0)
    assert row_group.num_columns == 15  # 12 columns of one value, and x, y, angle of the minutiae
    for index in range(row_group.num_columns):
        column = row_group.column(index)
        start = column.dictionary_page_offset or column.data_page_offset
        for position in range(start, start + column.total_compressed_size):
            damaged = bytearray(content)
            damaged[position] ^= 1
            with open(path, "wb") as file:
                file.write(damaged)
            try:
                unnoticed = not pq.read_table(path).equals(intact)
            except (pa.ArrowException, OSError):
                unnoticed = False  # this bit pyarrow misses by itself
            if unnoticed:
                break
        else:
            pytest.fail(f"Control: no bit of {column.path_in_schema} is read as another value")
        message = refusal_for_its_rows(directory, path)
        assert "; 0 of its 15 rows were read before that. " in message, column.path_in_schema
    with open(path, "wb") as file:
        file.write(content)
    assert samples_checksum(read_samples(str(directory))) == checksum


def test_a_part_without_checksums_of_its_pages_is_read(tmp_path):
    """
    Another tool need not write the checksums, and a part without them is
    read like any other. Control: written again by pyarrow alone the part is
    shorter, and it has its length again when the checksums are asked for.
    """
    directory = tmp_path / "table"
    (path,) = write_table(directory, ROWS, parts=1)
    expected = read_samples(str(directory))
    with_checksums = os.path.getsize(path)
    table = pq.read_table(path)
    pq.write_table(table, path, write_page_checksum=True)
    assert os.path.getsize(path) == with_checksums
    pq.write_table(table, path)
    assert os.path.getsize(path) < with_checksums
    assert_same_rows(read_samples(str(directory)), expected)


def test_a_part_whose_footer_does_not_fit_its_rows_is_refused(tmp_path):
    """
    A footer that can be read and passes for the footer of the part, but
    does not say where the rows of this file are: the footer of a file that
    holds the same rows in smaller row groups.
    """
    paths = write_table(tmp_path / "table", ROWS)
    other = str(tmp_path / "other.parquet")
    pq.write_table(pq.read_table(paths[0]), other, row_group_size=2)
    with open(paths[0], "rb") as file:
        content = file.read()
    with open(other, "rb") as file:
        other_content = file.read()
    with open(paths[0], "wb") as file:
        file.write(
            content[: footer_position(content)] + other_content[footer_position(other_content) :]
        )
    # Control: the file states the rows and the metadata of the part, and its rows cannot be read
    metadata = pq.read_metadata(paths[0])
    assert (metadata.num_rows, metadata.num_row_groups) == (8, 4)
    assert json.loads(metadata.metadata[b"flx.build"])["rows"] == 8
    with pytest.raises(OSError):
        pq.read_table(paths[0])
    message = refusal_for_its_rows(tmp_path / "table", paths[0])
    assert "; 0 of its 8 rows were read before that. " in message


def test_an_error_of_arrow_behind_the_first_rows_of_a_part_is_a_refusal(
    tmp_path, monkeypatch, large_rows
):
    """
    Damaged pages make pyarrow raise errors of several classes, and not only
    for the first rows of a file. Here it is one that a damaged page gave
    (ArrowInvalid), raised when the second batch of rows is asked for.
    """
    rows = large_rows[:20_000]
    (path,) = write_table(tmp_path / "table", rows, parts=1)
    iter_batches = pq.ParquetFile.iter_batches
    given = []

    def breaks_off(self, *arguments, **options):
        batches = iter_batches(self, *arguments, **options)
        given.append(next(batches))
        yield given[0]
        raise pa.ArrowInvalid("Output buffer size (48) must be 50 or larger.")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", breaks_off)
    message = refusal_for_its_rows(tmp_path / "table", path)
    assert "(ArrowInvalid: Output buffer size (48) must be 50 or larger.)" in message
    rows_read = given[0].num_rows
    assert 0 < rows_read < 20_000  # control: the error came behind the first rows
    assert f"; {rows_read:,} of its 20,000 rows were read before that. " in message


def test_a_parquet_file_that_is_no_sample_table_is_refused(tmp_path):
    """The columns alone do not make a table: the file metadata must say what it is."""
    (path,) = write_table(tmp_path / "table", ROWS, parts=1)
    change_part(path, metadata={"flx.table": None})
    message = refusal(tmp_path / "table")
    assert path in message and "not a part of a sample table" in message
    assert "flx.table" in message and BUILDER in message

    change_part(path, metadata={"flx.table": "embeddings"})
    message = refusal(tmp_path / "table")
    assert path in message and "'embeddings'" in message and "'samples'" in message
    assert f"Give the directory that {BUILDER} wrote" in message

    change_part(path, metadata={"flx.table": "samples", "flx.schema_version": None})
    assert "flx.schema_version" in refusal(tmp_path / "table")

    for key, value in (
        ("flx.target_ppi", None),
        ("flx.target_ppi", "high"),
        ("flx.build", None),
        ("flx.build", "{no json"),
        ("flx.build", '["a list"]'),
        ("flx.build", '{"id": 7, "part": 0, "parts": 1, "rows": 15}'),
        ("flx.build", '{"id": "a", "part": "0", "parts": 1, "rows": 15}'),
        ("flx.build", '{"id": "a", "part": 1, "parts": 1, "rows": 15}'),
    ):
        (path,) = write_table(tmp_path / "table", ROWS, parts=1)
        change_part(path, metadata={key: value})
        message = refusal(tmp_path / "table")
        assert path in message and key in message and BUILDER in message, (key, value)


def test_another_schema_version_is_refused_and_the_message_says_which_one_is_read(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    change_part(paths[1], metadata={"flx.schema_version": "2"})
    message = refusal(tmp_path / "table")
    assert paths[1] in message
    assert "schema version '2'" in message and "reads version 1 only" in message
    assert "version of this repository that wrote it" in message


@pytest.mark.parametrize("column", COLUMNS)
def test_a_missing_column_is_refused(tmp_path, column):
    paths = write_table(tmp_path / "table", ROWS)
    change_part(paths[0], lambda table: table.drop_columns([column]))
    message = refusal(tmp_path / "table")
    assert paths[0] in message and f"lacks the column(s) {column};" in message
    assert BUILDER in message


def test_a_column_that_is_there_twice_is_refused(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    change_part(paths[1], lambda table: table.append_column("ppi", table.column("ppi")))
    assert pq.read_schema(paths[1]).names.count("ppi") == 2  # control: Parquet allows it
    message = refusal(tmp_path / "table")
    assert paths[1] in message and "has the column ppi 2 times" in message
    assert BUILDER in message


def _minutiae_type(item_type=None, list_of=None, names=("x", "y", "angle")):
    item_type = pa.float32() if item_type is None else item_type
    return (pa.list_ if list_of is None else list_of)(
        pa.struct([(name, item_type) for name in names])
    )


def _other_minutiae(arrow_type):
    """The minutiae of a table written as another type (the fields are matched by name)."""
    return lambda table: pa.array(table.column("minutiae").to_pylist(), type=arrow_type)


@pytest.mark.parametrize(
    "column, values, found, expected",
    [
        ("ppi", lambda t: t.column("ppi").cast(pa.float32()), "float", "double"),
        ("subject", lambda t: t.column("subject").cast(pa.int64()), "int64", "int32"),
        ("impression", lambda t: t.column("impression").cast(pa.int16()), "int16", "int32"),
        ("file_size", lambda t: t.column("file_size").cast(pa.int32()), "int32", "int64"),
        ("width", lambda t: t.column("width").cast(pa.float64()), "double", "int32"),
        (
            "image_path",
            lambda t: t.column("image_path").cast(pa.large_string()),
            "large_string",
            "string",
        ),
        (
            "ridges",
            lambda t: t.column("ridges").dictionary_encode(),
            "dictionary<",
            "string",
        ),
        (
            "ridges_origin",
            lambda t: t.column("ridges_origin").cast(pa.binary()),
            "binary",
            "string",
        ),
        (
            "minutiae",
            lambda t: _other_minutiae(_minutiae_type(pa.float64()))(t),
            "double",
            "list<struct<x: float, y: float, angle: float>>",
        ),
        (
            "minutiae",
            lambda t: _other_minutiae(_minutiae_type(list_of=pa.large_list))(t),
            "large_list<",
            "list<struct<x: float, y: float, angle: float>>",
        ),
        (
            "minutiae",
            lambda t: _other_minutiae(_minutiae_type(names=("y", "x", "angle")))(t),
            "struct<y: float",
            "list<struct<x: float, y: float, angle: float>>",
        ),
        (
            "minutiae",
            lambda t: _other_minutiae(_minutiae_type(names=("x", "y", "angle", "quality")))(t),
            "quality",
            "list<struct<x: float, y: float, angle: float>>",
        ),
        (
            "minutiae",
            lambda t: pa.array([[1.0, 2.0, 0.5]] * t.num_rows, type=pa.list_(pa.float32())),
            "list<",
            "list<struct<x: float, y: float, angle: float>>",
        ),
    ],
)
def test_a_column_of_another_type_is_refused_and_not_cast(
    tmp_path, column, values, found, expected
):
    paths = write_table(tmp_path / "table", ROWS)
    change_part(paths[1], with_column(column, values))
    message = refusal(tmp_path / "table")
    assert paths[1] in message
    assert f"the column {column} has the type" in message
    position = message.index("has the type")
    assert found in message[position:] and f"where a sample table has {expected} " in message
    assert "not converted" in message and BUILDER in message


def test_a_ppi_of_32_bits_gives_another_prepared_size():
    """The control for the float64 column: with the rounded value the image has another size."""
    rounded = float(np.float32(251.2))
    assert rounded != 251.2
    assert resampled_size(785, 785, 251.2) == (1562, 1562)
    assert resampled_size(785, 785, rounded) == (1563, 1563)
    # A table whose ppi went through 32 bits no longer holds the sizes of its images
    row = sample_row(2, 0, source_width=785, source_height=785, ppi=251.2, width=1562, height=1562)
    validate_samples(samples_from_rows([row]))
    message = invalid([dataclasses.replace(row, ppi=rounded)])
    assert f"source 785x785 at {rounded!r} ppi gives 1563x1563 at 500 ppi" in message
    assert "the row holds 1562x1562" in message


def test_the_name_of_the_item_field_of_the_minutiae_list_is_free(tmp_path):
    """Parquet writers call the item of a list "element", Arrow calls it "item"."""
    paths = write_table(tmp_path / "table", ROWS)
    assert pq.read_schema(paths[0]).field("minutiae").type.value_field.name == "element"
    change_part(
        paths[0],
        with_column("minutiae", _other_minutiae(_minutiae_type())),
        use_compliant_nested_type=False,
    )
    # Control: this part names the item differently, and declares every field as nullable
    stored = pq.read_schema(paths[0]).field("minutiae").type
    assert stored.value_field.name == "item" and stored.value_field.nullable
    assert_same_rows(read_samples(str(tmp_path / "table")), samples_from_rows(ROWS))


@pytest.mark.parametrize("column", [name for name in COLUMNS if name != "minutiae"])
def test_a_null_outside_of_minutiae_is_refused(tmp_path, column):
    paths = write_table(tmp_path / "table", ROWS)

    def with_a_null(table):
        values = table.column(column).to_pylist()
        values[2] = None
        return pa.array(values, type=table.schema.field(column).type)

    path_of_the_row = pq.read_table(paths[1]).column("image_path")[2].as_py()
    change_part(paths[1], with_column(column, with_a_null))
    message = refusal(tmp_path / "table")
    assert paths[1] in message
    assert f"the column {column} holds 1 null value(s)" in message
    assert "row 2 of the part" in message and "Only minutiae can be null" in message
    if column != "image_path":
        assert repr(path_of_the_row) in message
    assert BUILDER in message


def test_a_null_is_named_by_its_row_in_the_part_also_behind_the_first_batch(tmp_path, large_rows):
    rows = large_rows[:20_000]
    assert len(rows) > sample_table._BATCH_ROWS  # control: one part, read in two batches
    (path,) = write_table(tmp_path / "table", rows, parts=1)

    def with_a_null(table):
        values = table.column("file_size").to_pylist()
        values[17_000] = None
        return pa.array(values, type=pa.int64())

    path_of_the_row = pq.read_table(path).column("image_path")[17_000].as_py()
    change_part(path, with_column("file_size", with_a_null))
    message = refusal(tmp_path / "table")
    assert "the column file_size holds 1 null value(s)" in message
    assert f"row 17,000 of the part (counted from 0), image_path {path_of_the_row!r}" in message


@pytest.mark.parametrize(
    "minutiae",
    [
        [{"x": 1.0, "y": 2.0, "angle": 0.5}, None],
        [{"x": 1.0, "y": None, "angle": 0.5}],
        [{"x": 1.0, "y": 2.0, "angle": None}],
    ],
)
def test_a_null_inside_a_list_of_minutiae_is_refused(tmp_path, minutiae):
    paths = write_table(tmp_path / "table", ROWS)

    def with_a_null(table):
        values = table.column("minutiae").to_pylist()
        values[1] = minutiae
        return pa.array(values, type=_minutiae_type())

    change_part(paths[0], with_column("minutiae", with_a_null))
    message = refusal(tmp_path / "table")
    assert paths[0] in message and "null minutia or a null x, y or angle" in message
    assert "Only the whole list of an image can be null" in message and BUILDER in message


@pytest.mark.parametrize(
    "column, text, allowed",
    [
        ("ppi_origin", "guessed", PPI_ORIGINS),
        ("ridges", "auto", ("dark", "light")),
        ("ridges_origin", "Argument", RIDGES_ORIGINS),
    ],
)
def test_a_text_that_is_no_value_of_its_column_is_refused(tmp_path, column, text, allowed):
    paths = write_table(tmp_path / "table", ROWS)

    def with_the_text(table):
        values = table.column(column).to_pylist()
        values[3] = text
        return pa.array(values, type=pa.string())

    path_of_the_row = pq.read_table(paths[0]).column("image_path")[3].as_py()
    change_part(paths[0], with_column(column, with_the_text))
    message = refusal(tmp_path / "table")
    assert paths[0] in message and f"the column {column} holds {text!r}" in message
    assert "row 3 of the part" in message and repr(path_of_the_row) in message
    for value in allowed:
        assert repr(value) in message
    assert BUILDER in message


@pytest.mark.parametrize(
    "column, allowed",
    [("ppi_origin", PPI_ORIGINS), ("ridges", ("dark", "light")), ("ridges_origin", RIDGES_ORIGINS)],
)
def test_a_text_that_is_not_utf8_is_refused_like_any_other_text(tmp_path, column, allowed):
    """
    A Parquet reader does not check that the bytes of a text are UTF-8, and
    such bytes cannot become a Python text. They are no value of the column,
    so the part and the row are named as for any other text.
    """
    paths = write_table(tmp_path / "table", ROWS)

    def with_other_bytes(table):
        texts = [text.encode() for text in table.column(column).to_pylist()]
        texts[3] = b"caf\xe9"  # Latin-1
        offsets = np.zeros(len(texts) + 1, dtype=np.int32)
        np.cumsum([len(text) for text in texts], out=offsets[1:])
        return pa.StringArray.from_buffers(
            len(texts), pa.py_buffer(offsets), pa.py_buffer(b"".join(texts))
        )

    path_of_the_row = pq.read_table(paths[0]).column("image_path")[3].as_py()
    change_part(paths[0], with_column(column, with_other_bytes))
    with pytest.raises(UnicodeDecodeError):  # control: the file holds these bytes
        pq.read_table(paths[0]).column(column).to_pylist()
    message = refusal(tmp_path / "table")
    assert paths[0] in message and f"the column {column} holds 'caf\\\\xe9'" in message
    assert "row 3 of the part" in message and repr(path_of_the_row) in message
    for value in allowed:
        assert repr(value) in message
    assert BUILDER in message


# ---------------------------------------------------------------------------
# What a reader refuses: parts that are not one whole build
# ---------------------------------------------------------------------------


def test_a_missing_part_is_refused(tmp_path):
    paths = write_table(tmp_path / "table", ROWS, parts=4)
    os.remove(paths[2])
    message = refusal(tmp_path / "table")
    assert str(tmp_path / "table") in message and "not a whole table" in message
    assert "it holds 3 of its 4 parts; missing: part-00002-of-00004.parquet. " in message
    assert BUILDER in message

    # Five names are given in full, and nothing is counted behind them
    paths = write_table(tmp_path / "six", ROWS, parts=6)
    for path in paths[1:]:
        os.remove(path)
    message = refusal(tmp_path / "six")
    assert "it holds 1 of its 6 parts; missing: part-00001-of-00006.parquet, " in message
    assert "part-00005-of-00006.parquet. " in message and " more" not in message


def test_a_part_that_is_there_twice_is_refused(tmp_path):
    """A copy of one part under the name of a missing one does not complete the table."""
    paths = write_table(tmp_path / "table", ROWS, parts=3)
    shutil.copyfile(paths[0], paths[2])
    message = refusal(tmp_path / "table")
    assert "more than once" in message
    assert "part 0: part-00000-of-00003.parquet, part-00002-of-00003.parquet" in message
    assert BUILDER in message


def test_a_renamed_part_is_refused(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    os.rename(paths[0], paths[0] + ".moved")
    os.rename(paths[1], paths[0])
    os.rename(paths[0] + ".moved", paths[1])
    message = refusal(tmp_path / "table")
    assert paths[0] in message and "is part 1 of 2 by its metadata" in message
    assert "part-00001-of-00002.parquet" in message and "renamed or copied" in message
    assert BUILDER in message


def test_parts_of_two_builds_are_refused(tmp_path):
    first = write_table(tmp_path / "table", ROWS, build={"id": "monday"})
    second = write_table(tmp_path / "other", ROWS, build={"id": "tuesday"})
    shutil.copyfile(second[1], first[1])
    message = refusal(tmp_path / "table")
    assert "parts of 2 builds" in message
    assert "build 'monday': part-00000-of-00002.parquet" in message
    assert "build 'tuesday': part-00001-of-00002.parquet" in message
    assert "into a new directory" in message and BUILDER in message


def test_parts_that_state_different_numbers_of_parts_are_refused(tmp_path):
    """What is left of a build with another number of parts in the same directory."""
    write_table(tmp_path / "table", ROWS, parts=2, build={"id": "a"})
    write_table(tmp_path / "table", ROWS, parts=3, build={"id": "a"})
    assert len(os.listdir(tmp_path / "table")) == 5
    message = refusal(tmp_path / "table")
    assert "different numbers of parts" in message and "(2, 3)" in message
    assert BUILDER in message


def test_a_part_with_another_number_of_rows_than_it_states_is_refused(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    change_part(paths[1], lambda table: table.slice(0, table.num_rows - 2))
    message = refusal(tmp_path / "table")
    assert paths[1] in message and "holds 5 rows, but its metadata state 7" in message
    assert BUILDER in message


def test_another_target_resolution_is_refused(tmp_path):
    # Rows that are right for prepared images of 1000 ppi
    rows = [
        dataclasses.replace(
            row,
            width=resampled_size(row.source_width, row.source_height, row.ppi, 1000.0)[0],
            height=resampled_size(row.source_width, row.source_height, row.ppi, 1000.0)[1],
        )
        for row in ROWS
    ]
    validate_samples(samples_from_rows(rows, target_ppi=1000.0))
    paths = write_table(tmp_path / "table", rows, target_ppi=1000.0)
    assert pq.read_metadata(paths[0]).metadata[b"flx.target_ppi"] == b"1000.0"
    message = refusal(tmp_path / "table")
    assert str(tmp_path / "table") in message
    assert "prepared at 1000.0 ppi" in message and "prepares images at 500.0 ppi" in message
    assert BUILDER in message

    paths = write_table(tmp_path / "mixed", ROWS)
    change_part(paths[1], metadata={"flx.target_ppi": "1000.0"})
    message = refusal(tmp_path / "mixed")
    assert "different target resolutions" in message and "500.0, 1000.0" in message
    assert BUILDER in message


def test_parts_built_with_different_settings_are_refused(tmp_path):
    directory = str(tmp_path / "table")
    paths = [
        write_part(
            directory,
            index,
            2,
            samples_from_rows(ROWS[index::2]),
            {"id": "a", "settings": {"ppi": ppi, "ridges": "dark"}},
        )
        for index, ppi in enumerate((None, 500.0))
    ]
    message = refusal(directory)
    assert paths[0] in message and paths[1] in message
    assert "other settings" in message and BUILDER in message
    # A part without settings beside one with settings is the same case
    write_part(directory, 1, 2, samples_from_rows(ROWS[1::2]), {"id": "a"})
    assert "other settings" in refusal(directory)


def test_columns_that_are_not_known_are_named_and_not_read(tmp_path, monkeypatch):
    paths = write_table(tmp_path / "table", ROWS)
    checksum = samples_checksum(read_samples(str(tmp_path / "table")))
    change_part(
        paths[0],
        lambda table: table.append_column(
            "person", pa.array(range(table.num_rows), type=pa.int64())
        ).append_column("finger_position", pa.array(["thumb"] * table.num_rows)),
    )
    change_part(
        paths[1],
        lambda table: table.append_column("split", pa.array(["train"] * table.num_rows)),
    )
    # What pyarrow hands out when the rows are read: another tool can have
    # added a column that is far larger than the table, pixels for example
    iter_batches = pq.ParquetFile.iter_batches
    handed_out = []

    def record(self, *arguments, **options):
        for batch in iter_batches(self, *arguments, **options):
            handed_out.append(sorted(batch.schema.names))
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", record)
    samples = read_samples(str(tmp_path / "table"))
    assert samples.unknown_columns == ("person", "finger_position", "split")
    assert handed_out == [sorted(COLUMNS)] * 2  # a batch per part, without the other columns
    assert_same_rows(samples, samples_from_rows(ROWS))
    assert samples_checksum(samples) == checksum


def test_other_files_in_the_directory_of_a_table_are_left_alone(tmp_path):
    paths = write_table(tmp_path / "table", ROWS)
    (tmp_path / "table" / "build_errors.txt").write_text("a.png\tunreadable\n")
    (tmp_path / "table" / "logs").mkdir()
    # What a build that was killed leaves of the part it was writing
    (tmp_path / "table" / (os.path.basename(paths[0]) + ".tmp4711")).write_bytes(b"half")
    assert_same_rows(read_samples(str(tmp_path / "table")), samples_from_rows(ROWS))


def refusal_for_its_number_of_rows(directory, path, rows: int) -> str:
    """
    The message with which a table is refused because pyarrow hands out
    another number of rows of the part than its metadata stated. It names
    both causes, and for each what to do.
    """
    message = refusal(directory)
    assert message.startswith(f"{path} does not hold the {rows:,} rows any more that its metadata")
    assert "the file changed while the table was read, or it was damaged. " in message
    assert "Read the table again when nothing writes to its directory; if it is refused" in message
    assert "copy the part again from where the table was built" in message and BUILDER in message
    return message


@pytest.mark.parametrize("rows_then", [3, 14])
def test_a_part_that_changes_while_the_table_is_read_is_refused(tmp_path, monkeypatch, rows_then):
    """
    The arrays of a table are made for the rows that the parts state before a
    row is read. A part that is written again in between, with fewer or with
    more rows, must not leave rows that were never filled.
    """
    paths = write_table(tmp_path / "table", ROWS)
    assert pq.read_metadata(paths[1]).num_rows == 7
    check_parts = sample_table._check_parts

    def write_a_part_again_then_check(directory, parts):
        # The metadata of all parts were read; their rows were not yet
        rows = pa.concat_tables([pq.read_table(paths[1])] * 2).slice(0, rows_then)
        change_part(paths[1], lambda table: rows, build={"rows": rows_then})
        check_parts(directory, parts)

    monkeypatch.setattr(sample_table, "_check_parts", write_a_part_again_then_check)
    refusal_for_its_number_of_rows(tmp_path / "table", paths[1], 7)


@pytest.mark.parametrize("handed_out", [0, 3])
def test_a_part_that_gives_fewer_rows_than_it_states_without_an_error_is_refused(
    tmp_path, monkeypatch, handed_out
):
    """
    A damaged part does not always make pyarrow raise an error. A footer in
    which one bit is inverted can still state the rows of its part and lead
    pyarrow to none of them, or to the first ones only. Nothing wrote to the
    file while the table was read then, and reading the table again gives the
    same refusal, so the refusal must not name a change of the file as its
    only cause. Here pyarrow hands out fewer rows of the second part than the
    footer of the part states, and raises nothing.
    """
    paths = write_table(tmp_path / "table", ROWS)
    iter_batches = pq.ParquetFile.iter_batches
    seen = []

    def fewer_rows_of_the_second_part(self, *arguments, **options):
        batches = iter_batches(self, *arguments, **options)
        if json.loads(self.metadata.metadata[b"flx.build"])["part"] != 1:
            yield from batches
            return
        first = next(batches).slice(0, handed_out)
        seen.append((self.metadata.num_rows, first.num_rows))
        if handed_out:
            yield first

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", fewer_rows_of_the_second_part)
    refusal_for_its_number_of_rows(tmp_path / "table", paths[1], 7)
    # Control: the rows that the footer of the part states, and the rows that were handed out
    assert seen == [(7, handed_out)]


@pytest.mark.parametrize(
    "then, error", [("removed", "FileNotFoundError"), ("no Parquet file", "ArrowInvalid")]
)
def test_a_part_that_cannot_be_opened_when_its_rows_are_read_is_refused(
    tmp_path, monkeypatch, then, error
):
    """A part that is removed or overwritten after the metadata of all parts were read."""
    paths = write_table(tmp_path / "table", ROWS)
    check_parts = sample_table._check_parts

    def change_a_part_then_check(directory, parts):
        os.remove(paths[1])
        if then == "no Parquet file":
            with open(paths[1], "wb") as file:
                file.write(b"not a Parquet file")
        check_parts(directory, parts)

    monkeypatch.setattr(sample_table, "_check_parts", change_a_part_then_check)
    message = refusal_for_its_rows(tmp_path / "table", paths[1])
    assert f"({error}: " in message and "; 0 of its 7 rows were read before that. " in message


def test_a_table_that_is_read_is_validated(tmp_path):
    """The same pair of numbers in two parts: each part alone is in order."""
    rows = ROWS + [sample_row(5, 2, image_path="again.png")]
    assert len(rows) % 2 == 0  # so that the two rows with (5, 2) go into different parts
    write_table(tmp_path / "table", rows)
    message = refusal(tmp_path / "table")
    assert message.startswith(str(tmp_path / "table") + ": ")
    assert "The same subject and impression in 2 of 16 rows" in message
    assert row_name(rows[0]) in message and "'again.png' (subject 5, impression 2)" in message


# ---------------------------------------------------------------------------
# The rules for the rows
# ---------------------------------------------------------------------------


def test_the_example_rows_are_valid():
    validate_samples(samples_from_rows(ROWS))
    validate_samples(samples_from_rows([]))


def test_negative_numbers_are_refused():
    lowest = dataclasses.replace(sample_row(4, 0), subject=-(2**31), impression=-(2**31))
    rows = ROWS + [sample_row(-1, 0), sample_row(4, -2), lowest]
    message = invalid(rows)
    assert "A negative subject or impression in 3 of 18 rows" in message
    for row in rows[-3:]:
        assert row_name(row) in message
    assert "0 or more" in message and BUILDER in message


def test_two_rows_with_the_same_numbers_are_refused():
    rows = ROWS + [sample_row(3, 1, image_path="other.png"), sample_row(17, 0, image_path="b.png")]
    message = invalid(rows)
    assert "The same subject and impression in 4 of 17 rows" in message
    for row in (sample_row(3, 1), rows[-2], sample_row(17, 0), rows[-1]):
        assert row_name(row) in message
    assert "a pair of numbers of its own" in message and BUILDER in message


def test_two_rows_with_the_same_image_path_are_refused():
    """One file under two pairs of numbers, far apart in the table."""
    rows = [sample_row(number // 4, number % 4) for number in range(2000)]
    validate_samples(samples_from_rows(rows))
    rows[1900] = dataclasses.replace(rows[1900], image_path=rows[3].image_path)
    rows[77] = dataclasses.replace(rows[77], image_path=rows[1500].image_path)
    rows[78] = dataclasses.replace(rows[78], image_path=rows[1500].image_path)
    message = invalid(rows)
    assert "The same image_path in 5 of 2,000 rows" in message
    for position in (3, 77, 78, 1500, 1900):
        assert row_name(rows[position]) in message
    assert " more" not in message  # five rows are named, and no rest is counted
    assert "a file of its own" in message and BUILDER in message


@pytest.mark.parametrize("count", [2, 7])
def test_a_table_in_which_every_row_has_the_same_image_path_is_refused(count):
    """The smallest case is what a list of files gives that names one file under two pairs."""
    rows = [sample_row(number, 0, image_path="sensor_a/1_1.png") for number in range(count)]
    message = invalid(rows)
    assert f"The same image_path in {count} of {count} rows" in message
    assert row_name(rows[0]) in message and row_name(rows[1]) in message


def test_more_equal_paths_than_are_compared_in_one_piece_are_all_found():
    """Every path twice, and more bytes in these pairs than one block of the comparison holds."""
    rows = [
        sample_row(number, 0, image_path=f"{'directory/' * 70}{number // 2:05d}.png")
        for number in range(3200)
    ]
    assert 1600 * len(rows[0].image_path) > sample_table._BLOCK_BYTES  # control
    assert "The same image_path in 3,200 of 3,200 rows" in invalid(rows)
    # The last pair is behind the first block: its rows are not counted any more once
    # they have paths of their own
    rows[-2] = sample_row(3198, 0, image_path="another.png")
    assert "The same image_path in 3,198 of 3,200 rows" in invalid(rows)
    validate_samples(samples_from_rows(rows[::2]))  # one row of every pair: a valid table


def thue_morse(length: int) -> str:
    """The Thue-Morse sequence as a text of the letters a and b."""
    return "".join("ab"[bin(position).count("1") % 2] for position in range(length))


def test_paths_are_compared_byte_by_byte_where_their_hashes_are_equal():
    """
    The first 2048 letters of the Thue-Morse sequence and the same text with
    the two letters exchanged have the same sum of (letter) * base**position
    modulo 2**64 for every odd base: the difference of the two sums is the
    product of (1 - base**(2**k)) for k = 0 .. 10, which 2**66 divides. So
    these two different paths have the same hash, whatever the base is.
    """
    text = thue_morse(2048)
    other = text.translate(str.maketrans("ab", "ba"))
    assert text != other
    rows = [sample_row(0, 0, image_path=text), sample_row(1, 0, image_path=other)]
    rows += [sample_row(2 + number, 0) for number in range(50)]
    # Control: the two paths do have the same hash in this module
    hashes = sample_table._path_hashes(samples_from_rows(rows))
    assert hashes[0] == hashes[1] and len(set(hashes.tolist())) == len(rows) - 1

    validate_samples(samples_from_rows(rows))  # different paths are not taken for the same

    # A row with the same path as the first, and two rows that share the other path
    message = invalid(rows + [sample_row(100, 0, image_path=text)])
    assert "The same image_path in 2 of 53 rows" in message
    assert "(subject 0, impression 0)" in message and "(subject 100, impression 0)" in message
    doubled = [sample_row(100, 0, image_path=other), sample_row(101, 0, image_path=other)]
    message = invalid(rows[:1] + rows[2:] + doubled)
    assert "The same image_path in 2 of 53 rows" in message
    assert "(subject 100, impression 0)" in message and "(subject 101, impression 0)" in message


def test_paths_of_different_lengths_are_different_where_their_hashes_are_equal():
    """
    With the signs s of the Thue-Morse sequence, the polynomial
    2 * (x + 1) * sum(s[j] * x**j, j < 2048) is 0 modulo 2**64 at every odd x.
    Its coefficients are the differences, position by position, between a
    text of 2048 letters and the text of 2048 times "e" and one more
    character, so these two paths of different lengths have the same hash.
    """
    signs = [1 - 2 * (bin(position).count("1") % 2) for position in range(2048)]
    steps = [2 * signs[0]] + [2 * (signs[j] + signs[j - 1]) for j in range(1, 2048)]
    shorter = "".join(chr(ord("e") + step) for step in steps)
    longer = "e" * 2048 + "\x01"
    assert set(shorter) == set("aegi") and len(longer) == len(shorter) + 1
    # The longer path first and the shorter one as the last bytes of the table
    rows = [sample_row(0, 0, image_path=longer)]
    rows += [sample_row(1 + number, 0) for number in range(50)]
    rows += [sample_row(100, 0, image_path=shorter)]
    samples = samples_from_rows(rows)
    hashes = sample_table._path_hashes(samples)
    assert hashes[0] == hashes[-1] and len(set(hashes.tolist())) == len(rows) - 1  # control
    assert samples.image_path(len(rows) - 1) == shorter
    validate_samples(samples)
    assert "The same image_path in 2 of 53" in invalid(
        rows + [sample_row(101, 0, image_path=longer)]
    )


@pytest.mark.parametrize(
    "path, fault",
    [
        ("", "it is empty"),
        ("/data/1_1.png", "it starts with /"),
        ("a\\1_1.png", "it holds a backslash"),
        ("../1_1.png", "it has the component '..'"),
        ("a/../../1_1.png", "it has the component '..'"),
        ("a/..", "it has the component '..'"),
        ("./1_1.png", "it has the component '.'"),
        ("a/./1_1.png", "it has the component '.'"),
        (".", "it has the component '.'"),
        ("a//1_1.png", "it has an empty component"),
        ("a/", "it has an empty component"),
        ("a/1_1.png\0.txt", "it holds a NUL character"),
    ],
)
def test_an_image_path_that_is_not_a_normalised_relative_path_is_refused(path, fault):
    rows = ROWS + [sample_row(40, 0, image_path=path)]
    message = invalid(rows)
    assert "An image_path that is not a normalised relative path in 1 of 16 rows" in message
    assert f"{path!r} (subject 40, impression 0): {fault}" in message
    assert "normalize_image_path" in message and BUILDER in message


def test_paths_that_look_like_faults_but_are_none():
    names = [
        "...",
        "..a",
        "a..",
        ".a",
        "a.",
        "a/.b/c..",
        "a b/c",
        "ä/ö.png",
        "..png",
        "a/...",
        ". /a",
    ]
    validate_samples(
        samples_from_rows([sample_row(n, 0, image_path=p) for n, p in enumerate(names)])
    )


def _is_allowed(path: str) -> bool:
    """The rule for the text of an image path, said with normalize_image_path."""
    return (
        path != ""
        and "\\" not in path
        and not path.startswith("/")
        and normalize_image_path(path) == path
        # normpath leaves "." and a ".." at the start as they are
        and path != "."
        and ".." not in path.split("/")
    )


def test_the_path_rules_are_the_normalised_form_for_every_short_path():
    """All texts of up to five characters over an alphabet that holds everything the rules name."""
    alphabet = ("a", ".", "/", "\\", "ä")
    paths = [""]
    for _ in range(5):
        paths = paths + [path + letter for path in paths for letter in alphabet]
    paths = sorted(set(paths))
    assert len(paths) == sum(5**length for length in range(6))
    allowed = [path for path in paths if _is_allowed(path)]
    refused = [path for path in paths if not _is_allowed(path)]
    assert len(allowed) > 500 and len(refused) > 2000

    # All paths in one table, so that every path has neighbours
    rows = [sample_row(number, 0, image_path=path) for number, path in enumerate(paths)]
    message = invalid(rows)
    assert f"in {len(refused):,} of {len(paths):,} rows" in message
    named = [row_name(row) for row in rows if not _is_allowed(row.image_path)][:5]
    position = 0
    for name in named:
        position = message.index(name, position)
    validate_samples(samples_from_rows([row for row in rows if _is_allowed(row.image_path)]))
    # And every refused path alone
    for number, path in enumerate(refused):
        assert "not a normalised relative path in 1 of 1 rows" in invalid(
            [sample_row(number, 0, image_path=path)]
        ), path


def _with_paths(samples: Samples, paths: list) -> Samples:
    """The table with other bytes as its image paths."""
    offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    np.cumsum([len(path) for path in paths], out=offsets[1:])
    return dataclasses.replace(
        samples, path_data=np.frombuffer(b"".join(paths), dtype=np.uint8), path_offsets=offsets
    )


def test_an_image_path_that_is_not_utf8_is_refused(tmp_path):
    samples = samples_from_rows([sample_row(number, 0) for number in range(4)])
    validate_samples(_with_paths(samples, [b"a.png", b"b\xc3\xa4.png", b"c.png", b"d.png"]))
    # The second path ends inside of a character and the third begins with the
    # rest of it: together the bytes are valid UTF-8, each path alone is not
    paths = [b"a.png", b"b\xc3", b"\xa4.png", b"d.png"]
    assert b"".join(paths).decode("utf-8") == "a.pngbä.pngd.png"
    message = invalid(_with_paths(samples, paths))
    assert "not a normalised relative path in 2 of 4 rows" in message
    assert "'b\\\\xc3' (subject 1, impression 0): it is not valid UTF-8" in message
    assert "'\\\\xa4.png' (subject 2, impression 0): it is not valid UTF-8" in message
    # And a byte that is valid nowhere
    message = invalid(_with_paths(samples, paths[:3] + [b"\xff.png"]))
    assert "not a normalised relative path in 3 of 4 rows" in message
    assert message.count("it is not valid UTF-8") == 3

    # Such bytes in a file: a Parquet reader does not check them
    (path,) = write_table(tmp_path / "table", ROWS, parts=1)

    def not_utf8(table):
        texts = [text.encode() for text in table.column("image_path").to_pylist()]
        texts[4] = b"sensor/\xe4.png"  # Latin-1
        offsets = np.zeros(len(texts) + 1, dtype=np.int32)
        np.cumsum([len(text) for text in texts], out=offsets[1:])
        return pa.StringArray.from_buffers(
            len(texts), pa.py_buffer(offsets), pa.py_buffer(b"".join(texts))
        )

    change_part(path, with_column("image_path", not_utf8))
    message = refusal(tmp_path / "table")
    assert "not a normalised relative path in 1 of 15 rows" in message
    assert "it is not valid UTF-8" in message and "\\\\xe4" in message


@pytest.mark.parametrize("ppi", [0.0, -500.0, float("nan"), float("inf")])
def test_a_ppi_that_is_not_a_positive_number_is_refused(ppi):
    rows = ROWS + [sample_row(40, 1, ppi=ppi)]
    message = invalid(rows)
    assert "A ppi that is not a positive number in 1 of 16 rows" in message
    assert f"(subject 40, impression 1): ppi {ppi!r}" in message
    assert BUILDER in message


def test_a_negative_file_size_is_refused():
    validate_samples(samples_from_rows([sample_row(1, 0, file_size=0)]))
    message = invalid(ROWS + [sample_row(40, 1, file_size=-1)])
    assert "A negative file_size in 1 of 16 rows" in message
    assert "(subject 40, impression 1): file_size -1" in message and BUILDER in message


@pytest.mark.parametrize("column", ["source_width", "source_height", "width", "height"])
@pytest.mark.parametrize("value", [0, -3])
def test_a_size_that_is_not_above_zero_is_refused(column, value):
    row = sample_row(40, 1, **{column: value})
    message = invalid(ROWS + [row])
    assert "A size that is not above 0 in 1 of 16 rows" in message
    assert (
        f"(subject 40, impression 1): source {row.source_width}x{row.source_height}, "
        f"prepared {row.width}x{row.height}" in message
    )
    assert BUILDER in message


def test_the_prepared_size_must_follow_from_the_source_size_and_the_ppi():
    good = sample_row(1, 0)  # 1000 ppi: half the size
    assert (good.source_width, good.source_height, good.ppi) == (307, 400, 1000.0)
    assert (good.width, good.height) == (154, 200)
    rows = [
        dataclasses.replace(good, width=153),
        sample_row(1, 1, height=good.height + 1),
        sample_row(1, 2),
        # Not resampled although the file has 1000 ppi
        sample_row(
            5, 0, width=sample_row(5, 0).source_width, height=sample_row(5, 0).source_height
        ),
    ]
    message = invalid(rows)
    assert "does not follow from the source size and the ppi in 3 of 4 rows" in message
    assert (
        f"{row_name(rows[0])}: source 307x400 at 1000.0 ppi gives 154x200 at 500 ppi, the row "
        "holds 153x200" in message
    )
    assert row_name(rows[1]) in message and row_name(rows[3]) in message
    assert row_name(rows[2]) not in message
    assert "resampled_size" in message and BUILDER in message


def test_a_ppi_so_small_that_no_size_follows_from_it_is_refused():
    """The smallest positive number: the size of the prepared image is beyond all numbers."""
    row = sample_row(40, 1, ppi=5e-324)
    message = invalid(ROWS + [row])
    assert "does not follow from the source size and the ppi in 1 of 16 rows" in message
    assert f"{row_name(row)}: source 580x403 at 5e-324 ppi gives infxinf at 500 ppi" in message


def test_the_size_rule_is_the_arithmetic_of_resampled_size():
    """The rule for all rows at once against the function, value by value."""
    rng = np.random.default_rng(7)
    cases = [(785, 785, 251.2), (157, 1413, 251.2), (1, 1, 2000.0), (1, 3, 1500.0), (3, 5, 1000.0)]
    cases += [(2**31 - 1, 2**31 - 1, 500.0), (2**31 - 1, 1, 2100.0), (2, 6, 2000.0)]
    for ppi in (1000.0, 2000.0, 251.2, 385.0, 508.0, 499.9998, 96.0, 72.0):
        cases += [(int(w), int(h), ppi) for w, h in rng.integers(1, 6000, size=(1500, 2))]
    cases += [
        (int(w), int(h), float(ppi))
        for w, h, ppi in zip(
            rng.integers(1, 6000, size=8000),
            rng.integers(1, 6000, size=8000),
            rng.uniform(100.0, 2100.0, size=8000),
        )
    ]
    # Control: sides that land exactly between two whole numbers are among them
    halves = [w for w, _, ppi in cases if (w * (500.0 / ppi)) % 1 == 0.5]
    assert len(halves) > 1000 and 785 in halves

    rows = []
    for number, (source_width, source_height, ppi) in enumerate(cases):
        width, height = resampled_size(source_width, source_height, ppi)
        rows.append(
            sample_row(
                number,
                0,
                source_width=source_width,
                source_height=source_height,
                ppi=ppi,
                width=width,
                height=height,
                minutiae=None,
            )
        )
    validate_samples(samples_from_rows(rows))
    # One pixel more or less is refused in every row, so the rule is not passed by chance
    for column, step in (("width", 1), ("height", 1), ("width", -1), ("height", -1)):
        wrong = [
            dataclasses.replace(row, **{column: getattr(row, column) + step})
            for row in rows
            if 0 < getattr(row, column) + step < 2**31
        ]
        assert len(wrong) > 0.99 * len(rows)
        assert f"in {len(wrong):,} of {len(wrong):,} rows" in invalid(wrong)


def test_an_origin_that_is_none_of_the_values_is_refused():
    samples = samples_from_rows(ROWS)
    origin = samples.ppi_origin.copy()
    origin[4] = len(PPI_ORIGINS)
    message = invalid(dataclasses.replace(samples, ppi_origin=origin))
    assert (
        "A ppi_origin or ridges_origin that is none of the values of its column in 1 of 15"
        in message
    )
    assert f"{samples.image_path(4)!r}" in message and "ppi_origin 4" in message
    origin = samples.ridges_origin.copy()
    origin[0] = 200
    assert "ridges_origin 200" in invalid(dataclasses.replace(samples, ridges_origin=origin))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("column", [0, 1, 2])
def test_minutiae_must_be_finite(value, column):
    minutiae = np.array([[10.0, 20.0, 0.5], [30.0, 40.0, 1.5], [50.0, 60.0, 2.5]], dtype=np.float32)
    minutiae[1, column] = value
    rows = ROWS + [sample_row(40, 1, minutiae=minutiae), sample_row(41, 0, minutiae=minutiae[:1])]
    message = invalid(rows)
    assert "Minutiae with an x, y or angle that is not a finite number in 1 of 17 rows" in message
    assert row_name(rows[-2]) in message and row_name(rows[-1]) not in message
    assert BUILDER in message


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("column", [0, 1, 2])
def test_the_first_minutia_of_a_row_must_be_finite_too(value, column):
    """
    The first minutia of a row is where the minutiae of the row before it
    end, and the first one of the table is where the search begins: the
    places at which the row of a minutia is most easily taken wrongly.
    """
    good = np.array([[10.0, 20.0, 0.5], [30.0, 40.0, 1.5]], dtype=np.float32)
    bad = good.copy()
    bad[0, column] = value
    # The first minutia of the whole table
    rows = [sample_row(0, 0, minutiae=bad), sample_row(1, 0, minutiae=good)]
    message = invalid(rows)
    assert "not a finite number in 1 of 2 rows" in message
    assert row_name(rows[0]) in message and row_name(rows[1]) not in message
    # The first minutia of a later row, behind a row with minutiae, a row
    # whose minutiae are not known and a row that has none
    rows = [
        sample_row(0, 0, minutiae=good),
        sample_row(1, 0, minutiae=None),
        sample_row(2, 0, minutiae=np.zeros((0, 3), dtype=np.float32)),
        sample_row(3, 0, minutiae=bad),
        sample_row(4, 0, minutiae=good),
    ]
    message = invalid(rows)
    assert "not a finite number in 1 of 5 rows" in message
    assert row_name(rows[3]) in message
    assert not any(row_name(row) in message for row in rows[:3] + rows[4:])


def test_the_rules_are_checked_in_every_block_of_a_large_table(large_samples):
    """A fault in a late row, behind the first blocks of path bytes and of minutiae."""
    samples = large_samples
    validate_samples(samples)
    offsets = samples.path_offsets
    late = 49_000
    assert offsets[late] > sample_table._BLOCK_BYTES
    assert 3 * samples.minutiae_offsets[late] > sample_table._BLOCK_VALUES
    name = f"{samples.image_path(late)!r} (subject {samples.subject[late]}, impression"

    paths = samples.path_data.copy()
    paths[offsets[late] + 8] = ord("\\")
    message = invalid(dataclasses.replace(samples, path_data=paths))
    assert "not a normalised relative path in 1 of 50,000 rows" in message
    assert name.replace("/", "\\\\", 1) in message and "it holds a backslash" in message

    # The path of an early row once more: all paths of this table have one length
    early = 10
    assert offsets[early + 1] - offsets[early] == offsets[late + 1] - offsets[late]
    paths = samples.path_data.copy()
    paths[offsets[late] : offsets[late + 1]] = paths[offsets[early] : offsets[early + 1]]
    message = invalid(dataclasses.replace(samples, path_data=paths))
    assert "The same image_path in 2 of 50,000 rows" in message
    for row in (early, late):
        assert f"(subject {samples.subject[row]}, impression {samples.impression[row]})" in message

    assert samples.minutiae_known[late] and len(samples.minutiae_of(late)) > 0
    minutiae = samples.minutiae.copy()
    minutiae[samples.minutiae_offsets[late + 1] - 1, 2] = np.nan
    message = invalid(dataclasses.replace(samples, minutiae=minutiae))
    assert "not a finite number in 1 of 50,000 rows" in message and name in message


def test_a_path_that_is_longer_than_a_block_is_checked_like_any_other():
    long = "a" * (sample_table._BLOCK_BYTES + 5000)
    rows = ROWS + [sample_row(40, 0, image_path=long + "/1.png"), sample_row(41, 0)]
    validate_samples(samples_from_rows(rows))
    message = invalid(rows + [sample_row(42, 0, image_path=long + "/1.png")])
    assert "The same image_path in 2 of 18 rows" in message
    assert "(subject 40, impression 0)" in message and "(subject 42, impression 0)" in message
    message = invalid(rows + [sample_row(42, 0, image_path=long + "/")])
    assert "not a normalised relative path in 1 of 18 rows" in message
    assert "(subject 42, impression 0): it has an empty component" in message
    # One byte differs at the very end
    validate_samples(samples_from_rows(rows + [sample_row(42, 0, image_path=long + "/2.png")]))


def test_a_target_resolution_that_is_no_resolution_is_refused():
    for target_ppi in (0.0, -500.0, float("nan"), float("inf")):
        message = invalid(samples_from_rows(ROWS, target_ppi=target_ppi))
        assert "target resolution" in message and repr(target_ppi) in message


def test_a_message_names_the_first_rows_and_counts_the_rest():
    rows = [sample_row(number, 0, file_size=-1 - number) for number in range(1234)]
    message = invalid(rows)
    assert "A negative file_size in 1,234 of 1,234 rows" in message
    for row in rows[:5]:
        assert f"{row_name(row)}: file_size {row.file_size}" in message
    assert row_name(rows[5]) not in message and "and 1,229 more" in message


# ---------------------------------------------------------------------------
# The checksum
# ---------------------------------------------------------------------------


def expected_checksum(rows, target_ppi: float = 500.0) -> str:
    """The checksum as the module docstring defines it, with struct and hashlib alone."""
    rows = sorted(rows, key=lambda row: (row.subject, row.impression))
    data = b"flx-samples-1\n"
    data += struct.pack("<q", len(rows)) + struct.pack("<d", target_ppi)
    data += b"".join(struct.pack("<i", row.subject) for row in rows)
    data += b"".join(struct.pack("<i", row.impression) for row in rows)
    data += b"".join(struct.pack("<q", row.file_size) for row in rows)
    data += b"".join(struct.pack("<d", row.ppi) for row in rows)
    data += bytes({"dark": 0, "light": 1}[row.ridges] for row in rows)
    for column in ("source_width", "source_height", "width", "height"):
        data += b"".join(struct.pack("<i", getattr(row, column)) for row in rows)
    paths = [row.image_path.encode("utf-8") for row in rows]
    data += b"".join(struct.pack("<i", len(path)) for path in paths)
    data += b"".join(paths)
    data += bytes(0 if row.minutiae is None else 1 for row in rows)
    data += b"".join(
        struct.pack("<i", 0 if row.minutiae is None else len(row.minutiae)) for row in rows
    )
    for row in rows:
        for x, y, angle in [] if row.minutiae is None else row.minutiae.tolist():
            data += struct.pack("<f", x) + struct.pack("<f", y) + struct.pack("<f", angle)
    return hashlib.sha256(data).hexdigest()


def test_the_checksum_is_the_sha256_of_the_values_as_the_docstring_defines_them(tmp_path):
    assert samples_checksum(samples_from_rows(ROWS)) == expected_checksum(ROWS)
    write_table(tmp_path / "table", ROWS)
    assert samples_checksum(read_samples(str(tmp_path / "table"))) == expected_checksum(ROWS)
    assert samples_checksum(samples_from_rows([])) == expected_checksum([])
    rows = many_rows(300)
    assert samples_checksum(samples_from_rows(rows)) == expected_checksum(rows)
    assert samples_checksum(samples_from_rows(ROWS, target_ppi=1000.0)) == expected_checksum(
        ROWS, 1000.0
    )


def test_the_checksum_of_a_fixed_table_stays_what_it_is():
    """
    Stored trainings name their table by this value, so the definition cannot
    change: a change must show here, also if the same change is made to the
    function above.
    """
    rows = [
        SampleRow(
            subject=16,
            impression=2,
            image_path="sensor_a/17_3.png",
            file_size=51_234,
            ppi=1000.0,
            ppi_origin="manifest",
            ridges="light",
            ridges_origin="argument",
            source_width=826,
            source_height=1274,
            width=413,
            height=637,
            minutiae=np.array([[175.0, 320.5, 1.25], [12.0625, 600.0, -0.5]], dtype=np.float32),
        ),
        SampleRow(
            subject=3,
            impression=0,
            image_path="größe/4_1.tif",
            file_size=7,
            ppi=251.2,
            ppi_origin="metadata",
            ridges="dark",
            ridges_origin="default",
            source_width=785,
            source_height=157,
            width=1562,
            height=312,
            minutiae=None,
        ),
        SampleRow(
            subject=3,
            impression=1,
            image_path="4_2.png",
            file_size=0,
            ppi=500.0,
            ppi_origin="assumed",
            ridges="dark",
            ridges_origin="estimated",
            source_width=1,
            source_height=1,
            width=1,
            height=1,
            minutiae=np.zeros((0, 3), dtype=np.float32),
        ),
    ]
    validate_samples(samples_from_rows(rows))
    assert expected_checksum(rows) == FIXED_CHECKSUM
    assert samples_checksum(samples_from_rows(rows)) == FIXED_CHECKSUM


FIXED_CHECKSUM = "2a5c71f3c766d0950ca6a71cdf819f8267fc33e21ede30a43d6cf665d6a502d3"


def test_the_checksum_of_a_second_fixed_table_stays_what_it_is():
    """
    In the table above only the last row has minutiae, so the number of
    minutiae of every row and the position at which they end are the same
    numbers there, and a checksum over the one could not be told from a
    checksum over the other. Here minutiae are in three rows.
    """
    values = np.arange(18, dtype=np.float32).reshape(6, 3) / 4
    common = dict(source_width=4, source_height=5, width=4, height=5, ppi=500.0)
    origins = dict(ppi_origin="argument", ridges_origin="argument")
    rows = [
        SampleRow(
            subject=7,
            impression=3,
            image_path="b/8_4.png",
            file_size=14,
            ridges="dark",
            minutiae=values[3:],
            **common,
            **origins,
        ),
        SampleRow(
            subject=1,
            impression=1,
            image_path="a/2_2.png",
            file_size=12,
            ridges="light",
            minutiae=None,
            **common,
            **origins,
        ),
        SampleRow(
            subject=7,
            impression=0,
            image_path="b/8_1.png",
            file_size=13,
            ppi=1000.0,
            ridges="dark",
            source_width=8,
            source_height=6,
            width=4,
            height=3,
            minutiae=values[2:3],
            **origins,
        ),
        SampleRow(
            subject=1,
            impression=0,
            image_path="a/2_1.png",
            file_size=11,
            ridges="dark",
            minutiae=values[:2],
            **common,
            **origins,
        ),
    ]
    samples = samples_from_rows(rows)
    validate_samples(samples)
    # Control: the numbers of minutiae of the rows are not the positions at which they end
    assert np.diff(samples.minutiae_offsets).tolist() == [2, 0, 1, 3]
    assert samples.minutiae_offsets[1:].tolist() == [2, 2, 3, 6]
    assert expected_checksum(rows) == SECOND_FIXED_CHECKSUM
    assert samples_checksum(samples) == SECOND_FIXED_CHECKSUM


SECOND_FIXED_CHECKSUM = "20e47e516426421a44b8e18bad207941662d9875167d4a20de051a92d5d925ee"


def test_the_checksum_does_not_depend_on_parts_and_the_order_of_rows(tmp_path):
    checksum = expected_checksum(ROWS)
    for parts in (1, 2, 5):
        directory = tmp_path / f"table-{parts}"
        paths = write_table(directory, ROWS, parts=parts, build={"id": f"build of {parts}"})
        assert samples_checksum(read_samples(str(directory))) == checksum
    # The last table again with the rows of every file turned round, in small row groups
    for path in paths:
        change_part(
            path, lambda table: table.take(list(range(table.num_rows))[::-1]), row_group_size=2
        )
    assert samples_checksum(read_samples(str(directory))) == checksum
    assert samples_checksum(samples_from_rows(ROWS[::-1])) == checksum


def _other_minutiae_value(row: SampleRow, column: int) -> SampleRow:
    minutiae = row.minutiae.copy()
    minutiae[-1, column] += 0.5
    return dataclasses.replace(row, minutiae=minutiae)


FILLED = next(
    position for position, row in enumerate(ROWS) if row.minutiae is not None and len(row.minutiae)
)
EMPTY = next(
    position
    for position, row in enumerate(ROWS)
    if row.minutiae is not None and not len(row.minutiae)
)
UNKNOWN = next(position for position, row in enumerate(ROWS) if row.minutiae is None)


@pytest.mark.parametrize(
    "position, change",
    [
        (0, lambda row: dataclasses.replace(row, subject=row.subject + 1)),
        (0, lambda row: dataclasses.replace(row, impression=row.impression + 5)),
        (0, lambda row: dataclasses.replace(row, image_path=row.image_path + "g")),
        (0, lambda row: dataclasses.replace(row, image_path=row.image_path.upper())),
        (0, lambda row: dataclasses.replace(row, file_size=row.file_size + 1)),
        (0, lambda row: dataclasses.replace(row, ppi=np.nextafter(row.ppi, 0.0))),
        (
            0,
            lambda row: dataclasses.replace(
                row, ridges="light" if row.ridges == "dark" else "dark"
            ),
        ),
        (0, lambda row: dataclasses.replace(row, source_width=row.source_width + 1)),
        (0, lambda row: dataclasses.replace(row, source_height=row.source_height + 1)),
        (0, lambda row: dataclasses.replace(row, width=row.width + 1)),
        (0, lambda row: dataclasses.replace(row, height=row.height + 1)),
        (FILLED, lambda row: _other_minutiae_value(row, 0)),
        (FILLED, lambda row: _other_minutiae_value(row, 1)),
        (FILLED, lambda row: _other_minutiae_value(row, 2)),
        (FILLED, lambda row: dataclasses.replace(row, minutiae=row.minutiae[:-1])),
        (FILLED, lambda row: dataclasses.replace(row, minutiae=row.minutiae[::-1])),
        (FILLED, lambda row: dataclasses.replace(row, minutiae=None)),
        (EMPTY, lambda row: dataclasses.replace(row, minutiae=None)),
        (UNKNOWN, lambda row: dataclasses.replace(row, minutiae=np.zeros((0, 3), np.float32))),
    ],
)
def test_the_checksum_changes_with_every_value_it_includes(position, change):
    rows = list(ROWS)
    rows[position] = change(rows[position])
    checksum = samples_checksum(samples_from_rows(rows))
    assert checksum != samples_checksum(samples_from_rows(ROWS))
    assert checksum == expected_checksum(rows)


def test_the_checksum_changes_with_the_rows_and_the_target_resolution():
    checksum = samples_checksum(samples_from_rows(ROWS))
    assert samples_checksum(samples_from_rows(ROWS[:-1])) != checksum
    assert samples_checksum(samples_from_rows(ROWS + [sample_row(50, 0)])) != checksum
    assert samples_checksum(samples_from_rows(ROWS, target_ppi=1000.0)) != checksum
    # Minutiae that move from one row to the next: the same values in the same order
    first, second = sample_row(1, 0, minutiae=None), sample_row(1, 1, minutiae=None)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    split_early = [
        dataclasses.replace(first, minutiae=values[:1]),
        dataclasses.replace(second, minutiae=values[1:]),
    ]
    split_late = [
        dataclasses.replace(first, minutiae=values[:3]),
        dataclasses.replace(second, minutiae=values[3:]),
    ]
    assert samples_checksum(samples_from_rows(split_early)) != samples_checksum(
        samples_from_rows(split_late)
    )


def test_the_checksum_leaves_out_the_origins_and_the_build(tmp_path):
    checksum = samples_checksum(samples_from_rows(ROWS))
    other_origins = [
        dataclasses.replace(
            row,
            ppi_origin=PPI_ORIGINS[(PPI_ORIGINS.index(row.ppi_origin) + 1) % 4],
            ridges_origin=RIDGES_ORIGINS[(RIDGES_ORIGINS.index(row.ridges_origin) + 2) % 4],
        )
        for row in ROWS
    ]
    assert samples_checksum(samples_from_rows(other_origins)) == checksum
    assert samples_checksum(samples_from_rows(ROWS, build={"id": "x", "files": 3})) == checksum
    for build in ({"id": "one"}, {"id": "two", "files": 99, "settings": {"ppi": 500.0}}):
        directory = tmp_path / build["id"]
        write_table(directory, ROWS, build=build)
        samples = read_samples(str(directory))
        assert samples.build["id"] == build["id"]
        assert samples_checksum(samples) == checksum


# ---------------------------------------------------------------------------
# Writing a part
# ---------------------------------------------------------------------------


def test_part_names_have_five_digits_for_index_and_count():
    assert part_name(0, 3) == "part-00000-of-00003.parquet"
    assert part_name(12, 345) == "part-00012-of-00345.parquet"


def test_part_names_grow_beyond_five_digits(tmp_path):
    path = write_part(str(tmp_path / "table"), 0, 1, samples_from_rows(ROWS), {"id": "a"})
    # As if it were one of very many parts: the name gets longer
    change_part(path, build={"part": 123_456, "parts": 123_457})
    os.rename(path, tmp_path / "table" / part_name(123_456, 123_457))
    message = refusal(tmp_path / "table")
    assert "it holds 1 of its 123,457 parts; missing: part-00000-of-123457.parquet" in message
    assert "part-00004-of-123457.parquet and 123,451 more. " in message

    # A number of parts that no directory holds is not counted through
    path = write_part(str(tmp_path / "huge"), 0, 1, samples_from_rows(ROWS), {"id": "a"})
    change_part(path, build={"parts": 10**15})
    os.rename(path, tmp_path / "huge" / part_name(0, 10**15))
    message = refusal(tmp_path / "huge")
    assert "it holds 1 of its 1,000,000,000,000,000 parts" in message
    assert "missing: part-00001-of-1000000000000000.parquet" in message


@pytest.mark.parametrize("failing", ["while the file is written", "when the file is renamed"])
def test_a_write_that_fails_leaves_no_file(tmp_path, monkeypatch, failing):
    directory = tmp_path / "table"
    path = write_part(str(directory), 0, 2, samples_from_rows(ROWS[:8]), {"id": "a"})
    with open(path, "rb") as file:
        complete = file.read()

    def disk_full(table, where, **options):
        with open(where, "wb") as file:
            file.write(b"half a part")
        raise OSError(28, "No space left on device")

    def no_rename(source, target):
        raise OSError(13, "Permission denied")

    if failing == "while the file is written":
        monkeypatch.setattr(pq, "write_table", disk_full)
    else:
        monkeypatch.setattr(os, "replace", no_rename)
    with pytest.raises(OSError):
        write_part(str(directory), 1, 2, samples_from_rows(ROWS[8:]), {"id": "a"})
    # Neither the part nor a temporary file
    assert os.listdir(directory) == [os.path.basename(path)]
    # A part that was there before is still the complete one
    with pytest.raises(OSError):
        write_part(str(directory), 0, 2, samples_from_rows(ROWS[8:]), {"id": "a"})
    assert os.listdir(directory) == [os.path.basename(path)]
    with open(path, "rb") as file:
        assert file.read() == complete


def test_a_part_is_written_under_a_temporary_name_first(tmp_path, monkeypatch):
    seen = []
    write_table = pq.write_table

    def record(table, where, **options):
        seen.append((where, sorted(os.listdir(tmp_path))))
        write_table(table, where, **options)
        seen.append(sorted(os.listdir(tmp_path)))

    monkeypatch.setattr(pq, "write_table", record)
    path = write_part(str(tmp_path), 0, 1, samples_from_rows(ROWS), {"id": "a"})
    (temporary, before), during = seen
    assert before == [] and os.path.dirname(temporary) == str(tmp_path)
    assert temporary != path and during == [os.path.basename(temporary)]
    assert os.listdir(tmp_path) == [os.path.basename(path)]


def test_arguments_of_write_part_that_make_no_sense_are_refused(tmp_path):
    samples = samples_from_rows(ROWS)
    for index, count in ((0, 0), (2, 2), (-1, 2), (3, 2)):
        with pytest.raises(ValueError, match="index from 0 to count - 1"):
            write_part(str(tmp_path), index, count, samples, {"id": "a"})
    for build in ({}, {"id": ""}, {"id": 7}, {"files": 3}, None, "a"):
        with pytest.raises(ValueError, match='"id"'):
            write_part(str(tmp_path), 0, 1, samples, build)
    for build in ({"id": "a", "when": object()}, {"id": "a", "share": float("nan")}):
        with pytest.raises(ValueError, match="plain JSON values"):
            write_part(str(tmp_path), 0, 1, samples, build)
    assert os.listdir(tmp_path) == []


def test_an_existing_part_is_replaced(tmp_path):
    write_part(str(tmp_path), 0, 1, samples_from_rows(ROWS[:3]), {"id": "a"})
    write_part(str(tmp_path), 0, 1, samples_from_rows(ROWS), {"id": "b"})
    samples = read_samples(str(tmp_path))
    assert len(samples) == len(ROWS) and samples.build["id"] == "b"


def test_rows_that_cannot_be_written_are_refused(tmp_path):
    samples = samples_from_rows(ROWS)
    origin = samples.ridges_origin.copy()
    origin[:2] = 9
    with pytest.raises(SampleTableError, match="ridges_origin of 2 rows is not an index"):
        write_part(
            str(tmp_path), 0, 1, dataclasses.replace(samples, ridges_origin=origin), {"id": "a"}
        )
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# Rows as Python values
# ---------------------------------------------------------------------------


def test_rows_are_put_into_the_order_of_subject_and_impression():
    samples = samples_from_rows(ROWS, build={"id": "x"})
    pairs = list(zip(samples.subject.tolist(), samples.impression.tolist()))
    assert pairs == sorted((row.subject, row.impression) for row in ROWS)
    assert samples.target_ppi == TARGET_PPI and samples.build == {"id": "x"}
    assert samples_from_rows(iter(ROWS)).subject.tolist() == samples.subject.tolist()
    assert samples.subject.dtype == np.int32 and samples.file_size.dtype == np.int64
    assert samples.ppi.dtype == np.float64 and samples.minutiae.dtype == np.float32
    assert samples.path_offsets.dtype == np.int64 and samples.minutiae_offsets.dtype == np.int64
    # Minutiae can be given as any array of numbers (k, 3)
    row = sample_row(1, 2, minutiae=[[1, 2, 0.5], [3, 4, 1.5]])
    assert samples_from_rows([row]).minutiae.tolist() == [[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]]


@pytest.mark.parametrize(
    "changes, said",
    [
        (dict(subject=2**31), "subject is 2147483648; the column holds numbers from -2147483648"),
        (dict(impression=-(2**31) - 1), "impression is -2147483649"),
        (dict(file_size=2**63), "file_size is 9223372036854775808"),
        (dict(width=2**31), "width is 2147483648"),
        (dict(subject=1.5), "subject must be a whole number, got 1.5"),
        (dict(height="640"), "height must be a whole number, got '640'"),
        (dict(ppi="high"), "ppi must be a number, got 'high'"),
        (dict(ppi=None), "ppi must be a number, got None"),
        (dict(ppi_origin="guessed"), "ppi_origin is 'guessed'; it must be one of 'argument'"),
        (dict(ridges="auto"), "ridges is 'auto'; it must be one of 'dark', 'light'"),
        (dict(ridges_origin=None), "ridges_origin is None; it must be one of 'argument'"),
        (dict(image_path="caf\udce9.png"), "image_path must be text that can be written as UTF-8"),
        (dict(image_path=b"a.png"), "image_path must be text that can be written as UTF-8"),
        (dict(minutiae=np.zeros((2, 2))), "minutiae must be an array (k, 3)"),
        (dict(minutiae=np.zeros(3)), "minutiae must be an array (k, 3)"),
        (dict(minutiae=np.zeros(0)), "got an array (0,)"),
        (dict(minutiae=[["a", "b", "c"]]), "got no array of numbers"),
    ],
)
def test_a_row_that_the_table_cannot_hold_is_refused_with_its_name(changes, said):
    row = dataclasses.replace(sample_row(6, 1), **changes)
    with pytest.raises(SampleTableError) as error:
        samples_from_rows(ROWS + [row])
    message = str(error.value)
    assert said in message
    assert message.startswith(f"Row {row.image_path!r} (subject {row.subject}, impression")


def test_image_paths_are_normalised_with_a_slash_as_separator():
    assert normalize_image_path("a/1_1.png") == "a/1_1.png"
    assert normalize_image_path("./sensor_a//session/../1_1.png") == "sensor_a/1_1.png"
    assert normalize_image_path("a/") == "a"
    assert normalize_image_path(os.path.join("sensor_a", "1_1.png")) == "sensor_a/1_1.png"
    # Whether a path stays inside the root directory is a rule of the table
    assert normalize_image_path("a/../../1_1.png") == "../1_1.png"
    assert normalize_image_path("") == "."


def test_samples_refuses_arrays_that_are_not_a_table():
    samples = samples_from_rows(ROWS)
    values = {name: getattr(samples, name) for name in ARRAYS}
    Samples(**values)
    with pytest.raises(ValueError, match="Samples.subject must be an array of int32, got int64"):
        Samples(**dict(values, subject=samples.subject.astype(np.int64)))
    with pytest.raises(ValueError, match="Samples.ppi must be an array of float64, got list"):
        Samples(**dict(values, ppi=samples.ppi.tolist()))
    with pytest.raises(ValueError, match="Samples.minutiae must have the shape"):
        Samples(**dict(values, minutiae=samples.minutiae.reshape(-1)))
    with pytest.raises(ValueError, match="Samples.width holds 14 values for 15 rows"):
        Samples(**dict(values, width=samples.width[:-1]))
    with pytest.raises(ValueError, match="Samples.path_offsets"):
        Samples(**dict(values, path_offsets=samples.path_offsets[:-1]))
    with pytest.raises(ValueError, match="Samples.minutiae_offsets must go from 0"):
        Samples(**dict(values, minutiae_offsets=samples.minutiae_offsets + 1))
    with pytest.raises(ValueError, match="whose minutiae are not known"):
        Samples(**dict(values, minutiae_known=np.zeros(len(samples), dtype=bool)))
    with pytest.raises(ValueError, match="order of \\(subject, impression\\)"):
        Samples(**dict(values, subject=samples.subject[::-1].copy()))
    # The arrays that were given stay as they are; the table has views of them
    subject = samples.subject.copy()
    table = Samples(**dict(values, subject=subject))
    assert subject.flags.writeable and not table.subject.flags.writeable


def test_samples_refuses_rows_that_are_out_of_order_in_any_way():
    """Not only where a row has a smaller subject than the first row."""
    samples = samples_from_rows(ROWS)
    values = {name: getattr(samples, name) for name in ARRAYS}
    # Two impressions of the first subject exchanged: the subjects stay in order
    impression = samples.impression.copy()
    assert samples.subject[0] == samples.subject[1]
    impression[[0, 1]] = impression[[1, 0]]
    with pytest.raises(ValueError, match="order of \\(subject, impression\\)"):
        Samples(**dict(values, impression=impression))
    # Two later subjects exchanged: every subject is still larger than the first one
    subject = samples.subject.copy()
    assert subject[0] < subject[3] < subject[6]
    subject[3:6], subject[6:9] = samples.subject[6:9], samples.subject[3:6]
    assert subject[:9].tolist() == [0, 0, 0, 5, 5, 5, 3, 3, 3]
    with pytest.raises(ValueError, match="order of \\(subject, impression\\)"):
        Samples(**dict(values, subject=subject))
