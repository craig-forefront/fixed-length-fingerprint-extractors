"""
Tests for scripts/build_sample_table.py: the script that builds a sample
table from image files and their .mnt files, and checks a table that exists.

The script is run through its main(), and in a few tests as a program, on
small prints that are written here and on .mnt files that are written as
text, as the minutiae extraction writes them. What a build wrote is read back
with read_samples and compared with what SensorNormalizer.load and
read_mnt_file give for the same files.
"""

import dataclasses
import errno
import importlib.util
import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import time
from os.path import dirname, join

import numpy as np
import pytest
from PIL import Image

from flx.data.fingerflow_minutia_loader import read_mnt_file
from flx.preparation import sample_table
from flx.preparation.sample_table import (
    PPI_ORIGINS,
    RIDGES_ORIGINS,
    SampleRow,
    Samples,
    read_samples,
    samples_checksum,
    samples_from_rows,
    write_part,
)
from flx.preparation.sensor_normalization import SensorNormalizer, resampled_size

REPOSITORY = dirname(dirname(__file__))
SCRIPT = join(REPOSITORY, "scripts", "build_sample_table.py")

FAILURES_LOG = "build_failures.txt"
WARNINGS_LOG = "build_warnings.txt"

# The arrays of Samples, by the names of its fields
ARRAYS = [field.name for field in dataclasses.fields(Samples) if field.type is np.ndarray]


def load_script():
    spec = importlib.util.spec_from_file_location("build_sample_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder(monkeypatch):
    """The script as a module. A test that takes it builds or reads a table, which needs pyarrow."""
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script adds a directory
    return load_script()


@pytest.fixture
def builder_without_pyarrow(monkeypatch):
    """
    The script where pyarrow cannot be imported. The name is blocked, so the
    tests that take this fixture run the same way with and without pyarrow
    installed, and are never skipped.
    """
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    return load_script()


def run(script, monkeypatch, *arguments) -> int:
    """Runs main() with the arguments and returns the exit status."""
    monkeypatch.setattr(sys, "argv", ["build_sample_table.py"] + [str(a) for a in arguments])
    return script.main()


def build(script, monkeypatch, tmp_path, *options, table="table") -> int:
    """Builds tmp_path/<table> from the images in tmp_path/images."""
    return run(
        script,
        monkeypatch,
        *("--dataset_dir", tmp_path / "images", "--samples", tmp_path / table),
        *options,
    )


def refused(script, monkeypatch, capsys, tmp_path, *options) -> str:
    """The message with which a build of tmp_path/table is refused (exit status 2)."""
    capsys.readouterr()
    assert build(script, monkeypatch, tmp_path, *options) == 2
    message = capsys.readouterr().err
    assert message.startswith("ERROR: ") or "\nERROR: " in message
    return " ".join(message.split())


def usage_error(script, monkeypatch, capsys, *arguments) -> str:
    """The message of an option error: argparse leaves with the exit status 2."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as leaving:
        run(script, monkeypatch, *arguments)
    assert leaving.value.code == 2
    return " ".join(capsys.readouterr().err.split())


def printed(capsys) -> list:
    """The lines that were printed, each with single spaces."""
    return [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]


# ---------------------------------------------------------------------------
# Prints and .mnt files
# ---------------------------------------------------------------------------


def ridge_image(height: int = 160, width: int = 128, period: float = 9.5) -> np.ndarray:
    """Dark ridges on a white background: a print in the middle of a white frame."""
    y, x = np.mgrid[0:height, 0:width]
    ridges = 127.5 + 127.5 * np.cos(2 * np.pi * (x + 0.3 * y) / period)
    image = np.full((height, width), 255.0)
    inside = (y > height // 6) & (y < 5 * height // 6) & (x > width // 6) & (x < 5 * width // 6)
    image[inside] = ridges[inside]
    return image.astype(np.uint8)


def save(path, image: np.ndarray = None, ppi=500) -> str:
    """Saves a print, with the resolution in the file metadata unless ppi is None."""
    os.makedirs(dirname(str(path)), exist_ok=True)
    image = ridge_image() if image is None else image
    if ppi is None:
        Image.fromarray(image).save(str(path))
    else:
        Image.fromarray(image).save(str(path), dpi=(ppi, ppi))
    return str(path)


# What the extraction is taken to have found in a print: x, y, angle
MINUTIAE = [(10.5, 20.25, 0.5), (100.0, 150.0, 6.0), (63.75, 80.0, 3.141593)]


def mnt_text(info, minutiae=MINUTIAE, comment: bool = True) -> str:
    """A .mnt file as flx/preparation/minutiae_extraction.py writes it for a prepared image."""
    lines = [str(len(minutiae)), f"{info.width} {info.height}"]
    lines += ["%.4f %.4f %.6f %d %.4f" % (x, y, angle, -1, 0.5) for x, y, angle in minutiae]
    if comment:
        lines.append("# " + info.describe())
    return "\n".join(lines) + "\n"


def write_mnt(path, text: str) -> str:
    os.makedirs(dirname(str(path)), exist_ok=True)
    with open(str(path), "w") as file:
        file.write(text)
    return str(path)


def extracted(tmp_path, name: str, normalizer: SensorNormalizer = None, **options) -> str:
    """
    Writes the .mnt file of tmp_path/images/<name> where the extraction
    writes it: at the path of the image below the minutiae directory, with
    the ending .mnt. Returns its path.
    """
    normalizer = SensorNormalizer() if normalizer is None else normalizer
    _, info = normalizer.load(str(tmp_path / "images" / name), check_polarity=True)
    path = tmp_path / "minutiae" / (os.path.splitext(name)[0] + ".mnt")
    return write_mnt(path, mnt_text(info, **options))


def stored_minutiae(mnt_path) -> np.ndarray:
    """The minutiae of a .mnt file as a table is to hold them: what read_mnt_file returns."""
    locations, angles, _ = read_mnt_file(str(mnt_path))
    return np.column_stack([locations.reshape(-1, 2), angles]).astype(np.float32)


def numbers(samples) -> list:
    """(subject, impression) of the rows of a table."""
    return list(zip(samples.subject.tolist(), samples.impression.tolist()))


def paths(samples) -> list:
    return [samples.image_path(row) for row in range(len(samples))]


def row_of(samples, subject: int, impression: int) -> dict:
    """A row of a table as Python values."""
    row = samples.find(subject, impression)
    assert row >= 0, f"the table has no row for subject {subject}, impression {impression}"
    return dict(
        image_path=samples.image_path(row),
        file_size=int(samples.file_size[row]),
        ppi=float(samples.ppi[row]),
        ppi_origin=PPI_ORIGINS[samples.ppi_origin[row]],
        ridges="light" if samples.ridges_light[row] else "dark",
        ridges_origin=RIDGES_ORIGINS[samples.ridges_origin[row]],
        source_size=(int(samples.source_width[row]), int(samples.source_height[row])),
        size=(int(samples.width[row]), int(samples.height[row])),
        minutiae=samples.minutiae_of(row) if samples.minutiae_known[row] else None,
    )


def as_loaded(path, normalizer: SensorNormalizer) -> dict:
    """What a row holds about an image file: the facts of SensorNormalizer.load, and its size."""
    _, info = normalizer.load(str(path), check_polarity=True)
    return dict(
        file_size=os.path.getsize(path),
        ppi=info.ppi,
        ppi_origin=info.ppi_origin,
        ridges=info.ridges,
        ridges_origin=info.ridges_origin,
        source_size=(info.source_width, info.source_height),
        size=(info.width, info.height),
    )


def log(directory, name: str) -> dict:
    """A log of a build as {image path: message}; empty if the log is not there."""
    path = join(str(directory), name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    entries = dict(line.split("\t") for line in lines)  # one tab in every line
    assert len(entries) == len(lines), "a file has two lines in the log"
    return entries


def snapshot(directory, with_itself: bool = True) -> dict:
    """
    Every file and directory below a directory, and the directory itself,
    with its size and the time of its last change. A file that was made and
    removed again shows in the time of its directory.
    """
    found = {}
    for base, directories, files in os.walk(str(directory)):
        for name in directories + files + [""]:
            if name or (with_itself and base == str(directory)):
                path = join(base, name)
                status = os.lstat(path)
                found[os.path.relpath(path, str(directory))] = (status.st_size, status.st_mtime_ns)
    return found


def tables(tmp_path) -> list:
    """What a build left in tmp_path: tables, and temporary directories of tables."""
    return sorted(name for name in os.listdir(tmp_path) if name.startswith("table"))


def stored_parts(directory) -> list:
    """
    Per part of a table, in the order of the parts: (the build information of
    the part, [(subject, impression) of its rows in the order of the file]).
    """
    import pyarrow.parquet as pq

    parts = []
    for name in sorted(os.listdir(str(directory))):
        if name.endswith(".parquet"):
            table = pq.read_table(join(str(directory), name))
            pairs = zip(table.column("subject").to_pylist(), table.column("impression").to_pylist())
            parts.append((json.loads(table.schema.metadata[b"flx.build"]), list(pairs)))
    return parts


def forbid_loading(script, monkeypatch) -> None:
    """A build that loads an image after this fails: what is refused must be refused before."""

    def load_sample(*arguments):
        raise AssertionError(f"an image was loaded: {arguments[1:]}")

    monkeypatch.setattr(script, "load_sample", load_sample)


# ---------------------------------------------------------------------------
# A directory of files named as today
# ---------------------------------------------------------------------------

# Files in the root and in subdirectories, with two endings; where files are
# listed by name, 10_1 comes before 2_1
NAMED = ["1_1.png", "1_2.png", "sensor_a/2_1.png", "sensor_a/10_1.png", "sensor_b/deep/17_3.tif"]


def write_named(tmp_path) -> None:
    for position, name in enumerate(NAMED):
        # Every file has a size of its own, in pixels and in bytes
        save(tmp_path / "images" / name, ridge_image(160 + 16 * position, 128 + 8 * position))
        extracted(tmp_path, name, minutiae=MINUTIAE[: 1 + position % 3])


def test_a_directory_of_files_named_as_today_becomes_a_table(
    builder, monkeypatch, tmp_path, capsys
):
    write_named(tmp_path)
    before = snapshot(tmp_path / "images"), snapshot(tmp_path / "minutiae")
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0

    samples = read_samples(tmp_path / "table")
    # The numbers of the file names minus 1, as the training reads these names
    assert numbers(samples) == [(0, 0), (0, 1), (1, 0), (9, 0), (16, 2)]
    normalizer = SensorNormalizer()
    for name in NAMED:
        subject, impression = os.path.splitext(os.path.basename(name))[0].split("_")
        row = row_of(samples, int(subject) - 1, int(impression) - 1)
        # The path relative to the image directory, with / as separator
        assert row.pop("image_path") == name
        minutiae = row.pop("minutiae")
        assert row == as_loaded(tmp_path / "images" / name, normalizer)
        expected = stored_minutiae(tmp_path / "minutiae" / (os.path.splitext(name)[0] + ".mnt"))
        assert minutiae.dtype == np.float32 and len(minutiae) == len(expected) > 0
        assert minutiae.tobytes() == expected.tobytes()
    # Control: the files differ in what the rows hold, so no row passes for another
    assert len(set(samples.file_size.tolist())) == len(set(samples.width.tolist())) == 5

    assert samples.build["files"] == 5 and samples.build["failed"] == 0
    assert samples.build["parts"] == 1 and samples.build["rows"] == 5
    assert samples.build["settings"] == normalizer.settings()
    assert samples.build["with_minutiae"] is True
    assert len(samples.build["id"]) == 64 and int(samples.build["id"], 16) >= 0

    # Only the part: no log without lines, no temporary directory, nothing in the other directories
    assert os.listdir(tmp_path / "table") == ["part-00000-of-00001.parquet"]
    assert sorted(os.listdir(tmp_path)) == ["images", "minutiae", "table"]
    assert (snapshot(tmp_path / "images"), snapshot(tmp_path / "minutiae")) == before

    lines = printed(capsys)
    for line in [
        f"Sample table: {tmp_path / 'table'}",
        "Files listed: 5",
        "Rows: 5",
        "Files that failed: 0",
        "Rows with minutiae: 5 (9 minutiae)",
        "Rows with empty minutiae: 0",
        "Rows with null minutiae: 0",
        "Subjects: 4",
        "Impressions per subject: smallest 1, median 1, largest 2",
        "Parts: 1",
        f"Checksum (samples_sha256): {samples_checksum(samples)}",
        "Failures log: none",
        "Warnings log: none",
        "Resolution of the 5 image file(s) of the table:",
        "5 500 ppi from metadata of the file",
        "Ridge polarity:",
        "5 dark ridges, from --ridges",
    ]:
        assert line in lines, line


def test_a_part_holds_the_rows_as_any_parquet_reader_sees_them(builder, monkeypatch, tmp_path):
    """Read with pyarrow alone, not with the module that the script wrote the part with."""
    import pyarrow.parquet as pq

    write_named(tmp_path)
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    rows = pq.read_table(tmp_path / "table" / "part-00000-of-00001.parquet").to_pylist()
    # In the order of the numbers, which is not the order of the names: 2_1 before 10_1
    assert [row["image_path"] for row in rows] == NAMED
    assert [(row["subject"], row["impression"]) for row in rows][2:4] == [(1, 0), (9, 0)]
    row = rows[-1]
    path = tmp_path / "images" / "sensor_b" / "deep" / "17_3.tif"
    _, info = SensorNormalizer().load(str(path))
    assert (row["subject"], row["impression"]) == (16, 2)
    assert row["file_size"] == os.path.getsize(path)
    assert (row["ppi"], row["ppi_origin"]) == (500.0, "metadata")
    assert (row["ridges"], row["ridges_origin"]) == ("dark", "argument")
    assert (row["source_width"], row["source_height"]) == (info.source_width, info.source_height)
    assert (row["width"], row["height"]) == (160, 224) == (info.width, info.height)
    assert row["minutiae"] == [
        {"x": float(np.float32(x)), "y": float(np.float32(y)), "angle": float(np.float32(angle))}
        for x, y, angle in MINUTIAE[:2]
    ]


def test_the_resolution_of_a_file_comes_from_where_the_extraction_takes_it(
    builder, monkeypatch, tmp_path, capsys
):
    images = tmp_path / "images"
    # Where each resolution comes from: the manifest row of the directory, the
    # manifest row of the file (which wins over its metadata), the metadata,
    # and the assumption, for a file without metadata and for one whose
    # metadata are not believed
    save(images / "sensor_a" / "1_1.png", ridge_image(320, 256, 19), ppi=None)
    save(images / "sensor_a" / "1_2.png", ridge_image(157, 157), ppi=500)
    save(images / "2_1.png", ridge_image(246, 197, 7.3), ppi=385)
    save(images / "2_2.png", ppi=None)
    save(images / "2_3.png", ppi=96)
    manifest = tmp_path / "resolutions.csv"
    manifest.write_text("path,ppi\nsensor_a,1000\nsensor_a/1_2.png,251.2\n")
    options = ["--ppi_manifest", manifest, "--assume_ppi", 500]
    assert build(builder, monkeypatch, tmp_path, *options) == 0

    samples = read_samples(tmp_path / "table")
    normalizer = SensorNormalizer(root_dir=str(images), manifest=str(manifest), assume_ppi=500)
    found = {}
    for name, subject, impression in [
        ("sensor_a/1_1.png", 0, 0),
        ("sensor_a/1_2.png", 0, 1),
        ("2_1.png", 1, 0),
        ("2_2.png", 1, 1),
        ("2_3.png", 1, 2),
    ]:
        row = row_of(samples, subject, impression)
        assert row.pop("image_path") == name and row.pop("minutiae") is None
        assert row == as_loaded(images / name, normalizer), name
        found[name] = (row["ppi"], row["ppi_origin"], row["size"])
    assert found == {
        "sensor_a/1_1.png": (1000.0, "manifest", (128, 160)),
        # 64 bits: with the resolution rounded to 32 bits the prepared image is 313 pixels wide
        "sensor_a/1_2.png": (251.2, "manifest", (312, 312)),
        "2_1.png": (385.0, "metadata", resampled_size(197, 246, 385)),
        "2_2.png": (500.0, "assumed", (128, 160)),
        "2_3.png": (500.0, "assumed", (128, 160)),
    }
    assert resampled_size(157, 157, float(np.float32(251.2))) == (313, 313)  # control
    assert samples.build["settings"] == normalizer.settings()
    assert samples.build["settings"]["ppi_manifest_entries"] == 2

    lines = printed(capsys)
    assert "Resolution of the 5 image file(s) of the table:" in lines
    assert "2 500 ppi from --assume_ppi" in lines
    assert "1 1000 ppi from manifest (resampled to 500 ppi)" in lines
    assert "1 251.2 ppi from manifest (resampled to 500 ppi)" in lines
    assert "1 385 ppi from metadata of the file (resampled to 500 ppi)" in lines

    # --ppi is for all files and wins over manifest and metadata
    assert build(builder, monkeypatch, tmp_path, *options, "--ppi", 1000, table="table_ppi") == 0
    samples = read_samples(tmp_path / "table_ppi")
    assert samples.ppi.tolist() == [1000.0] * 5
    assert {PPI_ORIGINS[origin] for origin in samples.ppi_origin} == {"argument"}
    assert row_of(samples, 1, 1)["size"] == (64, 80)
    assert "5 1000 ppi from --ppi (resampled to 500 ppi)" in printed(capsys)


def test_the_ridge_polarity_of_a_file_is_stored_as_it_was_resolved(
    builder, monkeypatch, tmp_path, capsys
):
    images = tmp_path / "images"
    save(images / "1_1.png", 255 - ridge_image())  # light ridges on a dark background
    save(images / "1_2.png", ridge_image())
    # No background: the estimate gives no decision
    full = np.tile((127.5 + 127.5 * np.cos(np.arange(128) / 1.5)).astype(np.uint8), (160, 1))
    save(images / "2_1.png", full)
    save(images / "inverting_sensor" / "3_1.png", 255 - ridge_image())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,ridges\ninverting_sensor,light\n")

    options = ["--ridges", "auto", "--ppi_manifest", manifest]
    assert build(builder, monkeypatch, tmp_path, *options) == 0
    samples = read_samples(tmp_path / "table")
    polarity = {}
    for pair in numbers(samples):
        row = row_of(samples, *pair)
        polarity[pair] = (row["ridges"], row["ridges_origin"])
    # Never "auto": the table holds what each image was taken to hold
    assert polarity == {
        (0, 0): ("light", "estimated"),
        (0, 1): ("dark", "estimated"),
        (1, 0): ("dark", "default"),
        (2, 0): ("light", "manifest"),
    }
    assert samples.build["settings"]["ridges"] == "auto"
    # The image that allowed no estimate is named in the warnings log
    warnings = log(tmp_path / "table", WARNINGS_LOG)
    assert list(warnings) == ["2_1.png"] and "could not be estimated" in warnings["2_1.png"]
    lines = printed(capsys)
    assert "1 light ridges, estimated from the image" in lines
    assert "1 dark ridges, estimated from the image" in lines
    assert "1 dark ridges, by default: the image allowed no estimate" in lines
    assert "1 light ridges, from the manifest" in lines
    assert f"Warnings log: {tmp_path / 'table' / WARNINGS_LOG} (1 line(s))" in lines

    assert build(builder, monkeypatch, tmp_path, "--ridges", "light", table="table_light") == 0
    samples = read_samples(tmp_path / "table_light")
    assert samples.ridges_light.all()
    assert {RIDGES_ORIGINS[origin] for origin in samples.ridges_origin} == {"argument"}


def test_an_image_that_looks_inverted_is_named_in_the_warnings_log(builder, monkeypatch, tmp_path):
    save(tmp_path / "images" / "1_1.png", ridge_image())
    save(tmp_path / "images" / "a" / "1_2.png", 255 - ridge_image())
    # A warning is no failure: the exit status is 0 and the image has its row
    assert build(builder, monkeypatch, tmp_path) == 0
    samples = read_samples(tmp_path / "table")
    assert len(samples) == 2 and not samples.ridges_light.any()
    warnings = log(tmp_path / "table", WARNINGS_LOG)
    assert list(warnings) == ["a/1_2.png"]
    assert "looks like light ridges" in warnings["a/1_2.png"]
    assert not (tmp_path / "table" / FAILURES_LOG).exists()


def test_the_minutiae_can_lie_next_to_the_images(builder, monkeypatch, tmp_path):
    """One directory for both: the .mnt file of an image lies beside it."""
    save(tmp_path / "images" / "a" / "1_1.png")
    _, info = SensorNormalizer().load(str(tmp_path / "images" / "a" / "1_1.png"))
    path = write_mnt(tmp_path / "images" / "a" / "1_1.mnt", mnt_text(info))
    before = snapshot(tmp_path / "images")
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "images") == 0
    samples = read_samples(tmp_path / "table")
    assert samples.minutiae_of(0).tobytes() == stored_minutiae(path).tobytes()
    assert snapshot(tmp_path / "images") == before


def test_an_image_is_a_row_under_its_own_name_also_where_it_is_a_link(
    builder, monkeypatch, tmp_path
):
    """
    The path of a row is the name of the file as it is, also where its ending
    is written in capitals. For a symbolic link the row holds the size of the
    file that the link leads to, which is what a reader of the table loads.
    """
    images = tmp_path / "images"
    save(images / "1_1.PNG")
    extracted(tmp_path, "1_1.PNG")
    scan = save(tmp_path / "scans" / "00002239.png", ridge_image(200, 144))
    os.symlink(scan, images / "1_2.png")
    extracted(tmp_path, "1_2.png", minutiae=MINUTIAE[:2])
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    samples = read_samples(tmp_path / "table")
    assert paths(samples) == ["1_1.PNG", "1_2.png"]
    # The .mnt file of 1_1.PNG is 1_1.mnt, as the extraction names it
    assert np.diff(samples.minutiae_offsets).tolist() == [3, 2]
    sizes = [os.path.getsize(images / "1_1.PNG"), os.path.getsize(scan)]
    assert samples.file_size.tolist() == sizes
    assert row_of(samples, 0, 1)["source_size"] == (144, 200)
    # Control: the link itself has another size, that of the path it holds
    assert os.lstat(images / "1_2.png").st_size == len(scan) != os.path.getsize(scan)


# ---------------------------------------------------------------------------
# A sample list
# ---------------------------------------------------------------------------

LISTED = ["00002239_plain_05.png", "sensor b/roll 05.tif", "x/1_1.png", "x/y/z.jpg", "w.bmp"]
HEADER = "path,subject,impression\n"


def write_listed(tmp_path) -> None:
    for name in LISTED:
        save(tmp_path / "images" / name)


def with_list(tmp_path, text: str) -> list:
    path = tmp_path / "files.csv"
    path.write_text(text, encoding="utf-8")
    return ["--sample_list", path]


def test_a_sample_list_says_which_finger_each_file_shows(builder, monkeypatch, tmp_path, capsys):
    write_listed(tmp_path)
    extracted(tmp_path, "x/y/z.jpg")
    options = with_list(
        tmp_path,
        "path,subject,impression,person\n"
        "00002239_plain_05.png,7,0,anna\n"
        '"sensor b/roll 05.tif",7,3,anna\n'
        "./x//y/../1_1.png,7,9,anna\n"  # not normalised, and named like a file of today
        "x/y/z.jpg,40,1,bert\n"
        "w.bmp,0,0,carl\n",
    )
    options += ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, monkeypatch, tmp_path, *options) == 1  # four have no .mnt file

    samples = read_samples(tmp_path / "table")
    # The numbers as they are written: nothing is subtracted, 0 is a number, gaps
    # are kept, and the fingers have 1, 3 and 1 impressions
    assert dict(zip(paths(samples), numbers(samples))) == {
        "w.bmp": (0, 0),
        "00002239_plain_05.png": (7, 0),
        "sensor b/roll 05.tif": (7, 3),
        "x/1_1.png": (7, 9),
        "x/y/z.jpg": (40, 1),
    }
    row = row_of(samples, 40, 1)
    expected = stored_minutiae(tmp_path / "minutiae" / "x" / "y" / "z.mnt")
    assert row["minutiae"].tobytes() == expected.tobytes() != b""
    assert row["file_size"] == os.path.getsize(tmp_path / "images" / "x" / "y" / "z.jpg")
    # The files without a .mnt file are in the table, with null minutiae
    assert row_of(samples, 7, 9)["minutiae"] is None
    assert samples.build["files"] == 5 and samples.build["failed"] == 4

    lines = printed(capsys)
    assert f"Note: the column(s) 'person' of {tmp_path / 'files.csv'} are not used." in lines
    assert "Subjects: 3" in lines
    assert "Impressions per subject: smallest 1, median 1, largest 3" in lines


def test_a_sample_list_can_begin_with_a_byte_order_mark_and_pad_its_numbers(
    builder, monkeypatch, tmp_path
):
    write_listed(tmp_path)
    # White space around the names of the columns and around the numbers, and
    # a number with more zeros in front than the largest number has digits
    text = "\N{BYTE ORDER MARK}path , subject, impression \r\nw.bmp, 3 , 0000000000004 \r\n\r\n"
    assert build(builder, monkeypatch, tmp_path, *with_list(tmp_path, text)) == 0
    samples = read_samples(tmp_path / "table")
    assert (paths(samples), numbers(samples)) == (["w.bmp"], [(3, 4)])


def test_a_path_of_a_sample_list_is_taken_as_it_is_written(builder, monkeypatch, tmp_path):
    """
    White space at the ends of a path belongs to the name of the file, and a
    listed file needs no ending. A list that names " x.png" must not get a
    row of x.png, with the numbers of the other file.
    """
    images = tmp_path / "images"
    sizes = {
        " x.png": (144, 200),
        "x.png": (128, 160),  # the twin of the first without the white space
        "y.png ": (136, 176),
        "a/z.png\n": (152, 184),
        "scans/0007": (120, 168),
    }
    for name, (width, height) in sizes.items():
        # Written under a name that tells the image library the format
        written = save(images / "written.png", ridge_image(height, width))
        os.makedirs(dirname(str(images / name)), exist_ok=True)
        os.rename(written, images / name)
    rows = '" x.png",5,0\n"y.png ",6,0\n"a/z.png\n",7,0\nscans/0007,8,0\n'
    # ... and a file that is not there is not found under a name that is like it
    rows += "x.png ,9,0\n"
    assert build(builder, monkeypatch, tmp_path, *with_list(tmp_path, HEADER + rows)) == 1

    samples = read_samples(tmp_path / "table")
    assert dict(zip(numbers(samples), paths(samples))) == {
        (5, 0): " x.png",
        (6, 0): "y.png ",
        (7, 0): "a/z.png\n",
        (8, 0): "scans/0007",
    }
    for subject, name in ((5, " x.png"), (6, "y.png "), (7, "a/z.png\n"), (8, "scans/0007")):
        row = row_of(samples, subject, 0)
        assert row["source_size"] == sizes[name], name
        assert row["file_size"] == os.path.getsize(images / name)
    failures = log(tmp_path / "table", FAILURES_LOG)
    assert list(failures) == ["x.png "]
    assert failures["x.png "].startswith("no row: FileNotFoundError")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("path,subject\nw.bmp,1\n", ["not name the column(s) impression", "'path', 'subject'"]),
        ("path,subject,impression,subject\nw.bmp,1,2,3\n", ["column(s) subject exactly once"]),
        ("", ["column(s) path, subject, impression", "names: none"]),
        (HEADER, ["names no file"]),
        (HEADER + "w.bmp,0,0\nx/1_1.png,1.0,2\n", ["line 3: the subject '1.0' is not a whole"]),
        (HEADER + "w.bmp,-1,2\n", ["line 2: the subject '-1' is not a whole number of 0 or more"]),
        (HEADER + "w.bmp,1,two\n", ["line 2: the impression 'two' is not a whole number"]),
        (HEADER + "w.bmp,1\n", ["line 2: it names no impression"]),
        (HEADER + "w.bmp,,1\n", ["line 2: it names no subject"]),
        (HEADER + "w.bmp,2147483648,1\n", ["line 2: the subject 2147483648 is above 2147483647"]),
        (HEADER + "w.bmp,1," + "9" * 5000 + "\n", ["line 2: the impression 9999", "is above 2147"]),
        (HEADER + "\n,1,2\n", ["line 3: the path is empty"]),
        # A path of white space is not empty, it is a name: this line lacks a number
        (HEADER + '"  ",1\n', ["line 2: it names no impression"]),
        (HEADER + "./,1,2\n", ["line 2: the path './' names no file"]),
        (HEADER + "../w.bmp,1,2\n", ["line 2: the path '../w.bmp' leaves the image directory"]),
        (HEADER + "x/../../w.bmp,1,2\n", ["line 2: the path 'x/../../w.bmp' leaves the image"]),
        (HEADER + "/data/w.bmp,1,2\n", ["line 2: the path '/data/w.bmp' is absolute"]),
        (HEADER + "x\\1_1.png,1,2\n", ["line 2: the path 'x\\1_1.png' holds a backslash"]),
        (HEADER + "w\x00.bmp,1,2\n", ["line 2: the path 'w\\x00.bmp' holds a NUL character"]),
        (HEADER + "w.bmp,1,2,3\n", ["line 2: it has more fields than the header line"]),
        (
            HEADER + "".join(f"w.bmp,x{n},1\n" for n in range(8)),
            ["8 line(s)", "line 2: the subject 'x0'", "line 6: the subject 'x4'", "; and 3 more"],
        ),
    ],
)
def test_a_sample_list_that_cannot_be_used_refuses_the_build(
    builder, monkeypatch, tmp_path, capsys, text, expected
):
    write_listed(tmp_path)
    forbid_loading(builder, monkeypatch)
    message = refused(builder, monkeypatch, capsys, tmp_path, *with_list(tmp_path, text))
    for part in expected:
        assert part in message, message
    # The list, what a list is, and where it is described
    assert str(tmp_path / "files.csv") in message
    assert "the columns path, subject and impression" in message
    assert "docs/sample-table.md" in message
    assert tables(tmp_path) == []


def test_a_sample_list_that_is_not_utf8_refuses_the_build(builder, monkeypatch, tmp_path, capsys):
    write_listed(tmp_path)
    (tmp_path / "files.csv").write_bytes(b"path,subject,impression\ncaf\xe9.png,1,2\n")
    forbid_loading(builder, monkeypatch)
    options = ["--sample_list", tmp_path / "files.csv"]
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert "cannot be read as a CSV file in UTF-8 (UnicodeDecodeError" in message
    assert tables(tmp_path) == []


def test_files_that_a_list_cannot_tell_apart_refuse_the_build(
    builder, monkeypatch, tmp_path, capsys
):
    write_listed(tmp_path)
    save(tmp_path / "images" / "w.png")
    forbid_loading(builder, monkeypatch)

    # The same file twice, also where it is written in two ways
    options = with_list(tmp_path, HEADER + "w.bmp,0,0\nx/1_1.png,1,0\n./x/y/../1_1.png,1,1\n")
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert f"{tmp_path / 'files.csv'} names 1 file(s) more than once" in message
    assert "x/1_1.png (line 3 and line 4)" in message
    assert "remove the repeated lines" in message

    # Two files with the same pair of numbers
    options = with_list(tmp_path, HEADER + "w.bmp,5,7\nx/1_1.png,5,8\nx/y/z.jpg,5,7\n")
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert "1 pair(s) of image files have the same subject and impression" in message
    assert "subject 5, impression 7: line 2 (w.bmp) and line 4 (x/y/z.jpg)" in message
    assert "Give every line a pair of numbers of its own" in message

    # Two files that differ only in their ending share one .mnt file
    options = with_list(tmp_path, HEADER + "w.bmp,0,0\nx/1_1.png,1,0\nw.png,2,0\n")
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert "1 pair(s) of image files differ only in their ending" in message
    assert "line 2 (w.bmp) and line 4 (w.png)" in message
    assert "share one .mnt file" in message and "Rename one file of each pair" in message
    assert tables(tmp_path) == []


def test_a_table_does_not_depend_on_how_a_list_writes_its_paths(builder, monkeypatch, tmp_path):
    write_listed(tmp_path)
    plain = with_list(tmp_path, HEADER + "x/1_1.png,3,1\nw.bmp,3,0\nx/y/z.jpg,1,5\n")
    assert build(builder, monkeypatch, tmp_path, *plain, table="table_plain") == 0
    # The same files in another order, with paths that are not normalised
    other = with_list(tmp_path, HEADER + "./x/y/z.jpg,1,5\nx/y/../../w.bmp,3,0\nx//1_1.png,3,1\n")
    assert build(builder, monkeypatch, tmp_path, *other, table="table_other") == 0
    plain, other = read_samples(tmp_path / "table_plain"), read_samples(tmp_path / "table_other")
    assert paths(other) == ["x/y/z.jpg", "w.bmp", "x/1_1.png"]
    assert samples_checksum(plain) == samples_checksum(other)
    assert plain.build["id"] == other.build["id"]


def test_a_listed_file_that_is_missing_has_no_row(builder, monkeypatch, tmp_path):
    write_listed(tmp_path)
    options = with_list(tmp_path, HEADER + "w.bmp,0,0\nnot/there.png,0,1\n")
    assert build(builder, monkeypatch, tmp_path, *options) == 1
    assert paths(read_samples(tmp_path / "table")) == ["w.bmp"]
    failures = log(tmp_path / "table", FAILURES_LOG)
    assert list(failures) == ["not/there.png"]
    assert failures["not/there.png"].startswith("no row: FileNotFoundError")


# ---------------------------------------------------------------------------
# Workers and parts
# ---------------------------------------------------------------------------


def write_many(tmp_path, subjects=range(1, 8), impressions=(1, 2)) -> list:
    """
    A set with everything a build can meet: minutiae, a .mnt file without
    minutiae and one without a header, .mnt files that are refused and
    missing, an image that cannot be loaded and one that looks inverted.
    Returns the names in the order of (subject, impression).
    """
    names = []
    for subject in subjects:
        for impression in impressions:
            name = join(f"sensor_{subject % 3}", f"{subject}_{impression}.png")
            names.append(name)
            kind = (subject + 3 * impression) % 7
            image = ridge_image(160 + 8 * subject, 128 + 16 * impression)
            save(tmp_path / "images" / name, 255 - image if kind == 6 else image)
            mnt_path = tmp_path / "minutiae" / name.replace(".png", ".mnt")
            if kind == 0:
                (tmp_path / "images" / name).write_bytes(b"not an image")
            elif kind == 1:
                write_mnt(mnt_path, "")
            elif kind == 2:
                write_mnt(mnt_path, "1\n8 8\n4 4 0.5\n")
            elif kind == 3:
                extracted(tmp_path, name, minutiae=[])
            elif kind != 4:  # 4: no .mnt file
                extracted(tmp_path, name, minutiae=MINUTIAE[: subject % 3 + 1])
    return names


def test_worker_processes_give_the_same_table_as_one_process(
    builder, monkeypatch, tmp_path, capsys
):
    write_many(tmp_path)
    options = ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, monkeypatch, tmp_path, *options, table="table_1") == 1
    one_process = printed(capsys)
    assert build(builder, monkeypatch, tmp_path, *options, "--workers", 3, table="table_3") == 1
    workers = printed(capsys)

    one, three = read_samples(tmp_path / "table_1"), read_samples(tmp_path / "table_3")
    # Control: the set holds rows with, with empty and with null minutiae, and a file without a row
    counts = np.diff(one.minutiae_offsets)
    assert (counts > 0).any() and (one.minutiae_known & (counts == 0)).any()
    assert not one.minutiae_known.all() and len(one) < one.build["files"]
    assert samples_checksum(one) == samples_checksum(three)
    assert one.build == three.build
    for name in ARRAYS:  # every array, also the origins, which the checksum leaves out
        assert getattr(one, name).tobytes() == getattr(three, name).tobytes(), name
    # The logs do not depend on the order in which the workers finish
    for name in (FAILURES_LOG, WARNINGS_LOG):
        text = (tmp_path / "table_1" / name).read_text()
        assert text and (tmp_path / "table_3" / name).read_text() == text
    # ... and neither does what is printed, apart from the directory
    assert [line.replace("table_3", "table_1") for line in workers] == one_process


def test_the_files_are_cut_into_parts_in_the_order_of_their_numbers(builder, monkeypatch, tmp_path):
    names = write_many(tmp_path, subjects=(1, 2, 3, 10, 11, 20, 100))
    options = ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, monkeypatch, tmp_path, *options, table="table_one") == 1
    monkeypatch.setattr(builder, "FILES_PER_PART", 4)
    assert build(builder, monkeypatch, tmp_path, *options, table="table_parts") == 1
    options += ["--workers", 2]
    assert build(builder, monkeypatch, tmp_path, *options, table="table_workers") == 1

    # Control: by name the files are listed in another order than by number
    listed = list(builder.iter_image_paths(str(tmp_path / "images")))
    assert sorted(listed) == sorted(names) and listed != names
    one = read_samples(tmp_path / "table_one")
    for table in ("table_parts", "table_workers"):
        assert sorted(name for name in os.listdir(tmp_path / table) if "part" in name) == [
            f"part-0000{index}-of-00004.parquet" for index in range(4)
        ]
        parts = stored_parts(tmp_path / table)
        # 14 files: three parts of four listed files and one of two, whatever became of the files
        assert [information["files"] for information, _ in parts] == [4, 4, 4, 2]
        assert [information["rows"] for information, _ in parts] == [
            len(pairs) for _, pairs in parts
        ]
        assert sum(len(pairs) for _, pairs in parts) == len(one) < 14
        assert sum(information["failed"] for information, _ in parts) == one.build["failed"]
        # Every part is in the order of the rows, and so are the parts one after the other
        in_files = [pair for _, pairs in parts for pair in pairs]
        assert in_files == sorted(in_files) == numbers(one)
        for index, (information, _) in enumerate(parts):
            assert (information["part"], information["parts"]) == (index, 4)
            assert information["id"] == one.build["id"]
            assert information["settings"] == one.build["settings"]
        samples = read_samples(tmp_path / table)
        assert samples_checksum(samples) == samples_checksum(one)
        assert dict(samples.build, parts=1) == one.build
        for name in (FAILURES_LOG, WARNINGS_LOG):
            text = (tmp_path / "table_one" / name).read_text()
            assert (tmp_path / table / name).read_text() == text


def test_a_part_whose_files_all_failed_holds_no_rows(builder, monkeypatch, tmp_path, capsys):
    for subject in range(1, 7):
        save(tmp_path / "images" / f"{subject}_1.png")
    for subject in (3, 4):
        (tmp_path / "images" / f"{subject}_1.png").write_bytes(b"not an image")
    monkeypatch.setattr(builder, "FILES_PER_PART", 2)
    assert build(builder, monkeypatch, tmp_path) == 1
    parts = stored_parts(tmp_path / "table")
    assert [pairs for _, pairs in parts] == [[(0, 0), (1, 0)], [], [(4, 0), (5, 0)]]
    assert [(part["files"], part["failed"], part["rows"]) for part, _ in parts] == [
        (2, 0, 2),
        (2, 2, 0),
        (2, 0, 2),
    ]
    samples = read_samples(tmp_path / "table")
    assert (len(samples), samples.build["files"], samples.build["failed"]) == (4, 6, 2)
    lines = printed(capsys)
    assert "Files that failed: 2 (2 without a row, 0 with null minutiae)" in lines
    assert "Parts: 3" in lines


def test_a_build_has_the_id_of_what_was_asked_for(builder, monkeypatch, tmp_path):
    for name in ("1_1.png", "1_2.png", "2_1.png"):
        save(tmp_path / "images" / name)
        extracted(tmp_path, name)
    minutiae = ["--minutiae_dir", tmp_path / "minutiae"]

    def id_of_build(table, *options) -> str:
        assert build(builder, monkeypatch, tmp_path, *options, table=table) == 0
        return read_samples(tmp_path / table).build["id"]

    first = id_of_build("table_a", *minutiae)
    assert id_of_build("table_b", *minutiae, "--workers", 2) == first  # the same build again
    assert id_of_build("table_c") != first  # without minutiae
    assert id_of_build("table_d", *minutiae, "--ridges", "auto") != first  # other settings
    listed = with_list(tmp_path, HEADER + "1_1.png,0,0\n1_2.png,0,1\n2_1.png,1,0\n")
    assert id_of_build("table_e", *minutiae, *listed) == first  # the same files and numbers
    listed = with_list(tmp_path, HEADER + "1_1.png,0,0\n1_2.png,0,1\n2_1.png,1,1\n")
    assert id_of_build("table_f", *minutiae, *listed) != first  # another number
    # Another file under the same numbers, with a name of the same length
    os.rename(tmp_path / "images" / "2_1.png", tmp_path / "images" / "2_2.png")
    os.rename(tmp_path / "minutiae" / "2_1.mnt", tmp_path / "minutiae" / "2_2.mnt")
    listed = with_list(tmp_path, HEADER + "1_1.png,0,0\n1_2.png,0,1\n2_2.png,1,0\n")
    assert id_of_build("table_g", *minutiae, *listed) != first
    os.remove(tmp_path / "images" / "2_2.png")
    assert id_of_build("table_h", *minutiae) != first  # one file less


def test_a_worker_that_dies_costs_the_file_it_was_loading_and_no_other(
    builder, monkeypatch, tmp_path
):
    names = write_many(tmp_path, subjects=range(1, 11))
    options = ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, monkeypatch, tmp_path, *options, table="table_whole") == 1
    whole = read_samples(tmp_path / "table_whole")

    killer = str(tmp_path / "images" / "sensor_2" / "5_1.png")
    once = str(tmp_path / "images" / "sensor_1" / "7_2.png")
    marker = tmp_path / "died_once"
    load_sample = builder.load_sample

    def dying(source, subject, impression, image_path):
        path = join(source[0], image_path)
        if path == killer:
            os._exit(1)  # the worker ends at once, every time: a file that it does not survive
        if path == once and not marker.exists():
            marker.touch()
            os._exit(1)  # the worker ends once: not the fault of the file
        return load_sample(source, subject, impression, image_path)

    # The workers are forked from this process, so they have the function that is put in here
    monkeypatch.setattr(builder, "load_sample", dying)
    children = multiprocessing.active_children()
    assert build(builder, monkeypatch, tmp_path, *options, "--workers", 3) == 1
    assert marker.exists()  # control: a worker died for the other file as well

    samples = read_samples(tmp_path / "table")
    # The file whose worker died twice has no row. The file that was loaded
    # again and every other file have the rows of an undisturbed build.
    assert whole.find(4, 0) >= 0 and samples.find(4, 0) == -1
    assert numbers(samples) == [pair for pair in numbers(whole) if pair != (4, 0)]
    assert (6, 1) in numbers(samples)
    for row, pair in enumerate(numbers(samples)):
        other = whole.find(*pair)
        assert samples.image_path(row) == whole.image_path(other)
        assert samples.minutiae_known[row] == whole.minutiae_known[other]
        assert samples.minutiae_of(row).tobytes() == whole.minutiae_of(other).tobytes()
    failures = log(tmp_path / "table", FAILURES_LOG)
    assert failures.pop("sensor_2/5_1.png").startswith("no row: the worker process died")
    assert failures == log(tmp_path / "table_whole", FAILURES_LOG)
    assert len(names) == 20 == samples.build["files"]
    assert samples.build["failed"] == whole.build["failed"] + 1
    # The workers that died and those that took their places have all ended
    assert multiprocessing.active_children() == children


def test_a_worker_that_dies_while_it_has_no_file_is_replaced_and_costs_no_file(
    builder, monkeypatch, tmp_path
):
    """
    A worker has no file while files are left when one file takes long: with
    two workers no file is handed out that is 32 or more files behind the
    one that is waited for. Here the first file takes as long as the other
    worker needs for the 31 files after it, that worker is killed when it
    has returned the last of them, and the files behind are held back until
    then, so that one of them is handed to the worker that is dead.
    """
    window, count = 32, 36
    for subject in range(1, count + 1):
        save(tmp_path / "images" / f"{subject}_1.png")
    slow, loaded, killed = tmp_path / "slow", tmp_path / "loaded", tmp_path / "killed"
    record = tmp_path / "record"
    load_sample = builder.load_sample

    def staged(source, subject, impression, image_path):
        with open(record, "a") as file:
            file.write(f"{subject} {os.getpid()}\n")
        if subject == 0:
            slow.write_text(str(os.getpid()))
            wait_until(loaded.exists)
        if subject >= window:
            wait_until(killed.exists)
        result = load_sample(source, subject, impression, image_path)
        if subject == window - 1:
            loaded.touch()
        return result

    write_part = builder.write_part
    children = multiprocessing.active_children()

    def killing(directory, index, *arguments):
        # With one file per part, the part of file 31 is written when the
        # files 0 to 31 are done: the worker that loaded 1 to 31 has returned
        # them all, and has been handed no other file since
        if index == window - 1:
            (idle,) = [
                process
                for process in multiprocessing.active_children()
                if process not in children and process.pid != int(slow.read_text())
            ]
            idle.kill()
            idle.join()
            killed.write_text(str(idle.pid))
        return write_part(directory, index, *arguments)

    # The workers are forked from this process, so they have the function that is put in here
    monkeypatch.setattr(builder, "load_sample", staged)
    monkeypatch.setattr(builder, "write_part", killing)
    monkeypatch.setattr(builder, "FILES_PER_PART", 1)
    assert build(builder, monkeypatch, tmp_path, "--workers", 2) == 0

    samples = read_samples(tmp_path / "table")
    assert numbers(samples) == [(subject, 0) for subject in range(count)]
    assert samples.build["failed"] == 0 and not (tmp_path / "table" / FAILURES_LOG).exists()
    loads = [line.split() for line in record.read_text().splitlines()]
    # Every file was loaded once, also the one that was handed to the dead worker
    assert sorted(int(subject) for subject, _ in loads) == list(range(count))
    # ... by the two workers of the start and by the one that took the place
    # of the dead one, which had loaded the 31 files
    workers = [pid for _, pid in loads]
    assert len(set(workers)) == 3 and workers.count(killed.read_text()) == window - 1
    assert multiprocessing.active_children() == children


# ---------------------------------------------------------------------------
# The .mnt file of an image
# ---------------------------------------------------------------------------


def write_pair(tmp_path) -> None:
    """Two prints of one finger; the first has a .mnt file that is accepted."""
    save(tmp_path / "images" / "1_1.png")
    save(tmp_path / "images" / "1_2.png", ridge_image(200, 144))
    extracted(tmp_path, "1_1.png")


def info_of(tmp_path, name: str = "1_2.png"):
    return SensorNormalizer().load(str(tmp_path / "images" / name))[1]


@pytest.mark.parametrize(
    "case, reasons",
    [
        ("missing", ["the .mnt file is missing"]),
        ("other size", ["states an image of 128x160 pixels, and the prepared image has 144x200"]),
        ("other width", ["states an image of 145x200 pixels, and the prepared image has 144x"]),
        ("other height", ["states an image of 144x199 pixels, and the prepared image has 144x"]),
        ("swapped size", ["states an image of 200x144 pixels, and the prepared image has 144x"]),
        ("size 0", ["states an image of 0x0 pixels"]),
        ("other polarity", ["names light ridges, and this build took the image to have dark"]),
        ("other polarity, then empty lines", ["names light ridges, and this build took the"]),
        ("other source", ["names a source image of 288x400 pixels, and the image file has 144x"]),
        ("other source width", ["names a source image of 288x200 pixels, and the image file"]),
        ("other source height", ["names a source image of 144x400 pixels, and the image file"]),
        ("no .mnt file", ["the .mnt file cannot be read (ValueError"]),
        ("no text", ["the .mnt file cannot be read (UnicodeDecodeError"]),
        ("a directory", ["the .mnt file cannot be read (IsADirectoryError"]),
        ("three numbers", ["the .mnt file cannot be read (ValueError"]),
        ("count above the rows", ["the .mnt file cannot be read (ValueError"]),
        ("not a number", ["2 of the 3 minutiae", "is minutia 2 (100.0 nan 6.0)"]),
        ("infinite", ["1 of the 3 minutiae", "is minutia 3 (63.75 80.0 inf)"]),
        ("too large for 32 bits", ["1 of the 3 minutiae", "is minutia 1 (inf 20.25 0.5)"]),
    ],
)
def test_a_mnt_file_that_is_refused_leaves_its_image_in_the_table_without_minutiae(
    builder, monkeypatch, tmp_path, capsys, case, reasons
):
    write_pair(tmp_path)
    path = tmp_path / "minutiae" / "1_2.mnt"
    good = mnt_text(info_of(tmp_path))
    # Control: what the cases replace is in the file once
    assert good.startswith("3\n144 200\n10.5000 20.2500 0.500000 -1 0.5000\n")
    assert good.count("ridges dark (argument)") == good.count("source 144x200") == 1
    changed = {
        "other size": good.replace("144 200\n", "128 160\n"),
        "other width": good.replace("144 200\n", "145 200\n"),
        "other height": good.replace("144 200\n", "144 199\n"),
        "swapped size": good.replace("144 200\n", "200 144\n"),
        "size 0": "0\n0 0\n",
        "other polarity": good.replace("ridges dark (argument)", "ridges light (argument)"),
        # The comment of the extraction is the last line that is not empty
        "other polarity, then empty lines": good.replace("ridges dark (", "ridges light (")
        + "\n   \n\n",
        "other source": good.replace("source 144x200", "source 288x400"),
        "other source width": good.replace("source 144x200", "source 288x200"),
        "other source height": good.replace("source 144x200", "source 144x400"),
        "no .mnt file": "These are no minutiae.\nNot at all.\n",
        "three numbers": good.replace("144 200\n", "144 200 3\n"),
        # The comment line is then read as the fourth minutia
        "count above the rows": good.replace("3\n", "4\n", 1),
        "not a number": good.replace("150.0000", "nan").replace("3.141593", "NaN"),
        "infinite": good.replace("3.141593", "inf"),
        "too large for 32 bits": good.replace("10.5000", "1e39"),
    }
    if case in changed:
        write_mnt(path, changed[case])
    elif case == "no text":
        os.makedirs(path.parent, exist_ok=True)
        path.write_bytes(b"\xff\xfe\x00\x01\n\x80\x81\n")
    elif case == "a directory":
        os.makedirs(path)
    else:
        assert case == "missing"

    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    samples = read_samples(tmp_path / "table")
    # The row is there, with everything but the minutiae: the image can be used without them
    row = row_of(samples, 0, 1)
    assert row.pop("minutiae") is None and row.pop("image_path") == "1_2.png"
    assert row == as_loaded(tmp_path / "images" / "1_2.png", SensorNormalizer())
    assert len(row_of(samples, 0, 0)["minutiae"]) == 3
    failures = log(tmp_path / "table", FAILURES_LOG)
    assert list(failures) == ["1_2.png"]
    assert failures["1_2.png"].startswith("null minutiae: ")
    for reason in reasons:
        assert reason in failures["1_2.png"], failures["1_2.png"]
    assert (samples.build["files"], samples.build["failed"]) == (2, 1)
    lines = printed(capsys)
    assert "Files that failed: 1 (0 without a row, 1 with null minutiae)" in lines
    assert "Rows with null minutiae: 1 (the .mnt file was missing or refused)" in lines
    assert f"Failures log: {tmp_path / 'table' / FAILURES_LOG} (1 line(s))" in lines
    assert f"1_2.png: {failures['1_2.png']}" in lines  # the first failures are printed


def test_a_mnt_file_without_minutiae_gives_an_empty_list(builder, monkeypatch, tmp_path, capsys):
    for name in ("1_1.png", "1_2.png", "2_1.png", "2_2.png", "3_1.png"):
        save(tmp_path / "images" / name)
    extracted(tmp_path, "1_1.png")
    write_mnt(tmp_path / "minutiae" / "1_2.mnt", "")  # 0 bytes
    write_mnt(tmp_path / "minutiae" / "2_1.mnt", "0\n128 160\n")  # the header 0
    extracted(tmp_path, "2_2.png", minutiae=[])  # ... and with the comment of the extraction
    write_mnt(tmp_path / "minutiae" / "3_1.mnt", "7\n")  # one line: no header either
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0

    samples = read_samples(tmp_path / "table")
    # Empty, not null: the images have no minutiae, as the training takes these files
    assert samples.minutiae_known.all()
    assert np.diff(samples.minutiae_offsets).tolist() == [3, 0, 0, 0, 0]
    for row in range(1, 5):
        minutiae = samples.minutiae_of(row)
        assert minutiae.shape == (0, 3) and minutiae.dtype == np.float32
    assert not (tmp_path / "table" / FAILURES_LOG).exists()
    assert samples.build["failed"] == 0
    # The files without a header are named, those with the header 0 are not
    warnings = log(tmp_path / "table", WARNINGS_LOG)
    assert sorted(warnings) == ["1_2.png", "3_1.png"]
    assert "holds no header (0 line(s))" in warnings["1_2.png"]
    assert "holds no header (1 line(s))" in warnings["3_1.png"]
    lines = printed(capsys)
    assert "Rows with minutiae: 1 (3 minutiae)" in lines
    assert "Rows with empty minutiae: 4" in lines
    assert "Rows with null minutiae: 0" in lines
    # ... and counted
    note = "Note: 2 .mnt file(s) hold no header and were taken as files without minutiae."
    assert f"{note} They are named in {tmp_path / 'table' / WARNINGS_LOG}." in lines


def test_two_warnings_about_one_file_are_one_line_of_the_warnings_log(
    builder, monkeypatch, tmp_path
):
    # An image that looks inverted, with a .mnt file that holds no header
    save(tmp_path / "images" / "1_1.png", 255 - ridge_image())
    write_mnt(tmp_path / "minutiae" / "1_1.mnt", "")
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    warnings = log(tmp_path / "table", WARNINGS_LOG)
    assert list(warnings) == ["1_1.png"]
    assert "looks like light ridges" in warnings["1_1.png"]
    assert "the .mnt file holds no header (0 line(s))" in warnings["1_1.png"]


def test_without_a_minutiae_directory_every_row_has_null_minutiae(
    builder, monkeypatch, tmp_path, capsys
):
    write_pair(tmp_path)
    assert build(builder, monkeypatch, tmp_path) == 0  # no failure
    samples = read_samples(tmp_path / "table")
    assert len(samples) == 2 and not samples.minutiae_known.any() and len(samples.minutiae) == 0
    assert samples.build["with_minutiae"] is False and samples.build["failed"] == 0
    assert os.listdir(tmp_path / "table") == ["part-00000-of-00001.parquet"]
    lines = printed(capsys)
    assert "Minutiae: not read (no --minutiae_dir): every row gets null minutiae" in lines
    assert "Rows with null minutiae: 2 (the table was built without --minutiae_dir)" in lines


def test_a_mnt_file_without_the_comment_of_the_extraction_is_accepted_by_its_header(
    builder, monkeypatch, tmp_path, capsys
):
    write_pair(tmp_path)
    # Another comment, and no comment: neither says how the image was prepared
    text = mnt_text(info_of(tmp_path), comment=False) + "# by another tool\n"
    write_mnt(tmp_path / "minutiae" / "1_2.mnt", text)
    save(tmp_path / "images" / "1_3.png")
    extracted(tmp_path, "1_3.png", comment=False)
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    samples = read_samples(tmp_path / "table")
    assert np.diff(samples.minutiae_offsets).tolist() == [3, 3, 3]
    output = " ".join(capsys.readouterr().out.split())
    assert "Note: 2 .mnt file(s) do not end with the comment line" in output
    assert not os.path.exists(tmp_path / "table" / WARNINGS_LOG)


def test_fewer_minutiae_than_a_mnt_file_announces_are_stored_and_named(
    builder, monkeypatch, tmp_path
):
    write_pair(tmp_path)
    # Five are announced and three are there (with a comment line the file could not be read)
    text = mnt_text(info_of(tmp_path), comment=False).replace("3\n", "5\n", 1)
    path = write_mnt(tmp_path / "minutiae" / "1_2.mnt", text)
    save(tmp_path / "images" / "1_3.png")
    # No row of the file has three values: read_mnt_file returns another shape for it
    write_mnt(tmp_path / "minutiae" / "1_3.mnt", "2\n128 160\n10 20\n30 40\n")
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 0
    samples = read_samples(tmp_path / "table")
    assert samples.minutiae_of(1).tobytes() == stored_minutiae(path).tobytes()
    assert np.diff(samples.minutiae_offsets).tolist() == [3, 3, 0] and samples.minutiae_known.all()
    warnings = log(tmp_path / "table", WARNINGS_LOG)
    assert "announces 5 minutiae, and 3 could be read" in warnings["1_2.png"]
    assert "announces 2 minutiae, and 0 could be read" in warnings["1_3.png"]


def test_a_reason_that_quotes_a_file_is_cut_in_the_log_and_on_the_terminal(
    builder, monkeypatch, tmp_path, capsys
):
    """One file must not fill the log: here its .mnt file holds a value of 100,000 characters."""
    write_pair(tmp_path)

    def reason(value: str) -> str:
        """Why a .mnt file with this text in the place of a number is refused."""
        cause = f"ValueError: could not convert string to float: '{value}'"
        return f"null minutiae: the .mnt file cannot be read ({cause})"

    # Values with which the reason has 100,000 characters and more, exactly
    # 1,000 characters, and one more than that
    values = {"1_2.png": "9" * 100_000 + "x"}
    for name, length in (("1_3.png", 1000), ("1_4.png", 1001)):
        save(tmp_path / "images" / name, ridge_image(200, 144))
        values[name] = "9" * (length - len(reason("x"))) + "x"
        assert len(reason(values[name])) == length
    for name, value in values.items():
        text = mnt_text(info_of(tmp_path)).replace("10.5000", value)
        write_mnt(tmp_path / "minutiae" / name.replace(".png", ".mnt"), text)
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1

    failures = log(tmp_path / "table", FAILURES_LOG)
    # A reason of 1,000 characters is whole. Of a longer one the log holds
    # the first 1,000 characters, and how many more there are.
    assert failures["1_3.png"] == reason(values["1_3.png"])
    assert failures["1_4.png"] == reason(values["1_4.png"])[:1000] + "... (1 more characters)"
    whole = reason(values["1_2.png"])
    assert len(whole) == 100_096
    assert failures["1_2.png"] == whole[:1000] + "... (99,096 more characters)"
    assert os.path.getsize(tmp_path / "table" / FAILURES_LOG) < 3300
    lines = printed(capsys)
    for name in values:
        assert f"{name}: {failures[name]}" in lines
    assert max(len(line) for line in lines) < 1100
    assert len(read_samples(tmp_path / "table")) == 4


def test_a_mnt_file_with_thousands_of_minutiae_is_stored_whole(builder, monkeypatch, tmp_path):
    """More than a pipe between two processes holds: the worker has to wait for its reader."""
    write_pair(tmp_path)
    rng = np.random.default_rng(5)
    many = rng.uniform((0, 0, 0), (144, 200, 6.28), (20_000, 3)).tolist()
    path = extracted(tmp_path, "1_2.png", minutiae=many)
    options = ["--minutiae_dir", tmp_path / "minutiae", "--workers", 2]
    assert build(builder, monkeypatch, tmp_path, *options) == 0
    samples = read_samples(tmp_path / "table")
    assert np.diff(samples.minutiae_offsets).tolist() == [3, 20_000]
    assert samples.minutiae_of(1).tobytes() == stored_minutiae(path).tobytes()


def test_a_mnt_file_is_looked_for_where_the_extraction_writes_it(builder, monkeypatch, tmp_path):
    """At the path of the image, not by the name of the file anywhere in the directory."""
    save(tmp_path / "images" / "sensor_a" / "1_1.png")
    save(tmp_path / "images" / "sensor_b" / "1_2.tiff")
    info = SensorNormalizer().load(str(tmp_path / "images" / "sensor_a" / "1_1.png"))[1]
    write_mnt(tmp_path / "minutiae" / "1_1.mnt", mnt_text(info))  # not below sensor_a
    write_mnt(tmp_path / "minutiae" / "sensor_b" / "1_2.mnt", mnt_text(info, MINUTIAE[:2]))
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    samples = read_samples(tmp_path / "table")
    assert samples.minutiae_known.tolist() == [False, True]
    assert len(samples.minutiae_of(1)) == 2
    assert list(log(tmp_path / "table", FAILURES_LOG)) == ["sensor_a/1_1.png"]


def test_no_row_holds_minutiae_from_a_directory_laid_out_in_another_way(
    builder, monkeypatch, tmp_path, capsys
):
    save(tmp_path / "images" / "sensor_a" / "1_1.png")
    save(tmp_path / "images" / "sensor_a" / "1_2.png")
    os.makedirs(tmp_path / "minutiae")
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    output = " ".join(capsys.readouterr().out.split())
    assert "WARNING: no row of the table holds minutiae" in output
    assert "scripts/extract_minutiae.py" in output


def test_minutiae_of_another_ridge_polarity_than_the_build_takes_are_refused(
    builder, monkeypatch, tmp_path
):
    """
    What the comment line of a .mnt file is compared for: the extraction took
    the image to have light ridges and inverted it. A build that takes it to
    have dark ridges prepares an image of the same size, which the header
    cannot tell from the image that the minutiae fit.
    """
    save(tmp_path / "images" / "1_1.png", 255 - ridge_image())
    path = extracted(tmp_path, "1_1.png", SensorNormalizer(ridges="auto"))
    with open(path) as file:
        assert file.read().splitlines()[-1].endswith("ridges light (estimated), prepared 128x160")

    options = ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, monkeypatch, tmp_path, *options, "--ridges", "auto") == 0
    samples = read_samples(tmp_path / "table")
    assert samples.ridges_light.tolist() == [True]
    assert samples.minutiae_of(0).tobytes() == stored_minutiae(path).tobytes() != b""

    assert build(builder, monkeypatch, tmp_path, *options, table="table_dark") == 1
    samples = read_samples(tmp_path / "table_dark")
    assert samples.ridges_light.tolist() == [False] and samples.minutiae_known.tolist() == [False]
    message = log(tmp_path / "table_dark", FAILURES_LOG)["1_1.png"]
    assert "names light ridges, and this build took the image to have dark ridges" in message
    assert "looks like light ridges" in log(tmp_path / "table_dark", WARNINGS_LOG)["1_1.png"]

    # The other way round: the extraction was told that the ridges are dark,
    # and the build finds out from the image that they are light
    path = extracted(tmp_path, "1_1.png", SensorNormalizer())
    with open(path) as file:
        assert file.read().splitlines()[-1].endswith("ridges dark (argument), prepared 128x160")
    assert build(builder, monkeypatch, tmp_path, *options, "--ridges", "auto", table="auto") == 1
    samples = read_samples(tmp_path / "auto")
    assert samples.ridges_light.tolist() == [True] and samples.minutiae_known.tolist() == [False]
    message = log(tmp_path / "auto", FAILURES_LOG)["1_1.png"]
    assert "names dark ridges, and this build took the image to have light ridges" in message
    assert "(estimated)" in message


# ---------------------------------------------------------------------------
# Images that cannot be loaded
# ---------------------------------------------------------------------------


def test_an_image_that_cannot_be_loaded_has_a_line_in_the_failures_log_and_no_row(
    builder, monkeypatch, tmp_path, capsys
):
    images = tmp_path / "images"
    save(images / "1_1.png")
    extracted(tmp_path, "1_1.png")
    (images / "a").mkdir()
    (images / "a" / "1_2.png").write_bytes(b"not an image")
    save(images / "2_1.png", ppi=None)  # no resolution in the file, and none is given
    save(images / "2_2.png", ppi=96)  # a resolution that is not believed
    Image.fromarray(ridge_image()).save(str(images / "3_1.png"), dpi=(500, 1000))  # not square
    write_mnt(tmp_path / "minutiae" / "a" / "1_2.mnt", "0\n128 160\n")  # a .mnt file does not help

    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    samples = read_samples(tmp_path / "table")
    assert paths(samples) == ["1_1.png"]
    failures = log(tmp_path / "table", FAILURES_LOG)
    assert sorted(failures) == ["2_1.png", "2_2.png", "3_1.png", "a/1_2.png"]
    assert all(message.startswith("no row: ") for message in failures.values())
    assert "ImageReadError: Could not read the file as an image" in failures["a/1_2.png"]
    assert "UnknownResolutionError: Resolution unknown" in failures["2_1.png"]
    assert "--assume_ppi" in failures["2_1.png"]  # the preparation says what to do
    assert "96 ppi" in failures["2_2.png"] and "not square" in failures["3_1.png"]
    assert (samples.build["files"], samples.build["failed"], samples.build["rows"]) == (5, 4, 1)

    lines = printed(capsys)
    assert "Rows: 1" in lines
    assert "Files that failed: 4 (4 without a row, 0 with null minutiae)" in lines
    path = tmp_path / "table" / FAILURES_LOG
    assert f"4 file(s) failed. The first 4 of them (all are in {path}):" in lines
    for name, message in failures.items():
        assert f"{name}: {message}" in lines
    assert '"no row: ..." names an image that could not be loaded' in " ".join(lines)

    # With an assumption for the resolution the same files load, but for the two that are no prints
    assert build(builder, monkeypatch, tmp_path, "--assume_ppi", 500, table="table_assumed") == 1
    assert sorted(log(tmp_path / "table_assumed", FAILURES_LOG)) == ["3_1.png", "a/1_2.png"]
    assert paths(read_samples(tmp_path / "table_assumed")) == ["1_1.png", "2_1.png", "2_2.png"]


@pytest.mark.parametrize("minutiae", ["no minutiae directory", "a minutiae directory"])
def test_a_build_in_which_no_image_could_be_loaded_writes_no_table(
    builder, monkeypatch, tmp_path, capsys, minutiae
):
    for name in ("1_1.png", "1_2.png", "2_1.png"):
        save(tmp_path / "images" / name, ppi=None)
    options = []
    if minutiae == "a minutiae directory":
        os.makedirs(tmp_path / "minutiae")
        options = ["--minutiae_dir", tmp_path / "minutiae"]
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert "None of the 3 listed files could be loaded as an image" in message
    # A log is not kept without its table: the message gives the first reasons, and what to do
    assert "1_1.png: no row: UnknownResolutionError: Resolution unknown" in message
    assert "1_2.png: no row: UnknownResolutionError" in message and "; and 1 more." in message
    assert "give --ppi, --ppi_manifest or --assume_ppi" in message
    assert tables(tmp_path) == []


@pytest.mark.parametrize("workers", [1, 2])
def test_a_build_is_given_up_when_none_of_its_first_files_could_be_loaded(
    builder, monkeypatch, tmp_path, capsys, workers
):
    """
    What keeps every file from loading is a mistake of the options (here:
    no resolution is known). A build of millions of files must not load them
    all to say so: it stops after its first files, here after 3 of 7.
    """
    monkeypatch.setattr(builder, "FILES_BEFORE_GIVING_UP", 3)
    names = [f"{subject}_{impression}.png" for subject in (1, 2, 3) for impression in (1, 2)]
    for name in names + ["4_1.png"]:
        save(tmp_path / "images" / name, ppi=None)
    loaded = []
    load_sample = builder.load_sample

    def counting(source, subject, impression, image_path):
        loaded.append(image_path)
        return load_sample(source, subject, impression, image_path)

    monkeypatch.setattr(builder, "load_sample", counting)
    before = set(multiprocessing.active_children())
    message = refused(builder, monkeypatch, capsys, tmp_path, "--workers", workers)
    assert "None of the first 3 of the 7 listed files could be loaded as an image" in message
    assert "the build was given up and no table was written" in message
    assert "1_1.png: no row: UnknownResolutionError: Resolution unknown" in message
    assert "1_2.png: no row: UnknownResolutionError" in message and "; and 1 more." in message
    assert "give --ppi, --ppi_manifest or --assume_ppi" in message
    assert tables(tmp_path) == []
    assert set(multiprocessing.active_children()) == before
    if workers == 1:
        # The files after the third were not loaded at all
        assert loaded == names[:3]


def test_one_image_among_the_first_files_that_loads_lets_the_build_go_on(
    builder, monkeypatch, tmp_path
):
    """The third of five files is an image with a resolution: the build is not given up."""
    monkeypatch.setattr(builder, "FILES_BEFORE_GIVING_UP", 3)
    for name in ("1_1.png", "1_2.png", "2_2.png", "3_1.png"):
        save(tmp_path / "images" / name, ppi=None)
    save(tmp_path / "images" / "2_1.png")
    assert build(builder, monkeypatch, tmp_path) == 1
    assert paths(read_samples(tmp_path / "table")) == ["2_1.png"]
    assert sorted(log(tmp_path / "table", FAILURES_LOG)) == [
        "1_1.png",
        "1_2.png",
        "2_2.png",
        "3_1.png",
    ]


def test_a_build_of_no_more_files_than_the_limit_says_that_none_could_be_loaded(
    builder, monkeypatch, tmp_path, capsys
):
    """Then all files were loaded, and the refusal says so instead of naming the first ones."""
    monkeypatch.setattr(builder, "FILES_BEFORE_GIVING_UP", 3)
    for name in ("1_1.png", "1_2.png", "2_1.png"):
        save(tmp_path / "images" / name, ppi=None)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert "None of the 3 listed files could be loaded as an image" in message
    assert "given up" not in message
    assert tables(tmp_path) == []


def test_what_the_table_refuses_for_one_file_costs_that_file_and_not_the_table(
    builder, monkeypatch, tmp_path
):
    """
    The rules of the table are applied to every file when it is loaded. Here
    the preparation states, for two files, sizes and a resolution that the
    table refuses, as nothing but an error in it could: without the check per
    file the whole table would be refused when it is read at the end.
    """
    write_many(tmp_path, subjects=range(1, 4))
    load = SensorNormalizer.load

    def wrong(self, path, check_polarity=False):
        image, info = load(self, path, check_polarity)
        if path.endswith("2_1.png"):
            info = dataclasses.replace(info, width=info.width + 1)
        if path.endswith("3_2.png"):
            info = dataclasses.replace(info, ppi=float("nan"))
        return image, info

    monkeypatch.setattr(SensorNormalizer, "load", wrong)
    assert build(builder, monkeypatch, tmp_path) == 1
    samples = read_samples(tmp_path / "table")
    assert paths(samples) == ["sensor_1/1_1.png", "sensor_2/2_2.png", "sensor_0/3_1.png"]
    failures = log(tmp_path / "table", FAILURES_LOG)
    assert sorted(failures) == ["sensor_0/3_2.png", "sensor_1/1_2.png", "sensor_2/2_1.png"]
    message = failures["sensor_2/2_1.png"]
    assert message.startswith("no row: SampleTableError: A prepared size that does not follow")
    message = failures["sensor_0/3_2.png"]
    assert message.startswith("no row: SampleTableError: A ppi that is not a positive number")


def test_file_names_with_unusual_characters_are_rows_and_single_lines_of_the_log(
    builder, monkeypatch, tmp_path
):
    names = ["zeile\num bruch.png", "tab\tulator.png", "größe ü/指紋 1.png", "a b/c'd\"e.png"]
    for name in names:
        save(tmp_path / "images" / name)
    extracted(tmp_path, names[2])
    rows = "".join(f'"{name}",{number},0\n' for number, name in enumerate(names[:3]))
    options = with_list(tmp_path, HEADER + rows + '"a b/c\'d""e.png",3,0\n')
    options += ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder, monkeypatch, tmp_path, *options) == 1
    samples = read_samples(tmp_path / "table")
    assert paths(samples) == names
    assert samples.minutiae_known.tolist() == [False, False, True, False]
    with open(tmp_path / "table" / FAILURES_LOG, encoding="utf-8") as file:
        lines = file.read().split("\n")
    # One line per file: what would break a line is written as an escape sequence
    assert lines == [
        "zeile\\num bruch.png\tnull minutiae: the .mnt file is missing",
        "tab\\tulator.png\tnull minutiae: the .mnt file is missing",
        "a b/c'd\"e.png\tnull minutiae: the .mnt file is missing",
        "",
    ]


# ---------------------------------------------------------------------------
# Builds that are refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("existing", ["a table", "an empty directory", "a file", "a broken link"])
def test_an_output_directory_that_exists_refuses_the_build(
    builder, monkeypatch, tmp_path, capsys, existing
):
    write_pair(tmp_path)
    if existing == "a table":
        assert build(builder, monkeypatch, tmp_path) == 0
    elif existing == "an empty directory":
        (tmp_path / "table").mkdir()
    elif existing == "a file":
        (tmp_path / "table").write_text("something else")
    else:
        os.symlink(tmp_path / "nowhere", tmp_path / "table")
    before = snapshot(tmp_path)
    forbid_loading(builder, monkeypatch)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert f"{tmp_path / 'table'} exists" in message
    assert "give a directory that does not exist yet as --samples" in message
    assert "or remove the old one first" in message
    # Nothing was touched, and no temporary directory was made
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "names, expected",
    [
        (["1_1.png", "a/readme.png"], ["1 of the 2 image files", "a/readme.png: it is not named"]),
        (["1_1.png", "1_1_1.png"], ["1_1_1.png: it is not named <subject>_<impression>"]),
        (["1_1.png", "0_1.png", "1_0.png"], ["2 of the 3", "0_1.png: it is", "1_0.png: it is"]),
        (["1_1.png", "a_1.png"], ["a_1.png: it is not named"]),
        (["1_1.png", "5.png"], ["5.png: it is not named"]),
        (["1_1.png", "2147483649_1.png"], ["the subject 2147483648 is above 2147483647"]),
        (["1_1.png", "1_99999999999.png"], ["the impression 99999999998 is above 2147483647"]),
        (["1_1.png", "sensor\\a/2_1.png"], ["sensor\\a/2_1.png: its path holds a backslash"]),
        (["1_1.png", b"3_1\xff.png"], ["3_1\\xff.png: its path is not valid UTF-8"]),
        (
            [f"{name}.png" for name in "abcdefgh"],
            ["8 of the 8 image files", "a.png: it is", "e.png: it is", "; and 3 more"],
        ),
    ],
)
def test_file_names_that_a_table_cannot_take_refuse_the_build(
    builder, monkeypatch, tmp_path, capsys, names, expected
):
    for name in names:
        if isinstance(name, bytes):  # a name that is not UTF-8, as a file system can hold it
            name = os.fsdecode(name)
        save(tmp_path / "images" / name)
    before = snapshot(tmp_path / "images")
    forbid_loading(builder, monkeypatch)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    for part in expected:
        assert part in message, message
    # What to do: the form of the names, and the way for other names
    assert "named <subject>_<impression> with whole numbers from 1" in message
    assert "17_3.png becomes subject 16, impression 2" in message
    assert "--sample_list" in message and "docs/sample-table.md" in message
    assert tables(tmp_path) == []
    assert snapshot(tmp_path / "images") == before


def test_files_that_cannot_be_told_apart_refuse_the_build(builder, monkeypatch, tmp_path, capsys):
    forbid_loading(builder, monkeypatch)
    # Two files with the same name in two directories: the same subject and impression
    for name in ("sensor_a/17_3.png", "sensor_b/17_3.png", "sensor_b/4_1.png"):
        save(tmp_path / "images" / name)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert "1 pair(s) of image files have the same subject and impression" in message
    assert "subject 16, impression 2: sensor_a/17_3.png and sensor_b/17_3.png" in message
    assert "rename one of them" in message and "--sample_list" in message
    assert tables(tmp_path) == []

    # Two images that differ only in their ending: one .mnt file for both
    os.remove(tmp_path / "images" / "sensor_b" / "17_3.png")
    save(tmp_path / "images" / "sensor_b" / "4_1.tif")
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert "1 pair(s) of image files differ only in their ending" in message
    assert "sensor_b/4_1.png and sensor_b/4_1.tif" in message
    assert "share one .mnt file" in message and "Rename one file of each pair" in message
    assert tables(tmp_path) == []


def test_a_directory_without_image_files_refuses_the_build(builder, monkeypatch, tmp_path, capsys):
    (tmp_path / "images" / "sensor_a").mkdir(parents=True)
    (tmp_path / "images" / "sensor_a" / "1_1.mnt").write_text("0\n8 8\n")
    (tmp_path / "images" / "notes.txt").write_text("no image")
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert f"No image file was found below {tmp_path / 'images'}" in message
    assert ".png, .jpg, .jpeg, .bmp, .tif, .tiff" in message and "--dataset_dir" in message
    assert tables(tmp_path) == []


@pytest.mark.parametrize("closed", [["b", "c/deep"], ["."]])
def test_a_directory_of_images_that_cannot_be_listed_refuses_the_build(
    builder, monkeypatch, tmp_path, capsys, closed
):
    """
    Such a directory is not passed over: the table would lack its files
    without a word. The tests may run as a user for whom no directory is
    closed, so the listing of the directories fails here as it does for a
    directory that the user may not read.
    """
    images = tmp_path / "images"
    for name in ("a/1_1.png", "b/2_1.png", "c/3_1.png", "c/deep/4_1.png"):
        save(images / name)
    closed = [os.path.normpath(images / name) for name in closed]
    scandir = os.scandir

    def scan(path):
        if path in closed:
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    # Control: the listing of the extraction leaves out the files of these directories
    listed = list(builder.iter_image_paths(str(images)))
    assert listed == ([] if closed == [str(images)] else ["a/1_1.png", "c/3_1.png"])

    with monkeypatch.context() as patch:
        forbid_loading(builder, patch)
        message = refused(builder, patch, capsys, tmp_path)
    assert f"{images} cannot be listed completely" in message
    for directory in closed:
        assert f"{directory} (PermissionError: Permission denied)" in message
    assert "Make these directories readable" in message and "--sample_list" in message
    assert "No image file was found" not in message
    assert tables(tmp_path) == []

    # A sample list names the files itself: no directory is listed for it
    options = with_list(tmp_path, HEADER + "a/1_1.png,0,0\n")
    assert build(builder, monkeypatch, tmp_path, *options) == 0
    assert paths(read_samples(tmp_path / "table")) == ["a/1_1.png"]


# ---------------------------------------------------------------------------
# The settings of the extraction
# ---------------------------------------------------------------------------


def write_extraction(tmp_path, **settings) -> SensorNormalizer:
    """
    1000 ppi prints without resolution metadata, extracted with --ppi 1000,
    and the settings file that scripts/extract_minutiae.py stores (the
    settings of the preparation and two of its own). Returns the normalizer
    of the extraction.
    """
    normalizer = SensorNormalizer(ppi=1000)
    for name in ("1_1.png", "1_2.png"):
        save(tmp_path / "images" / name, ridge_image(320, 256, 19), ppi=None)
        extracted(tmp_path, name, normalizer)
    stored = dict(normalizer.settings(), image_dpi=500, classified=False)
    stored.update(settings)
    with open(tmp_path / "minutiae" / "extraction_settings.json", "w") as file:
        file.write(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    return normalizer


def test_the_options_of_the_extraction_build_a_table(builder, monkeypatch, tmp_path, capsys):
    normalizer = write_extraction(tmp_path)
    before = snapshot(tmp_path / "minutiae")
    options = ["--minutiae_dir", tmp_path / "minutiae", "--ppi", 1000]
    assert build(builder, monkeypatch, tmp_path, *options) == 0
    samples = read_samples(tmp_path / "table")
    assert np.diff(samples.minutiae_offsets).tolist() == [3, 3]
    assert samples.build["settings"] == normalizer.settings()
    settings_file = tmp_path / "minutiae" / "extraction_settings.json"
    assert f"Settings: those of the extraction ({settings_file})" in printed(capsys)
    # The settings file is read and nothing else: the directory is as it was
    assert snapshot(tmp_path / "minutiae") == before


def test_other_settings_than_those_of_the_extraction_refuse_the_build(
    builder, monkeypatch, tmp_path, capsys
):
    write_extraction(tmp_path)
    before = snapshot(tmp_path / "minutiae"), snapshot(tmp_path / "images")
    forbid_loading(builder, monkeypatch)
    options = ["--minutiae_dir", tmp_path / "minutiae", "--assume_ppi", 500, "--ridges", "auto"]
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    # The keys with both values
    assert f"{tmp_path / 'minutiae' / 'extraction_settings.json'} holds other settings" in message
    assert " ppi is 1000.0 there and null here" in message
    assert " assume_ppi is null there and 500.0 here" in message
    assert ' ridges is "dark" there and "auto" here' in message
    # What is equal is not named, and neither are the settings that the builder does not have
    for name in ("target_ppi", "ppi_manifest_", "image_dpi", "classified"):
        assert name not in message
    # Both ways out
    assert "give this script the options that the extraction was given" in message
    assert "(--ppi, --ppi_manifest, --assume_ppi, --ridges)" in message
    assert "pass --allow_other_extraction_settings" in message
    assert tables(tmp_path) == []
    assert (snapshot(tmp_path / "minutiae"), snapshot(tmp_path / "images")) == before


def test_other_settings_can_be_allowed_and_the_checks_per_file_decide(
    builder, monkeypatch, tmp_path, capsys
):
    write_extraction(tmp_path)
    save(tmp_path / "images" / "1_2.png", ridge_image(320, 256, 19), ppi=500)
    before = snapshot(tmp_path / "minutiae")
    options = ["--minutiae_dir", tmp_path / "minutiae", "--assume_ppi", 1000]
    assert build(builder, monkeypatch, tmp_path, *options, "--allow_other_extraction_settings") == 1
    output = " ".join(capsys.readouterr().out.split())
    assert "Note: " in output and "holds other settings than this build uses" in output
    assert " ppi is 1000.0 there and null here" in output
    samples = read_samples(tmp_path / "table")
    # The file that is prepared as the extraction prepared it has its minutiae. The file
    # that now states 500 ppi is prepared in another size than its .mnt file says.
    assert samples.minutiae_known.tolist() == [True, False]
    message = log(tmp_path / "table", FAILURES_LOG)["1_2.png"]
    assert "states an image of 128x160 pixels, and the prepared image has 256x320" in message
    assert snapshot(tmp_path / "minutiae") == before


@pytest.mark.parametrize(
    "stored, expected",
    [
        ({"ppi_manifest_sha256": "0" * 64}, ' ppi_manifest_sha256 is "0000'),
        ({"ppi_manifest_entries": 3}, " ppi_manifest_entries is 3 there and 0 here"),
        ({"target_ppi": 1000.0}, " target_ppi is 1000.0 there and 500.0 here"),
        ({"assume_ppi": 1000.0}, " assume_ppi is 1000.0 there and null here"),
        ({"ridges": "light"}, ' ridges is "light" there and "dark" here'),
        ({"ppi": None}, " ppi is null there and 1000.0 here"),
        ({"ppi_manifest_entries": False}, " ppi_manifest_entries is false there and 0 here"),
    ],
)
def test_every_setting_of_the_preparation_is_compared(
    builder, monkeypatch, tmp_path, capsys, stored, expected
):
    write_extraction(tmp_path, **stored)
    options = ["--minutiae_dir", tmp_path / "minutiae", "--ppi", 1000]
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert expected in message, message
    assert message.count(" there and ") == 1  # what is equal is not named


def test_a_setting_that_the_extraction_did_not_store_counts_as_another(
    builder, monkeypatch, tmp_path, capsys
):
    write_extraction(tmp_path)
    path = tmp_path / "minutiae" / "extraction_settings.json"
    stored = json.loads(path.read_text())
    del stored["assume_ppi"]
    path.write_text(json.dumps(stored))
    options = ["--minutiae_dir", tmp_path / "minutiae", "--ppi", 1000]
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert " assume_ppi is not stored there and null here" in message


def test_a_missing_settings_file_is_said_and_the_build_goes_on(
    builder, monkeypatch, tmp_path, capsys
):
    write_extraction(tmp_path)
    os.remove(tmp_path / "minutiae" / "extraction_settings.json")
    before = snapshot(tmp_path / "minutiae")
    options = ["--minutiae_dir", tmp_path / "minutiae", "--ppi", 1000]
    assert build(builder, monkeypatch, tmp_path, *options) == 0
    notes = [line for line in printed(capsys) if line.startswith("Note: ")]
    assert len(notes) == 1  # one line
    assert f"{tmp_path / 'minutiae' / 'extraction_settings.json'} is missing" in notes[0]
    assert "cannot be compared" in notes[0]
    assert np.diff(read_samples(tmp_path / "table").minutiae_offsets).tolist() == [3, 3]
    # The builder does not write the file that is missing
    assert snapshot(tmp_path / "minutiae") == before


@pytest.mark.parametrize(
    "content",
    ["{ no JSON", "[1, 2]", "", pytest.param("[" * 100_000, id="brackets without end")],
)
def test_a_settings_file_that_cannot_be_read_refuses_the_build(
    builder, monkeypatch, tmp_path, capsys, content
):
    write_extraction(tmp_path)
    (tmp_path / "minutiae" / "extraction_settings.json").write_text(content)
    options = ["--minutiae_dir", tmp_path / "minutiae", "--ppi", 1000]
    with monkeypatch.context() as patch:
        forbid_loading(builder, patch)
        message = refused(builder, patch, capsys, tmp_path, *options)
    assert "extraction_settings.json cannot be read (" in message
    assert "restore it, or remove it if it is damaged" in message
    assert "pass --allow_other_extraction_settings" in message
    assert tables(tmp_path) == []

    assert build(builder, monkeypatch, tmp_path, *options, "--allow_other_extraction_settings") == 0
    assert "extraction_settings.json cannot be read (" in " ".join(capsys.readouterr().out.split())
    assert read_samples(tmp_path / "table").minutiae_known.all()


# ---------------------------------------------------------------------------
# Nothing is left behind
# ---------------------------------------------------------------------------


def test_a_table_gets_its_name_only_when_it_is_complete(builder, monkeypatch, tmp_path):
    write_many(tmp_path, subjects=range(1, 4))
    seen = []
    load_sample = builder.load_sample

    def watching(*arguments):
        # While the images are loaded there is a temporary directory next to the table, and no table
        seen.append(tables(tmp_path))
        return load_sample(*arguments)

    monkeypatch.setattr(builder, "load_sample", watching)
    assert build(builder, monkeypatch, tmp_path) == 1
    assert seen == [[f"table.tmp{os.getpid()}"]] * 6
    assert tables(tmp_path) == ["table"]


@pytest.fixture
def signals_of_a_terminal():
    """
    SIGINT and SIGHUP as a program has them that was started in a terminal:
    not ignored. The tests themselves may run in a process that ignores one
    of them (a background job of a shell script ignores SIGINT, a process
    that was started with nohup ignores SIGHUP), and so does every program
    that such a process starts; a build leaves an ignored SIGHUP ignored.
    While a test runs, a signal that is ignored is taken by a handler here:
    a program that the test starts then has the signal at its default, and a
    build in this process finds a signal that is not ignored.
    """
    handlers = {signal.SIGINT: signal.default_int_handler, signal.SIGHUP: lambda *_: None}
    ignored = [number for number in handlers if signal.getsignal(number) is signal.SIG_IGN]
    for number in ignored:
        signal.signal(number, handlers[number])
    yield
    for number in ignored:
        signal.signal(number, signal.SIG_IGN)


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError, "SIGTERM", "SIGHUP"])
@pytest.mark.parametrize("workers", [1, 2])
def test_a_build_that_is_interrupted_or_fails_leaves_nothing_behind(
    builder, monkeypatch, tmp_path, capsys, signals_of_a_terminal, error, workers
):
    write_many(tmp_path)
    monkeypatch.setattr(builder, "FILES_PER_PART", 4)
    before = snapshot(tmp_path, with_itself=False)
    # A signal: what kill and a job scheduler send, or what a build gets
    # when its terminal is closed
    number = getattr(signal, error) if isinstance(error, str) else None
    handlers = [signal.getsignal(taken) for taken in (signal.SIGTERM, signal.SIGHUP)]
    children = multiprocessing.active_children()
    write_part = builder.write_part
    written = []

    def failing(directory, index, *arguments):
        written.append(index)
        if index == 1:  # the first part is written, the second is not
            assert f"table.tmp{os.getpid()}" in os.listdir(tmp_path / "new")
            if number is None:
                raise error("in the middle of the build")
            # Only if the build takes the signal: with the handler of before
            # it would end the tests, or do nothing
            assert signal.getsignal(number) not in handlers
            os.kill(os.getpid(), number)
        return write_part(directory, index, *arguments)

    monkeypatch.setattr(builder, "write_part", failing)
    arguments = ["--dataset_dir", tmp_path / "images", "--samples", tmp_path / "new" / "table"]
    with pytest.raises(SystemExit if number else error) as raised:
        run(builder, monkeypatch, *arguments, "--workers", workers)
    if number:
        assert raised.value.code == 128 + number
    else:
        assert str(raised.value) == "in the middle of the build"
    assert written == [0, 0, 1]  # control: the trial part, the first part, the part that failed
    # No table, no temporary directory, and not the directory that was made for the table
    assert snapshot(tmp_path, with_itself=False) == before
    assert "The build did not finish: no table was written" in capsys.readouterr().err
    # The handlers of the signals are those that were there, and the workers are ended
    assert [signal.getsignal(taken) for taken in (signal.SIGTERM, signal.SIGHUP)] == handlers
    assert multiprocessing.active_children() == children


def test_a_build_that_was_started_to_outlive_its_terminal_goes_on_when_it_is_closed(
    builder, monkeypatch, tmp_path
):
    """nohup starts a program with SIGHUP ignored, and a build leaves it ignored."""
    write_pair(tmp_path)
    load_sample = builder.load_sample
    sent = []

    def hung_up(*arguments):
        # Only if the build left the signal ignored: its own handler would end the build here
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        os.kill(os.getpid(), signal.SIGHUP)
        sent.append(signal.SIGHUP)
        return load_sample(*arguments)

    monkeypatch.setattr(builder, "load_sample", hung_up)
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        assert build(builder, monkeypatch, tmp_path) == 0
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)
    assert len(sent) == 2 and len(read_samples(tmp_path / "table")) == 2


def test_an_output_directory_that_appears_during_the_build_is_not_replaced(
    builder, monkeypatch, tmp_path, capsys
):
    write_pair(tmp_path)
    load_sample = builder.load_sample

    def meanwhile(*arguments):
        os.makedirs(tmp_path / "table", exist_ok=True)  # something else makes the directory
        return load_sample(*arguments)

    monkeypatch.setattr(builder, "load_sample", meanwhile)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert f"{tmp_path / 'table'} was made by something else while the table was built" in message
    assert "build it again" in message
    # What is there is left as it is, and the table that was built is not kept
    assert tables(tmp_path) == ["table"] and os.listdir(tmp_path / "table") == []


@pytest.mark.parametrize(
    "fault, expected",
    [
        (
            "a part that lost its end",
            ["does not pass its own check, and it was not kept", "cannot be read as a Parquet"],
        ),
        ("a part without its rows", ["holds 0 rows of 2 files where 2 rows of 2 files were"]),
        ("a part of other files", ["holds 2 rows of 3 files where 2 rows of 2 files were"]),
    ],
)
def test_a_table_that_does_not_come_back_as_it_was_written_is_not_published(
    builder, monkeypatch, tmp_path, capsys, fault, expected
):
    """
    At its end a build reads its table as a reader does, and only a table
    that passes gets its name. Here the part is not what the build wrote,
    as on a disk that does not keep what it is given.
    """
    write_pair(tmp_path)
    write_part = builder.write_part

    def faulty(directory, index, count, samples, build):
        if len(samples) == 0:  # the part without rows that a build tries pyarrow with
            return write_part(directory, index, count, samples, build)
        if fault == "a part without its rows":
            samples = samples_from_rows([])
        if fault == "a part of other files":
            build = dict(build, files=build["files"] + 1)
        path = write_part(directory, index, count, samples, build)
        if fault == "a part that lost its end":
            os.truncate(path, os.path.getsize(path) - 4)
        return path

    monkeypatch.setattr(builder, "write_part", faulty)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    for part in expected:
        assert part in message, message
    if fault == "a part that lost its end":
        # The message of the reader, and what to do about a build of which nothing is left
        assert f"{tmp_path / 'table'}.tmp{os.getpid()}/part-00000-of-00001.parquet" in message
        assert "Of this build nothing is left to copy or to read again" in message
        assert "Build the table again." in message
    else:
        assert "It was not kept." in message
    assert tables(tmp_path) == []


def test_a_table_is_written_also_where_what_it_holds_cannot_be_printed(
    builder, monkeypatch, tmp_path, capsys
):
    """
    The reader of the output can go away while a build runs, as the reader
    of a pipe. The table has its name before its summary is printed, so the
    status says that it is written: 1 here, for the image without a .mnt file.
    """
    write_pair(tmp_path)

    def no_reader(directory, samples):
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr(builder, "describe_table", no_reader)
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    assert len(read_samples(tmp_path / "table")) == 2
    errors = " ".join(capsys.readouterr().err.split())
    table = tmp_path / "table"
    assert f"The table {table} is written, but what it holds could not be printed" in errors
    assert f"(BrokenPipeError: [Errno 32] Broken pipe). To see it: --check {table}" in errors


def test_a_table_is_written_also_where_its_summary_fails_for_another_reason(
    builder, monkeypatch, tmp_path, capsys
):
    """Not an error of the system: a name that the output cannot encode, for example."""
    write_pair(tmp_path)

    def not_encoded(directory, samples):
        raise UnicodeEncodeError("ascii", "t\xe4ble", 1, 2, "ordinal not in range(128)")

    monkeypatch.setattr(builder, "describe_table", not_encoded)
    assert build(builder, monkeypatch, tmp_path) == 0
    assert len(read_samples(tmp_path / "table")) == 2
    errors = " ".join(capsys.readouterr().err.split())
    assert "is written, but what it holds could not be printed (UnicodeEncodeError: " in errors
    assert f"To see it: --check {tmp_path / 'table'}" in errors


def test_a_table_is_written_also_where_nothing_can_be_printed_any_more(
    builder, monkeypatch, tmp_path
):
    """
    A build can outlive its terminal: it was started to go on when the
    terminal is closed, and its output still goes there. Then nothing can be
    written any more: not what the table holds, and not that it could not be
    printed. The status still says that the table is written, and here that
    no file failed.
    """
    write_pair(tmp_path)
    describe_table = builder.describe_table
    closed = []

    class Terminal:
        """Takes what is written to it until it is closed, and refuses it from then on."""

        refused = 0

        def write(self, text):
            if closed:
                self.refused += 1
                raise OSError(errno.EIO, "Input/output error")
            return len(text)

        def flush(self):
            pass

    def closing(directory, samples):
        closed.append(directory)  # the terminal goes when the table has its name
        return describe_table(directory, samples)

    monkeypatch.setattr(builder, "describe_table", closing)
    output, errors = Terminal(), Terminal()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    assert build(builder, monkeypatch, tmp_path) == 0
    # Control: the table had its name, and neither stream took what was written to it then
    assert closed == [str(tmp_path / "table")]
    assert output.refused and errors.refused
    assert len(read_samples(tmp_path / "table")) == 2


def test_a_temporary_directory_that_is_there_already_is_left_as_it_is(
    builder, monkeypatch, tmp_path, capsys
):
    """
    It is the work of another build: one that runs under the same process
    number (on another machine that writes to the same disk), or one that
    was killed. The build that finds it is refused and removes nothing.
    """
    write_pair(tmp_path)
    theirs = tmp_path / f"table.tmp{os.getpid()}"
    theirs.mkdir()
    (theirs / "part-00000-of-00002.parquet").write_text("the work of another build")
    (theirs / FAILURES_LOG).write_text("its log\n")
    before = snapshot(tmp_path)
    forbid_loading(builder, monkeypatch)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert f"The temporary directory {theirs}, in which the table would be built, exists" in message
    assert f"another build into {tmp_path / 'table'} is running, or one was killed" in message
    assert "It was not touched. If no build is running, remove it" in message
    assert snapshot(tmp_path) == before


def test_a_temporary_directory_that_cannot_be_made_is_not_taken_for_one_that_exists(
    builder, monkeypatch, tmp_path, capsys
):
    """
    The directory in which the table is to lie cannot be written to. The
    tests may run as a user for whom no directory is closed, so the making
    of the temporary directory fails here as it does for a user who may not
    write there.
    """
    write_pair(tmp_path)
    before = snapshot(tmp_path)
    forbid_loading(builder, monkeypatch)
    temporary = f"{tmp_path / 'table'}.tmp{os.getpid()}"
    mkdir = os.mkdir

    def closed(path, *arguments, **options):
        if str(path) == temporary:
            raise PermissionError(errno.EACCES, "Permission denied", temporary)
        return mkdir(path, *arguments, **options)

    monkeypatch.setattr(os, "mkdir", closed)
    message = refused(builder, monkeypatch, capsys, tmp_path)
    assert f"The temporary directory {temporary}, in which the table is built, cannot" in message
    assert "cannot be made (PermissionError: [Errno 13] Permission denied" in message
    assert f"{tmp_path} must be a directory that can be written to" in message
    # Not the refusal of a temporary directory that is there: nothing is to be removed
    assert "exists already" not in message and "remove" not in message
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("below", ["notes", "notes/new"])
def test_a_table_below_a_file_is_refused_and_the_file_is_left_alone(
    builder, monkeypatch, tmp_path, capsys, below
):
    write_pair(tmp_path)
    (tmp_path / "notes").write_text("a file, not a directory")
    before = snapshot(tmp_path)
    forbid_loading(builder, monkeypatch)
    options = ["--samples", tmp_path / below / "table"]
    message = refused(builder, monkeypatch, capsys, tmp_path, *options)
    assert f"The directory {tmp_path / below}, in which the table is to be written" in message
    assert "cannot be made (" in message
    assert "Give as --samples a path in a directory that exists or can be made" in message
    # Nothing in the message tells the user to remove what is there
    assert "remove" not in message
    assert snapshot(tmp_path) == before


def test_directories_above_the_table_are_made_and_removed_again_if_no_table_is_written(
    builder, monkeypatch, tmp_path
):
    write_pair(tmp_path)
    (tmp_path / "tables").mkdir()
    deep = tmp_path / "tables" / "new" / "deeper" / "table"
    save(tmp_path / "images" / "readme.png")  # a build that is refused after it has begun
    assert build(builder, monkeypatch, tmp_path, "--samples", deep) == 2
    assert os.listdir(tmp_path / "tables") == []
    os.remove(tmp_path / "images" / "readme.png")
    assert build(builder, monkeypatch, tmp_path, "--samples", deep) == 0
    assert len(read_samples(deep)) == 2
    assert os.listdir(tmp_path / "tables" / "new" / "deeper") == ["table"]


# ---------------------------------------------------------------------------
# Checking a table
# ---------------------------------------------------------------------------


def summary(lines: list) -> list:
    """The lines of the summary: from the first line of = signs to the last count of ridges."""
    start = lines.index("=" * 60)
    counts = [i for i, line in enumerate(lines) if re.match(r"[\d,]+ (dark|light) ridges, ", line)]
    return lines[start : counts[-1] + 1]


def test_check_prints_the_summary_of_the_build(builder, monkeypatch, tmp_path, capsys):
    write_many(tmp_path)
    monkeypatch.setattr(builder, "FILES_PER_PART", 5)
    assert build(builder, monkeypatch, tmp_path, "--minutiae_dir", tmp_path / "minutiae") == 1
    built = printed(capsys)
    before = snapshot(tmp_path)

    assert run(builder, monkeypatch, "--check", tmp_path / "table") == 0
    checked = printed(capsys)
    assert checked == summary(built)
    samples = read_samples(tmp_path / "table")
    for line in [
        f"Sample table: {tmp_path / 'table'}",
        "Files listed: 14",
        "Rows: 12",
        "Files that failed: 6 (2 without a row, 4 with null minutiae)",
        "Rows with minutiae: 4 (7 minutiae)",
        "Rows with empty minutiae: 4",
        "Rows with null minutiae: 4 (the .mnt file was missing or refused)",
        "Subjects: 7",
        "Impressions per subject: smallest 1, median 2, largest 2",
        "Parts: 3",
        f"Checksum (samples_sha256): {samples_checksum(samples)}",
        f"Failures log: {tmp_path / 'table' / FAILURES_LOG} (6 line(s))",
        f"Warnings log: {tmp_path / 'table' / WARNINGS_LOG} (4 line(s))",
        "12 500 ppi from metadata of the file",
        "12 dark ridges, from --ridges",
    ]:
        assert line in checked, line
    assert snapshot(tmp_path) == before  # a check changes nothing

    # A table that was copied without its logs: the check says what is missing
    os.remove(tmp_path / "table" / FAILURES_LOG)
    os.remove(tmp_path / "table" / WARNINGS_LOG)
    assert run(builder, monkeypatch, "--check", tmp_path / "table") == 0
    checked = printed(capsys)
    path = tmp_path / "table" / FAILURES_LOG
    assert f"Failures log: missing, although 6 file(s) failed ({path})" in checked
    assert "Warnings log: none" in checked


def test_check_refuses_what_is_no_table(builder, monkeypatch, tmp_path, capsys):
    write_many(tmp_path)
    monkeypatch.setattr(builder, "FILES_PER_PART", 5)
    assert build(builder, monkeypatch, tmp_path) == 1

    def check(directory) -> str:
        capsys.readouterr()
        assert run(builder, monkeypatch, "--check", directory) == 2
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith("ERROR: ")
        return captured.err

    # A directory that is no table, a path that is no directory, and one that is not there
    message = check(tmp_path / "images")
    assert "holds no part of a sample table" in message
    assert "scripts/build_sample_table.py" in message
    for path in (tmp_path / "table" / "part-00000-of-00003.parquet", tmp_path / "nowhere"):
        assert "cannot be read as the directory of a sample table" in check(path)

    # A table that lacks a part
    os.remove(tmp_path / "table" / "part-00001-of-00003.parquet")
    message = check(tmp_path / "table")
    assert "is not a whole table: it holds 2 of its 3 parts" in message
    assert "missing: part-00001-of-00003.parquet" in message

    # A part that is no Parquet file
    (tmp_path / "table" / "part-00001-of-00003.parquet").write_bytes(b"PAR1 and nothing else")
    assert "cannot be read as a Parquet file" in check(tmp_path / "table")


def test_check_names_columns_that_this_code_does_not_know(builder, monkeypatch, tmp_path, capsys):
    import pyarrow as pa
    import pyarrow.parquet as pq

    write_pair(tmp_path)
    assert build(builder, monkeypatch, tmp_path) == 0
    capsys.readouterr()
    path = tmp_path / "table" / "part-00000-of-00001.parquet"
    table = pq.read_table(path).append_column("person", pa.array(["anna", "anna"]))
    pq.write_table(table.append_column("finger", pa.array([2, 2], type=pa.int8())), path)
    assert run(builder, monkeypatch, "--check", tmp_path / "table") == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "Rows: 2" in output
    assert "Note: the table holds 2 column(s) that this code does not know" in output
    assert "does not use: person, finger." in output


def test_check_reads_a_table_that_this_script_did_not_build(builder, monkeypatch, tmp_path, capsys):
    """Written with the module alone: such a table records no files and no failures."""
    row = SampleRow(
        subject=3,
        impression=1,
        image_path="a/b.png",
        file_size=100,
        ppi=1000.0,
        ppi_origin="manifest",
        ridges="light",
        ridges_origin="manifest",
        source_width=20,
        source_height=24,
        width=10,
        height=12,
        minutiae=np.zeros((0, 3), dtype=np.float32),
    )
    write_part(str(tmp_path / "table"), 0, 1, samples_from_rows([row]), {"id": "another tool"})
    assert run(builder, monkeypatch, "--check", tmp_path / "table") == 0
    lines = printed(capsys)
    for line in [
        "Files listed: not recorded",
        "Rows: 1",
        "Files that failed: not recorded",
        "Rows with minutiae: 0",
        "Rows with empty minutiae: 1",
        "Rows with null minutiae: 0",
        "Failures log: none",
        "1 1000 ppi from manifest (resampled to 500 ppi)",
        "1 light ridges, from the manifest",
    ]:
        assert line in lines, line


def test_the_note_about_metadata_counts_the_files_of_all_resolutions(
    builder, monkeypatch, tmp_path, capsys
):
    """
    A table with more resolutions than the summary lists. The note about the
    files that took the resolution of their own metadata, although the build
    was given an assumption, counts all of them, also those whose resolution
    is not among the listed ones, and it stands behind the line about the
    resolutions that are not listed.
    """
    rows = ["path,ppi"]
    for subject in range(1, 26):
        for impression in (1, 2):
            name = f"{subject}_{impression}.png"
            if subject <= 23:
                # A resolution of its own for each of 23 subjects, by a manifest
                save(tmp_path / "images" / name, ppi=None)
                rows.append(f"{name},{400 + subject}")
            else:
                # Two subjects whose files state 300 and 600 ppi themselves
                save(tmp_path / "images" / name, ppi=300 * (subject - 23))
    (tmp_path / "resolutions.csv").write_text("\n".join(rows) + "\n")
    options = ["--ppi_manifest", tmp_path / "resolutions.csv", "--assume_ppi", 500]
    assert build(builder, monkeypatch, tmp_path, *options) == 0
    built = printed(capsys)
    assert run(builder, monkeypatch, "--check", tmp_path / "table") == 0
    for lines in (built, printed(capsys)):
        first = lines.index("Resolution of the 50 image file(s) of the table:")
        # Twenty of the 25 resolutions, which have two files each, are listed
        assert lines[first + 1] == "2 401 ppi from manifest (resampled to 500 ppi)"
        assert lines[first + 20] == "2 420 ppi from manifest (resampled to 500 ppi)"
        assert lines[first + 21] == "10 with 5 further resolutions"
        assert lines[first + 22].startswith(
            "Note: 4 file(s) took the resolution from their own metadata, not from "
            "--assume_ppi 500."
        )
        assert lines[first + 23] == "Ridge polarity:"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_options_that_do_not_fit_are_refused_before_anything_is_done(
    builder, monkeypatch, tmp_path, capsys
):
    write_pair(tmp_path)
    forbid_loading(builder, monkeypatch)
    images, table = tmp_path / "images", tmp_path / "table"
    both = ["--dataset_dir", images, "--samples", table]

    def error(*arguments) -> str:
        message = usage_error(builder, monkeypatch, capsys, *arguments)
        assert tables(tmp_path) == []
        return message

    assert "a build needs --dataset_dir and --samples" in error()
    assert "(missing: --dataset_dir, --samples)" in error()
    assert "(missing: --samples)" in error("--dataset_dir", images)
    assert "to check a table that exists, give --check alone" in error("--samples", table)
    assert "--workers must be at least 1" in error(*both, "--workers", 0)
    message = error("--dataset_dir", tmp_path / "nowhere", "--samples", table)
    assert f"--dataset_dir {tmp_path / 'nowhere'} is not a directory" in message
    message = error(*both, "--minutiae_dir", tmp_path / "nowhere")
    assert f"--minutiae_dir {tmp_path / 'nowhere'} is not a directory" in message
    message = error(*both, "--sample_list", tmp_path / "nowhere.csv")
    assert f"--sample_list {tmp_path / 'nowhere.csv'} is not a file" in message
    message = error(*both, "--allow_other_extraction_settings")
    assert "--allow_other_extraction_settings has no meaning without --minutiae_dir" in message
    assert "ppi must be a positive number" in error(*both, "--ppi", -500)
    message = error(*both, "--ppi_manifest", tmp_path / "nowhere.csv")
    assert f"--ppi_manifest {tmp_path / 'nowhere.csv'} cannot be read" in message
    (tmp_path / "manifest.csv").write_text("file,resolution\n1_1.png,500\n")
    assert "column 'path'" in error(*both, "--ppi_manifest", tmp_path / "manifest.csv")

    # Nothing is written into the directories of the images and of the minutiae
    message = error("--dataset_dir", images, "--samples", images / "table")
    assert "lies in --dataset_dir" in message and "give a directory outside of them" in message
    # ... also where the path leads into one of them through a link
    os.symlink(images, tmp_path / "link")
    message = error("--dataset_dir", images, "--samples", tmp_path / "link" / "table")
    assert f"--samples {tmp_path / 'link' / 'table'} lies in --dataset_dir {images}" in message
    minutiae = tmp_path / "minutiae"
    inside = ["--samples", minutiae / "new" / "table", "--minutiae_dir", minutiae]
    assert "lies in --minutiae_dir" in error("--dataset_dir", images, *inside)
    assert not (minutiae / "new").exists()


def test_check_goes_with_no_other_option(builder, monkeypatch, tmp_path, capsys):
    write_pair(tmp_path)
    assert build(builder, monkeypatch, tmp_path) == 0
    for others, named in [
        (["--dataset_dir", tmp_path / "images"], "--dataset_dir"),
        (["--samples", tmp_path / "other"], "--samples"),
        # Also where an option is given the value that it has anyway
        (["--workers", 1], "--workers"),
        (["--ridges", "dark", "--assume_ppi", 500], "--assume_ppi, --ridges"),
        (["--allow_other_extraction_settings"], "--allow_other_extraction_settings"),
    ]:
        message = usage_error(builder, monkeypatch, capsys, "--check", tmp_path / "table", *others)
        assert f"--check goes with no other option (given: {named})" in message
    assert tables(tmp_path) == ["table"]


def test_every_option_has_a_help_text_that_states_its_default(builder, monkeypatch, capsys):
    with pytest.raises(SystemExit) as leaving:
        run(builder, monkeypatch, "--help")
    assert leaving.value.code == 0
    text = capsys.readouterr().out
    assert "docs/sample-table.md" in text
    # The options of the script itself; those of the preparation are described by its module
    own = text[: text.index("image preparation:")].split("\n  --")[1:]
    assert [block.split()[0] for block in own] == [
        "dataset_dir",
        "samples",
        "minutiae_dir",
        "sample_list",
        "workers",
        "allow_other_extraction_settings",
        "check",
    ]
    for block in own:
        assert "(default: " in " ".join(block.split()), block
    for option in ("--ppi PPI", "--ppi_manifest", "--assume_ppi", "--ridges {dark,light,auto}"):
        assert option in text


def test_the_script_runs_as_a_program_from_any_directory(tmp_path):
    """With worker processes, and with flx found although the working directory is another one."""
    pytest.importorskip("pyarrow")
    write_many(tmp_path)

    def program(*arguments):
        return subprocess.run(
            [sys.executable, SCRIPT] + [str(argument) for argument in arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    result = program("--dataset_dir", "images", "--samples", "table", "--workers", 2)
    assert result.returncode == 1, result.stderr  # the set holds images that cannot be loaded
    assert "Sample table: table" in result.stdout
    samples = read_samples(tmp_path / "table")
    assert len(samples) == 12 and samples.build["files"] == 14
    assert sorted(os.listdir(tmp_path)) == ["images", "minutiae", "table"]

    result = program("--check", "table")
    assert result.returncode == 0, result.stderr
    assert f"Checksum (samples_sha256): {samples_checksum(samples)}" in result.stdout


def test_an_error_that_is_no_refusal_ends_the_program_with_the_status_2(
    builder, monkeypatch, tmp_path
):
    """
    The status 1 stands for a table that was written, so it is not the
    status of an error that leaves none. Here a part cannot be written: the
    program may not write a file of its size, which stands for a disk that
    is full.
    """
    write_pair(tmp_path)
    rng = np.random.default_rng(5)
    many = rng.uniform((0, 0, 0), (144, 200, 6.28), (20_000, 3)).tolist()
    extracted(tmp_path, "1_2.png", minutiae=many)
    options = ["--minutiae_dir", tmp_path / "minutiae"]
    limit = 50_000
    # Control: the part is larger than the limit, and without a limit it is written
    assert build(builder, monkeypatch, tmp_path, *options, table="table_whole") == 0
    assert os.path.getsize(tmp_path / "table_whole" / "part-00000-of-00001.parquet") > 2 * limit

    arguments = ["--dataset_dir", "images", "--samples", "table", "--minutiae_dir", "minutiae"]
    code = (
        "import resource, runpy, sys\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({limit}, {limit}))\n"
        f"sys.argv = {[SCRIPT] + arguments!r}\n"
        f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n"
    )
    # -B: the interpreter writes no file of its own under the limit
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 2, result.stderr
    # Not a refusal: the error is printed as it was raised, after what became of the build
    assert "ERROR: " not in result.stderr
    assert "The build did not finish: no table was written" in result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    assert "OSError" in result.stderr.splitlines()[-1]
    assert tables(tmp_path) == ["table_whole"]


def test_an_error_that_cannot_be_printed_ends_the_program_with_the_status_2_as_well(tmp_path):
    """
    Standard error has no reader any more, as after "2>&1 | head". The build
    fails when it writes its progress, and neither that it did not finish
    nor the traceback of the error can be written, as they cannot in a
    terminal that was closed. The status is still not 1, which stands for a
    table that was written.
    """
    pytest.importorskip("pyarrow")
    write_pair(tmp_path)
    reader, writer = os.pipe()
    os.close(reader)
    # The progress bar is what the build writes to standard error: a setting
    # of the environment that switches the bar off (TQDM_DISABLE) is not passed on
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("TQDM_")
    }
    try:
        # -u: nothing waits in a buffer for the end of the interpreter, which
        # has a status of its own for what it cannot write then
        result = subprocess.run(
            [sys.executable, "-u", SCRIPT, "--dataset_dir", "images", "--samples", "table"],
            cwd=tmp_path,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=writer,
        )
    finally:
        os.close(writer)
    assert result.returncode == 2
    assert tables(tmp_path) == []


def test_an_error_of_any_kind_that_is_no_refusal_ends_the_program_with_the_status_2(tmp_path):
    """Not only an error of the system: here one that a function of the table module raises."""
    pytest.importorskip("pyarrow")
    write_pair(tmp_path)
    arguments = ["--dataset_dir", "images", "--samples", "table"]
    code = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {REPOSITORY!r})\n"
        "from flx.preparation import sample_table\n"
        "def not_foreseen(path):\n"
        "    raise RuntimeError('not foreseen')\n"
        "sample_table.normalize_image_path = not_foreseen\n"
        f"sys.argv = {[SCRIPT] + arguments!r}\n"
        f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 2, result.stderr
    assert "ERROR: " not in result.stderr
    assert "The build did not finish: no table was written" in result.stderr
    assert result.stderr.splitlines()[-1] == "RuntimeError: not foreseen"
    assert tables(tmp_path) == []


def children(pid: int) -> list:
    """The processes that run the script as children of a process: its worker processes."""
    found = []
    for entry in os.listdir("/proc"):
        try:
            with open(f"/proc/{entry}/stat") as file:
                parent = int(file.read().rpartition(")")[2].split()[1])
            with open(f"/proc/{entry}/cmdline", "rb") as file:
                if parent == pid and b"build_sample_table" in file.read():
                    found.append(int(entry))
        except (OSError, ValueError):
            pass  # no process, or one that has ended meanwhile
    return found


def running(pids: list) -> list:
    """Those of the processes that have not ended."""
    found = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as file:
                if file.read().rpartition(")")[2].split()[0] != "Z":  # Z: ended, not yet collected
                    found.append(pid)
        except OSError:
            pass
    return found


def wait_until(condition, seconds: float = 60) -> bool:
    """
    Asks the condition again and again until it is true or the time is
    over, and says whether it became true. A condition that was true is not
    asked once more: it need not stay true (a pipe can be written to as long
    as its reader waits, and no longer).
    """
    end = time.monotonic() + seconds
    while True:
        if condition():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.05)


@pytest.fixture
def stuck_program(tmp_path, signals_of_a_terminal):
    """
    The script as a program with two workers, in the middle of a build that
    gets no further: one of its image files is a pipe that nothing writes to,
    and a worker waits for it. Gives (the process, its two workers).
    """
    pytest.importorskip("pyarrow")
    if not os.path.isdir("/proc/self"):
        pytest.skip("the worker processes are found in /proc")
    for subject in range(2, 9):
        save(tmp_path / "images" / f"{subject}_1.png")
    os.mkfifo(tmp_path / "images" / "1_1.png")
    command = [sys.executable, SCRIPT, "--dataset_dir", tmp_path / "images", "--workers", "2"]
    process = subprocess.Popen(
        [str(argument) for argument in command + ["--samples", tmp_path / "table"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # a process group of its own, as a program in a terminal has
    )
    workers = []
    try:
        wait_until(lambda: len(children(process.pid)) == 2 or process.poll() is not None)
        workers = children(process.pid)
        assert len(workers) == 2, "the program did not get as far as its two workers"
        time.sleep(0.5)  # the workers have taken their files, and one of them the pipe
        assert tables(tmp_path) == [f"table.tmp{process.pid}"]
        yield process, workers
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)  # the program and every worker it started
        except ProcessLookupError:
            pass  # they have all ended
        process.communicate()


def test_an_interruption_from_the_keyboard_ends_the_program_and_its_workers(
    stuck_program, tmp_path
):
    process, workers = stuck_program
    # A terminal sends the signal to the whole process group. The workers
    # leave it to the main process to end them: alone, they go on.
    for pid in workers:
        os.kill(pid, signal.SIGINT)
    time.sleep(0.5)
    assert running(workers) == workers and process.poll() is None
    os.killpg(process.pid, signal.SIGINT)
    _, errors = process.communicate(timeout=60)
    # Ended by the signal, with the traceback of the main process and no other
    assert process.returncode == -signal.SIGINT
    assert errors.count("KeyboardInterrupt") == 1
    assert "The build did not finish: no table was written" in errors
    assert tables(tmp_path) == []
    assert running(workers) == []


def test_a_terminal_that_is_closed_ends_the_program_and_its_workers(stuck_program, tmp_path):
    process, workers = stuck_program
    # The signal of a closed terminal can reach the whole process group, as
    # an interruption does. The workers leave it to the main process as well.
    for pid in workers:
        os.kill(pid, signal.SIGHUP)
    time.sleep(0.5)
    assert running(workers) == workers and process.poll() is None
    os.killpg(process.pid, signal.SIGHUP)
    _, errors = process.communicate(timeout=60)
    # Ended as by the signal, without a traceback, and nothing is left
    assert process.returncode == 128 + signal.SIGHUP
    assert "Traceback" not in errors
    assert "The build did not finish: no table was written" in errors
    assert tables(tmp_path) == []
    assert running(workers) == []


def test_the_workers_end_by_themselves_when_the_program_is_killed(stuck_program, tmp_path):
    """
    A program that is killed can neither end its workers nor remove its
    temporary directory. A worker notices that the main process is gone by
    its pipe to it, which is closed then: the worker without a file at once,
    the other one when it has loaded its file.
    """
    process, workers = stuck_program
    process.kill()
    process.wait()
    assert wait_until(lambda: len(running(workers)) == 1, 30), running(workers)

    def written() -> bool:
        """Writes the image file that the other worker waits for (as an empty file)."""
        try:
            os.close(os.open(tmp_path / "images" / "1_1.png", os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            return False  # the pipe has no reader: the worker does not wait for it yet
        return True

    assert wait_until(written, 30)
    assert wait_until(lambda: running(workers) == [], 30), running(workers)
    assert tables(tmp_path) == [f"table.tmp{process.pid}"]  # what a killed build leaves


# ---------------------------------------------------------------------------
# Without pyarrow
# ---------------------------------------------------------------------------


def test_without_pyarrow_a_build_stops_before_an_image_is_loaded(
    builder_without_pyarrow, monkeypatch, tmp_path, capsys
):
    write_pair(tmp_path)
    before = snapshot(tmp_path, with_itself=False)
    forbid_loading(builder_without_pyarrow, monkeypatch)
    options = ["--minutiae_dir", tmp_path / "minutiae"]
    assert build(builder_without_pyarrow, monkeypatch, tmp_path, *options) == 2
    message = capsys.readouterr().err
    assert message.startswith("ERROR: ") and "needs the package pyarrow" in message
    assert 'pip install -e ".[table]"' in message
    assert f'pip install "{sample_table.PYARROW_REQUIREMENT}"' in message
    assert snapshot(tmp_path, with_itself=False) == before


def test_without_pyarrow_a_check_says_how_to_install_it(
    builder_without_pyarrow, monkeypatch, tmp_path, capsys
):
    (tmp_path / "table").mkdir()
    assert run(builder_without_pyarrow, monkeypatch, "--check", tmp_path / "table") == 2
    message = capsys.readouterr().err
    assert "needs the package pyarrow" in message and 'pip install -e ".[table]"' in message


def test_without_pyarrow_the_script_is_loaded_and_checks_its_options():
    """In a new interpreter in which pyarrow cannot be imported from its start."""
    code = (
        "import importlib.util, sys\n"
        "sys.modules['pyarrow'] = None\n"
        f"spec = importlib.util.spec_from_file_location('build_sample_table', {SCRIPT!r})\n"
        "script = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(script)\n"
        "sys.argv = ['build_sample_table.py', '--check', 'nowhere', '--workers', '2']\n"
        "script.main()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPOSITORY, capture_output=True, text=True
    )
    assert result.returncode == 2, result.stderr
    assert "--check goes with no other option (given: --workers)" in result.stderr
    assert "Traceback" not in result.stderr
