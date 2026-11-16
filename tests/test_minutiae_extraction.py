"""
Tests for the minutiae extraction over images from different sensors and
large file lists: scripts/extract_minutiae.py and
flx/preparation/minutiae_extraction.py. The extractor (fingerflow package of
the minutiae_extractor repository) is replaced by a stand-in that reports the
dark square drawn into each test image as its one minutia.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from os.path import dirname, join

import numpy as np
import pytest
import torch
from PIL import Image

from flx.data.dataset import Identifier
from flx.data.fingerflow_minutia_loader import FingerFlowMinutiaMapLoader, read_mnt_file
from flx.preparation.minutiae_extraction import (
    SettingsChangedError,
    check_settings,
    iter_image_paths,
    select_shard,
    write_mnt_file,
)

pd = pytest.importorskip("pandas")

REPOSITORY = dirname(dirname(__file__))
SCRIPT = join(REPOSITORY, "scripts", "extract_minutiae.py")

# Size of the test print at 500 ppi and the centre of its marker
WIDTH_500, HEIGHT_500 = 200, 260
MARKER_500 = (60.0, 150.0)  # x, y


def marker_image(ppi: float, marker=MARKER_500, light_ridges: bool = False) -> np.ndarray:
    """
    The same physical print at the given resolution: white with a dark square
    of 0.04 inch around the marker position.
    """
    scale = ppi / 500
    width, height = int(round(WIDTH_500 * scale)), int(round(HEIGHT_500 * scale))
    y, x = np.mgrid[0:height, 0:width]
    # Pixel centres in 500 ppi units
    x_500, y_500 = (x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5
    square = (np.abs(x_500 - marker[0]) <= 10) & (np.abs(y_500 - marker[1]) <= 10)
    image = np.where(square, 0, 255).astype(np.uint8)
    return 255 - image if light_ridges else image


def save(path, image: np.ndarray, ppi=None) -> str:
    os.makedirs(dirname(str(path)), exist_ok=True)
    if ppi is None:
        Image.fromarray(image).save(str(path))
    else:
        Image.fromarray(image).save(str(path), dpi=(ppi, ppi))
    return str(path)


def marker_minutiae(image: np.ndarray, classify: bool = True):
    """The centre of the dark pixels as the one minutia of the image."""
    ys, xs = np.nonzero(image < 128)
    table = pd.DataFrame(
        {"x": [xs.mean()], "y": [ys.mean()], "angle": [1.0], "score": [0.9]}
    )
    table["class"] = 1.0 if classify else np.nan
    table["direction_confidence"] = 0.5
    return table


class FakeExtractor:
    """Has the interface of fingerflow.extractor.Extractor"""

    instances = []

    def __init__(self, coarse_net, fine_net, classify_net, core_net, image_dpi=500):
        self.classify_net = classify_net
        self.core_net = core_net
        self.images = []
        FakeExtractor.instances.append(self)

    def extract_minutiae(self, image):
        self.images.append(image)
        if (image < 128).all():
            raise RuntimeError("the extractor can not work on this image")
        return {"core": None, "minutiae": marker_minutiae(image, self.classify_net is not None)}


@pytest.fixture
def fake_fingerflow(monkeypatch):
    fingerflow = types.ModuleType("fingerflow")
    extractor = types.ModuleType("fingerflow.extractor")
    extractor.Extractor = FakeExtractor
    fingerflow.extractor = extractor
    monkeypatch.setitem(sys.modules, "fingerflow", fingerflow)
    monkeypatch.setitem(sys.modules, "fingerflow.extractor", extractor)
    monkeypatch.setattr(FakeExtractor, "instances", [])
    return FakeExtractor


def run_script(monkeypatch, input_dir, output_dir, *options) -> int:
    spec = importlib.util.spec_from_file_location("extract_minutiae", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_minutiae.py", "--input_dir", str(input_dir), "--output_dir", str(output_dir)]
        + ["--coarse_net", "coarse", "--fine_net", "fine"]
        + [str(option) for option in options],
    )
    return script.main()


def images_seen(fake) -> int:
    return sum(len(extractor.images) for extractor in fake.instances)


# ---------------------------------------------------------------------------
# Resolution and polarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ppi", [500, 1000, 385])
def test_minutiae_are_extracted_at_500_ppi(tmp_path, monkeypatch, fake_fingerflow, ppi):
    """The extractor gets the 500 ppi image, and the .mnt file is in its pixels."""
    save(tmp_path / "images" / "1_1.png", marker_image(ppi), ppi=ppi)
    assert run_script(monkeypatch, tmp_path / "images", tmp_path / "minutiae") == 0

    (image,) = fake_fingerflow.instances[0].images
    assert abs(image.shape[0] - HEIGHT_500) <= 1 and abs(image.shape[1] - WIDTH_500) <= 1
    locs, _, (width, height) = read_mnt_file(str(tmp_path / "minutiae" / "1_1.mnt"))
    assert (height, width) == image.shape
    assert locs[0] == pytest.approx(MARKER_500, abs=0.75)
    last_line = (tmp_path / "minutiae" / "1_1.mnt").read_text().splitlines()[-1]
    assert last_line.startswith("# source ") and f"at {ppi} ppi (metadata)" in last_line


def test_a_file_of_unknown_resolution_fails_and_the_run_goes_on(
    tmp_path, monkeypatch, fake_fingerflow
):
    save(tmp_path / "images" / "1_1.png", marker_image(500), ppi=500)
    save(tmp_path / "images" / "1_2.png", marker_image(1000))  # no resolution in the file
    save(tmp_path / "images" / "1_3.png", marker_image(500), ppi=500)
    output_dir = tmp_path / "minutiae"
    assert run_script(monkeypatch, tmp_path / "images", output_dir) == 1

    assert sorted(p.name for p in output_dir.glob("*.mnt")) == ["1_1.mnt", "1_3.mnt"]
    (line,) = (output_dir / "extraction_errors.txt").read_text().splitlines()
    path, message = line.split("\t")
    assert path == str(tmp_path / "images" / "1_2.png")
    assert "Resolution unknown" in message and "--assume_ppi" in message
    # The extractor was not run on an image of unknown resolution
    assert images_seen(fake_fingerflow) == 2


def test_plausible_metadata_wins_over_the_assumption_and_the_run_says_so(
    tmp_path, monkeypatch, fake_fingerflow, capsys
):
    # 500 ppi prints; an image program stamped 300 into two of them
    save(tmp_path / "images" / "1_1.png", marker_image(500), ppi=300)
    save(tmp_path / "images" / "1_2.png", marker_image(500), ppi=300)
    save(tmp_path / "images" / "1_3.png", marker_image(500))
    out = tmp_path / "minutiae"
    assert run_script(monkeypatch, tmp_path / "images", out, "--assume_ppi", 500) == 0
    output = capsys.readouterr().out
    assert "Resolution of the files extracted in this run:" in output
    lines = [" ".join(line.split()) for line in output.splitlines()]
    assert "2 300 ppi from metadata of the file (resampled to 500 ppi)" in lines
    assert "1 500 ppi from --assume_ppi" in lines
    assert "2 file(s) took the resolution from their own metadata" in output
    assert "does not override" in output

    # --ppi is the option for a corpus of one resolution: no file is resampled
    capsys.readouterr()
    out = tmp_path / "minutiae_ppi"
    assert run_script(monkeypatch, tmp_path / "images", out, "--ppi", 500) == 0
    output = capsys.readouterr().out
    assert "3 500 ppi from --ppi" in [" ".join(l.split()) for l in output.splitlines()]
    assert "Note:" not in output
    for extractor in fake_fingerflow.instances[-1:]:
        assert all(image.shape == (HEIGHT_500, WIDTH_500) for image in extractor.images)


def test_resolution_from_the_argument_and_from_the_manifest(
    tmp_path, monkeypatch, fake_fingerflow
):
    images = tmp_path / "images"
    save(images / "a" / "1_1.png", marker_image(1000))
    save(images / "b" / "2_1.png", marker_image(385))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,ppi\na,1000\nb/2_1.png,385\n")
    assert run_script(monkeypatch, images, tmp_path / "m1", "--ppi_manifest", manifest) == 0
    for name in ("a/1_1.mnt", "b/2_1.mnt"):
        locs, _, (width, height) = read_mnt_file(str(tmp_path / "m1" / name))
        assert abs(width - WIDTH_500) <= 1 and abs(height - HEIGHT_500) <= 1
        assert locs[0] == pytest.approx(MARKER_500, abs=0.75)

    # --ppi is for all files: right for a, wrong for b
    assert run_script(monkeypatch, images, tmp_path / "m2", "--ppi", 1000) == 0
    _, _, (width, _) = read_mnt_file(str(tmp_path / "m2" / "a" / "1_1.mnt"))
    assert width == WIDTH_500
    _, _, (width, _) = read_mnt_file(str(tmp_path / "m2" / "b" / "2_1.mnt"))
    assert width == 77  # 154 pixels taken as 1000 ppi


def test_light_ridges_are_inverted_and_a_wrong_setting_is_reported(
    tmp_path, monkeypatch, fake_fingerflow
):
    images = tmp_path / "images"
    # A print with ridges in the middle of a white image
    y, x = np.mgrid[0:HEIGHT_500, 0:WIDTH_500]
    print_area = (np.abs(x - 100) < 60) & (np.abs(y - 130) < 80)
    ridges = 127.5 + 127.5 * np.cos(2 * np.pi * (x + 0.3 * y) / 9.5)
    dark_ridges = np.where(print_area, ridges, 255).astype(np.uint8)
    save(images / "1_1.png", 255 - dark_ridges, ppi=500)
    save(images / "1_2.png", dark_ridges, ppi=500)

    # Declared per file: no warning, both images reach the extractor with dark ridges
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,ridges\n1_1.png,light\n")
    out = tmp_path / "m1"
    assert run_script(monkeypatch, images, out, "--ppi_manifest", manifest) == 0
    for image in fake_fingerflow.instances[0].images:
        assert np.array_equal(image, dark_ridges)
    assert not (out / "extraction_warnings.txt").exists()
    assert "ridges light (manifest)" in (out / "1_1.mnt").read_text()

    # Not declared: the image is extracted as it is and the check reports it
    out = tmp_path / "m2"
    assert run_script(monkeypatch, images, out) == 0
    (line,) = (out / "extraction_warnings.txt").read_text().splitlines()
    assert line.startswith(str(images / "1_1.png") + "\t")
    assert "looks like light ridges" in line


# ---------------------------------------------------------------------------
# ClassifyNet is optional
# ---------------------------------------------------------------------------


def test_classify_net_is_optional(tmp_path, monkeypatch, fake_fingerflow):
    images = tmp_path / "images"
    save(images / "1_1.png", marker_image(500), ppi=500)
    assert run_script(monkeypatch, images, tmp_path / "plain") == 0
    assert run_script(
        monkeypatch, images, tmp_path / "classified", "--classify_net", "c", "--core_net", "d"
    ) == 0
    plain, classified = fake_fingerflow.instances
    assert (plain.classify_net, plain.core_net) == (None, None)
    # CoreNet is never loaded: the cores are not written
    assert (classified.classify_net, classified.core_net) == ("c", None)

    rows = {
        name: (tmp_path / name / "1_1.mnt").read_text().splitlines()[2].split()
        for name in ("plain", "classified")
    }
    assert rows["plain"][3] == "-1" and rows["classified"][3] == "1"
    assert rows["plain"][:3] == rows["classified"][:3]
    assert rows["plain"][4] == rows["classified"][4]

    # The loader reads both, and the minutia map does not depend on the class
    maps = [
        FingerFlowMinutiaMapLoader(str(tmp_path / name)).get(Identifier(0, 0))
        for name in ("plain", "classified")
    ]
    assert maps[0][1] == maps[1][1] == 1.0
    assert torch.equal(maps[0][0], maps[1][0])
    assert maps[0][0].max() > 0.4


# ---------------------------------------------------------------------------
# Very large file lists: continue, shards, failures
# ---------------------------------------------------------------------------


def make_dataset(images, count: int = 7) -> list[str]:
    names = []
    for i in range(count):
        name = join(f"sensor_{i % 2}", f"{i + 1}_1.png")
        save(images / name, marker_image(500, marker=(40.0 + 10 * i, 100.0)), ppi=500)
        names.append(name)
    return sorted(names)


def mnt_files(output_dir) -> list[str]:
    return sorted(
        os.path.relpath(join(directory, name), output_dir)
        for directory, _, files in os.walk(output_dir)
        for name in files
        if name.endswith(".mnt")
    )


def as_mnt(names) -> list[str]:
    return sorted(os.path.splitext(name)[0] + ".mnt" for name in names)


def test_skip_existing_continues_a_run(tmp_path, monkeypatch, fake_fingerflow):
    images, out = tmp_path / "images", tmp_path / "minutiae"
    names = make_dataset(images)
    # A run that was stopped: two finished files and one that was being written
    for name in names[:2]:
        write_mnt_file(str(out / name).replace(".png", ".mnt"), marker_minutiae(marker_image(500)), 1, 1)
    half_written = str(out / names[2]).replace(".png", ".mnt") + ".tmp123"
    with open(half_written, "w") as file:
        file.write("1\n")
    finished = (out / as_mnt(names)[0]).read_text()

    assert run_script(monkeypatch, images, out, "--skip_existing") == 0
    assert images_seen(fake_fingerflow) == len(names) - 2
    assert mnt_files(out) == as_mnt(names)
    assert (out / as_mnt(names)[0]).read_text() == finished  # not extracted again

    # Everything is finished: nothing is extracted
    assert run_script(monkeypatch, images, out, "--skip_existing") == 0
    assert images_seen(fake_fingerflow) == len(names) - 2

    # Without the option every file is extracted again
    assert run_script(monkeypatch, images, out) == 0
    assert images_seen(fake_fingerflow) == 2 * len(names) - 2
    assert (out / as_mnt(names)[0]).read_text() != finished


def test_failures_are_logged_and_tried_again(tmp_path, monkeypatch, fake_fingerflow):
    images, out = tmp_path / "images", tmp_path / "minutiae"
    names = make_dataset(images)
    bad = [images / "sensor_0" / "9_1.png", images / "sensor_1" / "9_2.png"]
    bad[0].write_bytes(b"not an image")
    save(bad[1], np.zeros((50, 50), dtype=np.uint8), ppi=500)  # the extractor raises

    assert run_script(monkeypatch, images, out) == 1  # the exit status tells
    assert mnt_files(out) == as_mnt(names)  # every other file is done
    log = dict(line.split("\t") for line in (out / "extraction_errors.txt").read_text().splitlines())
    assert set(log) == {str(path) for path in bad}
    assert "Could not read the file as an image" in log[str(bad[0])]
    assert "RuntimeError: the extractor can not work on this image" in log[str(bad[1])]

    # One file is repaired: the next run tries only the two failed files
    save(bad[0], marker_image(500), ppi=500)
    seen = images_seen(fake_fingerflow)
    assert run_script(monkeypatch, images, out, "--skip_existing") == 1
    assert images_seen(fake_fingerflow) == seen + 2
    (line,) = (out / "extraction_errors.txt").read_text().splitlines()
    assert line.startswith(str(bad[1]) + "\t")

    # The other one is repaired too: exit status 0 and no error log
    save(bad[1], marker_image(500), ppi=500)
    assert run_script(monkeypatch, images, out, "--skip_existing") == 0
    assert not (out / "extraction_errors.txt").exists()
    assert len(mnt_files(out)) == len(names) + 2


def test_shards_hold_every_file_exactly_once(tmp_path, monkeypatch, fake_fingerflow):
    images = tmp_path / "images"
    names = make_dataset(images)
    (images / "sensor_0" / "broken.png").write_bytes(b"not an image")
    all_names = sorted(names + [join("sensor_0", "broken.png")])
    assert list(iter_image_paths(str(images))) == all_names

    per_shard = []
    for shard in range(3):
        out = tmp_path / f"shard{shard}"
        status = run_script(
            monkeypatch, images, out, "--num_shards", 3, "--shard_index", shard
        )
        expected = all_names[shard::3]
        log = out / f"extraction_errors.shard-{shard:05d}-of-00003.txt"
        # Only the shard with the broken file fails, and its log has its own name
        assert status == (1 if join("sensor_0", "broken.png") in expected else 0)
        assert log.exists() == (status == 1)
        assert not (out / "extraction_errors.txt").exists()
        assert mnt_files(out) == as_mnt(n for n in expected if "broken" not in n)
        per_shard.append(mnt_files(out))
    assert sorted(sum(per_shard, [])) == as_mnt(names)

    assert list(select_shard("abcdefg", 3, 1)) == ["b", "e"]
    with pytest.raises(ValueError):
        list(select_shard("abc", 3, 3))
    with pytest.raises(SystemExit):
        run_script(monkeypatch, images, tmp_path / "x", "--num_shards", 3, "--shard_index", 3)


def test_file_list(tmp_path, monkeypatch, fake_fingerflow):
    images, out = tmp_path / "images", tmp_path / "minutiae"
    names = make_dataset(images)
    file_list = tmp_path / "files.txt"
    file_list.write_text("# my list\n" + "\n".join(names[:3] + ["sensor_0/missing.png"]) + "\n\n")
    assert run_script(monkeypatch, images, out, "--file_list", file_list) == 1
    assert mnt_files(out) == as_mnt(names[:3])
    (line,) = (out / "extraction_errors.txt").read_text().splitlines()
    assert line.startswith(str(images / "sensor_0" / "missing.png") + "\t")


def test_image_list_order_does_not_depend_on_the_file_system(tmp_path):
    for name in ["b/2.PNG", "a/1.tif", "a/0.bmp", "c.jpeg", "a/notes.txt", "a/x/9.png"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert list(iter_image_paths(str(tmp_path))) == [
        "c.jpeg", "a/0.bmp", "a/1.tif", "a/x/9.png", "b/2.PNG",
    ]


def test_one_directory_holds_one_setting(tmp_path, monkeypatch, fake_fingerflow, capsys):
    images, out = tmp_path / "images", tmp_path / "minutiae"
    names = make_dataset(images, count=2)
    assert run_script(monkeypatch, images, out, "--assume_ppi", 500) == 0
    stored = json.loads((out / "extraction_settings.json").read_text())
    assert stored["assume_ppi"] == 500 and stored["classified"] is False

    # The same settings again: fine. Other settings: the run does not start
    assert run_script(monkeypatch, images, out, "--assume_ppi", 500, "--skip_existing") == 0
    seen = images_seen(fake_fingerflow)
    assert run_script(monkeypatch, images, out, "--ppi", 1000) == 2
    assert "other settings (different: assume_ppi, ppi)" in capsys.readouterr().err
    assert images_seen(fake_fingerflow) == seen
    assert json.loads((out / "extraction_settings.json").read_text()) == stored

    # A full new extraction with the new settings
    assert run_script(monkeypatch, images, out, "--ppi", 1000, "--allow_settings_change") == 0
    assert json.loads((out / "extraction_settings.json").read_text())["ppi"] == 1000
    assert images_seen(fake_fingerflow) == seen + len(names)
    with pytest.raises(SystemExit):  # not together with files of the former settings
        run_script(monkeypatch, images, out, "--allow_settings_change", "--skip_existing")

    with pytest.raises(SettingsChangedError):
        check_settings(str(out), {"ppi": 500})


def test_mnt_file_appears_only_when_complete(tmp_path):
    path = tmp_path / "deep" / "er" / "1_1.mnt"
    write_mnt_file(str(path), marker_minutiae(marker_image(500)), 200, 260, "how\nprepared")
    assert os.listdir(path.parent) == ["1_1.mnt"]  # no temporary file is left
    lines = path.read_text().splitlines()
    assert lines[:2] == ["1", "200 260"] and lines[-1] == "# how prepared"

    # No minutiae: a finished file that the loader gives the weight 0
    write_mnt_file(str(path), pd.DataFrame([], columns=["x", "y", "angle", "score", "class"]), 200, 260)
    locs, oris, size = read_mnt_file(str(path))
    assert len(locs) == 0 and size == (200, 260)


FAKE_FINGERFLOW = '''
import os
import numpy as np
import pandas as pd


class Extractor:
    def __init__(self, coarse_net, fine_net, classify_net, core_net, image_dpi=500):
        pass

    def extract_minutiae(self, image):
        if image.shape[0] == 77:  # this image kills the worker process
            os._exit(1)
        ys, xs = np.nonzero(image < 128)
        table = pd.DataFrame({"x": [xs.mean()], "y": [ys.mean()], "angle": [1.0], "score": [0.9]})
        table["class"] = np.nan
        table["direction_confidence"] = 0.5
        return {"core": None, "minutiae": table}
'''


def test_worker_processes_and_a_worker_that_dies(tmp_path):
    """The script as a program with two worker processes and a stand-in extractor package."""
    package = tmp_path / "packages" / "fingerflow"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "extractor.py").write_text(textwrap.dedent(FAKE_FINGERFLOW))
    images, out = tmp_path / "images", tmp_path / "minutiae"
    names = make_dataset(images, count=30)
    (images / "sensor_0" / "broken.png").write_bytes(b"not an image")
    killer = images / "sensor_1" / "killer.png"
    save(killer, np.full((77, 60), 255, dtype=np.uint8), ppi=500)

    def run(*options):
        environment = dict(os.environ, PYTHONPATH=str(tmp_path / "packages"))
        command = [sys.executable, SCRIPT, "--input_dir", str(images), "--output_dir", str(out)]
        command += ["--coarse_net", "c", "--fine_net", "f", "--workers", "2", *options]
        return subprocess.run(command, env=environment, capture_output=True, text=True)

    result = run()
    assert result.returncode == 1, result.stderr
    failed = {
        line.split("\t")[0]: line.split("\t")[1]
        for line in (out / "extraction_errors.txt").read_text().splitlines()
    }
    assert "Could not read the file as an image" in failed[str(images / "sensor_0" / "broken.png")]
    assert "a worker process died" in failed[str(killer)]
    # Every file is either extracted or in the error log, and the run went on
    # after the worker died: the jobs that were handed out at that time (at
    # most 8 with 2 workers) are the only good files that can be missing
    done = set(mnt_files(out))
    not_done = set(as_mnt(names)) - done
    assert done <= set(as_mnt(names))
    assert {str(images / n.replace(".mnt", ".png")) for n in not_done} == set(failed) - {
        str(images / "sensor_0" / "broken.png"), str(killer)
    }
    assert len(not_done) <= 7
    assert len(done) >= len(names) - 7

    # Without the killing image, the next run finishes the rest
    killer.unlink()
    result = run("--skip_existing")
    assert result.returncode == 1, result.stderr  # the broken file is still there
    assert mnt_files(out) == as_mnt(names)
    (line,) = (out / "extraction_errors.txt").read_text().splitlines()
    assert "broken.png" in line
    assert f"Skipped:    {len(done)}" in result.stdout
