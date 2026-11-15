"""
Tests for the minutiae extraction scripts. The extractor (fingerflow package of
the minutiae_extractor repository) is replaced by a stand-in, so that no model
weights are needed.
"""

import importlib.util
import sys
import types
from os.path import join, dirname

import cv2
import numpy as np
import pytest

from flx.data.fingerflow_minutia_loader import read_mnt_file

pd = pytest.importorskip("pandas")

SCRIPTS_DIR = join(dirname(dirname(__file__)), "scripts")
TEST_IMAGE_DIR = join(dirname(__file__), "data", "SFingeExample")
# Size of the test images. The width is not a multiple of 8.
WIDTH = 292
HEIGHT = 392

MINUTIAE = pd.DataFrame(
    {
        "x": [100.0, 250.5],
        "y": [150.0, 380.25],
        "angle": [0.5, 6.0],
        "score": [0.9, 0.6],
        "class": [0.0, 1.0],
        "direction_confidence": [0.8, 0.1],
    }
)


class FakeExtractor:
    """Has the interface of fingerflow.extractor.Extractor"""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.images = []
        FakeExtractor.instances.append(self)

    def extract_minutiae(self, image):
        self.images.append(image)
        return {"core": pd.DataFrame(), "minutiae": MINUTIAE.copy()}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, join(SCRIPTS_DIR, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _assert_is_extractor_output(mnt_path: str) -> None:
    with open(mnt_path) as file:
        lines = file.read().splitlines()
    assert lines[0] == str(len(MINUTIAE))
    # Size of the full image, not trimmed to a multiple of 8
    assert lines[1] == f"{WIDTH} {HEIGHT}"
    rows = np.array([[float(v) for v in line.split()] for line in lines[2:]])
    expected = MINUTIAE[["x", "y", "angle", "class", "direction_confidence"]]
    assert rows == pytest.approx(expected.to_numpy(), abs=1e-4)

    locs, oris, size = read_mnt_file(mnt_path)
    assert size == (WIDTH, HEIGHT)
    assert locs == pytest.approx(MINUTIAE[["x", "y"]].to_numpy(), abs=1e-4)
    assert oris == pytest.approx(MINUTIAE["angle"].to_numpy(), abs=1e-4)


def _assert_got_full_grayscale_image(extractor: FakeExtractor) -> None:
    (image,) = extractor.images
    assert image.shape == (HEIGHT, WIDTH)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("dpi_args, image_dpi", [([], 500), (["--image_dpi", "400"], 400)])
def test_extract_minutiae(tmp_path, monkeypatch, fake_fingerflow, dpi_args, image_dpi):
    script = _load_script("extract_minutiae")
    output_dir = join(tmp_path, "minutiae")
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_minutiae.py", "--input_dir", TEST_IMAGE_DIR, "--output_dir", output_dir]
        + ["--coarse_net", "coarse", "--fine_net", "fine"]
        + ["--classify_net", "classify", "--core_net", "core"]
        + dpi_args,
    )
    assert script.main() == 0

    # One extractor for all images
    (extractor,) = fake_fingerflow.instances
    assert extractor.args == ("coarse", "fine", "classify", "core")
    assert extractor.kwargs == {"image_dpi": image_dpi}
    assert len(extractor.images) == 8
    assert not (tmp_path / "minutiae" / "extraction_errors.txt").exists()
    _assert_is_extractor_output(join(output_dir, "1_1.mnt"))


MODEL_PATHS = {k: k for k in ["coarse_net", "fine_net", "classify_net", "core_net"]}


def test_extract_minutiae_single_image(tmp_path, fake_fingerflow):
    script = _load_script("extract_minutiae")
    image_path = join(TEST_IMAGE_DIR, "1_1.png")
    mnt_path = join(tmp_path, "1_1.mnt")
    result = script.extract_single_image((image_path, mnt_path, MODEL_PATHS, 500))
    assert result == (image_path, True, None)
    _assert_got_full_grayscale_image(fake_fingerflow.instances[0])
    _assert_is_extractor_output(mnt_path)


def _extract_minutiae_argv(input_dir: str, output_dir: str) -> list[str]:
    return (
        ["extract_minutiae.py", "--input_dir", input_dir, "--output_dir", output_dir]
        + ["--coarse_net", "coarse", "--fine_net", "fine"]
        + ["--classify_net", "classify", "--core_net", "core"]
    )


def test_extract_minutiae_reports_failed_images(tmp_path, monkeypatch, fake_fingerflow):
    script = _load_script("extract_minutiae")
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    image = cv2.imread(join(TEST_IMAGE_DIR, "1_1.png"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(input_dir / "1_1.png"), image)
    (input_dir / "1_2.png").write_bytes(b"not an image")
    output_dir = tmp_path / "minutiae"
    monkeypatch.setattr(
        sys, "argv", _extract_minutiae_argv(str(input_dir), str(output_dir))
    )
    assert script.main() == 1

    # The other image is extracted, the failed one is in the error log
    _assert_is_extractor_output(str(output_dir / "1_1.mnt"))
    assert not (output_dir / "1_2.mnt").exists()
    error_log = (output_dir / "extraction_errors.txt").read_text().splitlines()
    assert len(error_log) == 1
    assert error_log[0].startswith(str(input_dir / "1_2.png"))


def test_extract_minutiae_stops_without_extractor(tmp_path, monkeypatch, fake_fingerflow):
    """E.g. a fingerflow version whose Extractor has no image_dpi option"""
    attempts = []

    class ExtractorWithoutImageDpi:
        def __init__(self, *args, **kwargs):
            attempts.append(kwargs)
            raise TypeError("unexpected keyword argument 'image_dpi'")

    monkeypatch.setattr(
        sys.modules["fingerflow.extractor"], "Extractor", ExtractorWithoutImageDpi
    )
    script = _load_script("extract_minutiae")
    output_dir = tmp_path / "minutiae"
    monkeypatch.setattr(
        sys, "argv", _extract_minutiae_argv(TEST_IMAGE_DIR, str(output_dir))
    )
    with pytest.raises(TypeError, match="image_dpi"):
        script.main()
    # Not tried again for each of the 8 images
    assert attempts == [{"image_dpi": 500}]
    assert not list(output_dir.glob("*.mnt"))

    image_path = join(TEST_IMAGE_DIR, "1_1.png")
    with pytest.raises(KeyError):
        script.extract_single_image((image_path, join(tmp_path, "1_1.mnt"), {}, 500))


def test_run_minutiae_extractor_process_image(tmp_path, fake_fingerflow):
    script = _load_script("run_minutiae_extractor")
    extractor = fake_fingerflow()
    mnt_path = join(tmp_path, "1_1.mnt")
    df = script.process_image(extractor, join(TEST_IMAGE_DIR, "1_1.png"), mnt_path)
    assert len(df) == len(MINUTIAE)
    _assert_got_full_grayscale_image(extractor)
    _assert_is_extractor_output(mnt_path)


@pytest.mark.parametrize("dpi_args, image_dpi", [([], 500), (["--image_dpi", "400"], 400)])
def test_run_minutiae_extractor(tmp_path, monkeypatch, fake_fingerflow, dpi_args, image_dpi):
    script = _load_script("run_minutiae_extractor")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_files = ["CoarseNet.h5", "FineNet.h5", "ClassifyNet_6_classes.h5", "CoreNet.weights"]
    for model_file in model_files:
        (models_dir / model_file).touch()
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    image = cv2.imread(join(TEST_IMAGE_DIR, "1_1.png"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(input_dir / "a.bmp"), image)
    output_dir = join(tmp_path, "minutiae")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_minutiae_extractor.py", "--input_dir", str(input_dir)]
        + ["--output_dir", output_dir, "--models_dir", str(models_dir)]
        + dpi_args,
    )
    script.main()

    (extractor,) = fake_fingerflow.instances
    assert extractor.args == tuple(str(models_dir / f) for f in model_files)
    assert extractor.kwargs == {"image_dpi": image_dpi}
    _assert_got_full_grayscale_image(extractor)
    _assert_is_extractor_output(join(output_dir, "a.mnt"))
