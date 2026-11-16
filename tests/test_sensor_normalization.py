"""
Tests for flx/preparation/sensor_normalization.py: where the resolution of an
image file comes from, the resampling to 500 ppi and the ridge polarity.
"""

import subprocess
import sys
from os.path import dirname, join

import cv2
import numpy as np
import pytest
from PIL import Image

from flx.preparation.sensor_normalization import (
    ImageReadError,
    SensorNormalizer,
    UnknownResolutionError,
    estimate_ridge_polarity,
    read_manifest,
    read_metadata_ppi,
    resample_to_ppi,
    resampled_size,
)

SFINGE_EXAMPLE = join(dirname(__file__), "data", "SFingeExample", "1_1.png")


def ridge_image(height: int = 320, width: int = 256, period: float = 9.5) -> np.ndarray:
    """Dark ridges on a white background: a print in the middle of a white frame."""
    y, x = np.mgrid[0:height, 0:width]
    ridges = 127.5 + 127.5 * np.cos(2 * np.pi * (x + 0.3 * y) / period)
    image = np.full((height, width), 255.0)
    inside = (
        (y > height // 6) & (y < 5 * height // 6) & (x > width // 6) & (x < 5 * width // 6)
    )
    image[inside] = ridges[inside]
    return image.astype(np.uint8)


def save(path, image: np.ndarray, ppi=None) -> str:
    """Saves the image, with the resolution in the file metadata if ppi is given."""
    pil_image = Image.fromarray(image)
    if ppi is None:
        pil_image.save(str(path))
    else:
        pil_image.save(str(path), dpi=(ppi, ppi))
    return str(path)


# ---------------------------------------------------------------------------
# Resolution: argument, then manifest, then metadata, then assumption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension", [".png", ".tif", ".jpg", ".bmp"])
def test_resolution_is_read_from_the_metadata_of_the_file(tmp_path, extension):
    path = save(tmp_path / ("a" + extension), ridge_image(), ppi=1000)
    assert read_metadata_ppi(path) == pytest.approx(1000, abs=0.5)
    ppi, origin = SensorNormalizer().resolve_ppi(path)
    assert ppi == pytest.approx(1000, abs=0.5)
    assert origin == "metadata"


def test_unknown_resolution_fails_with_a_clear_message(tmp_path):
    path = save(tmp_path / "a.png", ridge_image())
    assert read_metadata_ppi(path) is None
    with pytest.raises(UnknownResolutionError) as error:
        SensorNormalizer().load(path)
    message = str(error.value)
    assert "Resolution unknown" in message
    for option in ("--ppi", "--ppi_manifest", "--assume_ppi"):
        assert option in message


def test_implausible_metadata_is_not_used():
    """The SFinGe example files state 96 ppi, the default of an image program."""
    assert read_metadata_ppi(SFINGE_EXAMPLE) == pytest.approx(96, abs=0.1)
    with pytest.raises(UnknownResolutionError, match="96"):
        SensorNormalizer().resolve_ppi(SFINGE_EXAMPLE)
    # The assumption applies to such a file
    assert SensorNormalizer(assume_ppi=500).resolve_ppi(SFINGE_EXAMPLE) == (500, "assumed")


def test_non_square_pixels_are_rejected(tmp_path):
    path = str(tmp_path / "a.png")
    Image.fromarray(ridge_image()).save(path, dpi=(500, 1000))
    with pytest.raises(ValueError, match="not square"):
        SensorNormalizer(assume_ppi=500).load(path)


def test_order_of_the_resolution_sources(tmp_path):
    root = tmp_path / "images"
    (root / "sensor_a" / "session1").mkdir(parents=True)
    (root / "sensor_b").mkdir()
    in_a = save(root / "sensor_a" / "session1" / "1_1.png", ridge_image(), ppi=500)
    special = save(root / "sensor_a" / "session1" / "1_2.png", ridge_image(), ppi=500)
    in_b = save(root / "sensor_b" / "2_1.png", ridge_image(), ppi=700)
    no_metadata = save(root / "sensor_b" / "2_2.png", ridge_image())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,ppi\n"
        "sensor_a,1000\n"  # a directory
        "sensor_a/session1/1_2.png,385\n"  # a file wins over its directory
    )

    normalizer = SensorNormalizer(root_dir=str(root), manifest=str(manifest), assume_ppi=600)
    # Manifest before metadata
    assert normalizer.resolve_ppi(in_a) == (1000, "manifest")
    assert normalizer.resolve_ppi(special) == (385, "manifest")
    # Metadata before the assumption
    ppi, origin = normalizer.resolve_ppi(in_b)
    assert (round(ppi), origin) == (700, "metadata")
    assert normalizer.resolve_ppi(no_metadata) == (600, "assumed")

    # The explicit value before everything
    explicit = SensorNormalizer(root_dir=str(root), manifest=str(manifest), ppi=2000)
    for path in (in_a, special, in_b, no_metadata):
        assert explicit.resolve_ppi(path) == (2000, "argument")


def test_manifest_is_checked(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file,resolution\na.png,500\n")
    with pytest.raises(ValueError, match="column 'path'"):
        read_manifest(str(manifest))
    manifest.write_text("path,ppi\na.png,high\n")
    with pytest.raises(ValueError, match="line 2"):
        read_manifest(str(manifest))
    manifest.write_text("path,ppi,ridges\na.png,500,white\n")
    with pytest.raises(ValueError, match="ridges"):
        read_manifest(str(manifest))
    manifest.write_text("path,ppi,ridges\na.png,,light\n./b/,1000,\n")
    assert read_manifest(str(manifest)) == {"a.png": (None, "light"), "b": (1000.0, None)}
    with pytest.raises(ValueError, match="root directory"):
        SensorNormalizer(manifest=str(manifest))


@pytest.mark.parametrize("ppi", [0, -500, float("nan")])
def test_invalid_resolutions_are_rejected(ppi):
    with pytest.raises(ValueError):
        SensorNormalizer(ppi=ppi)
    with pytest.raises(ValueError):
        SensorNormalizer(assume_ppi=ppi)


def test_unreadable_file(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ImageReadError):
        SensorNormalizer(ppi=500).load(str(path))


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def test_a_500_ppi_image_is_not_changed(tmp_path):
    image = ridge_image()
    path = save(tmp_path / "a.png", image, ppi=500)
    prepared, info = SensorNormalizer().load(path)
    assert np.array_equal(prepared, image)
    assert (info.width, info.height) == (image.shape[1], image.shape[0])
    assert info.ppi == pytest.approx(500, abs=0.01)


@pytest.mark.parametrize("ppi", [1000, 385, 508, 250])
def test_images_are_resampled_to_500_ppi(tmp_path, ppi):
    """The ridge period of the prepared image is that of a 500 ppi image."""
    period_500 = 9.5
    scale = ppi / 500
    height, width = int(round(320 * scale)), int(round(256 * scale))
    image = ridge_image(height, width, period=period_500 * scale)
    path = save(tmp_path / "a.png", image, ppi=ppi)

    prepared, info = SensorNormalizer().load(path)
    assert prepared.dtype == np.uint8
    assert prepared.shape == (info.height, info.width)
    assert (info.width, info.height) == resampled_size(width, height, ppi)
    assert abs(info.width - 256) <= 1 and abs(info.height - 320) <= 1
    assert (info.source_width, info.source_height) == (width, height)

    # Ridge period along a row in the middle of the print, from the spectrum
    row = prepared[info.height // 2, info.width // 4 : 3 * info.width // 4].astype(float)
    spectrum = np.abs(np.fft.rfft(row - row.mean(), n=4096))
    period = 4096 / np.argmax(spectrum)
    assert period == pytest.approx(period_500, rel=0.03)


def test_resample_to_ppi_keeps_the_position_of_a_point():
    """A point at a physical position is at the same physical position afterwards."""
    image = np.zeros((400, 600), dtype=np.uint8)
    image[196:204, 296:304] = 255  # centre (299.5, 199.5) at 1000 ppi
    half = resample_to_ppi(image, 1000)
    assert half.shape == (200, 300)
    ys, xs = np.nonzero(half > 127)
    assert (xs.mean(), ys.mean()) == pytest.approx((149.5, 99.5), abs=0.01)


# ---------------------------------------------------------------------------
# Ridge polarity
# ---------------------------------------------------------------------------


def test_polarity_is_estimated_from_the_background():
    image = ridge_image()
    assert estimate_ridge_polarity(image) == "dark"
    assert estimate_ridge_polarity(255 - image) == "light"
    # No background: no decision
    y, x = np.mgrid[0:320, 0:256]
    full = (127.5 + 127.5 * np.cos(2 * np.pi * x / 9.5)).astype(np.uint8)
    assert estimate_ridge_polarity(full) is None
    assert estimate_ridge_polarity(np.full((320, 256), 200, dtype=np.uint8)) is None


@pytest.mark.parametrize("ridges", ["light", "auto"])
def test_light_ridges_are_inverted(tmp_path, ridges):
    image = ridge_image()
    path = save(tmp_path / "a.png", 255 - image, ppi=500)
    prepared, info = SensorNormalizer(ridges=ridges).load(path)
    assert np.array_equal(prepared, image)
    assert info.ridges == "light"
    assert info.ridges_origin == ("argument" if ridges == "light" else "estimated")


def test_auto_leaves_dark_ridges_and_images_without_background(tmp_path):
    image = ridge_image()
    prepared, info = SensorNormalizer(ridges="auto").load(save(tmp_path / "a.png", image, 500))
    assert np.array_equal(prepared, image)
    assert (info.ridges, info.ridges_origin) == ("dark", "estimated")
    assert info.polarity_warning is None

    full = np.tile((127.5 + 127.5 * np.cos(np.arange(256) / 1.5)).astype(np.uint8), (320, 1))
    prepared, info = SensorNormalizer(ridges="auto").load(save(tmp_path / "b.png", full, 500))
    assert np.array_equal(prepared, full)
    assert (info.ridges, info.ridges_origin) == ("dark", "default")
    assert "could not be estimated" in info.polarity_warning


def test_polarity_from_the_manifest_wins(tmp_path):
    root = tmp_path / "images"
    (root / "inverting_sensor").mkdir(parents=True)
    image = ridge_image()
    light = save(root / "inverting_sensor" / "1_1.png", 255 - image, ppi=500)
    dark = save(root / "2_1.png", image, ppi=500)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,ridges\ninverting_sensor,light\n")
    normalizer = SensorNormalizer(root_dir=str(root), manifest=str(manifest))
    for path in (light, dark):
        prepared, info = normalizer.load(path)
        assert np.array_equal(prepared, image)
    assert normalizer.load(light)[1].ridges_origin == "manifest"


def test_the_check_warns_about_an_image_that_looks_inverted(tmp_path):
    image = ridge_image()
    path = save(tmp_path / "a.png", 255 - image, ppi=500)
    # Taken as dark ridges (the default), not changed, but the check notices
    prepared, info = SensorNormalizer().load(path, check_polarity=True)
    assert np.array_equal(prepared, 255 - image)
    assert "looks like light ridges" in info.polarity_warning
    # Without the check nothing is estimated
    assert SensorNormalizer().load(path)[1].polarity_warning is None
    # The right setting gives no warning
    _, info = SensorNormalizer(ridges="light").load(path, check_polarity=True)
    assert info.polarity_warning is None


def test_the_preparation_does_not_need_torch():
    """The minutiae extraction runs in an environment without torch."""
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['torchvision'] = None\n"
        "import flx.preparation.sensor_normalization, flx.preparation.minutiae_extraction\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=dirname(dirname(__file__)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
