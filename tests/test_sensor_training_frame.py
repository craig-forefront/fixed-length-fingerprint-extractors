"""
The proof that image and minutia map agree for prints from different sensors.

The same physical prints are written as image files at 500, 1000 and 385
pixels per inch (ppi), in different sizes per print. scripts/extract_minutiae.py
(with a stand-in for the extractor that reports the marker in the image as the
minutia) writes the .mnt files, and the image and minutia datasets of
scripts/train_500k.py load both. The marker in the image that the model gets
must be where the minutia map has its peak.
"""

import importlib.util
import random
import sys
import types
from functools import partial
from os.path import dirname, join

import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms.functional as VTF
from PIL import Image

from flx.data.dataset import Identifier, IdentifierSet
from flx.data.fingerflow_minutia_loader import read_mnt_file
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.minutia_map_loader import MINUTIA_MAP_SIZE
from flx.preparation.sensor_normalization import SensorNormalizer, UnknownResolutionError
from flx.setup.config import INPUT_SIZE

pd = pytest.importorskip("pandas")

SCRIPTS_DIR = join(dirname(dirname(__file__)), "scripts")

# Distance of the second marker, that shows the direction of the minutia
DIRECTION_MARKER_DISTANCE = 30

# The prints as they are at 500 ppi: (width, height) and one minutia (x, y,
# angle). Different sizes, like the images of different sensors.
PRINTS = [
    ((292, 392), (146.0, 196.0, 0.0)),
    ((400, 300), (220.0, 130.0, np.pi / 3)),
    ((256, 256), (110.0, 140.0, 4.0)),
    ((500, 640), (270.0, 300.0, 5.5)),
]
IDS = IdentifierSet([Identifier(i, 0) for i in range(len(PRINTS))])


def print_image(index: int, ppi: float) -> np.ndarray:
    """
    Print number `index` as a sensor of the given resolution delivers it: white,
    with a gray square of 0.022 inch at the minutia and one of 0.01 inch at the
    tip of its direction.
    """
    (width, height), (x, y, angle) = PRINTS[index]
    scale = ppi / 500
    columns, rows = int(round(width * scale)), int(round(height * scale))
    grid_y, grid_x = np.mgrid[0:rows, 0:columns]
    # Centres of the pixels in units of 500 ppi pixels
    x_500, y_500 = (grid_x + 0.5) / scale - 0.5, (grid_y + 0.5) / scale - 0.5
    tip_x = x + DIRECTION_MARKER_DISTANCE * np.cos(angle)
    tip_y = y + DIRECTION_MARKER_DISTANCE * np.sin(angle)
    image = np.full((rows, columns), 255, dtype=np.uint8)
    image[(np.abs(x_500 - x) <= 5.5) & (np.abs(y_500 - y) <= 5.5)] = 128
    image[(np.abs(x_500 - tip_x) <= 2.5) & (np.abs(y_500 - tip_y) <= 2.5)] = 128
    return image


def marker_positions(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of the large and of the small marker of an image with a white background."""
    darkness = 255.0 - image.astype(float)
    count, labels = cv2.connectedComponents((darkness > 25).astype(np.uint8))
    assert count == 3  # Background and two markers
    grid_y, grid_x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    markers = []
    for label in (1, 2):
        weights = np.where(labels == label, darkness, 0.0)
        centre = np.array([(weights * grid_x).sum(), (weights * grid_y).sum()]) / weights.sum()
        markers.append((weights.sum(), centre))
    markers.sort(key=lambda marker: marker[0], reverse=True)
    return markers[0][1], markers[1][1]


class MarkerExtractor:
    """Has the interface of fingerflow.extractor.Extractor; finds the markers."""

    def __init__(self, *args, **kwargs):
        pass

    def extract_minutiae(self, image):
        position, tip = marker_positions(image)
        angle = np.arctan2(tip[1] - position[1], tip[0] - position[0]) % (2 * np.pi)
        table = pd.DataFrame(
            {"x": [position[0]], "y": [position[1]], "angle": [angle], "score": [0.9]}
        )
        table["class"] = np.nan
        table["direction_confidence"] = 1.0
        return {"core": None, "minutiae": table}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, join(SCRIPTS_DIR, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def white_resize(img: torch.Tensor, roi=None) -> torch.Tensor:
    """
    pad_and_resize_to_deepprint_input_size with white instead of black around
    the image, so that the markers can be found. The geometry is the same.
    """
    if roi is not None:
        img = 1.0 - VTF.center_crop(1.0 - img, roi)
    return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


def identity(img: torch.Tensor) -> torch.Tensor:
    return img


@pytest.fixture
def train_500k(monkeypatch):
    script = load_script("train_500k")
    # Keep the markers as they are: no change of contrast, no binarization
    monkeypatch.setattr(script, "QUALITY_AUGMENTATION", identity)
    monkeypatch.setattr(script, "LazilyAllocatedBinarizer", lambda _: identity)
    monkeypatch.setattr(script, "pad_and_resize_to_deepprint_input_size", white_resize)
    return script


def write_images(image_dir, ppi: float, metadata: bool = True) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for identifier in IDS:
        image = Image.fromarray(print_image(identifier.subject, ppi))
        path = str(image_dir / f"{identifier.subject + 1}_1.png")
        if metadata:
            image.save(path, dpi=(ppi, ppi))
        else:
            image.save(path)


def extract(monkeypatch, image_dir, minutiae_dir, *options) -> None:
    """Runs scripts/extract_minutiae.py with the marker extractor."""
    fingerflow = types.ModuleType("fingerflow")
    extractor = types.ModuleType("fingerflow.extractor")
    extractor.Extractor = MarkerExtractor
    fingerflow.extractor = extractor
    monkeypatch.setitem(sys.modules, "fingerflow", fingerflow)
    monkeypatch.setitem(sys.modules, "fingerflow.extractor", extractor)
    script = load_script("extract_minutiae")
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_minutiae.py", "--input_dir", str(image_dir)]
        + ["--output_dir", str(minutiae_dir), "--coarse_net", "c", "--fine_net", "f"]
        + [str(option) for option in options],
    )
    assert script.main() == 0


def model_input_markers(img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    assert img.shape == (1, INPUT_SIZE, INPUT_SIZE)
    return marker_positions((img[0] * 255).numpy())


def assert_minutia_map_matches_image(minutia_map, position, tip) -> None:
    """
    The peak of the minutia map is at the marker of the image, in the layer of
    the direction from the marker to the small marker.
    """
    assert minutia_map.shape == (6, MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE)
    peak_y, peak_x = np.unravel_index(
        int(minutia_map.sum(dim=0).argmax()), (MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE)
    )
    # The position in the minutia map is rounded down
    scale = MINUTIA_MAP_SIZE / INPUT_SIZE
    assert abs(peak_x - position[0] * scale) <= 1.5
    assert abs(peak_y - position[1] * scale) <= 1.5
    # The k-th layer is the direction k * 60 degrees
    layer = np.arctan2(tip[1] - position[1], tip[0] - position[0]) / (np.pi / 3)
    if abs(layer - round(layer)) < 0.4:
        assert minutia_map[:, peak_y, peak_x].argmax() == round(layer) % 6


def expected_marker_in_model_input(index: int, canvas_size: int) -> np.ndarray:
    """Where the minutia of the print is in the model input, by calculation."""
    (width, height), (x, y, _) = PRINTS[index]
    if canvas_size:
        side = canvas_size
        left, top = (width - canvas_size) / 2, (height - canvas_size) / 2
    else:
        side = max(width, height)
        left, top = -((side - width) // 2), -((side - height) // 2)
    return (np.array([x - left, y - top]) + 0.5) * INPUT_SIZE / side - 0.5


@pytest.mark.parametrize("canvas_size", [0, 512, 300])
@pytest.mark.parametrize("ppi", [500, 1000, 385])
def test_image_and_minutia_map_agree(tmp_path, monkeypatch, train_500k, ppi, canvas_size):
    image_dir, minutiae_dir = tmp_path / "images", tmp_path / "minutiae"
    write_images(image_dir, ppi)
    extract(monkeypatch, image_dir, minutiae_dir)

    # The .mnt files are in the pixels of the 500 ppi image, whatever the sensor
    for identifier in IDS:
        (width, height), (x, y, angle) = PRINTS[identifier.subject]
        locs, oris, size = read_mnt_file(str(minutiae_dir / f"{identifier.subject + 1}_1.mnt"))
        assert abs(size[0] - width) <= 1 and abs(size[1] - height) <= 1
        assert locs[0] == pytest.approx((x, y), abs=0.75)
        assert np.cos(oris[0] - angle) > np.cos(np.radians(3))

    normalizer = SensorNormalizer()
    # Without augmentation, as in validation
    images = train_500k.create_validation_image_loader(str(image_dir), normalizer, canvas_size)
    maps = train_500k.create_minutia_dataset(str(minutiae_dir), IDS, images, canvas_size)
    for identifier in IDS:
        position, tip = model_input_markers(images.get(identifier))
        # The image is where it should be: every sensor gives the same model input
        expected = expected_marker_in_model_input(identifier.subject, canvas_size)
        assert position == pytest.approx(expected, abs=1.0)
        minutia_map, weight = maps.get(identifier)
        assert weight == 1.0
        assert_minutia_map_matches_image(minutia_map, position, tip)

    # With the random poses of the training
    random.seed(3)
    training_images = train_500k.create_image_dataset(str(image_dir), IDS, normalizer, canvas_size)
    training_maps = train_500k.create_minutia_dataset(
        str(minutiae_dir), IDS, training_images.data_loader, canvas_size
    )
    shifts = []
    for _ in range(3):
        for identifier in IDS:
            position, tip = model_input_markers(training_images.get(identifier))
            minutia_map, weight = training_maps.get(identifier)
            assert weight == 1.0
            assert_minutia_map_matches_image(minutia_map, position, tip)
            expected = expected_marker_in_model_input(identifier.subject, canvas_size)
            shifts.append(float(np.linalg.norm(position - expected)))
    assert max(shifts) > 5  # the poses were random


def test_canvas_gives_every_sensor_the_same_scale(tmp_path, train_500k):
    """Distance of the two markers (30 pixels at 500 ppi) in the model input."""
    image_dir = tmp_path / "images"
    write_images(image_dir, 1000)
    normalizer = SensorNormalizer()

    def distances(canvas_size: int) -> list[float]:
        images = train_500k.create_validation_image_loader(str(image_dir), normalizer, canvas_size)
        result = []
        for identifier in IDS:
            position, tip = model_input_markers(images.get(identifier))
            result.append(float(np.linalg.norm(tip - position)))
        return result

    # With a canvas: one scale, INPUT_SIZE / canvas_size
    for distance in distances(512):
        assert distance == pytest.approx(DIRECTION_MARKER_DISTANCE * INPUT_SIZE / 512, abs=0.5)
    # Without: the scale follows the longer side of each image
    for distance, ((width, height), _) in zip(distances(0), PRINTS):
        expected = DIRECTION_MARKER_DISTANCE * INPUT_SIZE / max(width, height)
        assert distance == pytest.approx(expected, abs=0.5)


def test_training_reports_minutiae_from_differently_prepared_images(
    tmp_path, monkeypatch, train_500k
):
    """The minutiae were extracted from the 500 ppi image, the training loads another one."""
    image_dir, minutiae_dir = tmp_path / "images", tmp_path / "minutiae"
    write_images(image_dir, 1000, metadata=False)
    extract(monkeypatch, image_dir, minutiae_dir, "--ppi", 1000)
    identifier = Identifier(0, 0)

    # The training is told 500 ppi and loads the image at twice the size. Without
    # a canvas, image and minutiae are both scaled by their longer side and still
    # agree...
    wrong = SensorNormalizer(assume_ppi=500)
    images = train_500k.create_validation_image_loader(str(image_dir), wrong)
    maps = train_500k.create_minutia_dataset(str(minutiae_dir), IDS, images)
    position, tip = model_input_markers(images.get(identifier))
    assert_minutia_map_matches_image(maps.get(identifier)[0], position, tip)
    # ... with a canvas they do not, and the loader says so
    images = train_500k.create_validation_image_loader(str(image_dir), wrong, 512)
    maps = train_500k.create_minutia_dataset(str(minutiae_dir), IDS, images, 512)
    images.get(identifier)
    with pytest.raises(ValueError, match="292x392 pixels.*584x784 pixels.*--ppi"):
        maps.get(identifier)

    # An image of another shape than the one of the extraction is always an error
    Image.fromarray(print_image(1, 500)).save(str(image_dir / "1_1.png"))
    images = train_500k.create_validation_image_loader(str(image_dir), wrong)
    maps = train_500k.create_minutia_dataset(str(minutiae_dir), IDS, images)
    images.get(identifier)
    with pytest.raises(ValueError, match="292x392 pixels.*400x300 pixels.*another shape"):
        maps.get(identifier)


def test_training_stops_on_an_image_of_unknown_resolution(tmp_path, train_500k):
    image_dir = tmp_path / "images"
    write_images(image_dir, 500, metadata=False)
    loader = train_500k.CustomFingerprintLoader(str(image_dir), SensorNormalizer())
    with pytest.raises(UnknownResolutionError, match=r"1_1\.png: Resolution unknown.*--assume_ppi"):
        loader.get(Identifier(0, 0))
    # Without a normalizer the file is loaded as it is, as before
    assert train_500k.CustomFingerprintLoader(str(image_dir)).get(Identifier(0, 0)).shape == (
        1, 392, 292,
    )


def test_training_inverts_light_ridges_like_the_extraction(tmp_path, train_500k):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dark = print_image(0, 500)
    Image.fromarray(255 - dark).save(str(image_dir / "1_1.png"), dpi=(500, 500))
    loader = train_500k.CustomFingerprintLoader(str(image_dir), SensorNormalizer(ridges="light"))
    assert torch.equal(loader.get(Identifier(0, 0)), VTF.to_tensor(dark))


def test_train_500k_stores_the_preparation(tmp_path, monkeypatch):
    """The script hands its options to the loaders and writes them to the output directory."""
    import json

    script = load_script("train_500k")
    image_dir = tmp_path / "images"
    write_images(image_dir, 1000, metadata=False)
    calls = {}

    class Extractor:
        def fit(self, fingerprints, minutia_maps, labels, **kwargs):
            calls["image"] = fingerprints.get(Identifier(0, 0))
            calls["validation"] = kwargs["validation_fingerprints"]

    monkeypatch.setattr(script, "create_extractor", lambda *args, **kwargs: Extractor())
    monkeypatch.setattr(script, "POSE_AUGMENTATION", None)
    monkeypatch.setattr(script, "QUALITY_AUGMENTATION", identity)
    monkeypatch.setattr(script, "LazilyAllocatedBinarizer", lambda _: identity)
    monkeypatch.setattr(script, "pad_and_resize_to_deepprint_input_size", white_resize)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_500k.py", "--dataset_dir", str(image_dir), "--output_dir", str(tmp_path / "model")]
        + ["--num_subjects", "4", "--num_impressions", "1", "--validation_split", "0"]
        + ["--ppi", "1000", "--canvas_size", "512"],
    )
    script.main()

    stored = json.loads((tmp_path / "model" / "data_preparation.json").read_text())
    assert stored["ppi"] == 1000 and stored["canvas_size"] == 512
    assert stored["target_ppi"] == 500 and stored["ridges"] == "dark"
    position, _ = model_input_markers(calls["image"])
    assert position == pytest.approx(expected_marker_in_model_input(0, 512), abs=1.0)
