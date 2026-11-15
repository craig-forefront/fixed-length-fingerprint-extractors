from functools import partial
from os.path import join
import random

import numpy as np
import pytest
import torch

from flx.data.dataset import Dataset, DataLoader, Identifier, IdentifierSet
from flx.data.fingerflow_minutia_loader import FingerFlowMinutiaMapLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.minutia_map_loader import (
    MINUTIA_MAP_SIZE,
    PoseTransformedMinutiaMapLoader,
)
from flx.data.pose_dataset import PoseLoader
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.image_processing.augmentation import PoseTransform, RandomPoseTransform
from flx.setup.config import INPUT_SIZE

WIDTH = 292
HEIGHT = 392
# Distance of the second marker, that shows the direction of the minutia
DIRECTION_MARKER_DISTANCE = 30

# One minutia (x, y, angle) per sample
MINUTIAE = [
    (146.0, 196.0, 0.0),
    (120.0, 150.0, np.pi / 3),
    (180.0, 230.0, 4.0),
    (100.0, 260.0, 5.5),
    (200.0, 140.0, 2.0),
]
IDS = IdentifierSet([Identifier(i, 0) for i in range(len(MINUTIAE))])

TRAINING_POSES = RandomPoseTransform(
    pad=0,
    angle_min=-30,
    angle_max=30,
    shift_horizontal_min=-40,
    shift_horizontal_max=40,
    shift_vertical_min=-40,
    shift_vertical_max=40,
)

# Like in the training, but the padding is white like the fill of the pose transform
RESIZE = partial(pad_and_resize_to_deepprint_input_size, fill=1.0)


class _MarkerImages(DataLoader):
    """
    White images with a gray marker at the minutia of the sample, or at the tip
    of its direction vector.
    """

    def __init__(self, distance: float = 0.0):
        self._distance = distance

    @property
    def ids(self) -> IdentifierSet:
        return IDS

    def get(self, identifier: Identifier) -> torch.Tensor:
        x, y, angle = MINUTIAE[identifier.subject]
        x = round(x + self._distance * np.cos(angle))
        y = round(y + self._distance * np.sin(angle))
        img = torch.ones(1, HEIGHT, WIDTH)
        img[0, y - 2 : y + 3, x - 2 : x + 3] = 0.5
        return img


def _marker_position(img: torch.Tensor) -> np.ndarray:
    """
    @returns (x, y) of the marker in the image, in pixels
    """
    weights = (1.0 - img[0]).double()
    assert weights.sum() > 0
    y, x = torch.meshgrid(
        torch.arange(img.shape[1]).double(),
        torch.arange(img.shape[2]).double(),
        indexing="ij",
    )
    return np.array(
        [float((weights * x).sum()), float((weights * y).sum())]
    ) / float(weights.sum())


def _write_mnt_files(root_dir: str) -> None:
    for identifier in IDS:
        x, y, angle = MINUTIAE[identifier.subject]
        with open(join(root_dir, f"{identifier.subject + 1}_1.mnt"), "w") as file:
            file.write(f"1\n{WIDTH} {HEIGHT}\n{x} {y} {angle} 0 1.0\n")


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


@pytest.mark.parametrize(
    "pose",
    [
        PoseTransform(angle=25),
        PoseTransform(angle=-30, shift_horizontal=40, shift_vertical=-35),
        PoseTransform(pad=80, angle=60, shift_horizontal=-80, shift_vertical=80),
        PoseTransform(pad=10),
    ],
)
def test_pose_transform_of_coordinates_and_orientations_matches_image(pose):
    for index, (x, y, angle) in enumerate(MINUTIAE):
        identifier = Identifier(index, 0)
        position = _marker_position(pose(_MarkerImages().get(identifier)))
        tip = _marker_position(
            pose(_MarkerImages(DIRECTION_MARKER_DISTANCE).get(identifier))
        )

        coordinates = pose.transform_coordinates(np.array([[x, y]]), HEIGHT, WIDTH)
        assert coordinates.shape == (1, 2)
        assert coordinates[0] == pytest.approx(position, abs=0.2)

        orientation = pose.transform_orientations(np.array([angle]))[0]
        direction = tip - position
        assert _angle_difference(
            orientation, np.arctan2(direction[1], direction[0])
        ) < np.radians(2)


def _assert_minutia_matches_image(
    minutia_maps: PoseTransformedMinutiaMapLoader,
    identifier: Identifier,
    position: np.ndarray,
    tip: np.ndarray,
) -> None:
    """
    @param position : position of the marker of the minutia in the transformed image
    @param tip : position of the marker of the direction in the transformed image
    """
    direction = np.arctan2(tip[1] - position[1], tip[0] - position[0])

    locs, oris = minutia_maps.get_minutiae(identifier)
    assert locs.shape == (1, 2)
    assert locs[0] == pytest.approx(position, abs=0.75)
    assert _angle_difference(oris[0], direction) < np.radians(2)

    minutia_map, weight = minutia_maps.get(identifier)
    assert minutia_map.shape == (6, MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE)
    assert weight == 1.0
    peak_y, peak_x = np.unravel_index(
        int(minutia_map.sum(dim=0).argmax()), (MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE)
    )
    # The position in the minutia map is rounded down
    scale = MINUTIA_MAP_SIZE / INPUT_SIZE
    assert abs(peak_x - position[0] * scale) <= 1.5
    assert abs(peak_y - position[1] * scale) <= 1.5
    # The k-th layer is the direction k * 60 degrees. Between two layers both
    # are equally strong.
    layer = direction / (np.pi / 3)
    if abs(layer - round(layer)) < 0.4:
        assert minutia_map[:, peak_y, peak_x].argmax() == round(layer) % 6


def test_minutia_map_follows_random_pose_of_image(tmp_path):
    _write_mnt_files(str(tmp_path))
    random.seed(1)
    images = TransformedImageLoader(
        _MarkerImages(), poses=TRAINING_POSES, transforms=[RESIZE]
    )
    minutia_maps = PoseTransformedMinutiaMapLoader(
        FingerFlowMinutiaMapLoader(str(tmp_path)), images
    )
    # The order of the datasets is the one of the training
    samples = Dataset.zip(Dataset(images, IDS), Dataset(minutia_maps, IDS))

    poses = []
    for _ in range(10):
        for index, identifier in enumerate(IDS):
            img, (minutia_map, weight) = samples[index]
            assert img.shape == (1, INPUT_SIZE, INPUT_SIZE)

            pose, size = images.get_last_pose(identifier)
            assert size == (HEIGHT, WIDTH)
            poses.append(pose)
            # The marker of the direction with the same pose
            tip = RESIZE(pose(_MarkerImages(DIRECTION_MARKER_DISTANCE).get(identifier)))

            _assert_minutia_matches_image(
                minutia_maps, identifier, _marker_position(img), _marker_position(tip)
            )
            assert torch.equal(minutia_map, minutia_maps.get(identifier)[0])

    # Each sample had its own pose
    assert len({(p.angle, p.shift_horizontal, p.shift_vertical) for p in poses}) == len(poses)
    assert max(abs(p.angle) for p in poses) > 20


def test_minutia_map_follows_pose_dataset_of_image(tmp_path):
    _write_mnt_files(str(tmp_path))
    # With padding the image is larger before it is resized to the input size
    poses = PoseLoader(
        IDS,
        [
            PoseTransform(pad=80, angle=20.0 * i - 40, shift_horizontal=-30, shift_vertical=50)
            for i in range(len(IDS))
        ],
    )
    images = TransformedImageLoader(_MarkerImages(), poses=poses, transforms=[RESIZE])
    tips = TransformedImageLoader(
        _MarkerImages(DIRECTION_MARKER_DISTANCE), poses=poses, transforms=[RESIZE]
    )
    minutia_maps = PoseTransformedMinutiaMapLoader(
        FingerFlowMinutiaMapLoader(str(tmp_path)), images
    )
    for identifier in IDS:
        position = _marker_position(images.get(identifier))
        tip = _marker_position(tips.get(identifier))
        _assert_minutia_matches_image(minutia_maps, identifier, position, tip)


def test_minutia_map_follows_pose_of_image_in_input_size(tmp_path):
    # Like the MCYT images of the training set, the images are resized to the
    # input size when they are loaded, before the pose transform
    class ResizedMarkerImages(_MarkerImages):
        def get(self, identifier: Identifier) -> torch.Tensor:
            return RESIZE(super().get(identifier))

    _write_mnt_files(str(tmp_path))
    random.seed(2)
    images = TransformedImageLoader(ResizedMarkerImages(), poses=TRAINING_POSES)
    minutia_maps = PoseTransformedMinutiaMapLoader(
        FingerFlowMinutiaMapLoader(str(tmp_path)), images
    )
    for identifier in IDS:
        position = _marker_position(images.get(identifier))
        pose, size = images.get_last_pose(identifier)
        assert size == (INPUT_SIZE, INPUT_SIZE)
        tip = pose(ResizedMarkerImages(DIRECTION_MARKER_DISTANCE).get(identifier))
        _assert_minutia_matches_image(
            minutia_maps, identifier, position, _marker_position(tip)
        )


def test_minutia_map_without_pose_is_unchanged(tmp_path):
    _write_mnt_files(str(tmp_path))
    images = TransformedImageLoader(_MarkerImages(), poses=None, transforms=[RESIZE])
    original = FingerFlowMinutiaMapLoader(str(tmp_path))
    minutia_maps = PoseTransformedMinutiaMapLoader(original, images)
    assert minutia_maps.ids == original.ids
    for identifier in IDS:
        images.get(identifier)
        locs, oris = minutia_maps.get_minutiae(identifier)
        expected_locs, expected_oris = original.get_minutiae(identifier)
        assert np.array_equal(locs, expected_locs)
        assert np.array_equal(oris, expected_oris)
        assert torch.equal(minutia_maps.get(identifier)[0], original.get(identifier)[0])


def test_minutiae_outside_of_transformed_image_are_removed(tmp_path):
    _write_mnt_files(str(tmp_path))
    poses = PoseLoader(
        IDS,
        [
            PoseTransform(shift_horizontal=-250),
            PoseTransform(shift_horizontal=250),
            PoseTransform(shift_vertical=-300),
            PoseTransform(shift_vertical=300),
            PoseTransform(angle=180, shift_horizontal=200, shift_vertical=200),
        ],
    )
    images = TransformedImageLoader(_MarkerImages(), poses=poses, transforms=[RESIZE])
    minutia_maps = PoseTransformedMinutiaMapLoader(
        FingerFlowMinutiaMapLoader(str(tmp_path)), images
    )
    for identifier in IDS:
        img = images.get(identifier)
        assert img.min() > 0.99  # The marker is not in the image
        locs, oris = minutia_maps.get_minutiae(identifier)
        assert locs.shape == (0, 2)
        assert oris.shape == (0,)
        assert not minutia_maps.get(identifier)[0].any()


def test_minutia_map_needs_pose_of_its_image(tmp_path):
    _write_mnt_files(str(tmp_path))
    images = TransformedImageLoader(
        _MarkerImages(), poses=TRAINING_POSES, transforms=[RESIZE]
    )
    minutia_maps = PoseTransformedMinutiaMapLoader(
        FingerFlowMinutiaMapLoader(str(tmp_path)), images
    )
    with pytest.raises(RuntimeError):
        minutia_maps.get(IDS[0])
    images.get(IDS[1])
    with pytest.raises(RuntimeError):
        minutia_maps.get(IDS[0])
    minutia_maps.get(IDS[1])
