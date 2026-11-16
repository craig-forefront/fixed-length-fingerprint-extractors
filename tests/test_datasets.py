from os.path import join, dirname

import pytest
import cv2
import torch
import numpy as np

from flx.data.transformed_image_loader import TransformedImageLoader
from flx.data.dataset import Identifier, IdentifierSet
from flx.data.label_index import LabelIndex
from flx.data.embedding_loader import (
    EmbeddingLoader,
)
from flx.data.image_loader import (
    ImageLoader,
    SFINGE_CROPPED_ROWS,
    SFingeLoader,
    FVC2004Loader,
    NistSD4Dataset,
    MCYTOpticalLoader,
    MCYTCapacitiveLoader,
)
from flx.data.minutia_map_loader import (
    SFingeMinutiaMapLoader,
    MCYTOpticalMinutiaMapLoader,
    MCYTCapacitiveMinutiaMapLoader,
)
from flx.data.pose_dataset import PoseLoader
from flx.setup.config import INPUT_SIZE
from flx.setup.paths import get_fingerprint_dataset_path
from flx.image_processing.augmentation import (
    RandomPoseTransform,
    PoseTransform,
    RandomQualityTransform,
)
from flx.visualization.show_with_opencv import (
    show_tensor_as_image,
    show_minutia_maps_from_tensor,
    wait_for_key_and_close_windows,
)

TEST_DATA_DIR = join(dirname(__file__), "data")


def assert_is_image_tensor(tensor: torch.Tensor) -> None:
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1 or tensor.shape[0] == 0
    assert tensor.shape[1] > 0
    assert tensor.shape[2] > 0


def assert_identifiers(ds: IdentifierSet, num_subjects: int, num_samples: int):
    assert len(ds) == num_samples
    assert ds.num_subjects == num_subjects


def test_identifier_set():
    all_ids = [
        Identifier(4, 1),
        Identifier(3, 1),
        Identifier(5, 2),
        Identifier(3, 0),
        Identifier(1, 2),
        Identifier(4, 0),
        Identifier(1, 0),
        Identifier(1, 1),
        Identifier(2, 0),
        Identifier(2, 1),
        Identifier(5, 0),
        Identifier(5, 1),
    ]

    ds = IdentifierSet(all_ids)
    assert_identifiers(ds, 5, 12)

    # Test filter by subject
    subset = ds.filter_by_subject([2, 4])
    assert_identifiers(subset, 2, 4)
    assert subset[0] == Identifier(2, 0)
    assert subset[1] == Identifier(2, 1)
    assert subset[2] == Identifier(4, 0)
    assert subset[3] == Identifier(4, 1)

    print("\nFilter by id")
    print("Ids for subset " + " ".join(str(i) for i in all_ids[4:6]))
    subset = ds.filter_by_id(IdentifierSet(all_ids[4:6]))
    assert_identifiers(subset, 2, 2)
    assert subset[0] == all_ids[4]
    assert subset[1] == all_ids[5]

    subset = ds.filter_by_index(range(3))
    assert_identifiers(subset, 1, 3)
    assert subset[0] == Identifier(1, 0)
    assert subset[1] == Identifier(1, 1)
    assert subset[2] == Identifier(1, 2)


def test_label_dataset():
    label_index: LabelIndex = LabelIndex(
        IdentifierSet(
            [
                Identifier(4, 1),
                Identifier(3, 1),
                Identifier(5, 2),
                Identifier(3, 0),
                Identifier(1, 2),
            ]
        )
    )
    assert len(label_index.ids) == 5
    assert label_index.ids.num_subjects == 4
    assert label_index.get(Identifier(1, 2)) == 0
    assert label_index.get(Identifier(3, 0)) == 1
    assert label_index.get(Identifier(3, 1)) == 1
    assert label_index.get(Identifier(4, 1)) == 2
    assert label_index.get(Identifier(5, 2)) == 3


def test_embedding_dataset():
    ids = [Identifier(0, 0), Identifier(0, 1), Identifier(1, 0), Identifier(1, 1)]
    embeddings = np.array([np.random.random(4) for _ in ids])

    loader = EmbeddingLoader(IdentifierSet(ids), embeddings)
    for id, emb in zip(loader.ids, loader.numpy()):
        print(id)
        print(emb)

    dataset2 = EmbeddingLoader.combine(loader, loader)
    for id, emb in zip(dataset2.ids, dataset2.numpy()):
        print(id)
        print(emb)


def test_minutia_map_sfinge():
    loader = SFingeMinutiaMapLoader(join(TEST_DATA_DIR, "SFingeExample"))
    minutia_map, mask = loader.get(Identifier(0, 0))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_sfinge_minutia_orientations_are_radians():
    root_dir = join(TEST_DATA_DIR, "SFingeExample")
    loader = SFingeMinutiaMapLoader(root_dir)
    locs, oris = loader.get_minutiae(Identifier(0, 0))
    assert len(locs) == len(oris) > 0

    # In the ISO 19794-2:2005 file each minutia has 6 bytes, the 5th is the
    # angle in units of 360 / 256 degrees, counterclockwise. The loader returns
    # the clockwise angle (y points down).
    with open(join(root_dir, "1_1.ist"), "rb") as file:
        template = file.read()
    angle_bytes = np.array([template[28 + 6 * i + 4] for i in range(template[27])])
    expected = 2 * np.pi * (1 - angle_bytes / 256)
    assert oris == pytest.approx(expected)
    assert oris.min() >= 0
    assert oris.max() <= 2 * np.pi


def test_sfinge_minutiae_are_complete():
    root_dir = join(TEST_DATA_DIR, "SFingeExample")
    loader = SFingeMinutiaMapLoader(root_dir)
    for identifier in loader.ids:
        filename = f"{identifier.subject + 1}_{identifier.impression + 1}.ist"
        with open(join(root_dir, filename), "rb") as file:
            template = file.read()
        # The 28 bytes of the headers end with the number of minutiae, then
        # there are 6 bytes for each minutia. The first two are the type (2 bits)
        # and x (14 bits), the next two are y.
        num_minutiae = template[27]
        assert len(template) >= 28 + 6 * num_minutiae
        x = [
            int.from_bytes(template[28 + 6 * i : 30 + 6 * i], "big") & 0x3FFF
            for i in range(num_minutiae)
        ]
        y = [
            int.from_bytes(template[30 + 6 * i : 32 + 6 * i], "big")
            for i in range(num_minutiae)
        ]
        scale = INPUT_SIZE / 560
        expected = np.array([x, y]).T * scale + np.array([(560 - 416) / 2 * scale, 0])

        locs, oris = loader.get_minutiae(identifier)
        assert len(locs) == len(oris) == num_minutiae
        assert locs == pytest.approx(expected, abs=0.5)


def test_minutia_map_mcyt_capactive():
    loader = MCYTCapacitiveMinutiaMapLoader(
        join(TEST_DATA_DIR, "MCYTCapacitiveExample")
    )
    minutia_map, mask = loader.get(Identifier(0, 0))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_minutia_map_mcyt_optical():
    loader = MCYTOpticalMinutiaMapLoader(join(TEST_DATA_DIR, "MCYTOpticalExample"))
    minutia_map, mask = loader.get(Identifier(0, 8))
    print(minutia_map.shape)
    show_minutia_maps_from_tensor(minutia_map)


def test_sfinge_loader_cuts_rows_off_the_bottom():
    directory = join(TEST_DATA_DIR, "SFingeExample")
    loader = SFingeLoader(directory)
    assert len(loader.ids) > 0
    # The example images have 392 rows; the loader must return the top 360.
    assert SFINGE_CROPPED_ROWS == 32
    for bid in loader.ids:
        filename = f"{bid.subject + 1}_{bid.impression + 1}.png"
        file_image = cv2.imread(join(directory, filename), cv2.IMREAD_GRAYSCALE)
        image = loader.get(bid)
        assert image.shape[1] == file_image.shape[0] - SFINGE_CROPPED_ROWS
        assert image.shape[1] == 360
        assert image.shape[2] == file_image.shape[1]
        assert np.array_equal(
            (image[0] * 255).round().numpy(), file_image[:-SFINGE_CROPPED_ROWS]
        )


def test_pose_dataset(tmp_path):
    def make_random_pose_dataset(
        ids: list[Identifier], pose_distribution: RandomPoseTransform
    ) -> PoseLoader:
        return PoseLoader(ids, [pose_distribution.sample() for _ in ids])

    image_loader = SFingeLoader(join(TEST_DATA_DIR, "SFingeExample"))
    pose_loader = make_random_pose_dataset(
        image_loader.ids, pose_distribution=RandomPoseTransform()
    )
    # Not get_pose_dataset_path: the test must not write into data/poses
    path = str(tmp_path / "PoseTest.json")
    pose_loader.save(path)

    loaded = PoseLoader.load(path)
    assert list(loaded.ids) == list(image_loader.ids)
    for bid in image_loader.ids:
        assert loaded.get(bid) == pose_loader.get(bid)
        fp = image_loader.get(bid)
        posed = loaded.get(bid)(fp)
        assert_is_image_tensor(posed)
        # RandomPoseTransform pads 80 pixels on each side
        assert posed.shape[1] == fp.shape[1] + 160
        assert posed.shape[2] == fp.shape[2] + 160
        assert torch.equal(posed, pose_loader.get(bid)(fp))
        show_tensor_as_image(posed, winname=f"Random pose: {bid}", wait=False)
    wait_for_key_and_close_windows()


def test_transformed_dataset():
    images = SFingeLoader(join(TEST_DATA_DIR, "SFingeExample"))

    transformed_images = TransformedImageLoader(images)
    for bid in transformed_images.ids:
        image = transformed_images.get(bid)
        assert torch.equal(image, images.get(bid))
        show_tensor_as_image(
            image, winname=f"NoPoseTrafo, NoQualityTrafo: {bid}", wait=False
        )
    wait_for_key_and_close_windows()

    transformed_images = TransformedImageLoader(
        images, poses=RandomPoseTransform(), transforms=[RandomQualityTransform()]
    )
    for bid in transformed_images.ids:
        image = transformed_images.get(bid)
        assert_is_image_tensor(image)
        assert image.shape[1] == images.get(bid).shape[1] + 160
        assert image.shape[2] == images.get(bid).shape[2] + 160
        assert 0.0 <= image.min() and image.max() <= 1.0
        show_tensor_as_image(
            image, winname=f"PoseDistribution, QualityDistribution: {bid}", wait=False
        )
    wait_for_key_and_close_windows()

    pose_loader = PoseLoader(
        images.ids,
        [
            PoseTransform(pad=80, angle=i / len(images.ids) * 90)
            for i in range(len(images.ids))
        ],
    )
    transformed_images = TransformedImageLoader(images, poses=pose_loader)
    for bid in transformed_images.ids:
        image = transformed_images.get(bid)
        assert torch.equal(image, pose_loader.get(bid)(images.get(bid)))
        pose, (height, width) = transformed_images.get_last_pose(bid)
        assert pose == pose_loader.get(bid)
        assert (height, width) == tuple(images.get(bid).shape[1:])
        show_tensor_as_image(
            image, winname=f"PoseDataset, NoQualityTrafo: {bid}", wait=False
        )
    wait_for_key_and_close_windows()


def _test_image_dataset(
    dataset_type: type,
    root_dir_name: str,
    expected_num_subjects: int,
    expected_num_samples: int,
):
    if root_dir_name.startswith("test_data_dir:"):
        root_dir_name = join(TEST_DATA_DIR, root_dir_name[14:])
    dataset: ImageLoader = dataset_type(get_fingerprint_dataset_path(root_dir_name))
    assert dataset.ids.num_subjects == expected_num_subjects
    assert len(dataset.ids) == expected_num_samples
    image: torch.Tensor = dataset.get(dataset.ids[0])
    assert_is_image_tensor(image)
    show_tensor_as_image(image, f"{root_dir_name}", wait=True)


def test_dataset_SFingeExample():
    _test_image_dataset(SFingeLoader, "test_data_dir:SFingeExample", 4, 4 * 2)


# # Uncomment to test that the datasets are present on the system
#
# def test_dataset_SFingev2ValidationSeparateSubjects():
#     _test_image_dataset(SFingeDataset, "SFingev2ValidationSeparateSubjects", 2000, 2000 * 4)
#
# def test_dataset_SFingev2():
#     _test_image_dataset(SFingeDataset, "SFingev2", 8000, 8000 * 10)
#
# def test_dataset_MCYT330_Optical():
#     _test_image_dataset(MCYTOpticalDataset, "mcyt330_optical", 3300, 3300 * 12)
#
# def test_dataset_MCYT330_Capacitive():
#     _test_image_dataset(MCYTCapacitiveDataset, "mcyt330_capacitive", 3300, 3300 * 12)


def test_embeddings_are_saved_and_loaded_with_their_identifiers(tmp_path):
    # save() referred to an attribute that does not exist and always failed
    ids = [Identifier(1, 1), Identifier(0, 0), Identifier(1, 0), Identifier(0, 1)]
    embeddings = np.arange(12, dtype=np.float32).reshape(4, 3)
    loader = EmbeddingLoader(ids, embeddings)
    loader.save(str(tmp_path / "embeddings"))
    loaded = EmbeddingLoader.load(str(tmp_path / "embeddings"))
    assert len(loaded.ids) == 4
    for identifier, row in zip(ids, embeddings):
        assert (loaded.get(identifier) == row).all()
