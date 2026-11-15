from typing import Union, Callable

import torch

from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.image_loader import ImageLoader
from flx.data.pose_dataset import PoseLoader
from flx.image_processing.augmentation import (
    PoseTransform,
    RandomPoseTransform,
)


class TransformedImageLoader(DataLoader):
    def __init__(
        self,
        images: ImageLoader,
        poses: Union[None, PoseLoader, RandomPoseTransform] = None,
        transforms: list[Callable[[torch.Tensor], torch.Tensor]] = [],
    ):
        """
        A lightweight wrapper for applying image augmentation to an existing fingerprint dataset.

        First, if 'augment_image_pose' is True, a pose transform can be applied, it consists of:
            - Padding
            - Rotation
            - Shift in x and y direction
        If 'poses' is specified the rotation and shift are taken from this FingerprintPoseDataset,
        otherwise they are randomly generated.

        Then, if 'augment_image_quality' is True, the following properties are randomly transformed:
            - Gain (aka brightness)
            - Contrast
            - Gamma

        """
        self._images: ImageLoader = images
        self._pose_distribution: RandomPoseTransform = None
        self._pose_dataset: PoseLoader = None
        self._transforms: list[Callable[[torch.Tensor], torch.Tensor]] = transforms
        # Identifier, pose and image size (height, width) of the last call of get
        self._last_pose: tuple[Identifier, PoseTransform, tuple[int, int]] = None

        if not (
            poses is None
            or isinstance(poses, PoseLoader)
            or isinstance(poses, RandomPoseTransform)
        ):
            raise TypeError(
                "Parameter 'poses' must be None, a PoseDataset or a PoseDistribution"
            )

        if isinstance(poses, PoseLoader):
            if not set(images.ids).issubset(set(poses.ids)):
                raise RuntimeError("Received an incomplete pose dataset")
            self._pose_dataset = poses

        if isinstance(poses, RandomPoseTransform):
            self._pose_distribution = poses

    @property
    def ids(self) -> IdentifierSet:
        return self._images.ids

    def get(self, identifier: Identifier) -> torch.Tensor:
        """
        Get a fingerprint sample from the underlying dataset with the described augmentations applied.

        The transforms are only applied to the fingerprint image. To apply the pose
        transform to the minutia map as well, wrap the minutia map loader with a
        PoseTransformedMinutiaMapLoader (see minutia_map_loader.py).

        @returns the transformed fingerprint image
        """
        img = self._images.get(identifier)

        pose = None
        if self._pose_dataset is not None:
            pose = self._pose_dataset.get(identifier)
        elif self._pose_distribution is not None:
            pose = self._pose_distribution.sample()
        self._last_pose = (identifier, pose, (img.shape[-2], img.shape[-1]))

        if pose is not None:
            img = pose(img)

        for transform in self._transforms:
            img = transform(img)

        return img

    def get_last_pose(
        self, identifier: Identifier
    ) -> tuple[Union[None, PoseTransform], tuple[int, int]]:
        """
        @returns the pose transform (None if there was none) that the last call of get
            applied to the image of the identifier, and the (height, width) of the
            image before the pose transform.
        """
        if self._last_pose is None or self._last_pose[0] != identifier:
            raise RuntimeError(
                f"The image of {identifier} must be loaded directly before its pose is used"
            )
        return self._last_pose[1:]
