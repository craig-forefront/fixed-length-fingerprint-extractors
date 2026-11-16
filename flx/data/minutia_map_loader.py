import os
from typing import Union
from abc import abstractmethod

import numpy as np
import torch
import torchvision.transforms.functional as VTF

from flx.data.dataset import Identifier, DataLoader, IdentifierSet
from flx.data.minutia_map import create_minutia_map
from flx.data.file_index import FileIndex
from flx.data.iso_encoder_decoder import decode

from flx.data.image_helpers import (
    get_input_resolution,
    transform_from_input_size,
    transform_to_input_size,
)
from flx.data.transformed_image_loader import TransformedImageLoader

"""
The size of the original images created by SFinge.
The minutia locations in the .ist files refer to the original image,
even if the images are resampled to a smaller resolution during generation.
Therefore we need to transform the minutia locations w.r.t. the original
resolution and not w.r.t. the resolution of the actual image.
"""

MINUTIA_MAP_CHANNELS = 6
MINUTIA_MAP_SIZE = 128


class MinutiaMapLoader(DataLoader):
    @abstractmethod
    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        pass

    def get(self, identifier: Identifier) -> tuple[torch.Tensor, float]:
        """
        @returns : Minutia map generated from the minutia points in the given file,
            and the weight of the map in the minutia map loss: 1.0, or 0.0 if the
            sample has no minutiae.

        A sample without minutiae gets the weight 0.0, so it does not count in the
        minutia map loss (it still counts in the other parts of the loss). A
        fingerprint always has minutiae; an empty list means that the extraction
        failed, that the file is empty or that the pose augmentation moved all
        minutiae out of the image. An empty map with weight 1.0 would teach the
        model that such an image has no minutiae.
        """
        minutiae_loc, minutiae_ori = self.get_minutiae(identifier)
        if len(minutiae_loc) == 0:
            return (
                torch.zeros(
                    (MINUTIA_MAP_CHANNELS, MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE),
                ),
                0.0,
            )
        # Minutiae are often extracted from a different resolution than
        # that of the resized images that serve as input to DeepPrint
        minu_map = create_minutia_map(
            minutiae_loc,
            minutiae_ori,
            in_resolution=get_input_resolution(),
            out_resolution=(MINUTIA_MAP_SIZE, MINUTIA_MAP_SIZE),
            n_layers=MINUTIA_MAP_CHANNELS,
            sigma=1.5,
        )
        return (VTF.to_tensor(minu_map), 1.0)


class PoseTransformedMinutiaMapLoader(MinutiaMapLoader):
    """
    Applies the pose transform of the fingerprint image to its minutiae, so that the
    minutia map matches the image when the pose of the image is augmented.

    The pose is taken from the TransformedImageLoader of the images. It is the pose of
    the last image that was loaded, therefore the image must be loaded directly before
    its minutia map. This is the case when the image dataset comes before the minutia
    map dataset in Dataset.zip, like in the training.

    After the pose transform, the images must either be resized with
    pad_and_resize_to_deepprint_input_size or already have the input size. If the
    images are cut or padded to a fixed size first (the roi argument of
    pad_and_resize_to_deepprint_input_size), give the same roi here.

    If the minutia loader knows the size of the image that its minutiae were
    extracted from (get_frame_size, the header of a .mnt file), it is compared with
    the size of the image that was loaded, see _check_frame_size.

    The alignment of the localization network (Loc* variants) is not applied here,
    because it is only known inside the model. The loss moves the minutia map with
    it, see align_minutia_maps in flx/models/deep_print_loss.py.
    """

    def __init__(
        self,
        minutia_maps: MinutiaMapLoader,
        images: TransformedImageLoader,
        roi: Union[None, tuple[int, int]] = None,
    ):
        """
        @param roi : (height, width) that the images are centre-cropped or padded to
            after the pose transform and before they are resized to the input size.
        """
        self._minutia_maps: MinutiaMapLoader = minutia_maps
        self._images: TransformedImageLoader = images
        self._roi: Union[None, tuple[int, int]] = roi

    def _check_frame_size(self, identifier: Identifier, height: int, width: int) -> None:
        """
        Raises a ValueError if the minutiae were extracted from an image of another
        shape than the one that was loaded for the training.

        Without roi, both are scaled to the input size by their longer side, so an
        image that is the same picture at another size (for example 1000 instead of
        500 pixels per inch) still fits its minutiae: only a different aspect ratio
        is an error. With roi the pixels count, and the sizes must be equal.
        An image that already has the input size is not checked without roi.
        """
        get_frame_size = getattr(self._minutia_maps, "get_frame_size", None)
        if get_frame_size is None:
            return
        if self._roi is None and (height, width) == get_input_resolution():
            # The image loader has already padded and resized the image to the
            # input size (like the MCYT loaders): its shape says nothing
            return
        frame_width, frame_height = get_frame_size(identifier)
        if frame_width <= 0 or frame_height <= 0:
            return
        if self._roi is not None:
            fits = (frame_width, frame_height) == (width, height)
        else:
            scale = frame_width / width
            fits = abs(frame_height - height * scale) <= 1 + scale
        if not fits:
            raise ValueError(
                f"The minutiae of {identifier} were extracted from an image of "
                f"{frame_width}x{frame_height} pixels (width x height), but the "
                f"training loads this image with {width}x{height} pixels"
                + (
                    " and cuts it to a fixed size, which needs equal sizes"
                    if self._roi is not None
                    else ", which is another shape"
                )
                + ". The minutiae would be in the wrong places. Extraction and "
                "training must prepare the images in the same way: use the same "
                "--ppi, --ppi_manifest and --assume_ppi options for "
                "scripts/extract_minutiae.py and for the training, and extract "
                "the minutiae again if the image files changed."
            )

    @property
    def ids(self) -> IdentifierSet:
        return self._minutia_maps.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians.
        """
        locs, oris = self._minutia_maps.get_minutiae(identifier)
        pose, (height, width) = self._images.get_last_pose(identifier)
        if locs.shape[0] == 0:
            return locs, oris
        self._check_frame_size(identifier, height, width)
        if pose is None and self._roi is None:
            return locs, oris

        # The pose transform is applied to the image before it is cut to the roi and
        # resized to the input size
        locs = transform_from_input_size(locs, height, width)
        pad = 0
        if pose is not None:
            locs = pose.transform_coordinates(locs, height, width)
            oris = pose.transform_orientations(oris)
            pad = pose.pad
        locs = transform_to_input_size(
            locs, height + 2 * pad, width + 2 * pad, roi=self._roi
        )

        # Minutiae can be moved out of the image
        input_height, input_width = get_input_resolution()
        inside = (
            (locs[:, 0] >= 0)
            & (locs[:, 0] < input_width)
            & (locs[:, 1] >= 0)
            & (locs[:, 1] < input_height)
        )
        return locs[inside], oris[inside]


class SFingeMinutiaMapLoader(MinutiaMapLoader):
    def __init__(
        self,
        root_dir: str,
    ):
        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <dir>/<subject_id>_<impression_id>.png
            filename, _ = os.path.splitext(filename)  # Remove extension(s)
            subject_id, impression_id = filename.split("_")
            # We must start indexing at 0 instead of 1 to be compatible with pytorch
            return Identifier(int(subject_id) - 1, int(impression_id) - 1)

        self._files = FileIndex(root_dir, ".ist", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        minutiae = decode.load_iso19794(
            self._files.get(identifier), format="19794-2-2005"
        )
        locs = np.array([(m[1], m[2]) for m in minutiae])
        # The decoder returns the orientation in degrees
        oris = np.radians([m[3] for m in minutiae])
        return transform_to_input_size(locs, 560, 416), oris


def _read_mnt_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, "r") as file:
        lines = file.readlines()
        locs = []
        oris = []
        for line_number, line in enumerate(lines[2:], start=3):
            # x, y and angle; further columns (minutia class, score) are not used
            parts = line.split()
            if len(parts) == 0:
                continue
            if len(parts) < 3:
                raise ValueError(
                    f"{filepath}, line {line_number}: expected x, y and angle, "
                    f"found {len(parts)} value(s)"
                )
            locs.append((float(parts[0]), float(parts[1])))
            oris.append(float(parts[2]))
    return np.array(locs, dtype=np.float16), np.array(oris, dtype=np.float16)


class MCYTOpticalMinutiaMapLoader(MinutiaMapLoader):
    def __init__(
        self,
        root_dir: str,
    ):
        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <dir>/<person:04>_<finger>_<impression>.mnt
            _, person, finger, impression = filename.split("_")
            # 12 impressions per finger
            subject = (10 * int(person)) + int(finger)
            return Identifier(subject, int(impression))

        self._files: FileIndex = FileIndex(root_dir, ".mnt", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        locs, oris = _read_mnt_file(self._files.get(identifier))
        return (
            transform_to_input_size(
                locs, original_height=400, original_width=256, roi=(310, 256)
            ),
            oris,
        )


class MCYTCapacitiveMinutiaMapLoader(MinutiaMapLoader):
    def __init__(
        self,
        root_dir: str,
    ):
        def file_to_id_fun(_: str, filename: str) -> Identifier:
            # Pattern: <dir>/<person:04>_<finger>_<impression>.bmp.mnt
            #       or <dir>/<person:04>_<finger>_<impression>.mnt
            if filename.lower().endswith(".bmp"):
                filename = filename[: len(filename) - len(".bmp")]
            _, person, finger, impression = filename.split("_")
            # 12 impressions per finger
            subject = (10 * int(person)) + int(finger)
            return Identifier(subject, int(impression))

        self._files: FileIndex = FileIndex(root_dir, ".mnt", file_to_id_fun)

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get_minutiae(self, identifier: Identifier) -> tuple[np.ndarray, np.ndarray]:
        """
        @returns : Two numpy arrays of equal length. The first array is a 2-D array where each row is
            one coordinate (x, y) of a minutia. The second array is a 1-D array containing the orientation
            in radians (values in range [0.0, 2 * pi]).
        """
        locs, oris = _read_mnt_file(self._files.get(identifier))
        return transform_to_input_size(locs, 300, 300), oris
