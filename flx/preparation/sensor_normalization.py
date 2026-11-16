"""
Brings fingerprint images from different sensors into one form: 8-bit
grayscale, 500 pixels per inch (ppi), dark ridges on a light background.

Why: the minutiae extractor only works near 500 ppi and expects dark ridges,
and it does not rescale or invert anything. Sensors deliver 500, 1000 or other
resolutions, and some deliver light ridges on a dark background.

The minutiae extraction (scripts/extract_minutiae.py) and the image loader of
the training (scripts/train_500k.py) both load an image file with
SensorNormalizer.load. The minutiae are therefore found in exactly the pixels
that the training loads, and the size in the header of a .mnt file is the size
of the image that the training loads.

Where the resolution of a file comes from, in this order:
    1. the explicit value for all files (ppi argument),
    2. the manifest, a CSV file with a row for the file or for one of the
       directories above it,
    3. the metadata of the image file (pHYs chunk of PNG, resolution tags of
       TIFF, JFIF density of JPEG, pixels per metre of BMP), if the value is
       plausible for a fingerprint sensor,
    4. the assumption given by the user (assume_ppi argument).
If none of them gives a value, UnknownResolutionError is raised for the file.
A resolution is never guessed.

Manifest format (CSV with a header line):
    path,ppi,ridges
    sensor_a,1000,dark          <- a directory: all files below it
    sensor_b/7_1.png,385,light  <- one file
The paths are relative to the root directory of the dataset. The row of the
file itself wins over a directory row, a closer directory over one further up.
The "ridges" column is optional, and cells can be left empty.
"""

import csv
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

# Resolution that the minutiae extractor needs, and that the training uses
TARGET_PPI = 500.0

# Resolutions in image metadata outside of this range are not believed. Image
# editors write 72 or 96 into files without knowing the sensor; fingerprint
# sensors are between about 250 and 2000 ppi.
PLAUSIBLE_METADATA_PPI = (200.0, 2100.0)

# The two pixel axes of a file may differ by this share before the file is
# rejected as having non-square pixels
MAX_ASPECT_DIFFERENCE = 0.01

RIDGES_DARK = "dark"  # dark ridges on a light background (ink, most sensors)
RIDGES_LIGHT = "light"  # light ridges on a dark background
RIDGES_AUTO = "auto"  # decide per image with estimate_ridge_polarity
RIDGE_CHOICES = (RIDGES_DARK, RIDGES_LIGHT, RIDGES_AUTO)

# estimate_ridge_polarity: side of the square blocks in pixels at 500 ppi
_POLARITY_BLOCK = 16
# Smallest share of the blocks that must be background for a decision
_POLARITY_MIN_BACKGROUND_SHARE = 0.02
# Smallest difference between background and print in gray levels (0 to 255)
_POLARITY_MIN_DIFFERENCE = 10.0


def resolution_source_label(ppi: float, origin: str) -> str:
    """Key of the counts per resolution source, e.g. "metadata, 300 ppi"."""
    return f"{origin}, {ppi:g} ppi"


# What the origins mean, for messages
_ORIGIN_TEXT = {
    "argument": "--ppi",
    "manifest": "manifest",
    "metadata": "metadata of the file",
    "assumed": "--assume_ppi",
}


def describe_resolution_sources(
    counts: dict, assume_ppi: Optional[float] = None, target_ppi: float = TARGET_PPI
) -> list[str]:
    """
    Lines that say how many files took their resolution from where: one line
    per source and value (the keys of counts are those of
    resolution_source_label), the most frequent first. If files took a
    resolution from their metadata although an assumption was given, a note
    says that the assumption does not override metadata: a corpus whose files
    carry a wrong but plausible value (300 or 600 written by an image
    program) would be resampled by the wrong factor without any other sign.
    """
    lines = []
    from_metadata = 0
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        origin, _, value = key.partition(", ")
        resampled = "" if value == f"{target_ppi:g} ppi" else f" (resampled to {target_ppi:g} ppi)"
        lines.append(f"  {count:>12,}  {value} from {_ORIGIN_TEXT.get(origin, origin)}{resampled}")
        if origin == "metadata":
            from_metadata += count
    if from_metadata and assume_ppi is not None:
        lines.append(
            f"  Note: {from_metadata:,} file(s) took the resolution from their own "
            f"metadata, not from --assume_ppi {assume_ppi:g}. --assume_ppi is only "
            "used for files without a plausible value; it does not override "
            "metadata. If every file has one known resolution, give it with "
            "--ppi; if the metadata of some files is wrong, list them in a manifest."
        )
    return lines


class UnknownResolutionError(ValueError):
    """The resolution of an image file is not known."""


class ImageReadError(ValueError):
    """An image file could not be read."""


def read_metadata_ppi(path: str) -> Optional[float]:
    """
    The resolution in pixels per inch that the image file states about itself,
    or None if it states none. The value is returned as it is in the file,
    also if it is implausible.

    Raises ValueError if the file states different resolutions for the two
    axes (non-square pixels), which the preparation does not handle.
    """
    # Pillow reads the header only. It is imported here, so that everything
    # that does not use metadata works without it.
    from PIL import Image

    try:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
    except Exception:  # unreadable for Pillow: the caller reads the pixels with OpenCV
        return None
    if dpi is None:
        return None
    try:
        x_ppi, y_ppi = float(dpi[0]), float(dpi[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(x_ppi) and np.isfinite(y_ppi)) or x_ppi <= 0 or y_ppi <= 0:
        return None
    if abs(x_ppi - y_ppi) > MAX_ASPECT_DIFFERENCE * max(x_ppi, y_ppi):
        raise ValueError(
            f"The file states {x_ppi:g} x {y_ppi:g} pixels per inch. Pixels that "
            "are not square are not supported; resample the file first."
        )
    ppi = (x_ppi + y_ppi) / 2
    # PNG and BMP store whole pixels per metre, which turns 500 ppi into
    # 499.9998: take the nearest tenth if the value is that close to it
    if abs(ppi - round(ppi, 1)) < 0.02:
        ppi = round(ppi, 1)
    return ppi


def resampled_size(width: int, height: int, ppi: float, target_ppi: float = TARGET_PPI):
    """(width, height) of an image of the given resolution at the target resolution."""
    scale = target_ppi / ppi
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resample_to_ppi(
    image: np.ndarray, ppi: float, target_ppi: float = TARGET_PPI
) -> np.ndarray:
    """
    Resamples the image from its resolution to the target resolution. An image
    whose size does not change is returned as it is, so 500 ppi images stay
    bit-identical.

    Shrinking averages the source pixels (INTER_AREA: 1000 to 500 ppi is the
    mean of 2x2 pixels), enlarging interpolates (INTER_CUBIC).
    """
    height, width = image.shape[:2]
    new_width, new_height = resampled_size(width, height, ppi, target_ppi)
    if (new_width, new_height) == (width, height):
        return image
    interpolation = cv2.INTER_AREA if target_ppi < ppi else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def estimate_ridge_polarity(image: np.ndarray) -> Optional[str]:
    """
    Estimates from the image whether the ridges are dark ("dark") or light
    ("light"). Returns None if the image does not allow a decision.

    The estimate uses the background: where no finger touched, a sensor with
    dark ridges delivers a light area and a sensor with light ridges a dark
    one. The image is cut into blocks of 16 pixels. Blocks with little contrast
    are background, blocks with much contrast are print, and the gray level of
    the two is compared. An image that the print fills completely has no
    background and gives None.
    """
    block = _POLARITY_BLOCK
    rows, columns = image.shape[0] // block, image.shape[1] // block
    if rows < 2 or columns < 2:
        return None
    blocks = (
        image[: rows * block, : columns * block]
        .astype(np.float32)
        .reshape(rows, block, columns, block)
    )
    contrast = blocks.std(axis=(1, 3))
    gray = blocks.mean(axis=(1, 3))
    high_contrast = float(np.percentile(contrast, 90))
    if high_contrast < 5.0:  # no ridges anywhere
        return None
    background = contrast < 0.25 * high_contrast
    print_area = contrast > 0.5 * high_contrast
    if (
        background.mean() < _POLARITY_MIN_BACKGROUND_SHARE
        or background.sum() < 4
        or print_area.sum() < 4
    ):
        return None
    difference = float(np.median(gray[background]) - np.median(gray[print_area]))
    if difference > _POLARITY_MIN_DIFFERENCE:
        return RIDGES_DARK
    if difference < -_POLARITY_MIN_DIFFERENCE:
        return RIDGES_LIGHT
    return None


def _manifest_key(path: str) -> str:
    key = os.path.normpath(path.strip()).replace(os.sep, "/")
    return "" if key == "." else key


def read_manifest(path: str) -> dict[str, tuple[Optional[float], Optional[str]]]:
    """
    Reads a manifest (see the module docstring). Returns a dictionary from the
    relative path of a file or directory to (ppi or None, ridges or None).
    """
    entries: dict[str, tuple[Optional[float], Optional[str]]] = {}
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        if "path" not in columns or not ({"ppi", "ridges"} & set(columns)):
            raise ValueError(
                f"The manifest {path} needs a header line with the column 'path' "
                f"and at least one of 'ppi' and 'ridges', found: {columns}"
            )
        for number, row in enumerate(reader, start=2):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if not row.get("path"):
                continue
            ppi = None
            if row.get("ppi"):
                try:
                    ppi = float(row["ppi"])
                except ValueError:
                    ppi = float("nan")
                if not (np.isfinite(ppi) and ppi > 0):
                    raise ValueError(
                        f"{path}, line {number}: ppi must be a positive number, "
                        f"got '{row['ppi']}'"
                    )
            ridges = row.get("ridges") or None
            if ridges is not None and ridges not in (RIDGES_DARK, RIDGES_LIGHT):
                raise ValueError(
                    f"{path}, line {number}: ridges must be '{RIDGES_DARK}' or "
                    f"'{RIDGES_LIGHT}', got '{ridges}'"
                )
            entries[_manifest_key(row["path"])] = (ppi, ridges)
    return entries


@dataclass(frozen=True)
class PreparedImageInfo:
    """What SensorNormalizer.load did with one image file."""

    source_width: int
    source_height: int
    width: int  # of the prepared image, at the target resolution
    height: int
    ppi: float  # resolution of the file
    ppi_origin: str  # "argument", "manifest", "metadata" or "assumed"
    ridges: str  # "dark" or "light": what the file was taken to hold
    ridges_origin: str  # "argument", "manifest", "estimated" or "default"
    # Result of estimate_ridge_polarity ("dark", "light" or None for no
    # decision); "not checked" if the estimate was not made
    polarity_estimate: Optional[str] = "not checked"

    @property
    def polarity_warning(self) -> Optional[str]:
        """A message if the image looks different from what it was taken to be."""
        if self.polarity_estimate in (None, "not checked"):
            if self.polarity_estimate is None and self.ridges_origin == "default":
                return (
                    "the ridge polarity could not be estimated (no background "
                    "in the image); the image was taken to have dark ridges"
                )
            return None
        if self.polarity_estimate != self.ridges:
            return (
                f"the image was taken to have {self.ridges} ridges "
                f"({self.ridges_origin}), but it looks like {self.polarity_estimate} ridges"
            )
        return None

    @property
    def resolution_source(self) -> str:
        """Where the resolution came from and its value, e.g. "metadata, 300 ppi"."""
        return resolution_source_label(self.ppi, self.ppi_origin)

    def describe(self) -> str:
        """One line for the end of a .mnt file."""
        return (
            f"source {self.source_width}x{self.source_height} at {self.ppi:g} ppi "
            f"({self.ppi_origin}), ridges {self.ridges} ({self.ridges_origin}), "
            f"prepared {self.width}x{self.height}"
        )


class SensorNormalizer:
    """
    Loads image files as 8-bit grayscale at the target resolution with dark
    ridges. See the module docstring for where resolution and polarity come
    from.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        ppi: Optional[float] = None,
        manifest: Union[None, str, dict] = None,
        assume_ppi: Optional[float] = None,
        ridges: str = RIDGES_DARK,
        target_ppi: float = TARGET_PPI,
    ):
        """
        Args:
            root_dir: Root directory of the dataset; the paths in the manifest
                are relative to it. Needed with a manifest.
            ppi: Resolution of all files. Wins over manifest and metadata.
            manifest: Path of a manifest file, or the dictionary that
                read_manifest returns.
            assume_ppi: Resolution to assume for files whose resolution is not
                known otherwise.
            ridges: "dark" (the default), "light" or "auto" for the files that
                have no ridges value in the manifest. "auto" estimates the
                polarity of each image and takes "dark" where the estimate
                gives no decision.
            target_ppi: Resolution of the prepared images.
        """
        for name, value in (("ppi", ppi), ("assume_ppi", assume_ppi)):
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value}")
        if not (np.isfinite(target_ppi) and target_ppi > 0):
            raise ValueError(f"target_ppi must be a positive number, got {target_ppi}")
        if ridges not in RIDGE_CHOICES:
            raise ValueError(f"ridges must be one of {RIDGE_CHOICES}, got '{ridges}'")
        if manifest is not None and root_dir is None:
            raise ValueError("A manifest needs the root directory of the dataset")

        self._root_dir = None if root_dir is None else os.path.abspath(root_dir)
        self._ppi = None if ppi is None else float(ppi)
        self._assume_ppi = None if assume_ppi is None else float(assume_ppi)
        self._ridges = ridges
        self._target_ppi = float(target_ppi)
        self._manifest_path = None
        self._manifest_sha256 = None
        if isinstance(manifest, (str, os.PathLike)):
            self._manifest_path = str(manifest)
            with open(manifest, "rb") as file:
                self._manifest_sha256 = hashlib.sha256(file.read()).hexdigest()
            manifest = read_manifest(self._manifest_path)
        self._manifest: dict[str, tuple[Optional[float], Optional[str]]] = manifest or {}

    @property
    def target_ppi(self) -> float:
        return self._target_ppi

    def settings(self) -> dict:
        """
        The settings as a dictionary of plain values, to be stored with the
        results. The manifest is named by the SHA-256 of its content, so that
        the same manifest under another path counts as the same setting.
        """
        return {
            "target_ppi": self._target_ppi,
            "ppi": self._ppi,
            "ppi_manifest_sha256": self._manifest_sha256,
            "ppi_manifest_entries": len(self._manifest),
            "assume_ppi": self._assume_ppi,
            "ridges": self._ridges,
        }

    @property
    def assume_ppi(self) -> Optional[float]:
        return self._assume_ppi

    def describe(self) -> str:
        """The settings in one sentence."""
        sources = []
        if self._ppi is not None:
            sources.append(f"{self._ppi:g} ppi for all files")
        else:
            if self._manifest:
                sources.append(f"manifest ({len(self._manifest)} rows)")
            sources.append("file metadata")
            if self._assume_ppi is not None:
                sources.append(f"else assumed {self._assume_ppi:g} ppi")
            else:
                sources.append("else the file fails")
        return (
            f"resampled to {self._target_ppi:g} ppi; resolution from "
            + ", ".join(sources)
            + f"; ridges {self._ridges}"
        )

    def _manifest_entry(self, path: str):
        """Yields the manifest rows for the file: its own, then those of the directories above."""
        if not self._manifest:
            return
        relative = os.path.relpath(os.path.abspath(path), self._root_dir)
        key = _manifest_key(relative)
        if key.startswith("../") or key == "..":
            return  # not below the root directory: no row can apply
        while True:
            if key in self._manifest:
                yield self._manifest[key]
            if key == "":
                return
            key = key.rpartition("/")[0]

    def resolve_ppi(self, path: str) -> tuple[float, str]:
        """
        Returns (resolution of the file in ppi, where it comes from).
        Raises UnknownResolutionError if it is not known.
        """
        if self._ppi is not None:
            return self._ppi, "argument"
        for ppi, _ in self._manifest_entry(path):
            if ppi is not None:
                return ppi, "manifest"
        metadata_ppi = read_metadata_ppi(path)
        low, high = PLAUSIBLE_METADATA_PPI
        if metadata_ppi is not None and low <= metadata_ppi <= high:
            return metadata_ppi, "metadata"
        if self._assume_ppi is not None:
            return self._assume_ppi, "assumed"
        if metadata_ppi is None:
            found = "The file holds no resolution"
        else:
            found = (
                f"The file states {metadata_ppi:g} ppi, which is outside of the "
                f"plausible range {low:g} to {high:g} for a fingerprint sensor "
                "and is not used"
            )
        raise UnknownResolutionError(
            f"Resolution unknown. {found}, and neither an explicit resolution "
            "nor a manifest row gives one. Give the resolution of all files "
            "(--ppi), list it per file or directory in a manifest "
            "(--ppi_manifest), or state an assumption for files like this one "
            "(--assume_ppi)."
        )

    def resolve_ridges(self, path: str) -> tuple[str, str]:
        """
        Returns (ridges setting for the file, where it comes from). The setting
        can be "auto"; load then estimates it from the image.
        """
        for _, ridges in self._manifest_entry(path):
            if ridges is not None:
                return ridges, "manifest"
        return self._ridges, "argument"

    def load(
        self, path: str, check_polarity: bool = False
    ) -> tuple[np.ndarray, PreparedImageInfo]:
        """
        Loads an image file and prepares it.

        Args:
            path: Path of the image file
            check_polarity: Also estimate the ridge polarity when it is given,
                so that the returned info can warn about a contradiction.

        Returns:
            (image, info): the image as a uint8 array (height, width) at the
            target resolution with dark ridges, and what was done to get it.

        Raises ImageReadError if the file can not be read as an image and
        UnknownResolutionError if its resolution is not known.
        """
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ImageReadError(f"Could not read the file as an image: {path}")
        source_height, source_width = image.shape[:2]

        ppi, ppi_origin = self.resolve_ppi(str(path))
        image = resample_to_ppi(image, ppi, self._target_ppi)

        ridges, ridges_origin = self.resolve_ridges(str(path))
        estimate = "not checked"
        if ridges == RIDGES_AUTO or check_polarity:
            # On the resampled image, so that the blocks have one physical size
            estimate = estimate_ridge_polarity(image)
        if ridges == RIDGES_AUTO:
            if estimate is None:
                ridges, ridges_origin = RIDGES_DARK, "default"
            else:
                ridges, ridges_origin = estimate, "estimated"
        if ridges == RIDGES_LIGHT:
            image = 255 - image

        info = PreparedImageInfo(
            source_width=source_width,
            source_height=source_height,
            width=image.shape[1],
            height=image.shape[0],
            ppi=ppi,
            ppi_origin=ppi_origin,
            ridges=ridges,
            ridges_origin=ridges_origin,
            polarity_estimate=estimate,
        )
        return image, info


def add_arguments(parser) -> None:
    """Adds the options of the preparation to an argparse parser."""
    group = parser.add_argument_group(
        "image preparation",
        "All images are resampled to 500 pixels per inch (ppi) and brought to "
        "dark ridges. The resolution of a file is taken from --ppi, else from "
        "the manifest, else from the metadata of the file, else from "
        "--assume_ppi. A file whose resolution is not known this way fails.",
    )
    group.add_argument("--ppi", type=float, default=None,
                       help="Resolution of ALL images in pixels per inch. Use it only if "
                            "every file has this resolution. It wins over the manifest and "
                            "over the metadata of the files, so it is the option for a "
                            "corpus of one known resolution")
    group.add_argument("--ppi_manifest", type=str, default=None,
                       help="CSV file with the columns path,ppi[,ridges]: resolution (and "
                            "ridge polarity, dark or light) per file or per directory, "
                            "paths relative to the image directory")
    group.add_argument("--assume_ppi", type=float, default=None,
                       help="Resolution to assume for files that state none (or one "
                            "outside of 200 to 2100 ppi) and have no manifest row. Without "
                            "it such files fail. It does NOT override a plausible value in "
                            "the metadata of a file: a 500 ppi file that states 300 or 600 "
                            "is resampled by that value. Use --ppi if all files have one "
                            "known resolution")
    group.add_argument("--ridges", type=str, default=RIDGES_DARK, choices=RIDGE_CHOICES,
                       help="What the images hold: dark ridges on a light background "
                            "(default), light ridges on a dark background (these images "
                            "are inverted), or auto to estimate it per image from its "
                            "background. A ridges column in the manifest wins")


def from_arguments(args, root_dir: str) -> SensorNormalizer:
    """The SensorNormalizer for the options that add_arguments added."""
    return SensorNormalizer(
        root_dir=root_dir,
        ppi=args.ppi,
        manifest=args.ppi_manifest,
        assume_ppi=args.assume_ppi,
        ridges=args.ridges,
    )
