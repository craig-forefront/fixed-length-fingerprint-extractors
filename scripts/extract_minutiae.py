"""
Batch minutiae extraction script using the minutiae_extractor (FingerFlow).

This script extracts minutiae from all fingerprint images and saves them
as .mnt files for use in DeepPrint training. It is made for prints from
different sensors and for very large numbers of files.

Prerequisites:
    The `fingerflow` package of the minutiae_extractor repository, e.g.
        pip install -e /path/to/minutiae_extractor
    in a revision whose Extractor has the image_dpi option and accepts None
    for the ClassifyNet and CoreNet paths (the `fingerflow` release on PyPI
    has neither). See AIRGAP_INSTALLATION.md. Pillow is needed to read the
    resolution from the image files. torch is not needed.

Usage:
    python scripts/extract_minutiae.py \
        --input_dir /path/to/fingerprints \
        --output_dir /path/to/minutiae \
        --coarse_net /path/to/CoarseNet.h5 \
        --fine_net /path/to/FineNet.h5 \
        --ppi_manifest /path/to/resolutions.csv

Resolution (ppi = pixels per inch):
    The minutiae extractor only works near 500 ppi and does not rescale. This
    script therefore resamples every image to 500 ppi before the extraction.
    The resolution of a file is taken from, in this order:
        --ppi 1000           all files have this resolution
        --ppi_manifest FILE  CSV with the columns path,ppi[,ridges]; a row is
                             for one file or for a directory (all files below
                             it); paths relative to --input_dir
        the metadata of the image file, if the value is between 200 and 2100
        --assume_ppi 500     for files without a usable value
    A file whose resolution is not known this way fails (it is written to the
    error log); the resolution is never guessed. Images that are at 500 ppi are
    used unchanged.

    The minutiae are written in the pixels of the 500 ppi image, and the header
    of the .mnt file holds its size. scripts/train_500k.py loads the images in
    the same way (same options), so the minutiae fit the training images.

Ridge polarity:
    The extractor expects dark ridges on a light background.
        --ridges dark    the images are like that (default)
        --ridges light   light ridges on a dark background: the images are
                         inverted
        --ridges auto    decided per image from its background; images
                         without background are taken as dark
    A "ridges" column in the manifest sets it per file or directory. Every
    image is checked: one that looks different from what it was taken to be
    is written to extraction_warnings.txt (and still extracted).

Very large file lists:
    --skip_existing              leave out images that already have a .mnt
                                 file; this continues an interrupted run. A
                                 .mnt file is only visible when it is complete.
    --num_shards N --shard_index I
                                 extract every N-th image, starting with image
                                 I (0 .. N-1) of the list. Start one process
                                 per shard, on as many machines as you like.
    --file_list FILE             text file with one image path per line,
                                 relative to --input_dir, instead of scanning
                                 --input_dir. Make it once, so that all shards
                                 use the same list.
    --workers W                  worker processes of this shard

    One bad file never stops the run. Failures are written to
    extraction_errors.txt in the output directory as they happen (with
    shards: extraction_errors.shard-00003-of-00016.txt), one line per file:
    path, tab, reason. The exit status is 1 if the extraction failed for any
    image of this run and 0 otherwise; 2 means that the run did not start.
    The script stops with an exception if the extractor itself can not be
    set up.

    All .mnt files of one output directory must come from the same settings.
    The first run stores them in extraction_settings.json; a run with other
    settings refuses to start (see --allow_settings_change).

Minutia class:
    --classify_net is optional. Without it the minutiae are not classified
    and the class index in the .mnt files is -1. DeepPrint does not use the
    class, and leaving it out saves one network call per minutia. CoreNet is
    never run, because the core points are not written (--core_net is accepted
    and ignored).

--image_dpi is only used to find the direction of the minutiae. The images
are at 500 ppi after the resampling, so leave it at 500.

Output format (.mnt files):
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>   (of the image at 500 ppi)
    Lines 3+: <x> <y> <angle_radians> <class_index> <direction_confidence>
    Last line: # <how the image was prepared>

    x is the column and y the row of the 500 ppi image (y pointing down). The
    angle is the direction of the minutia, the direction vector is
    (cos(angle), sin(angle)) in these axes.
"""

import argparse
import os
import sys
from pathlib import Path

# The flx package of this repository, also when it is not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tqdm

from flx.preparation import minutiae_extraction, sensor_normalization
from flx.preparation.minutiae_extraction import (
    ExtractionConfig,
    SettingsChangedError,
    check_settings,
    iter_file_list,
    iter_image_paths,
    run_extraction,
    select_shard,
    shard_suffix,
)


def extract_single_image(image_path, output_path, model_paths, normalizer, image_dpi=500):
    """
    Extract minutiae from a single image.

    Returns:
        (image_path, success, error_message, warning, num_minutiae)

    An extractor that can not be set up (missing package or model files,
    version without the options used here) is not an error of the image:
    that exception is raised.
    """
    minutiae_extraction.init_worker(ExtractionConfig(model_paths, normalizer, image_dpi))
    return minutiae_extraction.extract_one((str(image_path), str(output_path)))


def find_images(input_dir: str, extensions: tuple = minutiae_extraction.IMAGE_EXTENSIONS):
    """Find all image files in directory recursively, in a fixed order."""
    return [Path(input_dir) / path for path in iter_image_paths(input_dir, extensions)]


def get_output_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Convert image path to corresponding .mnt output path."""
    relative = Path(image_path).relative_to(input_dir)
    return Path(output_dir) / relative.with_suffix('.mnt')


def main() -> int:
    """Returns the exit status: 1 if the extraction failed for any image, else 0."""
    parser = argparse.ArgumentParser(
        description="Extract minutiae from fingerprint images using FingerFlow",
        epilog="See the docstring at the top of this script and the section "
               "'Dataset Preparation' of TRAINING_PIPELINE_GUIDE.md.",
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing fingerprint images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save .mnt files")
    parser.add_argument("--file_list", type=str, default=None,
                        help="Text file with one image path per line, relative to "
                             "--input_dir (default: all images below --input_dir)")
    parser.add_argument("--coarse_net", type=str, required=True,
                        help="Path to CoarseNet model weights")
    parser.add_argument("--fine_net", type=str, required=True,
                        help="Path to FineNet model weights")
    parser.add_argument("--classify_net", type=str, default=None,
                        help="Path to ClassifyNet model weights. Optional: without it the "
                             "minutiae are not classified (class index -1), which DeepPrint "
                             "does not need, and the extraction is faster")
    parser.add_argument("--core_net", type=str, default=None,
                        help="Not used: the core points are not written, so CoreNet is not run")
    parser.add_argument("--image_dpi", type=float, default=500,
                        help="Used to find the direction of the minutiae only. The images are "
                             "at 500 ppi after the resampling: leave it at 500 (default)")
    sensor_normalization.add_arguments(parser)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1, GPU memory limited)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have .mnt files (continues a run)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the image list into this many shards (default: 1)")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="The shard that this process extracts, 0 .. num_shards-1 (default: 0)")
    parser.add_argument("--allow_settings_change", action="store_true",
                        help="Start although the output directory was written with other "
                             "settings, and store the new ones. Only together with a full "
                             "new extraction (no --skip_existing), else the directory "
                             "holds files from two settings")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be from 0 to --num_shards - 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.allow_settings_change and args.skip_existing:
        parser.error("--allow_settings_change can not be combined with --skip_existing: "
                     "the files that are skipped were extracted with the former settings")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    model_paths = {
        'coarse_net': args.coarse_net,
        'fine_net': args.fine_net,
        'classify_net': args.classify_net,
    }
    try:
        normalizer = sensor_normalization.from_arguments(args, str(input_dir))
    except (ValueError, OSError) as error:
        parser.error(str(error))

    settings = dict(normalizer.settings())
    settings["image_dpi"] = args.image_dpi
    settings["classified"] = args.classify_net is not None
    try:
        check_settings(str(output_dir), settings, args.allow_settings_change)
    except SettingsChangedError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.file_list is not None:
        print(f"Reading the image list {args.file_list}")
        relative_paths = iter_file_list(args.file_list)
    else:
        print(f"Scanning {input_dir} for images...")
        relative_paths = iter_image_paths(str(input_dir))
    relative_paths = select_shard(relative_paths, args.num_shards, args.shard_index)
    jobs = (
        (str(input_dir / path), str(get_output_path(input_dir / path, input_dir, output_dir)))
        for path in relative_paths
    )

    suffix = shard_suffix(args.num_shards, args.shard_index)
    error_log = output_dir / f"extraction_errors{suffix}.txt"
    warning_log = output_dir / f"extraction_warnings{suffix}.txt"

    print(f"Images:   {normalizer.describe()}")
    print(f"Classes:  {'ClassifyNet' if args.classify_net else 'not classified (class index -1)'}")
    if args.num_shards > 1:
        print(f"Shard:    {args.shard_index} of {args.num_shards} (index 0 .. {args.num_shards - 1})")

    with tqdm.tqdm(desc="Extracting minutiae", unit="img") as progress:
        summary = run_extraction(
            ExtractionConfig(model_paths, normalizer, args.image_dpi),
            jobs,
            str(error_log),
            str(warning_log),
            workers=args.workers,
            skip_existing=args.skip_existing,
            progress=progress,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"  Successful: {summary.extracted:,}")
    print(f"  Skipped:    {summary.skipped:,} (already had a .mnt file)")
    print(f"  Failed:     {summary.failed:,}")
    print(f"  Warnings:   {summary.warnings:,}")
    print(f"  Extracted without any minutia: {summary.without_minutiae:,}")
    print(f"  Output dir: {output_dir}")
    print(f"{'=' * 60}")

    if summary.warnings:
        print(f"\nWarnings (files that were extracted but look wrong): {warning_log}")
    if summary.failed:
        print(f"\nFirst {len(summary.first_failures)} errors:")
        for path, error in summary.first_failures:
            print(f"  {path}: {error}")
        print(f"\nFull error log saved to: {error_log}")
        print("Files in the error log have no .mnt file. After the cause is removed, "
              "run the same command with --skip_existing to extract only them.")

    return 1 if summary.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
