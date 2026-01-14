"""
Batch minutiae extraction script using tourmii/minutiae_extractor (FingerFlow).

This script extracts minutiae from all fingerprint images and saves them
as .mnt files for use in DeepPrint training.

Prerequisites:
    pip install fingerflow

Usage:
    python scripts/extract_minutiae.py \
        --input_dir /path/to/fingerprints \
        --output_dir /path/to/minutiae \
        --coarse_net /path/to/CoarseNet.h5 \
        --fine_net /path/to/FineNet.h5 \
        --classify_net /path/to/ClassifyNet.h5 \
        --core_net /path/to/CoreNet.h5

Output format (.mnt files):
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians>
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import tqdm


def extract_single_image(args_tuple):
    """
    Extract minutiae from a single image.

    Args:
        args_tuple: (image_path, output_path, model_paths)

    Returns:
        (image_path, success, error_message)
    """
    image_path, output_path, model_paths = args_tuple

    try:
        # Import inside function for multiprocessing compatibility
        from fingerflow.extractor import Extractor

        # Initialize extractor (cached per process)
        if not hasattr(extract_single_image, '_extractor'):
            extract_single_image._extractor = Extractor(
                model_paths['coarse_net'],
                model_paths['fine_net'],
                model_paths['classify_net'],
                model_paths['core_net'],
            )
        extractor = extract_single_image._extractor

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return (image_path, False, "Could not load image")

        height, width = image.shape[:2]

        # Extract minutiae
        results = extractor.extract_minutiae(image)
        minutiae_df = results.minutiae

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save in .mnt format
        with open(output_path, 'w') as f:
            f.write(f"{len(minutiae_df)}\n")
            f.write(f"{width} {height}\n")
            for _, row in minutiae_df.iterrows():
                f.write(f"{row['x']} {row['y']} {row['angle']}\n")

        return (image_path, True, None)

    except Exception as e:
        return (image_path, False, str(e))


def find_images(input_dir: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    """Find all image files in directory recursively."""
    input_path = Path(input_dir)
    images = []
    for ext in extensions:
        images.extend(input_path.rglob(f"*{ext}"))
        images.extend(input_path.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def get_output_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Convert image path to corresponding .mnt output path."""
    relative = image_path.relative_to(input_dir)
    return output_dir / relative.with_suffix('.mnt')


def main():
    parser = argparse.ArgumentParser(
        description="Extract minutiae from fingerprint images using FingerFlow"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing fingerprint images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save .mnt files")
    parser.add_argument("--coarse_net", type=str, required=True,
                        help="Path to CoarseNet model weights")
    parser.add_argument("--fine_net", type=str, required=True,
                        help="Path to FineNet model weights")
    parser.add_argument("--classify_net", type=str, required=True,
                        help="Path to ClassifyNet model weights")
    parser.add_argument("--core_net", type=str, required=True,
                        help="Path to CoreNet model weights")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, GPU memory limited)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have .mnt files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    model_paths = {
        'coarse_net': args.coarse_net,
        'fine_net': args.fine_net,
        'classify_net': args.classify_net,
        'core_net': args.core_net,
    }

    # Find all images
    print(f"Scanning {input_dir} for images...")
    images = find_images(input_dir)
    print(f"Found {len(images):,} images")

    # Prepare work items
    work_items = []
    skipped = 0
    for image_path in images:
        output_path = get_output_path(image_path, input_dir, output_dir)
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        work_items.append((image_path, str(output_path), model_paths))

    if skipped > 0:
        print(f"Skipping {skipped:,} images with existing .mnt files")

    print(f"Processing {len(work_items):,} images...")

    if len(work_items) == 0:
        print("Nothing to process!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    success_count = 0
    error_count = 0
    errors = []

    if args.workers == 1:
        # Single process (simpler, better for GPU)
        for item in tqdm.tqdm(work_items, desc="Extracting minutiae"):
            image_path, success, error = extract_single_image(item)
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append((image_path, error))
    else:
        # Multi-process (for CPU-based extraction)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_single_image, item): item for item in work_items}
            for future in tqdm.tqdm(as_completed(futures), total=len(work_items), desc="Extracting minutiae"):
                image_path, success, error = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append((image_path, error))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"  Successful: {success_count:,}")
    print(f"  Failed:     {error_count:,}")
    print(f"  Output dir: {output_dir}")
    print(f"{'=' * 60}")

    if errors:
        print(f"\nFirst 10 errors:")
        for path, error in errors[:10]:
            print(f"  {path}: {error}")

        # Save error log
        error_log = output_dir / "extraction_errors.txt"
        with open(error_log, 'w') as f:
            for path, error in errors:
                f.write(f"{path}\t{error}\n")
        print(f"\nFull error log saved to: {error_log}")


if __name__ == "__main__":
    main()
