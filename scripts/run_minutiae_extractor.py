"""
Run the minutiae extractor (package fingerflow of the minutiae_extractor
repository) on a directory of fingerprint images.

Development tool for comparing minutiae extractors. It is NOT part of the
training route; the route is scripts/extract_minutiae.py (see
docs/training-guide.md). It looks for a copy of the minutiae_extractor
repository in <repository>/minutiae_extractor-main (not part of this
repository) and puts its src directory first on the Python path; --models_dir
defaults to its models directory. This script was not run for the documents in
docs/.

Uses the Extractor of that package, i.e. MinutiaeNet (CoarseNet + FineNet),
the image-based refinement of the minutia direction and ClassifyNet, which
assigns one of six minutiae types to every extracted point (left out with
--no_classify).  CoreNet is not run, because the core points are not written.

The images are prepared like in scripts/extract_minutiae.py (see there and
flx/preparation/sensor_normalization.py): the minutiae extractor only works
near 500 ppi, so every image is resampled to 500 ppi first.  The resolution
of a file comes from --ppi, else from --ppi_manifest, else from the metadata
of the file, else from --assume_ppi; a file whose resolution is not known
fails.  --ridges light inverts images with light ridges.  --image_dpi is only
used to find the direction of the minutiae; leave it at 500.

The exit status is 1 if the extraction failed for any image.  For large
datasets use scripts/extract_minutiae.py, which can continue a run and work in
shards.

Output .mnt format (compatible with scripts/compare_minutiae_outputs.py):
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>   (of the image at 500 ppi)
    Lines 3+: <x> <y> <angle_radians> <class_index> <direction_confidence>
              class index: 0=ending  1=bifurcation  2=fragment
                           3=enclosure  4=crossbar  5=other
                           -1=not classified
    Last line: # <how the image was prepared>

    x is the column and y the row of the 500 ppi image (y pointing down). The
    angle is the direction of the minutia, the direction vector is
    (cos(angle), sin(angle)) in these axes.

Usage:
    python scripts/run_minutiae_extractor.py \\
        --input_dir MinutiaeNet-master/Dataset/CoarseNet_test/img_files \\
        --output_dir /tmp/me_mnt \\
        --models_dir minutiae_extractor-main/models \\
        --assume_ppi 500
"""

import argparse
import glob
import os
import sys

# ---------------------------------------------------------------------------
# Add minutiae_extractor-main/src to sys.path so the *local* fingerflow
# package shadows any pip-installed version.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ME_SRC = os.path.join(os.path.dirname(_HERE),
                        'minutiae_extractor-main', 'src')
if _ME_SRC not in sys.path:
    sys.path.insert(0, _ME_SRC)

# The flx package of this repository, also when it is not installed
_REPOSITORY = os.path.dirname(_HERE)
if _REPOSITORY not in sys.path:
    sys.path.insert(0, _REPOSITORY)

from flx.preparation import sensor_normalization
from flx.preparation.minutiae_extraction import extract_to_file, make_extractor


CLASS_NAMES = {
    0: 'ending',
    1: 'bifurcation',
    2: 'fragment',
    3: 'enclosure',
    4: 'crossbar',
    5: 'other',
}


def process_image(extractor, image_path, output_path, normalizer):
    """Prepare one image, extract its minutiae and write them to output_path.

    normalizer: the SensorNormalizer that loads the image (resolution and
    ridge polarity, see flx/preparation/sensor_normalization.py).

    Returns the minutiae DataFrame of the extractor.
    """
    df, info = extract_to_file(extractor, normalizer, image_path, output_path)
    if info.polarity_warning:
        print('WARNING %s: %s' % (image_path, info.polarity_warning))
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Run minutiae_extractor-main (ModernFingerFlow) on fingerprint images.'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing fingerprint images')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save .mnt output files')
    parser.add_argument('--models_dir',
                        default=os.path.join(os.path.dirname(_HERE),
                                             'minutiae_extractor-main', 'models'),
                        help='Directory containing model weight files '
                             '(default: minutiae_extractor-main/models)')
    parser.add_argument('--ext', default='.bmp',
                        help='Image file extension (default: .bmp)')
    parser.add_argument('--image_dpi', type=float, default=500,
                        help='Used to find the direction of the minutiae only. The images are '
                             'at 500 ppi after the resampling: leave it at 500 (default)')
    parser.add_argument('--no_classify', action='store_true',
                        help='Do not run ClassifyNet: the class index is -1 and the '
                             'extraction is faster')
    sensor_normalization.add_arguments(parser)
    args = parser.parse_args()

    try:
        normalizer = sensor_normalization.from_arguments(args, args.input_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    models_dir = args.models_dir
    coarse_net  = os.path.join(models_dir, 'CoarseNet.h5')
    fine_net    = os.path.join(models_dir, 'FineNet.h5')
    classify_net = os.path.join(models_dir, 'ClassifyNet_6_classes.h5')
    if args.no_classify:
        classify_net = None

    for path, label in [(coarse_net, 'CoarseNet'), (fine_net, 'FineNet'),
                        (classify_net, 'ClassifyNet')]:
        if path is None:
            continue
        if not os.path.isfile(path):
            print('ERROR: %s not found: %s' % (label, path))
            sys.exit(1)

    # The extractor is imported after the path checks (TF startup is slow)
    print('Loading Extractor...')
    extractor = make_extractor(
        {'coarse_net': coarse_net, 'fine_net': fine_net, 'classify_net': classify_net},
        image_dpi=args.image_dpi)
    print('Images: %s' % normalizer.describe())

    print('Models loaded.')

    os.makedirs(args.output_dir, exist_ok=True)

    ext = args.ext if args.ext.startswith('.') else '.' + args.ext
    image_files = sorted(glob.glob(os.path.join(args.input_dir, '*' + ext)))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(args.input_dir, '*' + ext.upper())))
    if not image_files:
        print('No images found in %s with extension %s' % (args.input_dir, ext))
        sys.exit(1)

    print('Found %d images. Processing...' % len(image_files))

    success, errors = 0, []
    for i, image_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output_dir, basename + '.mnt')
        try:
            df = process_image(extractor, image_path, output_path, normalizer)
            cls_info = (
                ' (endings=%d bifurcs=%d)' % (
                    (df['class'] == 0).sum(), (df['class'] == 1).sum())
                if len(df) > 0 and not args.no_classify else '')
            print('[%d/%d] %s — %d minutiae%s' % (
                i + 1, len(image_files), basename, len(df), cls_info))
            success += 1
        except Exception as e:
            import traceback
            print('[%d/%d] ERROR %s: %s' % (i + 1, len(image_files), basename, e))
            traceback.print_exc()
            errors.append((basename, str(e)))

    print('\n' + '=' * 40)
    print('Done. %d/%d succeeded.' % (success, len(image_files)))
    if errors:
        for name, err in errors:
            print('  ERROR %s: %s' % (name, err))
    print('Output: %s' % args.output_dir)
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
