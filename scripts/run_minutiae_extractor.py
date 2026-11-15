"""
Run minutiae_extractor-main (modernised FingerFlow) on a directory of
fingerprint images.

Uses the Extractor of that package, i.e. MinutiaeNet (CoarseNet + FineNet),
the image-based refinement of the minutia direction and ClassifyNet, which
assigns one of six minutiae types to every extracted point.  The core points
found by CoreNet are not written.

Feed the images in their native resolution.  --image_dpi is only used to find
the direction of the minutiae and depends on the ridge period of the images:
roughly 500 * period_px / 9.5 (500 for ordinary 500 ppi prints).  Extract all
.mnt files of a dataset with the same setting.

Output .mnt format (compatible with scripts/compare_minutiae_outputs.py):
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians> <class_index> <direction_confidence>
              class index: 0=ending  1=bifurcation  2=fragment
                           3=enclosure  4=crossbar  5=other

    x is the column and y the row of the image (y pointing down). The angle is
    the direction of the minutia, the direction vector is (cos(angle), sin(angle))
    in these axes.

Usage:
    python scripts/run_minutiae_extractor.py \\
        --input_dir MinutiaeNet-master/Dataset/CoarseNet_test/img_files \\
        --output_dir /tmp/me_mnt \\
        --models_dir minutiae_extractor-main/models
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

import cv2
import numpy as np


CLASS_NAMES = {
    0: 'ending',
    1: 'bifurcation',
    2: 'fragment',
    3: 'enclosure',
    4: 'crossbar',
    5: 'other',
}


def _write_mnt(filepath, df, width, height):
    """Write .mnt file — standard 3-column format plus class and direction confidence."""
    with open(filepath, 'w') as f:
        f.write('%d\n' % len(df))
        f.write('%d %d\n' % (width, height))
        for _, row in df.iterrows():
            cls = int(row['class']) if 'class' in df.columns and not np.isnan(row['class']) else -1
            f.write('%.4f %.4f %.6f %d %.4f\n' % (
                row['x'], row['y'], row['angle'], cls, row['direction_confidence']))


def process_image(extractor, image_path, output_path):
    """Extract the minutiae of one image and write them to output_path.

    Returns the minutiae DataFrame of the extractor.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Could not read image: %s' % image_path)

    df = extractor.extract_minutiae(image)['minutiae']

    # The minutiae coordinates refer to the full image
    height, width = image.shape[:2]
    _write_mnt(output_path, df, width, height)
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
                        help='Resolution of the images, used to find the direction of the '
                             'minutiae; roughly 500 * ridge_period_px / 9.5 (default: 500)')
    args = parser.parse_args()

    models_dir = args.models_dir
    coarse_net  = os.path.join(models_dir, 'CoarseNet.h5')
    fine_net    = os.path.join(models_dir, 'FineNet.h5')
    classify_net = os.path.join(models_dir, 'ClassifyNet_6_classes.h5')
    core_net    = os.path.join(models_dir, 'CoreNet.weights')

    for path, label in [(coarse_net, 'CoarseNet'), (fine_net, 'FineNet'),
                        (classify_net, 'ClassifyNet'), (core_net, 'CoreNet')]:
        if not os.path.isfile(path):
            print('ERROR: %s not found: %s' % (label, path))
            sys.exit(1)

    # Lazy import after path checks (TF startup is slow)
    from fingerflow.extractor import Extractor

    print('Loading Extractor...')
    extractor = Extractor(coarse_net, fine_net, classify_net, core_net,
                          image_dpi=args.image_dpi)

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
            df = process_image(extractor, image_path, output_path)
            cls_info = (
                ' (endings=%d bifurcs=%d)' % (
                    (df['class'] == 0).sum(), (df['class'] == 1).sum())
                if len(df) > 0 else '')
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


if __name__ == '__main__':
    main()
