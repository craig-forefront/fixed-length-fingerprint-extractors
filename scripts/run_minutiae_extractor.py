"""
Run minutiae_extractor-main (modernised FingerFlow) on a directory of
fingerprint images.

Uses MinutiaeNet (CoarseNet + FineNet) followed by ClassifyNet to assign one
of six minutiae types to every extracted point.  CoreNet (core-point
detection) is intentionally skipped here — it is not needed for comparing
minutiae output.

Output .mnt format (compatible with scripts/compare_minutiae_outputs.py):
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians> [<class_index>]
              class index: 0=ending  1=bifurcation  2=fragment
                           3=enclosure  4=crossbar  5=other

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
    """Write .mnt file — standard 3-column format plus optional class column."""
    with open(filepath, 'w') as f:
        f.write('%d\n' % len(df))
        f.write('%d %d\n' % (width, height))
        for _, row in df.iterrows():
            cls = int(row['class']) if 'class' in df.columns and not np.isnan(row['class']) else -1
            f.write('%.4f %.4f %.6f %d\n' % (row['x'], row['y'], row['angle'], cls))


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
    parser.add_argument('--no_classify', action='store_true',
                        help='Skip ClassifyNet (faster, no class labels)')
    args = parser.parse_args()

    models_dir = args.models_dir
    coarse_net  = os.path.join(models_dir, 'CoarseNet.h5')
    fine_net    = os.path.join(models_dir, 'FineNet.h5')
    classify_net = os.path.join(models_dir, 'ClassifyNet_6_classes.h5')

    for path, label in [(coarse_net, 'CoarseNet'), (fine_net, 'FineNet')]:
        if not os.path.isfile(path):
            print('ERROR: %s not found: %s' % (label, path))
            sys.exit(1)

    if not args.no_classify and not os.path.isfile(classify_net):
        print('WARNING: ClassifyNet not found (%s); running without classification.' % classify_net)
        args.no_classify = True

    # Lazy import after path checks (TF startup is slow)
    from fingerflow.extractor.minutiae_net import MinutiaeNet
    from fingerflow.extractor import utils as fe_utils

    print('Loading MinutiaeNet...')
    net = MinutiaeNet(coarse_net, fine_net)

    # -----------------------------------------------------------------------
    # Bug workaround: minutiae_extractor-main overwrites correctly-loaded
    # Gabor filter weights with freshly-computed ones, but the saved H5
    # weights diverged during training (mean ~8 vs computed mean ~0.001).
    # Reload the true Gabor weights from the H5 file.
    # -----------------------------------------------------------------------
    try:
        import h5py
        with h5py.File(coarse_net, 'r') as hf:
            for layer_name in ('enh_img_real_1', 'enh_img_imag_1'):
                layer = net._MinutiaeNet__coarse_net.get_layer(layer_name)
                grp = hf[layer_name][layer_name]
                saved_kernel = grp['kernel:0'][()]
                saved_bias   = grp['bias:0'][()]
                layer.set_weights([saved_kernel, saved_bias])
        print('Gabor filter weights restored from H5 (bug workaround applied).')
    except Exception as e:
        print('WARNING: could not restore Gabor weights: %s' % e)

    classify = None
    if not args.no_classify:
        from fingerflow.extractor.classify_net import ClassifyNet
        from fingerflow.extractor.ClassifyNet.utils import format_classified_data
        print('Loading ClassifyNet...')
        classify = ClassifyNet(classify_net)

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
            # Load as BGR — preprocess_image_data converts to grayscale internally
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError('Could not read image: %s' % image_path)

            preprocessed = fe_utils.preprocess_image_data(image_bgr)
            trimmed   = preprocessed['image']           # grayscale, 8-multiple
            original  = preprocessed['original_image']  # grayscale original

            mnt_array = net.extract_minutiae_points(trimmed, original)

            if classify is not None and mnt_array.shape[0] > 0:
                df = classify.classify_minutiae_points(original, mnt_array)
            else:
                # Build minimal DataFrame without class labels
                import pandas as pd
                df = pd.DataFrame(
                    mnt_array[:, :4] if mnt_array.shape[0] > 0 else [],
                    columns=['x', 'y', 'angle', 'score']
                )

            height, width = trimmed.shape[:2]
            _write_mnt(output_path, df, width, height)
            cls_info = '' if classify is None else (
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
