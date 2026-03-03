"""
Run FingerFlow's MinutiaeNet + ClassifyNet + CoreNet on a directory of
fingerprint images.

Uses the pip-installed fingerflow package (Python 3.11, fingerflow 3.0.1).

Output .mnt format:
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians> <class_index>
              class index: 0=ending  1=bifurcation  2=fragment
                           3=enclosure  4=crossbar  5=other

CoreNet results (if enabled) are saved as a <stem>.core.csv sidecar alongside
each .mnt file: x1,y1,x2,y2,score,w,h

IMPORTANT – CoreNet must be initialised BEFORE MinutiaeNet/ClassifyNet because
its constructor calls backend.clear_session(), which invalidates any already-
loaded models.

Usage:
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \\
    python scripts/run_fingerflow_minutiaenet.py \\
        --input_dir  MinutiaeNet-master/Dataset/CoarseNet_test/img_files \\
        --output_dir /tmp/fingerflow_mnt \\
        --coarse_net MinutiaeNet-master/Models/CoarseNet.h5 \\
        --fine_net   MinutiaeNet-master/Models/FineNet.h5 \\
        --models_dir minutiae_extractor-main/models
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import scipy.signal
import numpy.lib
import skimage.filters

# ---------------------------------------------------------------------------
# Compatibility shims for the pip fingerflow package (targets older APIs)
# ---------------------------------------------------------------------------

# scipy.signal.gaussian removed in scipy 1.12
if not hasattr(scipy.signal, 'gaussian'):
    scipy.signal.gaussian = scipy.signal.windows.gaussian

# numpy.lib.pad was an alias for numpy.pad
if not hasattr(numpy.lib, 'pad'):
    numpy.lib.pad = numpy.pad

# np.int / np.float / np.bool / np.complex removed in NumPy 1.24
for _name, _type in [('int', int), ('float', float), ('bool', bool), ('complex', complex)]:
    if not hasattr(np, _name):
        setattr(np, _name, _type)

# skimage.filters.gaussian dropped 'multichannel' kwarg in 0.20
_orig_skimage_gaussian = skimage.filters.gaussian
def _gaussian_compat(*args, **kwargs):
    if 'multichannel' in kwargs:
        mc = kwargs.pop('multichannel')
        if 'channel_axis' not in kwargs:
            kwargs['channel_axis'] = -1 if mc else None
    return _orig_skimage_gaussian(*args, **kwargs)
skimage.filters.gaussian = _gaussian_compat


# ---------------------------------------------------------------------------
# .mnt / .core.csv I/O
# ---------------------------------------------------------------------------

def _write_mnt(filepath, df, width, height):
    """Write .mnt file — 4-column format: x y angle class."""
    with open(filepath, 'w') as f:
        f.write('%d\n' % len(df))
        f.write('%d %d\n' % (width, height))
        for _, row in df.iterrows():
            cls = int(row['class']) if 'class' in df.columns else -1
            f.write('%.4f %.4f %.6f %d\n' % (row['x'], row['y'], row['angle'], cls))


def _write_core_csv(filepath, core_df):
    """Save CoreNet bounding-box results to a CSV sidecar file."""
    core_df.to_csv(filepath, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run FingerFlow (MinutiaeNet + ClassifyNet + CoreNet) on fingerprint images.'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing fingerprint images')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save .mnt (and .core.csv) output files')
    parser.add_argument('--coarse_net', required=True,
                        help='Path to CoarseNet.h5')
    parser.add_argument('--fine_net', required=True,
                        help='Path to FineNet.h5')
    parser.add_argument('--models_dir',
                        default=os.path.normpath(
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', 'minutiae_extractor-main', 'models')),
                        help='Directory with ClassifyNet_6_classes.h5 and CoreNet.weights '
                             '(default: minutiae_extractor-main/models)')
    parser.add_argument('--ext', default='.bmp',
                        help='Image file extension (default: .bmp)')
    parser.add_argument('--no_classify', action='store_true',
                        help='Skip ClassifyNet (faster, no class labels)')
    parser.add_argument('--no_core', action='store_true',
                        help='Skip CoreNet (no core-point detection)')
    args = parser.parse_args()

    classify_wt = os.path.join(args.models_dir, 'ClassifyNet_6_classes.h5')
    core_wt     = os.path.join(args.models_dir, 'CoreNet.weights')

    for path, label in [(args.coarse_net, 'CoarseNet'), (args.fine_net, 'FineNet')]:
        if not os.path.isfile(path):
            print('ERROR: %s not found: %s' % (label, path))
            sys.exit(1)

    if not args.no_classify and not os.path.isfile(classify_wt):
        print('WARNING: ClassifyNet weights not found (%s); skipping classification.' % classify_wt)
        args.no_classify = True

    if not args.no_core and not os.path.isfile(core_wt):
        print('WARNING: CoreNet weights not found (%s); skipping core detection.' % core_wt)
        args.no_core = True

    # ------------------------------------------------------------------
    # Load models — CoreNet MUST be first (calls backend.clear_session())
    # ------------------------------------------------------------------
    from fingerflow.extractor.core_net import CoreNet
    from fingerflow.extractor.minutiae_net import MinutiaeNet
    from fingerflow.extractor import utils as fe_utils

    core_net = None
    if not args.no_core:
        print('Loading CoreNet...')
        try:
            core_net = CoreNet(core_wt)
            print('CoreNet loaded.')
        except Exception as e:
            print('WARNING: CoreNet failed to load: %s' % e)
            core_net = None

    print('Loading MinutiaeNet...')
    mnt_net = MinutiaeNet(args.coarse_net, args.fine_net)

    classify_net = None
    if not args.no_classify:
        from fingerflow.extractor.classify_net import ClassifyNet
        print('Loading ClassifyNet...')
        classify_net = ClassifyNet(classify_wt)

    print('All models loaded.')

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
        mnt_path  = os.path.join(args.output_dir, basename + '.mnt')
        core_path = os.path.join(args.output_dir, basename + '.core.csv')
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError('Could not read image: %s' % image_path)

            preprocessed = fe_utils.preprocess_image_data(image_bgr)
            trimmed  = preprocessed['image']           # grayscale, 8-multiple
            original = preprocessed['original_image']  # grayscale, full size

            # CoreNet
            if core_net is not None:
                try:
                    core_df = core_net.detect_fingerprint_core(image_bgr)
                    _write_core_csv(core_path, core_df)
                except Exception as e:
                    print('  CoreNet inference failed for %s: %s' % (basename, e))

            # MinutiaeNet
            mnt_array = mnt_net.extract_minutiae_points(trimmed, original)

            # ClassifyNet
            if classify_net is not None and mnt_array.shape[0] > 0:
                df = classify_net.classify_minutiae_points(original, mnt_array)
            else:
                import pandas as pd
                cols = ['x', 'y', 'angle', 'score']
                df = pd.DataFrame(
                    mnt_array[:, :4] if mnt_array.shape[0] > 0 else [],
                    columns=cols
                )

            height, width = trimmed.shape[:2]
            _write_mnt(mnt_path, df, width, height)

            cls_info = ''
            if classify_net is not None and len(df) > 0:
                cls_info = ' (endings=%d bifurcs=%d)' % (
                    (df['class'] == 0).sum(), (df['class'] == 1).sum())
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
