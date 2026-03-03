# -*- coding: utf-8 -*-
"""
Run inference with the original MinutiaeNet (2018) on a directory of fingerprint images.

Outputs .mnt files in FingerFlow-compatible format so they can be compared with
the updated FingerFlow extractor using scripts/compare_minutiae_outputs.py.

IMPORTANT: This script must be run under the 'minutiaenet' conda environment:
    conda activate minutiaenet
    python scripts/run_original_minutiaenet.py --input_dir ... --output_dir ...

Output .mnt format (FingerFlow-compatible, 3 columns):
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians>

References:
    CoarseNet_model.py:inference() for the original pipeline
    CoarseNet_model.py:fuse_minu_orientation()
    CoarseNet_utils.py:label2mnt()
    MinutiaeNet_utils.py:py_cpu_nms(), nms(), fuse_nms(), FastEnhanceTexture(), get_maps_STFT()
"""
from __future__ import print_function, division

import os
import sys
import argparse
import glob

# ---------------------------------------------------------------------------
# Add MinutiaeNet source directories to sys.path so the original modules
# can be imported without modification.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_MINUTIAENET_ROOT = os.path.join(_REPO_ROOT, 'MinutiaeNet-master')
sys.path.insert(0, os.path.join(_MINUTIAENET_ROOT, 'CoarseNet'))
sys.path.insert(0, os.path.join(_MINUTIAENET_ROOT, 'FineNet'))

# ---------------------------------------------------------------------------
# Keras / TF imports (must be after sys.path is set up)
# ---------------------------------------------------------------------------
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU — TF 1.7 needs CUDA 9.0 for GPU

import numpy as np
import cv2
from scipy import misc, ndimage

from keras import backend as K
from keras.optimizers import Adam

from CoarseNet_model import CoarseNetmodel, fuse_minu_orientation
from CoarseNet_utils import label2mnt
from MinutiaeNet_utils import py_cpu_nms, nms, fuse_nms, FastEnhanceTexture, get_maps_STFT
from FineNet_model import FineNetmodel


def _write_mnt_fingerflow(filepath, mnt, width, height):
    """Write .mnt file in FingerFlow-compatible format (3 columns: x y angle_rad).

    Args:
        filepath: Output path for .mnt file
        mnt: Nx4 numpy array [x, y, orientation_rad, score] — only x/y/ori written
        width: Original image width in pixels
        height: Original image height in pixels
    """
    with open(filepath, 'w') as f:
        f.write('%d\n' % mnt.shape[0])
        f.write('%d %d\n' % (width, height))
        for i in range(mnt.shape[0]):
            f.write('%.4f %.4f %.6f\n' % (mnt[i, 0], mnt[i, 1], mnt[i, 2]))


def _process_single_image(image_path, main_net_model, model_FineNet,
                           max_num_minu=20, min_num_minu=6):
    """Run the full MinutiaeNet inference pipeline on one image.

    Pipeline mirrors CoarseNet_model.py:inference() exactly:
        1. Load grayscale, trim to multiple of 8
        2. STFT orientation/frequency map
        3. CoarseNet predict (deploy mode → 10 outputs)
        4. Segmentation morphological post-processing
        5. label2mnt → raw candidatess
        6. Dual NMS: py_cpu_nms + nms → fuse_nms
        7. Adaptive threshold (lowers until min_num_minu candidates remain)
        8. Optional FineNet soft-decision score fusion
        9. Final score threshold
        10. fuse_minu_orientation (mode=3: average CoarseNet + STFT)

    Args:
        image_path: Path to fingerprint image (grayscale)
        main_net_model: Loaded CoarseNet Keras model (deploy mode)
        model_FineNet: Loaded FineNet Keras model, or None to skip refinement
        max_num_minu: Maximum number of minutiae to keep after NMS
        min_num_minu: Minimum target for adaptive threshold loop

    Returns:
        (mnt_nms, img_size)
        - mnt_nms: Nx4 array [x, y, orientation_rad, score], may be (0,4) if none found
        - img_size: (height, width) tuple in pixels (trimmed to multiple of 8)
    """
    # 1. Load image
    image = misc.imread(image_path, mode='L')

    # Trim to multiple of 8 (required by CoarseNet's pooling structure)
    img_size = np.array(image.shape, dtype=np.int32) // 8 * 8
    image = image[:img_size[0], :img_size[1]]
    original_image = image.copy()

    # 2. Orientation/frequency map via STFT
    texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
    dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)

    # 3. CoarseNet prediction
    image_in = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
    (enh_img, enh_img_imag, enhance_img,
     ori_out_1, ori_out_2,
     seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out) = main_net_model.predict(image_in)

    # 4. Segmentation post-processing (from model output)
    round_seg = np.round(np.squeeze(seg_out))
    seg_out = 1 - round_seg
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    seg_out = cv2.dilate(seg_out, kernel)

    # 5. Convert label maps to minutiae candidates
    mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)),
                    mnt_w_out, mnt_h_out, mnt_o_out, thresh=0)

    # 6. Dual NMS
    mnt_nms_1 = py_cpu_nms(mnt, 0.5)
    mnt_nms_2 = nms(mnt)
    # Sort mnt_nms_1 by score descending
    mnt_nms_1.view('f8,f8,f8,f8').sort(order=['f3'], axis=0)
    mnt_nms_1 = mnt_nms_1[::-1]

    # 7. Adaptive threshold: lower until we have enough candidates
    early_thres = 0.5
    mnt_nms_1_copy = mnt_nms_1.copy()
    mnt_nms_2_copy = mnt_nms_2.copy()

    while early_thres > 0:
        mnt_nms_1 = mnt_nms_1_copy[mnt_nms_1_copy[:, 3] > early_thres, :]
        mnt_nms_2 = mnt_nms_2_copy[mnt_nms_2_copy[:, 3] > early_thres, :]

        if mnt_nms_1.shape[0] > max_num_minu or mnt_nms_2.shape[0] > max_num_minu:
            mnt_nms_1 = mnt_nms_1[:max_num_minu, :]
            mnt_nms_2 = mnt_nms_2[:max_num_minu, :]

        if mnt_nms_1.shape[0] > min_num_minu and mnt_nms_2.shape[0] > min_num_minu:
            break

        early_thres -= 0.05

    mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)
    final_score_threshold = early_thres - 0.05

    # 8. FineNet soft-decision score fusion (optional)
    if model_FineNet is not None:
        patch_radio = 22
        mnt_refined = []
        for idx in range(mnt_nms.shape[0]):
            try:
                x_begin = int(mnt_nms[idx, 1]) - patch_radio
                y_begin = int(mnt_nms[idx, 0]) - patch_radio
                patch = original_image[x_begin:x_begin + 2*patch_radio,
                                       y_begin:y_begin + 2*patch_radio]
                patch = cv2.resize(patch, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
                rgb = np.empty((224, 224, 3), dtype=np.uint8)
                rgb[:, :, 0] = patch
                rgb[:, :, 1] = patch
                rgb[:, :, 2] = patch
                patch_in = np.expand_dims(rgb, axis=0)

                [prob] = model_FineNet.predict(patch_in)
                is_minutiae_prob = prob[0]
                tmp = mnt_nms[idx, :].copy()
                tmp[3] = (4 * tmp[3] + is_minutiae_prob) / 5
                mnt_refined.append(tmp)
            except Exception:
                mnt_refined.append(mnt_nms[idx, :])
        mnt_nms = np.array(mnt_refined)

    # 9. Final score threshold
    if mnt_nms.shape[0] > 0:
        mnt_nms = mnt_nms[mnt_nms[:, 3] > final_score_threshold, :]

    # 10. Fuse minutiae orientation with STFT orientation field (mode=3: average)
    fuse_minu_orientation(dir_map, mnt_nms, mode=3)

    return mnt_nms, img_size


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run original MinutiaeNet (2018) inference on fingerprint images. '
            'Must be run under the minutiaenet conda environment. '
            'Outputs .mnt files in FingerFlow-compatible format for comparison.'
        )
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing fingerprint images')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save .mnt output files')
    parser.add_argument('--coarse_net', required=True,
                        help='Path to CoarseNet.h5 model weights')
    parser.add_argument('--fine_net', default=None,
                        help='Path to FineNet.h5 model weights (optional)')
    parser.add_argument('--no_finenet', action='store_true',
                        help='Skip FineNet refinement step')
    parser.add_argument('--ext', default='.bmp',
                        help='Image file extension to search for (default: .bmp)')
    parser.add_argument('--max_minu', type=int, default=20,
                        help='Maximum minutiae to keep after NMS (default: 20)')
    parser.add_argument('--min_minu', type=int, default=6,
                        help='Minimum target for adaptive threshold (default: 6)')
    args = parser.parse_args()

    # TF session setup (matches CoarseNet_run.py)
    tf_config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
    sess = K.tf.Session(config=tf_config)
    K.set_session(sess)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load CoarseNet
    print('Loading CoarseNet from: %s' % args.coarse_net)
    main_net_model = CoarseNetmodel((None, None, 1), args.coarse_net, mode='deploy')

    # Load FineNet (optional)
    model_FineNet = None
    use_finenet = (not args.no_finenet) and (args.fine_net is not None)
    if use_finenet:
        print('Loading FineNet from: %s' % args.fine_net)
        model_FineNet = FineNetmodel(num_classes=2, pretrained_path=args.fine_net,
                                     input_shape=(224, 224, 3))
        model_FineNet.compile(loss='categorical_crossentropy',
                              optimizer=Adam(lr=0), metrics=['accuracy'])
    else:
        print('FineNet: disabled (using CoarseNet scores only)')

    # Find images
    ext = args.ext if args.ext.startswith('.') else '.' + args.ext
    pattern = os.path.join(args.input_dir, '*' + ext)
    image_files = sorted(glob.glob(pattern))
    if not image_files:
        # Also try uppercase extension
        pattern_upper = os.path.join(args.input_dir, '*' + ext.upper())
        image_files = sorted(glob.glob(pattern_upper))

    if not image_files:
        print('No images found in %s with extension %s' % (args.input_dir, ext))
        sys.exit(1)

    print('Found %d images. Processing...' % len(image_files))

    success_count = 0
    errors = []

    for i, image_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output_dir, basename + '.mnt')

        try:
            mnt_nms, img_size = _process_single_image(
                image_path, main_net_model, model_FineNet,
                max_num_minu=args.max_minu, min_num_minu=args.min_minu,
            )
            height, width = img_size[0], img_size[1]
            _write_mnt_fingerflow(output_path, mnt_nms, width, height)
            print('[%d/%d] %s — %d minutiae written to %s' % (
                i + 1, len(image_files), basename, mnt_nms.shape[0], output_path))
            success_count += 1
        except Exception as e:
            print('[%d/%d] ERROR %s: %s' % (i + 1, len(image_files), basename, str(e)))
            errors.append((basename, str(e)))

    print('\n========================================')
    print('Done. %d/%d images succeeded.' % (success_count, len(image_files)))
    if errors:
        print('%d errors:' % len(errors))
        for name, err in errors:
            print('  %s: %s' % (name, err))
    print('Output directory: %s' % args.output_dir)


if __name__ == '__main__':
    main()
