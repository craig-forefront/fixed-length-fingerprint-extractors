"""
4-panel visualization: raw image | Original MinutiaeNet | FingerFlow | minutiae_extractor.

Every marking panel shows:
  - Minutiae typed by ClassifyNet (circle=ending, triangle=bifurcation, diamond=others)
  - Colour by type (green/cyan/yellow/orange/magenta/grey)
  - Orientation arrow
  - Core region bounding box + crosshair (red) from CoreNet

Usage:
    python scripts/visualize_three_way.py \
        --input_dir  MinutiaeNet-master/Dataset/CoarseNet_test/img_files \
        --orig_dir   /tmp/original_mnt \
        --ff_dir     /tmp/fingerflow_mnt \
        --me_dir     /tmp/me_mnt \
        --output_dir .  \
        --models_dir minutiae_extractor-main/models  \
        [--panel_height 420]
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make minutiae_extractor-main/src the first fingerflow on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ME_SRC = os.path.normpath(os.path.join(_HERE, '..', 'minutiae_extractor-main', 'src'))
sys.path.insert(0, _ME_SRC)

# ---------------------------------------------------------------------------
# Type colour / name / shape mapping
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    0:  (  0, 230,   0),   # ending      – green
    1:  (  0, 220, 220),   # bifurcation – cyan
    2:  (  0, 180, 255),   # fragment    – sky-blue
    3:  (  0, 120, 255),   # enclosure   – orange-ish blue
    4:  (200,   0, 255),   # crossbar    – magenta
    5:  (130, 130, 130),   # other       – grey
    -1: (220, 220, 220),   # unknown     – light grey
}
TYPE_NAMES = {
    0: 'ending', 1: 'bifurcation', 2: 'fragment',
    3: 'enclosure', 4: 'crossbar', 5: 'other', -1: '?',
}
CORE_COLOR  = (0, 0, 230)   # red (BGR)


# ---------------------------------------------------------------------------
# .mnt I/O
# ---------------------------------------------------------------------------

def read_mnt(path):
    """Return list of (x, y, angle_rad, cls) – cls=-1 when column absent."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    result = []
    for line in lines[2: 2 + n]:
        parts = line.split()
        if len(parts) < 3:
            continue
        x, y, angle = float(parts[0]), float(parts[1]), float(parts[2])
        cls = int(parts[3]) if len(parts) >= 4 else -1
        result.append((x, y, angle, cls))
    return result


# ---------------------------------------------------------------------------
# ClassifyNet helper
# ---------------------------------------------------------------------------

def classify_all(minutiae, gray_image, classify_net, fe_utils):
    """Fill in cls=-1 entries by running ClassifyNet on image patches."""
    result = []
    for (x, y, angle, cls) in minutiae:
        if cls == -1:
            patch = fe_utils.get_minutiae_patch(x, y, gray_image)
            if patch.size > 0:
                try:
                    cls = int(classify_net.classify_minutiae_patch(patch))
                except Exception:
                    cls = -1
        result.append((x, y, angle, cls))
    return result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _triangle_pts(cx, cy, r, angle):
    angles = [angle + math.pi / 2 + i * 2 * math.pi / 3 for i in range(3)]
    pts = [[int(cx + r * math.cos(a)), int(cy - r * math.sin(a))] for a in angles]
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _diamond_pts(cx, cy, r, angle):
    angles = [angle + i * math.pi / 2 for i in range(4)]
    pts = [[int(cx + r * math.cos(a)), int(cy - r * math.sin(a))] for a in angles]
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def draw_minutia(panel, x, y, angle, cls, r=8):
    color = TYPE_COLORS.get(int(cls), TYPE_COLORS[-1])
    xi, yi = int(round(x)), int(round(y))
    if cls == 0:
        cv2.circle(panel, (xi, yi), r, color, 2)
    elif cls == 1:
        cv2.polylines(panel, [_triangle_pts(xi, yi, r, angle)], True, color, 2)
    else:
        cv2.polylines(panel, [_diamond_pts(xi, yi, r, angle)], True, color, 2)
    # Orientation arrow (forward direction)
    ar = int(r * 1.8)
    ax = xi + int(ar * math.cos(angle))
    ay = yi - int(ar * math.sin(angle))
    cv2.arrowedLine(panel, (xi, yi), (ax, ay), color, 1, tipLength=0.4)


def draw_core(panel, core_df, scale):
    if len(core_df) == 0:
        return
    row = core_df.iloc[core_df['score'].argmax()]
    x1 = int(row['x1'] * scale)
    y1 = int(row['y1'] * scale)
    x2 = int(row['x2'] * scale)
    y2 = int(row['y2'] * scale)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(panel, (x1, y1), (x2, y2), CORE_COLOR, 2)
    cr = 10
    cv2.line(panel, (cx - cr, cy), (cx + cr, cy), CORE_COLOR, 2)
    cv2.line(panel, (cx, cy - cr), (cx, cy + cr), CORE_COLOR, 2)


def draw_legend(panel):
    """Overlay a small type legend in the bottom-left corner."""
    ph = panel.shape[0]
    items = list(TYPE_NAMES.items())[:6]   # skip -1
    block_h = 16 * (len(items) + 1) + 8
    y0 = ph - block_h - 4
    # Translucent backing
    overlay = panel.copy()
    cv2.rectangle(overlay, (2, y0), (110, ph - 2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

    y = y0 + 14
    for cls, name in items:
        color = TYPE_COLORS[cls]
        if cls == 0:
            cv2.circle(panel, (12, y - 4), 5, color, -1)
        elif cls == 1:
            pts = _triangle_pts(12, y - 4, 5, 0)
            cv2.fillPoly(panel, [pts], color)
        else:
            pts = _diamond_pts(12, y - 4, 5, 0)
            cv2.fillPoly(panel, [pts], color)
        cv2.putText(panel, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, color, 1, cv2.LINE_AA)
        y += 16
    # Core indicator
    cv2.rectangle(panel, (6, y - 6), (18, y + 4), CORE_COLOR, 2)
    cv2.putText(panel, 'core', (22, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, CORE_COLOR, 1, cv2.LINE_AA)


def add_title(panel, text):
    bar = np.full((28, panel.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def make_marking_panel(gray_full, minutiae, core_df, title, target_h, with_legend=False):
    h, w = gray_full.shape[:2]
    scale = target_h / h
    panel = cv2.resize(cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR),
                       (int(w * scale), target_h))

    # Core
    draw_core(panel, core_df, scale)

    # Minutiae
    for (x, y, angle, cls) in minutiae:
        draw_minutia(panel, x * scale, y * scale, angle, cls)

    # Count overlay
    ph = panel.shape[0]
    cnt_text = 'n=%d' % len(minutiae)
    cv2.putText(panel, cnt_text, (5, ph - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if with_legend:
        draw_legend(panel)

    return add_title(panel, title)


def make_raw_panel(gray_full, target_h):
    h, w = gray_full.shape[:2]
    scale = target_h / h
    panel = cv2.resize(cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR),
                       (int(w * scale), target_h))
    return add_title(panel, 'Original image')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate 4-panel three-way minutiae comparison images.')
    parser.add_argument('--input_dir',  required=True,
                        help='Directory of fingerprint images')
    parser.add_argument('--orig_dir',   required=True,
                        help='Original MinutiaeNet .mnt directory')
    parser.add_argument('--ff_dir',     required=True,
                        help='FingerFlow .mnt directory')
    parser.add_argument('--me_dir',     required=True,
                        help='minutiae_extractor .mnt directory')
    parser.add_argument('--output_dir', default='.',
                        help='Where to save output images (default: .)')
    parser.add_argument('--models_dir',
                        default=os.path.normpath(
                            os.path.join(_HERE, '..', 'minutiae_extractor-main', 'models')),
                        help='Directory with HuggingFace-downloaded model weights')
    parser.add_argument('--panel_height', type=int, default=420,
                        help='Panel height in pixels (default: 420)')
    args = parser.parse_args()

    models_dir  = args.models_dir
    coarse_net  = os.path.join(models_dir, 'CoarseNet.h5')
    fine_net    = os.path.join(models_dir, 'FineNet.h5')
    classify_wt = os.path.join(models_dir, 'ClassifyNet_6_classes.h5')
    core_wt     = os.path.join(models_dir, 'CoreNet.weights')

    # ------------------------------------------------------------------
    # Load models.  CoreNet MUST be initialised first — its constructor
    # calls backend.clear_session(), which would invalidate any already-
    # loaded models.
    # ------------------------------------------------------------------
    core_net_obj = None
    if os.path.isfile(core_wt):
        print('Loading CoreNet …')
        try:
            from fingerflow.extractor.core_net import CoreNet
            core_net_obj = CoreNet(core_wt)
            print('CoreNet loaded.')
        except Exception as e:
            print('WARNING: CoreNet failed to load: %s' % e)
            core_net_obj = None
    else:
        print('WARNING: CoreNet.weights not found, skipping core detection.')

    print('Loading MinutiaeNet …')
    from fingerflow.extractor.minutiae_net import MinutiaeNet
    from fingerflow.extractor import utils as fe_utils

    mnt_net = MinutiaeNet(coarse_net, fine_net)

    # Bug-fix: restore correct Gabor weights from H5
    try:
        import h5py
        with h5py.File(coarse_net, 'r') as hf:
            for lname in ('enh_img_real_1', 'enh_img_imag_1'):
                layer = mnt_net._MinutiaeNet__coarse_net.get_layer(lname)
                grp = hf[lname][lname]
                layer.set_weights([grp['kernel:0'][()], grp['bias:0'][()]])
        print('Gabor filter weights restored from H5.')
    except Exception as e:
        print('WARNING: could not restore Gabor weights: %s' % e)

    print('Loading ClassifyNet …')
    from fingerflow.extractor.classify_net import ClassifyNet
    classify_net = ClassifyNet(classify_wt)
    print('All models loaded.')

    # ------------------------------------------------------------------
    # Discover images that have .mnt files in all three directories
    # ------------------------------------------------------------------
    exts = ('.bmp', '.BMP', '.png', '.PNG', '.jpg', '.JPG', '.tif', '.TIF')
    image_files = [
        f for f in sorted(os.listdir(args.input_dir))
        if os.path.splitext(f)[1] in exts
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in image_files:
        stem = os.path.splitext(fname)[0]
        orig_mnt = os.path.join(args.orig_dir, stem + '.mnt')
        ff_mnt   = os.path.join(args.ff_dir,   stem + '.mnt')
        me_mnt   = os.path.join(args.me_dir,   stem + '.mnt')

        if not (os.path.isfile(orig_mnt) and os.path.isfile(ff_mnt)
                and os.path.isfile(me_mnt)):
            print('Skipping %s (missing .mnt files)' % fname)
            continue

        img_path = os.path.join(args.input_dir, fname)
        image_bgr  = cv2.imread(img_path)
        if image_bgr is None:
            print('Could not read %s, skipping.' % img_path)
            continue
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # CoreNet
        core_df_raw = None
        if core_net_obj is not None:
            try:
                core_df_raw = core_net_obj.detect_fingerprint_core(image_bgr)
            except Exception as e:
                print('CoreNet inference failed for %s: %s' % (stem, e))
                core_df_raw = None
        import pandas as pd
        core_df = core_df_raw if core_df_raw is not None else pd.DataFrame(
            columns=['x1', 'y1', 'x2', 'y2', 'score', 'w', 'h'])

        # Load minutiae
        mnt_orig = read_mnt(orig_mnt)
        mnt_ff   = read_mnt(ff_mnt)
        mnt_me   = read_mnt(me_mnt)

        # ClassifyNet on original and fingerflow (they have cls=-1)
        print('Classifying %s minutiae …' % stem)
        mnt_orig = classify_all(mnt_orig, image_gray, classify_net, fe_utils)
        mnt_ff   = classify_all(mnt_ff,   image_gray, classify_net, fe_utils)
        # mnt_me already has class from ClassifyNet baked in

        TARGET_H = args.panel_height
        panel_raw  = make_raw_panel(image_gray, TARGET_H)
        panel_orig = make_marking_panel(image_gray, mnt_orig, core_df,
                                        'Original MinutiaeNet  (n=%d)' % len(mnt_orig),
                                        TARGET_H, with_legend=True)
        panel_ff   = make_marking_panel(image_gray, mnt_ff,   core_df,
                                        'FingerFlow pip  (n=%d)' % len(mnt_ff),
                                        TARGET_H)
        panel_me   = make_marking_panel(image_gray, mnt_me,   core_df,
                                        'minutiae_extractor  (n=%d)' % len(mnt_me),
                                        TARGET_H)

        # Pad all to the same height (title bars may differ)
        max_h = max(p.shape[0] for p in [panel_raw, panel_orig, panel_ff, panel_me])
        def pad_h(p, target):
            if p.shape[0] < target:
                p = np.vstack([p, np.zeros((target - p.shape[0], p.shape[1], 3), np.uint8)])
            return p
        panels = [pad_h(p, max_h) for p in [panel_raw, panel_orig, panel_ff, panel_me]]

        # Add thin separator columns
        sep = np.full((max_h, 3, 3), 60, dtype=np.uint8)
        combined = np.hstack([panels[0], sep, panels[1], sep, panels[2], sep, panels[3]])

        out_path = os.path.join(args.output_dir, 'threeway_%s.png' % stem)
        cv2.imwrite(out_path, combined)
        print('Saved: %s' % out_path)

    print('Done.')


if __name__ == '__main__':
    main()
