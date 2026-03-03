"""
Compare minutiae extraction outputs from two directories of .mnt files.

Designed to compare:
  - Original MinutiaeNet (run via scripts/run_original_minutiaenet.py)
  - Updated FingerFlow extractor (run via scripts/extract_minutiae.py)

Both directories must contain .mnt files in FingerFlow-compatible format:
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians>

Matching algorithm per image pair:
    Greedy nearest-neighbour with:
        - Distance threshold: 16 pixels
        - Orientation threshold: π/6 radians (30°)
    Metrics: Precision, Recall, F1, mean location error, mean orientation error

Usage:
    python scripts/compare_minutiae_outputs.py \\
        --original_dir /tmp/original_mnt \\
        --updated_dir /tmp/fingerflow_mnt \\
        [--dist_thresh 16] [--ori_thresh 0.5236] [--output_csv comparison_results.csv]
"""

import argparse
import csv
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# .mnt file I/O
# ---------------------------------------------------------------------------

def read_mnt_file(filepath):
    """Read a .mnt file in FingerFlow-compatible format.

    Args:
        filepath: Path to .mnt file

    Returns:
        (locations, orientations, (width, height))
        - locations: np.ndarray shape (N, 2) with (x, y) float32
        - orientations: np.ndarray shape (N,) float32, angles in radians
        - (width, height): original image dimensions
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), (0, 0)

    num_minutiae = int(lines[0].strip())
    width, height = map(int, lines[1].strip().split())

    if num_minutiae == 0 or len(lines) < 3:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), (width, height)

    locs, oris = [], []
    for line in lines[2:2 + num_minutiae]:
        parts = line.strip().split()
        if len(parts) >= 3:
            locs.append((float(parts[0]), float(parts[1])))
            oris.append(float(parts[2]))

    return (
        np.array(locs, dtype=np.float32),
        np.array(oris, dtype=np.float32),
        (width, height),
    )


# ---------------------------------------------------------------------------
# Matching / metrics
# ---------------------------------------------------------------------------

def _angle_delta(a, b):
    """Smallest angular difference between two angles in [0, 2π]."""
    delta = abs(a - b) % (2 * np.pi)
    return min(delta, 2 * np.pi - delta)


def match_minutiae(locs_a, oris_a, locs_b, oris_b, dist_thresh=16.0, ori_thresh=np.pi / 6):
    """Greedy nearest-neighbour matching between two minutiae sets.

    A minutia in set A is matched to the nearest unmatched minutia in set B
    that satisfies both the distance and orientation thresholds.

    Args:
        locs_a: (Na, 2) array of (x, y) positions for set A
        oris_a: (Na,) array of orientations for set A
        locs_b: (Nb, 2) array of (x, y) positions for set B
        oris_b: (Nb,) array of orientations for set B
        dist_thresh: Maximum spatial distance for a match (pixels)
        ori_thresh: Maximum orientation difference for a match (radians)

    Returns:
        (matched_a, matched_b, dist_errors, ori_errors)
        - matched_a: indices of matched minutiae in A
        - matched_b: indices of matched minutiae in B
        - dist_errors: spatial distances for each match (pixels)
        - ori_errors: orientation differences for each match (radians)
    """
    if len(locs_a) == 0 or len(locs_b) == 0:
        return [], [], [], []

    # Pairwise Euclidean distances
    diff = locs_a[:, np.newaxis, :] - locs_b[np.newaxis, :, :]  # (Na, Nb, 2)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))              # (Na, Nb)

    # Pairwise orientation differences
    ori_matrix = np.abs(oris_a[:, np.newaxis] - oris_b[np.newaxis, :])  # (Na, Nb)
    ori_matrix = np.minimum(ori_matrix, 2 * np.pi - ori_matrix)

    # Candidates: within thresholds
    valid = (dist_matrix <= dist_thresh) & (ori_matrix <= ori_thresh)

    # Greedy matching: sort all valid pairs by distance, pick greedily
    rows, cols = np.where(valid)
    if len(rows) == 0:
        return [], [], [], []

    distances = dist_matrix[rows, cols]
    sort_idx = np.argsort(distances)
    rows, cols, distances = rows[sort_idx], cols[sort_idx], distances[sort_idx]
    ori_diffs = ori_matrix[rows, cols]

    used_a, used_b = set(), set()
    matched_a, matched_b = [], []
    dist_errors, ori_errors = [], []

    for i in range(len(rows)):
        ia, ib = rows[i], cols[i]
        if ia not in used_a and ib not in used_b:
            matched_a.append(ia)
            matched_b.append(ib)
            dist_errors.append(distances[i])
            ori_errors.append(ori_diffs[i])
            used_a.add(ia)
            used_b.add(ib)

    return matched_a, matched_b, dist_errors, ori_errors


def compute_prf(n_pred, n_true, n_matched):
    """Precision, Recall, F1 from counts."""
    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare_directories(original_dir, updated_dir, dist_thresh, ori_thresh):
    """Compare all matching .mnt file pairs in two directories.

    Args:
        original_dir: Directory with original MinutiaeNet .mnt files
        updated_dir: Directory with updated FingerFlow .mnt files
        dist_thresh: Spatial match threshold in pixels
        ori_thresh: Orientation match threshold in radians

    Returns:
        List of per-image result dicts, one per matched filename pair
    """
    orig_files = {
        os.path.splitext(f)[0]: os.path.join(original_dir, f)
        for f in os.listdir(original_dir)
        if f.endswith('.mnt')
    }
    updated_files = {
        os.path.splitext(f)[0]: os.path.join(updated_dir, f)
        for f in os.listdir(updated_dir)
        if f.endswith('.mnt')
    }

    common = sorted(set(orig_files.keys()) & set(updated_files.keys()))
    only_original = sorted(set(orig_files.keys()) - set(updated_files.keys()))
    only_updated = sorted(set(updated_files.keys()) - set(orig_files.keys()))

    if only_original:
        print('WARNING: %d files only in original_dir (no updated counterpart): %s ...' % (
            len(only_original), ', '.join(only_original[:3])))
    if only_updated:
        print('WARNING: %d files only in updated_dir (no original counterpart): %s ...' % (
            len(only_updated), ', '.join(only_updated[:3])))
    if not common:
        print('ERROR: No matching filenames found between the two directories.')
        sys.exit(1)

    print('Comparing %d matched .mnt file pairs...' % len(common))

    results = []
    for name in common:
        locs_o, oris_o, (w_o, h_o) = read_mnt_file(orig_files[name])
        locs_u, oris_u, (w_u, h_u) = read_mnt_file(updated_files[name])

        matched_a, matched_b, dist_errs, ori_errs = match_minutiae(
            locs_o, oris_o, locs_u, oris_u,
            dist_thresh=dist_thresh, ori_thresh=ori_thresh,
        )

        n_orig = len(locs_o)
        n_upd = len(locs_u)
        n_matched = len(matched_a)

        # Treat original as ground truth, updated as predictions
        prec, rec, f1 = compute_prf(n_upd, n_orig, n_matched)

        mean_dist = float(np.mean(dist_errs)) if dist_errs else 0.0
        mean_ori = float(np.mean(ori_errs)) if ori_errs else 0.0

        results.append({
            'name': name,
            'count_original': n_orig,
            'count_updated': n_upd,
            'count_matched': n_matched,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mean_dist_error_px': mean_dist,
            'mean_ori_error_rad': mean_ori,
        })

    return results


def print_summary(results, dist_thresh, ori_thresh):
    """Print a formatted summary table and aggregate statistics."""
    print('\n' + '=' * 72)
    print('MINUTIAE COMPARISON RESULTS')
    print('Match thresholds: distance <= %g px, orientation <= %.4f rad (%.1f deg)' % (
        dist_thresh, ori_thresh, np.degrees(ori_thresh)))
    print('=' * 72)

    header = '%-30s %6s %6s %7s %7s %7s %7s %8s %9s' % (
        'Image', 'N_orig', 'N_upd', 'Matched', 'Prec', 'Rec', 'F1', 'DistErr', 'OriErr(d)')
    print(header)
    print('-' * 72)

    for r in results:
        print('%-30s %6d %6d %7d %7.3f %7.3f %7.3f %8.2f %9.2f' % (
            r['name'][:30],
            r['count_original'],
            r['count_updated'],
            r['count_matched'],
            r['precision'],
            r['recall'],
            r['f1'],
            r['mean_dist_error_px'],
            np.degrees(r['mean_ori_error_rad']),
        ))

    print('=' * 72)

    # Aggregate
    n = len(results)
    avg_n_orig = np.mean([r['count_original'] for r in results])
    avg_n_upd = np.mean([r['count_updated'] for r in results])
    avg_matched = np.mean([r['count_matched'] for r in results])
    avg_prec = np.mean([r['precision'] for r in results])
    avg_rec = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_dist = np.mean([r['mean_dist_error_px'] for r in results])
    avg_ori = np.mean([r['mean_ori_error_rad'] for r in results])

    print('\nAGGREGATE OVER %d IMAGES:' % n)
    print('  Mean count (original): %.1f' % avg_n_orig)
    print('  Mean count (updated):  %.1f' % avg_n_upd)
    print('  Mean matched:          %.1f' % avg_matched)
    print('  Mean Precision:        %.4f' % avg_prec)
    print('  Mean Recall:           %.4f' % avg_rec)
    print('  Mean F1:               %.4f' % avg_f1)
    print('  Mean dist error:       %.2f px' % avg_dist)
    print('  Mean ori error:        %.2f deg' % np.degrees(avg_ori))

    if avg_f1 >= 0.8:
        verdict = 'GOOD AGREEMENT (F1 >= 0.80)'
    elif avg_f1 >= 0.6:
        verdict = 'MODERATE AGREEMENT (F1 in [0.60, 0.80))'
    else:
        verdict = 'LOW AGREEMENT (F1 < 0.60) — outputs differ substantially'
    print('\n  Verdict: %s' % verdict)
    print('=' * 72)


def save_csv(results, output_path):
    """Save per-image results to a CSV file."""
    fieldnames = [
        'name', 'count_original', 'count_updated', 'count_matched',
        'precision', 'recall', 'f1', 'mean_dist_error_px', 'mean_ori_error_rad',
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print('\nResults saved to: %s' % output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compare .mnt minutiae files from original MinutiaeNet vs. updated FingerFlow '
            'extractor. Both directories must contain .mnt files in FingerFlow format.'
        )
    )
    parser.add_argument('--original_dir', required=True,
                        help='Directory of .mnt files from run_original_minutiaenet.py')
    parser.add_argument('--updated_dir', required=True,
                        help='Directory of .mnt files from extract_minutiae.py (FingerFlow)')
    parser.add_argument('--dist_thresh', type=float, default=16.0,
                        help='Spatial match threshold in pixels (default: 16)')
    parser.add_argument('--ori_thresh', type=float, default=np.pi / 6,
                        help='Orientation match threshold in radians (default: pi/6 ≈ 0.524)')
    parser.add_argument('--output_csv', default='comparison_results.csv',
                        help='Path for per-image CSV output (default: comparison_results.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.original_dir):
        print('ERROR: original_dir does not exist: %s' % args.original_dir)
        sys.exit(1)
    if not os.path.isdir(args.updated_dir):
        print('ERROR: updated_dir does not exist: %s' % args.updated_dir)
        sys.exit(1)

    results = compare_directories(
        args.original_dir, args.updated_dir,
        dist_thresh=args.dist_thresh, ori_thresh=args.ori_thresh,
    )

    print_summary(results, args.dist_thresh, args.ori_thresh)
    save_csv(results, args.output_csv)


if __name__ == '__main__':
    main()
