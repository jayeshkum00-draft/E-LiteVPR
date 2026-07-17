"""
Task 1 -- Flicker quantification for E-LiteVPR (night-robustness evidence).

Context: Gehrig et al. (DSEC, RA-L 2021, Fig. 9) document that flashing
street lamps at night "periodically trigger events for a large number of
pixels", spiking event rates to 30-40 MEPS. This script quantifies that
effect in OUR inter-frame windows and measures what our 99.5th-percentile
hot-pixel mask actually removes, separating:

  (a) point-source flicker: few pixels with extreme counts
      -> the mask CAN catch these (they sit above the percentile), and
  (b) area-illumination flicker: moderate counts smeared over many pixels
      -> structurally below any percentile threshold, passes to the model.

Per sampled inter-frame window it reports:
  - total events, active-pixel count
  - share of events from the top-100 / top-1000 pixels (PRE-mask)
  - hot-mask size and share of events it removes
  - top-100 / top-1000 share POST-mask (the residue the model sees)
Plus a time-aggregated log-count heatmap with top pixels marked (pre and
post mask), for visual comparison against streetlight positions in RGB.

Run it on one DAY and one NIGHT sequence (raw events.h5 + rectify_map.h5 +
image timestamps -- same DSEC layout the preprocessor uses) and compare:

    python analyze_flicker.py --dsec_path /path/to/DSEC \
        --sequences zurich_city_03_a zurich_city_09_a \
        --out flicker_report

Needs only the Events/<seq>/ files and RGB/<seq>/<seq>_image_timestamps.txt
per sequence -- NOT the RGB images -- so the Drive download per sequence is
just the h5 files (a few GB).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SENSOR_HW = (480, 640)
HOT_PIXEL_PCT = 99.5      # must match cfg.datasets.event_hot_pixels_threshold
TOP_KS = (100, 1000)


# --------------------------------------------------------------------------
# Pure analysis core (no I/O -- unit-testable)
# --------------------------------------------------------------------------

def window_concentration_stats(x_r, y_r, sensor_hw=SENSOR_HW,
                               hot_pct=HOT_PIXEL_PCT, top_ks=TOP_KS):
    """
    Given rectified event coordinates of ONE window, compute concentration
    metrics before and after the same hot-pixel mask the preprocessor uses.
    Returns a flat dict of scalars.
    """
    h, w = sensor_hw
    counts = np.zeros((h, w), dtype=np.int64)
    np.add.at(counts, (y_r, x_r), 1)
    flat = counts.ravel()
    total = int(flat.sum())
    stats = {'total_events': total,
             'active_pixels': int(np.count_nonzero(flat))}
    if total == 0:
        for k in top_ks:
            stats[f'top{k}_share_pre'] = 0.0
            stats[f'top{k}_share_post'] = 0.0
        stats.update(hot_pixels=0, hot_removed_share=0.0)
        return stats

    sorted_counts = np.sort(flat)[::-1]
    for k in top_ks:
        stats[f'top{k}_share_pre'] = float(sorted_counts[:k].sum() / total)

    # Same rule as preprocess_dsec.compute_hot_pixel_mask: percentile of
    # NONZERO counts, mask strictly-above pixels.
    nonzero = flat[flat > 0]
    thr = np.percentile(nonzero, hot_pct)
    hot = flat > thr
    removed = int(flat[hot].sum())
    stats['hot_pixels'] = int(hot.sum())
    stats['hot_removed_share'] = float(removed / total)

    flat_post = flat.copy()
    flat_post[hot] = 0
    total_post = int(flat_post.sum())
    sorted_post = np.sort(flat_post)[::-1]
    for k in top_ks:
        stats[f'top{k}_share_post'] = (
            float(sorted_post[:k].sum() / total_post) if total_post > 0 else 0.0)
    return stats


def aggregate_stats(per_window, q=(50, 95, 100)):
    keys = per_window[0].keys()
    return {k: {f'p{p}': float(np.percentile([s[k] for s in per_window], p)) for p in q}
            for k in keys}


def heatmap_png(count_image, top_k_mark=1000):
    """Log-scale count heatmap (BGR) with the top-K pixels circled-ish
    (dilated mask overlay) for visual localisation of dominant firers."""
    log_img = np.log1p(count_image.astype(np.float64))
    vis = (255 * log_img / max(log_img.max(), 1e-9)).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    flat = count_image.ravel()
    if np.count_nonzero(flat) > 0:
        k = min(top_k_mark, int(np.count_nonzero(flat)))
        thr = np.sort(flat)[::-1][k - 1]
        mark = (count_image >= max(thr, 1)).astype(np.uint8)
        mark = cv2.dilate(mark, np.ones((5, 5), np.uint8))
        color[mark > 0] = (0, 255, 0)
    return color


# --------------------------------------------------------------------------
# I/O + driver
# --------------------------------------------------------------------------

def rectify(events, rectify_map, sensor_hw=SENSOR_HW):
    h, w = sensor_hw
    xy = rectify_map[events['y'], events['x']]
    x_r = np.rint(xy[:, 0]).astype(np.int64)
    y_r = np.rint(xy[:, 1]).astype(np.int64)
    keep = (x_r >= 0) & (x_r < w) & (y_r >= 0) & (y_r < h)
    return x_r[keep], y_r[keep]


def analyze_sequence(dsec_path: Path, seq: str, out_dir: Path,
                     max_windows=60, window_stride=None):
    import h5py, hdf5plugin  # noqa: F401  (blosc codec registration)
    from ../utils/dsec_eventslicer import EventSlicer

    ts_file = dsec_path / "RGB" / seq / f"{seq}_image_timestamps.txt"
    ev_file = dsec_path / "Events" / seq / "events.h5"
    rm_file = dsec_path / "Events" / seq / "rectify_map.h5"
    timestamps = np.loadtxt(ts_file, dtype=np.int64)
    with h5py.File(rm_file, 'r') as f:
        rectify_map = f['rectify_map'][:]

    n_windows = len(timestamps) - 1
    stride = window_stride or max(1, n_windows // max_windows)
    per_window = []
    agg_counts = np.zeros(SENSOR_HW, dtype=np.int64)

    with h5py.File(ev_file, 'r') as h5f:
        slicer = EventSlicer(h5f)
        for i in range(1, len(timestamps), stride):
            ev = slicer.get_events(timestamps[i - 1], timestamps[i])
            if ev is None or ev['t'].size == 0:
                continue
            x_r, y_r = rectify(ev, rectify_map)
            if x_r.size == 0:
                continue
            per_window.append(window_concentration_stats(x_r, y_r))
            np.add.at(agg_counts, (y_r, x_r), 1)

    assert per_window, f"{seq}: no usable windows sampled"
    agg = aggregate_stats(per_window)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{seq}_heatmap_pre.png"), heatmap_png(agg_counts))
    # post-mask aggregate view: zero the pixels the per-aggregate mask hits
    nonzero = agg_counts[agg_counts > 0]
    post = agg_counts.copy()
    post[agg_counts > np.percentile(nonzero, HOT_PIXEL_PCT)] = 0
    cv2.imwrite(str(out_dir / f"{seq}_heatmap_post.png"), heatmap_png(post))

    return agg, len(per_window)


def print_report(results):
    cols = ['total_events', 'active_pixels',
            'top100_share_pre', 'top1000_share_pre',
            'hot_pixels', 'hot_removed_share',
            'top100_share_post', 'top1000_share_post']
    header = f"{'metric (median/window)':<26}" + "".join(f"{s:>22}" for s in results)
    print("\n" + header)
    print("-" * len(header))
    for c in cols:
        row = f"{c:<26}"
        for seq in results:
            v = results[seq][c]
            row += f"{v:>22.4f}" if isinstance(v, float) and v < 1000 else f"{v:>22,.0f}"
        print(row)
    print("\nReading guide:")
    print("  top-K share PRE  high on night, low on day  -> flicker concentration confirmed")
    print("  hot_removed_share                            -> what the mask deletes")
    print("  top-K share POST still elevated on night     -> area-flicker residue reaches the model")
    print("  Heatmap PNGs: green marks = top-1000 pixels; compare against streetlight")
    print("  positions in the RGB frames (cf. Gehrig et al. DSEC Fig. 9).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dsec_path', type=Path, required=True)
    ap.add_argument('--sequences', nargs='+', required=True,
                    help='e.g. one day and one night sequence, same route if possible')
    ap.add_argument('--out', type=Path, default=Path('flicker_report'))
    ap.add_argument('--max_windows', type=int, default=60)
    args = ap.parse_args()

    results = {}
    for seq in args.sequences:
        agg, n = analyze_sequence(args.dsec_path, seq, args.out, args.max_windows)
        print(f"{seq}: {n} windows sampled")
        results[seq] = agg
    print_report(results)


if __name__ == "__main__":
    main()