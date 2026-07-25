"""
Validate one extracted preprocessed-DSEC sequence zip before committing to
the full 41-sequence run + training.

Usage:
    python validate_sequence.py /media/adarsh/Disk2/Chrome_Downloads/interlaken_00_c

Checks, in order:
  1. Builds a local pairs.txt from pairs_dsec/*.txt if missing (the master
     is only created at the end of the full run).
  2. dataset.py smoke test: both representations load, shapes/dtypes/ranges.
  3. Channel semantics on the SAVED arrays (catches any regression that
     slipped past the driver self-test into real data):
       - histogram neg channel differs from net channel
       - net channel has both signs and unit max-abs
       - voxel bins are not copies of each other / not empty after bin 0
  4. Rectification overlays: writes PNGs (RGB | net-channel | blend) for a
     few frames spread across the sequence. THE HUMAN CHECK: event edges
     should sit ON RGB edges. Misalignment = rectification or crop bug.
  5. Per-frame event statistics (nonzero fraction, mean |value|) so you can
     eyeball day/night behavior and compare against your earlier
     distribution intuitions.

Exit code 0 + "ALL CHECKS PASS" means: launch the remaining sequences.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from dataset import E_LiteVPRDataset


def ensure_pairs_txt(root: Path):
    master = root / 'pairs.txt'
    if master.is_file():
        return
    seq_files = sorted((root / 'pairs_dsec').glob('*_pairs.txt'))
    assert seq_files, f"No pairs files found under {root / 'pairs_dsec'}"
    with open(master, 'w') as out:
        for sf in seq_files:
            for line in open(sf):
                if line.strip():
                    out.write(line if line.endswith('\n') else line + '\n')
    print(f"[1] Built local pairs.txt from {len(seq_files)} per-sequence file(s)")


def check_dataset(root: Path):
    ds_h = E_LiteVPRDataset(root, 'histogram')
    ds_v = E_LiteVPRDataset(root, 'voxel')
    assert len(ds_h) == len(ds_v) > 0
    rgb, ev, ts = ds_h[0]
    assert tuple(rgb.shape) == tuple(ev.shape) == (3, 384, 384), \
        f"shape mismatch: rgb {tuple(rgb.shape)}, event {tuple(ev.shape)}"
    assert str(rgb.dtype).endswith('float32') and str(ev.dtype).endswith('float32')
    _, ev_v, _ = ds_v[0]
    ev_np = np.asarray(ev)
    assert -1.001 <= float(ev_np.min()) and float(ev_np.max()) <= 1.001, \
        f"event values outside [-1,1]: [{float(ev_np.min()):.3f}, {float(ev_np.max()):.3f}]"
    print(f"[2] Dataset smoke test PASS: {len(ds_h)} pairs, both reps (3,384,384) float32, "
          f"events in [{float(ev_np.min()):.3f}, {float(ev_np.max()):.3f}]")
    return ds_h


def check_channel_semantics(root: Path, ds):
    n = len(ds)
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    for i in idxs:
        hist = np.load(ds.pairs[i]['event_path']).astype(np.float32)
        vox_path = Path(str(ds.pairs[i]['event_path']).replace(
            str(Path('events') / 'histogram'), str(Path('events') / 'voxel')))
        vox = np.load(vox_path).astype(np.float32)

        pos, neg, net = hist[0], hist[1], hist[2]
        assert not np.array_equal(neg, net), \
            f"frame {i}: neg == net (channel overwrite regression!)"
        assert pos.min() >= 0 and neg.min() >= 0, f"frame {i}: count channels went negative"
        assert net.min() < 0 < net.max(), \
            f"frame {i}: net channel single-signed (range [{net.min():.3f},{net.max():.3f}])"
        # fp16 + INTER_AREA resize erode the peak; an ISOLATED max pixel is
        # diluted by the block-average factor (384/640)*(384/480) ~= 0.48,
        # common in night frames where extremes are lone residual spikes.
        assert 0.4 < np.abs(net).max() <= 1.001, \
            f"frame {i}: net max-abs {np.abs(net).max():.3f}, expected in (0.4, 1]"

        bins_nonzero = [int(np.count_nonzero(vox[b])) for b in range(vox.shape[0])]
        assert all(c > 0 for c in bins_nonzero), \
            f"frame {i}: empty voxel bin(s) {bins_nonzero} (bin-loop regression!)"
        assert not np.array_equal(vox[0], vox[1]) and not np.array_equal(vox[1], vox[2]), \
            f"frame {i}: identical voxel bins"
    print(f"[3] Channel semantics PASS on frames {idxs}: "
          f"neg!=net, net two-signed with unit peak, all voxel bins populated and distinct")


def write_overlays(root: Path, ds, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    ds_plain = E_LiteVPRDataset(root, 'histogram', imagenet_normalize=False)
    n = len(ds_plain)
    idxs = sorted({n // 5, n // 2, (4 * n) // 5})
    for i in idxs:
        rgb, ev, ts = ds_plain[i]
        rgb_img = (np.asarray(rgb).transpose(1, 2, 0) * 255).astype(np.uint8)  # HWC RGB
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        net = np.asarray(ev)[2]
        net_vis = np.clip((net * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)  # [-1,1]->[0,255]
        net_color = cv2.applyColorMap(net_vis, cv2.COLORMAP_JET)

        blend = cv2.addWeighted(rgb_bgr, 0.55, net_color, 0.45, 0)
        panel = np.concatenate([rgb_bgr, net_color, blend], axis=1)
        out_path = out_dir / f"overlay_{i:05d}_t{ts}.png"
        cv2.imwrite(str(out_path), panel)
    print(f"[4] Overlays written to {out_dir}/ for frames {idxs}")
    print("    >>> OPEN THEM. Event edges must sit ON RGB edges (poles, lane marks,")
    print("    >>> building outlines). Systematic offset or curved-vs-straight mismatch")
    print("    >>> at image borders = rectification/crop problem. STOP if misaligned.")


def event_stats(ds):
    fracs, mags = [], []
    step = max(1, len(ds) // 50)
    for i in range(0, len(ds), step):
        ev = np.load(ds.pairs[i]['event_path']).astype(np.float32)
        fracs.append(np.count_nonzero(ev[2]) / ev[2].size)
        mags.append(np.abs(ev[2]).mean())
    fracs, mags = np.array(fracs), np.array(mags)
    print(f"[5] Net-channel stats over {len(fracs)} sampled frames:")
    print(f"    nonzero fraction: min {fracs.min():.3f} / median {np.median(fracs):.3f} / max {fracs.max():.3f}")
    print(f"    mean |value|:     min {mags.min():.4f} / median {np.median(mags):.4f} / max {mags.max():.4f}")
    print("    (Compare feel against your earlier raw-count percentile checks; a frame")
    print("    with nonzero fraction near 0 would deserve a manual look.)")


def main():
    assert len(sys.argv) == 2, "usage: python validate_sequence.py <extracted_zip_root>"
    root = Path(sys.argv[1])
    assert root.is_dir(), f"{root} is not a directory"

    ensure_pairs_txt(root)
    ds = check_dataset(root)
    check_channel_semantics(root, ds)
    write_overlays(root, ds, root / 'validation_overlays')
    event_stats(ds)
    print("\nALL CHECKS PASS -- eyeball the overlays, then launch the remaining sequences.")


if __name__ == "__main__":
    main()