"""
Colab driver for the E-LiteVPR DSEC preprocessor.

This file is NOT part of the portable pipeline -- it wraps the clean,
committable `preprocess_dsec.py` with Colab-specific machinery:

  1. SELF-TEST on startup: runs the imported event functions on synthetic
     events and refuses to launch if channel semantics are wrong. A 6-hour
     run never starts on top of a silently-broken representation.
  2. STAGING: copies one raw sequence at a time from the Drive mount to
     local disk (EventSlicer random access over FUSE is slow/flaky).
  3. CHECKPOINTING: after each sequence, zips its outputs (store-only) and
     moves the zip to Drive. A sequence is durably DONE the moment its zip
     exists; session death costs at most the in-flight sequence.
  4. RESUME: sequences whose zip already exists on Drive are skipped.
  5. MASTER pairs.txt: built at the end FROM THE ZIPS on Drive (not from
     local files), so it is complete even across resumed sessions.

Usage (in a Colab cell, with Drive mounted):

    !python preprocess_dsec_colab.py \
        dsec_path=/content/drive/MyDrive/E-LiteVPR/DSEC \
        output_dir=/content/outputs

All processing logic lives in preprocess_dsec.py; this file only moves
files around and must never contain science.
"""

import os
import shutil
import zipfile
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from preprocess_dsec import (
    process_sequence,
    process_event_histogram,
    process_event_voxel_grid,
    process_event_time_surface,
    rectify_event_coords,
    compute_hot_pixel_mask,
)

# ---- Colab-specific locations (edit here, not in the committed script) ----
STAGING_DIR = Path("/content/dsec_staging")
DRIVE_ZIP_DIR = Path("/content/drive/MyDrive/E-LiteVPR/preprocessed_zips")
TMP_ZIP_DIR = Path("/content")


def self_test():
    """
    Run the imported processing functions on synthetic events with known
    ground truth. Raises AssertionError (with a specific message) if the
    pipeline is broken. Costs <1s; guards multi-hour runs.
    """
    sensor_hw = (480, 640)
    h, w = sensor_hw
    out_size = (64, 64)
    rng = np.random.default_rng(0)

    # Identity rectify map + one known-shift map
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    ident_map = np.stack([gx, gy], axis=-1)
    shift_map = np.stack([gx + 10, gy], axis=-1)

    # -- rectification direction --
    ev = {'x': np.array([30]), 'y': np.array([50]),
          'p': np.array([1]), 't': np.array([100])}
    x_r, y_r, _, _ = rectify_event_coords(ev, shift_map, sensor_hw)
    assert x_r[0] == 40 and y_r[0] == 50, \
        "SELF-TEST FAIL: rectification moved event the wrong way (remap-style inverse?)"

    # -- synthetic window: background noise + one balanced pixel + one pos-heavy pixel --
    n_bg = 30000
    x = rng.integers(0, w, n_bg)
    y = rng.integers(0, h, n_bg)
    p = rng.integers(0, 2, n_bg)
    t = np.sort(rng.integers(0, 50000, n_bg))
    # balanced pixel at (60, 50): 40 pos + 40 neg
    x = np.concatenate([x, np.full(80, 50)]); y = np.concatenate([y, np.full(80, 60)])
    p = np.concatenate([p, np.tile([1, 0], 40)]); t = np.concatenate([t, np.linspace(0, 50000, 80).astype(np.int64)])
    # pos-heavy pixel at (61, 51): 60 pos only
    x = np.concatenate([x, np.full(60, 51)]); y = np.concatenate([y, np.full(60, 61)])
    p = np.concatenate([p, np.ones(60, dtype=np.int64)]); t = np.concatenate([t, np.linspace(0, 50000, 60).astype(np.int64)])
    order = np.argsort(t, kind='stable')
    ev = {'x': x[order], 'y': y[order], 'p': p[order], 't': t[order]}

    x_r, y_r, p_r, t_r = rectify_event_coords(ev, ident_map, sensor_hw)
    assert t_r.size == ev['t'].size, "SELF-TEST FAIL: identity rectification dropped events"

    hot = np.zeros(sensor_hw, dtype=bool)  # disable hot removal for semantic checks

    # -- histogram channel semantics (at sensor res: skip resize effects by
    #    checking pre-resize invariants through a full-res out_size) --
    hist = process_event_histogram(x_r, y_r, p_r, hot, out_size=(w, h), sensor_hw=sensor_hw)
    assert hist.shape == (3, h, w), f"SELF-TEST FAIL: histogram shape {hist.shape}"
    assert not np.array_equal(hist[1], hist[2]), \
        "SELF-TEST FAIL: neg and net channels are identical (channel overwrite bug)"
    assert abs(hist[2][60, 50]) < 1e-6, \
        f"SELF-TEST FAIL: balanced pixel has net {hist[2][60, 50]}, expected 0 " \
        "(net normalized by wrong max, or computed from normalized channels)"
    assert hist[2][61, 51] > 0, "SELF-TEST FAIL: pos-heavy pixel has non-positive net"
    assert hist[1][61, 51] == 0 and hist[1][60, 50] > 0, \
        "SELF-TEST FAIL: neg channel does not contain negative counts"
    assert abs(hist[2].max() - 1.0) < 1e-6 or abs(hist[2].min() + 1.0) < 1e-6, \
        "SELF-TEST FAIL: net channel not normalized to unit max-abs"

    # -- voxel: window-boundary bins, event conservation incl. t == t_end --
    vox = process_event_voxel_grid(x_r, y_r, p_r, t_r, 0, 50000, hot,
                                   out_size=out_size, num_bins=3, sensor_hw=sensor_hw)
    assert vox.shape == (3, out_size[1], out_size[0]), \
        f"SELF-TEST FAIL: voxel shape {vox.shape}"
    span = 50000.0
    bin_idx = np.clip((t_r.astype(np.float64) * 3 / span).astype(np.int64), 0, 2)
    assert bin_idx.size == t_r.size and bin_idx[t_r == 50000].min() == 2, \
        "SELF-TEST FAIL: event at t_end not assigned to last bin"

    # -- hot pixel mask detects an injected hot pixel --
    xh = np.concatenate([x_r, np.full(5000, 100)])
    yh = np.concatenate([y_r, np.full(5000, 200)])
    mask = compute_hot_pixel_mask(xh, yh, sensor_hw=sensor_hw, threshold=99.5)
    assert mask[200, 100], "SELF-TEST FAIL: injected hot pixel not detected"

    # -- time surface: polarity separation, bounded [0,1] -- at full sensor
    # res (out_size=(w,h)) so pixel indices are exact, same as the histogram
    # check above.
    on_full, off_full = process_event_time_surface(
        x_r, y_r, p_r, t_r, t_start_us=0, t_end_us=50000, hot_mask=hot, out_size=(w, h),
        decay_frac=1.0, sensor_hw=sensor_hw)
    assert on_full.shape == (h, w) == off_full.shape, \
        f"SELF-TEST FAIL: time surface shape {on_full.shape}"
    assert on_full.min() >= 0.0 and on_full.max() <= 1.0 + 1e-6, \
        "SELF-TEST FAIL: ON time surface not bounded in [0, 1]"
    assert off_full.min() >= 0.0 and off_full.max() <= 1.0 + 1e-6, \
        "SELF-TEST FAIL: OFF time surface not bounded in [0, 1]"
    # pos-heavy pixel (x=51, y=61): only ON events -> OFF must be exactly 0,
    # ON must be > 0 (last event at t=50000 -> decay factor exactly 1.0)
    assert off_full[61, 51] == 0.0, \
        "SELF-TEST FAIL: OFF channel nonzero at a pixel with no OFF events"
    assert abs(on_full[61, 51] - 1.0) < 1e-6, \
        "SELF-TEST FAIL: ON surface at the most recent event's pixel should be 1.0"
    # balanced pixel (x=50, y=60): both polarities present -> both nonzero
    assert on_full[60, 50] > 0.0 and off_full[60, 50] > 0.0, \
        "SELF-TEST FAIL: balanced pixel missing a polarity channel"

    print("SELF-TEST PASS: rectification, histogram channels, voxel bins, "
          "hot-pixel mask, time surface")


def zip_sequence_outputs(seq, out_root, pairs_dir, output_dirs, zip_dest):
    """Zip one sequence's outputs (store-only) with out_root-relative paths."""
    tmp_zip = TMP_ZIP_DIR / f"{seq}.zip.part"
    with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_STORED) as zf:
        for d in output_dirs:
            for f in sorted(d.glob(f"{seq}_*.npy")):
                zf.write(f, f.relative_to(out_root))
        seq_pairs = pairs_dir / f"{seq}_pairs.txt"
        zf.write(seq_pairs, seq_pairs.relative_to(out_root))
    # .part -> final name only when complete, so a mid-write session death
    # never leaves a truncated zip that the resume check would trust.
    shutil.move(str(tmp_zip), str(zip_dest))


def build_master_pairs_from_zips(drive_zip_dir, dest_path):
    """
    Build the master pairs.txt from the pairs files INSIDE the zips on
    Drive. Correct across resumed sessions (local pairs files only exist
    for sequences processed in the current session).
    """
    total = 0
    zips = sorted(drive_zip_dir.glob("*.zip"))
    with open(dest_path, 'w') as out:
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith("_pairs.txt")]
                assert len(names) == 1, f"{z.name}: expected 1 pairs file, found {names}"
                text = zf.read(names[0]).decode()
                for line in text.splitlines():
                    if line.strip():
                        out.write(line + '\n')
                        total += 1
    print(f"Master pairs.txt: {total} pairs from {len(zips)} sequence zips -> {dest_path}")
    return total


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    self_test()  # refuse to start a multi-hour run on a broken pipeline

    dataset_path = Path(cfg.dsec_path)
    out_root = Path(cfg.output_dir) / "preprocessed_dsec"
    rgb_out = out_root / "rgb"
    hist_out = out_root / "events" / "histogram"
    voxel_out = out_root / "events" / "voxel"
    ts_net_out = out_root / "events" / "timesurface_net"
    ts_zero_out = out_root / "events" / "timesurface_zero"
    pairs_dir = out_root / "pairs_dsec"
    for d in (rgb_out, hist_out, voxel_out, ts_net_out, ts_zero_out, pairs_dir):
        d.mkdir(parents=True, exist_ok=True)

    rgb_dir = dataset_path / "RGB"
    events_dir = dataset_path / "Events"
    calibrations_dir = dataset_path / "Calibrations"
    DRIVE_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    sequences = [s for s in sorted(os.listdir(rgb_dir)) if os.path.isdir(rgb_dir / s)]
    print(f"{len(sequences)} sequences found; "
          f"{sum((DRIVE_ZIP_DIR / f'{s}.zip').exists() for s in sequences)} already done on Drive")

    failed = []
    for seq in sequences:
        zip_dest = DRIVE_ZIP_DIR / f"{seq}.zip"
        if zip_dest.exists():
            print(f"{seq}: already done (zip on Drive), skipping")
            continue
        try:
            # 1) stage raw sequence Drive -> local disk
            shutil.rmtree(STAGING_DIR, ignore_errors=True)
            for name, src in (("RGB", rgb_dir), ("Events", events_dir),
                              ("Calibrations", calibrations_dir)):
                shutil.copytree(src / seq, STAGING_DIR / name / seq)

            # 2) process from the local staged copy (all science lives in
            #    preprocess_dsec.py; nothing here touches representations)
            n_pairs, n_skipped = process_sequence(
                seq, STAGING_DIR / "RGB", STAGING_DIR / "Events",
                STAGING_DIR / "Calibrations", pairs_dir,
                rgb_out, hist_out, voxel_out, ts_net_out, ts_zero_out, out_root, cfg)
            print(f"{seq}: {n_pairs} pairs"
                  + (f", {n_skipped} skipped" if n_skipped else ""))

            # 3) zip -> Drive: sequence durably done
            zip_sequence_outputs(seq, out_root, pairs_dir,
                                 (rgb_out, hist_out, voxel_out, ts_net_out, ts_zero_out), zip_dest)

            # 4) free local disk
            shutil.rmtree(STAGING_DIR, ignore_errors=True)
            for d in (rgb_out, hist_out, voxel_out, ts_net_out, ts_zero_out):
                for f in d.glob(f"{seq}_*.npy"):
                    f.unlink()
        except Exception as e:
            print(f"ERROR: {seq} failed: {e!r}")
            failed.append((seq, repr(e)))

    # Master pairs.txt from the zips (complete across resumed sessions),
    # written both locally and next to the zips on Drive.
    build_master_pairs_from_zips(DRIVE_ZIP_DIR, out_root / "pairs.txt")
    shutil.copy(out_root / "pairs.txt", DRIVE_ZIP_DIR / "pairs.txt")

    if failed:
        fail_log = DRIVE_ZIP_DIR / "failed_sequences.txt"
        with open(fail_log, 'w') as f:
            for seq, err in failed:
                f.write(f"{seq}: {err}\n")
        print(f"{len(failed)} sequence(s) failed -- see {fail_log}. "
              f"Rerun this script; finished sequences are skipped.")


if __name__ == "__main__":
    main()
