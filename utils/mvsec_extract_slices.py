from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

SENSOR_HW = (260, 346)
BLOCK = 10_000_000          # rows per sequential read (~320 MB)

def load_positions(gt_path):
    """(ts, xy) from */pose (loop-closure corrected; NOT */odometry, which
    drifts and corrupts positives at loop closures)."""
    with h5py.File(gt_path, "r") as g:
        cands = []
        g.visititems(lambda n, o: cands.append(n)
                     if isinstance(o, h5py.Dataset) and o.ndim == 3
                     and o.shape[1:] == (4, 4) else None)
        pose_named = [n for n in cands if n.endswith("pose")]
        if not pose_named:
            raise KeyError(f"no */pose dataset in {gt_path}; found {cands}")
        name = pose_named[0]
        pose = g[name][:]
        ts = g[name + "_ts"][:]
    xy = pose[:, :2, 3].astype(np.float64)
    print(f"  poses: {name} n={len(ts)}  t {ts[0]:.1f}..{ts[-1]:.1f}")
    return ts, xy

def load_rect_maps(calib_dir, prefix):
    x_map = np.loadtxt(Path(calib_dir) / f"{prefix}_left_x_map.txt")
    y_map = np.loadtxt(Path(calib_dir) / f"{prefix}_left_y_map.txt")
    assert x_map.shape == SENSOR_HW, x_map.shape
    return x_map.astype(np.float32), y_map.astype(np.float32)

def extract_sequence(data_path, gt_path, out_path, rect, slice_ms, hz,
                     min_speed):
    pose_ts, pose_xy = load_positions(gt_path)
    half = slice_ms / 2e3

    with h5py.File(data_path, "r") as f:
        ev = f["davis/left/events"]
        n = ev.shape[0]
        t0 = float(ev[0, 2])
        t1 = float(ev[-1, 2])

        lo = max(t0, pose_ts[0]) + half
        hi = min(t1, pose_ts[-1]) - half
        centers = np.arange(lo, hi, 1.0 / hz)
        cx = np.interp(centers, pose_ts, pose_xy[:, 0])
        cy = np.interp(centers, pose_ts, pose_xy[:, 1])
        speed = np.hypot(np.gradient(cx, 1.0 / hz),
                         np.gradient(cy, 1.0 / hz))
        keep = speed >= min_speed
        centers, cx, cy = centers[keep], cx[keep], cy[keep]
        print(f"  {keep.sum()}/{len(keep)} samples moving "
              f"(>= {min_speed} m/s) @ {hz} Hz")
        t_lo, t_hi = centers - half, centers + half

        xs, ys, ts_us, ps = [], [], [], []
        offsets = np.zeros(len(centers) + 1, np.int64)
        n_empty = 0

        def emit(j, buf):
            bt = buf[:, 2]
            a = np.searchsorted(bt, t_lo[j], "left")
            b = np.searchsorted(bt, t_hi[j], "left")
            blk = buf[a:b]
            if len(blk) == 0:
                nonlocal n_empty
                n_empty += 1
                offsets[j + 1] = offsets[j]
                return
            x = blk[:, 0].astype(np.int32)
            y = blk[:, 1].astype(np.int32)
            p = (blk[:, 3] > 0).astype(np.uint8)
            t = blk[:, 2]
            if rect is not None:
                xr = np.rint(rect[0][y, x]).astype(np.int32)
                yr = np.rint(rect[1][y, x]).astype(np.int32)
                ok = ((xr >= 0) & (xr < SENSOR_HW[1])
                      & (yr >= 0) & (yr < SENSOR_HW[0]))
                x, y, p, t = xr[ok], yr[ok], p[ok], t[ok]
            xs.append(x.astype(np.uint16))
            ys.append(y.astype(np.uint16))
            ts_us.append(np.round((t - t_lo[j]) * 1e6).astype(np.uint32))
            ps.append(p)
            offsets[j + 1] = offsets[j] + len(x)

        # single sequential pass; buffer holds current block + carry-over,
        # a slice is emitted once its whole window is inside the buffer
        buf = np.empty((0, 4), np.float64)
        j = 0
        last_t = -np.inf
        for i0 in tqdm(range(0, n, BLOCK), desc="events", unit="blk"):
            blk = ev[i0:min(i0 + BLOCK, n)]
            if blk[0, 2] < last_t or np.any(np.diff(blk[:, 2]) < 0):
                raise RuntimeError("events not time-sorted")
            last_t = blk[-1, 2]
            buf = np.concatenate([buf, blk]) if len(buf) else blk
            while j < len(centers) and len(buf) and t_hi[j] <= buf[-1, 2]:
                emit(j, buf)
                j += 1
                if j < len(centers):
                    keep_from = np.searchsorted(buf[:, 2], t_lo[j], "left")
                    buf = buf[keep_from:]
        while j < len(centers):
            emit(j, buf)
            j += 1
        if n_empty:
            print(f"  warning: {n_empty} empty slices")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as o:
        for k, v in [("events_x", xs), ("events_y", ys),
                     ("events_t_us", ts_us), ("events_p", ps)]:
            o.create_dataset(k, data=np.concatenate(v), compression="lzf")
        o.create_dataset("slice_offsets", data=offsets)
        o.create_dataset("slice_center_ts", data=centers)
        o.create_dataset("slice_xy_local", data=np.stack([cx, cy], 1))
        o.attrs.update(dict(
            sequence=Path(data_path).stem, slice_ms=slice_ms,
            hz=hz, min_speed=min_speed,
            rectified=rect is not None, sensor_hw=SENSOR_HW,
            frame="local-pose"))
    print(f"  -> {out_path}  ({len(centers)} slices, {offsets[-1]/1e6:.1f}M "
          f"events, {out_path.stat().st_size/1e6:.0f} MB)")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    m = cfg.datasets
    root = Path(m.root)
    names = list(m.sequences.keys()) if m.sequence == "all" else [m.sequence]
    for seq in names:
        cond = m.sequences[seq]
        print(f"===== {seq} =====")
        rect = (load_rect_maps(root / cond / f"{cond}_calib", cond)
                if m.rectify else None)
        extract_sequence(
            data_path=root / cond / f"{seq}_data.hdf5",
            gt_path=root / cond / f"{seq}_gt.hdf5",
            out_path=Path(m.slices_dir) / f"{seq}.h5",
            rect=rect, slice_ms=float(m.slice_ms), hz=float(m.hz),
            min_speed=float(m.min_speed))

if __name__ == "__main__":
    main()