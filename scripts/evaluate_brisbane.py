import csv
from datetime import datetime, timezone
import math
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from brisbane_representation import build_representation
from model import EventViTStudent
from sequence_matching import apply_sequence_matching

DAY_DAY = {"sunrise", "morning", "daytime", "sunset2"}

def parse_nmea_rmc(path):
    """Parse $__RMC sentences -> (t_epoch[s], lat[deg], lon[deg]) arrays."""
    times, lats, lons = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "RMC" not in line:
                continue
            fields = line.strip().split(",")
            try:
                if len(fields) < 10 or fields[2] != "A":
                    continue
                hhmmss, lat_raw, ns, lon_raw, ew, date = (
                    fields[1], fields[3], fields[4], fields[5], fields[6], fields[9])
                lat = int(lat_raw[:2]) + float(lat_raw[2:]) / 60.0
                lon = int(lon_raw[:3]) + float(lon_raw[3:]) / 60.0
                if ns == "S":
                    lat = -lat
                if ew == "W":
                    lon = -lon
                base = datetime.strptime(
                    date + hhmmss.split(".")[0], "%d%m%y%H%M%S"
                ).replace(tzinfo=timezone.utc).timestamp()
                frac = float("0." + hhmmss.split(".")[1]) if "." in hhmmss else 0.0
                times.append(base + frac)
                lats.append(lat)
                lons.append(lon)
            except (ValueError, IndexError):
                continue
    if not times:
        raise RuntimeError(f"No valid RMC fixes parsed from {path}")
    order = np.argsort(times)
    return (np.asarray(times)[order],
            np.asarray(lats)[order],
            np.asarray(lons)[order])


def latlon_to_xy(lat, lon, lat0, lon0):
    """Local equirectangular projection (metres); fine at 25 m over 8 km."""
    R = 6371000.0
    x = np.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R
    return np.stack([x, y], axis=1)


def _resolve_columns(pf):
    names = {n.lower(): n for n in pf.schema_arrow.names}
    def pick(*cands):
        for c in cands:
            if c in names:
                return names[c]
        raise KeyError(f"None of {cands} in parquet columns {pf.schema_arrow.names}")
    return pick("x"), pick("y"), pick("t", "timestamp", "time"), pick("p", "pol", "polarity")


def event_time_range(parquet_path, time_offset=0.0):
    """(t_min, t_max) in epoch seconds + unit scale, reading only the first
    and last row group instead of the whole file."""
    pf = pq.ParquetFile(parquet_path)
    _, _, ct, _ = _resolve_columns(pf)
    n_rg = pf.metadata.num_row_groups
    t_first = float(pf.read_row_group(0, columns=[ct]).column(0)[0].as_py())
    t_last = float(pf.read_row_group(n_rg - 1, columns=[ct]).column(0)[-1].as_py())

    if t_last > 1e17:
        scale = 1e-9
    elif t_last > 1e14:
        scale = 1e-6
    elif t_last > 1e11:
        scale = 1e-3
    elif t_last < 1e7:
        raise RuntimeError(
            f"{parquet_path}: event timestamps look RELATIVE (max={t_last:.3f}). "
            "GPS alignment needs absolute epoch time; set datasets.time_offset.")
    else:
        scale = 1.0
    return t_first * scale + time_offset, t_last * scale + time_offset, scale


def stream_window_events(parquet_path, frame_times, dt, scale, time_offset=0.0,
                         batch_rows=2_000_000):
    """Single sequential pass over a time-sorted parquet. Yields
    (frame_idx, x, y, t, p) per window; buffer only ever holds the current
    read batch + the active window."""
    pf = pq.ParquetFile(parquet_path)
    cx, cy, ct, cp = _resolve_columns(pf)

    bx = np.empty(0, np.int64)
    by = np.empty(0, np.int64)
    bt = np.empty(0, np.float64)
    bp = np.empty(0, np.int64)
    frame_idx = 0
    last_t = -np.inf

    def emit(f_idx):
        f0 = frame_times[f_idx]
        lo = np.searchsorted(bt, f0, side="left")
        hi = np.searchsorted(bt, f0 + dt, side="left")
        return f_idx, bx[lo:hi], by[lo:hi], bt[lo:hi], bp[lo:hi]

    for rb in pf.iter_batches(batch_size=batch_rows, columns=[cx, cy, ct, cp]):
        x = rb.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        y = rb.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        t = rb.column(2).to_numpy(zero_copy_only=False).astype(np.float64) * scale + time_offset
        p = np.where(rb.column(3).to_numpy(zero_copy_only=False).astype(np.int64) > 0, 1, 0)

        if t.size == 0:
            continue
        if t[0] < last_t or np.any(np.diff(t) < 0):
            raise RuntimeError(f"{parquet_path}: events not time-sorted; "
                               "streaming requires a sorted file.")
        last_t = t[-1]

        bx = np.concatenate((bx, x))
        by = np.concatenate((by, y))
        bt = np.concatenate((bt, t))
        bp = np.concatenate((bp, p))

        # bt can be emptied by the post-emit trim when dt < grid step
        # (windows are then disjoint); fall through and read the next batch
        while (frame_idx < len(frame_times) and bt.size > 0
               and frame_times[frame_idx] + dt <= bt[-1]):
            yield emit(frame_idx)
            frame_idx += 1
            if frame_idx < len(frame_times):
                keep = np.searchsorted(bt, frame_times[frame_idx], side="left")
                bx, by, bt, bp = bx[keep:], by[keep:], bt[keep:], bp[keep:]

    while frame_idx < len(frame_times):
        yield emit(frame_idx)
        frame_idx += 1


def resolve_file(directory, key, suffix):
    """Unique file in `directory` matching *key* (dashes/underscores agnostic)."""
    directory = Path(directory)
    norm = lambda s: s.replace("-", "").replace("_", "")
    hits = sorted(f for f in directory.rglob(f"*{suffix}")
                  if norm(key) in norm(f.name))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 '*{suffix}' matching '{key}' under {directory}, "
            f"found {len(hits)}: {[f.name for f in hits]}. Available: "
            f"{[f.name for f in directory.rglob(f'*{suffix}')]}")
    return hits[0]


def process_traverse(name, cfg, model, device):
    """-> descriptors (N, D) L2-normalised, positions (N, 2) metres."""
    modality = cfg.data.modality
    clock_offset = float(cfg.datasets.get("clock_offsets", {}).get(name, 0.0))
    cache_dir = Path(cfg.datasets.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # clock_offset is in the key: it changes the stored xy
    cache = (cache_dir /
             f"{name}_{modality}_dt{cfg.datasets.dt}_co{clock_offset}.pt")
    if cache.exists():
        blob = torch.load(cache, map_location="cpu")
        return blob["desc"], blob["xy"]

    bag = cfg.datasets.traverses[name]
    ev_path = resolve_file(cfg.datasets.root, bag, ".parquet")
    gps_path = resolve_file(cfg.datasets.root, bag, ".nmea")

    time_offset = float(cfg.datasets.get("time_offset", 0.0))
    t_min, t_max, scale = event_time_range(ev_path, time_offset)
    gps_t, lat, lon = parse_nmea_rmc(gps_path)

    overlap = min(t_max, gps_t[-1]) - max(t_min, gps_t[0])
    print(f"  [{name}] events {t_min:.1f}..{t_max:.1f}  gps {gps_t[0]:.1f}.."
          f"{gps_t[-1]:.1f}  overlap {overlap:.1f}s")
    if overlap <= 0:
        raise RuntimeError(
            f"{name}: no time overlap between events and GPS. Check units / "
            "set datasets.time_offset.")

    dt = float(cfg.datasets.dt)
    step = 1.0 / float(cfg.datasets.sample_hz)
    t_start = max(t_min, gps_t[0] - clock_offset)
    t_end = min(t_max - dt, gps_t[-1] - dt - clock_offset)
    frame_times = np.arange(t_start, t_end, step)

    xy_all = latlon_to_xy(lat, lon, float(cfg.datasets.lat0), float(cfg.datasets.lon0))
    # per-recording event-vs-GPS clock correction (see brisbane.yaml)
    tc = frame_times + dt / 2.0 + clock_offset
    xy = np.stack([np.interp(tc, gps_t, xy_all[:, 0]),
                   np.interp(tc, gps_t, xy_all[:, 1])], axis=1)

    model.eval()
    descs, pending = [], []
    n_empty = 0
    batch_size = int(cfg.datasets.batch_size)

    def flush():
        batch = torch.stack(pending).to(device, non_blocking=True)
        _, d = model(batch)
        descs.append(d.float().cpu())
        pending.clear()

    with torch.no_grad():
        stream = stream_window_events(ev_path, frame_times, dt, scale, time_offset)
        for i, ex, ey, et, ep in tqdm(stream, total=len(frame_times),
                                      desc="descriptors", leave=False):
            if et.size == 0:
                n_empty += 1  # kept as zero frames to preserve the regular grid
            pending.append(build_representation(
                ex, ey, et, ep, frame_times[i], dt, modality, cfg))
            if len(pending) == batch_size:
                flush()
        if pending:
            flush()

    print(f"  [{name}] {len(frame_times)} frames @ {cfg.datasets.sample_hz} Hz, "
          f"dt={dt}s, {n_empty} empty bins")

    desc = torch.cat(descs, dim=0)
    desc = torch.nn.functional.normalize(desc, p=2, dim=1)

    xy = torch.from_numpy(xy).float()
    torch.save({"desc": desc, "xy": xy,
                "frame_times": torch.from_numpy(frame_times)}, cache)
    return desc, xy


def recall_at_1(preds, q_xy, r_xy, threshold):
    err = torch.linalg.norm(q_xy - r_xy[preds], dim=1)
    correct = err <= threshold
    r1_total = 100.0 * correct.float().mean().item()
    has_pos = torch.cdist(q_xy, r_xy).min(dim=1).values <= threshold
    r1_valid = (100.0 * correct[has_pos].float().mean().item()
                if has_pos.any() else float("nan"))
    return r1_total, r1_valid, int(has_pos.sum()), len(preds)


def build_model(cfg, device):
    # mirrors train.py exactly
    model = EventViTStudent(
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,
        img_size=cfg.model.img_hw[0],
        in_channels=cfg.data.input_channels,
    )
    print(f"Loading weights: {cfg.phase1_weights}")
    state = torch.load(cfg.phase1_weights, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]   # tolerate a last_*.pth full-state file
    model.load_state_dict(state)
    return model.to(device)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.device)
    modality = cfg.data.modality
    threshold = float(cfg.datasets.gt_threshold_m)
    ref_name = cfg.datasets.reference
    queries = [n for n in cfg.datasets.traverses if n != ref_name]

    model = build_model(cfg, device)

    desc, xy = {}, {}
    for name in cfg.datasets.traverses:
        print(f"Processing {name}")
        desc[name], xy[name] = process_traverse(name, cfg, model, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for q_name in queries:
        dist = -torch.cdist(desc[q_name].to(device), desc[ref_name].to(device))
        for length in cfg.datasets.seq_lengths:
            dist_L = dist if length == 1 else apply_sequence_matching(dist, length)
            preds = dist_L.argmax(dim=1).cpu()
            r1, r1_valid, n_valid, n_q = recall_at_1(
                preds, xy[q_name], xy[ref_name], threshold)
            cond = "day-day" if q_name in DAY_DAY else "day-night"
            rows.append(dict(modality=modality, reference=ref_name, query=q_name,
                             condition=cond, seq_len=length, recall_at_1=r1,
                             recall_at_1_validonly=r1_valid, n_queries=n_q,
                             n_queries_with_gt=n_valid,
                             dt=float(cfg.datasets.dt), threshold_m=threshold))
            print(f"  {ref_name} vs {q_name:8s} L={length:2d}: R@1={r1:6.2f}%  "
                  f"(valid-only {r1_valid:6.2f}%, {n_valid}/{n_q} queries "
                  f"have a {threshold:.0f}m positive)")

    out_csv = Path(str(cfg.datasets.results_csv).format(
        modality=modality, dt=cfg.datasets.dt))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {out_csv}")

    print(f"\n{'='*70}\nSUMMARY [{modality}, dt={cfg.datasets.dt}s] "
          f"(mean over seq lengths > 1)")
    for cond in ("day-day", "day-night"):
        vals = [r["recall_at_1"] for r in rows
                if r["condition"] == cond and r["seq_len"] > 1]
        if vals:
            print(f"  {cond:10s}: mean R@1 = {np.mean(vals):.2f}%")


if __name__ == "__main__":
    main()
