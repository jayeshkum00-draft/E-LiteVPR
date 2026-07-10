import csv
from datetime import datetime, timezone
import math
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
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

def load_events(parquet_path, time_offset=0.0):
    """-> x:int64, y:int64, t:float64 epoch seconds (sorted), p:int64 {0,1}."""
    df = pd.read_parquet(parquet_path)
    cols = {c.lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return df[cols[n]].to_numpy()
        raise KeyError(f"None of {names} in parquet columns {list(df.columns)}")

    x = col("x").astype(np.int64)
    y = col("y").astype(np.int64)
    t = col("t", "timestamp", "time").astype(np.float64)
    p = col("p", "pol", "polarity").astype(np.int64)

    # time-unit detection: epoch s ~1.6e9, ms ~1.6e12, us ~1.6e15, ns ~1.6e18
    tmax = t.max()
    if tmax > 1e17:
        t *= 1e-9
    elif tmax > 1e14:
        t *= 1e-6
    elif tmax > 1e11:
        t *= 1e-3
    elif tmax < 1e7:
        raise RuntimeError(
            f"{parquet_path}: event timestamps look RELATIVE (max={tmax:.3f}). "
            "GPS alignment needs absolute epoch time; set brisbane.time_offset.")
    t += time_offset

    p = np.where(p > 0, 1, 0).astype(np.int64)  # {-1,1}/bool -> {0,1}

    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    return x, y, t, p


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

class BrisbaneBinDataset(Dataset):
    def __init__(self, events, frame_times, dt, modality, cfg):
        self.x, self.y, self.t, self.p = events
        self.frame_times = frame_times
        self.dt = dt
        self.modality = modality
        self.cfg = cfg

    def __len__(self):
        return len(self.frame_times)

    def __getitem__(self, i):
        t0 = self.frame_times[i]
        lo = np.searchsorted(self.t, t0, side="left")
        hi = np.searchsorted(self.t, t0 + self.dt, side="left")
        return build_representation(
            self.x[lo:hi], self.y[lo:hi], self.t[lo:hi], self.p[lo:hi],
            t0, self.dt, self.modality, self.cfg)


@torch.no_grad()
def extract_descriptors(model, loader, device):
    model.eval()
    out = []
    for batch in tqdm(loader, desc="descriptors", leave=False):
        batch = batch.to(device, non_blocking=True)
        _, desc = model(batch)   # (projected_patches, GeM global descriptor)
        out.append(desc.float().cpu())
    return torch.cat(out, dim=0)


def process_traverse(name, cfg, model, device):
    """-> descriptors (N, D) L2-normalised, positions (N, 2) metres."""
    modality = cfg.data.modality
    cache_dir = Path(cfg.datasets.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}_{modality}_dt{cfg.datasets.dt}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu")
        return blob["desc"], blob["xy"]

    bag = cfg.datasets.traverses[name]
    ev_path = resolve_file(cfg.datasets.root, bag, ".parquet")
    gps_path = resolve_file(cfg.datasets.root, bag, ".nmea")

    x, y, t, p = load_events(ev_path, float(cfg.datasets.get("time_offset", 0.0)))
    gps_t, lat, lon = parse_nmea_rmc(gps_path)

    overlap = min(t[-1], gps_t[-1]) - max(t[0], gps_t[0])
    print(f"  [{name}] events {t[0]:.1f}..{t[-1]:.1f}  gps {gps_t[0]:.1f}.."
          f"{gps_t[-1]:.1f}  overlap {overlap:.1f}s")
    if overlap <= 0:
        raise RuntimeError(
            f"{name}: no time overlap between events and GPS. Check units / "
            "set brisbane.time_offset.")

    dt = float(cfg.datasets.dt)
    step = 1.0 / float(cfg.datasets.sample_hz)   # paper: 1 Hz grid
    t_start = max(t[0], gps_t[0])
    t_end = min(t[-1] - dt, gps_t[-1] - dt)
    frame_times = np.arange(t_start, t_end, step)

    # GT position at the bin centre
    xy_all = latlon_to_xy(lat, lon, float(cfg.datasets.lat0), float(cfg.datasets.lon0))
    tc = frame_times + dt / 2.0
    xy = np.stack([np.interp(tc, gps_t, xy_all[:, 0]),
                   np.interp(tc, gps_t, xy_all[:, 1])], axis=1)

    # empty bins kept as all-zero representations to preserve the regular
    # grid that sequence matching relies on
    lo = np.searchsorted(t, frame_times, side="left")
    hi = np.searchsorted(t, frame_times + dt, side="left")
    n_empty = int((hi - lo == 0).sum())
    print(f"  [{name}] {len(frame_times)} frames @ {cfg.datasets.sample_hz} Hz, "
          f"dt={dt}s, {n_empty} empty bins")

    ds = BrisbaneBinDataset((x, y, t, p), frame_times, dt, modality, cfg)
    loader = DataLoader(ds, batch_size=cfg.datasets.batch_size, shuffle=False,
                        num_workers=cfg.datasets.num_workers, pin_memory=True)
    desc = extract_descriptors(model, loader, device)
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
    state = torch.load(cfg.model_weights, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]   # tolerate a last_*.pth full-state file
    model.load_state_dict(state)
    return model.to(device)

@hydra.main(version_base=None, config_path="../configs", config_name="brisbane_eval")
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