"""Post-hoc diagnostics over the descriptor caches written by
evaluate_brisbane.py. No model or events needed -- just the *.pt caches.

Reports, per (modality, dt):
  1. Descriptor-space health per traverse: mean/std off-diagonal cosine
     similarity (near 1.0 = collapsed around a common component) and the
     top-5 PCA explained-variance ratios of the centred descriptors.
  2. Retrieval vs sunset1 at L=1, raw and reference-centred (subtract the
     reference-set mean descriptor, re-normalise both sides): R@1 @ 25 m
     and best-match GPS-error percentiles. Error percentiles separate
     "near miss" (median slightly over 25 m) from "uniform over route".
  3. Similarity-matrix PNGs (query x reference) per traverse -- a visible
     diagonal that the threshold misses reads very differently from noise.

Usage:
  python diagnose_brisbane.py --cache_dir /kaggle/working/brisbane_cache \
      --modality voxel --dt 1.0 --out_dir /kaggle/working/diagnostics
"""
import argparse
from pathlib import Path

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:          # stats still print without matplotlib
    plt = None

TRAVERSES = ["sunset1", "sunset2", "daytime", "morning", "sunrise", "night"]
REFERENCE = "sunset1"
THRESHOLD_M = 25.0

# recording keys, mirrors configs/datasets/brisbane.yaml
KEYS = {
    "sunset1": "2020-04-21-17-03-03",
    "sunset2": "2020-04-22-17-24-21",
    "daytime": "2020-04-24-15-12-03",
    "morning": "2020-04-28-09-14-11",
    "sunrise": "2020-04-29-06-20-23",
    "night":   "2020-04-27-18-13-29",
}


def load_cache(cache_dir, name, modality, dt):
    path = Path(cache_dir) / f"{name}_{modality}_dt{dt}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run evaluate_brisbane.py first)")
    blob = torch.load(path, map_location="cpu")
    return blob["desc"], blob["xy"]


def descriptor_stats(desc):
    G = desc @ desc.T
    n = G.shape[0]
    off = G[~torch.eye(n, dtype=torch.bool)]
    centred = desc - desc.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centred)
    evr = (sv**2 / (sv**2).sum())[:5]
    return off.mean().item(), off.std().item(), evr.tolist()


def retrieval(q_desc, q_xy, r_desc, r_xy):
    pred = (q_desc @ r_desc.T).argmax(dim=1)
    err = torch.linalg.norm(q_xy - r_xy[pred], dim=1)
    r1 = 100.0 * (err <= THRESHOLD_M).float().mean().item()
    pct = torch.quantile(err, torch.tensor([0.25, 0.5, 0.75, 0.9])).tolist()
    return r1, pct


def centre(desc, mean_vec):
    return torch.nn.functional.normalize(desc - mean_vec, p=2, dim=1)


def fit_whitener(ref_desc, shrink=0.1):
    """PCA whitening learned on the reference map only (query-blind, so
    still zero-shot). shrink regularises small eigenvalues."""
    mean_vec = ref_desc.mean(dim=0, keepdim=True)
    X = ref_desc - mean_vec
    _, s, Vh = torch.linalg.svd(X, full_matrices=False)
    scale = 1.0 / torch.sqrt(s**2 / (X.shape[0] - 1) + shrink * (s**2).mean()
                             / (X.shape[0] - 1))
    W = Vh.T * scale.unsqueeze(0)
    return mean_vec, W


def whiten(desc, mean_vec, W):
    return torch.nn.functional.normalize((desc - mean_vec) @ W, p=2, dim=1)


def shift_along_track(xy, delta_s, step_s=1.0):
    """Approximate the position delta_s seconds later along the same
    trajectory, using finite-difference velocity from the 1 Hz grid."""
    vel = torch.zeros_like(xy)
    vel[:-1] = (xy[1:] - xy[:-1]) / step_s
    vel[-1] = vel[-2]
    return xy + delta_s * vel


def _parse_rmc_time_speed(path):
    """(t_epoch[s], speed[m/s]) from $__RMC sentences (field 7 = knots)."""
    from datetime import datetime, timezone
    times, speeds = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "RMC" not in line:
                continue
            fields = line.strip().split(",")
            try:
                if len(fields) < 10 or fields[2] != "A":
                    continue
                base = datetime.strptime(
                    fields[9] + fields[1].split(".")[0], "%d%m%y%H%M%S"
                ).replace(tzinfo=timezone.utc).timestamp()
                frac = (float("0." + fields[1].split(".")[1])
                        if "." in fields[1] else 0.0)
                speeds.append(float(fields[7]) * 0.514444)
                times.append(base + frac)
            except (ValueError, IndexError):
                continue
    if not times:
        raise RuntimeError(f"no RMC speed fixes in {path}")
    import numpy as np
    order = np.argsort(times)
    return (np.asarray(times)[order], np.asarray(speeds)[order])


def _event_rate(parquet_path, res):
    """Event count per `res`-second bin + the bin grid start (epoch s)."""
    import numpy as np
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    names = {n.lower(): n for n in pf.schema_arrow.names}
    ct = next(names[c] for c in ("t", "timestamp", "time") if c in names)
    t_last = float(pf.read_row_group(
        pf.metadata.num_row_groups - 1, columns=[ct]).column(0)[-1].as_py())
    scale = 1e-9 if t_last > 1e17 else 1e-6 if t_last > 1e14 else \
        1e-3 if t_last > 1e11 else 1.0
    counts, t0 = None, None
    for rb in pf.iter_batches(batch_size=5_000_000, columns=[ct]):
        t = rb.column(0).to_numpy(zero_copy_only=False).astype(np.float64) * scale
        if t0 is None:
            t0 = t[0]
        idx = ((t - t0) / res).astype(np.int64)
        c = np.bincount(idx)
        if counts is None:
            counts = c
        else:
            if len(c) > len(counts):
                counts = np.pad(counts, (0, len(c) - len(counts)))
            counts[:len(c)] += np.pad(c, (0, len(counts) - len(c)))
    return counts.astype(np.float64), t0


def estimate_clock_offset(ev_path, gps_path, res=0.1, max_lag=15.0):
    """Descriptor-free event/GPS clock offset via cross-correlation of
    event rate against GPS speed-over-ground. Returns (offset, peak_corr):
    GPS should be sampled at event_time + offset. Positive offset = event
    clock behind GPS clock (same sign convention as the retrieval sweep)."""
    import numpy as np
    rate, t0 = _event_rate(ev_path, res)
    gps_t, sp = _parse_rmc_time_speed(gps_path)
    grid = t0 + np.arange(len(rate)) * res
    valid = (grid >= gps_t[0] - max_lag) & (grid <= gps_t[-1] + max_lag)
    a = rate[valid] - rate[valid].mean()
    g = grid[valid]
    max_k = int(round(max_lag / res))
    best = (0.0, -np.inf)
    for k in range(-max_k, max_k + 1):
        s = np.interp(g + k * res, gps_t, sp)
        s = s - s.mean()
        denom = a.std() * s.std()
        if denom == 0:
            continue
        c = float((a * s).mean() / denom)
        if c > best[1]:
            best = (k * res, c)
    return best


def offset_sweep(q_desc, q_xy, r_desc, r_xy, deltas):
    """R@1 and median best-match error as a function of a time offset
    applied to the QUERY GPS interpolation. A sharp off-zero optimum means
    the event and GPS clocks disagree (-> datasets.time_offset)."""
    pred = (q_desc @ r_desc.T).argmax(dim=1)
    rows = []
    for d in deltas:
        err = torch.linalg.norm(
            shift_along_track(q_xy, d) - r_xy[pred], dim=1)
        rows.append((d, 100.0 * (err <= THRESHOLD_M).float().mean().item(),
                     err.median().item()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--dt", required=True,
                    help="as it appears in the cache filename, e.g. 1.0 or 0.05")
    ap.add_argument("--out_dir", default="diagnostics")
    ap.add_argument("--root", default=None,
                    help="dataset root with *.parquet/*.nmea; enables the "
                         "descriptor-free clock-offset estimation")
    ap.add_argument("--max_lag", type=float, default=15.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.root:
        print("=== descriptor-free clock offsets: event rate x GPS speed "
              f"(sample GPS at event_time + offset; lag range +-{args.max_lag:.0f}s) ===")
        norm = lambda s: s.replace("-", "").replace("_", "")
        root = Path(args.root)
        for name in TRAVERSES:
            find = lambda suf: next(
                f for f in sorted(root.rglob(f"*{suf}"))
                if norm(KEYS[name]) in norm(f.name))
            off, corr = estimate_clock_offset(
                find(".parquet"), find(".nmea"), max_lag=args.max_lag)
            flag = "" if corr > 0.3 else "   (weak peak -- treat as unreliable)"
            print(f"  {name:10s} offset = {off:+5.1f}s   peak_corr = "
                  f"{corr:.3f}{flag}")
        print()

    desc, xy = {}, {}
    print(f"=== descriptor health [{args.modality}, dt={args.dt}] ===")
    print(f"{'traverse':10s} {'cos_mean':>8s} {'cos_std':>8s}  top-5 PCA EVR")
    for name in TRAVERSES:
        desc[name], xy[name] = load_cache(
            args.cache_dir, name, args.modality, args.dt)
        m, s, evr = descriptor_stats(desc[name])
        evr_s = " ".join(f"{v:.3f}" for v in evr)
        print(f"{name:10s} {m:8.4f} {s:8.4f}  [{evr_s}]")

    ref_mean = desc[REFERENCE].mean(dim=0, keepdim=True)
    r_raw = desc[REFERENCE]
    r_cen = centre(desc[REFERENCE], ref_mean)
    w_mean, W = fit_whitener(desc[REFERENCE])
    r_wht = whiten(desc[REFERENCE], w_mean, W)

    print(f"\n=== L=1 retrieval vs {REFERENCE} (R@1 @ {THRESHOLD_M:.0f} m; "
          f"best-match error percentiles, metres) ===")
    print(f"{'query':10s} {'raw R@1':>8s} {'ctr R@1':>8s} {'wht R@1':>8s}   "
          f"{'raw p25/p50/p75/p90':>24s}   {'wht p25/p50/p75/p90':>24s}")
    for name in TRAVERSES:
        if name == REFERENCE:
            continue
        r1, pct = retrieval(desc[name], xy[name], r_raw, xy[REFERENCE])
        r1c, _ = retrieval(centre(desc[name], ref_mean), xy[name],
                           r_cen, xy[REFERENCE])
        r1w, pctw = retrieval(whiten(desc[name], w_mean, W), xy[name],
                              r_wht, xy[REFERENCE])
        fmt = lambda p: "/".join(f"{v:.0f}" for v in p)
        print(f"{name:10s} {r1:7.2f}% {r1c:7.2f}% {r1w:7.2f}%   "
              f"{fmt(pct):>24s}   {fmt(pctw):>24s}")

    print(f"\n=== query-clock offset sweep (whitened desc; positions shifted "
          f"delta s along query trajectory) ===")
    print("A sharp optimum away from 0 = event/GPS clock disagreement for "
          "that traverse pair -> per-traverse time_offset, NOT a descriptor "
          "problem.")
    deltas = [d / 2.0 for d in range(-20, 21)]
    print(f"{'query':10s} {'R@1(d=0)':>9s} {'best d':>7s} {'R@1(d*)':>8s} "
          f"{'med_err(d*)':>11s}")
    for name in TRAVERSES:
        if name == REFERENCE:
            continue
        rows = offset_sweep(whiten(desc[name], w_mean, W), xy[name],
                            r_wht, xy[REFERENCE], deltas)
        r1_0 = next(r for r in rows if r[0] == 0.0)[1]
        d_star, r1_star, med_star = max(rows, key=lambda r: (r[1], -r[2]))
        print(f"{name:10s} {r1_0:8.2f}% {d_star:6.1f}s {r1_star:7.2f}% "
              f"{med_star:10.1f}m")
        if name == "sunset2":
            curve = "  ".join(f"{d:+.1f}s:{r1:.1f}" for d, r1, _ in rows)
            print(f"           full curve: {curve}")

    for name in TRAVERSES:
        if name == REFERENCE or plt is None:
            continue
        sim = (centre(desc[name], ref_mean) @ r_cen.T).numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(sim, aspect="auto", cmap="viridis")
        ax.set_xlabel(f"{REFERENCE} (reference idx)")
        ax.set_ylabel(f"{name} (query idx)")
        ax.set_title(f"centred cosine sim: {name} vs {REFERENCE} "
                     f"[{args.modality}, dt={args.dt}]")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        png = out_dir / f"sim_{name}_{args.modality}_dt{args.dt}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"           -> {png}")


if __name__ == "__main__":
    main()
