"""Recall@1 vs sequence length from the descriptor caches, with the
stationarity analysis: does longer sequence matching degrade recall, and if
so, is the degradation confined to windows that contain stationary frames?

Three curves per query traverse:
  all          -- every frame (protocol of the raw tables)
  all/no-stop  -- 'all' recall restricted to queries whose L-window is
                  fully moving (stratum A)
  all/stop     -- 'all' recall restricted to queries whose L-window contains
                  a stationary segment (stratum B; the slope-1 diagonal
                  assumption of sequence matching is violated here)
  moving       -- stationary frames removed from query AND reference before
                  matching (the literature-standard protocol)

Usage:
  python recall_vs_length.py --cache_dir /kaggle/working/brisbane_cache \
      --modality histogram --dt 1.0 --out_dir /kaggle/working/diagnostics
"""
import argparse
import csv
from pathlib import Path

import torch

from sequence_matching import apply_sequence_matching

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

TRAVERSES = ["sunset1", "sunset2", "daytime", "morning", "sunrise", "night"]
REFERENCE = "sunset1"


def load_cache(cache_dir, name, modality, dt):
    hits = sorted(Path(cache_dir).glob(f"{name}_{modality}_dt{dt}*.pt"),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(
            f"no cache {name}_{modality}_dt{dt}*.pt in {cache_dir}")
    if len(hits) > 1:
        print(f"  note: {len(hits)} caches match {name}; using newest "
              f"({hits[-1].name})")
    blob = torch.load(hits[-1], map_location="cpu")
    return blob["desc"], blob["xy"]


def frame_speeds(xy, step):
    if len(xy) < 2:
        return torch.zeros(len(xy))
    v = torch.linalg.norm(xy[1:] - xy[:-1], dim=1) / step
    return torch.cat([v, v[-1:]])


def window_has_stop(stop_mask, L):
    """bool per query: any stationary frame in its causal L-window."""
    out = torch.zeros(len(stop_mask), dtype=torch.bool)
    s = stop_mask.float()
    c = torch.cat([torch.zeros(1), s.cumsum(0)])
    for q in range(len(s)):
        lo = max(0, q - L + 1)
        out[q] = (c[q + 1] - c[lo]) > 0
    return out


def r1_of(preds, q_xy, r_xy, thr, sel=None):
    err = torch.linalg.norm(q_xy - r_xy[preds], dim=1)
    ok = err <= thr
    if sel is not None:
        ok = ok[sel]
    return (100.0 * ok.float().mean().item(), int(ok.numel())) \
        if ok.numel() else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--threshold_m", type=float, default=25.0)
    ap.add_argument("--min_speed", type=float, default=1.0)
    ap.add_argument("--sample_hz", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=40)
    ap.add_argument("--out_dir", default="diagnostics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 1.0 / args.sample_hz

    desc, xy, stop = {}, {}, {}
    for n in TRAVERSES:
        desc[n], xy[n] = load_cache(args.cache_dir, n, args.modality, args.dt)
        stop[n] = frame_speeds(xy[n], step) < args.min_speed
        print(f"  {n}: {int(stop[n].sum())}/{len(stop[n])} stationary frames "
              f"(< {args.min_speed} m/s)")

    rk = ~stop[REFERENCE]
    rows = []
    for q in TRAVERSES:
        if q == REFERENCE:
            continue
        qk = ~stop[q]
        d_all = -torch.cdist(desc[q], desc[REFERENCE])
        d_mov = -torch.cdist(desc[q][qk], desc[REFERENCE][rk])
        for L in range(1, args.max_len + 1):
            dl = d_all if L == 1 else apply_sequence_matching(d_all, L)
            preds = dl.argmax(dim=1)
            has_stop = window_has_stop(stop[q], L)
            for name, sel in [("all", None), ("all/no-stop", ~has_stop),
                              ("all/stop", has_stop)]:
                r1, n = r1_of(preds, xy[q], xy[REFERENCE],
                              args.threshold_m, sel)
                rows.append(dict(query=q, variant=name, seq_len=L,
                                 recall_at_1=r1, n_queries=n))
            dl = d_mov if L == 1 else apply_sequence_matching(d_mov, L)
            r1, n = r1_of(dl.argmax(dim=1), xy[q][qk], xy[REFERENCE][rk],
                          args.threshold_m)
            rows.append(dict(query=q, variant="moving", seq_len=L,
                             recall_at_1=r1, n_queries=n))
        best = {v: max((r for r in rows if r["query"] == q
                        and r["variant"] == v), key=lambda r: r["recall_at_1"])
                for v in ("all", "moving")}
        print(f"  {q}: all peaks {best['all']['recall_at_1']:.1f}% @ "
              f"L={best['all']['seq_len']}, moving peaks "
              f"{best['moving']['recall_at_1']:.1f}% @ "
              f"L={best['moving']['seq_len']}")

    out_csv = out_dir / f"recall_vs_length_{args.modality}_dt{args.dt}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out_csv}")

    if plt is not None:
        for q in TRAVERSES:
            if q == REFERENCE:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for v, style in [("all", "-"), ("all/no-stop", "--"),
                             ("all/stop", ":"), ("moving", "-.")]:
                pts = [(r["seq_len"], r["recall_at_1"]) for r in rows
                       if r["query"] == q and r["variant"] == v]
                ax.plot(*zip(*pts), style, label=v)
            ax.set_xlabel("sequence length L")
            ax.set_ylabel("R@1 (%)")
            ax.set_title(f"{q} vs {REFERENCE} [{args.modality}, dt={args.dt}]")
            ax.legend()
            fig.tight_layout()
            png = out_dir / f"recall_vs_L_{q}_{args.modality}_dt{args.dt}.png"
            fig.savefig(png, dpi=120)
            plt.close(fig)
            print(f"  -> {png}")


if __name__ == "__main__":
    main()
