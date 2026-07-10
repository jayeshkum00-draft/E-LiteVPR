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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--dt", required=True,
                    help="as it appears in the cache filename, e.g. 1.0 or 0.05")
    ap.add_argument("--out_dir", default="diagnostics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

        if plt is None:
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
