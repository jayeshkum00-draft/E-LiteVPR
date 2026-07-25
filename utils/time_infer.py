"""End-to-end per-query inference latency for each method, on one GPU.

end-to-end = reconstruction (event window -> frame) + descriptor (model
forward). Sequence matching reuses cached descriptors and is ~free per query,
so it is omitted. Both stages are measured on the same box so the numbers are
directly comparable.

- Reconstruction: eliteHistogram (ours) vs eventCount / timeSurface / e2vid.
  E2VID is a GPU model (the baselines' expensive step); eliteHistogram /
  eventCount / timeSurface are cheap CPU/array ops.
- Descriptor: elitevpr (ViT-S) vs mixvpr / cosplace / netvlad / megaloc.

Compose per method:  end_to_end = recon_ms[its recon] + desc_ms[its model].
E-LiteVPR uses eliteHistogram; the reconstruction-VPR baselines use e2vid
(their strongest config) — that E2VID cost is what E-LiteVPR avoids.

Lives in <repo>/utils/, imports the bench in <repo>/external/
ensemble_event_vpr_bench/. Pre-warm torch.hub methods once before running.

Usage:  python utils/time_infer.py   ->  writes inference_times.txt in the CWD
"""
import importlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, ".."))
BENCH = os.path.join(REPO, "external", "ensemble_event_vpr_bench")
OUT = os.path.join(os.getcwd(), "inference_times.txt")

sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(BENCH, "vpr_methods_evaluation"))
os.chdir(BENCH)

import numpy as np
import pyarrow.parquet as pq
import torch
import vpr_models
from vpr_methods_evaluation.parse import parse_arguments

WARMUP, ITERS = 20, 200
DEVICE = "cuda"
SENSOR = (346, 260)                                    # Brisbane DAVIS346 (W, H)
RECON_METHODS = ["eliteHistogram", "eventCount", "timeSurface", "e2vid"]
VPR_METHODS = ["elitevpr", "mixvpr", "cosplace", "netvlad", "megaloc"]

_DS = os.path.join(BENCH, "datasample_for_ensem_event_bench",
                   "Brisbane", "paraquet_data", "sunset1")


def load_frames(dt=1.0, max_frames=None):
    """sunset1 events as dt-second frames (real per-frame event density).
    max_frames caps how many frames are timed (keeps fine dt fast); None = all
    (full traverse). Hot mask amortised over the frames timed, as the pipeline
    does it."""
    tab = pq.read_table(os.path.join(_DS, "events.parquet"),
                        columns=["t", "x", "y", "p"])
    t = tab.column("t").to_numpy()
    x = tab.column("x").to_numpy().astype(np.int64)
    y = tab.column("y").to_numpy().astype(np.int64)
    p = tab.column("p").to_numpy().astype(np.int64)
    bin_starts = np.arange(t[0], t[-1] - dt, dt)
    if max_frames is not None:
        bin_starts = bin_starts[:max_frames]
    hi = np.searchsorted(t, bin_starts[-1] + dt, side="right")   # only load events needed
    t, x, y, p = t[:hi], x[:hi], y[:hi], p[:hi]
    ev = np.core.records.fromarrays(
        [t, x, y, p],
        dtype=np.dtype([("t", np.float64), ("x", int), ("y", int), ("p", int)]))
    s_idx = np.searchsorted(t, bin_starts, side="left")
    e_idx = np.searchsorted(t, bin_starts + dt, side="right")
    keep = s_idx != e_idx
    return ev, s_idx[keep], e_idx[keep]


def time_recon(name, ev, s_idx, e_idx):
    """Amortized ms/frame: reconstruct the whole N-frame run, divide by N."""
    mod = importlib.import_module(f"reconstruction.{name}")
    rec = mod.EventReconstructor()
    hp = os.path.join(_DS, "hot_pixels.txt")
    hp = hp if os.path.exists(hp) else None
    n_frames = len(s_idx)
    call = (lambda: rec.reconstruct(ev, SENSOR, s_idx, e_idx, hp_loc=hp))
    for _ in range(1):                                  # warmup (whole traverse)
        call()
    torch.cuda.synchronize()
    K = 3
    t0 = time.time()
    for _ in range(K):
        call()
    torch.cuda.synchronize()
    per_run = (time.time() - t0) / K
    return per_run / n_frames * 1000.0, n_frames        # ms per frame


def time_desc(m):
    args = parse_arguments(m)
    model = vpr_models.get_model(m, args.backbone, args.descriptors_dimension)
    model = model.eval().to(DEVICE)
    S = args.image_size[0] if getattr(args, "image_size", None) else 224
    x = torch.randn(1, 3, S, S, device=DEVICE)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ITERS):
            model(x)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / ITERS * 1000.0
    del model
    torch.cuda.empty_cache()
    return ms, S, args.descriptors_dimension


def time_match(D, n_ref=700):
    """Per-query retrieval: query descriptor vs an n_ref reference database
    (matrix-vector + argmax). Value-independent, so random tensors are fine."""
    q = torch.randn(1, D, device=DEVICE)
    db = torch.randn(n_ref, D, device=DEVICE)
    with torch.no_grad():
        for _ in range(20):
            _ = (q @ db.T).argmax()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(500):
            _ = (q @ db.T).argmax()
        torch.cuda.synchronize()
    return (time.time() - t0) / 500 * 1000.0


def main():
    lines = []
    ev, s_idx, e_idx = load_frames(dt=1.0)
    lines.append(f"# FULL sunset1 traverse: {len(s_idx)} frames, {len(ev):,} events "
                 f"({len(ev)/len(s_idx):,.0f} ev/frame avg); reconstruction = amortized ms/frame\n")
    lines.append("== reconstruction (ms / frame, amortized) ==")
    recon_all = {}
    for r in RECON_METHODS:
        try:
            ms, nf = time_recon(r, ev, s_idx, e_idx)
            recon_all[r] = ms
            lines.append(f"{r:16s} {ms:8.2f} ms/frame  (over {nf} frames)")
        except Exception as e:
            lines.append(f"{r:16s} FAILED: {e}")
    lines.append("\n== descriptor forward (ms / query) ==")
    desc_all, desc_dim = {}, {}
    for m in VPR_METHODS:
        try:
            ms, S, dim = time_desc(m)
            desc_all[m], desc_dim[m] = ms, dim
            lines.append(f"{m:16s} {ms:8.2f} ms/query   (input {S}x{S}, desc_dim {dim})")
        except Exception as e:
            lines.append(f"{m:16s} FAILED: {e}")
    lines.append("\n== matching (retrieval vs 700-ref DB, ms / query) ==")
    match_ms = {}
    for m in VPR_METHODS:
        try:
            match_ms[m] = time_match(desc_dim[m])
            lines.append(f"{m:16s} {match_ms[m]:8.3f} ms/query  (desc_dim {desc_dim[m]})")
        except Exception as e:
            lines.append(f"{m:16s} FAILED: {e}")

    # E-LiteVPR single, fully measured end-to-end on this box
    elite = (recon_all.get("eliteHistogram", float("nan"))
             + desc_all.get("elitevpr", float("nan"))
             + match_ms.get("elitevpr", 0.0))
    lines.append(f"\n# E-LiteVPR single end-to-end = "
                 f"eliteHistogram {recon_all.get('eliteHistogram', float('nan')):.1f}"
                 f" + ViT {desc_all.get('elitevpr', float('nan')):.1f}"
                 f" + match {match_ms.get('elitevpr', 0.0):.2f}"
                 f" = {elite:.1f} ms")

    # ---- combined ensemble per query (measured on this box) ----
    # 4 recon x 4 time-res = 16 reconstructed frames; each -> 4 VPR = 64 forwards.
    ENS_RECON = ["eventCount", "eventCount_noPolarity", "timeSurface", "e2vid"]
    ENS_VPR = ["mixvpr", "cosplace", "netvlad", "megaloc"]
    ENS_TR = [0.1, 0.25, 0.5, 1.0]
    lines.append("\n== combined ensemble per query (measured) ==")
    recon_total = 0.0
    for tr in ENS_TR:
        ev_tr, s_tr, e_tr = load_frames(dt=tr, max_frames=40)
        for r in ENS_RECON:
            try:
                ms, _ = time_recon(r, ev_tr, s_tr, e_tr)
                recon_total += ms
                lines.append(f"  recon {r:20s} @ {tr:<4}s = {ms:7.2f} ms")
            except Exception as ex:
                lines.append(f"  recon {r:20s} @ {tr:<4}s FAILED: {ex}")
    desc_total = 16 * sum(desc_all.get(m, float("nan")) for m in ENS_VPR)   # 64 forwards
    match_total = 16 * sum(match_ms.get(m, 0.0) for m in ENS_VPR)           # 64 retrievals
    ens = recon_total + desc_total + match_total
    lines.append(f"  reconstruction total (16)  = {recon_total:8.1f} ms")
    lines.append(f"  descriptor total (64 fwd)  = {desc_total:8.1f} ms")
    lines.append(f"  matching total (64 retr)   = {match_total:8.1f} ms")
    ratio = ens / elite if elite and elite == elite else float("nan")
    lines.append(f"  ENSEMBLE per query (+fusion ~0) ~ {ens:8.1f} ms   "
                 f"vs E-LiteVPR ~ {elite:.1f} ms   ({ratio:.1f}x, same hardware)")
    out = "\n".join(lines) + "\n"
    print(out)
    with open(OUT, "w") as f:
        f.write(out)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
