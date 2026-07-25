"""WiSE-FT-style weight interpolation between the Phase 1 student and the
Phase 2 fine-tuned backbone (Wortsman et al., CVPR 2022).

    theta_merged = (1 - alpha) * theta_phase1 + alpha * theta_phase2

Valid because Phase 2 was initialised FROM Phase 1 (aligned weights, same
architecture); the merge covers the student backbone only -- the Phase 2
head is a training-time device and is discarded. Output files are plain
EventViTStudent state dicts, drop-in for evaluate_brisbane.py via
`phase1_weights=<merged>.pth` (no phase2 flags).

Float parameters and BN running stats are interpolated; integer buffers
(num_batches_tracked) are copied from Phase 1. alpha=0.5 is the
pre-registered headline point; the other alphas exist for the ablation
curve only.

Usage:
  python utils/wise_merge.py \
      --phase1 best_phase1_histogram.pth \
      --phase2 best_phase2_histogram.pth \
      --alphas 0.25 0.5 0.75 --out_dir merged/
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", required=True,
                    help="EventViTStudent state dict (best_phase1_*.pth)")
    ap.add_argument("--phase2", required=True,
                    help="Phase2Net state dict (best_phase2_*.pth)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--out_dir", default="merged")
    args = ap.parse_args()

    p1 = torch.load(args.phase1, map_location="cpu")
    if isinstance(p1, dict) and "model_state" in p1:
        p1 = p1["model_state"]
    p2_full = torch.load(args.phase2, map_location="cpu")
    if isinstance(p2_full, dict) and "model_state" in p2_full:
        p2_full = p2_full["model_state"]

    # pull the backbone out of the Phase2Net checkpoint (head discarded)
    p2 = {k[len("student."):]: v for k, v in p2_full.items()
          if k.startswith("student.")}
    if set(p1.keys()) != set(p2.keys()):
        only1 = sorted(set(p1) - set(p2))[:5]
        only2 = sorted(set(p2) - set(p1))[:5]
        raise RuntimeError(
            f"checkpoints do not align: only-phase1 {only1}, "
            f"only-phase2 {only2}")

    # how far did fine-tuning move? (context for the merge)
    flat_deltas = [(p2[k] - p1[k]).float().abs().mean().item()
                   for k in p1 if p1[k].is_floating_point()]
    print(f"{len(p1)} tensors; mean |delta| over float tensors: "
          f"{sum(flat_deltas) / len(flat_deltas):.2e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in args.alphas:
        merged = {}
        for k in p1:
            if p1[k].is_floating_point():
                merged[k] = (1.0 - a) * p1[k] + a * p2[k]
            else:                        # e.g. num_batches_tracked (int)
                merged[k] = p1[k].clone()
        out = out_dir / f"merged_alpha{a:g}.pth"
        torch.save(merged, out)
        print(f"alpha={a:g} -> {out}")


if __name__ == "__main__":
    main()
