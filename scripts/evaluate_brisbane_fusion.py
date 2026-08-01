"""Is the MegaLoc student and the DINOv3 student's strength complementary?

CLAIM UNDER TEST
----------------
Measured on Brisbane [all], sunset1 ref, 25 m, histogram dt=1.0:

  night R@1        L=1    L=10   L=20   L=30   mean   L1->L30
  DINOv3 raw loss  1.92   18.58  39.82  53.39  37.27   27.8x
  MegaLoc + fix    5.31   15.78  26.11  34.81  25.57    6.6x

The MegaLoc student wins EVERY single-frame cell (day and night) and the
DINOv3 student wins every sequence cell. So they are not ranked -- they encode
different things: per-frame place discrimination vs temporally coherent
descriptors that 30-frame matching can integrate.

If that is real, fusing the two descriptors should beat BOTH on the night mean
while keeping MegaLoc's L=1. If it is not -- if fusion just interpolates
between the two curves -- then they are redundant and a multi-teacher training
run would be wasted GPU time. This script decides that from the two
checkpoints that already exist, with no training.

Descriptors are L2-normalised by process_traverse, so the fusion

    d = [sqrt(a) * d_A , sqrt(1-a) * d_B]

is itself unit-norm and its inner product is a*cos_A + (1-a)*cos_B: an exact
convex blend of the two similarity matrices, swept in one pass because
process_traverse caches per (traverse, model_tag).

a=1.0 reproduces model A alone and a=0.0 model B alone, so the sweep contains
its own controls -- if those two columns do not match the numbers above, the
harness is wrong and nothing else in the table means anything.

Reports [all] only: the ensemble baseline we compare against does not remove
stationary frames, so [moving] is not comparable.

Run:
    python scripts/evaluate_brisbane_fusion.py datasets=brisbane \
      datasets.root=.../Brisbane_6_set data.modality=histogram datasets.dt=1.0 \
      '+fusion.weights=[/path/megaloc_fix/best_phase1_histogram.pth,
                        /path/baseline/best_phase1_histogram.pth]' \
      '+fusion.tags=[megaloc_fix,baseline]'
"""

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import evaluate_brisbane as eb
from sequence_matching import apply_sequence_matching


def descriptors_for(cfg, device, path, tag):
    """{traverse: (N, D) L2-normalised}, {traverse: (N, 2) xy}.

    `tag` is passed through to process_traverse as model_tag so each model
    gets its own descriptor cache -- without it the second model would read
    the first model's cached descriptors and every fused row would be a lie.
    """
    OmegaConf.update(cfg, "phase1_weights", str(path), merge=False)
    OmegaConf.update(cfg, "phase2_weights", None, merge=False)
    model, _ = eb.build_model(cfg, device)

    desc, xy = {}, {}
    for name in cfg.datasets.traverses:
        print(f"  [{tag}] {name}")
        desc[name], xy[name] = eb.process_traverse(name, cfg, model, device, tag)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return desc, xy


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    fcfg = OmegaConf.select(cfg, "fusion")
    if fcfg is None or not OmegaConf.select(fcfg, "weights"):
        raise SystemExit(__doc__)

    weights = [str(w) for w in fcfg.weights]
    tags = [str(t) for t in (OmegaConf.select(fcfg, "tags") or
                             [f"m{i}" for i in range(len(weights))])]
    alphas = [float(a) for a in (OmegaConf.select(fcfg, "alphas") or
                                 [0.0, 0.25, 0.5, 0.75, 1.0])]
    if len(weights) != 2 or len(tags) != 2:
        raise SystemExit("fusion.weights and fusion.tags must each have 2 entries")
    for p in weights:
        if not Path(p).is_file():
            raise SystemExit(f"checkpoint not found: {p}")

    device = torch.device(cfg.device)
    threshold = float(cfg.datasets.gt_threshold_m)
    ref_name = cfg.datasets.reference
    queries = [n for n in cfg.datasets.traverses if n != ref_name]
    lengths = list(cfg.datasets.seq_lengths)

    print(f"A = {tags[0]}: {weights[0]}")
    print(f"B = {tags[1]}: {weights[1]}")
    desc_a, xy = descriptors_for(cfg, device, weights[0], tags[0])
    desc_b, _ = descriptors_for(cfg, device, weights[1], tags[1])

    for name in cfg.datasets.traverses:
        if desc_a[name].shape[0] != desc_b[name].shape[0]:
            raise SystemExit(
                f"{name}: {tags[0]} has {desc_a[name].shape[0]} descriptors but "
                f"{tags[1]} has {desc_b[name].shape[0]}. The two caches are not "
                f"frame-aligned and every fused row would be meaningless.")

    rows = []
    print(f"\n{'='*78}\nalpha = weight on {tags[0]}  "
          f"(1.0 = {tags[0]} alone, 0.0 = {tags[1]} alone)")
    for a in alphas:
        wa, wb = np.sqrt(a), np.sqrt(1.0 - a)
        fused = {n: torch.cat([wa * desc_a[n], wb * desc_b[n]], dim=1)
                 for n in cfg.datasets.traverses}

        r_desc = fused[ref_name].to(device)
        r_xy = xy[ref_name]
        per_cond = {}
        for q_name in queries:
            dist = -torch.cdist(fused[q_name].to(device), r_desc)
            for length in lengths:
                dist_L = dist if length == 1 else apply_sequence_matching(dist, length)
                preds = dist_L.argmax(dim=1).cpu()
                r1, r1_valid, n_valid, n_q = eb.recall_at_1(
                    preds, xy[q_name], r_xy, threshold)
                cond = "day-day" if q_name in eb.DAY_DAY else "day-night"
                per_cond.setdefault((cond, length), []).append(r1)
                rows.append(dict(alpha=a, query=q_name, condition=cond,
                                 seq_len=length, recall_at_1=r1,
                                 recall_at_1_validonly=r1_valid,
                                 n_queries=n_q, n_queries_with_gt=n_valid))

        def mean(cond, ls):
            v = [x for L in ls for x in per_cond.get((cond, L), [])]
            return float(np.mean(v)) if v else float("nan")

        night = {L: mean("day-night", [L]) for L in lengths}
        print(f"\n  alpha={a:.2f}   day-day {mean('day-day', [L for L in lengths if L > 1]):6.2f}"
              f"   day-night {mean('day-night', [L for L in lengths if L > 1]):6.2f}")
        print("            night  " + "  ".join(
            f"L={L:<2d} {night[L]:6.2f}" for L in lengths))

    out = Path(str(cfg.datasets.results_csv).format(
        modality=cfg.data.modality, dt=cfg.datasets.dt))
    out = out.with_name(out.stem + "_fusion.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {out}")
    print("\nDecide on the night column: if some intermediate alpha beats BOTH "
          "endpoints\non the day-night mean while holding L=1 near the "
          f"{tags[0]} endpoint, the two\nteachers are complementary and a "
          "multi-teacher run is justified. If every\nintermediate row sits "
          "between the endpoints, they are redundant -- do not\nspend the GPU "
          "time.")


if __name__ == "__main__":
    main()
