"""Is the M3ED-night teacher degenerate, and did it change the training target?

CLAIM UNDER TEST
----------------
Adding M3ED to the DSEC corpus took Brisbane day-night from 37.27 to 11.60
(bs=64) / 4.08 (bs=32) while day-day barely moved (82.33 -> 81.72). Day intact,
night destroyed, on the stock `train.py` with no other change.

The proposed mechanism is a property of the CACHED TEACHER FEATURES ALONE --
no student, no forward pass, no training:

  `build_day_night_sampler` (train.py:106) pins night at 50% of every batch by
  reweighting, so M3ED's 5 `car_urban_night_*` sequences did not ADD night, they
  DILUTED DSEC night inside a fixed budget. And `compute_structural_loss` builds
  its target from the teacher descriptors of exactly the B frames in the batch.
  If DINOv3-Large GeM on M3ED's dark car-camera RGB is near-constant across
  frames, the target inside a night-heavy batch says "every one of these is the
  same place", and the KL trains the student to collapse night to a point.

PREDICTIONS (stated before measuring; scored at the end)
  P1  within-sequence mean cosine on M3ED `car_urban_night_*` is materially
      higher than on DSEC night (>= +0.05).
      fails => M3ED night RGB is not degenerate; the collapse is elsewhere.
  P2  simulating the REAL sampler (50% night / 50% day, night pooled across
      sources in proportion to true frame counts), the DSEC+M3ED target has
      HIGHER off_cos and FEWER eff_nbrs than the DSEC-only target.
      fails => adding M3ED did not measurably change the training target, and
      the corpus-composition explanation should be discarded, not patched.
  P3  the damage is night-side: restricting the simulation to the DAY half
      shows no material change between the two corpora.
      fails => M3ED hurts day too and the story is not night-specific.

HONEST LIMIT: this measures the TARGET the loss is computed against. A worse
target is a necessary, not sufficient, condition for the observed collapse.

Reads the same caches training reads (`<features_dir>/<seq>/patches.npy`) and
applies the same GeM(p=3) the frozen teacher applies. Sequence membership and
night lists come from the dataset YAMLs so this cannot drift from the configs.

Mirrors stock `train.py`, which does NOT honour `to_exclude` -- pass
--apply-exclude to see what the excluded run would have seen instead.

Run:
    uv run python scripts/probe_teacher_degeneracy.py \
        --dsec-features /kaggle/input/.../teacher_features \
        --m3ed-features /kaggle/working/m3ed_features
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_EPS = 1e-6
_REPO = Path(__file__).resolve().parent.parent


def gem(patches, p=3.0):
    """model.py GeM(p=3) over the patch axis -- what train.py's frozen
    teacher_gem does to build teacher_global from the DINOv3 cache."""
    x = patches.clamp(min=_EPS).pow(p)
    return x.mean(dim=1).pow(1.0 / p)


def load_source(root, keep, per_seq, rng):
    """{seq: (desc (n,D) float32, true_n int)}.

    Rows stay CONTIGUOUS so within-sequence temporal structure survives.
    `true_n` is the FULL sequence length from the .npy header, kept separate so
    the sampler simulation can weight sources by their real frame counts even
    though only `per_seq` rows are loaded.
    """
    root = Path(root)
    paths = sorted(root.glob("*/patches.npy"))
    if not paths:
        raise SystemExit(f"no */patches.npy under {root}")

    out = {}
    for path in paths:
        name = path.parent.name
        if keep is not None and name not in keep:
            continue
        arr = np.load(path, mmap_mode="r")
        true_n = int(arr.shape[0])
        take = min(per_seq, true_n)
        start = int(rng.integers(0, max(1, true_n - take + 1)))
        block = np.ascontiguousarray(arr[start:start + take])
        out[name] = (gem(torch.from_numpy(block).float()), true_n)
        del block, arr
    return out


def batch_stats(desc, tau):
    """desc (B, D) -> (diag, informative, eff_nbrs, off_cos) for the target
    softmax(normalize(d) @ normalize(d).T / tau) the KL is computed against."""
    d = torch.nn.functional.normalize(desc, p=2, dim=-1)
    sim = d @ d.T
    B = sim.shape[0]
    eye = torch.eye(B, dtype=torch.bool)

    T = torch.softmax(sim / tau, dim=-1)
    diag = T[eye].mean().item()

    off = T.masked_fill(eye, 0.0)
    q = off / off.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    ent = -(q * (q + 1e-12).log()).sum(dim=-1)
    return (diag, 1.0 - diag, ent.exp().mean().item(),
            sim.masked_fill(eye, 0.0).sum().item() / (B * (B - 1)))


def within_seq_cos(desc):
    """Mean off-diagonal cosine over a whole sequence. ~1.0 = the teacher
    cannot tell any two frames of this sequence apart."""
    d = torch.nn.functional.normalize(desc, p=2, dim=-1)
    sim = d @ d.T
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    eye = torch.eye(n, dtype=torch.bool)
    return sim.masked_fill(eye, 0.0).sum().item() / (n * (n - 1))


def draw_pool(entries, k, rng):
    """Draw k rows from {seq: (desc, true_n)}, choosing the source sequence in
    proportion to its TRUE frame count -- what WeightedRandomSampler does once
    every frame inside a condition carries equal weight."""
    names = list(entries)
    if not names:
        return None
    counts = np.array([entries[n][1] for n in names], dtype=np.float64)
    probs = counts / counts.sum()
    picks = rng.choice(len(names), size=k, p=probs)
    rows = []
    for i in picks:
        d = entries[names[i]][0]
        rows.append(d[int(rng.integers(0, d.shape[0]))])
    return torch.stack(rows)


def simulate(day, night, B, tau, trials, rng, half=None):
    """The real training batch: B/2 night + B/2 day (`half` restricts to one
    condition to localise any change)."""
    acc = []
    for _ in range(trials):
        if half == "day":
            d = draw_pool(day, B, rng)
        elif half == "night":
            d = draw_pool(night, B, rng)
        else:
            a, b = draw_pool(night, B // 2, rng), draw_pool(day, B - B // 2, rng)
            d = None if a is None or b is None else torch.cat([a, b], dim=0)
        if d is None or d.shape[0] < 2:
            continue
        acc.append(batch_stats(d, tau))
    if not acc:
        return None
    return np.array(acc).mean(axis=0)


def header(title):
    print(f"\n{title}\n  {'setting':<34} {'diag':>7} {'informative':>12} "
          f"{'eff_nbrs':>9} {'off_cos':>8}")


def line(label, m):
    if m is None:
        print(f"  {label:<34} {'n/a':>7}")
        return None
    print(f"  {label:<34} {m[0]:>7.4f} {m[1]:>12.4f} {m[2]:>9.2f} {m[3]:>8.4f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsec-features", required=True)
    ap.add_argument("--m3ed-features", required=True)
    ap.add_argument("--per-seq", type=int, default=100,
                    help="contiguous rows loaded per sequence "
                         "(patches.npy is ~1.2 MB per frame)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply-exclude", action="store_true",
                    help="honour m3ed.yaml to_exclude; stock train.py does NOT, "
                         "so the default reflects the run that actually ran")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dsec_cfg = OmegaConf.load(_REPO / "configs/datasets/dsec.yaml")
    m3ed_cfg = OmegaConf.load(_REPO / "configs/datasets/m3ed.yaml")

    dsec_train = set(dsec_cfg.train_seq_list)
    dsec_night = set(dsec_cfg.night_sequences) & dsec_train
    m3ed_night = set(m3ed_cfg.night_sequences)
    excluded = set(m3ed_cfg.get("to_exclude") or []) if args.apply_exclude else set()

    print(f"loading DSEC train split from {args.dsec_features}")
    dsec = load_source(args.dsec_features, dsec_train, args.per_seq, rng)
    print(f"  {len(dsec)} sequences, "
          f"{sum(v[1] for v in dsec.values())} frames (true counts)")

    print(f"loading M3ED from {args.m3ed_features}")
    m3ed = load_source(args.m3ed_features, None, args.per_seq, rng)
    for name in sorted(excluded & set(m3ed)):
        print(f"  excluding {name} (--apply-exclude)")
        m3ed.pop(name)
    print(f"  {len(m3ed)} sequences, "
          f"{sum(v[1] for v in m3ed.values())} frames (true counts)")

    unknown = m3ed_night - set(m3ed)
    if unknown:
        print(f"  NOTE: night_sequences not found in cache: {sorted(unknown)}")

    # ---- per-sequence degeneracy -------------------------------------------
    rows = []
    for name, (d, n) in dsec.items():
        rows.append(("DSEC", "night" if name in dsec_night else "day",
                     name, n, within_seq_cos(d)))
    for name, (d, n) in m3ed.items():
        rows.append(("M3ED", "night" if name in m3ed_night else "day",
                     name, n, within_seq_cos(d)))

    print(f"\nWithin-sequence mean cosine (teacher GeM, {args.per_seq} "
          f"contiguous frames)\n  {'source':<6} {'cond':<6} {'sequence':<40} "
          f"{'frames':>8} {'cos':>7}")
    for src, cond, name, n, c in sorted(rows, key=lambda r: -r[4]):
        print(f"  {src:<6} {cond:<6} {name:<40} {n:>8} {c:>7.4f}")

    def mean_cos(src, cond):
        v = [r[4] for r in rows if r[0] == src and r[1] == cond]
        return float(np.mean(v)) if v else float("nan")

    dsec_night_cos, m3ed_night_cos = mean_cos("DSEC", "night"), mean_cos("M3ED", "night")
    dsec_day_cos, m3ed_day_cos = mean_cos("DSEC", "day"), mean_cos("M3ED", "day")
    print(f"\n  group means:  DSEC day {dsec_day_cos:.4f}   "
          f"DSEC night {dsec_night_cos:.4f}   "
          f"M3ED day {m3ed_day_cos:.4f}   M3ED night {m3ed_night_cos:.4f}")

    # ---- the target the sampler actually builds ----------------------------
    dsec_day_pool = {k: v for k, v in dsec.items() if k not in dsec_night}
    dsec_night_pool = {k: v for k, v in dsec.items() if k in dsec_night}
    both_day = {**dsec_day_pool,
                **{k: v for k, v in m3ed.items() if k not in m3ed_night}}
    both_night = {**dsec_night_pool,
                  **{k: v for k, v in m3ed.items() if k in m3ed_night}}

    n_dsec_n = sum(v[1] for v in dsec_night_pool.values())
    n_m3ed_n = sum(v[1] for k, v in m3ed.items() if k in m3ed_night)
    share = n_m3ed_n / max(n_dsec_n + n_m3ed_n, 1)
    print(f"\n  night budget: DSEC {n_dsec_n} frames, M3ED {n_m3ed_n} frames "
          f"-> M3ED takes {share:.1%} of the fixed 50% night half")

    B, tau, T = args.batch_size, args.tau, args.trials
    header(f"Training target, B={B}, tau={tau}  [50% night / 50% day, as sampled]")
    a_full = line("DSEC only", simulate(dsec_day_pool, dsec_night_pool, B, tau, T, rng))
    b_full = line("DSEC + M3ED", simulate(both_day, both_night, B, tau, T, rng))

    header(f"Localised: night half only, B={B}")
    a_night = line("DSEC only", simulate(None, dsec_night_pool, B, tau, T, rng, half="night"))
    b_night = line("DSEC + M3ED", simulate(None, both_night, B, tau, T, rng, half="night"))

    header(f"Localised: day half only, B={B}")
    a_day = line("DSEC only", simulate(dsec_day_pool, None, B, tau, T, rng, half="day"))
    b_day = line("DSEC + M3ED", simulate(both_day, None, B, tau, T, rng, half="day"))

    # ---- scoring ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PREDICTIONS (stated before measuring)")

    def score(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    score("P1 M3ED night more degenerate than DSEC night",
          m3ed_night_cos >= dsec_night_cos + 0.05,
          f"M3ED night {m3ed_night_cos:.4f} vs DSEC night {dsec_night_cos:.4f} "
          f"(delta {m3ed_night_cos - dsec_night_cos:+.4f})")

    if a_full is not None and b_full is not None:
        score("P2 combined target is worse (higher off_cos, fewer eff_nbrs)",
              b_full[3] > a_full[3] and b_full[2] < a_full[2],
              f"off_cos {a_full[3]:.4f} -> {b_full[3]:.4f}, "
              f"eff_nbrs {a_full[2]:.2f} -> {b_full[2]:.2f}, "
              f"informative {a_full[1]:.4f} -> {b_full[1]:.4f}")

    if a_day is not None and b_day is not None and a_night is not None:
        d_day = abs(b_day[3] - a_day[3])
        d_night = abs(b_night[3] - a_night[3])
        score("P3 damage is night-side, not day-side", d_night > d_day,
              f"|delta off_cos| night {d_night:.4f} vs day {d_day:.4f}")

    print("\nIf P2 FAILS the training target did not measurably change and the")
    print("corpus-composition explanation must be discarded, not patched --")
    print("look at the student/optimiser side instead.")


if __name__ == "__main__":
    main()
