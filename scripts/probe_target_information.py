"""How much information does the relational target actually carry?

CLAIM UNDER TEST
----------------
The MegaLoc run collapsed to train struct 0.0007 while val struct sat at 0.24
-- a 300x gap in the SAME loss. My explanation was that the two loaders batch
differently (train: WeightedRandomSampler over 14k frames spanning many
sequences; val: shuffle=False, i.e. 32 CONSECUTIVE frames), so random batches
contain no two views of the same place, the teacher similarity matrix is
essentially the identity, and softmax at tau=0.05 turns it into a near-one-hot
target the student satisfies by doing nothing but staying spread out.

That was inference from two numbers in a log. It is also a property of the
CACHED DESCRIPTORS ALONE -- no student, no forward pass, no training:

    T = softmax( normalize(d) @ normalize(d).T / tau )

so it can be measured directly. This script does that, and scores three
predictions made BEFORE the measurement so the result is not interpreted after
the fact:

  P1  random batches, B=32, tau=0.05, MegaLoc -> mean diagonal mass > 0.95
      (fails => "the target is essentially the identity" is wrong and the ~0
       train loss needs another explanation)
  P2  sequential batches -> materially lower diagonal mass than random
      (fails => batching is not the mechanism)
  P3  DINOv3 under random batching -> LOWER diagonal mass than MegaLoc
      (this is the counterintuitive half: a better VPR teacher makes different
       places more orthogonal, so the random-batch target carries LESS signal.
       fails => "MegaLoc made the target emptier" collapses)

It also sweeps the design space offline so the batch geometry and tau are
CHOSEN rather than guessed: batch size, temperature, and block structure
(K sequences x M contiguous frames). The point is one measurement then one
training run, not a series of training runs.

HONEST LIMIT: this measures the information content of the TARGET. It does not
show the student can exploit it -- only training shows that.

Metrics, per batch, averaged over trials:
  diag        mean_i T[i,i]  -- probability mass the target puts on "match
              yourself". 1.0 means the loss is a pure uniformity prior.
  informative 1 - diag. The share of the target that says anything about how
              frames relate to EACH OTHER. This is the quantity the whole
              argument is about.
  eff_nbrs    exp(entropy of the off-diagonal, renormalised): how many other
              frames a row meaningfully points at. ~1 = one clear neighbour,
              ~B = uniform mush.
  off_cos     mean off-diagonal cosine before temperature, for reference.

Run:
    uv run python scripts/probe_target_information.py \
        --megaloc /kaggle/working/megaloc_features \
        --dinov3  /kaggle/input/notebooks/jhag18/dsec-all/teacher_features
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

_EPS = 1e-6


def gem(patches, p=3.0):
    """model.py GeM(p=3) over the patch axis -- what train.py's frozen
    teacher_gem does to build teacher_global from the DINOv3 cache."""
    x = patches.clamp(min=_EPS).pow(p)
    return x.mean(dim=1).pow(1.0 / p)


def load_sequences(root, kind, per_seq, rng, limit_seqs=None):
    """{seq: (n, D) float32} keeping rows CONTIGUOUS so sequential/block
    sampling reflects real temporal adjacency."""
    root = Path(root)
    fname = "megaloc.npy" if kind == "megaloc" else "patches.npy"
    paths = sorted(root.glob(f"*/{fname}"))
    if not paths:
        raise SystemExit(f"no */{fname} under {root}")
    if limit_seqs:
        paths = paths[:limit_seqs]

    out = {}
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]
        take = min(per_seq, n)
        start = int(rng.integers(0, max(1, n - take + 1)))
        block = np.ascontiguousarray(arr[start:start + take])
        t = torch.from_numpy(block).float()
        if kind == "dinov3":
            t = gem(t)                      # (take, 576, 1024) -> (take, 1024)
        out[path.parent.name] = t
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
    off_sum = off.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    q = off / off_sum
    ent = -(q * (q + 1e-12).log()).sum(dim=-1)
    return (diag, 1.0 - diag, ent.exp().mean().item(),
            sim.masked_fill(eye, 0.0).sum().item() / (B * (B - 1)))


def sample(seqs, scheme, B, rng, K=None, M=None):
    """random | sequential | block(K x M). sequential == block(1, B)."""
    names = list(seqs)
    if scheme == "random":
        # pooled uniform draw across every sequence -- what
        # WeightedRandomSampler(replacement=True) produces once day/night
        # weights are applied across a corpus with no repeat traverses
        pool = torch.cat([seqs[n] for n in names], dim=0)
        idx = rng.integers(0, pool.shape[0], size=B)
        return pool[torch.from_numpy(idx)]

    if scheme == "sequential":
        K, M = 1, B
    parts = []
    for _ in range(K):
        n = names[int(rng.integers(0, len(names)))]
        a = seqs[n]
        if a.shape[0] < M:
            return None
        s = int(rng.integers(0, a.shape[0] - M + 1))
        parts.append(a[s:s + M])
    return torch.cat(parts, dim=0)


def run(seqs, scheme, B, tau, trials, rng, K=None, M=None):
    acc = []
    for _ in range(trials):
        d = sample(seqs, scheme, B, rng, K, M)
        if d is None or d.shape[0] < 2:
            continue
        acc.append(batch_stats(d, tau))
    if not acc:
        return None
    a = np.array(acc)
    return a.mean(axis=0), a.std(axis=0)


def header(title):
    print(f"\n{title}\n  {'setting':<26} {'diag':>7} {'informative':>12} "
          f"{'eff_nbrs':>9} {'off_cos':>8}")


def line(label, res):
    if res is None:
        print(f"  {label:<26} {'n/a':>7}")
        return None
    m, s = res
    print(f"  {label:<26} {m[0]:>7.4f} {m[1]:>12.4f} {m[2]:>9.2f} {m[3]:>8.4f}"
          f"   (+/-{s[1]:.4f})")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--megaloc", required=True)
    ap.add_argument("--dinov3", default=None,
                    help="teacher features_dir with */patches.npy (for P3)")
    ap.add_argument("--per-seq", type=int, default=400,
                    help="contiguous rows loaded per sequence")
    ap.add_argument("--dinov3-per-seq", type=int, default=120,
                    help="lower: patches.npy is ~1.2 MB per frame")
    ap.add_argument("--dinov3-seqs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"loading MegaLoc from {args.megaloc}")
    mega = load_sequences(args.megaloc, "megaloc", args.per_seq, rng)
    dims = {tuple(v.shape[1:]) for v in mega.values()}
    print(f"  {len(mega)} sequences, {sum(v.shape[0] for v in mega.values())} "
          f"frames, dim {dims}")

    T = args.trials

    header("MegaLoc target, tau=0.05, B=32  [the configuration that ran]")
    p_rand = line("random (train loader)", run(mega, "random", 32, 0.05, T, rng))
    p_seq = line("sequential (val loader)", run(mega, "sequential", 32, 0.05, T, rng))

    header("MegaLoc: batch size sweep, random sampling, tau=0.05")
    for B in (32, 64, 128, 256):
        line(f"random B={B}", run(mega, "random", B, 0.05, T, rng))

    header("MegaLoc: temperature sweep, random sampling, B=32")
    for tau in (0.05, 0.1, 0.2, 0.5):
        line(f"random tau={tau}", run(mega, "random", 32, tau, T, rng))

    header("MegaLoc: block structure K seqs x M contiguous, tau=0.05")
    for B in (32, 128):
        for M in (2, 4, 8, 16, 32):
            if B % M:
                continue
            line(f"B={B} K={B // M} x M={M}",
                 run(mega, "block", B, 0.05, T, rng, K=B // M, M=M))

    p_dino = None
    if args.dinov3:
        print(f"\nloading DINOv3 from {args.dinov3} (GeM p=3 over patches)")
        dino = load_sequences(args.dinov3, "dinov3", args.dinov3_per_seq, rng,
                              limit_seqs=args.dinov3_seqs)
        print(f"  {len(dino)} sequences, "
              f"{sum(v.shape[0] for v in dino.values())} frames")
        header("DINOv3 target, tau=0.05, B=32")
        p_dino = line("random (train loader)", run(dino, "random", 32, 0.05, T, rng))
        line("sequential (val loader)", run(dino, "sequential", 32, 0.05, T, rng))

    print("\n" + "=" * 74)
    print("PREDICTIONS (stated before measuring)")

    def score(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if p_rand is not None:
        score("P1 MegaLoc random diag > 0.95", p_rand[0] > 0.95,
              f"measured {p_rand[0]:.4f}")
    if p_rand is not None and p_seq is not None:
        score("P2 sequential materially lower", p_seq[0] < p_rand[0] - 0.05,
              f"random {p_rand[0]:.4f} vs sequential {p_seq[0]:.4f} "
              f"(informative {p_rand[1]:.4f} -> {p_seq[1]:.4f}, "
              f"{p_seq[1] / max(p_rand[1], 1e-12):.1f}x)")
    if p_dino is not None and p_rand is not None:
        score("P3 DINOv3 random diag < MegaLoc", p_dino[0] < p_rand[0],
              f"DINOv3 {p_dino[0]:.4f} vs MegaLoc {p_rand[0]:.4f}")
    else:
        print("  [ -- ] P3 not evaluated (pass --dinov3)")

    print("\nIf P1/P2 fail, the batching explanation for the 300x train/val gap")
    print("is wrong and should be discarded rather than patched.")


if __name__ == "__main__":
    main()
