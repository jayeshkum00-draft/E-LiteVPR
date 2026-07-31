"""Precompute teacher_global = GeM(p=3) over the cached DINOv3 patches.

WHY
---
Under structural-only training the patch grid is never used as a loss target;
train.py only ever reduces it to one vector:

    teacher_gem = GeM(p=3.0)            # train.py:341-343, requires_grad=False
    teacher_global = teacher_gem(teacher_patches)     # train.py:92

teacher_gem is FROZEN, so that vector is a constant per frame and recomputing
it every epoch is pure waste. Worse, it forces the dataloader to read
576 x 1024 fp16 = 1.18 MB PER FRAME. At batch_size=128 that is 151 MB per step
off network-backed storage, which is what dominates throughput on Kaggle.

Caching the pooled vector instead is 1024 fp16 = 2 KB per frame: a 576x
reduction, and bit-equivalent because the pooling is deterministic and frozen.

    <out_dir>/<seq>/gem.npy    float16, (N, 1024), row-aligned to
                               <features_dir>/<seq>/frames.txt

Same row contract as cache_megaloc.py, and trivially satisfied here: rows are
copied in patches.npy order, which IS frames.txt order.

No GPU and no model needed -- it is a pooling over an existing cache.

    python scripts/cache_teacher_gem.py \
        datasets.root_dir=$DSEC_ROOT datasets.output_dir=$DSEC_FEATS \
        datasets@datasets_extra=m3ed \
        datasets_extra.root_dir=$M3ED_ROOT datasets_extra.output_dir=$M3ED_FEATS \
        +gem.out_dir=$ROOT/feats/gem hydra.run.dir=$ROOT/hy/cache_gem

Then train with +training.teacher=dinov3_gem +training.teacher_dir=$ROOT/feats/gem,
which is mathematically the same objective as +training.teacher=dinov3 but
reads 576x less per step.
"""

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_EPS = 1e-6          # model.GeM default
_P = 3.0             # train.py:341  GeM(p=3.0)


def gem_rows(patches):
    """(n, N, D) fp16 -> (n, D) float32, identical to model.GeM(p=3)."""
    x = torch.from_numpy(np.ascontiguousarray(patches)).float()
    return x.clamp(min=_EPS).pow(_P).mean(dim=1).pow(1.0 / _P)


def process_sequence(seq, features_dir, out_dir, chunk):
    frames_file = features_dir / seq / "frames.txt"
    n_rows = sum(1 for ln in frames_file.read_text().splitlines() if ln.strip())

    out_path = out_dir / seq / "gem.npy"
    if out_path.is_file() and np.load(out_path, mmap_mode="r").shape[0] == n_rows:
        return n_rows, "skipped"

    patches = np.load(features_dir / seq / "patches.npy", mmap_mode="r")
    if patches.shape[0] != n_rows:
        raise RuntimeError(
            f"{seq}: patches.npy has {patches.shape[0]} rows but frames.txt has "
            f"{n_rows}; the DINOv3 cache for this sequence is inconsistent.")

    out = np.empty((n_rows, patches.shape[-1]), dtype=np.float16)
    for i in tqdm(range(0, n_rows, chunk), desc=f"  {seq}", leave=False):
        out[i:i + chunk] = gem_rows(patches[i:i + chunk]).numpy().astype(np.float16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    return n_rows, "written"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    gcfg = OmegaConf.select(cfg, "gem") or OmegaConf.create({})
    out_dir = OmegaConf.select(gcfg, "out_dir")
    if not out_dir:
        raise SystemExit("Pass +gem.out_dir=<writable dir>")
    out_dir = Path(str(out_dir))
    chunk = int(OmegaConf.select(gcfg, "chunk") or 128)

    sources = [cfg.datasets]
    extra = OmegaConf.select(cfg, "datasets_extra")
    if extra is not None:
        sources.append(extra)

    total = 0
    for dcfg in sources:
        features_dir = Path(str(dcfg.output_dir))
        seqs = sorted(p.parent.name for p in features_dir.glob("*/frames.txt"))
        if not seqs:
            raise SystemExit(f"No */frames.txt under {features_dir}")
        excluded = set(OmegaConf.select(dcfg, "to_exclude") or [])
        if excluded:
            dropped = [s for s in seqs if s in excluded]
            seqs = [s for s in seqs if s not in excluded]
            print(f"  to_exclude: skipping {dropped or 'nothing'}")
        print(f"\n{features_dir}: {len(seqs)} sequences")
        for seq in seqs:
            n, how = process_sequence(seq, features_dir, out_dir, chunk)
            print(f"  {seq:<40} {n:>7} rows  {how}")
            total += n

    print(f"\nDone: {total} descriptors -> {out_dir}")


if __name__ == "__main__":
    main()
