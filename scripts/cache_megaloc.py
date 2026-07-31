"""Cache MegaLoc descriptors for the RGB side of a preprocessed corpus.

WHY
---
The teacher probe showed DINOv3's descriptor is organised by ILLUMINATION:
on Brisbane event frames its R@1 at L=1 falls monotonically with illumination
distance from the reference (sunset 47.13 -> sunrise 22.12 -> morning 15.12
-> daytime 14.61 -> night 3.14). The structural loss copies that geometry into
the student, which is why every student so far gains 2-4x on daylight pairs
and LOSES on night (0.57x vs its own teacher).

MegaLoc is VPR-trained: it was optimised on labelled same-place/different-
condition pairs precisely so illumination is NOT a descriptor axis. Distilling
its pairwise geometry instead of DINOv3's is the direct test of that diagnosis,
and it needs no GPS -- unlike a place-target loss, which is capped by the three
M3ED route pairs that have usable pose ground truth.

WHAT THIS WRITES
----------------
    <out_dir>/<seq>/megaloc.npy    float16, (N, 8448), row-aligned to the
                                   EXISTING <features_dir>/<seq>/frames.txt

Row alignment to frames.txt is the whole contract: dataset.py maps a pair to a
cache row through `_row_index` built from that file, so a misaligned cache
silently trains on the wrong targets and every number afterwards is garbage.
This script therefore derives the RGB path FROM frames.txt (never from a
directory listing, whose order is not guaranteed to match) and asserts the row
count at the end.

`out_dir` is separate from `features_dir` on purpose: on Kaggle the teacher
feature cache lives under /kaggle/input, which is read-only.

MODEL / PREPROCESSING
---------------------
Mirrors external/ensemble_event_vpr_bench exactly, so the descriptors are the
same ones behind the paper's 44.16 Brisbane D-N number:
  * vpr_models/__init__.py:  torch.hub.load("gmberton/MegaLoc",
                                            "get_trained_model")
  * parse.py:                megaloc sets backbone/dim only and NEVER sets
                             image_size, so no Resize is applied -- MegaLoc
                             consumes images at native resolution. (The
                             224x224 in that repo belongs to `dinomix`.)
  * test_dataset.py:132-136: ToTensor() then ImageNet Normalize.

One deviation, made explicit: MegaLoc is DINOv2-based and wants H, W to be
multiples of 14. The benchmark feeds it real photos whose sizes happen to work;
our preprocessed RGB is 384x384 and 384 is NOT a multiple of 14. So images are
resized to the nearest multiple (384 -> 378), which is exactly what that repo's
ResizingWrapper("dino_v2_resize") does for every other DINOv2 model. Override
with +megaloc.size=N if you want a different one.

Run:
    python scripts/cache_megaloc.py \
        datasets.root_dir=/kaggle/input/datasets/jhag18/dsec-all-clean \
        datasets.output_dir=/kaggle/input/notebooks/jhag18/dsec-all/teacher_features \
        +megaloc.out_dir=/kaggle/working/megaloc_features \
        hydra.run.dir=/kaggle/working/hy_cache

Needs HYDRA_MAIN_MODULE unset (this file owns its own @hydra.main, unlike the
train_*/evaluate_* wrappers which import someone else's).
"""

from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_EXPECTED_DIM = 8448          # parse.py: megaloc -> descriptors_dimension 8448


def load_megaloc(device):
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                           trust_repo=True)
    return model.eval().to(device)


def dataset_sources(cfg):
    """The same one-or-two sources train.py trains over."""
    sources = [cfg.datasets]
    extra = OmegaConf.select(cfg, "datasets_extra")
    if extra is not None:
        sources.append(extra)
    return sources


def load_rgb_batch(paths, size, device):
    """Preprocessed RGB .npy -> normalised float tensor (B, 3, size, size).

    Stored CHW uint8 by the preprocessors; HWC is accepted too so this does not
    silently transpose a differently-written cache.
    """
    imgs = []
    for p in paths:
        a = np.load(p)
        if a.ndim != 3:
            raise ValueError(f"{p}: expected a 3-D RGB array, got {a.shape}")
        if a.shape[0] != 3:
            if a.shape[-1] != 3:
                raise ValueError(f"{p}: no 3-channel axis in {a.shape}")
            a = np.transpose(a, (2, 0, 1))
        imgs.append(torch.from_numpy(np.ascontiguousarray(a)))
    x = torch.stack(imgs).float() / 255.0            # ToTensor()
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD         # Normalize()
    if x.shape[-2:] != (size, size):
        x = TF.resize(x, [size, size], antialias=True)
    return x.to(device, non_blocking=True)


def process_sequence(seq, root, features_dir, out_dir, model, cfg, device):
    """Returns (n_rows, 'written'|'skipped')."""
    frames_file = features_dir / seq / "frames.txt"
    keys = [ln.strip() for ln in frames_file.read_text().splitlines() if ln.strip()]

    out_path = out_dir / seq / "megaloc.npy"
    if out_path.is_file():
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[0] == len(keys):
            return len(keys), "skipped"
        print(f"  {seq}: existing cache has {existing.shape[0]} rows but "
              f"frames.txt has {len(keys)} -- recomputing")

    # frames.txt stores '<seq>/rgb/<name>.npy', i.e. the path relative to root
    paths = [root / k for k in keys]
    missing = [p for p in paths[:64] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{seq}: {missing[0]} from frames.txt not found under root={root}. "
            f"datasets.root_dir must be the SAME corpus the teacher features "
            f"were extracted from.")

    size = int(cfg.size)
    bs = int(cfg.batch_size)
    out = np.empty((len(keys), _EXPECTED_DIM), dtype=np.float16)

    for i in tqdm(range(0, len(keys), bs), desc=f"  {seq}", leave=False):
        chunk = paths[i:i + bs]
        with torch.no_grad(), torch.autocast("cuda", enabled=bool(cfg.amp)):
            d = model(load_rgb_batch(chunk, size, device))
        if d.shape[-1] != _EXPECTED_DIM:
            raise ValueError(
                f"MegaLoc returned dim {d.shape[-1]}, expected {_EXPECTED_DIM}. "
                f"The hub checkpoint changed; update _EXPECTED_DIM only after "
                f"checking it against parse.py in the benchmark repo.")
        out[i:i + len(chunk)] = d.float().cpu().numpy().astype(np.float16)

    assert out.shape[0] == len(keys), "row count drifted from frames.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    return len(keys), "written"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mcfg = OmegaConf.select(cfg, "megaloc") or OmegaConf.create({})
    out_dir = OmegaConf.select(mcfg, "out_dir")
    if not out_dir:
        raise SystemExit(
            "Pass +megaloc.out_dir=<writable dir>. It must NOT be the teacher "
            "features_dir: on Kaggle that lives under /kaggle/input and is "
            "read-only.")
    mcfg = OmegaConf.merge(
        OmegaConf.create({"size": 378, "batch_size": 32, "amp": True}), mcfg)
    out_dir = Path(str(out_dir))

    device = torch.device(cfg.device)
    model = load_megaloc(device)
    print(f"MegaLoc loaded; input {mcfg.size}x{mcfg.size} "
          f"(nearest multiple of 14), descriptor {_EXPECTED_DIM}-d")

    total = 0
    for dcfg in dataset_sources(cfg):
        root = Path(str(dcfg.root_dir))
        features_dir = Path(str(dcfg.output_dir))
        seqs = sorted(p.parent.name for p in features_dir.glob("*/frames.txt"))
        if not seqs:
            raise SystemExit(f"No */frames.txt under {features_dir}")

        # `to_exclude` is honoured here as well as in train_ext.py: if the
        # DINOv3 pass was run over the unfiltered pairs.txt, the excluded
        # sequence already has a frames.txt and would otherwise be cached.
        excluded = set(OmegaConf.select(dcfg, "to_exclude") or [])
        if excluded:
            dropped = [s for s in seqs if s in excluded]
            seqs = [s for s in seqs if s not in excluded]
            print(f"  to_exclude: skipping {dropped or 'nothing'}")

        print(f"\n{features_dir}: {len(seqs)} sequences")
        for seq in seqs:
            n, how = process_sequence(seq, root, features_dir, out_dir,
                                      model, mcfg, device)
            print(f"  {seq:<28} {n:>7} rows  {how}")
            total += n

    print(f"\nDone: {total} descriptors -> {out_dir}")
    print(f"Now train with MEGALOC_DIR={out_dir}")


if __name__ == "__main__":
    main()
