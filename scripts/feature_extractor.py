import importlib.util
import os
import sys
import types
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset import sequence_name_from_rel_path
from model import SignedGeM

# ToTensor() then ImageNet Normalize, matching the benchmark's
# test_dataset.py:132-136 so descriptors are the ones behind its published
# numbers. Every teacher here takes the same normalisation.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_GEM_P = 3.0


def read_pairs(root_dir: str, pairs_name: str = 'pairs.txt'):
    """pairs.txt -> {sequence: [rgb path relative to root_dir]}, in file order."""
    pairs_path = Path(root_dir) / pairs_name
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"pairs.txt file not found at {pairs_path}")

    seq_to_frames: dict[str, list[str]] = {}
    with open(pairs_path, 'r') as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue # skip empty lines and comments
            entries = [e.strip() for e in line.split(',')]
            if len(entries) < 3:
                raise ValueError(f"Invalid line in pairs.txt at line {ln + 1}: {line}")
            rgb_path = entries[1]
            # The SAME key function dataset.py builds its row index with --
            # imported, not reimplemented, because a divergence here silently
            # aligns every teacher target to the wrong frame.
            seq_name = sequence_name_from_rel_path(rgb_path)
            if not seq_name:
                raise ValueError(f"Could not extract sequence name from rgb path: {rgb_path}")
            rgb_path = os.path.join(seq_name, rgb_path) # Store the path relative to the sequence directory
            seq_to_frames.setdefault(seq_name, []).append(rgb_path)

    n_total = sum(len(frames) for frames in seq_to_frames.values())
    print(f"Read {len(seq_to_frames)} sequences with a total of {n_total} frames from {pairs_path}")
    return seq_to_frames, n_total


def dataset_sources(cfg):
    """`datasets`, `datasets_extra`, and any further `datasets_extra<N>` group
    in name order -- the same corpora the matching training command reads."""
    sources = [cfg.datasets]
    extra = OmegaConf.select(cfg, "datasets_extra")
    if extra is not None:
        sources.append(extra)
    for key in sorted(str(k) for k in cfg.keys()
                      if str(k).startswith("datasets_extra")
                      and str(k) != "datasets_extra"):
        more = cfg.get(key)
        if more is not None:
            sources.append(more)
    return sources

def _bench_dir():
    """<repo>/external/ensemble_event_vpr_bench/vpr_methods_evaluation."""
    return (Path(__file__).resolve().parents[1] / "external"
            / "ensemble_event_vpr_bench" / "vpr_methods_evaluation")


def _bench_mixvpr_module():
    """Load `vpr_models/mixvpr.py` WITHOUT its package __init__, which imports
    the whole zoo (apgem needs einops, boq/dinomix hit torch.hub). mixvpr.py
    imports gdown at module scope only for a download path load_mixvpr never
    takes, so a stub suffices when it is not installed."""
    path = _bench_dir() / "vpr_models" / "mixvpr.py"
    if not path.is_file():
        raise SystemExit(f"benchmark MixVPR module not found at {path}")
    if "gdown" not in sys.modules:
        try:
            import gdown            # noqa: F401
        except ModuleNotFoundError:
            sys.modules["gdown"] = types.ModuleType("gdown")
    spec = importlib.util.spec_from_file_location("_bench_mixvpr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_teacher(tcfg: DictConfig, device: torch.device):
    """Frozen teacher + the callable that turns a normalised batch into
    (B, dim) descriptors. `source` is the only thing that differs."""
    source = str(tcfg.source)

    if source == "hub":
        model = torch.hub.load(str(tcfg.hub_repo), str(tcfg.hub_entry),
                               trust_repo=True).eval().to(device)
        return model, (lambda x: model(x))

    if source == "bench":
        bench = _bench_mixvpr_module()
        url, filename, out_channels, out_rows = bench.MODELS_INFO[int(tcfg.dim)]
        model = bench.MixVPRModel(agg_config={
            "in_channels": 1024, "in_h": 20, "in_w": 20,
            "out_channels": out_channels, "mix_depth": 4,
            "mlp_ratio": 1, "out_rows": out_rows,
        })
        weights = OmegaConf.select(tcfg, "weights")
        path = (Path(str(weights)) if weights else
                _bench_dir() / "trained_models" / "mixvpr" / filename)
        if not path.is_file():
            raise SystemExit(
                f"MixVPR weights not found at {path}.\n"
                f"Pass +model.teacher.weights=<path to '{filename}'>, or fetch it once with\n"
                f"    gdown --fuzzy '{url}' -O '{filename}'")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"MixVPR loaded from {path}")
        model = model.eval().to(device)
        return model, (lambda x: model(x))

    if source == "hf":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(tcfg.hf_model)).eval().to(device)
        n_prefix = 1 + (getattr(model.config, "num_register_tokens", None) or 0)
        # Signed GeM, the SAME pooler the student uses -- imported rather than
        # reimplemented so the two cannot drift. p is frozen at 3.
        pool = SignedGeM(p=_GEM_P).to(device)
        n_patches = int(tcfg.expected_patches)

        def encode(x):
            tokens = model(pixel_values=x).last_hidden_state[:, n_prefix:, :]
            if tokens.shape[1] != n_patches:
                raise ValueError(f"expected {n_patches} patch tokens, got {tokens.shape[1]}")
            return pool(tokens.float())

        return model, encode

    raise SystemExit(f"unknown teacher source {source!r} (expected hub/bench/hf)")


def load_rgb_batch(paths, size, device):
    """Preprocessed RGB .npy -> normalised float tensor (B, 3, size, size).

    Stored CHW uint8 by the preprocessors; HWC is accepted too so this does not
    silently transpose a differently-written cache. Resize happens AFTER
    normalisation, which is the order the MegaLoc cache was built with.
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

def process_sequence(seq, frames, root, out_dir, encode, tcfg, device):
    """Returns (n_rows, 'written'|'skipped')."""
    seq_dir = out_dir / seq
    out_path = seq_dir / str(tcfg.cache_file)
    dim = int(tcfg.dim)
    n = len(frames)

    if out_path.is_file():
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape == (n, dim) and (seq_dir / "frames.txt").is_file():
            return n, "skipped"
        print(f"  {seq}: existing cache is {existing.shape}, expected {(n, dim)} -- recomputing")

    paths = [root / k for k in frames]
    missing = [p for p in paths[:64] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{seq}: {missing[0]} from pairs.txt not found under root={root}. "
            f"datasets.root_dir must be the preprocessed corpus these pairs describe.")

    bs = int(tcfg.batch_size)
    amp = bool(tcfg.enable_amp) and device.type == "cuda"
    out = np.empty((n, dim), dtype=np.float16)

    for i in tqdm(range(0, n, bs), desc=f"  {seq}", leave=False):
        chunk = paths[i:i + bs]
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):
            d = encode(load_rgb_batch(chunk, int(tcfg.input_size), device))
        if d.shape[-1] != dim:
            raise ValueError(
                f"{tcfg.name} returned dim {d.shape[-1]}, expected {dim}. "
                f"Update the teacher config only after checking the checkpoint.")
        if not torch.isfinite(d).all():
            raise FloatingPointError(
                f"non-finite descriptor in {seq} near {chunk[0]}. "
                f"Re-run with +model.teacher.enable_amp=false to rule out fp16 overflow.")
        out[i:i + len(chunk)] = d.float().cpu().numpy().astype(np.float16)

    assert out.shape[0] == n, "row count drifted from the frame list"
    seq_dir.mkdir(parents=True, exist_ok=True)
    # descriptors first, frames.txt last => its presence marks a finished sequence
    np.save(out_path, out)
    with open(seq_dir / "frames.txt", "w") as f:
        f.write("\n".join(frames) + "\n")
    return n, "written"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    tcfg = OmegaConf.select(cfg, "model.teacher")
    if tcfg is None:
        raise SystemExit("no model/teacher group selected -- run with "
                         "`model/teacher=megaloc` (see configs/model/teacher/).")

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass device=cpu to run anyway.")

    model, encode = load_teacher(tcfg, device)
    print(f"{tcfg.name}: input {tcfg.input_size}x{tcfg.input_size}, "
          f"descriptor {tcfg.dim}-d -> <seq>/{tcfg.cache_file}")

    override_out = OmegaConf.select(cfg, "extract.out_dir")
    total = 0
    for dcfg in dataset_sources(cfg):
        root = Path(str(dcfg.root_dir))
        out_dir = Path(str(override_out or dcfg.features_dir))
        seq_to_frames, _ = read_pairs(str(root), str(dcfg.pairs_name))

        excluded = set(OmegaConf.select(dcfg, "to_exclude") or [])
        if excluded:
            dropped = sorted(s for s in seq_to_frames if s in excluded)
            seq_to_frames = {s: f for s, f in seq_to_frames.items() if s not in excluded}
            print(f"  to_exclude: skipping {dropped or 'nothing'}")

        print(f"\n{root} -> {out_dir}: {len(seq_to_frames)} sequences")
        for seq, frames in seq_to_frames.items():
            n, how = process_sequence(seq, frames, root, out_dir, encode, tcfg, device)
            print(f"  {seq:<40} {n:>7} rows  {how}")
            total += n

    print(f"\nDone: {total} descriptors ({tcfg.name}, {tcfg.dim}-d)")


if __name__ == '__main__':
    main()
