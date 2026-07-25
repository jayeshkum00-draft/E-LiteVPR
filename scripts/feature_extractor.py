import json
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
from transformers import AutoModel

def read_pairs(root_dir: str, pairs_name: str = 'pairs.txt'):
    """
    Read the pairs.txt file and return the list of rgb paths.
    This is returned as a dictionary with the following structure:
        'sequence_name': {
            'rgb': [list of rgb paths]
        }
    """
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
            if len(entries) < 4:
                raise ValueError(f"Invalid line in pairs.txt at line {ln + 1}: {line}")
            rgb_path = entries[1]
            seq_name = rgb_path.split('/')[1].rsplit('_', 1)[0] # Extract sequence name from the path
            if not seq_name:
                raise ValueError(f"Could not extract sequence name from rgb path: {rgb_path}")
            rgb_path = os.path.join(seq_name, rgb_path) # Store the path relative to the sequence directory
            seq_to_frames.setdefault(seq_name, []).append(rgb_path)

    n_total = sum(len(frames) for frames in seq_to_frames.values())
    print(f"Read {len(seq_to_frames)} sequences with a total of {n_total} frames from {pairs_path}")
    return seq_to_frames, n_total

def load_teacher(device: torch.device, cfg: DictConfig):
    print(f"Loading teacher model: {cfg.model.teacher_model}")
    model = AutoModel.from_pretrained(cfg.model.teacher_model)
    # eager attention is required for output_attentions=True (SDPA won't return them)
    model.set_attn_implementation('eager')
    model.to(device).eval()

    num_reg = getattr(model.config, "num_register_tokens", None)
    if num_reg is None:
        raise RuntimeError(
            "model.config has no num_register_tokens attribute. This is required for the teacher model to work."
        )
    
    n_prefix = 1 + num_reg # CLS + registers
    print(f"prefix tokens: 1 CLS + {num_reg} register tokens = {n_prefix} prefix tokens")
    return model, n_prefix

def load_rgb_batch(root_dir: str, rel_paths: list[str], cfg: DictConfig) -> torch.Tensor:
    """
    Loads a batch of RGB images from the given relative paths and returns them as a tensor.
    """
    arrs = []
    for path in rel_paths:
        img_npy = np.load(Path(root_dir) / path)
        if img_npy.dtype != np.uint8 or img_npy.shape != (3, *cfg.model.img_hw):
            raise ValueError(f"Unexpected shape or dtype for {path}: {img_npy.shape}, {img_npy.dtype}")
        arrs.append(img_npy)

    return torch.from_numpy(np.stack(arrs)) # (B,3,H,W) uint8

@torch.inference_mode()
def extract_batch(model, n_prefix: int, rgb_uint8: torch.Tensor, device, use_amp: bool, cfg):
    rgb = rgb_uint8.to(device, non_blocking=True).float().div_(255.0) 
    mean = torch.tensor(list(cfg.model.imagenet_mean), device=device).view(1, 3, 1, 1)
    std = torch.tensor(list(cfg.model.imagenet_std), device=device).view(1, 3, 1, 1)
    rgb = (rgb - mean) / std

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(pixel_values=rgb, output_attentions=True)

    hidden = out.last_hidden_state          # (B, n_prefix + P, D)
    cls_token = hidden[:, 0, :]             # (B, D)
    patches = hidden[:, n_prefix:, :]       # (B, P, D)
    attn = out.attentions[-1][:, :, 0, n_prefix:].mean(dim=1) # (B, P) CLS->patch, head-mean

    if patches.shape[1] != cfg.model.expected_patches:
        raise ValueError(f"Unexpected number of patches: {patches.shape[1]} (expected {cfg.model.expected_patches})")
    
    if patches.shape[2] != cfg.model.teacher_model_output_dim:
        raise ValueError(f"Unexpected patch embedding dimension: {patches.shape[2]} (expected {cfg.model.teacher_model_output_dim})")

    # fail fast — a NaN here poisons every epoch downstream
    for name, t in (("patches", patches), ("cls", cls_token), ("attn", attn)):
        if not torch.isfinite(t).all():
            raise FloatingPointError(
                f"non-finite values in teacher {name}; "
                f"re-run with 'cfg.model.enable_amp=False' to rule out fp16 overflow"
            )

    return (cls_token.half().cpu().numpy(),
            patches.half().cpu().numpy(),
            attn.half().cpu().numpy())

def sequence_done(seq_dir: str, n: int, cfg) -> bool:
    try:
        for fname, shape in (("patches.npy", (n, cfg.model.expected_patches, cfg.model.teacher_model_output_dim)),
                             ("cls.npy", (n, cfg.model.teacher_model_output_dim)),
                             ("attn.npy", (n, cfg.model.expected_patches))):
            a = np.load(os.path.join(seq_dir, fname), mmap_mode="r")
            if a.shape != shape or a.dtype != np.float16:
                return False
        with open(os.path.join(seq_dir, "frames.txt")) as f:
            if sum(1 for _ in f) != n:
                return False
        return True
    except (FileNotFoundError, ValueError):
        return False

def process_sequence(model, n_prefix, root, seq, frames, out_dir, batch_size, device, use_amp, cfg):
    seq_dir = os.path.join(out_dir, seq)
    n = len(frames)
    if sequence_done(seq_dir, n, cfg):
        print(f"[skip] {seq}: complete ({n} frames)")
        return

    os.makedirs(seq_dir, exist_ok=True)
    patches_all = np.empty((n, cfg.model.expected_patches, cfg.model.teacher_model_output_dim), dtype=np.float16)
    cls_all = np.empty((n, cfg.model.teacher_model_output_dim), dtype=np.float16)
    attn_all = np.empty((n, cfg.model.expected_patches), dtype=np.float16)

    for start in tqdm(range(0, n, batch_size), desc=seq, leave=False):
        chunk = frames[start:start + batch_size]
        x = load_rgb_batch(root, chunk, cfg)
        cls_np, patch_np, attn_np = extract_batch(model, n_prefix, x, device, use_amp, cfg)
        end = start + len(chunk)
        cls_all[start:end] = cls_np
        patches_all[start:end] = patch_np
        attn_all[start:end] = attn_np

    # write arrays first, frames.txt last => its presence marks a finished sequence
    np.save(os.path.join(seq_dir, "patches.npy"), patches_all)
    np.save(os.path.join(seq_dir, "cls.npy"), cls_all)
    np.save(os.path.join(seq_dir, "attn.npy"), attn_all)
    with open(os.path.join(seq_dir, "frames.txt"), "w") as f:
        f.write("\n".join(frames) + "\n")
    print(f"[done] {seq}: {n} frames -> {seq_dir}")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Do not run on CPU!")
    device = torch.device("cuda")
    
    seq_to_frames, n_total = read_pairs(cfg.datasets.root_dir, cfg.datasets.pairs_name)

    model, n_prefix = load_teacher(device, cfg)

    os.makedirs(cfg.datasets.output_dir, exist_ok=True)
    with open(os.path.join(cfg.datasets.output_dir, "meta.json"), "w") as f:
        json.dump({
            "model": cfg.model.teacher_model,
            "img_hw": list(cfg.model.img_hw),
            "patch_size": cfg.model.patch_size,
            "num_patches": cfg.model.expected_patches,
            "teacher_model_output_dim": cfg.model.teacher_model_output_dim,
            "prefix_tokens": n_prefix,
            "attn": "last layer, CLS->patch, mean over heads",
            "norm": "uint8/255 -> ImageNet mean/std",
            "dtype": "float16",
            "augmentation": "none",
        }, f, indent=2)

    # sanity check before starting the extraction
    first_seq = next(iter(seq_to_frames))
    _ = extract_batch(model, n_prefix, 
                      load_rgb_batch(cfg.datasets.root_dir, seq_to_frames[first_seq][:1], cfg),
                      device, cfg.model.enable_amp, cfg)
    print("startup self-test passed (1 frame end-to-end)")

    for seq, frames in seq_to_frames.items():
        process_sequence(model, n_prefix, cfg.datasets.root_dir, seq, frames,
                         cfg.datasets.output_dir, cfg.model.batch_size, device, cfg.model.enable_amp, cfg)

    print(f"\nAll sequences cached ({n_total} frames total).")
    
if __name__ == '__main__':
    main()