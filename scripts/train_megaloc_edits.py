"""Phase 1, cached-global teacher, STRUCTURAL KL ONLY, with the pooling flag.

WHAT THIS IS
------------
`train_megaloc.py` with InfoNCE deleted -- i.e. the `+mega.infonce_weight=0.0`
control (`ckpt_kl`, 83.85 / 36.53), which is also `train_bankkl.py` with both
its flags off. Everything that recipe holds constant is held constant here:

  * teacher  = cached GLOBAL descriptor, one row per frame, row-aligned to
               <features_dir>/<seq>/frames.txt
  * loss     = train_megaloc.structural_loss, IMPORTED not copied: drop the
               diagonal (B x B -> B x (B-1)), z-score both matrices, fp32.
               `training.temperature` is therefore in sigma units (use 1.0).
  * sampler  = BlockSampler, contiguous clips, block-level day/night balance
  * schedule = per-step warmup then cosine
  * patch loss = none. A cached global descriptor has no patch target, so
               `training.patch_loss_weight=0.0` is required, not optional.

Deleted relative to train_megaloc.py: the InfoNCE term, its 1024->8448 head,
the negative bank (and its ~345 MB of GPU memory), the +/-32 exclusion window,
and the (seq id, row) tail the dataset packed for that mask. Nothing else.

THE ONE NEW KNOB
----------------
`model.pooling` (configs/model/student_model.yaml, default `clamp`) selects the
student's patch aggregation. `signed` / `mean` register the pooler as `pool`
rather than `gem`, so the stock evaluator refuses those checkpoints instead of
scoring them through the clamped GeM -- use `evaluate_brisbane_gem.py
+gem.pooling=signed`.

`model.teacher_pooling` does NOTHING here. The target is pooled once, at cache
time (cache_teacher_gem.py:53 is the clamped GeM, cache_megaloc.py has no GeM at
all), so changing the teacher's pooling would mean writing a new cache, not
passing a flag.

`train_dino_edits.py` calls `run()` below with the DINOv3 cache, so the two
teachers differ by a filename and nothing else.

COMMANDS (Kaggle paths -- see the `kaggle-run-layout` memory note)
-----------------------------------------------------------------
  python scripts/train_megaloc_edits.py \
    datasets=dsec datasets@datasets_extra=m3ed \
    datasets.root_dir=$DSEC_ROOT   datasets.output_dir=$DSEC_FEATS \
    datasets_extra.root_dir=$M3ED_ROOT datasets_extra.output_dir=$M3ED_FEATS \
    training.patch_loss_weight=0.0 training.temperature=1.0 \
    +edits.teacher_dir=$MEGALOC_DIR \
    model.pooling=clamp \
    checkpoint_dir=/kaggle/working/ckpt_mega_clamp \
    output_dir=/kaggle/working/hydra_mega_clamp

Then `model.pooling=signed` with its own checkpoint_dir for the other arm. No
HYDRA_MAIN_MODULE needed -- this file owns its own @hydra.main.

Both `best` and `last` are written every epoch. Evaluate both: val is DSEC-only
and day-dominated (m3ed.yaml `val_seq_list: []`), and `last` beat `best` on
day-night in every expert run so far.
"""

import math
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from block_sampler import BlockSampler
from dataset import E_LiteVPRDataset
from model import EventViTStudent
import train as T
from train_megaloc import structural_loss


# Written in the MAIN process before the DataLoader workers fork; read inside
# _get_features. A worker cannot write it back, so it must never be lazy.
_CACHE = {"dir": None, "file": None}

_DEFAULTS = {
    "teacher_dir": None,
    "use_block_sampler": True,
    "block_size": 8,
    "night_frac": 0.5,
    "use_scheduler": True,
    "warmup_frac": 0.03,
    "min_lr_frac": 0.05,
}


def _e(cfg, key):
    v = OmegaConf.select(cfg, f"edits.{key}")
    return _DEFAULTS[key] if v is None else v


# --------------------------------------------------------------- dataset ---
def _open_cache(root, seq, features_dir, fname):
    """mmap <root>/<seq>/<fname>, asserting row alignment with frames.txt.

    dataset.py maps a pair to a cache row through _row_index, which is built
    from frames.txt. A cache with a different row count is a cache of different
    frames, so every target after it would be the wrong frame -- fail here, not
    silently.
    """
    path = Path(root) / seq / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- cache this corpus first "
            f"(cache_megaloc.py for megaloc.npy, cache_teacher_gem.py for gem.npy).")
    arr = np.load(path, mmap_mode="r")
    n_rows = sum(1 for ln in (Path(features_dir) / seq / "frames.txt")
                 .read_text().splitlines() if ln.strip())
    if arr.shape[0] != n_rows:
        raise RuntimeError(
            f"{path} has {arr.shape[0]} rows but {seq}/frames.txt has {n_rows}. "
            f"Re-run the caching script for {seq}.")
    return arr


class CachedGlobalDataset(E_LiteVPRDataset):
    """Serves (1, D): the cached teacher descriptor, nothing else.

    train_megaloc.MegaLocDataset packed a (seq id, row) tail so InfoNCE could
    mask same-place negatives. There are no negatives here, so the tail is gone
    and the row is served as-is. The (B, 1, D) collate contract and __init__'s
    `patches.ndim == 2 and attn.ndim == 1` probe are unchanged.
    """

    def _get_features(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            self._mmaps[seq] = _open_cache(_CACHE["dir"], seq,
                                           self.features_dir, _CACHE["file"])
        arr = self._mmaps[seq]
        row = self._row_index[pair["feature_key"]]
        d = torch.from_numpy(np.ascontiguousarray(arr[row]).astype(np.float32))
        return d.unsqueeze(0), torch.zeros(1)


def _probe_width(teacher_dir, seq, fname):
    return int(np.load(Path(teacher_dir) / seq / fname, mmap_mode="r").shape[1])


# ------------------------------------------------------------------ loop ---
def run_epoch(model, loader, optimizer, scaler, device, cfg, sched,
              train_mode, state):
    w = float(cfg.training.structural_loss_weight)
    tau = float(cfg.training.temperature)
    amp = bool(state["amp"])

    model.train(train_mode)
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Training" if train_mode else "Validation")
    for images, packed, _attn, _ts in pbar:
        images = images.to(device, non_blocking=True)
        teacher_global = packed.to(device, non_blocking=True).squeeze(1)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            # ONLY the forward runs under autocast. structural_loss .float()s
            # its inputs, but inside an autocast region that is not enough --
            # autocast re-casts the matmuls themselves, and a std/softmax over a
            # near-degenerate similarity matrix is exactly where fp16 turns into
            # inf/nan.
            with torch.autocast(device_type="cuda", enabled=amp):
                _patches, student_global = model(images)
            loss = w * structural_loss(student_global, teacher_global, tau)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {n}: {loss.item()}")

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if sched is not None:
                sched.step()

        total += loss.item()
        n += 1
        pbar.set_postfix({"kl": f"{loss.item():.4f}"})
    return total / max(n, 1)


# ------------------------------------------------------------------ main ---
def run(cfg: DictConfig, teacher_name, cache_file, cache_script):
    """The whole recipe. `train_dino_edits.py` calls this with the DINOv3 cache
    so the two teachers cannot drift apart in anything else."""
    teacher_dir = _e(cfg, "teacher_dir")
    if not teacher_dir:
        raise SystemExit(
            f"+edits.teacher_dir=<dir written by {cache_script}> is required "
            f"(this script reads <dir>/<seq>/{cache_file}).")

    torch.manual_seed(cfg.seed)
    # Fail rather than train 30 epochs on CPU: an earlier launch did exactly
    # that, fell back silently and exhausted host RAM instead of raising.
    if not torch.cuda.is_available() and not bool(cfg.get("allow_cpu", False)):
        raise SystemExit(
            "CUDA is not available -- torch.cuda.is_available() is False.\n"
            "Check the notebook accelerator and CUDA_VISIBLE_DEVICES.\n"
            "Pass +allow_cpu=true only to smoke-test the wiring.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.training.get("enable_amp", True)) and device.type == "cuda"

    if float(cfg.training.get("patch_loss_weight", 1.0)):
        raise SystemExit("a cached GLOBAL teacher descriptor has no patch "
                         "target: pass training.patch_loss_weight=0.0")

    pooling = str(cfg.model.get("pooling", "clamp"))
    dataset_cfgs = T.active_dataset_cfgs(cfg)
    _CACHE["dir"], _CACHE["file"] = str(teacher_dir), cache_file

    print("Initializing datasets...")
    orig = T.E_LiteVPRDataset
    T.E_LiteVPRDataset = CachedGlobalDataset
    try:
        train_ds = T.build_split(dataset_cfgs, "train_seq_list", cfg.data.modality)
        val_ds = T.build_split(dataset_cfgs, "val_seq_list", cfg.data.modality,
                               pair_stride_override=1)
    finally:
        T.E_LiteVPRDataset = orig
    print(f"train dataset size: {len(train_ds)}")
    print(f"val dataset size: {len(val_ds)}")

    # A typo'd night sequence would silently count as day. Check against the
    # sequences actually loaded, not the config lists.
    night_seqs = []
    for dcfg in dataset_cfgs:
        night_seqs.extend(dcfg.get("night_sequences") or [])
    known = set(train_ds.sequence_names()) | set(val_ds.sequence_names())
    unknown = set(night_seqs) - known
    if unknown:
        raise ValueError(f"night_sequences not in the loaded splits: "
                         f"{sorted(unknown)}")

    use_blocks = bool(_e(cfg, "use_block_sampler"))
    block = int(_e(cfg, "block_size"))
    if use_blocks:
        if int(cfg.training.batch_size) % block:
            raise SystemExit(f"batch_size={cfg.training.batch_size} must be a "
                             f"multiple of block_size={block}")
        sampler = BlockSampler(train_ds, night_seqs, block=block,
                               night_frac=float(_e(cfg, "night_frac")))
    else:
        sampler = T.build_day_night_sampler(train_ds, night_seqs)

    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                          sampler=sampler, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)

    print("Initializing model...")
    model = EventViTStudent(
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,
        img_size=cfg.model.img_hw[0],
        in_channels=cfg.data.input_channels,
        pooling=pooling,
    ).to(device)

    lr_base = cfg.training.get("lr_base_batch_size", cfg.training.batch_size)
    lr = cfg.training.learning_rate * (cfg.training.batch_size / lr_base) ** 0.5
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    sched = None
    if bool(_e(cfg, "use_scheduler")):
        total = len(train_dl) * int(cfg.training.epochs)
        warm = max(1, int(round(float(_e(cfg, "warmup_frac")) * total)))
        floor = float(_e(cfg, "min_lr_frac"))

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        sched = optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        print(f"  LR schedule: {warm} warmup then cosine to {floor:g}x over "
              f"{total} steps")

    width = _probe_width(teacher_dir, sorted(train_ds.sequence_names())[0],
                         cache_file)
    print("=" * 70)
    print(f"{teacher_name}_edits: teacher={teacher_name}({width}-d) "
          f"<- {teacher_dir}/<seq>/{cache_file}")
    print(f"  student pooling={pooling}  blocks={use_blocks}"
          + (f" (block={block}, night_frac={float(_e(cfg, 'night_frac'))})"
             if use_blocks else ""))
    print(f"  loss = {float(cfg.training.structural_loss_weight):g} * "
          f"KL(mask_diag, standardize, tau="
          f"{float(cfg.training.temperature):g})   [no patch, no InfoNCE]")
    if pooling != "clamp":
        print(f"  NOTE pooling={pooling} writes 'pool.*' keys: score it with "
              f"evaluate_brisbane_gem.py +gem.pooling={pooling}")
    print("=" * 70)

    wandb.init(project=cfg.get("wandb_project", "e-litevpr"),
               config=OmegaConf.to_container(cfg, resolve=True),
               mode=os.environ.get("WANDB_MODE", "online"))

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    mode = "phase1"
    best_path = os.path.join(cfg.checkpoint_dir,
                             f"best_{mode}_{cfg.data.modality}.pth")
    last_path = os.path.join(cfg.checkpoint_dir,
                             f"last_{mode}_{cfg.data.modality}.pth")

    state = {"amp": amp}
    best_val, patience = float("inf"), 0

    for epoch in range(int(cfg.training.epochs)):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        tr = run_epoch(model, train_dl, optimizer, scaler, device, cfg, sched,
                       True, state)
        va = run_epoch(model, val_dl, optimizer, None, device, cfg, None,
                       False, state)
        print(f"Train Loss: {tr:.6f}")
        print(f"Val   Loss: {va:.6f}")
        wandb.log({"epoch": epoch + 1, "train_loss": tr, "val_loss": va,
                   "lr": optimizer.param_groups[0]["lr"]})

        if va < best_val:
            best_val, patience = va, 0
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with val_loss: {va:.6f}")
        else:
            patience += 1

        # `last` is written EVERY epoch: it beat `best` on day-night in every
        # expert run, because val is DSEC-only and day-dominated.
        torch.save(model.state_dict(), last_path)
        T.save_checkpoint(os.path.join(cfg.checkpoint_dir, "resume.pth"), model,
                          optimizer, scaler, epoch, best_val, patience,
                          wandb.run.id, cfg)

        if cfg.early_stopping > 0 and patience >= cfg.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nDone. best_val_loss={best_val:.6f}")
    print(f"  {best_path}\n  {last_path}")
    print("  EVALUATE BOTH -- `last` beat `best` on day-night in every "
          "expert run.")
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run(cfg, "megaloc", "megaloc.npy", "cache_megaloc.py")


if __name__ == "__main__":
    main()
