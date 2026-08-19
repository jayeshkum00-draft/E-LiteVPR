"""Phase 1: distil a frozen RGB teacher's batch-wise retrieval geometry into an
event-native student.

The objective is structural ONLY. Both sides are reduced to one descriptor per
frame, the B x B cosine similarity matrices are compared, and nothing else
contributes a gradient:

    teacher   cached global descriptor, one row per frame, row-aligned to
              <cache_dir>/<seq>/frames.txt (written by feature_extractor.py)
    loss      KL between the two row distributions, self-term dropped and both
              matrices z-scored, so `training.temperature` is in sigma units
    sampler   contiguous clips, day/night balanced at the CLIP level
    schedule  per-step linear warmup then cosine

Teacher and aggregator are config groups, so one command trains every model in
the paper:

    python scripts/train.py model/teacher=megaloc model/aggregator=gem \\
        datasets=dsec datasets@datasets_extra=m3ed +datasets@datasets_extra2=vivid \\
        datasets.root_dir=... datasets.features_dir=... \\
        training.teacher_dir=<dir written by feature_extractor.py>

`training.teacher_dir` accepts a COMMA-SEPARATED list of roots, searched in
order, because each corpus keeps its own cache and they need not share a
filesystem. Hydra reads a bare comma as a sweep, so QUOTE it:

    "training.teacher_dir='/feats/dsec_m3ed,/feats/vivid'"

Left null, each source's own `datasets.features_dir` is used -- which is where
feature_extractor.py writes by default, so a single-corpus run needs nothing.

BATCH SIZE IS PART OF THE OBJECTIVE, not a throughput knob: the loss is a KL
between B x B matrices, so at B=32 each row is a 31-way target and at B=128 a
127-way one. `training.epoch_mult` lengthens the EPOCH rather than the
schedule (the block sampler draws with replacement, so epoch length is free),
which keeps steps/epoch fixed when batch size changes:

    epoch_mult = batch_size / 32   ->   the same steps/epoch at any batch size

`training.val_batch_size` pins validation for the same reason: a 127-way val KL
is not comparable to a 31-way one, so `best` would be chosen on a different
criterion.

Both `best` and `last` are written every epoch. EVALUATE BOTH -- validation is
DSEC-only and day-dominated, and `last` beat `best` on day-night in every run.
"""

import math
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import wandb

from block_sampler import BlockSampler
from dataset import ConcatE_LiteVPRDataset, E_LiteVPRDataset
from model import build_student

# Written in the MAIN process before the DataLoader workers fork; read inside
# _get_features. A worker cannot write it back, so it must never be lazy.
_CACHE = {"dirs": None, "file": None}      # "dirs" is a LIST of roots


# --- objective ----------------------------------------------------------------

def _drop_diag(sim):
    n = sim.shape[0]
    return sim[~torch.eye(n, dtype=torch.bool, device=sim.device)].view(n, n - 1)


def structural_loss(student_global, teacher_global, temperature):
    """Drop the self-term, z-score both matrices, tau in sigma units.

    The diagonal is deleted because S[i, i] = 1 by construction on both sides:
    under a row-wise softmax at tau=0.05 it takes 0.9994 of the row's mass,
    leaving 6e-4 for the entries that actually describe how frames relate.

    float32 throughout -- std/softmax on a near-degenerate similarity matrix is
    exactly where autocast fp16 turns into inf/nan.
    """
    s = F.normalize(student_global.float(), p=2, dim=-1)
    t = F.normalize(teacher_global.float(), p=2, dim=-1)
    s_sim, t_sim = _drop_diag(s @ s.T), _drop_diag(t @ t.T)
    s_sim = (s_sim - s_sim.mean()) / s_sim.std().clamp(min=1e-6)
    t_sim = (t_sim - t_sim.mean()) / t_sim.std().clamp(min=1e-6)
    return F.kl_div(F.log_softmax(s_sim / temperature, dim=-1),
                    F.softmax(t_sim / temperature, dim=-1),
                    reduction="batchmean")


# --- data ---------------------------------------------------------------------

def active_dataset_cfgs(cfg):
    """`datasets`, `datasets_extra`, and any further `datasets_extra<N>` group
    in name order, so a third or fourth corpus needs no code change."""
    cfgs = [cfg.datasets]
    extra = cfg.get("datasets_extra")
    if extra is not None:
        cfgs.append(extra)
    for key in sorted(str(k) for k in cfg.keys()
                      if str(k).startswith("datasets_extra")
                      and str(k) != "datasets_extra"):
        more = cfg.get(key)
        if more is not None:
            cfgs.append(more)
    return cfgs


def build_split(dataset_cfgs, split_key, modality, pair_stride_override=None,
                dataset_cls=E_LiteVPRDataset):
    """One split from every source that has one.

        null           -> every sequence in that source's pairs.txt
        [] or missing  -> that source sits this split out
        list of names  -> exactly those sequences
    """
    parts = []
    for dcfg in dataset_cfgs:
        if split_key not in dcfg:
            continue
        seq_list = dcfg[split_key]          # None means "all", [] means "none"
        if seq_list is not None and len(seq_list) == 0:
            continue
        stride = (pair_stride_override if pair_stride_override is not None
                  else dcfg.get("pair_stride", 1))
        parts.append(dataset_cls(
            root=dcfg.root_dir,
            features_dir=dcfg.features_dir,
            event_type=modality,
            sequences=list(seq_list) if seq_list is not None else None,
            pair_stride=stride,
        ))

    if not parts:
        raise ValueError(
            f"No dataset source contributes to the '{split_key}' split. "
            f"Set {split_key} on at least one dataset config (null = all sequences).")
    return parts[0] if len(parts) == 1 else ConcatE_LiteVPRDataset(parts)


def build_day_night_sampler(dataset, night_seqs):
    """Balance day vs night at the SAMPLE level (the `blocks -> random` ablation)."""
    seqs = [pair["sequence"] for pair in dataset.pairs]
    night_seqs = set(night_seqs)
    is_night = torch.tensor([s in night_seqs for s in seqs], dtype=torch.bool)

    n_night = int(is_night.sum())
    n_day = len(seqs) - n_night
    if n_night == 0 or n_day == 0:
        raise ValueError(
            f"Day/night sampler degenerate: {n_day} day / {n_night} night samples. "
            "Check night_sequences against the train split.")

    weights = torch.where(is_night, torch.tensor(1.0 / n_night),
                          torch.tensor(1.0 / n_day)).double()
    print(f"Day/night sampler: {n_day} day / {n_night} night samples "
          f"(night weight x{n_day / n_night:.2f})")
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def _find_cache(roots, seq, fname):
    """First root holding <seq>/<fname>. Sequence names are unique across
    corpora, so this cannot pick the wrong corpus's cache."""
    tried = []
    for root in roots:
        path = Path(root) / seq / fname
        tried.append(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{seq}/{fname} not found under any of {[str(t) for t in tried]} -- "
        f"cache that corpus first with feature_extractor.py.")


def _open_cache(roots, seq, features_dir, fname):
    """mmap <root>/<seq>/<fname>, asserting row alignment with frames.txt.

    dataset.py maps a pair to a cache row through _row_index, built from
    frames.txt. A cache with a different row count is a cache of different
    frames, so every target after it would be the wrong frame -- fail here.
    """
    path = _find_cache(roots, seq, fname)
    arr = np.load(path, mmap_mode="r")
    n_rows = sum(1 for ln in (Path(features_dir) / seq / "frames.txt")
                 .read_text().splitlines() if ln.strip())
    if arr.shape[0] != n_rows:
        raise RuntimeError(
            f"{path} has {arr.shape[0]} rows but {seq}/frames.txt has {n_rows}. "
            f"Re-run feature_extractor.py for {seq}.")
    return arr


class CachedGlobalDataset(E_LiteVPRDataset):
    """Serves (1, D): the cached teacher descriptor, nothing else. The (B, 1, D)
    collate contract and __init__'s shape probe are unchanged."""

    def _get_features(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            self._mmaps[seq] = _open_cache(_CACHE["dirs"], seq,
                                           self.features_dir, _CACHE["file"])
        arr = self._mmaps[seq]
        row = self._row_index[pair["feature_key"]]
        d = torch.from_numpy(np.ascontiguousarray(arr[row]).astype(np.float32))
        return d.unsqueeze(0), torch.zeros(1)


# --- loop ---------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_loss,
                    patience_counter, wandb_run_id, cfg):
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,  # last COMPLETED epoch (0-indexed)
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "wandb_run_id": wandb_run_id,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic: never leaves a half-written checkpoint


def run_epoch(model, loader, optimizer, scaler, device, cfg, sched,
              train_mode, amp):
    w = float(cfg.training.structural_loss_weight)
    tau = float(cfg.training.temperature)

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
            # autocast re-casts the matmuls themselves, and a std/softmax over
            # a near-degenerate similarity matrix is exactly where fp16 turns
            # into inf/nan.
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    tcfg = OmegaConf.select(cfg, "model.teacher")
    if tcfg is None:
        raise SystemExit("no model/teacher group selected -- run with "
                         "`model/teacher=megaloc` (see configs/model/teacher/).")

    torch.manual_seed(cfg.seed)
    # Fail rather than train 30 epochs on CPU: an earlier launch did exactly
    # that, fell back silently and exhausted host RAM instead of raising.
    if not torch.cuda.is_available() and not bool(cfg.get("allow_cpu", False)):
        raise SystemExit(
            "CUDA is not available -- torch.cuda.is_available() is False.\n"
            "Pass +allow_cpu=true only to smoke-test the wiring.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.training.get("enable_amp", True)) and device.type == "cuda"

    dataset_cfgs = active_dataset_cfgs(cfg)
    teacher_dir = cfg.training.get("teacher_dir", None)
    roots = ([t.strip() for t in str(teacher_dir).split(",") if t.strip()]
             if teacher_dir else [str(d.features_dir) for d in dataset_cfgs])
    _CACHE["dirs"], _CACHE["file"] = roots, str(tcfg.cache_file)

    print("Initializing datasets...")
    train_ds = build_split(dataset_cfgs, "train_seq_list", cfg.data.modality,
                           dataset_cls=CachedGlobalDataset)
    val_ds = build_split(dataset_cfgs, "val_seq_list", cfg.data.modality,
                         pair_stride_override=1, dataset_cls=CachedGlobalDataset)
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
        raise ValueError(f"night_sequences not in the loaded splits: {sorted(unknown)}")

    use_blocks = bool(cfg.training.get("use_block_sampler", True))
    block = int(cfg.training.get("block_size", 8))
    mult = float(cfg.training.get("epoch_mult", 1.0))
    if use_blocks:
        if int(cfg.training.batch_size) % block:
            raise SystemExit(f"batch_size={cfg.training.batch_size} must be a "
                             f"multiple of block_size={block}")
        sampler = BlockSampler(train_ds, night_seqs, block=block,
                               night_frac=float(cfg.training.get("night_frac", 0.5)),
                               num_samples=int(round(len(train_ds) * mult)))
    else:
        if mult != 1.0:
            raise SystemExit("training.epoch_mult applies to the block sampler "
                             "only; with use_block_sampler=false it is a no-op.")
        sampler = build_day_night_sampler(train_ds, night_seqs)

    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                          sampler=sampler, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=cfg.num_workers > 0)
    val_bs = cfg.training.get("val_batch_size", None)
    val_bs = int(cfg.training.batch_size) if val_bs is None else int(val_bs)
    val_dl = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)

    print("Initializing model...")
    model = build_student(cfg).to(device)

    lr_base = cfg.training.get("lr_base_batch_size", cfg.training.batch_size)
    lr = cfg.training.learning_rate * (cfg.training.batch_size / lr_base) ** 0.5
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    sched = None
    if bool(cfg.training.get("use_scheduler", True)):
        total_steps = len(train_dl) * int(cfg.training.epochs)
        warm = max(1, int(round(float(cfg.training.get("warmup_frac", 0.03)) * total_steps)))
        floor = float(cfg.training.get("min_lr_frac", 0.05))

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total_steps - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        sched = optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        print(f"  LR schedule: {warm} warmup then cosine to {floor:g}x over "
              f"{total_steps} steps")

    width = int(np.load(_find_cache(roots, sorted(train_ds.sequence_names())[0],
                                    _CACHE["file"]), mmap_mode="r").shape[1])
    print("=" * 70)
    print(f"teacher={tcfg.name} ({width}-d) <- {len(roots)} root(s)/<seq>/{_CACHE['file']}")
    for r in roots:
        print(f"    {r}")
    print(f"  corpora: {len(dataset_cfgs)} source(s), "
          f"{len(train_ds.sequence_names())} train sequences")
    print(f"  aggregator={cfg.model.aggregator.name}  blocks={use_blocks}"
          + (f" (block={block}, night_frac={float(cfg.training.get('night_frac', 0.5))})"
             if use_blocks else ""))
    print(f"  batch={int(cfg.training.batch_size)} (val {val_bs})  "
          f"epoch_mult={mult:g}  ->  {len(train_dl)} steps/epoch, "
          f"{len(train_dl) * int(cfg.training.epochs)} total steps")
    print(f"  loss = {float(cfg.training.structural_loss_weight):g} * "
          f"KL(mask_diag, standardize, tau={float(cfg.training.temperature):g})")
    print("=" * 70)

    wandb.init(project=cfg.get("project_name", "e-litevpr"),
               config=OmegaConf.to_container(cfg, resolve=True),
               mode=os.environ.get("WANDB_MODE", "online"))

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    mode = cfg.training.get("name", "phase1")
    best_path = os.path.join(cfg.checkpoint_dir, f"best_{mode}_{cfg.data.modality}.pth")
    last_path = os.path.join(cfg.checkpoint_dir, f"last_{mode}_{cfg.data.modality}.pth")

    best_val, patience = float("inf"), 0
    for epoch in range(int(cfg.training.epochs)):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        tr = run_epoch(model, train_dl, optimizer, scaler, device, cfg, sched, True, amp)
        va = run_epoch(model, val_dl, optimizer, None, device, cfg, None, False, amp)
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
        # run, because validation is DSEC-only and day-dominated.
        torch.save(model.state_dict(), last_path)
        save_checkpoint(os.path.join(cfg.checkpoint_dir, "resume.pth"), model,
                        optimizer, scaler, epoch, best_val, patience,
                        wandb.run.id, cfg)

        if cfg.early_stopping > 0 and patience >= cfg.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nDone. best_val_loss={best_val:.6f}")
    print(f"  {best_path}\n  {last_path}")
    print("  EVALUATE BOTH -- `last` beat `best` on day-night in every run.")
    wandb.finish()


if __name__ == "__main__":
    main()
