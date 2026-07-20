"""Phase 2: GPS-triplet metric learning on MVSEC + NYC slices, initialised
from the Phase 1 distilled student.

  python phase2_train.py datasets=phase2_corpus training=phase2

Loss: batch-hard triplet on L2-normalised global descriptors. Positives are
the sampled pair partners (miner CSR, cross-condition quota in the dataset);
negatives are mined inside the batch under the geometric validity mask from
phase2_dataset (different dataset, or same registration frame and farther
than neg_floor). Validation = night3 -> day2 retrieval R@1 (val probe);
best checkpoint maximises it.
"""
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import wandb

from model import EventViTStudent
from phase2_dataset import Phase2PairDataset, Phase2ProbeDataset

class Phase2Net(nn.Module):
    """Student + descriptor head -> normalised descriptor.

    The BatchNorm is load-bearing: Phase 1 descriptors share a dominant
    common component (cos_mean ~0.98), and both the raw GeM output and a
    random linear projection of it stay collapsed, stalling the triplet
    loss at the margin (verified in the first two Phase 2 runs). BN's
    per-dimension centering removes the common component and rescales the
    informative residual to unit variance, so training starts from a
    spread embedding. Inference uses BN running stats (deterministic)."""

    def __init__(self, student, desc_dim=0):
        super().__init__()
        self.student = student
        layers = [nn.BatchNorm1d(student.teacher_dim)]
        if desc_dim:
            layers.append(nn.Linear(student.teacher_dim, desc_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        _, g = self.student(x)
        return F.normalize(self.head(g), p=2, dim=-1)

def batch_hard_triplet(desc, xy, group, dataset, floor, margin):
    """desc (2B, D) normalised; pairs are (2i, 2i+1). Returns (loss,
    active_fraction, mean_pos_d, mean_neg_d)."""
    n = desc.shape[0]
    d = torch.cdist(desc, desc)                                  # (2B, 2B)

    partner = torch.arange(n, device=desc.device) ^ 1
    pos_d = d[torch.arange(n, device=desc.device), partner]

    geo = torch.cdist(xy, xy)                                    # (2B, 2B)
    fl = torch.maximum(floor[:, None], floor[None, :])
    valid = (dataset[:, None] != dataset[None, :]) | \
            ((group[:, None] == group[None, :]) & (geo > fl))

    d_masked = d.masked_fill(~valid, float("inf"))
    neg_d, _ = d_masked.min(dim=1)
    has_neg = torch.isfinite(neg_d)

    loss_all = F.relu(pos_d - neg_d + margin)[has_neg]
    active = float((loss_all > 0).float().mean()) if len(loss_all) else 0.0
    loss = loss_all.mean() if len(loss_all) else desc.sum() * 0.0
    return loss, active, float(pos_d.detach().mean()), \
        float(neg_d[has_neg].detach().mean()) if has_neg.any() else float("nan")

def flatten_batch(batch, device):
    reps = batch["reps"].flatten(0, 1).to(device, non_blocking=True)
    xy = batch["xy"].flatten(0, 1).to(device)
    group = batch["group"].flatten(0, 1).to(device)
    dataset = batch["dataset"].flatten(0, 1).to(device)
    floor = batch["floor"].flatten(0, 1).to(device)
    return reps, xy, group, dataset, floor

@torch.no_grad()
def run_probe(model, probe, device, batch_size, num_workers, use_head=True):
    """Probe R@1 with the head descriptor (use_head=True) or the backbone
    GeM descriptor (use_head=False -- what we deploy; the head is a
    training-time device a la SimCLR projection heads)."""
    model.eval()
    loader = DataLoader(probe, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    outs = [None] * len(probe)
    for reps, idx in tqdm(loader, desc="Probe"):
        reps = reps.to(device, non_blocking=True)
        if use_head:
            d = model(reps).cpu()
        else:
            _, g = model.student(reps)
            d = F.normalize(g, p=2, dim=-1).cpu()
        for k, i in enumerate(idx.tolist()):
            outs[i] = d[k]
    return probe.recall_at_1(torch.stack(outs))

def save_checkpoint(path, model, optimizer, scaler, epoch, best_r1,
                    patience, run_id, cfg):
    tmp = path + ".tmp"
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_r1": best_r1, "patience_counter": patience,
        "wandb_run_id": run_id,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, tmp)
    os.replace(tmp, path)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    t_cfg = cfg.training
    enable_amp = bool(t_cfg.get("enable_amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if enable_amp else None
    mode, modality = t_cfg.name, cfg.data.modality

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    last_path = os.path.join(cfg.checkpoint_dir, f"last_{mode}_{modality}.pth")
    best_path = os.path.join(cfg.checkpoint_dir, f"best_{mode}_{modality}.pth")

    resume = cfg.get("resume", True) and os.path.exists(last_path)
    prev_run_id = None
    if resume:
        prev_run_id = torch.load(last_path,
                                 map_location="cpu").get("wandb_run_id")
        print(f"Resuming from {last_path} (wandb run: {prev_run_id})")

    wandb.init(project=cfg.project_name, id=prev_run_id, resume="allow",
               name=f"{mode}_{modality}_bs{t_cfg.batch_size}"
                    f"_m{t_cfg.margin}",
               config=OmegaConf.to_container(t_cfg, resolve=True))

    print("Initializing corpus...")
    train_ds = Phase2PairDataset(cfg.datasets, t_cfg, modality,
                                 cfg.model.img_hw[0], seed=cfg.seed)
    probe = Phase2ProbeDataset(cfg.datasets, t_cfg, modality,
                               cfg.model.img_hw[0])
    n_night = int(train_ds.anchor_night.sum())
    n_day = len(train_ds) - n_night
    print(f"anchors: {len(train_ds)} ({n_day} day / {n_night} night), "
          f"probe: {len(probe)}")

    if bool(t_cfg.night_balance):
        w = np.where(train_ds.anchor_night, 1.0 / max(n_night, 1),
                     1.0 / max(n_day, 1))
        sampler = WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double),
            num_samples=int(t_cfg.steps_per_epoch) * int(t_cfg.batch_size),
            replacement=True)
    else:
        sampler = None

    loader = DataLoader(train_ds, batch_size=t_cfg.batch_size,
                        sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True,
                        persistent_workers=cfg.num_workers > 0)

    print("Initializing model...")
    student = EventViTStudent(
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,
        img_size=cfg.model.img_hw[0],
        in_channels=cfg.data.input_channels)
    p1 = cfg.get("phase1_weights")
    if p1 and os.path.exists(p1):
        student.load_state_dict(torch.load(p1, map_location="cpu"))
        print(f"Loaded Phase 1 weights: {p1}")
    else:
        raise FileNotFoundError(
            f"phase1_weights not found at {p1!r} -- Phase 2 must start from "
            "the distilled student (set phase1_weights=... to override)")
    model = Phase2Net(student, int(t_cfg.desc_dim)).to(device)

    # differential lrs: the head must learn fast (it is new and does the
    # de-collapsing); the backbone carries Phase 1 semantics and is
    # protected (prior submission: global 1e-4 crashed Phase 1 knowledge)
    lr_bb = float(t_cfg.get("lr_backbone", t_cfg.get("learning_rate", 5e-5)))
    lr_hd = float(t_cfg.get("lr_head", lr_bb))
    optimizer = optim.AdamW([
        {"params": model.student.parameters(), "lr": lr_bb},
        {"params": model.head.parameters(), "lr": lr_hd},
    ])
    print(f"lr backbone {lr_bb:g}, head {lr_hd:g}")

    start_epoch, best_r1, patience = 0, -1.0, 0
    if resume:
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_r1 = ckpt["best_val_r1"]
        patience = ckpt["patience_counter"]
        print(f"Resumed at epoch {start_epoch} (best_val_r1={best_r1:.4f})")

    r1h = run_probe(model, probe, device, t_cfg.batch_size, cfg.num_workers)
    r1g = run_probe(model, probe, device, t_cfg.batch_size, cfg.num_workers,
                    use_head=False)
    print(f"Probe R@1 before training: head {r1h:.4f}, gem {r1g:.4f}")
    wandb.log({"epoch": start_epoch, "val_probe_r1": r1h,
               "val_probe_r1_gem": r1g})

    for epoch in range(start_epoch, t_cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{t_cfg.epochs}")
        train_ds.set_epoch(epoch)
        model.train()

        totals, n_b, cross_frac = np.zeros(4), 0, 0.0
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            reps, xy, group, dataset, floor = flatten_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=enable_amp):
                desc = model(reps)
                loss, active, pd, nd = batch_hard_triplet(
                    desc.float(), xy, group, dataset, floor,
                    float(t_cfg.margin))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at batch {n_b}")
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
            totals += [loss.item(), active, pd, nd]
            cross_frac += float(batch["is_cross"].float().mean())
            n_b += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             active=f"{active:.2f}",
                             pos=f"{pd:.3f}", neg=f"{nd:.3f}")

        tr = totals / max(n_b, 1)
        r1h = run_probe(model, probe, device, t_cfg.batch_size,
                        cfg.num_workers)
        r1 = run_probe(model, probe, device, t_cfg.batch_size,
                       cfg.num_workers, use_head=False)   # selection metric
        print(f"Epoch {epoch + 1} - loss {tr[0]:.4f}, active {tr[1]:.2f}, "
              f"pos_d {tr[2]:.3f}, neg_d {tr[3]:.3f}, "
              f"probe R@1 head {r1h:.4f} / gem {r1:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": tr[0],
                   "active_triplets": tr[1], "pos_dist": tr[2],
                   "neg_dist": tr[3], "val_probe_r1": r1h,
                   "val_probe_r1_gem": r1,
                   "cross_pair_frac": cross_frac / max(n_b, 1)})

        if r1 > best_r1:
            best_r1, patience = r1, 0
            torch.save(model.state_dict(), best_path)
            print(f"New best GEM probe R@1: {r1:.4f} -> {best_path}")
        else:
            patience += 1
        save_checkpoint(last_path, model, optimizer, scaler, epoch,
                        best_r1, patience, wandb.run.id, cfg)
        if cfg.early_stopping > 0 and patience >= cfg.early_stopping:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    final = os.path.join(cfg.checkpoint_dir, f"final_{mode}_{modality}.pth")
    torch.save(model.state_dict(), final)
    print(f"\nDone. Best probe R@1: {best_r1:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()