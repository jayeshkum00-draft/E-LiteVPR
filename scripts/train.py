import os

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import wandb

from dataset import E_LiteVPRDataset
from model import EventViTStudent, GeM

def compute_agfd_loss(student_patches, teacher_patches, teacher_attn):
    error = (student_patches - teacher_patches).pow(2).mean(dim=-1)      # (B, N)
    attn_normalized = teacher_attn / teacher_attn.mean(dim=1, keepdim=True)
    return (error * attn_normalized).mean()


def compute_structural_loss(student_global, teacher_global, temperature=0.05):
    student_norm = F.normalize(student_global, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_global, p=2, dim=-1)

    student_sim = torch.matmul(student_norm, student_norm.T) / temperature
    teacher_sim = torch.matmul(teacher_norm, teacher_norm.T) / temperature

    return F.kl_div(
        F.log_softmax(student_sim, dim=-1),
        F.softmax(teacher_sim, dim=-1),
        reduction='batchmean',
    )


def compute_losses(model_out, teacher_patches, teacher_attn, teacher_gem,
                   use_agfd, structural_weight, temperature):
    student_patches, student_global = model_out

    with torch.no_grad():
        teacher_global = teacher_gem(teacher_patches)

    if use_agfd:
        patch_loss = compute_agfd_loss(student_patches, teacher_patches, teacher_attn)
    else:
        patch_loss = F.mse_loss(student_patches, teacher_patches)

    structural_loss = compute_structural_loss(student_global, teacher_global, temperature)
    loss = patch_loss + structural_weight * structural_loss
    return loss, patch_loss, structural_loss

# Day/night weighted sampler

def build_day_night_sampler(dataset, night_seqs):
    """Balance day vs night at the sample level.

    Reads the per-sample sequence name straight from dataset.pairs (already
    derived with the shared key function inside the dataset).
    """
    seqs = [pair['sequence'] for pair in dataset.pairs]

    night_seqs = set(night_seqs)
    is_night = torch.tensor([s in night_seqs for s in seqs], dtype=torch.bool)

    n_night = int(is_night.sum())
    n_day = len(seqs) - n_night
    if n_night == 0 or n_day == 0:
        raise ValueError(
            f"Day/night sampler degenerate: {n_day} day / {n_night} night samples. "
            "Check night_sequences against the train split."
        )

    weights = torch.where(
        is_night,
        torch.tensor(1.0 / n_night),
        torch.tensor(1.0 / n_day),
    ).double()

    print(f"Day/night sampler: {n_day} day / {n_night} night samples "
          f"(night weight x{n_day / n_night:.2f})")

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_loss,
                    patience_counter, wandb_run_id, cfg):
    tmp_path = path + '.tmp'
    torch.save({
        'epoch': epoch,  # last COMPLETED epoch (0-indexed)
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'wandb_run_id': wandb_run_id,
        'cfg': OmegaConf.to_container(cfg, resolve=True),
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic: never leaves a half-written checkpoint


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler is not None and ckpt.get('scaler_state') is not None:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt

def train_epoch(model, dataloader, optimizer, scaler, teacher_gem,
                use_agfd, structural_weight, temperature, device, enable_amp):
    model.train()
    teacher_gem.eval()

    totals = torch.zeros(3)
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, teacher_patches, teacher_attn, _ts in pbar:
        images = images.to(device, non_blocking=True)
        teacher_patches = teacher_patches.to(device, non_blocking=True)
        teacher_attn = teacher_attn.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', enabled=enable_amp):
            model_out = model(images)
            loss, patch_loss, structural_loss = compute_losses(
                model_out, teacher_patches, teacher_attn, teacher_gem,
                use_agfd, structural_weight, temperature)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at batch {num_batches}: "
                f"loss={loss.item()}, patch={patch_loss.item()}, "
                f"struct={structural_loss.item()}"
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        totals += torch.tensor([loss.item(), patch_loss.item(), structural_loss.item()])
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'patch': f"{patch_loss.item():.4f}",
            'struct': f"{structural_loss.item():.4f}",
        })

    return (totals / num_batches).tolist()


@torch.no_grad()
def validate_epoch(model, dataloader, teacher_gem,
                   use_agfd, structural_weight, temperature, device):
    model.eval()
    teacher_gem.eval()

    totals = torch.zeros(3)
    num_batches = 0

    for images, teacher_patches, teacher_attn, _ts in tqdm(dataloader, desc="Validation"):
        images = images.to(device, non_blocking=True)
        teacher_patches = teacher_patches.to(device, non_blocking=True)
        teacher_attn = teacher_attn.to(device, non_blocking=True)

        model_out = model(images)
        loss, patch_loss, structural_loss = compute_losses(
            model_out, teacher_patches, teacher_attn, teacher_gem,
            use_agfd, structural_weight, temperature)

        totals += torch.tensor([loss.item(), patch_loss.item(), structural_loss.item()])
        num_batches += 1

    return (totals / num_batches).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)   # covers the WeightedRandomSampler's global RNG too

    device = cfg.device
    print(f"Using device: {device}")

    mode = cfg.training.name
    enable_amp = bool(cfg.training.get('enable_amp', True)) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if enable_amp else None

    # checkpoint / resume setup (before wandb.init so run id survives)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    last_path = os.path.join(cfg.checkpoint_dir, f"last_{mode}_{cfg.data.modality}.pth")
    best_path = os.path.join(cfg.checkpoint_dir, f"best_{mode}_{cfg.data.modality}.pth")

    resume = cfg.get('resume', True) and os.path.exists(last_path)
    prev_run_id = None
    if resume:
        # peek at run id only; full load happens after model/optimizer exist
        prev_run_id = torch.load(last_path, map_location='cpu').get('wandb_run_id')
        print(f"Resuming from {last_path} (wandb run: {prev_run_id})")

    wandb.init(
        project=cfg.project_name,
        id=prev_run_id,
        resume="allow",
        name=f"{mode}_{cfg.data.modality}_bs{cfg.training.batch_size}_ep{cfg.training.epochs}",
        config={
            "mode": mode,
            "modality": cfg.data.modality,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "num_patches": cfg.model.num_patches,
            "teacher_dim": cfg.model.teacher_dim,
            "structural_weight": cfg.training.structural_loss_weight,
            "temperature": cfg.training.temperature,
            "enable_amp": enable_amp,
        },
    )

    print("Initializing datasets...")
    train_dataset = E_LiteVPRDataset(
        root=cfg.datasets.root_dir,
        features_dir=cfg.datasets.output_dir,
        event_type=cfg.data.modality,           # 'histogram' or 'voxel'
        sequences=list(cfg.datasets.train_seq_list),
        pair_stride=cfg.datasets.get('pair_stride', 1),
    )
    val_dataset = E_LiteVPRDataset(
        root=cfg.datasets.root_dir,
        features_dir=cfg.datasets.output_dir,
        event_type=cfg.data.modality,
        sequences=list(cfg.datasets.val_seq_list),
        pair_stride=1,                          
    )
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")

    # whitelist-assert: a typo'd night seq would silently count as day
    night_seqs = list(cfg.datasets.night_sequences)
    known = set(cfg.datasets.train_seq_list) | set(cfg.datasets.val_seq_list)
    unknown = set(night_seqs) - known
    if unknown:
        raise ValueError(
            f"night_sequences contains names not in train/val lists: {sorted(unknown)}"
        )

    sampler = build_day_night_sampler(train_dataset, night_seqs)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,            # sampler and shuffle are mutually exclusive
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,             # structural KL is batch-relational; avoid tiny last batch
        persistent_workers=cfg.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    print("Initializing model...")
    model = EventViTStudent(
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,   # must be 576 for the 384/ViT-L16 cache
        img_size=cfg.model.img_hw[0],        # must be 384
        in_channels=cfg.data.input_channels,
    ).to(device)

    teacher_gem = GeM(p=3.0).to(device)
    for param in teacher_gem.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    if resume:
        ckpt = load_checkpoint(last_path, model, optimizer, scaler, device)
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        patience_counter = ckpt['patience_counter']
        print(f"Resumed at epoch {start_epoch} "
              f"(best_val_loss={best_val_loss:.6f}, patience={patience_counter})")

    print(f"Starting train with {mode} mode + structural loss")

    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")

        train_loss, train_patch, train_struct = train_epoch(
            model, train_dataloader, optimizer, scaler, teacher_gem,
            cfg.training.attention_guided, cfg.training.structural_loss_weight,
            cfg.training.temperature, device, enable_amp)

        val_loss, val_patch, val_struct = validate_epoch(
            model, val_dataloader, teacher_gem,
            cfg.training.attention_guided, cfg.training.structural_loss_weight,
            cfg.training.temperature, device)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.6f} "
              f"(patch: {train_patch:.6f}, struct: {train_struct:.6f})")
        print(f"Val Loss: {val_loss:.6f} "
              f"(patch: {val_patch:.6f}, struct: {val_struct:.6f})")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_patch_loss": train_patch,
            "train_structural_loss": train_struct,
            "val_loss": val_loss,
            "val_patch_loss": val_patch,
            "val_structural_loss": val_struct,
            "bn_running_mean": model.input_norm.running_mean.mean().item(),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path) 
            print(f"New best model saved with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        save_checkpoint(last_path, model, optimizer, scaler, epoch,
                        best_val_loss, patience_counter, wandb.run.id, cfg)

        if cfg.early_stopping > 0 and patience_counter >= cfg.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_path = os.path.join(cfg.checkpoint_dir, f"final_{mode}_{cfg.data.modality}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()