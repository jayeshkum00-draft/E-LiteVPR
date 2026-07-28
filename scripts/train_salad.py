"""Phase 1 distillation with a SALAD aggregator instead of GeM.

Identical objective and data pipeline to train.py -- every helper is imported
from it, so the two runs differ ONLY in the aggregation head. The structural
KL compares B x B cosine-similarity matrices, so the student's 8448-d SALAD
descriptor and the teacher's 1024-d GeM target are directly comparable
without any place supervision.

Run (same overrides as train.py):
    python scripts/train_salad.py datasets=dsec
    python scripts/train_salad.py datasets=dsec model=student_salad \
        training.batch_size=32

Checkpoints are written as *_salad_*.pth so they never collide with the GeM
run's files.
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from model import GeM
from model_salad import EventViTStudentSALAD
from train import (
    active_dataset_cfgs,
    build_split,
    build_day_night_sampler,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    validate_epoch,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    device = cfg.device
    print(f"Using device: {device}")

    mode = f"{cfg.training.name}_salad"
    enable_amp = bool(cfg.training.get('enable_amp', True)) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if enable_amp else None

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    last_path = os.path.join(cfg.checkpoint_dir,
                             f"last_{mode}_{cfg.data.modality}.pth")
    best_path = os.path.join(cfg.checkpoint_dir,
                             f"best_{mode}_{cfg.data.modality}.pth")
    resume = bool(cfg.get('resume', True)) and os.path.exists(last_path)

    wandb_run_id = None
    if resume:
        wandb_run_id = torch.load(last_path, map_location='cpu').get('wandb_run_id')

    wandb.init(
        project=cfg.project_name,
        name=f"{mode}_{cfg.data.modality}_bs{cfg.training.batch_size}"
             f"_ep{cfg.training.epochs}",
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config={
            "aggregator": "salad",
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "structural_weight": cfg.training.structural_loss_weight,
            "temperature": cfg.training.temperature,
            "enable_amp": enable_amp,
            "num_clusters": cfg.model.get('num_clusters', 64),
            "cluster_dim": cfg.model.get('cluster_dim', 128),
            "token_dim": cfg.model.get('token_dim', 256),
        },
    )

    print("Initializing datasets...")
    dataset_cfgs = active_dataset_cfgs(cfg)
    if len(dataset_cfgs) > 1:
        print(f"Joint training over {len(dataset_cfgs)} dataset sources: "
              f"{[str(d.root_dir) for d in dataset_cfgs]}")

    train_dataset = build_split(dataset_cfgs, 'train_seq_list', cfg.data.modality)
    val_dataset = build_split(dataset_cfgs, 'val_seq_list', cfg.data.modality,
                              pair_stride_override=1)
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")

    night_seqs = []
    for dcfg in dataset_cfgs:
        night_seqs.extend(dcfg.get('night_sequences') or [])
    known = set(train_dataset.sequence_names()) | set(val_dataset.sequence_names())
    unknown = set(night_seqs) - known
    if unknown:
        raise ValueError(
            f"night_sequences contains names not present in the loaded train/val "
            f"splits: {sorted(unknown)}"
        )

    sampler = build_day_night_sampler(train_dataset, night_seqs)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,             # structural KL is batch-relational
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

    print("Initializing SALAD student...")
    model = EventViTStudentSALAD(
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,
        img_size=cfg.model.img_hw[0],
        in_channels=cfg.data.input_channels,
        num_clusters=cfg.model.get('num_clusters', 64),
        cluster_dim=cfg.model.get('cluster_dim', 128),
        token_dim=cfg.model.get('token_dim', 256),
        sinkhorn_iters=cfg.model.get('sinkhorn_iters', 3),
    ).to(device)

    # teacher target geometry stays GeM -- unchanged from train.py
    teacher_gem = GeM(p=3.0).to(device)
    for param in teacher_gem.parameters():
        param.requires_grad = False

    lr_base_batch_size = cfg.training.get('lr_base_batch_size', cfg.training.batch_size)
    scaled_lr = cfg.training.learning_rate * (cfg.training.batch_size / lr_base_batch_size) ** 0.5
    if scaled_lr != cfg.training.learning_rate:
        print(f"Scaling lr {cfg.training.learning_rate:.6g} -> {scaled_lr:.6g} "
              f"(batch_size={cfg.training.batch_size}, lr_base_batch_size={lr_base_batch_size})")
    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr)

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
            cfg.training.temperature, device, enable_amp,
            patch_weight=cfg.training.get('patch_loss_weight', 1.0))

        val_loss, val_patch, val_struct = validate_epoch(
            model, val_dataloader, teacher_gem,
            cfg.training.attention_guided, cfg.training.structural_loss_weight,
            cfg.training.temperature, device,
            patch_weight=cfg.training.get('patch_loss_weight', 1.0))

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
            "salad_dust_bin": model.salad.dust_bin.item(),
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

    final_path = os.path.join(cfg.checkpoint_dir,
                              f"final_{mode}_{cfg.data.modality}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
