"""Phase 1, MegaLoc teacher: structural KL + bank-relational KL, + pooling flag.

WHAT CHANGED FROM train_megaloc.py, AND WHY
-------------------------------------------
`train_megaloc.py` added InfoNCE against a 4096-row negative bank. Measured on
Brisbane (2026-08-05), R@1 at L=1 vs the KL-only control from the same launch:

    sunset2  98.41 -> 82.99      morning  76.78 -> 44.76
    daytime  67.48 -> 25.11      sunrise  83.41 -> 47.09
    night     7.52 ->  2.36

Uniform destruction, worst on the pairs FURTHEST from the reference condition.
That is the signature of the target distribution, not the weight: cross-entropy
asserts a ONE-HOT label -- this frame's own teacher descriptor is the positive,
all 4096 sampled rows are negatives. The `+/-32`-row exclusion only removed
temporal neighbours inside the anchor's own sequence, so every genuine same-
place frame from another traversal or a loop closure was labelled a negative
and actively pushed away. Re-running at a smaller `infonce_weight` would not
fix a wrong label.

The target set was right. The bank spans day and night across DSEC and M3ED,
which is exactly the coverage the B x B relational KL lacks: blocks are
contiguous and therefore single-condition (`block_sampler.py:44-48`), so a day
view and a night view of one place are essentially never co-sampled and the
loss never states that MegaLoc puts them close. MegaLoc demonstrably does --
`probe_teacher_invariance.py` on the M3ED day/night route pairs gives trace
inlier 0.947 / rho 0.796 on penno_small_loop against a control ceiling of
0.427 / 0.228, 3 of 5 routes clearing on both. The teacher's constraint exists;
nothing samples it.

So: keep the bank, keep the head, replace the one-hot with a SOFT target.

    student-vs-anchors similarity profile  ~=  teacher-vs-anchors profile

A same-place anchor scores high in the teacher profile, so the student is
pulled TOWARD it rather than away. No exclusion window -- under a soft target
scoring high on a same-place anchor is the desired behaviour, and masking it
would reintroduce the exact defect above. Each frame is now related to 4096
anchors spanning both conditions instead of 31 same-condition neighbours.

TWO AXES, ONE FLAG EACH
-----------------------
The calibration curve (cross d' 1.83 -> 98.4%, 1.37 -> 61.2%, 0.44 -> 7.7%)
puts night at cross d' ~1.4 for ~60% at L=1. Night's WITHIN-traverse d' is 1.52
and its retention (cross/within) is 0.29, so reaching 1.4 needs retention ~0.92
-- better than sunset2's 0.86 -- which is not reachable by retention alone.
Within d' must rise too. Those are different mechanisms:

  objective axis    `+bank.bankkl_weight`   raises RETENTION
  aggregation axis  `+bank.pooling`         raises WITHIN d'
                    (see model_gem_signed.py for the clamp mechanism)

They are independent, so run one flag at a time first, then together.

STANDARDISE BEFORE THE TEMPERATURE
----------------------------------
`bankkl_standardize` z-scores each row of both similarity profiles before the
softmax, and `bankkl_temperature` defaults to 1.0 accordingly. This is not
decoration. tau was shipped scale-free once (one shared 0.05 gave MegaLoc
gradient norm 2.1e-09 -- dead) and `teacher_mix` repeated the same class of bug
(mix=0.5 was a 75.7% gradient share). Student and teacher cosine ranges are
model-dependent and differ by more than the temperature; z-scoring removes that
degree of freedom. `first-batch scales:` prints both losses AND both gradient
norms on the first real batch so the weighting is measured, never assumed.

WHAT IS SHIPPED
---------------
The `nn.Linear(teacher_dim, 8448)` head exists to put the student in the
teacher's width for the loss only. It is trained with the model, clipped with
the model, and DISCARDED at save time -- only `model.state_dict()` is written
(SimCLR/BYOL arrangement, the representation beneath the head is the deployed
one). The teacher is never projected, so no basis is fit to DSEC/M3ED and
zero-shot transfer to Brisbane/NSAVP is untouched. teacher_dim stays 1024;
setting it to 8448 would widen `model.py:52`'s PER-PATCH projection to
(B, 576, 8448) = 622 MB a forward and OOM a T4.

With `+bank.pooling=clamp` (default) the checkpoint is architecturally
identical to every earlier run and loads in the stock evaluators. With `signed`
or `mean` the pooler registers as `pool` rather than `gem`, so the stock strict
`load_state_dict` RAISES instead of silently evaluating through the old
pooling -- use `evaluate_brisbane_gem.py`.

READ THE RESULT ON cross-condition d' AND retention (probe_all.py), NOT on the
day-night sequence mean. Two runs of one recipe (`ckpt_blk_m3ed` vs `ckpt_kl`)
differed by 0.15 points at L=1 and 5.6 points on that mean -- it cannot resolve
these changes. Baseline: eff rank 1.10/1024, night cross d' 0.4412,
retention 0.29 night / 0.68 daytime. Develop on DAYTIME retention.

CHECKPOINT SELECTION IS KNOWN-BROKEN: val is DSEC-only and day-dominated
(`m3ed.yaml val_seq_list: []`), and `last` beat `best` on day-night in both
expert runs. Both are always written. Evaluate both.

COMMANDS (Kaggle paths -- see the `kaggle-run-layout` memory note)
-----------------------------------------------------------------
Common prefix:

  HYDRA_MAIN_MODULE=__main__ python scripts/train_bankkl.py \
    datasets=dsec datasets@datasets_extra=m3ed \
    datasets.root_dir=$DSEC_ROOT   datasets.output_dir=$DSEC_FEATS \
    datasets_extra.root_dir=$M3ED_ROOT datasets_extra.output_dir=$M3ED_FEATS \
    training.patch_loss_weight=0.0 training.temperature=1.0 \
    +bank.teacher_dir=$MEGALOC_DIR

Then one flag per cell:

  A  aggregation only   +bank.bankkl_weight=0.0 +bank.pooling=signed \
                        checkpoint_dir=/kaggle/working/ckpt_signed
  A' the p=1 control    +bank.bankkl_weight=0.0 +bank.pooling=mean \
                        checkpoint_dir=/kaggle/working/ckpt_mean
  B  objective only     +bank.bankkl_weight=1.0 +bank.pooling=clamp \
                        checkpoint_dir=/kaggle/working/ckpt_bankkl
  AB both              +bank.bankkl_weight=1.0 +bank.pooling=signed \
                        checkpoint_dir=/kaggle/working/ckpt_signed_bankkl

Control already run: `ckpt_kl` (83.85 / 36.53, night L=1 7.52) is
`+bank.bankkl_weight=0.0 +bank.pooling=clamp`, i.e. this script with both flags
off. No need to repeat it.

Score with (add nothing for pooling=clamp):

  rm -rf /kaggle/working/brisbane_cache
  python scripts/evaluate_brisbane_gem.py +gem.pooling=signed \
    +gem.entry=probe_all datasets=brisbane datasets.root=$BRISBANE \
    phase1_weights=/kaggle/working/<ckpt>/last_phase1_histogram.pth \
    data.modality=histogram datasets.dt=1.0
"""

import math
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from block_sampler import BlockSampler
from model_gem_signed import build_student
import train as T
import train_megaloc as M


_DEFAULTS = {
    "teacher_dir": None,
    "bankkl_weight": 1.0,
    "bankkl_temperature": 1.0,
    "bankkl_anchors": 4096,
    "bankkl_standardize": True,
    "pooling": "clamp",
    "use_block_sampler": True,
    "block_size": 8,
    "night_frac": 0.5,
    "use_scheduler": True,
    "warmup_frac": 0.03,
    "min_lr_frac": 0.05,
}


def _b(cfg, key):
    v = OmegaConf.select(cfg, f"bank.{key}")
    return _DEFAULTS[key] if v is None else v


# ------------------------------------------------------------------ loss ---
def bankkl_loss(student_global, teacher_global, bank, n_anchors, tau,
                standardize, head):
    """Match the student's similarity profile over shared anchors to the
    teacher's.

    Both sides are compared against the SAME `n_anchors` rows drawn from the
    cached teacher bank, so the two profiles are commensurable and the target
    is the teacher's own soft distribution -- not a one-hot label. A frame that
    is genuinely the same place as an anchor is high in the teacher profile and
    the student is pulled toward it, which is precisely what the InfoNCE
    formulation got backwards.

    Anchors are shared across the batch (one draw per step): the head output is
    (B, 8448) and the anchor block is (K, 8448), so one 138 MB fp32 gather
    serves the whole batch -- the same footprint the InfoNCE run already ran at
    on a T4.

    Resampled every step, so an epoch covers far more of the bank than any one
    batch sees.
    """
    z_s = F.normalize(head(student_global.float()), p=2, dim=-1)
    z_t = F.normalize(teacher_global.float(), p=2, dim=-1)

    idx = torch.randint(bank.shape[0], (n_anchors,), device=bank.device)
    anchors = bank[idx].float()                                  # (K, D), L2'd
    s_sim = z_s @ anchors.T                                      # (B, K)
    t_sim = z_t @ anchors.T                                      # (B, K)

    if standardize:
        # Per ROW: each frame's profile is z-scored against its own anchors, so
        # a student whose cosines live in a narrower band than the teacher's is
        # not penalised for the band, only for the ordering and the shape.
        s_sim = ((s_sim - s_sim.mean(1, keepdim=True))
                 / s_sim.std(1, keepdim=True).clamp(min=1e-6))
        t_sim = ((t_sim - t_sim.mean(1, keepdim=True))
                 / t_sim.std(1, keepdim=True).clamp(min=1e-6))

    return F.kl_div(F.log_softmax(s_sim / tau, dim=-1),
                    F.softmax(t_sim / tau, dim=-1),
                    reduction="batchmean")


def _report_scales(struct, bkl, student_global):
    """Both losses and both gradient norms, once, on the first real batch.

    tau and teacher_mix were each shipped assuming two terms are comparable in
    scale and each was wrong. The loss VALUES do not settle this -- a KL over
    4096 anchors and a KL over 31 batch neighbours have different natural
    magnitudes at identical gradient strength.
    """
    parts = []
    for name, term in (("structural", struct), ("bankkl", bkl)):
        g = torch.autograd.grad(term, student_global, retain_graph=True,
                                allow_unused=True)[0]
        gn = float("nan") if g is None else g.norm().item()
        parts.append(f"{name} loss {term.item():.4f} |grad| {gn:.3e}")
    print("\n  first-batch scales: " + "   ".join(parts))


# ------------------------------------------------------------------ loop ---
def run_epoch(model, head, loader, optimizer, scaler, device, cfg, bank_t,
              sched, train_mode, state):
    bank = bank_t[0]
    struct_w = float(cfg.training.structural_loss_weight)
    bkl_w = float(_b(cfg, "bankkl_weight"))
    tau_kl = float(cfg.training.temperature)
    tau_bkl = float(_b(cfg, "bankkl_temperature"))
    n_anch = int(_b(cfg, "bankkl_anchors"))
    std = bool(_b(cfg, "bankkl_standardize"))
    amp = bool(state["amp"])
    dim = M._BANK["dim"]

    model.train(train_mode)
    if head is not None:
        head.train(train_mode)
    totals, n = torch.zeros(3), 0
    pbar = tqdm(loader, desc="Training" if train_mode else "Validation")
    for images, packed, _attn, _ts in pbar:
        images = images.to(device, non_blocking=True)
        packed = packed.to(device, non_blocking=True).squeeze(1)   # (B, D+2)
        # The (seq id, row) tail is unused here -- there is no exclusion mask
        # under a soft target. MegaLocDataset is reused unchanged so the cache
        # row-alignment contract it asserts stays byte-identical to the
        # validated version.
        teacher_global = packed[:, :dim]

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            # ONLY the forward runs under autocast. Calling .float() on the
            # inputs inside an autocast region is NOT enough -- autocast
            # re-casts the matmuls themselves, so the 8448-d anchor products
            # would still run in fp16.
            with torch.autocast(device_type="cuda", enabled=amp):
                _patches, student_global = model(images)
            struct = (M.structural_loss(student_global, teacher_global, tau_kl)
                      if struct_w else student_global.sum() * 0.0)
            bkl = (bankkl_loss(student_global, teacher_global, bank, n_anch,
                               tau_bkl, std, head)
                   if bkl_w else student_global.sum() * 0.0)
            loss = struct_w * struct + bkl_w * bkl

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: struct={struct.item()} "
                               f"bankkl={bkl.item()}")

        if train_mode:
            if not state["reported"] and struct_w and bkl_w:
                _report_scales(struct, bkl, student_global)
                state["reported"] = True
            # Clip EVERY optimised parameter, not just the model's: the head is
            # in the optimiser too, and leaving it unclipped lets it take
            # unbounded steps while the backbone is bounded -- the head would
            # absorb the alignment instead of the descriptor.
            clip = [p for g in optimizer.param_groups for p in g["params"]]
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip, 1.0)
                optimizer.step()
            if sched is not None:
                sched.step()

        totals += torch.tensor([loss.item(), struct.item(), bkl.item()])
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "kl": f"{struct.item():.4f}",
                          "bkl": f"{bkl.item():.4f}"})
    return (totals / max(n, 1)).tolist()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    teacher_dir = _b(cfg, "teacher_dir")
    if not teacher_dir:
        raise SystemExit(
            "train_bankkl.py requires +bank.teacher_dir=<dir written by "
            "cache_megaloc.py>. Note the namespace is +bank.*, not +mega.* -- "
            "read this file's docstring for the current commands.")

    pooling = str(_b(cfg, "pooling"))
    torch.manual_seed(cfg.seed)
    # Fail rather than train 30 epochs on CPU: an earlier launch did exactly
    # that, fell back silently and exhausted host RAM instead of raising.
    if not torch.cuda.is_available() and not bool(cfg.get("allow_cpu", False)):
        raise SystemExit(
            "CUDA is not available -- torch.cuda.is_available() is False.\n"
            "Check the Kaggle notebook accelerator is set to GPU (T4 x2) and "
            "that CUDA_VISIBLE_DEVICES names a real device.\n"
            "Pass +allow_cpu=true only to smoke-test the wiring.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.training.get("enable_amp", True)) and device.type == "cuda"

    if float(cfg.training.get("patch_loss_weight", 1.0)):
        raise SystemExit("a cached GLOBAL teacher descriptor has no patch "
                         "target: pass training.patch_loss_weight=0.0")

    dataset_cfgs = T.active_dataset_cfgs(cfg)
    M._BANK["dir"] = str(teacher_dir)

    print("Initializing datasets...")
    probe_ds = T.build_split(dataset_cfgs, "train_seq_list", cfg.data.modality)
    val_probe = T.build_split(dataset_cfgs, "val_seq_list", cfg.data.modality,
                              pair_stride_override=1)
    print(f"train dataset size: {len(probe_ds)}")
    print(f"val dataset size: {len(val_probe)}")

    night_seqs = []
    for dcfg in dataset_cfgs:
        night_seqs.extend(dcfg.get("night_sequences") or [])
    known = set(probe_ds.sequence_names()) | set(val_probe.sequence_names())
    unknown = set(night_seqs) - known
    if unknown:
        raise ValueError(f"night_sequences not in the loaded splits: "
                         f"{sorted(unknown)}")

    # Bank is built from TRAIN sequences only, so a val place can never become
    # an anchor. Row counts are asserted against frames.txt inside build_bank.
    print("Building anchor bank...")
    bank_t = M.build_bank(teacher_dir, probe_ds, val_probe, device)
    bank, _bseq, _bpos, seq_id = bank_t
    M._BANK["seq_id"] = seq_id
    M._BANK["dim"] = int(bank.shape[1])
    print(f"  bank: {bank.shape[0]} descriptors x {bank.shape[1]}-d "
          f"({bank.element_size() * bank.nelement() / 1e6:.0f} MB fp16) "
          f"over {len(probe_ds.sequence_names())} train sequences "
          f"({len(probe_ds)} pairs)")

    # rebuild both splits against the cached-descriptor dataset class
    orig = T.E_LiteVPRDataset
    T.E_LiteVPRDataset = M.MegaLocDataset
    try:
        train_ds = T.build_split(dataset_cfgs, "train_seq_list", cfg.data.modality)
        val_ds = T.build_split(dataset_cfgs, "val_seq_list", cfg.data.modality,
                               pair_stride_override=1)
    finally:
        T.E_LiteVPRDataset = orig

    use_blocks = bool(_b(cfg, "use_block_sampler"))
    block = int(_b(cfg, "block_size"))
    if use_blocks:
        if int(cfg.training.batch_size) % block:
            raise SystemExit(f"batch_size={cfg.training.batch_size} must be a "
                             f"multiple of block_size={block}")
        sampler = BlockSampler(train_ds, night_seqs, block=block,
                               night_frac=float(_b(cfg, "night_frac")))
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
    model = build_student(
        pooling=pooling,
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,
        img_size=cfg.model.img_hw[0],
        in_channels=cfg.data.input_channels,
    ).to(device)

    # Loss-only projection head, trained with the model and DISCARDED at save
    # time -- only model.state_dict() is written.
    head = None
    if float(_b(cfg, "bankkl_weight")):
        head = torch.nn.Linear(int(cfg.model.teacher_dim),
                               M._BANK["dim"]).to(device)
        print(f"  bank-KL head: {cfg.model.teacher_dim} -> {M._BANK['dim']} "
              f"({sum(p.numel() for p in head.parameters()) / 1e6:.1f}M params, "
              f"not saved)")

    lr_base = cfg.training.get("lr_base_batch_size", cfg.training.batch_size)
    lr = cfg.training.learning_rate * (cfg.training.batch_size / lr_base) ** 0.5
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    optimizer = optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    sched = None
    if bool(_b(cfg, "use_scheduler")):
        total = len(train_dl) * int(cfg.training.epochs)
        warm = max(1, int(round(float(_b(cfg, "warmup_frac")) * total)))
        floor = float(_b(cfg, "min_lr_frac"))

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        sched = optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        print(f"  LR schedule: {warm} warmup then cosine to {floor:g}x over "
              f"{total} steps")

    w_kl = float(cfg.training.structural_loss_weight)
    w_bkl = float(_b(cfg, "bankkl_weight"))
    t_kl = float(cfg.training.temperature)
    t_bkl = float(_b(cfg, "bankkl_temperature"))
    k_bkl = int(_b(cfg, "bankkl_anchors"))
    z_bkl = bool(_b(cfg, "bankkl_standardize"))
    print("=" * 70)
    print(f"train_bankkl: teacher=megaloc({M._BANK['dim']}-d)  "
          f"pooling={pooling}  blocks={use_blocks}"
          + (f" (block={block}, night_frac={float(_b(cfg, 'night_frac'))})"
             if use_blocks else ""))
    print(f"  loss = {w_kl:g} * KL(mask_diag,standardize,tau={t_kl:g})"
          f"  +  {w_bkl:g} * bankKL(tau={t_bkl:g}, K={k_bkl}, "
          f"standardize={z_bkl})")
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

    state = {"amp": amp, "reported": False}
    best_val, patience = float("inf"), 0

    for epoch in range(int(cfg.training.epochs)):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        tr = run_epoch(model, head, train_dl, optimizer, scaler, device, cfg,
                       bank_t, sched, True, state)
        va = run_epoch(model, head, val_dl, optimizer, None, device, cfg,
                       bank_t, None, False, state)
        print(f"Train Loss: {tr[0]:.6f} (kl: {tr[1]:.6f}, bkl: {tr[2]:.6f})")
        print(f"Val   Loss: {va[0]:.6f} (kl: {va[1]:.6f}, bkl: {va[2]:.6f})")
        wandb.log({"epoch": epoch + 1, "train_loss": tr[0], "train_kl": tr[1],
                   "train_bkl": tr[2], "val_loss": va[0], "val_kl": va[1],
                   "val_bkl": va[2], "lr": optimizer.param_groups[0]["lr"]})

        if va[0] < best_val:
            best_val, patience = va[0], 0
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with val_loss: {va[0]:.6f}")
        else:
            patience += 1

        # `last` is written EVERY epoch: `last` beat `best` on day-night in
        # both expert runs, because val is DSEC-only and day-dominated.
        torch.save(model.state_dict(), last_path)
        T.save_checkpoint(os.path.join(cfg.checkpoint_dir, "resume.pth"), model,
                          optimizer, scaler, epoch, best_val, patience,
                          wandb.run.id, cfg)

        if cfg.early_stopping > 0 and patience >= cfg.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nDone. best_val_loss={best_val:.6f}")
    print(f"  {best_path}\n  {last_path}")
    print("  EVALUATE BOTH -- `last` beat `best` on day-night in both expert runs.")
    wandb.finish()


if __name__ == "__main__":
    main()
