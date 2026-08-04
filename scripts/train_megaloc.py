"""Phase 1 against a single MegaLoc teacher: structural KL + InfoNCE.

WHY THIS EXISTS (measured 2026-08-04 -- reproduce with scripts/probe_all.py)
---------------------------------------------------------------------------
1. The invariance is already in the TEACHER. `probe_teacher_invariance.py` on
   the M3ED day/night route pairs: MegaLoc matches night RGB to day RGB with
   trace inlier 0.947 / rho 0.796 on penno_small_loop, against a control
   ceiling of 0.427 / 0.228 -- 3 of 5 routes clear it on both statistics.
   DINOv3-GeM does not (0.842 against its OWN control of 0.737, i.e. 1.14x
   versus MegaLoc's 4.5x) and its within-sequence cosine is 0.945-0.969, so it
   is a COLLAPSED target. => single teacher. teacher_b is dropped, not optional.

2. It does not reach the student. `diagnose_descriptor_collapse.py`: night
   cross-condition d' 0.4412 -> R@1 7.67% at L=1, while night WITHIN-traverse
   d' is 1.52. Retention (cross/within) is 0.86 sunset2 / 0.68 daytime /
   0.29 night. The representation encodes night places perfectly well; it just
   does not land near the day descriptor of the same place.

3. The old objective cannot transfer it. A B x B relational KL constrains
   ORDERING ONLY, and with the block sampler every batch is a single condition
   (block_sampler.py:44-48), so the day-night part of the teacher's geometry is
   never exercised. A collapsed embedding reproduces the teacher's ordering
   perfectly -- the loss is structurally blind to the collapse it permits.

InfoNCE targets exactly that: the positive is the frame's OWN cached MegaLoc
descriptor, pinning the student to the teacher's ABSOLUTE position, which is
where the invariance lives. Negatives are thousands of other frames from the
same cache -- no "which of these 4 clips am I in" shortcut satisfies a 4096-way
discrimination, and separating that many requires real margin, so it penalises
collapse directly.

NOTHING IS FIT TO THIS CORPUS, AND THE SHIPPED MODEL IS UNCHANGED. Alignment
uses a learned `nn.Linear(teacher_dim, 8448)` head on the STUDENT side only,
trained with the model and DISCARDED at save time -- the SimCLR/BYOL
arrangement, where the representation beneath the head is the one deployed.
The teacher is never projected, so no basis is fit to DSEC/M3ED and zero-shot
transfer is untouched.

teacher_dim stays 1024, as in every prior run. Setting it to 8448 to match
MegaLoc would widen model.py:51's PER-PATCH projection to (B, 576, 8448) =
622 MB a forward and OOM a T4 -- the descriptor is not the only thing that
scales. Keeping 1024 also means the .pth is architecturally identical to every
earlier checkpoint, so evaluate_brisbane.py and evaluate_nsavp.py load it with
no override at all.

READ THE RESULT ON cross-condition d' AND retention via probe_all.py, NOT on
the day-night sequence mean. Baseline to beat: effective rank 1.10/1024, night
cross d' 0.4412, retention 0.29. Use DAYTIME retention (0.68, R@1 61.20%) as
the development signal -- 8x the signal of night and outside the ~7-10 point
night noise band that makes single-seed night comparisons unreadable.

Sizing, from the calibration curve (d' 1.83 -> 98.4%, 1.37 -> 61.2%,
0.44 -> 7.7%): night needs cross d' ~1.4. At night's current within d' of 1.52
that would demand retention 0.90, better than sunset2's 0.86 -- implausible.
So within d' must rise too, which is the aggregation axis, NOT this one. Expect
this run to move retention; do not expect it to reach 1.4 alone.

CHECKPOINT SELECTION IS KNOWN-BROKEN. Val is DSEC-only and day-dominated
(m3ed.yaml `val_seq_list: []`), and in BOTH day/night expert runs `last` beat
`best` on day-night. This script always writes both. Evaluate both.

COMMANDS (Kaggle paths -- see the `kaggle-run-layout` memory note)
------------------------------------------------------------------
Common prefix:

  HYDRA_MAIN_MODULE=__main__ python scripts/train_megaloc.py \
    datasets=dsec datasets@datasets_extra=m3ed \
    datasets.root_dir=$DSEC_ROOT   datasets.output_dir=$DSEC_FEATS \
    datasets_extra.root_dir=$M3ED_ROOT datasets_extra.output_dir=$M3ED_FEATS \
    training.patch_loss_weight=0.0 training.temperature=1.0 \
    +mega.teacher_dir=$MEGALOC_DIR

Then one flag per cell:

  KL + InfoNCE (main)  +mega.infonce_weight=1.0 \
                       checkpoint_dir=/kaggle/working/ckpt_kl_nce
  InfoNCE only         +mega.infonce_weight=1.0 training.structural_loss_weight=0.0 \
                       checkpoint_dir=/kaggle/working/ckpt_nce
  KL only (control)    +mega.infonce_weight=0.0 \
                       checkpoint_dir=/kaggle/working/ckpt_kl
  no block sampler     +mega.infonce_weight=1.0 +mega.use_block_sampler=false \
                       checkpoint_dir=/kaggle/working/ckpt_kl_nce_noblk

Everything else is held at the settled configuration: mask_diag + standardize
+ temperature=1.0 (the fix that made MegaLoc trainable at all -- one shared
tau=0.05 gave gradient norm 2.1e-09), block_size=8, night_frac=0.5.

Score every checkpoint with:

  rm -rf /kaggle/working/brisbane_cache
  python scripts/probe_all.py datasets=brisbane datasets.root=$BRISBANE \
    phase1_weights=/kaggle/working/<ckpt>/last_phase1_histogram.pth \
    data.modality=histogram datasets.dt=1.0 \
    +probe.megaloc_dir=$MEGALOC_DIR
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from block_sampler import BlockSampler
from dataset import E_LiteVPRDataset
from model import EventViTStudent
import train as T


# Written in the MAIN process before DataLoader workers fork; read inside
# _get_features. A worker cannot write it back, so it must never be lazy.
_BANK = {"dir": None, "seq_id": None, "dim": None}

_DEFAULTS = {
    "teacher_dir": None,
    "infonce_weight": 1.0,
    "infonce_temperature": 0.07,
    "infonce_negatives": 4096,
    "infonce_exclude_window": 32,
    "use_block_sampler": True,
    "block_size": 8,
    "night_frac": 0.5,
    "use_scheduler": True,
    "warmup_frac": 0.03,
    "min_lr_frac": 0.05,
}


def _m(cfg, key):
    v = OmegaConf.select(cfg, f"mega.{key}")
    return _DEFAULTS[key] if v is None else v


# --------------------------------------------------------------- dataset ---
def _open_cache(root, seq, features_dir):
    """mmap <root>/<seq>/megaloc.npy, asserting row alignment with frames.txt.

    cache_megaloc.py:23 calls this "the whole contract": dataset.py maps a pair
    to a cache row through _row_index built from frames.txt, so a misaligned
    cache trains on the wrong targets and every number afterwards is garbage.
    """
    path = Path(root) / seq / "megaloc.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run cache_megaloc.py for this corpus first.")
    arr = np.load(path, mmap_mode="r")
    n_rows = sum(1 for ln in (Path(features_dir) / seq / "frames.txt")
                 .read_text().splitlines() if ln.strip())
    if arr.shape[0] != n_rows:
        raise RuntimeError(
            f"{path} has {arr.shape[0]} rows but {seq}/frames.txt has {n_rows}. "
            f"Re-run cache_megaloc.py for {seq}.")
    return arr


class MegaLocDataset(E_LiteVPRDataset):
    """Serves (1, D+1): the cached MegaLoc descriptor plus its BANK INDEX.

    The index rides in the last column rather than as a fifth tuple element so
    the collate, the (B, 1, D) contract and __init__'s
    `patches.ndim == 2 and attn.ndim == 1` probe are all untouched. float32
    represents integers exactly to 2^24; the corpus is ~21k frames.
    """

    def _get_features(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            self._mmaps[seq] = _open_cache(_BANK["dir"], seq, self.features_dir)
        arr = self._mmaps[seq]
        row = self._row_index[pair["feature_key"]]
        d = torch.from_numpy(np.ascontiguousarray(arr[row]).astype(np.float32))
        # (sequence id, row) rather than one global bank offset: VAL sequences
        # are deliberately absent from the bank, so an offset lookup KeyErrors
        # on the very first val probe. Only the exclusion mask needs these, and
        # it compares ids and rows directly.
        tail = torch.tensor([float(_BANK["seq_id"][seq]), float(row)])
        return torch.cat([d, tail]).unsqueeze(0), torch.zeros(1)


def _features_dirs(ds):
    """seq -> features_dir, across a single dataset or a Concat of sources."""
    parts = getattr(ds, "datasets", None) or [ds]
    return {s: p.features_dir for p in parts for s in p.sequence_names()}


def build_bank(teacher_dir, train_ds, val_ds, device):
    """All train-split MegaLoc descriptors, L2-normalised, fp16 on device.

    21401 x 8448 fp16 is ~345 MB, so it lives on the GPU and a 4096-way
    negative sample costs an index. Also returns per-row sequence id and
    within-sequence position, which is what makes same-place exclusion
    possible.

    Sequence ids are assigned over TRAIN + VAL so a val frame can still be
    compared against the mask, while only train rows enter the bank.

    Row counts are asserted against frames.txt here, not just lazily in
    _open_cache. An earlier version skipped it to stay Concat-safe and built a
    bank of 18245 rows for a 21401-pair corpus without complaining -- a short
    cache must fail at startup, not become a quietly truncated negative pool.
    """
    fdirs = {**_features_dirs(train_ds), **_features_dirs(val_ds)}
    all_seqs = sorted(set(train_ds.sequence_names())
                      | set(val_ds.sequence_names()))
    seq_id = {s: i for i, s in enumerate(all_seqs)}

    descs, seq_ids, positions, short = [], [], [], []
    for seq in sorted(train_ds.sequence_names()):
        path = Path(teacher_dir) / seq / "megaloc.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found -- run cache_megaloc.py for this corpus.")
        arr = np.load(path)
        n_rows = sum(1 for ln in (Path(fdirs[seq]) / seq / "frames.txt")
                     .read_text().splitlines() if ln.strip())
        if arr.shape[0] != n_rows:
            short.append(f"{seq}: cache {arr.shape[0]} vs frames.txt {n_rows}")
            continue
        descs.append(torch.from_numpy(arr.astype(np.float32)))
        seq_ids.append(torch.full((len(arr),), seq_id[seq], dtype=torch.int32))
        positions.append(torch.arange(len(arr), dtype=torch.int32))

    if short:
        raise SystemExit(
            "MegaLoc cache is not row-aligned with frames.txt for "
            f"{len(short)} sequence(s) -- every target from these would be the "
            "wrong frame. Re-run cache_megaloc.py for:\n  "
            + "\n  ".join(short))

    bank = F.normalize(torch.cat(descs), p=2, dim=1).half().to(device)
    return (bank, torch.cat(seq_ids).to(device),
            torch.cat(positions).to(device), seq_id)


# ------------------------------------------------------------------ loss ---
def _drop_diag(sim):
    n = sim.shape[0]
    return sim[~torch.eye(n, dtype=torch.bool, device=sim.device)].view(n, n - 1)


def structural_loss(student_global, teacher_global, temperature):
    """The settled fixed loss: drop the self-term, z-score, tau in sigma units.

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


def infonce_loss(student_global, teacher_global, anchor_seq, anchor_pos, bank,
                 bank_seq, bank_pos, n_neg, tau, window, head=None):
    """Align the student to the teacher's ABSOLUTE position.

    Positive: the frame's own teacher descriptor. Negatives: n_neg rows sampled
    from the cache, shared across the batch.

    The exclusion is not cosmetic. Frames a few rows apart within one sequence
    are the SAME PLACE, so sampling one as a negative teaches the model to push
    apart two views of one place -- the exact opposite of the objective. Any
    negative from the anchor's own sequence within +/- window rows is masked.
    """
    # `head` maps the student's descriptor into the teacher's width for the
    # loss ONLY, and is thrown away at inference (SimCLR/BYOL projection head).
    # The alternative -- model.teacher_dim=8448 -- would widen model.py:51's
    # PER-PATCH projection to (B, 576, 8448) = 622 MB a forward and OOM a T4;
    # every previous run was 1024-d for that reason. Discarding it also keeps
    # the shipped .pth architecturally identical to those runs, so both eval
    # scripts load it unchanged.
    s = student_global.float() if head is None else head(student_global.float())
    z_s = F.normalize(s, p=2, dim=-1)
    z_t = F.normalize(teacher_global.float(), p=2, dim=-1)
    pos = (z_s * z_t).sum(-1, keepdim=True)                        # (B, 1)

    idx = torch.randint(bank.shape[0], (n_neg,), device=bank.device)
    sim = z_s @ bank[idx].float().T                                # (B, K)

    same = anchor_seq[:, None] == bank_seq[idx][None, :]
    near = (anchor_pos[:, None] - bank_pos[idx][None, :]).abs() <= window
    sim = sim.masked_fill(same & near, float("-inf"))

    logits = torch.cat([pos, sim], dim=1) / tau
    return F.cross_entropy(
        logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def _report_scales(struct, nce, student_global):
    """Print both terms and their gradient norms once, on the first real batch.

    tau and teacher_mix were BOTH shipped assuming two loss terms are
    comparable in scale, and both were wrong (mix=0.5 turned out to be a 75.7%
    gradient share). Weighting two losses without measuring is how that
    happened twice; this makes it visible on the first step instead of after
    a full run.
    """
    parts = []
    for name, term in (("structural", struct), ("infonce", nce)):
        g = torch.autograd.grad(term, student_global, retain_graph=True,
                                allow_unused=True)[0]
        gn = float("nan") if g is None else g.norm().item()
        parts.append(f"{name} loss {term.item():.4f} |grad| {gn:.3e}")
    print("\n  first-batch scales: " + "   ".join(parts))


# ------------------------------------------------------------------ loop ---
def run_epoch(model, head, loader, optimizer, scaler, device, cfg, bank_t,
              sched, train_mode, state):
    bank, bank_seq, bank_pos, _ = bank_t
    struct_w = float(cfg.training.structural_loss_weight)
    nce_w = float(_m(cfg, "infonce_weight"))
    tau_kl = float(cfg.training.temperature)
    tau_nce = float(_m(cfg, "infonce_temperature"))
    n_neg = int(_m(cfg, "infonce_negatives"))
    window = int(_m(cfg, "infonce_exclude_window"))
    amp = bool(state["amp"])
    dim = _BANK["dim"]

    model.train(train_mode)
    if head is not None:
        head.train(train_mode)
    totals, n = torch.zeros(3), 0
    pbar = tqdm(loader, desc="Training" if train_mode else "Validation")
    for images, packed, _attn, _ts in pbar:
        images = images.to(device, non_blocking=True)
        packed = packed.to(device, non_blocking=True).squeeze(1)    # (B, D+2)
        teacher_global = packed[:, :dim]
        anchor_seq = packed[:, dim].long()
        anchor_pos = packed[:, dim + 1].long()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            # ONLY the forward runs under autocast. Both losses .float() their
            # inputs, but inside an autocast region that is not enough --
            # autocast re-casts the matmuls themselves, so `s @ s.T` and
            # `z_s @ neg.T` would still run in fp16. Computing them out here
            # makes the fp32 that structural_loss's docstring claims actually
            # true, which matters most for the 8448-d InfoNCE dot products.
            with torch.autocast(device_type="cuda", enabled=amp):
                _patches, student_global = model(images)
            struct = (structural_loss(student_global, teacher_global, tau_kl)
                      if struct_w else student_global.sum() * 0.0)
            nce = (infonce_loss(student_global, teacher_global, anchor_seq,
                                anchor_pos, bank, bank_seq, bank_pos,
                                n_neg, tau_nce, window, head)
                   if nce_w else student_global.sum() * 0.0)
            loss = struct_w * struct + nce_w * nce

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: struct={struct.item()} "
                               f"nce={nce.item()}")

        if train_mode:
            if not state["reported"] and struct_w and nce_w:
                _report_scales(struct, nce, student_global)
                state["reported"] = True
            # Clip EVERY optimised parameter, not just the model's: the
            # InfoNCE head is in the optimiser too, and leaving it unclipped
            # lets it take unbounded steps while the backbone is bounded --
            # the head would absorb the alignment instead of the descriptor.
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

        totals += torch.tensor([loss.item(), struct.item(), nce.item()])
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "kl": f"{struct.item():.4f}",
                          "nce": f"{nce.item():.4f}"})
    return (totals / max(n, 1)).tolist()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # This filename used to be a tombstone that failed loudly so an old
    # notebook cell could not silently launch a stale configuration. Requiring
    # a flag no old cell passes preserves that property.
    teacher_dir = _m(cfg, "teacher_dir")
    if not teacher_dir:
        raise SystemExit(
            "train_megaloc.py requires +mega.teacher_dir=<dir written by "
            "cache_megaloc.py>.\nIf you got here from an older notebook cell "
            "passing +training.megaloc_dir, that configuration no longer "
            "exists -- read this file's docstring for the current commands.")

    torch.manual_seed(cfg.seed)
    # Fail rather than train 30 epochs on CPU. The first launch of this script
    # did exactly that -- torch reported "no accelerator is found", both runs
    # fell back to CPU and exhausted host RAM instead of raising.
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
    _BANK["dir"] = str(teacher_dir)

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

    # Bank is built from TRAIN sequences only, so a val place can never be
    # drawn as a negative.
    print("Building negative bank...")
    bank_t = build_bank(teacher_dir, probe_ds, val_probe, device)
    bank, _bseq, _bpos, seq_id = bank_t
    _BANK["seq_id"] = seq_id
    _BANK["dim"] = int(bank.shape[1])
    print(f"  bank: {bank.shape[0]} descriptors x {bank.shape[1]}-d "
          f"({bank.element_size() * bank.nelement() / 1e6:.0f} MB fp16) "
          f"over {len(probe_ds.sequence_names())} train sequences "
          f"({len(probe_ds)} pairs)")

    # No teacher_dim constraint: the InfoNCE head bridges the widths, and the
    # KL never needed matching dims (it compares B x B similarity matrices on
    # each side separately, which is how every prior run trained a 1024-d
    # student against this 8448-d cache).

    # rebuild both splits against the cached-descriptor dataset class
    orig = T.E_LiteVPRDataset
    T.E_LiteVPRDataset = MegaLocDataset
    try:
        train_ds = T.build_split(dataset_cfgs, "train_seq_list", cfg.data.modality)
        val_ds = T.build_split(dataset_cfgs, "val_seq_list", cfg.data.modality,
                               pair_stride_override=1)
    finally:
        T.E_LiteVPRDataset = orig

    use_blocks = bool(_m(cfg, "use_block_sampler"))
    block = int(_m(cfg, "block_size"))
    if use_blocks:
        if int(cfg.training.batch_size) % block:
            raise SystemExit(f"batch_size={cfg.training.batch_size} must be a "
                             f"multiple of block_size={block}")
        sampler = BlockSampler(train_ds, night_seqs, block=block,
                               night_frac=float(_m(cfg, "night_frac")))
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
    ).to(device)

    # InfoNCE-only projection head, trained with the model and DISCARDED at
    # save time -- only model.state_dict() is written, so the checkpoint stays
    # loadable by the unmodified evaluate_brisbane/evaluate_nsavp build_model.
    head = None
    if float(_m(cfg, "infonce_weight")):
        head = torch.nn.Linear(int(cfg.model.teacher_dim), _BANK["dim"]).to(device)
        print(f"  InfoNCE head: {cfg.model.teacher_dim} -> {_BANK['dim']} "
              f"({sum(p.numel() for p in head.parameters()) / 1e6:.1f}M params, "
              f"not saved)")

    lr_base = cfg.training.get("lr_base_batch_size", cfg.training.batch_size)
    lr = cfg.training.learning_rate * (cfg.training.batch_size / lr_base) ** 0.5
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    optimizer = optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    sched = None
    if bool(_m(cfg, "use_scheduler")):
        total = len(train_dl) * int(cfg.training.epochs)
        warm = max(1, int(round(float(_m(cfg, "warmup_frac")) * total)))
        floor = float(_m(cfg, "min_lr_frac"))

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        sched = optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        print(f"  LR schedule: {warm} warmup then cosine to {floor:g}x over "
              f"{total} steps")

    print("=" * 70)
    print(f"train_megaloc: teacher=megaloc({_BANK['dim']}-d)  blocks={use_blocks}"
          + (f" (block={block}, night_frac={float(_m(cfg, 'night_frac'))})"
             if use_blocks else ""))
    print(f"  loss = {float(cfg.training.structural_loss_weight):g} * "
          f"KL(mask_diag,standardize,tau={float(cfg.training.temperature):g})"
          f"  +  {float(_m(cfg, 'infonce_weight')):g} * "
          f"InfoNCE(tau={float(_m(cfg, 'infonce_temperature')):g}, "
          f"K={int(_m(cfg, 'infonce_negatives'))}, "
          f"excl=+/-{int(_m(cfg, 'infonce_exclude_window'))})")
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
        print(f"Train Loss: {tr[0]:.6f} (kl: {tr[1]:.6f}, nce: {tr[2]:.6f})")
        print(f"Val   Loss: {va[0]:.6f} (kl: {va[1]:.6f}, nce: {va[2]:.6f})")
        wandb.log({"epoch": epoch + 1, "train_loss": tr[0], "train_kl": tr[1],
                   "train_nce": tr[2], "val_loss": va[0], "val_kl": va[1],
                   "val_nce": va[2], "lr": optimizer.param_groups[0]["lr"]})

        if va[0] < best_val:
            best_val, patience = va[0], 0
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with val_loss: {va[0]:.6f}")
        else:
            patience += 1

        # `last` is written EVERY epoch: in both day/night expert runs `last`
        # beat `best` on day-night, because val is DSEC-only and day-dominated.
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
