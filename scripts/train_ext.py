"""One training entry point; every addition is OFF by default and flag-gated.

With no `+training.*` overrides this is byte-identical to
`python scripts/train.py` -- same dataset, same losses, same sampler, same
optimiser. Each flag turns on exactly one change, so a run is fully described
by its override list and cells of an ablation table are directly comparable.

    +training.teacher=dinov3|megaloc   (default dinov3)
        dinov3  : teacher_global = GeM(p=3) over cached DINOv3 patches, i.e.
                  train.py unchanged.
        megaloc : teacher_global = cached MegaLoc descriptor (8448-d) from
                  +training.megaloc_dir. Forces structural-only; the patch term
                  has no target and patch_loss_weight must be 0.
                  Descriptor widths need not match: compute_structural_loss
                  reduces both sides to a B x B matrix of pairwise cosines
                  before comparing, so the student stays 1024-d and unchanged.

    +training.use_block_sampler=true   (default false)
    +training.block_size=8
    +training.night_frac=0.5
        Replaces the frame-level WeightedRandomSampler with contiguous blocks.
        Measured on the real MegaLoc cache (probe_target_information.py),
        B=32/tau=0.05: random batching gives 0.9993 diagonal / 0.0007
        informative target mass over 24.8 fake neighbours; K=4 x M=8 gives
        0.0659 informative over 3.2 real ones. The first MegaLoc run's train
        loss converged to exactly 0.0007 -- it hit the ceiling of its own
        objective in two epochs. batch_size must be a multiple of block_size;
        this is enforced, not assumed.

    +training.use_scheduler=true       (default false)
    +training.warmup_frac=0.03
    +training.min_lr_frac=0.05
        Per-STEP linear warmup then cosine decay to min_lr_frac x base lr.
        train.py has neither, and sqrt lr scaling puts bs=256 at 2.83e-4 from
        the first step. Total steps are derived from the actual train loader
        length x training.epochs, so it adapts to whatever corpus is loaded.

HOW IT PATCHES
--------------
train.py is imported, never edited or copied, so losses, checkpointing, early
stopping and wandb logging remain the code behind every recorded result. cfg is
captured by wrapping `train.active_dataset_cfgs` -- the first thing main() calls
that receives cfg (train.py:286), and it runs before build_split (292), the
DataLoaders (314) and the optimiser (350), all of which resolve
`E_LiteVPRDataset` / `build_day_night_sampler` / `DataLoader` / `optim` as
module globals at call time. So installing there is enough.

    HYDRA_MAIN_MODULE=__main__ python scripts/train_ext.py ...

is REQUIRED for any wrapper of this shape: hydra unwraps the decorated function
and reads `__module__`, which is "train", not "__main__", so it treats
config_path "../configs" as a package and dies with "Primary config module
'configs' not found" (hydra/_internal/utils.py:45-53).
"""

import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import train
from block_sampler import BlockSampler
from dataset import E_LiteVPRDataset, sequence_name_from_rel_path

_DEFAULTS = {
    "teacher": "dinov3",
    "teacher_dir": None,
    "megaloc_dir": None,          # back-compat alias for teacher_dir
    "use_block_sampler": False,
    "block_size": 8,
    "night_frac": 0.5,
    "use_scheduler": False,
    "warmup_frac": 0.03,
    "min_lr_frac": 0.05,
}

_state = {"steps_per_epoch": None, "teacher_dir": None, "teacher_file": None}

# teacher -> filename written by the corresponding cache script
_CACHED_GLOBAL = {
    "megaloc": ("megaloc.npy", "cache_megaloc.py"),
    "dinov3_gem": ("gem.npy", "cache_teacher_gem.py"),
}


def _flag(cfg, name):
    v = OmegaConf.select(cfg, f"training.{name}")
    return _DEFAULTS[name] if v is None else v


# --------------------------------------------------------------- megaloc ---
class CachedGlobalDataset(E_LiteVPRDataset):
    """Serves (cached_global[None, :], dummy_attn) in place of (patches, attn).

    Used for both `megaloc` (8448-d, cache_megaloc.py) and `dinov3_gem`
    (1024-d, cache_teacher_gem.py). The latter is mathematically identical to
    teacher=dinov3 -- train.py's frozen GeM(p=3) applied to the same patches --
    but reads 2 KB per frame instead of 1.18 MB, which is the difference
    between I/O-bound and GPU-bound on network-backed storage.

    Only `_get_features` is overridden, so pairs.txt parsing, the frames.txt
    row index, sequence filtering and the fail-fast probe in __init__ are the
    originals -- including its `patches.ndim == 2 and attn.ndim == 1` assertion,
    which the (1, D) / (1,) shapes below satisfy.
    """

    def _get_features(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            fname, script = _state["teacher_file"]
            path = Path(_state["teacher_dir"]) / seq / fname
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found -- run {script} for this corpus first. "
                    f"Sequence names must match the teacher cache.")
            arr = np.load(path, mmap_mode="r")
            n_rows = sum(1 for ln in
                         (self.features_dir / seq / "frames.txt").read_text()
                         .splitlines() if ln.strip())
            if arr.shape[0] != n_rows:
                raise RuntimeError(
                    f"{path} has {arr.shape[0]} rows but {seq}/frames.txt has "
                    f"{n_rows}. The cache is misaligned with the row index and "
                    f"every target would be the wrong frame. Re-run "
                    f"cache_megaloc.py for {seq}.")
            self._mmaps[seq] = arr
        row = self._row_index[pair["feature_key"]]
        desc = torch.from_numpy(
            np.ascontiguousarray(self._mmaps[seq][row]).astype(np.float32))
        return desc.unsqueeze(0), torch.zeros(1)


def compute_losses_global(model_out, teacher_desc, teacher_attn, teacher_gem,
                          use_agfd, structural_weight, temperature,
                          patch_weight=1.0):
    """Structural-only against a precomputed global teacher descriptor.

    Signature and 3-tuple return match train.compute_losses so train_epoch and
    validate_epoch call it unchanged; patch_loss is reported as a constant 0.

    train.compute_losses ALWAYS evaluates the patch term even at weight 0, so
    it would MSE (B, 576, 1024) student patches against a (B, 1, 8448) vector
    and die on the shape. It also never calls teacher_gem: GeM does
    clamp(min=eps).pow(p), which would destroy a MegaLoc descriptor's negative
    components.
    """
    if patch_weight:
        raise ValueError(
            f"training.patch_loss_weight={patch_weight} but a cached GLOBAL "
            f"teacher descriptor has no patch target. Pass "
            f"training.patch_loss_weight=0.")
    _student_patches, student_global = model_out
    teacher_global = teacher_desc.squeeze(1)              # (B, 1, D) -> (B, D)
    structural = train.compute_structural_loss(
        student_global, teacher_global, temperature)
    return structural_weight * structural, structural.new_zeros(()), structural


# ------------------------------------------------------------- scheduler ---
class ScheduledAdamW(torch.optim.AdamW):
    """AdamW with a per-step linear warmup then cosine decay.

    Implemented on the optimiser rather than as a torch scheduler because
    train.py's epoch loop has no place to call `scheduler.step()`, and warmup
    should advance per step anyway. Steps only advance when step() actually
    runs, so AMP batches skipped by GradScaler on non-finite grads do not eat
    warmup.
    """

    def __init__(self, params, lr, warmup_steps, total_steps, min_lr_frac,
                 **kw):
        super().__init__(params, lr=lr, **kw)
        self.base_lrs = [g["lr"] for g in self.param_groups]
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr_frac = float(min_lr_frac)
        self._step_n = 0

    def _factor(self):
        s = self._step_n
        if s < self.warmup_steps:
            return (s + 1) / self.warmup_steps
        p = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        p = min(1.0, p)
        return self.min_lr_frac + (1 - self.min_lr_frac) * 0.5 * (
            1 + math.cos(math.pi * p))

    @torch.no_grad()
    def step(self, closure=None):
        f = self._factor()
        for group, base in zip(self.param_groups, self.base_lrs):
            group["lr"] = base * f
        self._step_n += 1
        return super().step(closure)


# ---------------------------------------------------------------- install ---
def _make_dataloader(original, cfg):
    def guarded(dataset, *args, **kwargs):
        sampler = kwargs.get("sampler")
        bs = kwargs.get("batch_size")
        if isinstance(sampler, BlockSampler) and bs is not None:
            if bs % sampler.block:
                raise ValueError(
                    f"training.batch_size={bs} is not a multiple of "
                    f"training.block_size={sampler.block}; batches would "
                    f"straddle block boundaries and the geometry would not be "
                    f"the one that was measured.")
            print(f"  -> {bs // sampler.block} clips x {sampler.block} frames "
                  f"per batch")
        if _state["steps_per_epoch"] is None and bs and hasattr(sampler, "__len__"):
            # first DataLoader built is the TRAIN one (train.py:314)
            _state["steps_per_epoch"] = len(sampler) // bs
        return original(dataset, *args, **kwargs)
    return guarded


class _OptimProxy:
    """Stands in for `train.optim` so only AdamW is intercepted; the real
    torch.optim module is never mutated."""

    def __init__(self, real, factory):
        self._real, self._factory = real, factory

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def AdamW(self):
        return self._factory


def _install(cfg):
    teacher = str(_flag(cfg, "teacher")).lower()
    use_blocks = bool(_flag(cfg, "use_block_sampler"))
    use_sched = bool(_flag(cfg, "use_scheduler"))

    print("=" * 70)
    print(f"train_ext: teacher={teacher}  block_sampler={use_blocks}"
          + (f" (block={int(_flag(cfg, 'block_size'))}, "
             f"night_frac={float(_flag(cfg, 'night_frac'))})" if use_blocks else "")
          + f"  scheduler={use_sched}"
          + (f" (warmup_frac={float(_flag(cfg, 'warmup_frac'))}, "
             f"min_lr_frac={float(_flag(cfg, 'min_lr_frac'))})" if use_sched else ""))
    print("=" * 70)

    if teacher in _CACHED_GLOBAL:
        fname, script = _CACHED_GLOBAL[teacher]
        d = _flag(cfg, "teacher_dir") or _flag(cfg, "megaloc_dir")
        if not d:
            raise SystemExit(
                f"+training.teacher={teacher} requires "
                f"+training.teacher_dir=<dir written by {script}>")
        _state["teacher_dir"] = str(d)
        _state["teacher_file"] = (fname, script)
        train.E_LiteVPRDataset = CachedGlobalDataset
        train.compute_losses = compute_losses_global
        print(f"  cached global teacher: {d}/<seq>/{fname}")
    elif teacher != "dinov3":
        raise SystemExit(f"unknown +training.teacher={teacher!r} "
                         f"(dinov3|dinov3_gem|megaloc)")

    if use_blocks:
        block = int(_flag(cfg, "block_size"))
        nf = float(_flag(cfg, "night_frac"))
        train.build_day_night_sampler = (
            lambda dataset, night_seqs: BlockSampler(
                dataset, night_seqs, block=block, night_frac=nf))

    train.DataLoader = _make_dataloader(train.DataLoader, cfg)

    if use_sched:
        epochs = int(cfg.training.epochs)
        warmup_frac = float(_flag(cfg, "warmup_frac"))
        min_lr_frac = float(_flag(cfg, "min_lr_frac"))
        real_optim = train.optim

        def factory(params, lr, **kw):
            spe = _state["steps_per_epoch"]
            if not spe:
                raise RuntimeError(
                    "steps_per_epoch was not captured before the optimiser was "
                    "built; the DataLoader hook did not fire.")
            total = spe * epochs
            warm = max(1, int(round(warmup_frac * total)))
            print(f"  LR schedule: {warm} warmup steps then cosine to "
                  f"{min_lr_frac:g}x over {total} steps "
                  f"({spe} steps/epoch x {epochs} epochs)")
            return ScheduledAdamW(params, lr=lr, warmup_steps=warm,
                                  total_steps=total, min_lr_frac=min_lr_frac,
                                  **kw)

        train.optim = _OptimProxy(real_optim, factory)


# -------------------------------------------------------------- exclusions ---
def _sequences_in(root, pairs_name="pairs.txt"):
    """Sequence names present in a source's pairs.txt.

    Uses dataset.sequence_name_from_rel_path so the naming rule is not
    duplicated: the name comes from the FILENAME
    ('rgb/car_urban_day_horse_000123.npy' -> 'car_urban_day_horse'), not from a
    directory component.
    """
    seqs = set()
    with open(Path(root) / pairs_name) as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.add(sequence_name_from_rel_path(line.split(",")[1]))
    return seqs


def _apply_exclusions(dataset_cfgs):
    """Honour each source's `to_exclude` list.

    Nothing in train.py or dataset.py reads that key -- `build_split` filters
    only by train_seq_list/val_seq_list, and `train_seq_list: null` means
    "every sequence in pairs.txt". So an excluded sequence (e.g. M3ED's
    `falcon_outdoor_day_penno_cars`, a DRONE sequence whose altitude, viewpoint
    and motion model do not belong in a car corpus) would silently be trained
    on. This resolves the split lists explicitly instead.

    Also drops excluded names from `night_sequences`, otherwise train.py:304's
    whitelist assert would fail on a night sequence that was deliberately
    removed.
    """
    for dcfg in dataset_cfgs:
        excluded = set(dcfg.get("to_exclude") or [])
        if not excluded:
            continue

        present = _sequences_in(dcfg.root_dir,
                                dcfg.get("pairs_name") or "pairs.txt")
        missing = excluded - present
        hit = excluded & present

        for split_key in ("train_seq_list", "val_seq_list"):
            current = dcfg.get(split_key, None)
            if current is not None and len(current) == 0:
                continue                      # [] -> source sits this split out
            base = present if current is None else set(current)
            dcfg[split_key] = sorted(base - excluded)

        night = dcfg.get("night_sequences") or []
        if night:
            dcfg["night_sequences"] = [s for s in night if s not in excluded]

        print(f"  to_exclude ({Path(str(dcfg.root_dir)).name}): dropped "
              f"{sorted(hit) if hit else 'nothing'}"
              + (f"; NOT FOUND in pairs.txt: {sorted(missing)}" if missing else ""))


def _capture_cfg(module):
    original = module.active_dataset_cfgs

    def wrapper(cfg):
        _install(cfg)
        dataset_cfgs = original(cfg)
        _apply_exclusions(dataset_cfgs)
        return dataset_cfgs

    module.active_dataset_cfgs = wrapper


_capture_cfg(train)

if __name__ == "__main__":
    train.main()
