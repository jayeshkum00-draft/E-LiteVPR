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

    +training.teacher_b=dinov3_gem     (default null = single teacher)
    +training.teacher_b_dir=<dir written by cache_teacher_gem.py>
    +training.teacher_mix=0.2
    +training.teacher_b_mask_diag / _standardize / _temperature
        SECOND structural target, summed as

            mix * L(student, teacher_A) + (1 - mix) * L(student, teacher_B)

        with per-teacher loss settings, because the two teachers do NOT want
        the same objective. Measured (Brisbane [all], mean over L=10/20/30):
        MegaLoc needs mask_diag+z-score (raw tau=0.05 gave grad norm 2.1e-09 --
        no gradient at all), while the SAME fix HURT DINOv3 on every cell
        (37.27 -> 31.27) because its near-uniform raw target acts as a
        smoothing prior and smooth descriptors are what 30-frame sequence
        matching integrates. Sharing one loss setting throws away one of the
        two teachers. So teacher A uses training.temperature /
        struct_mask_diag / struct_standardize, and teacher B has its own trio,
        defaulting to the RAW loss (tau=0.05, no mask, no z-score) that
        produced the 37.27 model.

        Why this exists: descriptor-level fusion of those two checkpoints,
        [sqrt(a)*d_A, sqrt(1-a)*d_B], beats BOTH endpoints on EVERY cell --
        a=0.20 gives day-day 84.07 / day-night 51.92 vs 42.13 (A alone) and
        37.27 (B alone). That proves the targets carry non-redundant
        information, but it ships two backbones (~44M params, two forward
        passes) and makes `a` a test-time knob read off the benchmark. Summing
        the targets at train time gets one ~22M student and turns `a` into an
        ordinary loss weight.

        Both teachers must be cached-global (megaloc | dinov3_gem |
        dinov3_cls); teacher=dinov3 computes GeM in-training from patches and
        has no cache to pair. dinov3_gem IS that same GeM(p=3) over the same
        patches, just precomputed, so it reproduces teacher=dinov3 exactly.
        Widths need not match (8448 vs 1024): each loss reduces its side to a
        B x B cosine matrix first.

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

    +training.struct_mask_diag=true    (default false)
        Drop the self-similarity term from BOTH softmaxes (B x B -> B x (B-1)).
        T[i,i] is 1.0 by construction, carries no information about how frames
        relate to EACH OTHER, and is satisfied by any student. Measured at
        tau=0.05 (probe_target_information.py) it is where the target's mass
        went: MegaLoc put 0.9993 of it there, so 99.93% of that objective was a
        term that teaches nothing. DINOv3 GeM put 0.0684 there.

    +training.struct_standardize=true  (default false)
        Z-score each similarity matrix over its off-diagonal before dividing by
        tau, so tau is measured in units of WITHIN-BATCH sigma and stops
        depending on the teacher's cosine scale.
        Why: compute_structural_loss divides raw cosines by a single shared tau,
        so what the softmax sees is (cosine spread)/tau. Measured spreads are
        not comparable -- DINOv3 GeM off_cos 0.9606 (spread ~0.05, logit spread
        ~1 at tau=0.05 -> near-uniform target, eff_nbrs 30.15 of 31) vs MegaLoc
        off_cos 0.0548 (spread ~0.95, logit spread ~19 -> near-identity target).
        One tau cannot serve both, and a STRONGER VPR teacher makes distinct
        places more orthogonal, i.e. its fixed-tau target carries LESS
        information. That is the mechanism by which better teachers scored
        worse. After standardising, tau ~ 1.0 is the sensible setting for any
        teacher; the run WARNS if tau is left at the raw-cosine default.

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
    "teacher_b": None,
    "teacher_b_dir": None,
    "teacher_mix": 0.5,           # weight on teacher A; 0.5 = untuned default
    "teacher_b_mask_diag": False,  # defaults reproduce the RAW loss that
    "teacher_b_standardize": False,  # produced the 37.27 DINOv3 model
    "teacher_b_temperature": 0.05,
    "use_block_sampler": False,
    "block_size": 8,
    "night_frac": 0.5,
    "struct_mask_diag": False,
    "struct_standardize": False,
    "use_scheduler": False,
    "warmup_frac": 0.03,
    "min_lr_frac": 0.05,
}

_state = {"steps_per_epoch": None, "teacher_dir": None, "teacher_file": None}

# two-teacher state. dim_a is resolved in the MAIN process at install time (see
# _probe_width) -- a DataLoader worker cannot write it back, so it must never be
# discovered lazily inside _get_features.
_two = {"dir_a": None, "file_a": None, "dir_b": None, "file_b": None,
        "dim_a": None, "loss_a": None, "loss_b": None,
        "tau_a": None, "tau_b": None, "mix": None, "diag_done": False}

# teacher -> filename written by the corresponding cache script
_CACHED_GLOBAL = {
    "megaloc": ("megaloc.npy", "cache_megaloc.py"),
    "dinov3_gem": ("gem.npy", "cache_teacher_gem.py"),
    # cls.npy is already written by feature_extractor.py:161 alongside
    # patches.npy -- no new caching pass. Use it to sidestep GeM entirely:
    # GeM does clamp(min=1e-6).pow(3) on RAW SIGNED DINOv3 tokens
    # (feature_extractor.py:88 saves hidden[:, n_prefix:, :] unrectified), so
    # every negative coordinate collapses to a shared constant. Measured
    # consequence: within-sequence mean cosine 0.96-0.99 on all 59 sequences
    # regardless of scene, condition or dataset.
    "dinov3_cls": ("cls.npy", "feature_extractor.py"),
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


# ---------------------------------------------------------- two teachers ---
def _open_cache(root, fname, script, seq, features_dir):
    """mmap <root>/<seq>/<fname>, asserting it is row-aligned with frames.txt.

    Same contract as CachedGlobalDataset._get_features (left untouched so the
    single-teacher path that produced every recorded result is unchanged): the
    row index comes from frames.txt, so a cache with the wrong number of rows
    would silently pair every frame with a different frame's descriptor.
    """
    path = Path(root) / seq / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run {script} for this corpus first. "
            f"Sequence names must match the teacher cache.")
    arr = np.load(path, mmap_mode="r")
    n_rows = sum(1 for ln in
                 (features_dir / seq / "frames.txt").read_text().splitlines()
                 if ln.strip())
    if arr.shape[0] != n_rows:
        raise RuntimeError(
            f"{path} has {arr.shape[0]} rows but {seq}/frames.txt has {n_rows}. "
            f"The cache is misaligned with the row index and every target would "
            f"be the wrong frame. Re-run {script} for {seq}.")
    return arr


def _probe_width(root, fname, script):
    """Descriptor width from any one cached sequence, read in the main process.

    Doubles as the fail-fast check that the cache exists at all -- otherwise a
    missing teacher_b_dir would only surface inside a worker on the first batch.
    """
    hits = sorted(Path(root).glob(f"*/{fname}"))
    if not hits:
        raise SystemExit(
            f"no */{fname} under {root} -- run {script} for this corpus first.")
    return int(np.load(hits[0], mmap_mode="r").shape[1])


class TwoTeacherDataset(E_LiteVPRDataset):
    """Serves both teachers' descriptors concatenated into one (1, Da+Db) row.

    Packing rather than returning a tuple keeps the collate, the (B, 1, D)
    contract and __init__'s `patches.ndim == 2 and attn.ndim == 1` probe exactly
    as they are; compute_losses_two splits at _two["dim_a"].
    """

    def _get_features(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            self._mmaps[seq] = (
                _open_cache(_two["dir_a"], *_two["file_a"], seq, self.features_dir),
                _open_cache(_two["dir_b"], *_two["file_b"], seq, self.features_dir),
            )
        arr_a, arr_b = self._mmaps[seq]
        row = self._row_index[pair["feature_key"]]
        da = torch.from_numpy(np.ascontiguousarray(arr_a[row]).astype(np.float32))
        db = torch.from_numpy(np.ascontiguousarray(arr_b[row]).astype(np.float32))
        return torch.cat([da, db]).unsqueeze(0), torch.zeros(1)


def compute_losses_two(model_out, teacher_desc, teacher_attn, teacher_gem,
                       use_agfd, structural_weight, temperature,
                       patch_weight=1.0):
    """mix * L(student, A) + (1 - mix) * L(student, B), per-teacher settings.

    `temperature` (cfg.training.temperature) is teacher A's; teacher B uses
    training.teacher_b_temperature. Signature and 3-tuple return match
    train.compute_losses so train_epoch/validate_epoch call it unchanged.
    """
    if patch_weight:
        raise ValueError(
            f"training.patch_loss_weight={patch_weight} but cached GLOBAL "
            f"teacher descriptors have no patch target. Pass "
            f"training.patch_loss_weight=0.")
    _student_patches, student_global = model_out
    packed = teacher_desc.squeeze(1)                      # (B, 1, Da+Db)
    da = _two["dim_a"]
    loss_a = _two["loss_a"](student_global, packed[:, :da], _two["tau_a"])
    loss_b = _two["loss_b"](student_global, packed[:, da:], _two["tau_b"])
    mix = _two["mix"]
    if not _two["diag_done"]:
        _report_mix(student_global, loss_a, loss_b, mix)
    structural = mix * loss_a + (1.0 - mix) * loss_b
    return structural_weight * structural, structural.new_zeros(()), structural


def _report_mix(student_global, loss_a, loss_b, mix):
    """One-time: what fraction of the gradient does `mix` actually give A?

    teacher_mix is NOT scale-free, and the two scales disagree about which
    teacher dominates. Synthetic descriptors matched to the measured geometries
    (MegaLoc off_cos 0.055 / DINOv3 GeM 0.961) give L_b = 16x L_a but
    |g_a| = 2.3x |g_b| -- so at mix=0.2 teacher A is 1.5% of the loss VALUE and
    37% of the GRADIENT. The gradient is what trains the student, so that is
    the number to set `mix` by, and it must be read off the real caches rather
    than assumed. Same failure class as one shared tau across two teachers.

    Costs two extra backward passes through the head, once per run.
    """
    _two["diag_done"] = True
    if not (torch.is_grad_enabled() and student_global.requires_grad):
        return                                    # validation pass; try again later
    try:
        ga = torch.autograd.grad(loss_a, student_global, retain_graph=True)[0].norm()
        gb = torch.autograd.grad(loss_b, student_global, retain_graph=True)[0].norm()
    except RuntimeError as e:                     # never let a diagnostic kill a run
        print(f"  [teacher_mix] gradient probe skipped: {e}")
        return
    ga, gb = ga.item(), gb.item()
    denom = mix * ga + (1 - mix) * gb
    share = (mix * ga / denom) if denom > 0 else float("nan")
    print(f"\n  [teacher_mix={mix:g}] measured on the first real batch:"
          f"\n    A: loss {loss_a.item():9.4f}  |grad| {ga:.3e}"
          f"\n    B: loss {loss_b.item():9.4f}  |grad| {gb:.3e}"
          f"\n    -> teacher A supplies {100 * share:.1f}% of the structural "
          f"gradient (loss-value share {100 * mix * loss_a.item() / max(1e-12, mix * loss_a.item() + (1 - mix) * loss_b.item()):.1f}%)."
          f"\n    Set mix by the GRADIENT share; it is not the fusion alpha.\n")
    if ga < 1e-6 or gb < 1e-6:
        print(f"    WARNING: one teacher's gradient is ~0 -- that term is not "
              f"training anything. Check its tau/standardize settings.\n")


# ----------------------------------------------------------- struct loss ---
def _drop_diag(sim):
    """(B, B) -> (B, B-1), removing T[i,i].

    Selection rather than masked_fill(-inf): a masked logit gives target 0 at
    that position and F.kl_div then evaluates 0 * (log 0 - (-inf)) = nan. Only
    dropping the column avoids infinities entirely.
    """
    B = sim.shape[0]
    off = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    return sim.masked_select(off).view(B, B - 1)


def _standardize(sim):
    """Z-score a similarity matrix so tau is in units of within-batch sigma."""
    return (sim - sim.mean()) / sim.std().clamp(min=1e-6)


def make_structural_loss(mask_diag, standardize):
    """train.compute_structural_loss with the self-term and/or the teacher's
    cosine scale removed. Identical to the original when both are False."""

    def structural_loss(student_global, teacher_global, temperature=0.05):
        # float32: std/softmax under autocast fp16 is where a near-degenerate
        # similarity matrix turns into inf/nan.
        s = torch.nn.functional.normalize(student_global.float(), p=2, dim=-1)
        t = torch.nn.functional.normalize(teacher_global.float(), p=2, dim=-1)
        s_sim, t_sim = s @ s.T, t @ t.T

        if mask_diag:
            s_sim, t_sim = _drop_diag(s_sim), _drop_diag(t_sim)
        if standardize:
            s_sim, t_sim = _standardize(s_sim), _standardize(t_sim)

        s_sim, t_sim = s_sim / temperature, t_sim / temperature
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(s_sim, dim=-1),
            torch.nn.functional.softmax(t_sim, dim=-1),
            reduction="batchmean")

    return structural_loss


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
    mask_diag = bool(_flag(cfg, "struct_mask_diag"))
    standardize = bool(_flag(cfg, "struct_standardize"))

    print("=" * 70)
    print(f"train_ext: teacher={teacher}  block_sampler={use_blocks}"
          + (f" (block={int(_flag(cfg, 'block_size'))}, "
             f"night_frac={float(_flag(cfg, 'night_frac'))})" if use_blocks else "")
          + f"  scheduler={use_sched}"
          + (f" (warmup_frac={float(_flag(cfg, 'warmup_frac'))}, "
             f"min_lr_frac={float(_flag(cfg, 'min_lr_frac'))})" if use_sched else "")
          + f"  mask_diag={mask_diag}  standardize={standardize}")
    print("=" * 70)

    if mask_diag or standardize:
        tau = float(cfg.training.temperature)
        train.compute_structural_loss = make_structural_loss(mask_diag, standardize)
        print(f"  structural target: "
              f"{'B x (B-1), self-term dropped' if mask_diag else 'B x B'}"
              f"{', z-scored (tau in sigma units)' if standardize else ''}, "
              f"tau={tau:g}")
        if standardize and tau < 0.2:
            print(f"  WARNING: struct_standardize=true rescales the similarity "
                  f"matrix to unit variance, so tau is now in units of "
                  f"within-batch sigma. tau={tau:g} is the RAW-COSINE default "
                  f"and gives a logit spread of ~{1 / tau:.0f} sigma -- a "
                  f"near-one-hot target. Pass training.temperature=1.0 unless "
                  f"you are deliberately sweeping it.")

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

    teacher_b = _flag(cfg, "teacher_b")
    if teacher_b:
        teacher_b = str(teacher_b).lower()
        if teacher not in _CACHED_GLOBAL or teacher_b not in _CACHED_GLOBAL:
            raise SystemExit(
                f"+training.teacher_b requires BOTH teachers to be cached "
                f"globals ({'|'.join(_CACHED_GLOBAL)}); got teacher={teacher!r}, "
                f"teacher_b={teacher_b!r}. teacher=dinov3 computes GeM "
                f"in-training from patches and has no cache to pair -- use "
                f"dinov3_gem, which is the same GeM(p=3) over the same patches.")
        if teacher_b == teacher:
            raise SystemExit(
                f"+training.teacher_b={teacher_b!r} is the same as "
                f"+training.teacher -- that is one teacher counted twice, not "
                f"two targets.")

        dir_b = _flag(cfg, "teacher_b_dir")
        if not dir_b:
            raise SystemExit(
                f"+training.teacher_b={teacher_b} requires "
                f"+training.teacher_b_dir=<dir written by "
                f"{_CACHED_GLOBAL[teacher_b][1]}>")

        mix = float(_flag(cfg, "teacher_mix"))
        if not 0.0 <= mix <= 1.0:
            raise SystemExit(f"training.teacher_mix={mix} must be in [0, 1]")
        b_mask = bool(_flag(cfg, "teacher_b_mask_diag"))
        b_std = bool(_flag(cfg, "teacher_b_standardize"))
        tau_a = float(cfg.training.temperature)
        tau_b = float(_flag(cfg, "teacher_b_temperature"))

        _two.update(
            dir_a=_state["teacher_dir"], file_a=_state["teacher_file"],
            dir_b=str(dir_b), file_b=_CACHED_GLOBAL[teacher_b],
            dim_a=_probe_width(_state["teacher_dir"], *_state["teacher_file"]),
            loss_a=make_structural_loss(mask_diag, standardize),
            loss_b=make_structural_loss(b_mask, b_std),
            tau_a=tau_a, tau_b=tau_b, mix=mix)
        dim_b = _probe_width(str(dir_b), *_CACHED_GLOBAL[teacher_b])

        train.E_LiteVPRDataset = TwoTeacherDataset
        train.compute_losses = compute_losses_two
        print(f"  TWO TEACHERS: loss = {mix:g} * A + {1 - mix:g} * B")
        print(f"    A = {teacher:11s} {_two['dim_a']:5d}-d  {_state['teacher_dir']}"
              f"\n        mask_diag={mask_diag}  standardize={standardize}  "
              f"tau={tau_a:g}")
        print(f"    B = {teacher_b:11s} {dim_b:5d}-d  {dir_b}"
              f"\n        mask_diag={b_mask}  standardize={b_std}  tau={tau_b:g}")
        if b_std and tau_b < 0.2:
            print(f"    WARNING: teacher_b_standardize=true with "
                  f"teacher_b_temperature={tau_b:g} (the RAW-COSINE default) "
                  f"gives a ~{1 / tau_b:.0f} sigma logit spread. Pass "
                  f"+training.teacher_b_temperature=1.0.")
        if not b_std and tau_b >= 0.5:
            print(f"    WARNING: teacher_b_standardize=false with "
                  f"teacher_b_temperature={tau_b:g}. Without z-scoring, tau is "
                  f"in RAW cosine units; 0.05 is the setting that produced the "
                  f"37.27 DINOv3 model.")

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
