"""Pooling variants for the phase-1 student. `model.py` is NOT touched.

WHY (measured 2026-08-04, diagnose_descriptor_collapse.py)
----------------------------------------------------------
`model.py:13` is

    x = x.clamp(min=self.eps).pow(self.p)

and its input is `self.proj(patches)` (`model.py:44,52`) -- an `nn.Linear`
output, therefore SIGNED. GeM's canonical use (Radenovic et al.; and inside
MegaLoc / CosPlace) is on post-ReLU CNN feature maps, where the clamp is a
numerical guard on values that are already non-negative. Applied to a signed
projection it is not a guard, it is a truncation: 62-70% of token entries are
deleted and every descriptor is forced into the positive orthant, where similar
magnitude profiles force high cosine.

Measured consequence: effective rank 1.10 of 1024, top-1 direction carrying 95%
of the variance, mean cosine between DIFFERENT places 0.950-0.956 -- identical
day and night, and identical across two checkpoints trained on disjoint data,
which is what makes it architectural rather than learned.

This is a misuse of the operator, not a design choice, so fixing it is an
architecture correction rather than selective optimisation against an unpatched
baseline (the user's standing test: prefer changes that would improve ANY
event-VPR model).

THE VARIANTS
------------
  signed   sign-preserving generalised mean:
               mean_n( sign(x) |x|^p )  then  sign(m) |m|^(1/p)
           keeps GeM's contrast enhancement -- large-|x| tokens still dominate
           -- while carrying the sign bit through. Learnable p, init 3.0.
  mean     plain mean over patches. The p=1 control that separates "the clamp
           was the problem" from "GeM was the problem".
  clamp    stock GeM, unchanged, so the control can run from this same file.

STATE-DICT KEYS ARE DELIBERATELY INCOMPATIBLE
---------------------------------------------
`signed` and `mean` register their pooler as `pool`, not `gem`, so the stock
`evaluate_brisbane.build_model` (`:307`, a strict `load_state_dict`) RAISES on
these checkpoints instead of silently evaluating them through the clamped
pooling and printing plausible-looking numbers. `clamp` keeps `gem.p` and stays
loadable everywhere, as a control must be.

To evaluate a `signed`/`mean` checkpoint use `evaluate_brisbane_gem.py`, which
rebinds the name and prints which pooling it installed.
"""

import torch
import torch.nn as nn

from model import EventViTStudent


class SignedGeM(nn.Module):
    """Generalised mean that survives signed input.

    m = mean_n( sign(x) * |x|^p );  out = sign(m) * |m|^(1/p)

    At p=1 this is exactly mean pooling, so `p` interpolates between the two
    variants and its learned value is itself a readable diagnostic.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):                        # (B, N, D) -> (B, D)
        m = (torch.sign(x) * x.abs().clamp(min=self.eps).pow(self.p)).mean(dim=1)
        return torch.sign(m) * m.abs().clamp(min=self.eps).pow(1.0 / self.p)


class MeanPool(nn.Module):
    """No parameters, so a stock checkpoint cannot load into it by accident."""

    def forward(self, x):
        return x.mean(dim=1)


_POOLERS = {"signed": SignedGeM, "mean": MeanPool}


class EventViTStudentPooled(EventViTStudent):
    """Stock student with `gem` replaced by `pool`.

    Only the pooling differs -- backbone, input_norm and proj are inherited
    untouched, so `signed`/`mean`/`clamp` differ in exactly one operator and the
    comparison is clean.
    """

    def __init__(self, *args, pooling="signed", **kwargs):
        super().__init__(*args, **kwargs)
        if pooling not in _POOLERS:
            raise ValueError(f"pooling must be one of "
                             f"{sorted(_POOLERS) + ['clamp']}, got {pooling!r}")
        del self.gem                             # drops it from _modules too
        self.pool = _POOLERS[pooling]()
        self.pooling = pooling
        print(f"  pooling: {pooling} (state-dict key 'pool', NOT 'gem' -- "
              f"stock evaluate_brisbane will refuse this checkpoint)")

    def forward(self, x):
        x = self.input_norm(x)
        features = self.backbone.forward_features(x)
        patches = features[:, -self.num_patches:, :]
        projected_patches = self.proj(patches)
        return projected_patches, self.pool(projected_patches)


def build_student(pooling="signed", **kwargs):
    """Factory with the stock signature plus `pooling`.

    `clamp` returns the unmodified `EventViTStudent` rather than a subclass with
    a re-implemented GeM, so the control is the real thing and not a copy that
    could drift from `model.py`.
    """
    if pooling == "clamp":
        return EventViTStudent(**kwargs)
    return EventViTStudentPooled(pooling=pooling, **kwargs)


def install(module, pooling):
    """Point `module`'s EventViTStudent name at `pooling`.

    `evaluate_brisbane.build_model` (`:278`) looks the name up in
    `evaluate_brisbane`'s globals AT CALL TIME, so rebinding it here reaches
    every caller of that function -- including `probe_all.py` and
    `evaluate_nsavp.py`, whose `from evaluate_brisbane import build_model`
    binds the same function object.
    """
    def _factory(**kwargs):
        return build_student(pooling=pooling, **kwargs)

    module.EventViTStudent = _factory
    print(f"[gem] {module.__name__}.EventViTStudent -> pooling={pooling}")
