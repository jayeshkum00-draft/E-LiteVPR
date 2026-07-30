"""Per-sample input normalisation variant of EventViTStudent.

WHY THIS EXISTS
---------------
`EventViTStudent.__init__` sets `self.input_norm = nn.BatchNorm2d(in_channels)`
(model.py:42) and applies it as the very first op in `forward` (model.py:48).
BatchNorm uses *batch* statistics while training and *global running*
statistics at eval. Event frames have a systematic density gap between day and
night, so at eval time night frames enter the backbone with a different mean
and scale than day frames -- a condition shift injected mechanically, upstream
of every loss, and identical in every model trained so far. That is a candidate
explanation for night R@1 at L=1 sitting near 1.8% across grid, clean, phase-2
and structural-only runs while every other number moved.

Per-sample normalisation standardises each frame by its own statistics, so a
global density offset cannot survive the first layer. This is a
representation-level invariance mechanism and is orthogonal to the loss work.

    instance : InstanceNorm2d(affine=True) -- per sample, per channel, over HW.
               Removes a per-channel density offset. Default.
    layer    : LayerNorm over (C, H, W) -- per sample, across channels too, so
               the pos/neg/net channel BALANCE is preserved while overall
               magnitude is removed. Use if you suspect channel ratio carries
               the place signal.
    none     : identity, for an ablation row.

Nothing in model.py, train.py or evaluate_brisbane.py is modified; the wrapper
entry points below rebind the class in those modules before their `main` runs.

CHECKPOINTS ARE NOT INTERCHANGEABLE. BatchNorm2d carries running_mean /
running_var / num_batches_tracked which InstanceNorm2d(affine=True) does not,
so loading a BN checkpoint into this model fails loudly under strict
load_state_dict. That is intended -- these are different models.

Usage: see train_insnorm.py and evaluate_brisbane_insnorm.py.
"""

import os

import torch
import torch.nn as nn

from model import EventViTStudent


class _ChannelwiseLayerNorm(nn.Module):
    """LayerNorm over (C, H, W) for a fixed input size, written as a normalise
    + affine so the parameter shape stays (C,) and does not depend on H, W
    (img_size is a config knob, and a (C,H,W)-shaped LayerNorm would bake the
    resolution into the checkpoint)."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def build_input_norm(kind, in_channels):
    if kind == "instance":
        # track_running_stats=False -> same behaviour in train and eval, which
        # is the entire point; affine=True keeps a learnable per-channel gain.
        return nn.InstanceNorm2d(in_channels, affine=True,
                                 track_running_stats=False)
    if kind == "layer":
        return _ChannelwiseLayerNorm(in_channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown input norm {kind!r}; use instance|layer|none")


class EventViTStudentPerSampleNorm(EventViTStudent):
    """EventViTStudent with `input_norm` swapped for a per-sample normaliser.

    Subclassed rather than reimplemented so the backbone, projection, GeM and
    forward pass stay byte-identical to the model every recorded result used.
    """

    def __init__(self, *args, input_norm_kind=None, **kwargs):
        super().__init__(*args, **kwargs)
        kind = input_norm_kind or os.environ.get("INPUT_NORM", "instance")
        self.input_norm_kind = kind
        self.input_norm = build_input_norm(kind, self.in_channels)
        print(f"input_norm: {kind} (replaces BatchNorm2d)")


def install(*modules):
    """Rebind `EventViTStudent` inside already-imported pipeline modules.

    train.py and evaluate_brisbane.py both do `from model import
    EventViTStudent`, which binds the name into their own namespace at import
    time -- so patching `model.EventViTStudent` would have no effect and the
    module attribute has to be replaced directly.
    """
    for m in modules:
        if not hasattr(m, "EventViTStudent"):
            raise AttributeError(
                f"{m.__name__} has no EventViTStudent to replace; the import "
                f"in that script changed and this patch is now silently a "
                f"no-op. Fix here rather than editing the pipeline script.")
        m.EventViTStudent = EventViTStudentPerSampleNorm
