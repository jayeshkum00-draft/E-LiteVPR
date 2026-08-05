import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GeM(nn.Module):
    """Stock generalised mean. UNCHANGED -- this is the default and the control.

    Canonical GeM (Radenovic et al., and inside MegaLoc / CosPlace) pools
    POST-RELU CNN maps, where `clamp(min=eps)` is a numerical guard on values
    that are already non-negative. Also used here on the TEACHER's cached
    patches in train.py, which is why it must not change.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1, keepdim=True).pow(1.0 / self.p)
        return x.squeeze(1)


class SignedGeM(nn.Module):
    """Generalised mean that survives signed input.

        m = mean_n( sign(x) * |x|^p );   out = sign(m) * |m|^(1/p)

    The student's pooling input is `self.proj(patches)`, an nn.Linear output and
    therefore SIGNED, so GeM's clamp is not a guard here but a truncation:
    measured 2026-08-04 (diagnose_descriptor_collapse.py), 62-70% of token
    entries are deleted, those entries also receive ZERO gradient (clamp's
    derivative below the threshold), and every descriptor is forced into the
    positive orthant -- effective rank 1.10 of 1024, mean cosine 0.950-0.956
    between DIFFERENT places, identical day and night and across checkpoints
    trained on disjoint data.

    This keeps GeM's contrast enhancement (large-|x| tokens still dominate)
    while carrying the sign bit through. At p=1 it is exactly mean pooling, so
    the learned `p` is itself a readable diagnostic.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):                        # (B, N, D) -> (B, D)
        m = (torch.sign(x) * x.abs().clamp(min=self.eps).pow(self.p)).mean(dim=1)
        return torch.sign(m) * m.abs().clamp(min=self.eps).pow(1.0 / self.p)


class MeanPool(nn.Module):
    """Plain mean over patches: the p=1 control that separates "the clamp was
    the problem" from "GeM was the problem". No parameters."""

    def forward(self, x):
        return x.mean(dim=1)


# 'clamp' is handled separately: it keeps the historical `gem` attribute name.
_POOLERS = {'signed': SignedGeM, 'mean': MeanPool}


def build_pooler(pooling='clamp', p=3.0):
    """A pooler by name, for the TEACHER side as well as the student.

    train.py builds the structural target by pooling the cached DINOv3 PATCHES
    with GeM. Those tokens are signed too, so the clamp truncates the TARGET in
    exactly the way it truncates the student's descriptor -- which is worth
    being able to ablate independently of the student's own pooling
    (`model.teacher_pooling` vs `model.pooling`).
    """
    if pooling == 'clamp':
        return GeM(p=p)
    if pooling not in _POOLERS:
        raise ValueError(f"pooling must be one of "
                         f"{sorted(_POOLERS) + ['clamp']}, got {pooling!r}")
    cls = _POOLERS[pooling]
    return cls(p=p) if cls is SignedGeM else cls()


class EventViTStudent(nn.Module):
    def __init__(self,
                 backbone_name='vit_small_patch16_dinov3.lvd1689m',
                 teacher_dim=1024,
                 num_patches=576,
                 img_size=(480, 640),
                 in_channels=3,
                 pooling='clamp'):
        super().__init__()

        if pooling != 'clamp' and pooling not in _POOLERS:
            raise ValueError(f"model.pooling must be one of "
                             f"{sorted(_POOLERS) + ['clamp']}, got {pooling!r}")

        self.teacher_dim = teacher_dim
        self.num_patches = num_patches
        self.image_size = img_size
        self.in_channels = in_channels
        self.pooling = pooling

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=self.image_size
        )

        student_dim = self.backbone.embed_dim
        print(f"Student backbone dimension: {student_dim}")
        print(f"Teacher dimension: {teacher_dim}")

        self.input_norm = nn.BatchNorm2d(in_channels)

        self.proj = nn.Linear(student_dim, teacher_dim)

        # STATE-DICT KEYS ARE DELIBERATELY INCOMPATIBLE ACROSS THE FAMILIES.
        # 'clamp' keeps `gem.p`, so every checkpoint trained before this flag
        # existed still loads. 'signed'/'mean' register as `pool`, so a strict
        # load_state_dict RAISES on a checkpoint from the other family instead
        # of silently pooling those weights the wrong way and printing
        # plausible-looking numbers.
        if pooling == 'clamp':
            self.gem = GeM()
        else:
            self.pool = _POOLERS[pooling]()
        print(f"Pooling: {pooling} (state-dict key "
              f"{'gem' if pooling == 'clamp' else 'pool'})")

    def forward(self, x):
        x = self.input_norm(x)

        features = self.backbone.forward_features(x)
        patches = features[:, -self.num_patches:, :]
        projected_patches = self.proj(patches)
        pooler = self.gem if self.pooling == 'clamp' else self.pool
        global_descriptor = pooler(projected_patches)

        return projected_patches, global_descriptor
