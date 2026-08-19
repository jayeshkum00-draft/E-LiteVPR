"""Event-native student: DINOv3 ViT-S/16 + input BatchNorm + linear projection
+ an aggregator, selected by the `model/aggregator` config group.

    events (B, 3, 384, 384)
      -> BatchNorm2d           absorbs the mean/variance gap event histograms
                               leave behind after per-frame unit-max scaling
      -> ViT-S/16 backbone     DINOv3 weights, 576 patch tokens at 384-d
      -> Linear 384 -> 1024    shared per-patch projection
      -> aggregator            gem (signed) or salad
      -> global descriptor     the only output used at deployment

forward() returns (projected_patches, global_descriptor); the patch grid is
kept in the signature because callers unpack it, not because it carries a loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# --- aggregators --------------------------------------------------------------

class SignedGeM(nn.Module):
    """Generalised mean that survives signed input.

        m = mean_n( sign(x) * |x|^p );   out = sign(m) * |m|^(1/p)

    The pooling input is `self.proj(patches)`, an nn.Linear output and therefore
    SIGNED, so stock GeM's `clamp(min=eps)` is not a numerical guard here but a
    truncation: measured 2026-08-04, it deletes 62-70% of token entries, those
    entries also receive ZERO gradient, and every descriptor is forced into the
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


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    """Sinkhorn normalisation in log space (numerically stable)."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    """scores: (B, m, n) cluster-vs-token logits. `alpha` is the learnable
    dustbin, appended as an extra ROW so tokens can be assigned to "no
    cluster" -- the mechanism that lets SALAD discard uninformative patches.
    Returns (B, m+1, n) log-assignments."""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins = alpha.expand(b, 1, n)
    couplings = torch.cat([scores, bins], dim=1)          # (B, m+1, n)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu = log_mu[None].expand(b, -1)
    log_nu = log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return Z - norm


class SALAD(nn.Module):
    """Optimal-transport aggregation over patch tokens (Izquierdo & Civera,
    CVPR 2024) -- the aggregator MegaLoc itself uses.

    This trains on the structural objective with no place labels because that
    objective compares B x B cosine similarity matrices, which are
    dimension-agnostic: the student may emit an 8448-d SALAD descriptor while
    the teacher target stays a different width, and gradient still flows into
    the SALAD projections.

    Args:
        in_dim:       token dimension fed in (the projected tokens, so
                      aggregation lives in the distilled space)
        cls_dim:      dimension of the backbone CLS token
        num_clusters: m
        cluster_dim:  l, tokens are projected DOWN to this before clustering --
                      what keeps the descriptor at 8448 instead of m * in_dim
        token_dim:    width of the global (CLS) branch
    """

    def __init__(self, in_dim=1024, cls_dim=384, num_clusters=64,
                 cluster_dim=128, token_dim=256, sinkhorn_iters=3):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.sinkhorn_iters = sinkhorn_iters

        # local branch: per-token features to be aggregated
        self.cluster_features = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, cluster_dim),
        )
        # assignment branch: per-token logits over clusters
        self.score = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, num_clusters),
        )
        # global branch: CLS token -> compact global component
        self.token_features = nn.Sequential(
            nn.Linear(cls_dim, 512), nn.ReLU(),
            nn.Linear(512, token_dim),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def desc_dim(self):
        return self.num_clusters * self.cluster_dim + self.token_dim

    def forward(self, tokens, cls_token):
        """tokens: (B, N, in_dim)  cls_token: (B, cls_dim) -> (B, desc_dim)"""
        # log-domain Sinkhorn is fp16-unstable; run the head in fp32.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = tokens.float()
            cls_token = cls_token.float()

            f = self.cluster_features(tokens)          # (B, N, l)
            p = self.score(tokens)                     # (B, N, m)
            t = self.token_features(cls_token)         # (B, token_dim)

            p = p.transpose(1, 2)                      # (B, m, N)
            p = log_optimal_transport(p, self.dust_bin, self.sinkhorn_iters)
            p = torch.exp(p)
            p = p[:, :-1, :]                           # drop dustbin row -> (B, m, N)

            # V[b, k, :] = sum_n p[b, k, n] * f[b, n, :]
            v = torch.einsum('bkn,bnl->bkl', p, f)     # (B, m, l)
            v = F.normalize(v, p=2, dim=-1)            # intra-cluster norm
            v = v.flatten(1)                           # (B, m*l)

            desc = torch.cat([F.normalize(t, p=2, dim=-1), v], dim=-1)
            return F.normalize(desc, p=2, dim=-1)


_AGGREGATORS = ('gem', 'salad')


# --- student ------------------------------------------------------------------

class EventViTStudent(nn.Module):
    """The aggregator registers under its own state-dict key (`pool` for gem,
    `salad` for salad), so a strict load_state_dict refuses a checkpoint from
    the other family instead of silently scoring it through the wrong head.
    """

    def __init__(self,
                 backbone_name='vit_small_patch16_dinov3.lvd1689m',
                 proj_dim=1024,
                 num_patches=576,
                 img_size=(384, 384),
                 in_channels=3,
                 aggregator='gem',
                 gem_p=3.0,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 sinkhorn_iters=3):
        super().__init__()

        if aggregator not in _AGGREGATORS:
            raise ValueError(f"model/aggregator must be one of "
                             f"{list(_AGGREGATORS)}, got {aggregator!r}")

        self.proj_dim = proj_dim
        self.num_patches = num_patches
        self.image_size = img_size
        self.in_channels = in_channels
        self.aggregator = aggregator

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        student_dim = self.backbone.embed_dim

        self.input_norm = nn.BatchNorm2d(in_channels)
        self.proj = nn.Linear(student_dim, proj_dim)

        if aggregator == 'gem':
            self.pool = SignedGeM(p=gem_p)
            self.desc_dim = proj_dim
        else:
            self.salad = SALAD(in_dim=proj_dim, cls_dim=student_dim,
                               num_clusters=num_clusters, cluster_dim=cluster_dim,
                               token_dim=token_dim, sinkhorn_iters=sinkhorn_iters)
            self.desc_dim = self.salad.desc_dim

        print(f"Student backbone dimension: {student_dim}")
        print(f"Projection dimension: {proj_dim}")
        print(f"Aggregator: {aggregator} (state-dict key "
              f"{'pool' if aggregator == 'gem' else 'salad'}), "
              f"descriptor {self.desc_dim}-d")

    def forward(self, x):
        x = self.input_norm(x)
        features = self.backbone.forward_features(x)
        patches = features[:, -self.num_patches:, :]
        projected_patches = self.proj(patches)

        if self.aggregator == 'gem':
            global_descriptor = self.pool(projected_patches)
        else:
            global_descriptor = self.salad(projected_patches, features[:, 0])

        return projected_patches, global_descriptor


def build_student(cfg):
    """EventViTStudent from a composed config: `model.*` plus the selected
    `model/aggregator` group, whose keys are passed straight through."""
    agg = cfg.model.aggregator
    return EventViTStudent(
        backbone_name=cfg.model.backbone_name,
        proj_dim=cfg.model.proj_dim,
        num_patches=cfg.model.num_patches,
        img_size=tuple(cfg.model.img_hw),
        in_channels=cfg.data.input_channels,
        aggregator=str(agg.name),
        gem_p=float(agg.get('p', 3.0)),
        num_clusters=int(agg.get('num_clusters', 64)),
        cluster_dim=int(agg.get('cluster_dim', 128)),
        token_dim=int(agg.get('token_dim', 256)),
        sinkhorn_iters=int(agg.get('sinkhorn_iters', 3)),
    )
