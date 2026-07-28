"""SALAD aggregation variant of the Phase 1 student.

Drop-in replacement for model.py's EventViTStudent that swaps GeM for SALAD
(Izquierdo & Civera, "Optimal Transport Aggregation for Visual Place
Recognition", CVPR 2024) -- the aggregator MegaLoc uses.

Why this can train on the EXISTING phase-1 objective, with no place labels:
compute_structural_loss() compares COSINE SIMILARITY MATRICES (B x B), which
are dimension-agnostic. So the student may emit an 8448-d SALAD descriptor
while the teacher target stays a 1024-d GeM vector -- the KL still matches
their batch geometry, and gradient flows into the SALAD projections. The AGFD
patch term is untouched (it acts on the projected patch tokens, before
aggregation).

Descriptor dim = num_clusters * cluster_dim + token_dim
                = 64 * 128 + 256 = 8448   (MegaLoc-comparable default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Sinkhorn / optimal transport (SuperGlue formulation, as used by SALAD)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SALAD head
# ---------------------------------------------------------------------------

class SALAD(nn.Module):
    """Optimal-transport aggregation over patch tokens.

    Args:
        in_dim:       token dimension fed in (we use the projected 1024-d
                      tokens so aggregation lives in the distilled space)
        cls_dim:      dimension of the backbone CLS token
        num_clusters: m (64 in SALAD/MegaLoc)
        cluster_dim:  l, tokens are projected DOWN to this before clustering
                      -- this is what keeps the descriptor at 8448 instead of
                      m * in_dim = 65536
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


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------

class EventViTStudentSALAD(nn.Module):
    """Same backbone / input BN / linear projection as EventViTStudent; only
    the aggregator differs. forward() keeps the (patches, global) contract so
    compute_losses() and evaluate_brisbane.process_traverse() work unchanged."""

    def __init__(self,
                 backbone_name='vit_small_patch16_dinov3.lvd1689m',
                 teacher_dim=1024,
                 num_patches=576,
                 img_size=(480, 640),
                 in_channels=3,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 sinkhorn_iters=3):
        super().__init__()

        self.teacher_dim = teacher_dim
        self.num_patches = num_patches
        self.image_size = img_size
        self.in_channels = in_channels

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )

        student_dim = self.backbone.embed_dim
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.proj = nn.Linear(student_dim, teacher_dim)
        self.salad = SALAD(
            in_dim=teacher_dim,
            cls_dim=student_dim,
            num_clusters=num_clusters,
            cluster_dim=cluster_dim,
            token_dim=token_dim,
            sinkhorn_iters=sinkhorn_iters,
        )

        print(f"Student backbone dimension: {student_dim}")
        print(f"Teacher dimension: {teacher_dim}")
        print(f"SALAD descriptor dimension: {self.salad.desc_dim} "
              f"({num_clusters} clusters x {cluster_dim} + {token_dim} token)")

    def forward(self, x):
        x = self.input_norm(x)
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]                              # (B, student_dim)
        patches = features[:, -self.num_patches:, :]
        projected_patches = self.proj(patches)                  # (B, N, teacher_dim)
        global_descriptor = self.salad(projected_patches, cls_token)
        return projected_patches, global_descriptor
