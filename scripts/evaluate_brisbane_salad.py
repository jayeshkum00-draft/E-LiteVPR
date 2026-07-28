"""Brisbane evaluation for the SALAD student.

Reuses evaluate_brisbane.py wholesale -- retrieval, GT, sequence matching,
stationary filtering and CSV output are identical, so SALAD vs GeM is a
like-for-like comparison. Only build_model() is swapped.

process_traverse() already takes `out[1]` from the model tuple, so the 8448-d
SALAD descriptor flows through unchanged; the descriptor cache is tagged
"salad" and will not collide with existing GeM caches.

Run:
    python scripts/evaluate_brisbane_salad.py datasets=brisbane \
        phase1_weights=/kaggle/working/checkpoints/best_phase1_salad_histogram.pth
"""

import torch

import evaluate_brisbane
from model_salad import EventViTStudentSALAD


def build_model_salad(cfg, device):
    """Mirrors train_salad.py's construction exactly."""
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
    )

    weights = cfg.phase1_weights
    print(f"Loading SALAD weights: {weights}")
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    return model.to(device), "salad"


# evaluate_brisbane.main() calls build_model() from its own module namespace,
# so rebinding the name here is enough to redirect it.
evaluate_brisbane.build_model = build_model_salad


if __name__ == "__main__":
    evaluate_brisbane.main()
