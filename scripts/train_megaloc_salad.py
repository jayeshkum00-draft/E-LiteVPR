"""SALAD-aggregated student on the CURRENT MegaLoc objective.

Why this file exists rather than train_salad.py
-----------------------------------------------
train_salad.py is built on train.py: DINOv3 patch cache, day/night
WeightedRandomSampler, raw structural loss with no mask_diag/standardize, no
block sampler, no cached-MegaLoc teacher_dir. Running it would not be running
the objective this paper reports.

train_megaloc_edits.py IS that objective. It builds the student at line 357
from the module-global name `EventViTStudent` (imported line 116), so
rebinding that name is enough to swap the aggregator -- nothing in the
training script is edited, and every flag (`+edits.*`, teacher_dir, block
sampler, scheduler) behaves identically.

SALAD hyper-parameters
----------------------
The construction site passes only the kwargs EventViTStudent takes, so the
SALAD-specific ones cannot be read from cfg there. They come from the
environment instead -- the same pattern block_sampler.build_block_sampler
uses for BLOCK_M / NIGHT_FRAC.

    descriptor dim = SALAD_CLUSTERS * SALAD_CLUSTER_DIM + SALAD_TOKEN_DIM

    default (MegaLoc-comparable)  64 * 128 + 256 = 8448
    compact                       64 *  16 + 256 = 1280

`model.pooling` is accepted and ignored: SALAD has no pooling variant.

Run (no HYDRA_MAIN_MODULE -- train_megaloc_edits owns its own @hydra.main and
config_path resolves relative to that module, which is this same directory):
    SALAD_CLUSTER_DIM=128 \
        python scripts/train_megaloc_salad.py <same overrides as the GeM run>

Checkpoints: pass a distinct checkpoint_dir per width. The 8448-d and 1280-d
students are different shapes and will not load into each other, and neither
loads into the GeM student.
"""

import os

import train_megaloc_edits as E
from model_salad import EventViTStudentSALAD

_SALAD = {
    "num_clusters": ("SALAD_CLUSTERS", 64),
    "cluster_dim": ("SALAD_CLUSTER_DIM", 128),
    "token_dim": ("SALAD_TOKEN_DIM", 256),
    "sinkhorn_iters": ("SALAD_SINKHORN", 3),
}


def _salad_params():
    return {k: int(os.environ.get(env, default))
            for k, (env, default) in _SALAD.items()}


def _build(*args, pooling=None, **kwargs):
    """Drop-in for EventViTStudent. `pooling` is swallowed on purpose."""
    p = _salad_params()
    dim = p["num_clusters"] * p["cluster_dim"] + p["token_dim"]
    print(f"SALAD aggregator: {p['num_clusters']} clusters x "
          f"{p['cluster_dim']} + {p['token_dim']} token = {dim}-d descriptor")
    if pooling is not None:
        print(f"  (model.pooling={pooling!r} ignored -- SALAD has no pooling "
              f"variant)")
    return EventViTStudentSALAD(*args, **kwargs, **p)


E.EventViTStudent = _build

main = E.main

if __name__ == "__main__":
    main()
