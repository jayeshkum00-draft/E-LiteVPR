"""The train_megaloc_edits.py recipe with the DINOv3 teacher instead of MegaLoc.

ONE DIFFERENCE, BY CONSTRUCTION
-------------------------------
This file calls `train_megaloc_edits.run()` -- the same loop, the same
structural KL (mask diagonal + z-score, fp32), the same block sampler, the same
warmup/cosine schedule, the same checkpointing. The only change is which cached
global descriptor is read:

    megaloc   <teacher_dir>/<seq>/megaloc.npy   8448-d, cache_megaloc.py
    dinov3    <teacher_dir>/<seq>/gem.npy       1024-d, cache_teacher_gem.py

`gem.npy` is GeM(p=3) over the SAME cached DINOv3 patches that train.py pools in
the loop (`cache_teacher_gem.py:50-53`), so this is the DINOv3-GeM target, just
read at 2 KB/frame instead of 1.18 MB/frame.

The KL compares each side's own B x B similarity matrix, so the teacher's width
never has to match the student's -- `model.teacher_dim` stays 1024 for the
student's per-patch projection regardless of which cache is used.

`model.pooling` selects the STUDENT's aggregation, exactly as in the megaloc
arm. `model.teacher_pooling` does nothing here: gem.npy was pooled once, at
cache time, with the clamped GeM. Testing a signed TARGET means writing a
second cache, not passing a flag.

KNOWN RISK, STATED UP FRONT
---------------------------
This recipe -- mask_diag + z-score at tau=1.0, with block sampling -- is
MegaLoc-specific in the evidence so far. It has been refuted twice on DINOv3:
37.27 -> 31.27 under random sampling, and 45.13 -> 18.19 as teacher B under
block sampling. DINOv3's near-uniform raw target appears to act as a smoothing
prior that long sequence matching integrates, and sharpening it has cost recall
both times. Run this as the matched comparison it is, and read the result on
per-frame L=1 and on probe_all's cross-condition d'/retention rather than on the
day-night sequence mean.

COMMAND (Kaggle paths -- see the `kaggle-run-layout` memory note)
----------------------------------------------------------------
  python scripts/train_dino_edits.py \
    datasets=dsec datasets@datasets_extra=m3ed \
    datasets.root_dir=$DSEC_ROOT   datasets.output_dir=$DSEC_FEATS \
    datasets_extra.root_dir=$M3ED_ROOT datasets_extra.output_dir=$M3ED_FEATS \
    training.patch_loss_weight=0.0 training.temperature=1.0 \
    +edits.teacher_dir=$GEM_DIR \
    model.pooling=clamp \
    checkpoint_dir=/kaggle/working/ckpt_dino_clamp \
    output_dir=/kaggle/working/hydra_dino_clamp

`$GEM_DIR` is the DINOv3 cache (feats/gem), NOT feats/megaloc. Swap
`model.pooling=signed` with its own checkpoint_dir for the other arm. No
HYDRA_MAIN_MODULE needed -- this file owns its own @hydra.main.
"""

import hydra
from omegaconf import DictConfig

import train_megaloc_edits as E


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    E.run(cfg, "dinov3", "gem.npy", "cache_teacher_gem.py")


if __name__ == "__main__":
    main()
