"""Distil MegaLoc's pairwise geometry instead of DINOv3's, student unchanged.

Runs the real training loop from train.py with exactly two things rebound:

  1. `train.E_LiteVPRDataset` -> a subclass that serves the cached MegaLoc
     descriptor in the teacher slot (from MEGALOC_DIR, written by
     cache_megaloc.py) instead of the DINOv3 patch grid.
  2. `train.compute_losses`   -> a structural-only version.

THE STUDENT IS BYTE-IDENTICAL to the model that scored 82.33 / 37.27 on
Brisbane. That is possible because compute_structural_loss reduces BOTH sides
to a B x B matrix of pairwise cosines before comparing them:

    student_sim = f_s @ f_s.T / T        # from d_s-dim descriptors
    teacher_sim = f_t @ f_t.T / T        # from d_t-dim descriptors
    KL(log_softmax(student_sim) || softmax(teacher_sim))

so the teacher's 8448-d and the student's 1024-d never meet. MegaLoc's width
exists only at training time; nothing about the deployed 21.98 M-parameter
ViT-S or its 1024-d descriptor changes. (With the patch loss off, `teacher_dim`
is no longer pinned to the teacher's width at all -- it is now just the
student's descriptor size, and 512 or 256 are free ablation choices.)

WHY compute_losses IS REPLACED RATHER THAN LEFT ALONE
train.compute_losses ALWAYS evaluates the patch term (line 95/97) even when
patch_weight=0, so it would compute F.mse_loss between (B, 576, 1024) student
patches and a (B, 1, 8448) teacher vector and die on the shape. Multiplying by
zero afterwards does not save it. The replacement also never calls
`teacher_gem`: GeM does `x.clamp(min=eps).pow(p)`, which would destroy the
negative components of a MegaLoc descriptor.

Run:
    MEGALOC_DIR=/kaggle/working/megaloc_features \
    HYDRA_MAIN_MODULE=__main__ \
    python scripts/train_megaloc.py \
        datasets.root_dir=... datasets.output_dir=... \
        hydra.run.dir=/kaggle/working/hy

HYDRA_MAIN_MODULE=__main__ is required for every wrapper of this shape:
hydra unwraps the decorated function and reads `__module__`, which is "train",
not "__main__", so it resolves config_path "../configs" as a PACKAGE and fails
with "Primary config module 'configs' not found". The env var forces the
file-based branch (hydra/_internal/utils.py:45-53).

`training.patch_loss_weight=0` is REQUIRED (configs/training/phase1.yaml
defaults it to 1.0); a non-zero value raises, since there is no patch target.
"""

import os
from pathlib import Path

import numpy as np
import torch

import train
from dataset import E_LiteVPRDataset


def _megaloc_dir():
    d = os.environ.get("MEGALOC_DIR")
    if not d:
        raise SystemExit(
            "Set MEGALOC_DIR to the directory written by cache_megaloc.py "
            "(it is separate from datasets.output_dir because the teacher "
            "feature cache is read-only on Kaggle).")
    return Path(d)


class MegaLocDataset(E_LiteVPRDataset):
    """Serves (megaloc_desc[None, :], dummy_attn) in place of (patches, attn).

    Only `_get_features` is overridden, so pairs.txt parsing, the frames.txt
    row index, sequence filtering and the fail-fast probe in __init__ are the
    originals -- including its `patches.ndim == 2 and attn.ndim == 1` assertion,
    which the (1, D) / (1,) shapes below satisfy.
    """

    def _get_features(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            path = _megaloc_dir() / seq / "megaloc.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found -- run cache_megaloc.py for this corpus "
                    f"first. Sequence names must match the teacher cache.")
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


def compute_losses_megaloc(model_out, teacher_desc, teacher_attn, teacher_gem,
                           use_agfd, structural_weight, temperature,
                           patch_weight=1.0):
    """Structural-only, against a precomputed global teacher descriptor.

    Signature and 3-tuple return match train.compute_losses so train_epoch and
    validate_epoch call it unchanged; `patch_loss` is reported as a constant 0
    and shows up that way in the logs and in wandb.
    """
    if patch_weight:
        raise ValueError(
            f"patch_loss_weight={patch_weight} but a MegaLoc global descriptor "
            f"has no patch target. Leave it at 0.")
    _student_patches, student_global = model_out
    teacher_global = teacher_desc.squeeze(1)              # (B, 1, D) -> (B, D)
    structural = train.compute_structural_loss(
        student_global, teacher_global, temperature)
    return structural_weight * structural, structural.new_zeros(()), structural


train.E_LiteVPRDataset = MegaLocDataset
train.compute_losses = compute_losses_megaloc

if __name__ == "__main__":
    print(f"teacher: MegaLoc descriptors from {_megaloc_dir()} (structural-only)")
    train.main()
