"""Run the real training loop with per-sample input normalisation.

Identical to `python scripts/train.py ...` in every respect except that
`EventViTStudent.input_norm` is a per-sample normaliser instead of
BatchNorm2d. train.py is imported, not copied, so the losses, sampler,
scheduler-free optimiser, checkpointing and wandb logging are the same code
that produced every recorded result.

    INPUT_NORM=instance uv run python scripts/train_insnorm.py \
        training.patch_loss_weight=0 hydra.run.dir=/kaggle/working/hy

INPUT_NORM is instance|layer|none (default instance). Keep every other
override identical to the structural-only run so this stays one variable.
"""

import train
import model_insnorm

model_insnorm.install(train)

if __name__ == "__main__":
    train.main()
