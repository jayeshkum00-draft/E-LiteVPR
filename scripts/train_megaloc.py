"""SUPERSEDED by train_ext.py.

Everything this did is now a flag, so teacher choice, block sampling and the
LR schedule can be ablated independently instead of being welded into a
separate script. Equivalent command:

    HYDRA_MAIN_MODULE=__main__ python scripts/train_ext.py \
        +training.teacher=megaloc \
        +training.megaloc_dir=<dir written by cache_megaloc.py> \
        training.patch_loss_weight=0

This file intentionally fails loudly rather than staying runnable, so an old
notebook cell cannot silently launch a configuration that is no longer the one
described in the logs.
"""

raise SystemExit(__doc__)
