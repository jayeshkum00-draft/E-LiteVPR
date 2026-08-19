"""Contiguous-block sampler: make the relational target carry information.

WHY
---
`compute_structural_loss` is a KL between two B x B similarity matrices. It can
only teach from pairs of frames that are ACTUALLY RELATED and are IN THE SAME
BATCH. train.py's WeightedRandomSampler draws every index independently from a
14k-frame corpus spanning ~53 sequences, so a batch of 32 is 32 unrelated
places and the teacher matrix is nearly the identity.

Measured on the real MegaLoc cache (scripts/probe_target_information.py),
B=32, tau=0.05:

    sampler                 diag     informative   eff_nbrs
    random (train)        0.9993        0.0007        24.8
    sequential (val)      0.9283        0.0717         4.1
    K=4 x M=8             0.9341        0.0659         3.2

Informative mass 0.0007 is not a metaphor: the MegaLoc run's train structural
loss converged to 0.0007, i.e. it hit the ceiling of its own objective in two
epochs and then trained on nothing for ten more. `eff_nbrs` matters as much as
the mass -- random batching at B=256 raises informative to 0.0104 but spreads
it over 125 fake neighbours, while blocks concentrate it on ~3 real ones.
Bigger batches and higher tau do NOT substitute for block structure; they
manufacture diffuse mush (which is separately what the DINOv3 target already
was: off-diagonal cosine 0.96, eff_nbrs 29.6 of 31).

WHAT IT DOES
------------
Yields indices in contiguous runs of `block` frames from one sequence, so each
batch is (batch_size // block) short clips instead of batch_size unrelated
frames. Nothing else changes -- same DataLoader, same loop, same batch size.

Contiguity is by TIMESTAMP, not by position in pairs.txt: indices are grouped
per sequence and sorted on `timestamp_us`, so a block is temporally adjacent
even if the master pairs.txt was concatenated in some other order.

Day/night balance is preserved but moved to the BLOCK level -- picking a night
block with probability `night_frac` -- because balancing individual frames
would break contiguity.

LIMIT, STATED PLAINLY
---------------------
Contiguous frames within one traverse are the SAME CONDITION. This fixes "the
loss has nothing to optimise"; it does not by itself create cross-condition
invariance. That needs repeat traverses (M3ED day/night over one route) so a
batch can hold the same place under two illuminations, at which point a
VPR-trained teacher labels them as the same place with no GPS involved.
"""

import os

import torch
from torch.utils.data import Sampler


class BlockSampler(Sampler):
    """Contiguous blocks of `block` frames, balanced day/night per block.

    batch_size MUST be a multiple of `block`, otherwise the DataLoader slices
    across block boundaries and the batch geometry is not what was measured.
    `install_dataloader_guard` enforces that at construction time.
    """

    def __init__(self, dataset, night_seqs, block=8, night_frac=0.5,
                 num_samples=None, generator=None, verbose=True):
        self.block = int(block)
        self.night_frac = float(night_frac)
        self.generator = generator
        n = num_samples if num_samples is not None else len(dataset)
        self.n_blocks = max(1, int(n) // self.block)

        # group dataset indices by sequence, ordered in time
        by_seq = {}
        for idx, pair in enumerate(dataset.pairs):
            by_seq.setdefault(pair["sequence"], []).append(
                (pair["timestamp_us"], idx))

        night_seqs = set(night_seqs or [])
        self.day_groups, self.night_groups = [], []
        dropped = []
        for seq, rows in by_seq.items():
            rows.sort(key=lambda r: r[0])
            idxs = torch.tensor([r[1] for r in rows], dtype=torch.long)
            if idxs.numel() < self.block:
                dropped.append(seq)
                continue
            (self.night_groups if seq in night_seqs
             else self.day_groups).append(idxs)

        if not self.day_groups and not self.night_groups:
            raise ValueError(
                f"No sequence has at least block={self.block} frames. "
                f"Lower BLOCK_M or check the split lists.")
        if not self.night_groups:
            self.night_frac = 0.0
        if not self.day_groups:
            self.night_frac = 1.0

        if verbose:
            print(f"Block sampler: block={self.block}, "
                  f"night_frac={self.night_frac:.2f}, "
                  f"{len(self.day_groups)} day / {len(self.night_groups)} "
                  f"night sequences, {self.n_blocks} blocks/epoch "
                  f"({self.n_blocks * self.block} samples)")
            if dropped:
                print(f"  dropped {len(dropped)} sequences shorter than "
                      f"block: {sorted(dropped)[:5]}"
                      f"{' ...' if len(dropped) > 5 else ''}")

    def __len__(self):
        return self.n_blocks * self.block

    def __iter__(self):
        g = self.generator
        if g is None:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        def pick(seq_list):
            j = int(torch.randint(len(seq_list), (1,), generator=g).item())
            idxs = seq_list[j]
            hi = idxs.numel() - self.block + 1
            s = int(torch.randint(hi, (1,), generator=g).item())
            return idxs[s:s + self.block]

        for _ in range(self.n_blocks):
            use_night = (torch.rand((), generator=g).item() < self.night_frac)
            pool = self.night_groups if use_night else self.day_groups
            yield from pick(pool).tolist()


def build_block_sampler(dataset, night_seqs):
    """Drop-in replacement for train.build_day_night_sampler.

    Same signature, so it can be rebound without touching train.py. Reads
    BLOCK_M and NIGHT_FRAC from the environment to stay config-free.
    """
    return BlockSampler(
        dataset, night_seqs,
        block=int(os.environ.get("BLOCK_M", 8)),
        night_frac=float(os.environ.get("NIGHT_FRAC", 0.5)),
    )


# The DataLoader guard that enforces batch_size % block == 0 lives in
# train_hooks.install(), together with the cfg capture and the LR schedule, so
# `train` is patched from exactly one place.
