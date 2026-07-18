"""Phase 2 triplet dataset over slice h5s + mined indexes (MVSEC + NYC).

Each __getitem__ returns an (anchor, positive) representation pair plus the
geometry metadata the loss needs to mask in-batch negatives:
  - positives come from the miner's CSR lists (cross-condition drawn with
    probability `cross_quota` when available);
  - negatives are found *online* inside the batch: item j is a valid
    negative for item i if they come from different datasets (different
    cities), or share a registration frame and are farther apart than the
    index's neg_floor. Same-dataset pairs in unrelated frames are excluded
    (geometric relation unknown -- e.g. MVSEC night1 vs the day2 group).

Representations reuse the exact Phase 1 / eval pipeline
(brisbane_representation): per-window hot-pixel mask, histogram or voxel,
resized to the student's 384x384 input. A 50 ms training window is cut at a
random offset inside the stored slice (jitter; NYC slices are 100 ms so the
jitter is 0-50 ms, MVSEC 1200 ms so 0-1150 ms). Noise domain randomization
(event drop, BA noise, hot pixels, mains flicker) runs on raw events before
histogramming.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from brisbane_representation import (compute_hot_pixel_mask,
                                     process_event_histogram,
                                     process_event_voxel_grid)

class _Source:
    """One dataset (index npz + slice h5 dir); h5 handles opened lazily
    per dataloader worker."""

    def __init__(self, name, cfg_src):
        self.name = name
        self.slices_dir = Path(cfg_src.slices_dir)
        self.sensor_hw = tuple(int(v) for v in cfg_src.sensor_hw)
        idx = np.load(cfg_src.index, allow_pickle=False)
        self.xy = idx["xy"].astype(np.float64)
        self.seq_id = idx["seq_id"]
        self.is_night = idx["is_night"]
        self.local_index = idx["local_index"]
        self.seq_files = [str(s) for s in idx["seq_files"]]
        self.seq_frames = [str(s) for s in idx["seq_frames"]]
        self.ps_ptr, self.ps_idx = idx["pos_same_indptr"], idx["pos_same_indices"]
        self.pc_ptr, self.pc_idx = idx["pos_cross_indptr"], idx["pos_cross_indices"]
        self.neg_floor = float(idx["neg_floor"])
        excl = set(cfg_src.get("exclude_seqs", []) or [])
        unknown = excl - set(self.seq_files)
        if unknown:
            raise ValueError(f"{name}: exclude_seqs not in index: {unknown}")
        self.excluded = np.isin(self.seq_id,
                                [self.seq_files.index(e) for e in excl])
        self._h5 = {}

    def events(self, gid):
        """All events of slice `gid` (local arrays: x, y, t_us, p)."""
        f = self._file(int(self.seq_id[gid]))
        i = int(self.local_index[gid])
        a, b = f["slice_offsets"][i], f["slice_offsets"][i + 1]
        return (f["events_x"][a:b].astype(np.int64),
                f["events_y"][a:b].astype(np.int64),
                f["events_t_us"][a:b].astype(np.int64),
                f["events_p"][a:b].astype(np.int64))

    def slice_ms(self, gid):
        return float(self._file(int(self.seq_id[gid])).attrs["slice_ms"])

    def _file(self, seq):
        if seq not in self._h5:
            self._h5[seq] = h5py.File(self.slices_dir / self.seq_files[seq],
                                      "r")
        return self._h5[seq]

# noise randomization

def _rand_events(n, hw, w_us, rng):
    return (rng.integers(0, hw[1], n), rng.integers(0, hw[0], n),
            rng.integers(0, w_us, n), rng.integers(0, 2, n))


def augment_events(x, y, t_us, p, hw, w_us, rng, aug):
    """Event-level domain randomization; returns new arrays (unsorted --
    histogram/voxel binning does not require time order)."""
    if aug is None:
        return x, y, t_us, p
    parts_x, parts_y, parts_t, parts_p = [x], [y], [t_us], [p]

    # darkness simulation: low light collapses signal events while the
    # noise floor persists -> aggressive signal drop + full-strength BA
    dark = rng.random() < float(aug.get("dark_p", 0.0))
    if dark:
        lo, hi = (float(v) for v in aug.dark_drop)
        drop = rng.uniform(lo, hi)
    else:
        drop = rng.uniform(0.0, float(aug.drop_max))
    if drop > 0 and len(x):
        keep = rng.random(len(x)) >= drop
        parts_x[0], parts_y[0] = x[keep], y[keep]
        parts_t[0], parts_p[0] = t_us[keep], p[keep]

    n_px = hw[0] * hw[1]
    w_s = w_us * 1e-6

    ba_lo = 0.5 * float(aug.ba_rate_max) if dark else 0.0
    n_ba = rng.poisson(rng.uniform(ba_lo, float(aug.ba_rate_max)) * n_px * w_s)
    if n_ba:
        bx, by, bt, bp = _rand_events(int(n_ba), hw, w_us, rng)
        parts_x.append(bx); parts_y.append(by)
        parts_t.append(bt); parts_p.append(bp)

    for _ in range(rng.integers(0, int(aug.hot_pixels_max) + 1)):
        rate = rng.uniform(*[float(v) for v in aug.hot_rate])
        n = rng.poisson(rate * w_s)
        if n:
            hx, hy = rng.integers(0, hw[1]), rng.integers(0, hw[0])
            parts_x.append(np.full(n, hx)); parts_y.append(np.full(n, hy))
            parts_t.append(rng.integers(0, w_us, n))
            parts_p.append(np.full(n, rng.integers(0, 2)))

    if rng.random() < float(aug.flicker_p):
        freq = float(rng.choice([float(v) for v in aug.flicker_freq]))
        n = rng.poisson(rng.uniform(0.0, float(aug.flicker_rate_max))
                        * n_px * w_s * 0.5)          # on-phase duty ~0.5
        if n:
            fx, fy, ft, fp = _rand_events(int(n), hw, w_us, rng)
            period = 1e6 / freq
            # fold times into the on-half of each flicker period
            ft = ((ft // period).astype(np.int64) * period
                  + (ft % period) * 0.5).astype(np.int64)
            parts_x.append(fx); parts_y.append(fy)
            parts_t.append(np.clip(ft, 0, w_us - 1)); parts_p.append(fp)

    return (np.concatenate(parts_x), np.concatenate(parts_y),
            np.concatenate(parts_t), np.concatenate(parts_p))

# datasets

def build_rep(x, y, t_us, p, w_us, hw, modality, out_hw, hot_thr, voxel_bins):
    hot = compute_hot_pixel_mask(x, y, hw, threshold=hot_thr)
    if modality == "histogram":
        rep = process_event_histogram(x, y, p, hot, (out_hw, out_hw), hw)
    elif modality == "voxel":
        rep = process_event_voxel_grid(x, y, p, t_us, 0, w_us, hot,
                                       (out_hw, out_hw), voxel_bins, hw)
    else:
        raise ValueError(f"unknown modality {modality!r}")
    rep = rep.astype(np.float16).astype(np.float32)   # fp16 storage parity
    return torch.from_numpy(rep)

class Phase2PairDataset(Dataset):
    """Anchor-positive pairs; index space = anchors with >=1 usable
    positive after exclusions. Deterministic per (epoch, idx) via seed."""

    def __init__(self, d_cfg, t_cfg, modality, out_hw, seed=0):
        self.sources = [_Source(k, d_cfg.sources[k]) for k in d_cfg.sources]
        self.modality = modality
        self.out_hw = int(out_hw)
        self.hot_thr = float(d_cfg.hot_pixel_threshold)
        self.voxel_bins = int(d_cfg.voxel_bins)
        self.window_us = int(float(t_cfg.window_ms) * 1e3)
        self.cross_quota = float(t_cfg.cross_quota)
        self.aug = t_cfg.aug
        self.seed = seed
        self.epoch = 0

        # global anchor table: (source_idx, gid, usable pos lists)
        self.anchors = []
        self.anchor_night = []
        for si, src in enumerate(self.sources):
            ok = ~src.excluded
            for g in np.flatnonzero(ok):
                same = src.ps_idx[src.ps_ptr[g]:src.ps_ptr[g + 1]]
                cross = src.pc_idx[src.pc_ptr[g]:src.pc_ptr[g + 1]]
                same = same[~src.excluded[same]]
                cross = cross[~src.excluded[cross]]
                if len(same) or len(cross):
                    self.anchors.append((si, g, same, cross))
                    self.anchor_night.append(bool(src.is_night[g]))
        self.anchor_night = np.array(self.anchor_night)
        if not len(self.anchors):
            raise RuntimeError("no anchors with positives after exclusions")
        # frame-group ids unique across sources for the negative mask
        self._gid_of = {}
        for si, src in enumerate(self.sources):
            for fr in set(src.seq_frames):
                self._gid_of[(si, fr)] = len(self._gid_of)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.anchors)

    def _load(self, si, gid, rng, augment):
        src = self.sources[si]
        x, y, t, p = src.events(gid)
        stored_us = int(src.slice_ms(gid) * 1e3)
        off = rng.integers(0, max(stored_us - self.window_us, 0) + 1) \
            if augment else max(stored_us - self.window_us, 0) // 2
        m = (t >= off) & (t < off + self.window_us)
        x, y, t, p = x[m], y[m], t[m] - off, p[m]
        if augment:
            x, y, t, p = augment_events(x, y, t, p, src.sensor_hw,
                                        self.window_us, rng, self.aug)
        return build_rep(x, y, t, p, self.window_us, src.sensor_hw,
                         self.modality, self.out_hw, self.hot_thr,
                         self.voxel_bins)

    def _meta(self, si, gid):
        src = self.sources[si]
        fr = src.seq_frames[int(src.seq_id[gid])]
        return (np.float32(src.xy[gid]), self._gid_of[(si, fr)], si,
                np.float32(src.neg_floor))

    def __getitem__(self, idx):
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + self.epoch) * 2_000_003 + idx)
        si, g, same, cross = self.anchors[idx]
        pool = cross if (len(cross) and (not len(same) or
                                         rng.random() < self.cross_quota)) \
            else same
        pos = int(pool[rng.integers(0, len(pool))])

        reps = torch.stack([self._load(si, g, rng, True),
                            self._load(si, pos, rng, True)])
        xy_a, gid_a, ds_a, fl_a = self._meta(si, g)
        xy_p, gid_p, ds_p, fl_p = self._meta(si, pos)
        return dict(
            reps=reps,                                        # (2, C, H, W)
            xy=torch.from_numpy(np.stack([xy_a, xy_p])),      # (2, 2)
            group=torch.tensor([gid_a, gid_p]),
            dataset=torch.tensor([ds_a, ds_p]),
            floor=torch.tensor([fl_a, fl_p]),
            is_cross=torch.tensor(bool(self.sources[si].is_night[g])
                                  != bool(self.sources[si].is_night[pos])))

class Phase2ProbeDataset(Dataset):
    """Deterministic center-window descriptors for the val probe
    (query seq -> db seq retrieval, R@1 within radius)."""

    def __init__(self, d_cfg, t_cfg, modality, out_hw):
        p = d_cfg.val_probe
        self.src = _Source(str(p.source), d_cfg.sources[p.source])
        self.modality = modality
        self.out_hw = int(out_hw)
        self.hot_thr = float(d_cfg.hot_pixel_threshold)
        self.voxel_bins = int(d_cfg.voxel_bins)
        self.window_us = int(float(t_cfg.window_ms) * 1e3)
        q = self.src.seq_files.index(str(p.query_seq))
        d = self.src.seq_files.index(str(p.db_seq))
        self.ids = np.flatnonzero(np.isin(self.src.seq_id, [q, d]))
        self.is_query = self.src.seq_id[self.ids] == q
        self.xy = self.src.xy[self.ids]
        self.radius = float(p.radius)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        gid = int(self.ids[i])
        x, y, t, p = self.src.events(gid)
        stored_us = int(self.src.slice_ms(gid) * 1e3)
        off = max(stored_us - self.window_us, 0) // 2
        m = (t >= off) & (t < off + self.window_us)
        rep = build_rep(x[m], y[m], t[m] - off, p[m], self.window_us,
                        self.src.sensor_hw, self.modality, self.out_hw,
                        self.hot_thr, self.voxel_bins)
        return rep, i

    def recall_at_1(self, desc):
        """desc: (len(self), D) tensor in this dataset's order."""
        dq, dd = desc[self.is_query], desc[~self.is_query]
        xq, xd = self.xy[self.is_query], self.xy[~self.is_query]
        nn = torch.cdist(dq, dd).argmin(dim=1).cpu().numpy()
        geo = np.linalg.norm(xq - xd[nn], axis=1)
        return float((geo <= self.radius).mean())
