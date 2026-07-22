"""
contains functions to preprocess representations for evaluation on the Brisbane Event-VPR dataset.
This is similar to how the DSEC dataset is preprocessed for training.
"""

import cv2
import numpy as np
import torch

def _accumulate_events(x, y, sensor_hw):
    # per-pixel event count. np.bincount (C-level) is ~10-50x faster than
    # np.add.at and produces bit-identical counts.
    h, w = sensor_hw
    lin = y.astype(np.int64) * w + x.astype(np.int64)
    return np.bincount(lin, minlength=h * w).astype(np.float32).reshape(h, w)

def compute_hot_pixel_mask(x_r, y_r, sensor_hw, threshold=99.5):
    h, w = sensor_hw
    hist = _accumulate_events(x_r, y_r, sensor_hw)
    nonzero = hist[hist > 0]
    if len(nonzero) == 0:
        return np.zeros((h, w), dtype=bool)
    threshold_value = np.percentile(nonzero, threshold)
    return hist > threshold_value

def _norm_unit_max(arr):
    m = arr.max()
    return arr / m if m > 0 else arr

def _resize_to(arr, out_size):
    # out_size is (W, H) order for cv2.resize (square here, but explicit)
    return cv2.resize(arr, (out_size[0], out_size[1]),
                      interpolation=cv2.INTER_AREA)

def process_event_histogram(x_r, y_r, p, hot_mask, out_size, sensor_hw):
    """Channels: 0 pos count (unit-max), 1 neg count (unit-max),
    2 net polarity in [-1, 1], true zero where no events fired."""
    pos_mask = p == 1
    neg_mask = ~pos_mask

    pos_hist = _accumulate_events(x_r[pos_mask], y_r[pos_mask], sensor_hw)
    neg_hist = _accumulate_events(x_r[neg_mask], y_r[neg_mask], sensor_hw)

    pos_hist[hot_mask] = 0
    neg_hist[hot_mask] = 0

    net_hist = pos_hist - neg_hist
    net_max = np.max(np.abs(net_hist))
    if net_max > 0:
        net_hist /= net_max

    pos_hist = _norm_unit_max(pos_hist)
    neg_hist = _norm_unit_max(neg_hist)

    pos_hist = _resize_to(pos_hist, out_size)
    neg_hist = _resize_to(neg_hist, out_size)
    net_hist = _resize_to(net_hist, out_size)

    return np.stack([pos_hist, neg_hist, net_hist], axis=0)


def process_event_voxel_grid(x_r, y_r, p, t_us, t_start_us, t_end_us,
                             hot_mask, out_size, num_bins, sensor_hw):
    """num_bins temporal bins over the TRUE window [t_start_us, t_end_us];
    per bin pos/neg normalised together (shared max), combined as net."""
    voxels = np.zeros((num_bins, out_size[1], out_size[0]), dtype=np.float32)

    span = float(t_end_us - t_start_us)
    if span <= 0:
        return voxels

    bin_indices = ((t_us - t_start_us).astype(np.float64)
                   * num_bins / span).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    for b in range(num_bins):
        bin_mask = bin_indices == b
        pos_mask = bin_mask & (p == 1)
        neg_mask = bin_mask & (p == 0)

        pos_hist = _accumulate_events(x_r[pos_mask], y_r[pos_mask], sensor_hw)
        neg_hist = _accumulate_events(x_r[neg_mask], y_r[neg_mask], sensor_hw)

        pos_hist[hot_mask] = 0
        neg_hist[hot_mask] = 0

        m = max(pos_hist.max(), neg_hist.max())
        if m > 0:
            pos_hist /= m
            neg_hist /= m

        net_hist = pos_hist - neg_hist

        voxels[b] = _resize_to(net_hist, out_size)

    return voxels

def build_representation(x, y, t, p, t0, dt, modality, cfg):
    """x, y: int64 pixel coords; t: float64 epoch seconds (sorted);
    p: int64 {0,1}; t0/dt: bin start/duration in seconds.
    Returns float32 torch tensor (3, img_hw, img_hw), fp16 round-tripped."""
    sensor_hw = tuple(int(v) for v in cfg.datasets.sensor_hw)      # (260, 346)
    out_hw = int(cfg.model.img_hw[0])
    out_size = (out_hw, out_hw)                                     # (W, H)

    # back to relative integer microseconds (Brisbane parquet is natively
    # int64 epoch us, so np.round recovers the exact values)
    t_us = np.round((t - t0) * 1e6).astype(np.int64)
    t_end_us = int(round(dt * 1e6))

    # same per-window hot-pixel masking as training (applied on top of the
    # release's offline denoising -- the model was trained downstream of
    # this op, so it stays)
    hot_mask = compute_hot_pixel_mask(
        x, y, sensor_hw, threshold=float(cfg.datasets.hot_pixel_threshold))

    if modality == "histogram":
        rep = process_event_histogram(x, y, p, hot_mask, out_size, sensor_hw)
    elif modality == "voxel":
        rep = process_event_voxel_grid(
            x, y, p, t_us, 0, t_end_us, hot_mask, out_size,
            num_bins=int(cfg.datasets.voxel_bins), sensor_hw=sensor_hw)
    else:
        raise ValueError(f"Unknown modality {modality!r}")

    rep = rep.astype(np.float16).astype(np.float32)  # storage round-trip
    return torch.from_numpy(rep)
