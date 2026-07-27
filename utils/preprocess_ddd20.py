"""
DDD20 preprocessor for E-LiteVPR.

Reads raw ddd20-utils recordings (recXXXXXXXXXX.hdf5 -- see
~/Desktop/ddd20_h5_reading_guide.md.pdf for the on-disk layout) and writes
paired APS/event representations in the same output format as
preprocess_dsec.py: <seq>/rgb/ + <seq>/events/histogram/ .npy files plus a
master pairs.txt. Lines keep the 6-column DSEC format (timestamp_us, rgb,
histogram, voxel, timesurface_net, timesurface_zero -- paths relative to the
sequence dir, as dataset.py/feature_extractor.py expect) but only the
histogram representation is materialized; the voxel/timesurface columns are
placeholder paths that are never written (dataset.py only opens the column
it is asked for).

Differences from DSEC, because DDD20 is a single DAVIS346 chip (APS and DVS
share one pixel array -- no second RGB camera, no calibration, no
rectification):
  * APS frames are subsampled to a target rate (default 1 Hz) by comparing
    frame TIMESTAMPS, not frame indices -- DDD20's frame rate is not fixed,
    so an index stride would drift.
  * Frames are exposure-filtered (mean intensity band + clipped-pixel
    fraction on the 8-bit frame); a frame that fails is skipped and the next
    candidate frame is tried, without resetting the target-rate clock.
  * For each KEPT frame at time t, the paired event window is the preceding
    [t - event_window_ms, t] (not the span between two consecutive kept
    frames, since consecutive kept frames here are seconds apart).

Decode logic (header struct, frame payload layout, polarity event bit
packing) mirrors ddd20-utils/interfaces/caer.py's unpack_header /
unpack_frame / unpack_events exactly -- see docstrings below for the mapping
to that file. Do not re-derive the bit packing from scratch; if ddd20-utils
changes, re-derive from there.

Timestamps -- verified against a real 1 GB recording, not assumed: each dvs
row carries TWO different clocks. `dvs/timestamp` (the h5 column) is the
HOST's wall-clock arrival time for the whole packet (epoch microseconds --
the same clock as every CAN-bus/GPS channel in the file). The per-event/
per-frame ticks decoded from the payload (caer.py's `unpack_events`/
`unpack_frame`) are a separate DEVICE-side free-running microsecond counter.
Checking a real recording end-to-end showed the device counter's total span
did NOT match the host clock's span even though no ts-reset special event
(type 0) was ever seen -- i.e. the device clock silently re-initializes
(consistent with ddd20-utils' own datasets_ioerrors.txt: recordings restart
the sensor after I/O errors) without emitting the marker export.py relies on
to detect it. So the device counter is NOT trustworthy as a global timeline
across a whole file and is only used LOCALLY, within one packet (~10 ms
span), for sub-packet relative event spacing:
  * frame timestamp = the packet's own host arrival time (`ts[i]`) -- a
    single instant, so no sub-packet adjustment is needed.
  * each polarity event's timestamp = linearly interpolated between the
    CURRENT packet's host arrival time and the NEXT packet's host arrival
    time, using the event's fractional position (by device tick) within its
    packet. This keeps every timestamp on the one reliable, globally
    monotonic clock while still spreading a packet's ~1000+ batched events
    across its ~10 ms instead of collapsing them to one instant.

Usage (run with the `ddd20` datasets group, since config.yaml defaults to `dsec`):
    uv run python utils/preprocess_ddd20.py datasets=ddd20 \
        datasets.ddd20_path=/path/to/ddd20/recordings output_dir=/path/to/out
"""

import os
import struct
from pathlib import Path

import cv2
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig

from preprocess_dsec import (
    compute_hot_pixel_mask,
    process_event_histogram,
)

# --- raw cAER packet decoding -------------------------------------------------
# Struct layout matches caer_event_packet_header (28 bytes) as unpacked by
# ddd20-utils/interfaces/caer.py:unpack_header via struct.unpack('hhiiiiii', ...):
# etype, esource, esize, eoffset, eoverflow, ecapacity, enumber, evalid.
HEADER_STRUCT = struct.Struct('hhiiiiii')
FRAME_SHAPE = (260, 346)  # DAVIS346 (H, W)

EVENT_TYPE_SPECIAL = 0
EVENT_TYPE_POLARITY = 1
EVENT_TYPE_FRAME = 2
EVENT_TYPE_IMU6 = 3


def _iter_dvs_rows(h5file):
    """Yield (etype, esize, ecapacity, ts_us, payload_bytes) for every
    non-padding row of the dvs group. ts[i] == 0 marks a padding row from
    the chunked/padded storage (see ddd20-utils/view.py's ts==0 checks)."""
    d = h5file['dvs']['data'][:]
    ts = h5file['dvs']['timestamp'][:]
    for i in range(len(ts)):
        if ts[i] == 0:
            continue
        row = d[i]
        etype, esource, esize, eoffset, eoverflow, ecapacity, enumber, evalid = \
            HEADER_STRUCT.unpack(row[1].tobytes())
        yield etype, esize, ecapacity, int(ts[i]), row[2].tobytes()


def _decode_frame(payload):
    """FRAME_EVENT payload -> image uint16 (260, 346). Mirrors
    caer.py:unpack_frame's payload layout (36-byte header + raw 16-bit pixel
    buffer), but the frame's own header timestamp (unpack_frame's
    img_head[2]) is a device-clock tick, not the host clock -- see the
    module docstring. We use the packet's host arrival time instead, so it
    isn't extracted here."""
    img = np.frombuffer(payload[36:], dtype='<u2').reshape(FRAME_SHAPE)
    # DDD20's raw stream stores rows bottom-up (jAER/DAVIS convention,
    # y origin at bottom; ddd20-utils export.py leaves it as-is). Flip to
    # top-down so the teacher sees upright scenes like DSEC. Verified on a
    # real recording: flipud (no horizontal mirror) puts oncoming traffic on
    # the left, consistent with US right-hand driving; events get the
    # matching y flip in _decode_polarity.
    return img[::-1]


def _decode_polarity(payload, esize, ecapacity):
    """POLARITY_EVENT payload -> (tick, x, y, p) arrays. `tick` is the raw
    per-event DEVICE-clock counter (mirrors caer.py:unpack_events' second
    uint32 word) -- monotonic within this one packet, but not usable as a
    global timeline (see module docstring); callers use it only for
    intra-packet relative spacing via _assign_event_times."""
    if ecapacity == 0 or esize == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, empty_i, np.empty(0, dtype=np.uint8)
    p_arr = np.frombuffer(payload, dtype='<u4').reshape(ecapacity, esize // 4)
    data, tick = p_arr[:, 0], p_arr[:, 1]
    pol = (data >> 1) & 0b1
    y = (data >> 2) & 0b111111111111111  # 15 bits
    x = data >> 17
    # Same bottom-up y convention as the APS frames -- flip to match the
    # flipped frame rows (see _decode_frame).
    y = (FRAME_SHAPE[0] - 1) - y
    return tick.astype(np.int64), x.astype(np.int64), y.astype(np.int64), pol.astype(np.uint8)


def _assign_event_times(pol_packets):
    """Map each polarity packet's raw device ticks onto the host clock.
    `pol_packets` is a time-ordered list of (host_ts_us, tick, x, y, p) for
    one packet each. Every event in packet k is placed by linear
    interpolation between host_ts_us[k] and host_ts_us[k+1], at its
    fractional device-tick position within packet k -- so packet k's whole
    batch is spread across the interval up to the next packet's arrival,
    instead of collapsing every event in a packet onto one instant. The
    last packet has no "next" anchor, so its own device-tick span is used
    as the local duration estimate instead."""
    n = len(pol_packets)
    ev_t, ev_x, ev_y, ev_p = [], [], [], []
    for k in range(n):
        t0, tick, x, y, p = pol_packets[k]
        if k + 1 < n:
            t1 = pol_packets[k + 1][0]
        else:
            t1 = t0 + int(tick[-1] - tick[0])
        t1 = max(t1, t0)  # host arrivals can jitter slightly out of order

        span = tick[-1] - tick[0]
        if span > 0:
            frac = (tick - tick[0]).astype(np.float64) / span
        else:
            frac = np.zeros(tick.shape, dtype=np.float64)
        abs_t = (t0 + frac * (t1 - t0)).astype(np.int64)

        ev_t.append(abs_t); ev_x.append(x); ev_y.append(y); ev_p.append(p)

    events = {
        't': np.concatenate(ev_t) if ev_t else np.empty(0, dtype=np.int64),
        'x': np.concatenate(ev_x) if ev_x else np.empty(0, dtype=np.int64),
        'y': np.concatenate(ev_y) if ev_y else np.empty(0, dtype=np.int64),
        'p': np.concatenate(ev_p) if ev_p else np.empty(0, dtype=np.uint8),
    }
    # Packets are already host-time ordered and each packet's own abs_t is
    # internally non-decreasing (tick is monotonic within a packet), so this
    # is a cheap safety net rather than a full unordered sort.
    order = np.argsort(events['t'], kind='stable')
    return {k: v[order] for k, v in events.items()}


def load_recording(h5_path):
    """Decode a full recXXXXXXXXXX.hdf5 once into flat arrays, per the
    guide's recommendation not to keep re-parsing the packed/padded groups.
    Returns (frames, events): frames is a time-sorted list of
    (ts_us, img uint16 (260, 346)) on the host clock; events is a dict of
    int64/uint8 arrays, host-clock-anchored per _assign_event_times, sorted
    by 't'."""
    frames = []
    pol_packets = []
    with h5py.File(h5_path, 'r') as f:
        for etype, esize, ecapacity, ts_us, payload in _iter_dvs_rows(f):
            if etype == EVENT_TYPE_FRAME:
                img = _decode_frame(payload)
                frames.append((ts_us, img))
            elif etype == EVENT_TYPE_POLARITY:
                tick, x, y, p = _decode_polarity(payload, esize, ecapacity)
                if tick.size:
                    pol_packets.append((ts_us, tick, x, y, p))
            # special/imu6 events: skip, not needed for RGB+events extraction

    events = _assign_event_times(pol_packets)
    frames.sort(key=lambda fr: fr[0])
    return frames, events


def _passes_exposure(frame8, cfg):
    """Reject over/under-exposed frames: mean intensity outside
    [exposure_min_mean, exposure_max_mean], or too many clipped pixels.
    Dark (<=1) and bright (>=254) clipping have separate budgets: clipped
    sky on a bright day still leaves a teacher-visible road scene, while a
    mostly-black night frame gives the teacher nothing to encode (see
    ddd20.yaml for the tuning rationale)."""
    mean_val = float(frame8.mean())
    if mean_val < cfg.exposure_min_mean or mean_val > cfg.exposure_max_mean:
        return False
    if float((frame8 <= 1).mean()) > cfg.exposure_max_dark_frac:
        return False
    if float((frame8 >= 254).mean()) > cfg.exposure_max_bright_frac:
        return False
    return True


def select_frames(frames, cfg):
    """Time-based (not index-based) subsampling to ~cfg.target_hz, gated by
    the exposure filter. A frame that fails exposure is skipped and the next
    candidate is tried; the target-rate clock only advances on a kept frame,
    so a bad-exposure stretch doesn't shift the sampling grid."""
    dt_us = int(round(1e6 / cfg.target_hz))
    kept = []
    next_target_us = None
    for ts_us, img in frames:
        if next_target_us is not None and ts_us < next_target_us:
            continue
        frame8 = (img >> 8).astype(np.uint8)  # 16-bit -> 8-bit, same as ddd20-utils export.py's `d/256`
        if not _passes_exposure(frame8, cfg):
            continue
        kept.append((ts_us, frame8))
        next_target_us = ts_us + dt_us
    return kept


def get_event_window(events, t_end_us, window_us):
    """Events in (t_end_us - window_us, t_end_us], via searchsorted on the
    globally time-sorted event arrays."""
    t_start_us = t_end_us - window_us
    lo = np.searchsorted(events['t'], t_start_us, side='left')
    hi = np.searchsorted(events['t'], t_end_us, side='right')
    return {k: v[lo:hi] for k, v in events.items()}


def process_aps_frame(frame8, out_size):
    """Resize and replicate to 3 channels CHW uint8. DAVIS APS frames are
    single-channel grayscale (not RGB despite the loose naming) -- replicated
    to match the 3-channel RGB convention the student/teacher pipeline uses."""
    img_resized = cv2.resize(frame8, (out_size[0], out_size[1]), interpolation=cv2.INTER_AREA)
    img_chw = np.stack([img_resized, img_resized, img_resized], axis=0)
    return np.ascontiguousarray(img_chw)


def process_recording(seq, h5_path, pairs_dir, out_root, cfg):
    """Process a single recXXXXXXXXXX.hdf5 recording into out_root/<seq>/
    (DSEC layout: <seq>/rgb + <seq>/events/histogram, so dataset.py's
    root/<seq>/<rel_path> join resolves). Returns (n_pairs, n_skipped)."""
    out_size = tuple(cfg.model.img_hw)  # (W, H) for cv2.resize
    sensor_hw = tuple(cfg.datasets.SENSOR_HW)
    window_us = int(cfg.datasets.event_window_ms * 1000)
    seq_pairs_file = pairs_dir / f"{seq}_pairs.txt"

    seq_root = out_root / seq
    preprocessed_rgb_dir = seq_root / "rgb"
    preprocessed_events_hist_dir = seq_root / "events" / "histogram"
    for d in (preprocessed_rgb_dir, preprocessed_events_hist_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames, events = load_recording(h5_path)
    print(f"Recording {seq}: decoded {len(frames)} APS frames, {events['t'].size} events.")
    kept_frames = select_frames(frames, cfg.datasets)
    print(f"Recording {seq}: kept {len(kept_frames)} frames after "
          f"{cfg.datasets.target_hz} Hz + exposure filtering.")

    pairs_lines = []
    n_skipped = 0
    for i, (t_end_us, frame8) in enumerate(kept_frames):
        rgb_output_path = preprocessed_rgb_dir / f"{seq}_{i:06d}.npy"
        event_hist_output_path = preprocessed_events_hist_dir / f"{seq}_{i:06d}.npy"

        # 6-column DSEC line format; voxel/timesurface columns point at
        # paths that are never written, so a run that asks for them fails
        # fast at dataset.py's probe check instead of silently training on
        # the wrong representation -- we materialize histogram-only on DDD20.
        name = f"{seq}_{i:06d}.npy"
        pair_line = (f"{t_end_us},"
                     f"{rgb_output_path.relative_to(seq_root)},"
                     f"{event_hist_output_path.relative_to(seq_root)},"
                     f"events/voxel/{name},"
                     f"events/timesurface_net/{name},"
                     f"events/timesurface_zero/{name}")

        if rgb_output_path.exists() and event_hist_output_path.exists():
            pairs_lines.append(pair_line)
            continue

        window = get_event_window(events, t_end_us, window_us)
        if window['t'].size == 0:
            print(f"Warning: no events in [{t_end_us - window_us}, {t_end_us}] for {seq} frame {i}.")
            n_skipped += 1
            continue

        x_r, y_r, p = window['x'], window['y'], window['p']
        hot_mask = compute_hot_pixel_mask(x_r, y_r, sensor_hw=sensor_hw,
                                           threshold=cfg.datasets.event_hot_pixels_threshold)

        rgb_processed = process_aps_frame(frame8, out_size)
        event_histogram_processed = process_event_histogram(
            x_r, y_r, p, hot_mask, out_size=out_size, sensor_hw=sensor_hw)

        np.save(rgb_output_path, rgb_processed)
        np.save(event_hist_output_path, event_histogram_processed.astype(np.float16))

        pairs_lines.append(pair_line)

    with open(seq_pairs_file, 'w') as f:
        for line in pairs_lines:
            f.write(line + '\n')

    return (len(pairs_lines), n_skipped)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def build_pairs(cfg: DictConfig):
    dataset_path = Path(cfg.datasets.ddd20_path)
    out_root = Path(cfg.output_dir) / "preprocessed_ddd20"
    pairs_dir = out_root / "pairs_ddd20"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(dataset_path.glob("*.hdf5"))
    sequences = [p.stem for p in h5_paths]

    failed = []
    for seq, h5_path in zip(sequences, h5_paths):
        try:
            n_pairs, n_skipped = process_recording(
                seq, h5_path, pairs_dir, out_root, cfg)
            print(f"Recording {seq}: processed {n_pairs} pairs" +
                  (f", skipped {n_skipped} pairs" if n_skipped > 0 else ""))
        except Exception as e:
            print(f"Error processing recording {seq}: {e}")
            failed.append((seq, repr(e)))

    master_pairs_file = out_root / "pairs.txt"
    total = 0
    with open(master_pairs_file, 'w') as master_f:
        for seq in sequences:
            seq_pairs_file = pairs_dir / f"{seq}_pairs.txt"
            if seq_pairs_file.exists():
                with open(seq_pairs_file, 'r') as seq_f:
                    for line in seq_f:
                        if line.strip():
                            master_f.write(line if line.endswith('\n') else line + '\n')
                            total += 1

    print(f"Total pairs in master file: {total}")

    if failed:
        fail_log = out_root / "failed_recordings.txt"
        with open(fail_log, 'w') as f:
            for seq, err in failed:
                f.write(f"{seq}: {err}\n")
        print(f"Failed recordings logged in {fail_log}. "
              f"Re-run after fixing; existing outputs will be skipped (on resume)")


if __name__ == "__main__":
    build_pairs()
