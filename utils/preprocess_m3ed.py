import os
from pathlib import Path

import cv2
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig

from preprocess_dsec import (
    process_event_histogram,
    rectify_event_coords,
    compute_hot_pixel_mask,
)

EVENT_CAM = 'prophesee/left'
RGB_CAM = 'ovc/rgb'


# --- calibration / geometry ---------------------------------------------------

def load_calib(h5f, cam):
    """Return (K 3x3, D (4,), (W, H)) for a camera group. M3ED stores
    intrinsics as [fx, fy, cx, cy] and distortion as radtan [k1, k2, p1, p2]
    (verified: every camera in the file carries camera_model=b'pinhole',
    distortion_model=b'radtan')."""
    g = h5f[f'{cam}/calib']
    fx, fy, cx, cy = g['intrinsics'][:]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.asarray(g['distortion_coeffs'][:], dtype=np.float64).reshape(-1)
    w, h = (int(v) for v in g['resolution'][:])
    model = g['distortion_model'][()]
    assert model in (b'radtan', 'radtan'), f'{cam}: unexpected distortion model {model!r}'
    return K, D, (w, h)


def build_event_rectify_map(K, D, wh):
    """DSEC-style FORWARD map: rectify_map[y_raw, x_raw] = (x_undist, y_undist),
    float32 (H, W, 2). Built by undistorting the full pixel grid, so a raw
    event coordinate is a single array lookup. Kept fractional -- the
    downstream bilinear splat in preprocess_dsec._accumulate_events depends on
    that (rounding here reintroduces the unreachable-pixel lattice documented
    in rectify_event_coords)."""
    w, h = wh
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)[:, None, :]
    und = cv2.undistortPoints(pts, K, D, P=K)  # (N, 1, 2)
    return und.reshape(h, w, 2).astype(np.float32)


def build_rgb_to_event_maps(K_rgb, D_rgb, K_evt, R_evt_from_rgb, evt_wh):
    """Remap tables that resample the RAW RGB image into the UNDISTORTED
    event frame.

    cv2.initUndistortRectifyMap defines, for each destination pixel (u, v):
        X = R^-1 @ P^-1 @ [u, v, 1]        (X in the SOURCE camera's frame)
    then distorts X with D and projects it with K to get the source pixel.
    Destination is the event frame, so P = K_evt; X must land in the RGB
    camera's frame, so R^-1 = R_rgb<-evt, i.e. R = R_evt<-rgb -- exactly the
    rotation block of ovc/rgb/calib/T_to_prophesee_left, which is stored as
    the transform INTO the prophesee-left frame."""
    return cv2.initUndistortRectifyMap(
        K_rgb, D_rgb, R_evt_from_rgb, K_evt, evt_wh, cv2.CV_32FC1)


def common_valid_box(map_x, map_y, rgb_wh, margin=2):
    """Largest axis-aligned box in the event frame where the warped RGB is
    fully valid (the OVC's FOV is narrower than the event camera's, so the
    warp has invalid borders). Greedy shrink from the bounding box: the
    valid region is convex-ish here, so repeatedly trimming whichever side
    carries the most invalid pixels converges in a few dozen steps.
    Returns (x0, x1, y0, y1) as a half-open slice box."""
    w_rgb, h_rgb = rgb_wh
    valid = ((map_x >= margin) & (map_x < w_rgb - 1 - margin) &
             (map_y >= margin) & (map_y < h_rgb - 1 - margin))

    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        raise RuntimeError('RGB and event FOVs do not overlap -- check calibration.')
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    while True:
        sub = valid[y0:y1, x0:x1]
        if sub.all() or (x1 - x0) < 8 or (y1 - y0) < 8:
            break
        bad = ~sub
        counts = {'l': bad[:, 0].sum(), 'r': bad[:, -1].sum(),
                  't': bad[0, :].sum(), 'b': bad[-1, :].sum()}
        side = max(counts, key=counts.get)
        if counts[side] == 0:
            # No invalid pixel on any edge but some in the interior (should
            # not happen for a pinhole warp); trim all sides to be safe.
            x0 += 1; x1 -= 1; y0 += 1; y1 -= 1
            continue
        if side == 'l':
            x0 += 1
        elif side == 'r':
            x1 -= 1
        elif side == 't':
            y0 += 1
        else:
            y1 -= 1
    return x0, x1, y0, y1


def crop_box_to_aspect(box, target_aspect):
    """Shrink the common-FOV box about its centre until its width/height
    equals target_aspect. Returns the box unchanged when target_aspect is
    falsy.

    Why this exists
    ---------------
    Everything is resized to a SQUARE 384x384, so a source whose crop is not
    the same shape as the rest of the corpus lands in that square with a
    different anisotropic stretch. Verified with a synthetic circle pushed
    through cv2.resize: the M3ED box (1176x665) turns a circle into an
    ellipse of h/w = 1.769, while DSEC (640x480) gives 1.333 -- and the
    interpolation flag is irrelevant to this (INTER_AREA, INTER_LINEAR and
    INTER_NEAREST all give 1.769; INTER_AREA only controls aliasing, not the
    per-axis scale factors that cause the stretch).

    1.333 is not an arbitrary choice -- it is what every other source in the
    corpus already is, TRAINING and EVALUATION alike: DSEC 640x480, DDD20
    346x260, Brisbane 346x260 (DAVIS346), NSAVP 640x480 (DVXplorer). Left at
    1.769, M3ED would be the only source whose geometry disagrees with the
    Brisbane/NSAVP frames the student is actually scored on. Nothing about
    the evaluation sets changes to accommodate this -- they are untouched, so
    baseline comparability is unaffected; only this one new training source
    is brought into line with them.

    The cost is horizontal field of view (1176 -> 886 px, 25%), which is why
    it is a config knob: set datasets.match_aspect to null to keep the full
    FOV and ablate the choice.
    """
    x0, x1, y0, y1 = box
    if not target_aspect:
        return box
    w, h = x1 - x0, y1 - y0
    if w / h > target_aspect:          # too wide -> trim left/right
        new_w = int(round(h * target_aspect))
        off = (w - new_w) // 2
        x0, x1 = x0 + off, x0 + off + new_w
    else:                              # too tall -> trim top/bottom
        new_h = int(round(w / target_aspect))
        off = (h - new_h) // 2
        y0, y1 = y0 + off, y0 + off + new_h
    return x0, x1, y0, y1


# --- frame / event extraction -------------------------------------------------

def select_frame_indices(ts_us, target_hz):
    """Time-based (not index-based) subsampling of the OVC frames to
    ~target_hz. M3ED's OVC runs at a fixed 25 Hz, but selecting on
    timestamps keeps this correct for any sequence whose rate differs and
    matches preprocess_ddd20.select_frames' semantics."""
    if target_hz is None or target_hz <= 0:
        return list(range(len(ts_us)))
    dt_us = int(round(1e6 / float(target_hz)))
    keep, next_target = [], None
    for i, t in enumerate(ts_us):
        if next_target is not None and t < next_target:
            continue
        keep.append(i)
        next_target = int(t) + dt_us
    return keep


def _passes_exposure(gray, cfg):
    """Same over/under-exposure gate as DDD20 (mean-intensity band + separate
    dark/bright clipping budgets); see configs/datasets/ddd20.yaml for the
    tuning rationale. Applied to the grayscale of the OVC frame."""
    mean_val = float(gray.mean())
    if mean_val < cfg.exposure_min_mean or mean_val > cfg.exposure_max_mean:
        return False
    if float((gray <= 1).mean()) > cfg.exposure_max_dark_frac:
        return False
    if float((gray >= 254).mean()) > cfg.exposure_max_bright_frac:
        return False
    return True


def event_slice_bounds(ms_map_idx, n_events, t_start_us, t_end_us):
    """[lo, hi) event indices for the window [t_start_us, t_end_us) using
    ms_map_idx (first event index of each whole millisecond). Millisecond
    granularity is exact at the window edges we use (frame timestamps are
    whole milliseconds) and avoids a searchsorted over 4e8 on-disk elements."""
    n_ms = len(ms_map_idx)
    ms_lo = max(0, int(t_start_us) // 1000)
    ms_hi = int(np.ceil(int(t_end_us) / 1000.0))
    if ms_lo >= n_ms:
        return 0, 0
    lo = int(ms_map_idx[ms_lo])
    hi = int(ms_map_idx[ms_hi]) if ms_hi < n_ms else int(n_events)
    return lo, max(lo, hi)


def read_events(h5f, lo, hi):
    """Read one contiguous event slice as the dict layout
    rectify_event_coords expects."""
    g = h5f[EVENT_CAM]
    return {
        'x': g['x'][lo:hi].astype(np.int64),
        'y': g['y'][lo:hi].astype(np.int64),
        't': g['t'][lo:hi],
        'p': g['p'][lo:hi].astype(np.uint8),
    }


def process_rgb_frame(raw_bgr, map_x, map_y, box, out_size):
    """Raw OVC frame -> undistorted, rotation-aligned, FOV-cropped, resized
    CHW uint8 in RGB channel order. Stored raw uint8 and teacher-agnostic,
    exactly like DSEC: the dataloader does the /255 and any teacher
    normalisation.

    ovc/rgb/data is stored BGR, not RGB -- verified visually on
    car_urban_day_horse (a yellow road cone, red brick facades, red tail
    lights and blue jeans all only resolve correctly under the BGR reading;
    the RGB reading turns the whole scene teal). Everything downstream --
    preprocess_dsec.process_rgb_image, dataset.py, the DINOv3 teacher -- is
    RGB-ordered, so the swap happens here."""
    warped = cv2.remap(raw_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x0, x1, y0, y1 = box
    cropped = warped[y0:y1, x0:x1]
    resized = cv2.resize(cropped, (out_size[0], out_size[1]), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = np.stack([resized] * 3, axis=-1)
    else:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


def process_events(x_r, y_r, p, box, out_size, hot_threshold):
    """Undistorted event coordinates -> the same 3-channel histogram DSEC
    uses, but accumulated on the CROPPED sub-frame so it shares the RGB's
    field of view. Shifting the coordinates and shrinking the canvas (rather
    than histogramming full-frame then slicing) keeps the per-channel unit-max
    normalisation inside process_event_histogram computed over exactly the
    pixels that survive, so a hot spot outside the common FOV cannot suppress
    the visible scene."""
    x0, x1, y0, y1 = box
    keep = (x_r >= x0) & (x_r < x1) & (y_r >= y0) & (y_r < y1)
    xc = x_r[keep] - x0
    yc = y_r[keep] - y0
    pc = p[keep]
    crop_hw = (y1 - y0, x1 - x0)
    if xc.size == 0:
        return None
    hot_mask = compute_hot_pixel_mask(xc, yc, sensor_hw=crop_hw, threshold=hot_threshold)
    return process_event_histogram(xc, yc, pc, hot_mask, out_size=out_size, sensor_hw=crop_hw)


# --- debug overlays -----------------------------------------------------------

def _events_to_bgr(hist_chw):
    """Visualise a (3, H, W) histogram as BGR: green = positive, red = negative
    (matching the quick-look convention used to eyeball M3ED h5s)."""
    pos, neg = hist_chw[0], hist_chw[1]
    img = np.zeros(pos.shape + (3,), dtype=np.uint8)
    img[..., 1] = np.clip(pos * 255, 0, 255).astype(np.uint8)
    img[..., 2] = np.clip(neg * 255, 0, 255).astype(np.uint8)
    return img


def write_overlay(out_dir, seq, i, rgb_chw, hist_chw):
    """Side-by-side + blended overlay PNG for visual alignment checking."""
    rgb = np.ascontiguousarray(rgb_chw.transpose(1, 2, 0))       # RGB uint8
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ev = _events_to_bgr(hist_chw)
    blend = cv2.addWeighted(rgb_bgr, 0.6, ev, 1.0, 0)
    canvas = np.hstack([rgb_bgr, ev, blend])
    cv2.imwrite(str(out_dir / f'{seq}_{i:06d}.png'), canvas)


# --- driver -------------------------------------------------------------------

def process_sequence(seq, h5_path, pairs_dir, out_root, cfg, overlay_dir=None):
    """Process one M3ED <seq>_data.h5 into out_root/<seq>/ (DSEC layout:
    <seq>/rgb + <seq>/events/histogram, so dataset.py's root/<seq>/<rel_path>
    join resolves). Returns (n_pairs, n_skipped)."""
    dcfg = cfg.datasets
    out_size = tuple(cfg.model.img_hw)          # (W, H) for cv2.resize
    window_us = int(dcfg.event_window_ms * 1000)
    seq_pairs_file = pairs_dir / f'{seq}_pairs.txt'

    seq_root = out_root / seq
    rgb_dir = seq_root / 'rgb'
    hist_dir = seq_root / 'events' / 'histogram'
    if overlay_dir is None:
        for d in (rgb_dir, hist_dir):
            d.mkdir(parents=True, exist_ok=True)

    pairs_lines = []
    n_skipped = 0

    with h5py.File(h5_path, 'r') as f:
        K_evt, D_evt, evt_wh = load_calib(f, EVENT_CAM)
        K_rgb, D_rgb, rgb_wh = load_calib(f, RGB_CAM)
        assert tuple(dcfg.sensor_hw) == (evt_wh[1], evt_wh[0]), \
            f'{seq}: cfg sensor_hw {tuple(dcfg.sensor_hw)} != event resolution {(evt_wh[1], evt_wh[0])}'

        # T_to_prophesee_left maps points INTO the event frame, so its
        # rotation block is R_evt<-rgb -- what initUndistortRectifyMap wants.
        R_evt_from_rgb = np.asarray(f[f'{RGB_CAM}/calib/T_to_prophesee_left'][:])[:3, :3]

        rectify_map = build_event_rectify_map(K_evt, D_evt, evt_wh)
        map_x, map_y = build_rgb_to_event_maps(K_rgb, D_rgb, K_evt, R_evt_from_rgb, evt_wh)
        box = common_valid_box(map_x, map_y, rgb_wh)
        fov_w, fov_h = box[1] - box[0], box[3] - box[2]
        box = crop_box_to_aspect(box, dcfg.get('match_aspect', None))
        x0, x1, y0, y1 = box
        print(f'Sequence {seq}: common FOV {fov_w}x{fov_h} of {evt_wh[0]}x{evt_wh[1]}'
              f' -> box x[{x0}:{x1}] y[{y0}:{y1}] ({x1 - x0}x{y1 - y0}, '
              f'aspect {(x1 - x0) / (y1 - y0):.3f})')

        ts_us = f['ovc/ts'][:].astype(np.int64)
        rgb_data = f[f'{RGB_CAM}/data']
        ms_map_idx = f[f'{EVENT_CAM}/ms_map_idx'][:]
        n_events = f[f'{EVENT_CAM}/t'].shape[0]

        candidates = select_frame_indices(ts_us, dcfg.target_hz)
        max_frames = int(dcfg.get('debug_max_frames', 0) or 0)
        if overlay_dir is not None and max_frames:
            step = max(1, len(candidates) // max_frames)
            candidates = candidates[::step][:max_frames]
        print(f'Sequence {seq}: {len(ts_us)} OVC frames -> {len(candidates)} candidates '
              f'at {dcfg.target_hz} Hz.')

        kept = 0
        for fi in candidates:
            t_end_us = int(ts_us[fi])
            if t_end_us - window_us < 0:
                continue

            name = f'{seq}_{kept:06d}.npy'
            rgb_out = rgb_dir / name
            hist_out = hist_dir / name
            pair_line = (f'{t_end_us},'
                         f'rgb/{name},'
                         f'events/histogram/{name}')

            if overlay_dir is None and rgb_out.exists() and hist_out.exists():
                pairs_lines.append(pair_line)
                kept += 1
                continue

            raw = rgb_data[fi]
            if raw.ndim == 3 and raw.shape[-1] == 1:
                raw = raw[..., 0]
            gray = raw if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            if not _passes_exposure(gray, dcfg):
                continue

            lo, hi = event_slice_bounds(ms_map_idx, n_events, t_end_us - window_us, t_end_us)
            if hi <= lo:
                print(f'Warning: no events in [{t_end_us - window_us}, {t_end_us}] for {seq} frame {fi}.')
                n_skipped += 1
                continue
            events = read_events(f, lo, hi)
            # Trim the millisecond-granular slice to the exact window.
            m = (events['t'] >= t_end_us - window_us) & (events['t'] < t_end_us)
            events = {k: v[m] for k, v in events.items()}
            # Stationary-vehicle gate, expressed per sensor pixel so the
            # threshold transfers across sensor sizes instead of being tuned
            # to M3ED's HD array.
            #
            # M3ED's car sequences sit at traffic lights, and a stopped
            # window is not merely dimmer -- it is a different regime.
            # Measured on car_urban_day_horse at 50 ms: a driving window
            # holds 6.7e5-9.4e5 events and lands 22-27% of the output frame
            # above 0.05, while a stopped window holds 7.9e4 and lands 0.2%
            # -- a 60x collapse in occupied area, paired with a fully
            # detailed RGB frame. Distilling that pair teaches the student to
            # hallucinate scene structure from an empty input.
            #
            # Unit-max normalisation actively hides this: it rescales an
            # almost-empty window so its few surviving pixels look confident,
            # so nothing downstream can detect the frame is dead. That is why
            # it has to be filtered here, at the only point where the raw
            # event count is still visible.
            min_per_px = float(dcfg.get('min_events_per_px', 0.0) or 0.0)
            if events['t'].size < min_per_px * evt_wh[0] * evt_wh[1]:
                n_skipped += 1
                continue

            x_r, y_r, p, _ = rectify_event_coords(events, rectify_map,
                                                  sensor_hw=(evt_wh[1], evt_wh[0]))
            hist = process_events(x_r, y_r, p, box, out_size,
                                  dcfg.hot_pixel_threshold)
            if hist is None:
                print(f'Warning: all events outside the common FOV for {seq} frame {fi}.')
                n_skipped += 1
                continue

            rgb_processed = process_rgb_frame(raw, map_x, map_y, box, out_size)

            if overlay_dir is not None:
                write_overlay(overlay_dir, seq, kept, rgb_processed, hist)
            else:
                np.save(rgb_out, rgb_processed)
                np.save(hist_out, hist.astype(np.float16))

            pairs_lines.append(pair_line)
            kept += 1

    if overlay_dir is None:
        with open(seq_pairs_file, 'w') as fh:
            for line in pairs_lines:
                fh.write(line + '\n')

    return (len(pairs_lines), n_skipped)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def build_pairs(cfg: DictConfig):
    dataset_path = Path(cfg.datasets.m3ed_path)
    out_root = Path(cfg.datasets.root_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_root / 'pairs_m3ed'

    overlay_dir = cfg.datasets.get('debug_overlay_dir', None)
    if overlay_dir:
        overlay_dir = Path(os.path.expanduser(str(overlay_dir)))
        overlay_dir.mkdir(parents=True, exist_ok=True)
        print(f'DEBUG overlay mode: writing PNGs to {overlay_dir}, no .npy/pairs output.')
    else:
        overlay_dir = None
        pairs_dir.mkdir(parents=True, exist_ok=True)

    if dataset_path.is_file():
        h5_paths = [dataset_path]
    else:
        h5_paths = sorted(dataset_path.glob('*_data.h5')) or sorted(dataset_path.glob('*.h5'))
    # M3ED ships <sequence>_data.h5 alongside <sequence>_pose_gt.h5 / _depth_gt.h5;
    # the sequence name is the stem without the _data suffix.
    sequences = [p.stem[:-5] if p.stem.endswith('_data') else p.stem for p in h5_paths]

    failed = []
    for seq, h5_path in zip(sequences, h5_paths):
        try:
            n_pairs, n_skipped = process_sequence(seq, h5_path, pairs_dir, out_root, cfg,
                                                  overlay_dir=overlay_dir)
            print(f'Sequence {seq}: processed {n_pairs} pairs' +
                  (f', skipped {n_skipped} pairs' if n_skipped > 0 else ''))
        except Exception as e:
            print(f'Error processing sequence {seq}: {e}')
            failed.append((seq, repr(e)))

    if overlay_dir is not None:
        return

    master_pairs_file = out_root / cfg.datasets.pairs_name
    total = 0
    with open(master_pairs_file, 'w') as master_f:
        for seq in sequences:
            seq_pairs_file = pairs_dir / f'{seq}_pairs.txt'
            if seq_pairs_file.exists():
                with open(seq_pairs_file, 'r') as seq_f:
                    for line in seq_f:
                        if line.strip():
                            master_f.write(line if line.endswith('\n') else line + '\n')
                            total += 1

    print(f'Total pairs in master file: {total}')

    if failed:
        fail_log = out_root / 'failed_sequences.txt'
        with open(fail_log, 'w') as f:
            for seq, err in failed:
                f.write(f'{seq}: {err}\n')
        print(f'Failed sequences logged in {fail_log}. '
              f'Re-run after fixing; existing outputs will be skipped (on resume)')


if __name__ == '__main__':
    build_pairs()
