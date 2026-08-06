"""
ViViD++ preprocessor for E-LiteVPR.

Reads ViViD++ ROS1 bags (one bag = one sequence, e.g. campus_night.bag) and
writes paired RGB/event representations in the SAME on-disk contract as
preprocess_m3ed.py:

    <out_root>/<seq>/rgb/<seq>_<i>.npy                  uint8 (3, H, W)
    <out_root>/<seq>/events/histogram/<seq>_<i>.npy     float16 (3, H, W)
    <out_root>/pairs_vivid/<seq>_pairs.txt  ->  <out_root>/pairs.txt

pairs.txt lines are the 3-column DSEC form
    <t_us>,rgb/<name>.npy,events/histogram/<name>.npy
because that is what the live loader accepts: scripts/dataset.py:70 asserts
exactly 3 comma-separated entries and joins root/<seq>/<rel_path>, and
scripts/feature_extractor.py:34-38 takes the sequence name from the SECOND
path component. (Note: preprocess_m3ed.py writes 6 columns with placeholder
voxel/timesurface paths, which that assert rejects. Not touched here.)

Sensor layout -- verified by reading the two bags, not assumed
--------------------------------------------------------------
  /camera/image_color   sensor_msgs/Image, bgr8, 1280x1024 (Flea3)
  /dvs/events           dvs_msgs/EventArray, 640x480 (DVXplorer), ~33 Hz packets
  /camera/camera_info   PRESENT BUT EMPTY (width=height=0, K all zero) --
                        intrinsics must come from calibration_results/, not
                        from the bag.
  RGB rate is NOT constant across the corpus:
        campus_morning_manual_small  19.995 Hz (50.01 ms)
        campus_night                 29.970 Hz (33.37 ms)
  so frames are selected by TIMESTAMP against `target_hz` and the event
  window is a FIXED `event_window_ms` ending at the frame stamp, exactly like
  preprocess_m3ed. A DSEC-style "stride N, window = [t[i-1], t[i]]" would
  silently give a 50 ms window in one bag and 33 ms in the next.

  Events and RGB share one clock (both are Unix epoch header stamps, and the
  event stream leads the first image by ~32 ms), so no Brisbane-style offset
  table is needed. pairs.txt timestamps are epoch microseconds.

Geometry -- why this is DSEC's contract and not M3ED's
------------------------------------------------------
Like M3ED, ViViD ships both streams RAW from two physically separate cameras,
so the rectification happens here. Unlike M3ED, the numbers work out so that
NOTHING has to be cropped:

  events : radtan (cam1), undistorted with a DSEC-style FORWARD map (raw pixel
           -> fractional undistorted pixel) built once with cv2.undistortPoints
           over the full grid, then consumed by preprocess_dsec's
           rectify_event_coords + bilinear splat. Fractional on purpose -- see
           that function for why rounding bakes a lattice into the frames.
           Measured here: with the 2x2 bilinear footprint 100.00% of the
           undistorted 640x480 frame is reachable (nearest-neighbour rounding
           would leave 9.68% structurally unreachable -- the same failure mode
           documented for DSEC).
  RGB    : cam0 is EQUIDISTANT (fisheye), so it must be warped with
           cv2.fisheye.initUndistortRectifyMap, NOT the radtan
           cv2.initUndistortRectifyMap. R = the rotation block of cam1's
           T_cn_cnm1 (the cam0->cam1 transform, i.e. R_evt<-rgb) and P = K_evt,
           which removes fisheye distortion and the inter-camera rotation in
           one remap.

  The RGB is much wider than the event camera (fisheye half-angles ~42.5 deg
  horizontal / ~36.5 deg vertical vs the event camera's undistorted 44.4 deg
  HFOV / 34.2 deg VFOV): measured, the warp is valid over 100.00% of the event
  frame. So there are no invalid borders, no common-FOV box, and no
  match_aspect knob -- the output is the full undistorted 640x480 event frame,
  which is DSEC's 640x480 exactly, stretching into the square 384x384 by the
  same 1.333 as DSEC, DDD20, Brisbane and NSAVP.

The homography ignores the 12.95 cm baseline, i.e. it is exact only at
infinite depth. Residual parallax is fx*B/Z = 783.3*0.1295/Z ~= 101/Z px, so
~10 px at 10 m and ~2 px at 50 m on the 640-wide frame -- about 6 px and 1 px
after the resize to 384. That is ~3x tighter than DSEC, whose 60 cm baseline
is left entirely uncorrected, so this introduces no misalignment mode the
corpus does not already contain.

Measured, not assumed (correlation between the event magnitude image and the
RGB gradient magnitude, scanned over +-20 px on the 640x480 grid; 10 frames
spread across campus_morning_manual_small, 11 across campus_night):

    rotation     med dx   per-frame dx range   med dy   corr@(0,0)
    morning
      R (calib)   -9.0        -20 .. 0          +6.5      0.2682
      identity    +5.0        -20 .. +12        +7.0      0.2491
      R.T        +18.0        -12 .. +20        +7.0      0.2168
    night
      R (calib)  -13.0        -20 .. -6         +7.0      0.2104
      identity    +0.0         -7 .. +13        +7.0      0.2309
      R.T        +13.0         +7 .. +20        +7.0      0.1835

  R is right and identity is not, but NOT because of the correlation column:
  that column is not decisive, since it favours R in the morning bag and
  identity at night. It conflates rotation error with parallax, and it
  therefore rewards whichever rotation happens to cancel the average parallax
  of the bag.

  What is decisive is the SIGN of the per-frame horizontal residual. The
  infinite-depth homography can only displace nearer content ONE way, by
  -fx*B/Z = -101/Z px, so under the correct rotation dx must be <= 0 in every
  frame and must approach 0 for the most distant content. That is exactly what
  R gives in both bags (-20..0 morning, -20..-6 night), and the implied depths
  are ordinary: Z = 5 m..infinity by day, Z = 5..17 m at night, where only
  nearby illuminated structure fires events at all. Under identity dx reaches
  +12 and +13, which would require NEGATIVE depth. Identity is a rotation
  error that partly cancels parallax on average, nothing more.

  Note this reverses an earlier in-repo conclusion that favoured identity:
  that measurement warped the fisheye RGB with the radtan routine.

  The VERTICAL residual is the one thing every variant agrees on: +6.5 px
  (morning) and +7.0 px (night, per-frame +4..+9), independent of rotation, so
  it is not rotation; ty = -4.7 mm makes it 0.3 px of parallax at most, so it
  is not parallax either. It is a genuine offset between cam1's intrinsics
  (calibrated on /dvs/image_recon, i.e. E2VID reconstructions) and the raw
  event pixel grid. It is ~1.5% of frame height and is left UNCORRECTED by
  default -- set datasets.rgb_principal_shift: [0.0, -7.0] to fold it into the
  RGB destination K and ablate it. (Sign check, measured: a +6.5 shift moves
  the RGB DOWN and takes dy to +13.0, so the correction is negative.)

Usage
-----
    uv run --with rosbags python utils/preprocess_vivid.py datasets=vivid \\
        datasets.vivid_path=/path/to/bags output_dir=/path/to/out

    # visual check: [RGB | events | blend] PNGs instead of .npy/pairs output
    uv run --with rosbags python utils/preprocess_vivid.py datasets=vivid \\
        datasets.vivid_path=... datasets.debug_overlay_dir=~/Desktop/vivid_check

`rosbags` is NOT in the project venv, hence `--with`; the project venv is left
untouched. Bags are bz2-chunked, which caps a single reader at ~11 MB/s
(measured: 6.82 GiB in 10.2 min), so on the remote box run ONE PROCESS PER BAG
in parallel rather than trying to speed up a single pass.
"""

import os
from pathlib import Path

import cv2
import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from preprocess_dsec import (
    process_event_histogram,
    rectify_event_coords,
    compute_hot_pixel_mask,
)
from preprocess_m3ed import write_overlay

EVENT_TOPIC = '/dvs/events'
RGB_TOPIC = '/camera/image_color'

# ROS1 dvs_msgs/Event is UNALIGNED: uint16 x, uint16 y, time ts, bool polarity.
# Verified field-exact against the rosbags deserializer on 25 messages of each
# bag (x, y, sec, nsec, polarity all bit-identical, no trailing bytes).
# The deserializer builds one Python object per event, which is unusable at
# ~0.5 Mev/s over a 7-minute bag; np.frombuffer is what makes this tractable.
_EVENT_DT = np.dtype([('x', '<u2'), ('y', '<u2'),
                      ('sec', '<u4'), ('nsec', '<u4'), ('p', 'u1')])
assert _EVENT_DT.itemsize == 13
# What the rest of this module passes around: microsecond int64 time, matching
# preprocess_dsec's EventSlicer output.
_EV_DT = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])


# --- ROS1 message parsing (no deserializer) -----------------------------------

def _skip_header(buf):
    """std_msgs/Header = uint32 seq, time(uint32 sec, uint32 nsec), string
    frame_id. Returns (offset_after_header, stamp_us)."""
    sec, nsec = np.frombuffer(buf, '<u4', 2, 4)
    fid_len = int(np.frombuffer(buf, '<u4', 1, 12)[0])
    return 16 + fid_len, int(sec) * 1_000_000 + int(nsec) // 1000


def image_stamp_us(buf):
    """Header stamp of a sensor_msgs/Image without touching the payload."""
    return _skip_header(buf)[1]


def parse_image(buf):
    """sensor_msgs/Image (bgr8) -> (H, W, 3) uint8 view. ViViD's RGB stream is
    bgr8 1280x1024 in both bags; anything else is a corpus change we want to
    hear about rather than silently reinterpret."""
    off, _ = _skip_header(buf)
    h, w = (int(v) for v in np.frombuffer(buf, '<u4', 2, off)); off += 8
    enc_len = int(np.frombuffer(buf, '<u4', 1, off)[0]); off += 4
    enc = buf[off:off + enc_len].decode(); off += enc_len
    off += 1                                             # uint8 is_bigendian
    step = int(np.frombuffer(buf, '<u4', 1, off)[0]); off += 4
    dlen = int(np.frombuffer(buf, '<u4', 1, off)[0]); off += 4
    if enc != 'bgr8' or step != 3 * w or dlen != step * h:
        raise ValueError(f'unexpected Image layout: enc={enc} {w}x{h} step={step} len={dlen}')
    return np.frombuffer(buf, np.uint8, dlen, off).reshape(h, w, 3)


def parse_events(buf):
    """dvs_msgs/EventArray -> (h, w, structured (x, y, t_us, p) array sorted
    by time)."""
    off, _ = _skip_header(buf)
    h, w, n = (int(v) for v in np.frombuffer(buf, '<u4', 3, off)); off += 12
    raw = np.frombuffer(buf, _EVENT_DT, n, off)
    out = np.empty(n, _EV_DT)
    out['x'] = raw['x']
    out['y'] = raw['y']
    out['p'] = raw['p']
    out['t'] = (raw['sec'].astype(np.int64) * 1_000_000
                + raw['nsec'].astype(np.int64) // 1000)
    # Packets are internally monotonic in both bags, but sorting is cheap
    # insurance against a sequence where they are not.
    if n > 1 and not np.all(np.diff(out['t']) >= 0):
        out.sort(order='t')
    return h, w, out


# --- calibration / geometry ---------------------------------------------------

def load_camchain(path):
    """ViViD++ ships Kalibr camchains under calibration_results/. Returns
    (K_rgb, D_rgb, (W, H)_rgb, K_evt, D_evt, (W, H)_evt, R_evt_from_rgb, t).

    cam0 is the RGB (equidistant), cam1 the DVS (radtan), and cam1's
    T_cn_cnm1 is the cam0 -> cam1 transform, so its rotation block is
    R_evt<-rgb -- exactly what initUndistortRectifyMap wants as R."""
    with open(path, 'r') as f:
        chain = yaml.safe_load(f)
    rgb, evt = chain['cam0'], chain['cam1']
    if rgb['distortion_model'] != 'equidistant':
        raise ValueError(f'{path}: cam0 is {rgb["distortion_model"]}, expected equidistant '
                         f'(the RGB warp below uses cv2.fisheye)')
    if evt['distortion_model'] != 'radtan':
        raise ValueError(f'{path}: cam1 is {evt["distortion_model"]}, expected radtan')

    def _K(cam):
        fx, fy, cx, cy = cam['intrinsics']
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

    T = np.asarray(evt['T_cn_cnm1'], np.float64)
    return (_K(rgb), np.asarray(rgb['distortion_coeffs'], np.float64).reshape(4, 1),
            tuple(int(v) for v in rgb['resolution']),
            _K(evt), np.asarray(evt['distortion_coeffs'], np.float64).reshape(-1),
            tuple(int(v) for v in evt['resolution']),
            T[:3, :3], T[:3, 3])


def build_event_rectify_map(K, D, wh):
    """DSEC-style FORWARD map: rectify_map[y_raw, x_raw] = (x_undist, y_undist),
    float32 (H, W, 2), so a raw event coordinate is one array lookup. Kept
    fractional -- the bilinear splat in preprocess_dsec._accumulate_events
    depends on that."""
    w, h = wh
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)[:, None, :]
    und = cv2.undistortPoints(pts, K, D, P=K)
    return und.reshape(h, w, 2).astype(np.float32)


def build_rgb_to_event_maps(K_rgb, D_rgb, K_evt, R_evt_from_rgb, evt_wh, shift=None):
    """Remap tables resampling the RAW fisheye RGB into the UNDISTORTED event
    frame. `shift` is an optional (dx, dy) added to the destination principal
    point -- the ablation knob for the measured vertical residual (see the
    module docstring); a positive dy moves the RGB content DOWN."""
    K_dst = K_evt.copy()
    if shift is not None:
        K_dst[0, 2] += float(shift[0])
        K_dst[1, 2] += float(shift[1])
    return cv2.fisheye.initUndistortRectifyMap(
        K_rgb, D_rgb, R_evt_from_rgb, K_dst, evt_wh, cv2.CV_32FC1)


# --- frame gates --------------------------------------------------------------

def _passes_exposure(gray, cfg):
    """Same mean-intensity band + dark/bright clipping budgets as DDD20/M3ED.
    ViViD's night frames are legitimately dark, so the defaults in
    configs/datasets/vivid.yaml are set from the measured distribution rather
    than copied from M3ED -- see that file."""
    if float(gray.mean()) < cfg.exposure_min_mean or float(gray.mean()) > cfg.exposure_max_mean:
        return False
    if float((gray <= 1).mean()) > cfg.exposure_max_dark_frac:
        return False
    if float((gray >= 254).mean()) > cfg.exposure_max_bright_frac:
        return False
    return True


def process_rgb_frame(raw_bgr, map_x, map_y, out_size):
    """Raw fisheye BGR frame -> undistorted, rotation-aligned, resized CHW
    uint8 in RGB channel order. Stored raw uint8 and teacher-agnostic like
    every other source: the dataloader does the /255 and any teacher
    normalisation. No crop -- the warp is valid over the whole event frame."""
    warped = cv2.remap(raw_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    resized = cv2.resize(warped, (out_size[0], out_size[1]), interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


# --- one bag ------------------------------------------------------------------

def process_sequence(seq, bag_path, pairs_dir, out_root, cfg, overlay_dir=None):
    """Process one ViViD bag into out_root/<seq>/. Returns (n_pairs, n_skipped).

    Single pass: the bag interleaves both topics inside bz2 chunks, so reading
    them separately would decompress the whole file twice. Events are buffered
    and a selected frame is only emitted once the event stream has run
    `flush_guard_ms` past it, which makes the result independent of how far
    message write order lags header order.
    """
    from rosbags.highlevel import AnyReader          # optional dep, see docstring

    dcfg = cfg.datasets
    out_size = tuple(cfg.model.img_hw)
    window_us = int(dcfg.event_window_ms * 1000)
    guard_us = int(dcfg.get('flush_guard_ms', 200) * 1000)
    dt_us = int(round(1e6 / float(dcfg.target_hz))) if dcfg.target_hz else 0
    min_per_px = float(dcfg.get('min_events_per_px', 0.0) or 0.0)
    max_frames = int(dcfg.get('debug_max_frames', 0) or 0)

    seq_root = out_root / seq
    rgb_dir, hist_dir = seq_root / 'rgb', seq_root / 'events' / 'histogram'
    if overlay_dir is None:
        for d in (rgb_dir, hist_dir):
            d.mkdir(parents=True, exist_ok=True)

    calib = Path(os.path.expanduser(str(dcfg.camchain_path)))
    K_rgb, D_rgb, rgb_wh, K_evt, D_evt, evt_wh, R, t_vec = load_camchain(calib)
    shift = dcfg.get('rgb_principal_shift', None)
    rectify_map = build_event_rectify_map(K_evt, D_evt, evt_wh)
    map_x, map_y = build_rgb_to_event_maps(K_rgb, D_rgb, K_evt, R, evt_wh, shift)
    sensor_hw = (evt_wh[1], evt_wh[0])
    if tuple(dcfg.SENSOR_HW) != sensor_hw:
        raise ValueError(f'{seq}: cfg SENSOR_HW {tuple(dcfg.SENSOR_HW)} != '
                         f'camchain event resolution {sensor_hw}')
    print(f'Sequence {seq}: events {evt_wh[0]}x{evt_wh[1]} (radtan), rgb '
          f'{rgb_wh[0]}x{rgb_wh[1]} (equidistant), baseline '
          f'{np.linalg.norm(t_vec)*100:.2f} cm, principal shift {list(shift) if shift else None}')

    pairs_lines, n_skipped = [], 0
    state = {'kept': 0, 'ev': [], 'ev_t_max': -1, 'pending': []}

    def emit(t_us, img_buf):
        """Turn one buffered frame into an output pair (or drop it)."""
        nonlocal n_skipped
        name = f'{seq}_{state["kept"]:06d}.npy'
        rgb_out, hist_out = rgb_dir / name, hist_dir / name
        pair_line = f'{t_us},rgb/{name},events/histogram/{name}'

        if overlay_dir is None and rgb_out.exists() and hist_out.exists():
            pairs_lines.append(pair_line)
            state['kept'] += 1
            return

        lo = t_us - window_us
        sel = [c for c in state['ev'] if len(c) and c['t'][-1] >= lo and c['t'][0] < t_us]
        ev = np.concatenate(sel) if sel else np.empty(0, _EV_DT)
        ev = ev[(ev['t'] >= lo) & (ev['t'] < t_us)]

        # "Remove frames which have no events", as M3ED does, expressed per
        # sensor pixel so the threshold transfers across sensor sizes. ViViD's
        # driving bags stop at lights and junctions exactly like M3ED's; a
        # stopped window pairs a fully detailed RGB frame with a near-empty
        # event frame, and unit-max normalisation then rescales the few
        # survivors so nothing downstream can tell the input was dead. This is
        # the only point where the raw event count is still visible.
        # M3ED's own 0.25 ev/px does NOT transfer -- measured on
        # campus_morning_manual_small at 50 ms, the MEDIAN window is 0.136
        # ev/px, so 0.25 would delete most of the corpus. See vivid.yaml.
        if ev.size < min_per_px * evt_wh[0] * evt_wh[1]:
            n_skipped += 1
            return

        raw = parse_image(img_buf)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if not _passes_exposure(gray, dcfg):
            n_skipped += 1
            return

        x_r, y_r, p, _ = rectify_event_coords(ev, rectify_map, sensor_hw=sensor_hw)
        if p.size == 0:
            n_skipped += 1
            return
        hot = compute_hot_pixel_mask(x_r, y_r, sensor_hw=sensor_hw,
                                     threshold=dcfg.event_hot_pixels_threshold)
        hist = process_event_histogram(x_r, y_r, p, hot, out_size=out_size,
                                       sensor_hw=sensor_hw)
        rgb_processed = process_rgb_frame(raw, map_x, map_y, out_size)

        if overlay_dir is not None:
            write_overlay(overlay_dir, seq, state['kept'], rgb_processed, hist)
        else:
            np.save(rgb_out, rgb_processed)
            np.save(hist_out, hist.astype(np.float16))
        pairs_lines.append(pair_line)
        state['kept'] += 1

    def flush(force=False):
        keep = []
        for t_us, buf in state['pending']:
            if not force and state['ev_t_max'] < t_us + guard_us:
                keep.append((t_us, buf))
                continue
            emit(t_us, buf)
        state['pending'] = keep
        # Drop event chunks no future frame can need. The extra guard_us of
        # slack matters: with no frame pending, the only bound available is
        # ev_t_max, and the NEXT selected frame's window starts BEFORE that by
        # however far the event stream leads the image stream in write order.
        # Trimming to exactly one window would make correctness depend on that
        # lead staying under one event packet (~30 ms). guard_us of history is
        # ~6 MB at ViViD's peak event rate.
        cut = (min((p[0] for p in state['pending']), default=state['ev_t_max'])
               - window_us - guard_us)
        state['ev'] = [c for c in state['ev'] if len(c) == 0 or c['t'][-1] >= cut]

    with AnyReader([bag_path]) as reader:
        topics = {c.topic for c in reader.connections}
        missing = {EVENT_TOPIC, RGB_TOPIC} - topics
        if missing:
            raise ValueError(f'{seq}: bag is missing {sorted(missing)}; has {sorted(topics)}')
        conns = [c for c in reader.connections if c.topic in (EVENT_TOPIC, RGB_TOPIC)]
        n_rgb = sum(c.msgcount for c in conns if c.topic == RGB_TOPIC)
        dur_s = (reader.end_time - reader.start_time) / 1e9
        print(f'Sequence {seq}: {dur_s:.1f} s, {n_rgb} RGB msgs '
              f'({n_rgb/max(dur_s,1e-9):.2f} Hz) -> ~{int(dur_s * float(dcfg.target_hz))} '
              f'candidates at {dcfg.target_hz} Hz, {dcfg.event_window_ms} ms windows.')
        if overlay_dir is not None and max_frames:
            # Spread the overlay sample across the WHOLE bag by widening the
            # selection interval, not by stopping early: a bz2 bag has to be
            # read start-to-finish anyway, and frames from only the first 20 s
            # would not show whether alignment holds as the vehicle turns.
            dt_us = max(dt_us, int(dur_s * 1e6) // max_frames)
            print(f'Sequence {seq}: overlay mode, sampling every {dt_us/1e6:.2f} s '
                  f'for ~{max_frames} frames.')

        next_target = None
        for conn, _, raw in reader.messages(connections=conns):
            if conn.topic == EVENT_TOPIC:
                h, w, ev = parse_events(raw)
                if (w, h) != evt_wh:
                    raise ValueError(f'{seq}: EventArray is {w}x{h}, camchain says '
                                     f'{evt_wh[0]}x{evt_wh[1]}; wrong calibration file?')
                state['ev'].append(ev)
                if len(ev):
                    state['ev_t_max'] = max(state['ev_t_max'], int(ev['t'][-1]))
                flush()
            else:
                t_us = image_stamp_us(raw)
                if next_target is not None and t_us < next_target:
                    continue
                next_target = t_us + dt_us
                state['pending'].append((t_us, bytes(raw)))
        flush(force=True)

    if overlay_dir is None:
        with open(pairs_dir / f'{seq}_pairs.txt', 'w') as fh:
            for line in pairs_lines:
                fh.write(line + '\n')
    return len(pairs_lines), n_skipped


# --- driver -------------------------------------------------------------------

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def build_pairs(cfg: DictConfig):
    dataset_path = Path(os.path.expanduser(str(cfg.datasets.vivid_path)))
    out_root = Path(cfg.output_dir) / 'preprocessed_vivid'
    pairs_dir = out_root / 'pairs_vivid'

    overlay_dir = cfg.datasets.get('debug_overlay_dir', None)
    if overlay_dir:
        overlay_dir = Path(os.path.expanduser(str(overlay_dir)))
        overlay_dir.mkdir(parents=True, exist_ok=True)
        print(f'DEBUG overlay mode: writing PNGs to {overlay_dir}, no .npy/pairs output.')
    else:
        overlay_dir = None
        pairs_dir.mkdir(parents=True, exist_ok=True)

    bags = [dataset_path] if dataset_path.is_file() else sorted(dataset_path.glob('*.bag'))
    exclude = set(cfg.datasets.get('to_exclude', None) or [])
    bags = [b for b in bags if b.stem not in exclude]
    if not bags:
        raise FileNotFoundError(f'No .bag files found under {dataset_path}')
    print(f'{len(bags)} bag(s): {[b.stem for b in bags]}')

    failed = []
    for bag in bags:
        seq = bag.stem
        try:
            n_pairs, n_skipped = process_sequence(seq, bag, pairs_dir, out_root, cfg,
                                                  overlay_dir=overlay_dir)
            print(f'Sequence {seq}: processed {n_pairs} pairs'
                  + (f', skipped {n_skipped} pairs' if n_skipped else ''))
        except Exception as e:
            print(f'Error processing sequence {seq}: {e}')
            failed.append((seq, repr(e)))

    if overlay_dir is not None:
        return

    total = 0
    with open(out_root / 'pairs.txt', 'w') as master_f:
        for bag in bags:
            seq_pairs_file = pairs_dir / f'{bag.stem}_pairs.txt'
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
