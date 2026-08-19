import os
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import hydra
import numpy as np
from omegaconf import DictConfig
import yaml

from dsec_eventslicer import EventSlicer

def load_calibration_params(calib_file):
    # Load calibration parameters from the YAML file
    with open(calib_file, 'r') as f:
        calib_params = yaml.safe_load(f)
    
    fx0, fy0, cx0, cy0 = calib_params['intrinsics']['camRect0']['camera_matrix']
    K_rect_event = np.array([[fx0, 0, cx0],
                             [0, fy0, cy0],
                             [0, 0, 1]])
    
    fx1, fy1, cx1, cy1 = calib_params['intrinsics']['camRect1']['camera_matrix']
    K_rect_rgb = np.array([[fx1, 0, cx1],
                           [0, fy1, cy1],
                           [0, 0, 1]])
    
    return K_rect_event, K_rect_rgb

def load_rectify_map(rectify_map_path):
    # Load rectification map for the left event camera (file format: h5)
    # rectify_map[y_raw, x_raw] = (x_rect, y_rect): a FORWARD map for
    # per-event coordinate lookup (official DSEC usage).
    with h5py.File(rectify_map_path, 'r') as f:
        rectify_map = f['rectify_map'][:] # (H, W, 2)

    return rectify_map

def rectify_event_coords(events, rectify_map, sensor_hw=(480, 640)):
    """
    Look up rectified coordinates for each event and drop events that land
    outside the sensor frame.

    Returns (x_r, y_r, p, t) as aligned arrays with out-of-frame events removed.
    x_r/y_r are FRACTIONAL (float64) rectified coordinates 
    """
    h, w = sensor_hw
    xy_rect = rectify_map[events['y'], events['x']]  # (N, 2) float
    x_r = xy_rect[:, 0].astype(np.float64)
    y_r = xy_rect[:, 1].astype(np.float64)

    # Drop events that land outside the sensor frame
    valid_mask = (x_r >= 0) & (x_r < w) & (y_r >= 0) & (y_r < h)
    x_r = x_r[valid_mask]
    y_r = y_r[valid_mask]
    p = events['p'][valid_mask]
    t = events['t'][valid_mask]

    return x_r, y_r, p, t

def _accumulate_events(x, y, sensor_hw=(480, 640)):
    """
    Accumulate events into a histogram, splitting each event across its four
    neighbouring pixels by bilinear weight.
    """
    h, w = sensor_hw
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0

    # Four corners gathered into one bincount rather than four np.add.at
    # calls: np.add.at is an unbuffered ufunc and is ~10x slower, which is
    # material over a whole-corpus preprocessing pass.
    xs = np.concatenate([x0, x0 + 1, x0, x0 + 1])
    ys = np.concatenate([y0, y0, y0 + 1, y0 + 1])
    wt = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy),
                         (1 - fx) * fy, fx * fy])

    m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (wt > 0)
    hist = np.bincount(ys[m] * w + xs[m], weights=wt[m], minlength=h * w)
    return hist.reshape(h, w).astype(np.float32)

def compute_hot_pixel_mask(x_r, y_r, sensor_hw=(480, 640), threshold=99.5):
    """
    Identify hot pixels from the FULL-WINDOW combined count image.
    Returns a boolean (H, W) mask that is True at hot pixels.
    The same mask is applied to both representations and to every voxel bin,
    so denoising is identical across the histogram-vs-voxel comparison.
    """
    h, w = sensor_hw
    hist = _accumulate_events(x_r, y_r, sensor_hw)
    nonzero = hist[hist > 0]
    if len(nonzero) == 0:
        return np.zeros((h, w), dtype=bool)  # No events to process
    threshold_value = np.percentile(nonzero, threshold)
    hot_pixel_mask = hist > threshold_value
    return hot_pixel_mask

def _norm_unit_max(arr):
    m = arr.max()
    return arr / m if m > 0 else arr

def _resize_to(arr, out_size):
    # out_size is (W, H) order for cv2.resize; we pass square cfg.model.img_hw
    # so order is moot, but keep the convention explicit.
    return cv2.resize(arr, (out_size[0], out_size[1]), interpolation=cv2.INTER_AREA)

def _event_fov_crop_box(K_event, K_rgb, sensor_hw=(480, 640)):
    # Compute the cropping box for the RGB image based on the field of view of the event camera.
    # This function calculates the intersection of the two camera fields of view and returns the cropping box.
    w, h = sensor_hw[1], sensor_hw[0]  # width, height
    fx0, fy0, cx0, cy0 = K_event[0, 0], K_event[1, 1], K_event[0, 2], K_event[1, 2]
    fx1, fy1, cx1, cy1 = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    xs = [cx1 + (fx1 / fx0) * (x - cx0) for x, y in corners]
    ys = [cy1 + (fy1 / fy0) * (y - cy0) for x, y in corners]

    return min(xs), max(xs), min(ys), max(ys)  # (x_min, x_max, y_min, y_max)

def _crop_rgb_to_event_fov(rgb_image, K_rect_event, K_rect_rgb, sensor_hw=(480, 640)):
    # Crop the RGB image to the field of view of the event camera using the calibration parameters.
    # This function builds upon the fact that the RGB image is already rectified (for DSEC).
    # The cropping is based on the intrinsic parameters of both cameras.

    # Get the image dimensions
    h, w = rgb_image.shape[:2]

    x0, x1, y0, y1 = _event_fov_crop_box(K_rect_event, K_rect_rgb, sensor_hw)
    # Round to nearest int for slicing (numpy slices require ints; rounding
    # is unbiased vs truncation and the sub-pixel difference is negligible).
    x0 = max(0, int(round(x0)))
    x1 = min(w, int(round(x1)))
    y0 = max(0, int(round(y0)))
    y1 = min(h, int(round(y1)))

    return rgb_image[y0:y1, x0:x1] 

def process_rgb_image(rgb_image_path, K_rect_event, K_rect_rgb, image_size=(384, 384),
                      sensor_hw=(480, 640)):
    # Process the RGB image: read, convert to RGB, crop to event FOV, resize.
    img = cv2.imread(str(rgb_image_path))
    if img is None:
        raise FileNotFoundError(f"RGB image not found at {rgb_image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_cropped = _crop_rgb_to_event_fov(img, K_rect_event, K_rect_rgb, sensor_hw)
    img_resized = cv2.resize(img_cropped, (image_size[0], image_size[1]), interpolation=cv2.INTER_AREA)
    
    return np.ascontiguousarray(img_resized.transpose(2, 0, 1))  # Convert to CHW (3, H, W) uint8 format

def process_event_histogram(x_r, y_r, p, hot_mask, out_size, sensor_hw=(480, 640)):
    """
    Channels:
        0 : positive polarity count (normalised)
        1 : negative polarity count (normalised)
        2 : net polarity, true zero where no events fired
    
    Inputs are already-rectified event coordinates.
    """
    pos_mask = p == 1
    neg_mask = ~pos_mask 

    pos_hist = _accumulate_events(x_r[pos_mask], y_r[pos_mask], sensor_hw)
    neg_hist = _accumulate_events(x_r[neg_mask], y_r[neg_mask], sensor_hw)

    pos_hist[hot_mask] = 0
    neg_hist[hot_mask] = 0

    net_hist = pos_hist - neg_hist  # net polarity histogram before normalisation
    net_max = np.max(np.abs(net_hist))
    if net_max > 0:
        net_hist /= net_max  # normalise net polarity to [-1, 1]

    pos_hist = _norm_unit_max(pos_hist)
    neg_hist = _norm_unit_max(neg_hist)

    pos_hist = _resize_to(pos_hist, out_size)
    neg_hist = _resize_to(neg_hist, out_size)
    net_hist = _resize_to(net_hist, out_size)

    return np.stack([pos_hist, neg_hist, net_hist], axis=0)  # shape: (3, H_out, W_out)

def process_sequence(seq, rgb_dir, events_dir, calibrations_dir, pairs_dir, preprocessed_rgb_dir,
                     preprocessed_events_hist_dir,
                     out_root, cfg):
    # Process a single sequence: load calibration, rectify map, timestamps, and build pairs of RGB and Events. 
    # Returns (n_pairs, n_skipped)
    seq_rgb_dir = rgb_dir / seq
    seq_events_file = events_dir / seq / "events.h5"
    seq_rectify_map_file = events_dir / seq / "rectify_map.h5"
    seq_calib_file = calibrations_dir / seq / "cam_to_cam.yaml"
    seq_pairs_file = pairs_dir / f"{seq}_pairs.txt"

    out_size = tuple(cfg.model.img_hw)  # (W, H) for cv2.resize
    sensor_hw = tuple(cfg.datasets.sensor_hw)
    frame_stride = int(cfg.datasets.stride)
    assert frame_stride >= 1, "frame_stride must be >= 1"

    # Load calibration parameters
    K_rect_event, K_rect_rgb = load_calibration_params(seq_calib_file)
    rectify_map = load_rectify_map(seq_rectify_map_file)

    # Load timestamps for RGB images
    rgb_timestamps_file = seq_rgb_dir / f"{seq}_image_timestamps.txt"
    timestamps_us = np.loadtxt(rgb_timestamps_file, dtype=np.int64)  # Load timestamps in microseconds
    print(f"Sequence {seq}: Loaded {len(timestamps_us)} RGB timestamps.")

    seq_rgb_images = sorted(seq_rgb_dir.glob("*.png"))  # List of RGB image files sorted by name
    assert len(seq_rgb_images) == len(timestamps_us), \
        f"Mismatch in number of RGB images and timestamps for sequence {seq}."
    
    pairs_lines = []
    n_skipped = 0

    with h5py.File(seq_events_file, 'r') as h5f:
        event_slicer = EventSlicer(h5f)

        for i in range(1, len(timestamps_us), frame_stride):
            # Inter-frame window: exact, non-overlapping event coverage
            # matching RGB teacher frame transitions (~50ms at 20Hz).
            # frame_stride only selects WHICH pairs are saved; the window
            # is always the true consecutive [t[i-1], t[i]].
            t_start_us = timestamps_us[i-1]
            t_end_us = timestamps_us[i]

            rgb_output_path = preprocessed_rgb_dir / f"{seq}_{i:06d}.npy"
            event_hist_output_path = preprocessed_events_hist_dir / f"{seq}_{i:06d}.npy"

            # pairs.txt stores paths relative to the output root, not absolute paths
            # This is important for portability and consistency across different environments.
            pair_line = (f"{t_end_us},"
                         f"{rgb_output_path.relative_to(out_root)},"
                         f"{event_hist_output_path.relative_to(out_root)}")

            # skip frames whose output exists (if resumed)
            if (rgb_output_path.exists() and
                event_hist_output_path.exists()):
                pairs_lines.append(pair_line)
                continue

            # EventSlicer.get_events() expects microseconds and returns a dict
            # of (p, x, y, t) arrays, or None if the window is outside the
            # recorded range. A valid window with zero events returns EMPTY
            # arrays, not None -- guard both
            sliced_events = event_slicer.get_events(t_start_us, t_end_us)
            if sliced_events is None or sliced_events['t'].size == 0:
                print(f"Warning: No events found between {t_start_us} and {t_end_us} for sequence {seq}.")
                n_skipped += 1
                continue

            # Rectify once per window; share coords, timestamps, and the
            # hot-pixel mask between both representations.
            x_r, y_r, p, t = rectify_event_coords(sliced_events, rectify_map, sensor_hw)
            if t.size == 0: # all events rectified out of frame (unlikely)
                print(f"Warning: All events rectified out of frame between {t_start_us} and {t_end_us} for sequence {seq}.")
                n_skipped += 1
                continue
            hot_mask = compute_hot_pixel_mask(x_r, y_r, sensor_hw=sensor_hw,
                                              threshold=cfg.datasets.hot_pixel_threshold)

            rgb_processed = process_rgb_image(seq_rgb_images[i], K_rect_event, K_rect_rgb,
                                              image_size=out_size, sensor_hw=sensor_hw)
            event_histogram_processed = process_event_histogram(x_r, y_r, p, hot_mask, out_size=out_size, sensor_hw=sensor_hw)

            # uint8 RGB, fp16 events: ~2.2MB/frame vs ~5.3MB at full float32.
            np.save(rgb_output_path, rgb_processed)
            np.save(event_hist_output_path, event_histogram_processed.astype(np.float16))

            pairs_lines.append(pair_line)


    with open(seq_pairs_file, 'w') as f:
        for line in pairs_lines:
            f.write(line + '\n')    

    return (len(pairs_lines), n_skipped)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def build_pairs(cfg: DictConfig):
    # Builds pairs of RGB and Event data based on timestamps and saves them in the specified output directory.
    dataset_path = Path(cfg.dsec_path)
    out_root = Path(cfg.output_dir) / "preprocessed_dsec"
    preprocessed_rgb_dir = Path(cfg.output_dir) / "preprocessed_dsec" / "rgb"
    preprocessed_events_hist_dir = Path(cfg.output_dir) / "preprocessed_dsec" / "events" / "histogram"
    pairs_dir = out_root / "pairs_dsec"

    preprocessed_rgb_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_events_hist_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = dataset_path / "RGB"
    events_dir = dataset_path / "Events"
    calibrations_dir = dataset_path / "Calibrations"

    sequences = [seq for seq in sorted(os.listdir(rgb_dir)) if os.path.isdir(rgb_dir / seq)]

    failed = []
    for seq in sequences:
        # processing sequentially for each sequence (total: 53 sequences)
        try:
            n_pairs, n_skipped = process_sequence(seq, rgb_dir, events_dir, calibrations_dir, pairs_dir,
                                                  preprocessed_rgb_dir, preprocessed_events_hist_dir,
                                                  out_root, cfg)
            print(f"Sequence {seq}: Processed {n_pairs} pairs" + (f", skipped {n_skipped} pairs" if n_skipped > 0 else ""))
        except Exception as e:
            print(f"Error processing sequence {seq}: {e}")
            failed.append((seq, repr(e)))

    # Concatenate per-sequence pairs files into the single master pairs.txt
    # consumed by the dataloader. Per-sequence files are kept for reruns.
    master_pairs_file = out_root / cfg.datasets.pairs_name
    total = 0
    with open(master_pairs_file, 'w') as master_f:
        for seq in sequences:
            seq_pairs_file = pairs_dir / f"{seq}_pairs.txt"
            if seq_pairs_file.exists():
                with open(seq_pairs_file, 'r') as seq_f:
                    for line in seq_f:
                        if line.strip():  # Avoid writing empty lines
                            master_f.write(line if line.endswith('\n') else line + '\n')
                            total += 1
                            
    print(f"Total pairs in master file: {total}")

    if failed:
        fail_log = out_root / "failed_sequences.txt"
        with open(fail_log, 'w') as f:
            for seq, err in failed:
                f.write(f"{seq}: {err}\n")
        print(f"Failed sequences logged in {fail_log}. "
              f"Re-run after fixing; existing outputs will be skipped (on resume)")


if __name__ == "__main__":
    build_pairs()