"""Extract GPS-tagged event slices from NYC-Event-VPR raw recordings for
Phase 2 triplet training. Output format matches mvsec_extract_slices.py, so
phase2_mine_triplets.py consumes both unchanged. All sessions share one
geographic frame: slices carry slice_xy_map with map_frame='gps'.

Per session:
  events  : EVT3 .raw (single file, or dir of chunk files), decoded with
            the local spec-based evt3_decoder.py (expelliarmus dropped:
            its EVT3 decode doubles elapsed time on real Prophesee
            streams -- see evt3_decoder.py docstring), streamed
            chunk-wise (never fully in RAM). Raw timestamps are relative
            us; every file is rebased to its first event and anchored to
            its filename wall time (local tz) -- EVT3's 24-bit rollover
            makes cross-file offsets in the stream ambiguous, filenames
            are ~1 s accurate, good enough.
  gps     : GPS_data_*.csv; column detection is heuristic (lat/lon/time by
            name, epoch scale by magnitude) and fails loudly with the file
            head printed -- paste that back if it ever raises.

Usage (Colab):
  python nyc_extract_slices.py datasets=nyc                # all sessions
  python nyc_extract_slices.py datasets=nyc \
      datasets.session=sensor_data_2022-12-09_13-59-10
"""
from datetime import datetime
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

def filename_epoch(path, tz):
    """'..._YYYY-MM-DD_HH-MM-SS' (any prefix) -> epoch seconds."""
    stem = Path(path).stem
    token = "_".join(stem.split("_")[-2:])          # date_time tail
    dt = datetime.strptime(token, "%Y-%m-%d_%H-%M-%S")
    return dt.replace(tzinfo=ZoneInfo(tz)).timestamp()

def parse_gps_csv(path, tz):
    """(t_epoch[s], lat, lon) with heuristic column/format detection."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def find(*keys):
        for k, orig in cols.items():
            if any(key in k for key in keys):
                return orig
        return None

    lat_c = find("lat")
    lon_c = find("lon", "lng")
    t_c = find("time", "stamp", "utc", "epoch")
    if not (lat_c and lon_c and t_c):
        raise RuntimeError(
            f"GPS csv columns not recognised in {path}.\n"
            f"columns: {list(df.columns)}\nhead:\n{df.head(3)}\n"
            "-> paste this output back to adapt parse_gps_csv().")

    lat = df[lat_c].to_numpy(float)
    lon = df[lon_c].to_numpy(float)
    traw = df[t_c]

    def epoch_from_numeric(tv):
        m = np.nanmedian(tv)
        scale = 1e-9 if m > 1e17 else 1e-6 if m > 1e14 else \
            1e-3 if m > 1e11 else 1.0
        return tv * scale

    if pd.api.types.is_numeric_dtype(traw):
        t = epoch_from_numeric(traw.to_numpy(float))
    else:
        numeric = pd.to_numeric(traw, errors="coerce")   # epoch as string
        if numeric.notna().mean() > 0.9:
            t = epoch_from_numeric(numeric.to_numpy(float))
        else:                                            # datetime strings
            # NYC-Event-VPR format: 2022-12-09_13-59-12_937719 (local time)
            parsed = pd.to_datetime(traw, errors="coerce",
                                    format="%Y-%m-%d_%H-%M-%S_%f")
            if parsed.isna().all():                      # other formats
                parsed = pd.to_datetime(traw, errors="coerce")
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize(ZoneInfo(tz))
            # total_seconds is datetime-resolution-independent; NaT -> NaN
            t = (parsed - pd.Timestamp(0, tz="UTC")) \
                .dt.total_seconds().to_numpy()
    ok = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) \
        & (np.abs(lat) > 1) & (np.abs(lon) > 1)
    # plausible epoch seconds: 2001..2096
    ok &= (t > 1e9) & (t < 4e9)
    if ok.sum() < max(2, 0.5 * len(df)):
        with pd.option_context("display.width", 200,
                               "display.max_columns", None):
            raise RuntimeError(
                f"GPS time column {t_c!r} in {path} did not parse to "
                f"plausible epochs ({ok.sum()}/{len(df)} valid).\n"
                f"columns: {list(df.columns)}\ndtypes:\n{df.dtypes}\n"
                f"head:\n{df.head(5)}\n"
                "-> paste this output back to adapt parse_gps_csv().")
    t, lat, lon = t[ok], lat[ok], lon[ok]
    order = np.argsort(t)
    return t[order], lat[order], lon[order]

def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R
    return np.stack([x, y], axis=1)

def session_raw_files(sess_dir):
    """Sorted event-source list: data_*.raw, data_*.zip (each streamed
    without extraction), or data_*/ dir of chunk raws. Refuses to guess
    when both raws and zips are present."""
    direct = sorted(sess_dir.glob("data_*.raw"))
    zips = sorted(sess_dir.glob("data_*.zip"))
    if direct and zips:
        raise RuntimeError(
            f"{sess_dir} has both data_*.raw and data_*.zip -- remove one "
            "(truncated raws from the old wget download should be deleted)")
    if direct:
        return direct
    if zips:
        return zips
    sub = sorted(d for d in sess_dir.glob("data_*") if d.is_dir())
    if sub:
        chunks = sorted(sub[0].glob("*.raw"))
        if chunks:
            return chunks
    raise FileNotFoundError(f"no .raw/.zip files under {sess_dir}")

def stream_session_events(raw_files, tz, encoding):
    """Yield (x, y, t_epoch, p) arrays in time order across chunk files.
    Each file is rebased to its own first event and anchored to its
    filename wall time: EVT3's 24-bit time counter (~16.8 s rollover,
    wrap count not stored) makes decoded start offsets between chunk
    files ambiguous, so cross-file continuity cannot be inferred from
    the stream. Filename anchoring is ~1 s accurate -- within the
    training alignment budget. Leading events overlapping the previous
    file's tail (second-resolution filename jitter) are dropped."""
    from evt3_decoder import Evt3Decoder, stream_evt3
    if str(encoding).lower() != "evt3":
        raise NotImplementedError(
            f"encoding {encoding!r}: only evt3 is supported")
    if len(raw_files) > 1:
        print(f"  chunked session: {len(raw_files)} files, "
              f"per-file filename anchoring")
    hwm = -np.inf                       # high-water mark across files
    for f in raw_files:
        anchor = filename_epoch(f, tz)
        dec = Evt3Decoder()
        rebase = None
        for ex, ey, et_us, ep in stream_evt3(str(f), decoder=dec):
            t = et_us.astype(np.float64) * 1e-6
            if rebase is None:
                rebase = t[0]
            t = t - rebase + anchor
            if t[0] <= hwm:             # clip boundary overlap
                k = int(np.searchsorted(t, hwm, "right"))
                ex, ey, t, ep = ex[k:], ey[k:], t[k:], ep[k:]
                if t.size == 0:
                    continue
            hwm = t[-1]
            yield ex.astype(np.int32), ey.astype(np.int32), t, ep
        print(f"  {Path(f).name}: {dec.n_events} events, "
              f"{(hwm - anchor):.2f}s, max_backstep {dec.max_backstep_us} us"
              + (f", {dec.skipped_words} non-CD words skipped"
                 if dec.skipped_words else ""))

def extract_session(sess_dir, label, out_path, cfg_d):
    tz = cfg_d.timezone
    raw_files = session_raw_files(Path(sess_dir))
    gps_files = sorted(Path(sess_dir).glob("GPS_data_*.csv"))
    assert gps_files, f"no GPS csv in {sess_dir}"
    gps_t, lat, lon = parse_gps_csv(gps_files[0], tz)
    xy_all = latlon_to_xy(lat, lon, float(cfg_d.lat0), float(cfg_d.lon0))
    print(f"  gps: {len(gps_t)} fixes, {gps_t[0]:.0f}..{gps_t[-1]:.0f} "
          f"({gps_t[-1]-gps_t[0]:.0f}s)")

    half = float(cfg_d.slice_ms) / 2e3
    hz = float(cfg_d.hz)
    ev_start = filename_epoch(raw_files[0], tz)
    lo = max(ev_start, gps_t[0]) + half
    hi = gps_t[-1] - half
    centers = np.arange(lo, hi, 1.0 / hz)
    if centers.size < 2:
        raise RuntimeError(
            f"no event/GPS time overlap: events start {ev_start:.0f}, "
            f"gps {gps_t[0]:.0f}..{gps_t[-1]:.0f} -- check timezone "
            f"({tz}) and GPS time parsing")
    cx = np.interp(centers, gps_t, xy_all[:, 0])
    cy = np.interp(centers, gps_t, xy_all[:, 1])
    speed = np.hypot(np.gradient(cx, 1.0 / hz), np.gradient(cy, 1.0 / hz))
    keep = speed >= float(cfg_d.min_speed)
    centers, cx, cy = centers[keep], cx[keep], cy[keep]
    print(f"  {keep.sum()}/{len(keep)} samples moving @ {hz} Hz")
    t_lo, t_hi = centers - half, centers + half

    H, W = (int(v) for v in cfg_d.sensor_hw)
    offsets = np.zeros(len(centers) + 1, np.int64)
    n_empty, j = 0, 0

    # slices are streamed to disk as they complete -- a full NYC session
    # holds ~1e10 events, far beyond RAM
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o = h5py.File(out_path, "w")
    ev_ds = {k: o.create_dataset(k, shape=(0,), maxshape=(None,), dtype=dt,
                                 chunks=(1 << 20,), compression="lzf")
             for k, dt in [("events_x", np.uint16), ("events_y", np.uint16),
                           ("events_t_us", np.uint32),
                           ("events_p", np.uint8)]}

    def append(key, arr):
        d = ev_ds[key]
        n0 = d.shape[0]
        d.resize((n0 + len(arr),))
        d[n0:] = arr

    def emit(j, bx, by, bt, bp):
        a = np.searchsorted(bt, t_lo[j], "left")
        b = np.searchsorted(bt, t_hi[j], "left")
        if b <= a:
            nonlocal n_empty
            n_empty += 1
            offsets[j + 1] = offsets[j]
            return
        ok = (bx[a:b] >= 0) & (bx[a:b] < W) & (by[a:b] >= 0) & (by[a:b] < H)
        append("events_x", bx[a:b][ok].astype(np.uint16))
        append("events_y", by[a:b][ok].astype(np.uint16))
        append("events_t_us",
               np.round((bt[a:b][ok] - t_lo[j]) * 1e6).astype(np.uint32))
        append("events_p", (bp[a:b][ok] > 0).astype(np.uint8))
        offsets[j + 1] = offsets[j] + int(ok.sum())

    bx = np.empty(0, np.int32); by = np.empty(0, np.int32)
    bt = np.empty(0, np.float64); bp = np.empty(0, np.uint8)
    last_t = -np.inf
    for x, y, t, p in tqdm(stream_session_events(
            raw_files, tz, cfg_d.encoding), desc="chunks", unit="chunk"):
        if t.size == 0:
            continue
        if t[0] < last_t:
            raise RuntimeError("event chunks not time-ordered; check the "
                               "per-file/continuous anchor detection")
        last_t = t[-1]
        bx = np.concatenate([bx, x]); by = np.concatenate([by, y])
        bt = np.concatenate([bt, t]); bp = np.concatenate([bp, p])
        while j < len(centers) and bt.size and t_hi[j] <= bt[-1]:
            emit(j, bx, by, bt, bp)
            j += 1
            if j < len(centers):
                k = np.searchsorted(bt, t_lo[j], "left")
                bx, by, bt, bp = bx[k:], by[k:], bt[k:], bp[k:]
    while j < len(centers):
        emit(j, bx, by, bt, bp)
        j += 1
    if n_empty:
        print(f"  warning: {n_empty} empty slices")

    o.create_dataset("slice_offsets", data=offsets)
    o.create_dataset("slice_center_ts", data=centers)
    o.create_dataset("slice_xy_local", data=np.stack([cx, cy], 1))
    o.create_dataset("slice_xy_map", data=np.stack([cx, cy], 1))
    o.attrs.update(dict(
        sequence=Path(sess_dir).name, label=label,
        slice_ms=float(cfg_d.slice_ms), hz=hz,
        min_speed=float(cfg_d.min_speed), rectified=False,
        sensor_hw=(H, W), frame="gps", map_frame="gps"))
    o.close()
    print(f"  -> {out_path}  ({len(centers)} slices, "
          f"{offsets[-1]/1e6:.1f}M events, "
          f"{out_path.stat().st_size/1e6:.0f} MB)")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    d = cfg.datasets
    names = list(d.sessions.keys()) if d.session == "all" else [d.session]
    for name in names:
        label = d.sessions[name]
        print(f"===== {name} [{label}] =====")
        stamp = "_".join(name.split("_")[-2:])
        extract_session(Path(d.root) / name, label,
                        Path(d.slices_dir) / f"nyc_{stamp}_{label}.h5", d)

if __name__ == "__main__":
    main()
