"""
Normalisation-variant driver for the E-LiteVPR event preprocessors.

Nothing here is science-neutral plumbing: this file changes the event
representation itself. It exists as a SEPARATE driver so that
`preprocess_dsec.py` / `preprocess_ddd20.py` stay exactly as they were for
the published unit-max corpus, and the ablation is a different command
rather than a different git state.

WHY
---
The committed pipeline normalises each histogram channel by its own max
(`preprocess_dsec._norm_unit_max`, and the inline `net_max` divide). A single
surviving high-count pixel therefore sets the scale for the whole frame. The
99.5th-percentile hot-pixel mask removes only the extreme tail, so at night --
where events are sparse and dominated by a handful of point sources
(headlights, streetlights, specular reflections) -- the max can sit far above
the bulk of the scene, crushing genuine structure toward zero. That is the
"representation starvation" hypothesis: no aggregation head can recover
detail the input never encoded.

MODES (cfg.norm_mode)
---------------------
  p99  : divide by the q-th percentile of NONZERO pixels, then clip.
         Robust scale estimate; the point sources saturate at 1.0 instead of
         dictating the scale. Percentile is taken over nonzero pixels only --
         over ALL pixels it would be ~0 at night, where most of the frame is
         empty, and the divide would explode. This mirrors how
         `compute_hot_pixel_mask` already picks its threshold.
  raw  : no per-frame normalisation at all; store counts as-is. The true
         "unnormalised" control. Cross-frame comparable, but the dynamic
         range is dataset-dependent -- expect this to need a different LR.
  max  : identical to the committed pipeline. Present so the self-test can
         prove this driver reproduces the original bit-for-bit.

The percentile is applied AFTER hot-pixel masking (the mask has already
zeroed those pixels), so it reflects real scene content.

Usage
-----
    python utils/preprocess_unnorm.py source=dsec norm_mode=p99 \
        dsec_path=... output_dir=/path/to/preprocessed_dsec_p99

    python utils/preprocess_unnorm.py source=ddd20 norm_mode=p99 \
        datasets=ddd20 ddd20_path=... output_dir=...

MUST write to a NEW output_dir. Both preprocessors resume by skipping frames
whose .npy already exists, so pointing this at the unit-max output would
silently keep every old frame and produce a corpus that is a mix of two
normalisations. This driver refuses to start in that situation unless you
pass allow_existing=true.
"""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

import preprocess_dsec
from preprocess_dsec import _accumulate_events, _resize_to

DEFAULT_PERCENTILE = 99.0


# --- the normalisers ---------------------------------------------------------

def _norm_percentile(arr, q):
    """
    Scale by the q-th percentile of nonzero magnitude, then clip to [-1, 1].

    Handles the signed (net) channel and the unsigned (pos/neg) channels with
    the same code: magnitude drives the scale, the clip is symmetric, and a
    non-negative input simply never uses the lower bound.
    """
    nonzero = np.abs(arr[arr != 0])
    if nonzero.size == 0:
        return arr
    scale = np.percentile(nonzero, q)
    if scale <= 0:
        return arr
    return np.clip(arr / scale, -1.0, 1.0)


def _norm_unit_max(arr):
    m = np.max(np.abs(arr))
    return arr / m if m > 0 else arr


def _norm_identity(arr):
    return arr


def _get_normaliser(mode, q):
    if mode == "p99":
        return lambda a: _norm_percentile(a, q)
    if mode == "raw":
        return _norm_identity
    if mode == "max":
        return _norm_unit_max
    raise ValueError(f"unknown norm_mode {mode!r}; expected one of p99, raw, max")


# --- the replacement representation ------------------------------------------

def make_histogram_fn(mode, q=DEFAULT_PERCENTILE):
    """
    Build a drop-in replacement for preprocess_dsec.process_event_histogram
    with the channel normalisation swapped out.

    Channel semantics, event ordering, hot-pixel handling and resize are
    copied verbatim from the original so that `mode="max"` is bit-identical;
    only the three normalise calls differ.
    """
    normalise = _get_normaliser(mode, q)

    def process_event_histogram(x_r, y_r, p, hot_mask, out_size, sensor_hw=(480, 640)):
        pos_mask = p == 1
        neg_mask = ~pos_mask

        pos_hist = _accumulate_events(x_r[pos_mask], y_r[pos_mask], sensor_hw)
        neg_hist = _accumulate_events(x_r[neg_mask], y_r[neg_mask], sensor_hw)

        pos_hist[hot_mask] = 0
        neg_hist[hot_mask] = 0

        # net is formed from RAW counts, before either channel is scaled --
        # exactly as in the original. Normalising first would make a balanced
        # pixel non-zero whenever pos and neg had different maxima.
        net_hist = normalise(pos_hist - neg_hist)
        pos_hist = normalise(pos_hist)
        neg_hist = normalise(neg_hist)

        return np.stack([_resize_to(pos_hist, out_size),
                         _resize_to(neg_hist, out_size),
                         _resize_to(net_hist, out_size)], axis=0)

    process_event_histogram.norm_mode = mode
    process_event_histogram.norm_percentile = q
    return process_event_histogram


def patch(mode, q=DEFAULT_PERCENTILE):
    """
    Rebind the histogram function everywhere it is reachable.

    preprocess_ddd20 does `from preprocess_dsec import process_event_histogram`
    (preprocess_ddd20.py:72-75), which copies the function object into its own
    module globals. Patching only preprocess_dsec would leave DDD20 on unit-max
    and produce a corpus normalised two different ways -- so both namespaces
    are rebound here. preprocess_dsec.process_sequence resolves the name from
    its module globals at call time, so the rebinding takes effect.
    """
    fn = make_histogram_fn(mode, q)
    preprocess_dsec.process_event_histogram = fn

    # Optional: only patched if DDD20's own dependencies import cleanly.
    if "preprocess_ddd20" in sys.modules:
        sys.modules["preprocess_ddd20"].process_event_histogram = fn
    return fn


# --- guards ------------------------------------------------------------------

def self_test():
    """
    Prove three things before a multi-hour run starts:
      1. mode="max" reproduces the committed pipeline bit-for-bit,
      2. p99 actually rescues structure that unit-max crushes,
      3. channel semantics (balanced pixel -> net 0) survive the swap.
    """
    sensor_hw = (480, 640)
    h, w = sensor_hw
    rng = np.random.default_rng(0)
    out_size = (w, h)  # full-res, so resize is a no-op and semantics are exact

    n_bg = 30000
    x = rng.integers(0, w, n_bg)
    y = rng.integers(0, h, n_bg)
    p = rng.integers(0, 2, n_bg)
    # balanced pixel at (x=50, y=60): 40 pos + 40 neg
    x = np.concatenate([x, np.full(80, 50)]); y = np.concatenate([y, np.full(80, 60)])
    p = np.concatenate([p, np.tile([1, 0], 40)])
    # pos-heavy pixel at (x=51, y=61)
    x = np.concatenate([x, np.full(60, 51)]); y = np.concatenate([y, np.full(60, 61)])
    p = np.concatenate([p, np.ones(60, dtype=np.int64)])
    hot = np.zeros(sensor_hw, dtype=bool)

    ref = preprocess_dsec.process_event_histogram(x, y, p, hot, out_size, sensor_hw)
    got = make_histogram_fn("max")(x, y, p, hot, out_size, sensor_hw)
    assert np.allclose(ref, got, atol=1e-6), \
        "SELF-TEST FAIL: mode='max' does not reproduce the committed pipeline"

    h99 = make_histogram_fn("p99")(x, y, p, hot, out_size, sensor_hw)
    assert h99.shape == (3, h, w), f"SELF-TEST FAIL: shape {h99.shape}"
    assert abs(h99[2][60, 50]) < 1e-6, \
        f"SELF-TEST FAIL: balanced pixel net={h99[2][60, 50]}, expected 0"
    assert h99[2][61, 51] > 0, "SELF-TEST FAIL: pos-heavy pixel has non-positive net"
    assert h99.max() <= 1.0 + 1e-6 and h99.min() >= -1.0 - 1e-6, \
        "SELF-TEST FAIL: p99 output escaped [-1, 1] (clip not applied)"

    # The point of the variant: inject one dominant source and check that the
    # ordinary background retains far more amplitude under p99 than under max.
    xd = np.concatenate([x, np.full(4000, 300)])
    yd = np.concatenate([y, np.full(4000, 300)])
    pd = np.concatenate([p, np.ones(4000, dtype=np.int64)])
    bg_max = make_histogram_fn("max")(xd, yd, pd, hot, out_size, sensor_hw)[0][61, 51]
    bg_p99 = make_histogram_fn("p99")(xd, yd, pd, hot, out_size, sensor_hw)[0][61, 51]
    assert bg_p99 > bg_max * 5, (
        f"SELF-TEST FAIL: p99 did not rescue background structure "
        f"(max={bg_max:.5f}, p99={bg_p99:.5f}) -- the variant is a no-op")

    print(f"SELF-TEST PASS: max-mode identical to committed pipeline; "
          f"p99 lifts background {bg_p99 / max(bg_max, 1e-12):.0f}x")


def assert_clean_output_dir(out_root, allow_existing):
    """
    Both preprocessors resume by skipping frames whose .npy already exists.
    Reusing the unit-max output dir would therefore keep every old frame and
    yield a silently mixed-normalisation corpus -- the exact failure this
    ablation cannot survive.
    """
    if allow_existing:
        return
    out_root = Path(out_root)
    if not out_root.exists():
        return
    existing = next(out_root.rglob("*.npy"), None)
    if existing is not None:
        raise SystemExit(
            f"REFUSING TO START: {out_root} already contains .npy frames "
            f"(e.g. {existing}).\nThe preprocessors skip existing frames on "
            f"resume, so this run would mix normalisations. Point output_dir "
            f"at a new directory, or pass allow_existing=true if you are "
            f"deliberately resuming an interrupted run of THIS variant.")


# --- entry point -------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.get("norm_mode", "p99")
    q = float(cfg.get("norm_percentile", DEFAULT_PERCENTILE))
    source = cfg.get("source", "dsec")

    self_test()

    if source == "dsec":
        out_root = Path(cfg.output_dir) / "preprocessed_dsec"
        build = preprocess_dsec.build_pairs
    elif source == "ddd20":
        import preprocess_ddd20  # noqa: F401  (registers the module for patch())
        out_root = Path(cfg.output_dir) / "preprocessed_ddd20"
        build = preprocess_ddd20.build_pairs
    else:
        raise SystemExit(f"unknown source {source!r}; expected dsec or ddd20")

    assert_clean_output_dir(out_root, bool(cfg.get("allow_existing", False)))
    patch(mode, q)
    print(f"norm_mode={mode}" + (f" (q={q})" if mode == "p99" else "")
          + f", source={source}, out_root={out_root}")

    # .__wrapped__ is the undecorated body: build_pairs is itself a @hydra.main
    # entry point, and calling it directly would try to start a second Hydra
    # run and re-parse sys.argv. functools.wraps sets __wrapped__, and Hydra's
    # own unwrap loop depends on it, so relying on it is safe.
    build.__wrapped__(cfg)


if __name__ == "__main__":
    main()
