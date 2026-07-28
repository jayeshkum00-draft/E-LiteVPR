"""
Colab driver for the normalisation-variant preprocessor.

Combines the two halves that already exist:
  * preprocess_dsec_colab.py -- staging, zip checkpointing, resume, master
    pairs.txt built from the zips on Drive. All the Colab survival machinery.
  * preprocess_unnorm.py     -- the representation change itself.

It contains no science and no file-moving logic of its own; it only patches
the first with the second and installs the guards that stop the two runs from
contaminating each other.

WHY THIS FILE EXISTS RATHER THAN A FLAG
---------------------------------------
preprocess_dsec_colab.py treats a sequence as durably DONE the moment its zip
exists on Drive (`if zip_dest.exists(): continue`). Pointed at the unit-max
zip directory, a p99 run would skip all 41 sequences, rebuild pairs.txt from
the OLD zips, print a clean success, and hand you the unit-max corpus under a
new name. That failure is silent and would only surface as a confusing null
result after a full retrain. So the zip directory is mode-suffixed by default
and the driver refuses to share one with the baseline.

Usage (Colab cell, Drive mounted):

    !python utils/preprocess_unnorm_colab.py \
        norm_mode=p99 \
        dsec_path=/content/drive/MyDrive/E-LiteVPR/DSEC \
        output_dir=/content/outputs_p99

Zips land in <baseline_zip_dir>_p99 unless you pass zip_dir=... explicitly.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

import preprocess_dsec_colab as colab
import preprocess_unnorm as unnorm

BASELINE_ZIP_DIR = colab.DRIVE_ZIP_DIR


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.get("norm_mode", "p99")
    q = float(cfg.get("norm_percentile", unnorm.DEFAULT_PERCENTILE))

    # Baseline self-test FIRST, while the pipeline is still unpatched: it
    # asserts unit-max semantics, so it is only meaningful pre-patch. It
    # checks rectification direction and hot-pixel detection, which the
    # patch does not touch and which a broken environment would break.
    colab.self_test()
    # Then the variant's own: proves mode="max" reproduces the committed
    # pipeline bit-for-bit, and that p99 actually changes what it claims to.
    unnorm.self_test()

    fn = unnorm.patch(mode, q)
    # preprocess_dsec_colab did `from preprocess_dsec import
    # process_event_histogram`, so it holds its own reference. Rebind it too,
    # so anything reading it from that module sees the variant.
    colab.process_event_histogram = fn

    zip_dir = Path(cfg.get("zip_dir") or f"{BASELINE_ZIP_DIR}_{mode}")
    if zip_dir.resolve() == BASELINE_ZIP_DIR.resolve():
        raise SystemExit(
            f"REFUSING TO START: zip_dir is the baseline unit-max directory "
            f"({BASELINE_ZIP_DIR}).\nSequences with an existing zip are "
            f"skipped, so this run would silently reuse the unit-max corpus "
            f"and report success. Choose a different zip_dir.")

    out_root = Path(cfg.output_dir) / "preprocessed_dsec"
    unnorm.assert_clean_output_dir(out_root, bool(cfg.get("allow_existing", False)))

    colab.DRIVE_ZIP_DIR = zip_dir
    zip_dir.mkdir(parents=True, exist_ok=True)
    print(f"norm_mode={mode}" + (f" (q={q})" if mode == "p99" else "")
          + f"\nzips   -> {zip_dir}\noutputs-> {out_root}")

    # .__wrapped__ is the undecorated body; colab.main is itself a @hydra.main
    # entry point and calling it directly would start a second Hydra run.
    colab.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
