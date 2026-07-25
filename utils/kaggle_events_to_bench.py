"""Replace the ensemble-bench datasample's subsampled events with our FULL
Brisbane events.

The released datasample's events.parquet is temporally subsampled (most 1 s
windows are empty), so it yields too few frames. This copies the FULL events
from our Kaggle Brisbane parquet into the datasample's
paraquet_data/<seq>/events.parquet (columns t[sec], x, y, p), leaving the
datasample's gps.txt and hot_pixels.txt untouched. Same event clock as the
datasample, so no time offset is applied.

Usage:
  python utils/kaggle_events_to_bench.py \
      --kaggle_root /workspace/E-LiteVPR/datasets/Brisbane_6_set \
      --datasample  .../datasample_for_ensem_event_bench/Brisbane \
      --config      configs/datasets/brisbane.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def find_key_file(root, key, suffix):
    hits = sorted(f for f in Path(root).rglob(f"*{suffix}") if key in f.name)
    if not hits:
        raise FileNotFoundError(f"no *{suffix} containing '{key}' under {root}")
    return str(hits[0])


def resolve_cols(pf):
    names = {n.lower(): n for n in pf.schema_arrow.names}
    def pick(*cands):
        for c in cands:
            if c in names:
                return names[c]
        raise KeyError(f"none of {cands} in {pf.schema_arrow.names}")
    return pick("x"), pick("y"), pick("t", "timestamp", "time"), pick("p", "pol", "polarity")


def scale_of(t_last):
    if t_last > 1e17:
        return 1e-9
    if t_last > 1e14:
        return 1e-6
    if t_last > 1e11:
        return 1e-3
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_root", required=True)
    ap.add_argument("--datasample", required=True, help=".../datasample.../Brisbane")
    ap.add_argument("--config", default="configs/datasets/brisbane.yaml")
    args = ap.parse_args()

    traverses = yaml.safe_load(open(args.config))["traverses"]
    schema = pa.schema([("t", pa.float64()), ("x", pa.int16()),
                        ("y", pa.int16()), ("p", pa.int8())])

    for seq, key in traverses.items():
        src = find_key_file(args.kaggle_root, key, ".parquet")
        out = Path(args.datasample) / "paraquet_data" / seq / "events.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)

        pf = pq.ParquetFile(src)
        cx, cy, ct, cp = resolve_cols(pf)
        nrg = pf.metadata.num_row_groups
        t_last = float(pf.read_row_group(nrg - 1, columns=[ct]).column(0)[-1].as_py())
        sc = scale_of(t_last)

        writer = pq.ParquetWriter(out, schema)
        n = 0
        for rb in pf.iter_batches(batch_size=5_000_000, columns=[cx, cy, ct, cp]):
            x = rb.column(0).to_numpy(zero_copy_only=False).astype(np.int16)
            y = rb.column(1).to_numpy(zero_copy_only=False).astype(np.int16)
            t = rb.column(2).to_numpy(zero_copy_only=False).astype(np.float64) * sc
            p = np.where(np.nan_to_num(rb.column(3).to_numpy(zero_copy_only=False)) > 0,
                         1, 0).astype(np.int8)
            writer.write_table(pa.table({"t": t, "x": x, "y": y, "p": p},
                                        schema=schema))
            n += len(t)
        writer.close()
        print(f"{seq}: {n:,} events (scale {sc:g}) -> {out}")


if __name__ == "__main__":
    main()
