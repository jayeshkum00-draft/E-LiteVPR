"""Build the Phase 2 triplet-mining index from slice .h5 files
(mvsec_extract_slices.py output; NYC slices in the same format later).

Grouping logic:
  - sequences sharing a registration frame (slice_xy_map + map_frame attr,
    or being that frame's reference) are cross-minable;
  - sequences without registration (slice_xy_local only) mine within
    themselves.

Per anchor slice, two positive sets:
  same-condition positives  -- within pos_radius, same day/night tag,
                               temporal gap > min_dt if same sequence
                               (excludes trivial same-pass neighbours)
  cross-condition positives -- within cross_radius, opposite tag
                               (registration error budget -> wider radius)
Negatives are implicit at train time: any slice in the same frame group
farther than neg_floor (positions stored for the sampler).

Output .npz: slice table (file, index, xy, is_night, seq_id) + CSR arrays
(pos_same_indptr/indices, pos_cross_indptr/indices) over global slice ids.

Usage:
  python phase2_mine_triplets.py datasets=mvsec
  python phase2_mine_triplets.py datasets=mvsec \
      datasets.slices_dir=/content/slices datasets.index_out=/content/idx.npz
"""
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig


def load_slices(slices_dir, night_keyword):
    seqs = []
    for p in sorted(Path(slices_dir).glob("*.h5")):
        with h5py.File(p, "r") as f:
            if "slice_xy_map" in f:
                xy = f["slice_xy_map"][:]
                frame = f.attrs["map_frame"]
            else:
                xy = f["slice_xy_local"][:]
                frame = p.stem          # its own frame
            seqs.append(dict(
                file=p.name, stem=p.stem, xy=xy,
                ts=f["slice_center_ts"][:], frame=str(frame),
                is_night=night_keyword in p.stem.lower()))
        print(f"  {p.name}: {len(seqs[-1]['xy'])} slices, "
              f"frame={seqs[-1]['frame']}, night={seqs[-1]['is_night']}")
    if not seqs:
        raise FileNotFoundError(f"no *.h5 slice files in {slices_dir}")
    return seqs


def build_csr(pair_lists, n):
    indptr = np.zeros(n + 1, np.int64)
    for i, lst in enumerate(pair_lists):
        indptr[i + 1] = indptr[i] + len(lst)
    indices = np.concatenate([np.asarray(l, np.int64) for l in pair_lists]) \
        if indptr[-1] else np.empty(0, np.int64)
    return indptr, indices


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    m = cfg.datasets
    pos_radius = float(m.pos_radius)
    cross_radius = float(m.cross_radius)
    min_dt = float(m.min_dt)

    seqs = load_slices(m.slices_dir, m.night_keyword)

    # global slice table
    n = sum(len(s["xy"]) for s in seqs)
    xy = np.concatenate([s["xy"] for s in seqs]).astype(np.float64)
    ts = np.concatenate([s["ts"] for s in seqs])
    seq_id = np.concatenate([np.full(len(s["xy"]), i, np.int32)
                             for i, s in enumerate(seqs)])
    is_night = np.concatenate([np.full(len(s["xy"]), s["is_night"], bool)
                               for s in seqs])
    frames = np.array([s["frame"] for s in seqs])

    # per-frame-group neighbour search: local coordinates from different
    # frames are numerically comparable but geographically unrelated --
    # never share an index across frames. KD-tree scales to NYC-sized
    # corpora; block-numpy fallback if scipy is unavailable.
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    r_max = max(pos_radius, cross_radius)

    def neighbours(pts):
        if cKDTree is not None:
            return cKDTree(pts).query_ball_point(pts, r_max)
        out = []
        for i0 in range(0, len(pts), 2000):
            blk = pts[i0:i0 + 2000]
            dd = np.linalg.norm(blk[:, None, :] - pts[None, :, :], axis=-1)
            out.extend(list(np.flatnonzero(row <= r_max)) for row in dd)
        return out

    pos_same_lists = [[] for _ in range(n)]
    pos_cross_lists = [[] for _ in range(n)]
    for fr in np.unique(frames):
        ids = np.flatnonzero(frames[seq_id] == fr)
        nbrs = neighbours(xy[ids])
        for a_pos, a in enumerate(ids):
            for b_pos in nbrs[a_pos]:
                b = ids[b_pos]
                if b == a:
                    continue
                dist = np.linalg.norm(xy[a] - xy[b])
                if is_night[a] == is_night[b]:
                    if dist > pos_radius:
                        continue
                    if seq_id[a] == seq_id[b] and \
                            abs(ts[a] - ts[b]) < min_dt:
                        continue        # trivial same-pass neighbour
                    pos_same_lists[a].append(b)
                elif dist <= cross_radius:
                    pos_cross_lists[a].append(b)

    ps_ptr, ps_idx = build_csr(pos_same_lists, n)
    pc_ptr, pc_idx = build_csr(pos_cross_lists, n)

    out = Path(m.index_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, xy=xy.astype(np.float32), ts=ts, seq_id=seq_id,
        is_night=is_night, seq_files=np.array([s["file"] for s in seqs]),
        seq_frames=frames,
        local_index=np.concatenate([np.arange(len(s["xy"])) for s in seqs]),
        pos_same_indptr=ps_ptr, pos_same_indices=ps_idx,
        pos_cross_indptr=pc_ptr, pos_cross_indices=pc_idx,
        neg_floor=float(m.neg_floor), pos_radius=pos_radius,
        cross_radius=cross_radius)

    def stats(ptr, mask=None):
        counts = np.diff(ptr)
        if mask is not None:
            counts = counts[mask]
        return (f"{(counts > 0).sum()}/{len(counts)} anchors with >=1 "
                f"positive (median {int(np.median(counts[counts > 0]))} "
                f"per anchor)" if (counts > 0).any() else "NONE")

    print(f"\n{n} slices total")
    print(f"same-condition positives : {stats(ps_ptr)}")
    print(f"cross-condition positives: {stats(pc_ptr)}")
    print(f"  day anchors with night positive : {stats(pc_ptr, ~is_night)}")
    print(f"  night anchors with day positive : {stats(pc_ptr, is_night)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
