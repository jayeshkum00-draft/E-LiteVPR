"""
Localisation probe: is the night failure in the REPRESENTATION or in the MODEL?

Every phase-1 student we have trained scores ~68% day-day at L=1 and ~1.5%
day-night at L=1 -- the level recorded for LENS on this dataset, against 66%
for the ensemble. A 40x gap is not a corpus or a preprocessing-detail problem,
and it cannot be attributed to the student until we know whether a strong
generic extractor does any better on the IDENTICAL frames.

So this runs DINOv3-Large -- the very teacher phase 1 distils from -- directly
on the Brisbane event representations, with the same GT, same protocol, same
sequence matching. No training.

  teacher ~= student (both near chance)  -> the representation does not carry
      matchable night information. Fix the representation (dynamic range,
      window, channel encoding). More data and more training cannot help.
  teacher >> student                     -> the information IS there and phase 1
      is destroying it. Fix the training. Corpus work is beside the point.

Read it asymmetrically: event frames are out of distribution for DINOv3, so a
LOW teacher score is only a lower bound on extractable information. A HIGH
score is conclusive.

Nothing in evaluate_brisbane.py is modified -- its process_traverse, GT and
sequence matching are imported, so the numbers stay comparable to every run
already recorded. Descriptors cache under model_tag="teacher", so student
caches are untouched.

Also reports the CHANCE baseline for each pair, which none of our reporting has
ever included. Without it "1.8%" is uninterpretable.

Run (Colab/Kaggle):

    python scripts/probe_teacher_brisbane.py \
        datasets=brisbane datasets.root=/content/Brisbane_6_set \
        model=teacher_model data.modality=histogram datasets.dt=1.0 \
        hydra.run.dir=/content/hy

Note `model=teacher_model`: the probe needs teacher_model.yaml's img_hw,
expected_patches, imagenet_mean/std. phase1_weights is NOT used and may be
omitted. To also print a student column for the same frames in one table, add
    +probe.student_weights=/content/best_phase1_histogram_clean.pth
"""

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from evaluate_brisbane import (DAY_DAY, build_model, process_traverse,
                               recall_at_1)
from model import GeM
from sequence_matching import apply_sequence_matching


class TeacherEval(torch.nn.Module):
    """DINOv3-L on event frames, descriptor = GeM(p=3) over patch tokens --
    byte-for-byte the `teacher_global` that train.py:92 distils into, so the
    aggregation side is apples-to-apples against a phase-1 student.

    Input normalisation is the teacher's own (feature_extractor.py:79-81), NOT
    the student's `input_norm` BatchNorm2d: that BN is trained, so an untrained
    copy would be meaningless. The teacher expects [0, 1] images, so the net
    channel (which lives in [-1, 1]) is mapped with (x + 1) / 2; channels 0-1
    are already [0, 1] and pass through unchanged.
    """

    def __init__(self, cfg, device):
        super().__init__()
        from feature_extractor import load_teacher

        self.model, self.n_prefix = load_teacher(device, cfg)
        self.gem = GeM()
        self.expected_patches = int(cfg.model.expected_patches)
        self.register_buffer(
            "mean", torch.tensor(list(cfg.model.imagenet_mean)).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(list(cfg.model.imagenet_std)).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def forward(self, x):
        x = x.clone()
        x[:, 2] = (x[:, 2] + 1.0) / 2.0        # net channel [-1,1] -> [0,1]
        x = (x - self.mean) / self.std

        patches = self.model(pixel_values=x).last_hidden_state[:, self.n_prefix:, :]
        if patches.shape[1] != self.expected_patches:
            raise ValueError(
                f"Teacher returned {patches.shape[1]} patches, expected "
                f"{self.expected_patches}. Check model.img_hw matches the "
                f"representation size (both should be 384).")
        return self.gem(patches)


def chance_at_1(q_xy, r_xy, threshold):
    """Expected R@1 of a uniform-random predictor under this GT: mean over
    queries of (positives within threshold) / (references). This is the floor
    every reported number has to clear before it means anything."""
    d = torch.cdist(q_xy, r_xy)
    n_pos = (d <= threshold).sum(dim=1).float()
    return 100.0 * (n_pos / r_xy.shape[0]).mean().item()


def evaluate(tag, model, cfg, device, threshold, ref_name, queries, min_speed):
    """Same protocol as evaluate_brisbane.main, moving-frames variant only
    (the headline), so the rows line up with the recorded results."""
    desc, xy = {}, {}
    for name in cfg.datasets.traverses:
        print(f"[{tag}] processing {name}")
        desc[name], xy[name] = process_traverse(name, cfg, model, device, tag)

    step = 1.0 / float(cfg.datasets.sample_hz)

    def moving_mask(pos):
        if len(pos) < 2:
            return torch.ones(len(pos), dtype=torch.bool)
        v = torch.linalg.norm(pos[1:] - pos[:-1], dim=1) / step
        return torch.cat([v, v[-1:]]) >= min_speed

    keep = {n: moving_mask(xy[n]) for n in cfg.datasets.traverses}
    r_desc = desc[ref_name][keep[ref_name]].to(device)
    r_xy = xy[ref_name][keep[ref_name]]

    out = {}
    for q_name in queries:
        qk = keep[q_name]
        q_xy = xy[q_name][qk]
        dist = -torch.cdist(desc[q_name][qk].to(device), r_desc)
        for length in cfg.datasets.seq_lengths:
            dist_L = dist if length == 1 else apply_sequence_matching(dist, length)
            r1, _, _, _ = recall_at_1(dist_L.argmax(dim=1).cpu(), q_xy, r_xy,
                                      threshold)
            out[(q_name, length)] = r1
        out[(q_name, "chance")] = chance_at_1(q_xy, r_xy, threshold)

    del r_desc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.device)
    threshold = float(cfg.datasets.gt_threshold_m)
    ref_name = cfg.datasets.reference
    queries = [n for n in cfg.datasets.traverses if n != ref_name]
    min_speed = float(cfg.datasets.get("min_speed_ms", 1.0))
    lengths = list(cfg.datasets.seq_lengths)

    if int(cfg.model.img_hw[0]) != 384:
        raise SystemExit(
            f"model.img_hw={list(cfg.model.img_hw)} but the teacher cache and "
            f"representation are 384x384. Pass model=teacher_model.")

    results = {}
    teacher = TeacherEval(cfg, device)
    results["teacher"] = evaluate("teacher", teacher, cfg, device, threshold,
                                  ref_name, queries, min_speed)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sw = OmegaConf.select(cfg, "probe.student_weights")
    if sw:
        # build_model reads cfg.phase1_weights; set it without touching the file
        cfg.phase1_weights = str(sw)
        student, tag = build_model(cfg, device)
        results[f"student:{Path(str(sw)).stem}"] = evaluate(
            tag, student, cfg, device, threshold, ref_name, queries, min_speed)

    print(f"\n{'=' * 78}\nMOVING FRAMES, R@1 %  (ref={ref_name}, "
          f"{threshold:.0f} m, dt={cfg.datasets.dt}s)\n")
    hdr = f"  {'query':<10} {'cond':<10} {'chance':>7}" + \
          "".join(f"  {'L=' + str(L):>7}" for L in lengths)
    for who, res in results.items():
        print(f"  --- {who} ---")
        print(hdr)
        for q in queries:
            cond = "day-day" if q in DAY_DAY else "day-night"
            row = f"  {q:<10} {cond:<10} {res[(q, 'chance')]:>7.2f}"
            row += "".join(f"  {res[(q, L)]:>7.2f}" for L in lengths)
            print(row)
        night = [res[(q, L)] for q in queries if q not in DAY_DAY
                 for L in lengths if L > 1]
        day = [res[(q, L)] for q in queries if q in DAY_DAY
               for L in lengths if L > 1]
        print(f"    mean(L>1)  day-day {np.mean(day):.2f}   "
              f"day-night {np.mean(night):.2f}\n")

    print("  Interpretation: compare the day-night L=1 column against chance.\n"
          "  If the teacher is also at chance there, the representation is the\n"
          "  bottleneck and no amount of data or training fixes it. If the\n"
          "  teacher is well above it, phase 1 is losing information that the\n"
          "  frames already contain.\n")


if __name__ == "__main__":
    main()
