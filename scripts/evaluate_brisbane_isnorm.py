"""Evaluate a per-sample-input-norm checkpoint with the unmodified Brisbane
protocol.

Same relationship to evaluate_brisbane.py as train_insnorm.py has to train.py:
the class is swapped, nothing else. GT, sequence matching, moving-frame filter
and CSV output are the imported originals, so numbers stay comparable to every
run already recorded.

    INPUT_NORM=instance uv run python scripts/evaluate_brisbane_insnorm.py \
        datasets=brisbane datasets.root=/kaggle/input/... \
        phase1_weights=/path/to/best_phase1_histogram.pth \
        data.modality=histogram datasets.dt=1.0 hydra.run.dir=/kaggle/working/hy

INPUT_NORM must match the value used for training, or the strict
load_state_dict will fail (which is the intended guard, not a bug).

NOTE ON THE PROTOCOL-MATCHED COLUMN: arXiv 2509.01968 evaluates "without
metric subsampling, thereby preserving natural variations in speed and stop
duration", i.e. it does NOT drop stationary frames. Compare their AR@1 against
the [all] summary rows, not [moving].
"""

import evaluate_brisbane
import model_insnorm

model_insnorm.install(evaluate_brisbane)

if __name__ == "__main__":
    evaluate_brisbane.main()
