"""Run evaluate_brisbane / probe_all against a non-stock pooling.

WHY THIS FILE EXISTS
--------------------
`evaluate_brisbane.py:14` does `from model import EventViTStudent` and
`build_model` constructs it at `:278`, so there is no way to evaluate a
`signed`/`mean` checkpoint through the stock path -- and `model.py` and
`evaluate_brisbane.py` are not to be edited.

`model_gem_signed.install` rebinds that NAME rather than `build_model`, which
matters: `build_model` resolves `EventViTStudent` in `evaluate_brisbane`'s
globals at CALL time, so one rebinding reaches every caller of the function --
`probe_all.py`, and `evaluate_nsavp.py` too, whose
`from evaluate_brisbane import build_model` binds the same function object.

The safety net is the state dict, not this script: `signed`/`mean` register
their pooler as `pool`, so if you forget this wrapper the stock strict
`load_state_dict` RAISES rather than quietly evaluating your weights through
the clamped GeM and printing plausible numbers.

USAGE
-----
`+gem.pooling` and `+gem.entry` are read straight from argv, before Hydra
starts, because the model has to be chosen before `main` is called. They are
left in argv so they also appear in the resolved config.

  rm -rf /kaggle/working/brisbane_cache

  # recall + the sequence-matched summary
  python scripts/evaluate_brisbane_gem.py +gem.pooling=signed \
    datasets=brisbane datasets.root=$BRISBANE \
    phase1_weights=/kaggle/working/ckpt_signed/last_phase1_histogram.pth \
    data.modality=histogram datasets.dt=1.0

  # effective rank, cross-condition d', retention -- the numbers that decide
  python scripts/evaluate_brisbane_gem.py +gem.pooling=signed \
    +gem.entry=probe_all \
    datasets=brisbane datasets.root=$BRISBANE \
    phase1_weights=/kaggle/working/ckpt_signed/last_phase1_histogram.pth \
    data.modality=histogram datasets.dt=1.0

`+gem.entry` accepts `evaluate_brisbane` (default) or `probe_all`.
`+gem.pooling=clamp` is a no-op passthrough, useful only to confirm this
wrapper changes nothing when it should not.

CACHE WARNING: for phase-1 weights the descriptor cache key does NOT include
the checkpoint (`evaluate_brisbane.py:174` sets `tag=""`), so every phase-1
model shares one cache file. Delete `brisbane_cache` between runs or you will
score the previous checkpoint -- and here that would silently compare two
poolings using one set of descriptors.
"""

import importlib
import sys

import evaluate_brisbane as eb
import model_gem_signed as G


_ENTRIES = {"evaluate_brisbane", "probe_all"}


def _argv_opt(key, default):
    prefix = f"+gem.{key}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def main():
    pooling = _argv_opt("pooling", "signed")
    entry = _argv_opt("entry", "evaluate_brisbane")
    if entry not in _ENTRIES:
        raise SystemExit(f"+gem.entry must be one of {sorted(_ENTRIES)}, "
                         f"got {entry!r}")

    # Rebind before importing the entry module: probe_all does
    # `import evaluate_brisbane as eb` and calls eb.build_model, so it picks
    # this up either way, but doing it first keeps the print order readable.
    G.install(eb, pooling)

    mod = eb if entry == "evaluate_brisbane" else importlib.import_module(entry)
    print(f"[gem] entry={entry}")
    mod.main()


if __name__ == "__main__":
    main()
