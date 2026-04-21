#!/usr/bin/env python3
"""
Build a split JSON for the 10-sample overfit test.

Why repeat the train set?
-------------------------
The TrellisDataModule train DataLoader uses drop_last=True (see
trellis2/trainers/data.py), so len(loader) = len(dataset) // batch_size. With
only 10 unique samples and any per-GPU batch size > 10 (e.g. BS=12 to keep the
GPU busy for scaling tests), drop_last would discard everything and produce
zero training batches per epoch — training hangs.

So we oversample the 10 unique items to TRAIN_SIZE entries. This costs nothing
at I/O time (the dataset is just an index list and the underlying .vxz files
are cached/mmap'd). It is mathematically equivalent to a single epoch of 10
samples replayed TRAIN_SIZE/10 times, only the step boundary changes.

Val keeps the 10 unique samples once: val_dataloader uses drop_last=False so
no oversampling is needed there. With train==val (same 10 samples) we get a
clean "did the model memorize?" signal — val/loss should track train/loss and
approach zero as overfitting succeeds.

Paths are configurable via env vars so the script works both locally and on
the cluster (mount points differ: /local-scratch/localhome/dya78 vs
/localscratch/dya78). By default we auto-detect whichever mount exists:

  Local workstation: /local-scratch/localhome/dya78/lightgen_test
  Cluster (venus):   /localscratch/dya78/lightgen_test

Override explicitly:
    TEST_ROOT=/path/to/lightgen_test python data_toolkit/prep_10sample_as1k.py
    TRAIN_SIZE=240 python data_toolkit/prep_10sample_as1k.py
"""
import json
import os
import pandas as pd

_DEFAULT_TEST_ROOTS = [
    "/local-scratch/localhome/dya78/lightgen_test",
    "/localscratch/dya78/lightgen_test",
]


def _detect_test_root() -> str:
    env = os.environ.get("TEST_ROOT")
    if env:
        return env
    for cand in _DEFAULT_TEST_ROOTS:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "Could not find lightgen_test data root. Tried: "
        + ", ".join(_DEFAULT_TEST_ROOTS)
        + ". Set TEST_ROOT=/path/to/lightgen_test to override."
    )


TEST_ROOT   = _detect_test_root()
PARQUET     = f"{TEST_ROOT}/df_10samples.parquet"
OUT_JSON    = f"{TEST_ROOT}/data_splits_10sample_as1k.json"
# 120 = 10 batches/epoch at BS=12 per GPU (the current scaling target).
TRAIN_SIZE  = int(os.environ.get("TRAIN_SIZE", 120))
print(f"TEST_ROOT: {TEST_ROOT}")

df = pd.read_parquet(PARQUET)
n_unique = len(df)
print(f"Parquet: {PARQUET}  rows={n_unique}")

base = list(range(n_unique))
reps = (TRAIN_SIZE + n_unique - 1) // n_unique
train_indices = (base * reps)[:TRAIN_SIZE]
val_indices   = list(range(n_unique))

print(f"Train indices: {len(train_indices)}  "
      f"(each of {n_unique} unique samples appears ~{TRAIN_SIZE // n_unique}x)")
print(f"Val indices:   {len(val_indices)}  "
      f"(same {n_unique} unique samples once — overfit memorization check)")

splits = {
    "train": {"indices": train_indices},
    "val":   {"indices": val_indices},
    "test":  {"indices": []},
    "note": (f"10-sample overfit test: train oversampled to {TRAIN_SIZE} rows "
             f"so per-GPU batch size up to ~{TRAIN_SIZE} works cleanly with "
             f"drop_last=True. Val is the {n_unique} unique samples once — "
             f"val/loss should track train/loss and approach 0 as the model "
             f"memorizes the data."),
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(splits, f)
print(f"Wrote {OUT_JSON}")
