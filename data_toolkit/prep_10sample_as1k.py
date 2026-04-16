#!/usr/bin/env python3
"""
Build a split JSON that makes the 10-sample dataset behave like the 873-train
split of lightgen_1k: each of the 10 unique samples is repeated so the train
split has exactly 873 entries — matching the 1k finetune config row-for-row.

This lets us run the emission_pbr_finetune_1k config unmodified on the
10-sample data (same steps-per-epoch, same sampler behavior, same fp16 / LR /
EMA / grad-clip dynamics) to isolate whether lack of convergence is caused by
the training setup or by the data.

Output: /local-scratch/localhome/dya78/lightgen_test/data_splits_10sample_as1k.json
"""
import json
import os
import pandas as pd

TEST_ROOT   = "/local-scratch/localhome/dya78/lightgen_test"
PARQUET     = f"{TEST_ROOT}/df_10samples.parquet"
OUT_JSON    = f"{TEST_ROOT}/data_splits_10sample_as1k.json"
TRAIN_SIZE  = 873  # matches lightgen_1k train split exactly

df = pd.read_parquet(PARQUET)
n_unique = len(df)
print(f"Parquet: {PARQUET}  rows={n_unique}")

# Build positional indices 0..N-1, repeated up to TRAIN_SIZE entries total.
base = list(range(n_unique))
reps = (TRAIN_SIZE + n_unique - 1) // n_unique
train_indices = (base * reps)[:TRAIN_SIZE]

assert len(train_indices) == TRAIN_SIZE
print(f"Train indices: {len(train_indices)}  (each of {n_unique} unique samples "
      f"appears ~{TRAIN_SIZE // n_unique}-{(TRAIN_SIZE + n_unique - 1) // n_unique} times)")

splits = {
    "train": {"indices": train_indices},
    "val":   {"indices": []},
    "test":  {"indices": []},
    "note": (f"10-sample dataset expanded to {TRAIN_SIZE} train rows so the "
             f"emission_pbr_finetune_1k config can be run unmodified for a "
             f"training-setup isolation test. Each of {n_unique} unique samples "
             f"is repeated ~{TRAIN_SIZE // n_unique}x."),
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(splits, f)
print(f"Wrote {OUT_JSON}")
