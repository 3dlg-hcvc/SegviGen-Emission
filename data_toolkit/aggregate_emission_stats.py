#!/usr/bin/env python3
"""Compute emission-latent normalization stats by globbing a latent directory.

No model / GPU needed — just reads each .npz's `feats` and accumulates per-channel
sum / sumsq (memory-safe; never concatenates all feats). Use after a *sharded*
encode (script/vulcan/encode_emission_74k.sh with NSHARDS>1), where each shard
writes a disjoint slice to the shared scratch latent dir and no single job sees
everything. Writes <dir>/normalization_stats.json and prints the YAML block to
paste into the DiT config's emission_normalization.

Usage:
    python data_toolkit/aggregate_emission_stats.py \
        /scratch/dya78/lightgen/lightgen_74k/emission_latents/emission_pbr_vae_ft_74k
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    latent_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(latent_dir, "*.npz")))
    if not files:
        print(f"No .npz files in {latent_dir}", file=sys.stderr)
        sys.exit(1)

    n_tokens = 0
    s = None      # per-channel sum
    ss = None     # per-channel sum of squares
    fmin = np.inf
    fmax = -np.inf
    n_bad = 0
    for i, p in enumerate(files):
        try:
            with np.load(p) as d:
                f = d["feats"].astype(np.float64)
        except Exception as e:  # corrupt / partial npz — skip, count it
            n_bad += 1
            continue
        if f.ndim != 2:
            n_bad += 1
            continue
        if s is None:
            s = f.sum(axis=0)
            ss = (f * f).sum(axis=0)
        else:
            s += f.sum(axis=0)
            ss += (f * f).sum(axis=0)
        n_tokens += f.shape[0]
        fmin = min(fmin, float(f.min()))
        fmax = max(fmax, float(f.max()))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)} files, {n_tokens} tokens", file=sys.stderr)

    if n_tokens == 0:
        print("No valid feats accumulated; aborting.", file=sys.stderr)
        sys.exit(1)

    mean = s / n_tokens
    var = np.maximum(ss / n_tokens - mean * mean, 0.0)
    std = np.sqrt(var)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "total_tokens": int(n_tokens),
        "channels": int(mean.shape[0]),
        "feat_min": fmin,
        "feat_max": fmax,
        "n_samples": len(files) - n_bad,
        "n_bad": n_bad,
    }
    out_path = os.path.join(latent_dir, "normalization_stats.json")
    with open(out_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(
        f"\nWrote {out_path}  (samples={len(files) - n_bad}, bad={n_bad}, "
        f"tokens={n_tokens}, channels={mean.shape[0]})",
        file=sys.stderr,
    )
    print("\n=== Emission normalization stats (paste into DiT config) ===")
    print("    emission_normalization:")
    print(f"      mean: [{', '.join(f'{v:.6f}' for v in mean.tolist())}]")
    print(f"      std: [{', '.join(f'{v:.6f}' for v in std.tolist())}]")


if __name__ == "__main__":
    main()
