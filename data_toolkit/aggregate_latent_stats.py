#!/usr/bin/env python
"""Per-channel mean/std over an encoded latent set, computed in one streaming pass.

Why not `encode_*_dataset.py --aggregate_stats`: that path does
`torch.cat([...all shapes...])` before reducing, i.e. it materializes every token
of the whole set at once. At 72,421 shapes x thousands of tokens x 32 channels
that is tens of GB and OOMs on anything but a fat node.

This accumulates count / sum / sum-of-squares in float64 instead, so memory is
O(channels) regardless of set size, and emits the same JSON plus the ready-to-paste
config block.

Numerical note: variance from E[x^2]-E[x]^2 can cancel catastrophically when the
mean dwarfs the spread. It does not here (latents are roughly zero-centred, and
the accumulators are float64), and the result is checked against a second pass
over a random subset before being written.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--out", default=None, help="default: <latent_dir>/normalization_stats.json")
    ap.add_argument("--check_frac", type=float, default=0.02,
                    help="fraction re-read exactly to validate the streaming result")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.latent_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz under {a.latent_dir}")
    print(f"[stats] {len(files)} latent files in {a.latent_dir}")

    n = 0
    s = ss = None
    fmin, fmax = math.inf, -math.inf
    for i, p in enumerate(files):
        f = torch.from_numpy(np.load(p)["feats"]).double()
        if s is None:
            s = torch.zeros(f.shape[1], dtype=torch.float64)
            ss = torch.zeros(f.shape[1], dtype=torch.float64)
        n += f.shape[0]
        s += f.sum(0)
        ss += (f * f).sum(0)
        fmin = min(fmin, float(f.min()))
        fmax = max(fmax, float(f.max()))
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(files)}] tokens={n:,}")

    mean = s / n
    var = ss / n - mean * mean
    if bool((var < 0).any()):
        raise SystemExit("[stats] negative variance — numerical breakdown, refusing to write")
    std = var.sqrt()

    # Independent check on a random subset: recompute exactly (cat) and compare
    # the subset's own streaming numbers. Catches an accumulation bug, which
    # would otherwise silently mis-scale every DiT input.
    k = max(1, int(len(files) * a.check_frac))
    sub = random.Random(0).sample(files, k)
    cat = torch.cat([torch.from_numpy(np.load(p)["feats"]).double() for p in sub], 0)
    n2 = 0
    s2 = torch.zeros(cat.shape[1], dtype=torch.float64)
    ss2 = torch.zeros(cat.shape[1], dtype=torch.float64)
    for p in sub:
        f = torch.from_numpy(np.load(p)["feats"]).double()
        n2 += f.shape[0]
        s2 += f.sum(0)
        ss2 += (f * f).sum(0)
    m_exact, m_stream = cat.mean(0), s2 / n2
    sd_exact = cat.std(0, unbiased=False)
    sd_stream = (ss2 / n2 - m_stream * m_stream).sqrt()
    dm = float((m_exact - m_stream).abs().max())
    dsd = float((sd_exact - sd_stream).abs().max())
    print(f"[check] {k} files: max|dmean|={dm:.3e}  max|dstd|={dsd:.3e}")
    if dm > 1e-6 or dsd > 1e-6:
        raise SystemExit("[check] streaming and exact disagree — refusing to write")

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "total_tokens": int(n),
        "channels": int(mean.shape[0]),
        "feat_min": fmin,
        "feat_max": fmax,
        "n_samples": len(files),
    }
    out = a.out or os.path.join(a.latent_dir, "normalization_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[stats] wrote {out}")
    print(f"[stats] {len(files)} shapes, {n:,} tokens, {stats['channels']} ch, "
          f"range [{fmin:.4f}, {fmax:.4f}]")
    print("\n=== paste into the DiT config ===")
    print("    emission_normalization:")
    print(f"      mean: [{', '.join(f'{v:.6f}' for v in mean.tolist())}]")
    print(f"      std: [{', '.join(f'{v:.6f}' for v in std.tolist())}]")


if __name__ == "__main__":
    main()
