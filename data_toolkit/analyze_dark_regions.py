#!/usr/bin/env python3
"""Characterize dark-voxel patterns in emission VAE reconstructions.

For each sample in the 10-sample overfit:
  1. Load GT emission voxel (coords, RGB feats in [-1, 1]).
  2. Encode + decode via the trained VAE (channel-hijack for
     EmissionPbrFinetuneTrainer, two-stream for EmissionVaeTrainer).
  3. Convert to linear brightness = max_channel((feats + 1) / 2) in [0, 1].
  4. Bucket voxels by GT brightness and report:
       - fraction of voxels per bucket
       - mean pred brightness per GT bucket
       - pred L1 per GT bucket
       - how many GT-dark voxels sit on the boundary of a bright region
         (i.e. have at least one 26-neighbour voxel that is GT-bright)
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from vis.vis_utils import (
    load_samples, load_vae_from_handle, read_emission_voxel, resolve_experiment,
)

BUCKET_EDGES = [0.0, 0.01, 0.05, 0.10, 0.30, 1.01]
BUCKET_LABELS = ["=0-.01", ".01-.05", ".05-.10", ".10-.30", ".30-1.0"]
DARK_THR = 0.05  # boundary analysis threshold


def _brightness(feats_norm: torch.Tensor) -> torch.Tensor:
    """[-1, 1] RGB → linear brightness in [0, 1] (max over channels)."""
    lin = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    return lin.max(dim=-1).values


def _pbr_feats_for(path: str) -> torch.Tensor | None:
    if not os.path.exists(path):
        return None
    import o_voxel
    _, pbr_attr = o_voxel.io.read_vxz(path, num_threads=4)
    attrs = ["base_color", "metallic", "roughness", "alpha"]
    return torch.cat([pbr_attr[a].float() / 255.0 * 2.0 - 1.0 for a in attrs], dim=-1)


def _boundary_mask(coords: torch.Tensor, is_dark: torch.Tensor) -> torch.Tensor:
    """Return bool mask over coords: True iff the voxel is dark AND has at
    least one 26-neighbour in the active set that is bright.
    """
    coords_np = coords.cpu().numpy().astype(np.int64)
    dark_np = is_dark.cpu().numpy()
    # Map each coord to row index via a dict of tuples.
    key_to_idx: dict[tuple[int, int, int], int] = {
        (int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(coords_np)
    }
    out = np.zeros(len(coords_np), dtype=bool)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)]
    bright_np = ~dark_np
    for i, c in enumerate(coords_np):
        if not dark_np[i]:
            continue
        for dx, dy, dz in offs:
            j = key_to_idx.get((int(c[0]) + dx, int(c[1]) + dy, int(c[2]) + dz))
            if j is not None and bright_np[j]:
                out[i] = True
                break
    return torch.from_numpy(out)


def _bucket_idx(bright: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(bright, len(BUCKET_LABELS) - 1, dtype=torch.long)
    for b in range(len(BUCKET_LABELS)):
        lo, hi = BUCKET_EDGES[b], BUCKET_EDGES[b + 1]
        mask = (bright >= lo) & (bright < hi)
        out[mask] = b
    return out


def analyse(handle_path: str, dataset: str, data_root: str, device: str, samples: str, label: str) -> None:
    handle = resolve_experiment(label, handle_path)
    records = load_samples(
        data_root=data_root, resolution=handle.resolution, source=dataset,
        select=samples, require=("emission_voxel_path",),
    )
    print(f"Loaded {len(records)} samples from {dataset}")
    print(f"Run: {handle.ckpt_path}")
    decode = load_vae_from_handle(handle, device=device)

    per_sample_rows = []
    global_counts = np.zeros(len(BUCKET_LABELS), dtype=np.int64)
    global_pred_mean = np.zeros(len(BUCKET_LABELS), dtype=np.float64)
    global_l1 = np.zeros(len(BUCKET_LABELS), dtype=np.float64)

    for rec in records:
        gt_coords, gt_feats, res = read_emission_voxel(rec.emission_voxel_path)
        pbr_feats = _pbr_feats_for(rec.pbr_voxel_path) if handle.trainer_name == "EmissionVaeTrainer" else None
        pred_feats = decode(gt_coords, gt_feats, pbr_feats)

        gt_b = _brightness(gt_feats)
        pr_b = _brightness(pred_feats)

        buckets = _bucket_idx(gt_b)
        n = len(gt_b)

        # Per-bucket counts / mean pred / L1.
        rec_row = {"sha": rec.sha[:12], "n": n}
        for b, label in enumerate(BUCKET_LABELS):
            mask = buckets == b
            cnt = int(mask.sum().item())
            rec_row[f"%{label}"] = 100.0 * cnt / n
            rec_row[f"pr_mean_{label}"] = (pr_b[mask].mean().item() if cnt else float("nan"))
            rec_row[f"l1_{label}"] = ((pred_feats[mask] - gt_feats[mask]).abs().mean().item() if cnt else float("nan"))
            global_counts[b] += cnt
            if cnt:
                global_pred_mean[b] += pr_b[mask].sum().item()
                global_l1[b] += (pred_feats[mask] - gt_feats[mask]).abs().mean().item() * cnt

        # Boundary analysis on GT-dark (<DARK_THR) voxels.
        is_dark = gt_b < DARK_THR
        n_dark = int(is_dark.sum().item())
        rec_row["%dark"] = 100.0 * n_dark / n
        if n_dark:
            boundary = _boundary_mask(gt_coords, is_dark)
            n_bd = int(boundary.sum().item())
            rec_row["%bd_of_dark"] = 100.0 * n_bd / n_dark
            rec_row["pr_mean_dark_bd"] = pr_b[boundary].mean().item() if n_bd else float("nan")
            rec_row["pr_mean_dark_int"] = pr_b[is_dark & ~boundary].mean().item() if (n_dark - n_bd) else float("nan")
        else:
            rec_row["%bd_of_dark"] = 0.0
            rec_row["pr_mean_dark_bd"] = float("nan")
            rec_row["pr_mean_dark_int"] = float("nan")
        per_sample_rows.append(rec_row)
        print(f"  {rec.sha[:12]}  N={n:7d}  %dark={rec_row['%dark']:5.1f}  "
              f"%bd_of_dark={rec_row['%bd_of_dark']:4.1f}  "
              f"pr_mean(dark_bd)={rec_row['pr_mean_dark_bd']:.3f} "
              f"pr_mean(dark_int)={rec_row['pr_mean_dark_int']:.3f}")

    # Global aggregate table.
    print()
    print(f"=== Aggregate across {len(records)} samples ===")
    total = global_counts.sum()
    header = "  bucket   |  frac%  | pred_mean | L1(feat)"
    print(header)
    print("-" * len(header))
    for b, label in enumerate(BUCKET_LABELS):
        frac = 100.0 * global_counts[b] / total if total else 0
        pm = global_pred_mean[b] / global_counts[b] if global_counts[b] else float("nan")
        l1 = global_l1[b] / global_counts[b] if global_counts[b] else float("nan")
        print(f"  {label:8s} | {frac:6.2f} |  {pm:7.4f} |  {l1:7.4f}")

    # Write CSV.
    out_csv = f"reports/dark_region_analysis_{label}.csv"
    os.makedirs("reports", exist_ok=True)
    import csv
    keys = list(per_sample_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_sample_rows:
            w.writerow(r)
    print(f"\nWrote per-sample CSV: {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="path to run dir or .ckpt")
    ap.add_argument("--dataset", default="data/lightgen_test/df_10samples.parquet")
    ap.add_argument("--data_root", default="data/lightgen_test")
    ap.add_argument("--samples", default="all")
    ap.add_argument("--label", default="pbr_ft")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    analyse(args.experiment, args.dataset, args.data_root, args.device, args.samples, args.label)


if __name__ == "__main__":
    main()
