#!/usr/bin/env python3
"""
Compare emission VAE reconstructions at 500, 2000, and 5000 epochs.

Shows GT vs reconstructions at each epoch count
for all training samples, using max-luminance projections (XY, XZ).

Usage:
    conda run -n trellis2 python data_toolkit/compare_emission_vae_epochs.py
"""
import sys, os, types
sys.path.insert(0, "/localhome/dya78/code/TRELLIS.2-lightning")
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.modules import sparse as sp
import trellis2.models as trellis_models
from trellis2.trainers.builders import build_models

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
OUT_PNG   = "/localhome/dya78/code/TRELLIS.2-lightning/visualizations/emission_vae_epoch_comparison.png"

RUNS = {
    "500ep": {
        "run_dir": "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_500ep",
        "ckpt": None,  # auto-detect last.ckpt
        "label": "500ep",
    },
    "2000ep": {
        "run_dir": "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_10kep",
        "ckpt": "stepstep=0004000.ckpt",
        "label": "2000ep",
    },
    "5000ep": {
        "run_dir": "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_10kep",
        "ckpt": "stepstep=0010000.ckpt",
        "label": "5000ep",
    },
}

# ── Discover samples ─────────────────────────────────────────────────────────
df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
SAMPLES = list(df.index)
print(f"Found {len(SAMPLES)} samples")

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models_from_run(run_dir, ckpt_name=None):
    """Load encoder, decoder, and PBR encoder from a training run directory."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    models = build_models(cfg["models"])
    encoder, decoder = models["encoder"], models["decoder"]

    if ckpt_name is None:
        ckpt_path = os.path.join(run_dir, "ckpts", "last.ckpt")
    else:
        ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    enc_state = {k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")}
    dec_state = {k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.to(device).eval()
    decoder.to(device).eval()

    pbr_enc_name = cfg["trainer"]["args"].get("pretrained_pbr_encoder")
    pbr_encoder = trellis_models.from_pretrained(pbr_enc_name).eval().float().to(device) if pbr_enc_name else None

    step = ckpt.get("global_step", "?")
    return encoder, decoder, pbr_encoder, step


def max_proj(sparse_t, res=256):
    """Max-luminance projections along XY and XZ planes."""
    c   = sparse_t.coords[:, 1:].long()
    mag = (sparse_t.feats * 0.5 + 0.5).norm(dim=-1)
    xy = torch.zeros(res, res)
    xz = torch.zeros(res, res)
    for i in range(c.shape[0]):
        xi, yi, zi = c[i, 0].item(), c[i, 1].item(), c[i, 2].item()
        v = mag[i].item()
        if v > xy[yi, xi]: xy[yi, xi] = v
        if v > xz[zi, xi]: xz[zi, xi] = v
    return xy.numpy(), xz.numpy()


@torch.no_grad()
def reconstruct(encoder, decoder, pbr_encoder, coords, emit_feats, pbr_feats):
    """Run a single sample through encoder → decoder."""
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0 = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    x0_emit = x0.replace(emit_feats)

    x0_gpu = x0.to(device)
    z_emit = encoder(x0_emit.to(device))
    if pbr_encoder is not None:
        x_pbr = x0_gpu.replace(x0_gpu.feats[:, 3:])
        z_pbr = pbr_encoder(x_pbr)
        z_combined = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
    else:
        z_combined = z_emit
    y = decoder(z_combined).to("cpu")
    return x0_emit.to("cpu"), y


# ── Load both models ─────────────────────────────────────────────────────────
loaded_runs = {}
for tag, info in RUNS.items():
    print(f"\nLoading {tag} ({info['label']}):")
    enc, dec, pbr_enc, step = load_models_from_run(info["run_dir"], info["ckpt"])
    loaded_runs[tag] = dict(encoder=enc, decoder=dec, pbr_encoder=pbr_enc, step=step, label=info["label"])

# ── Reconstruct all samples with both models ─────────────────────────────────
results = []
for i, sha in enumerate(SAMPLES):
    print(f"  [{i+1}/{len(SAMPLES)}] {sha[:16]}...", end=" ", flush=True)

    coords, emit_attr = o_voxel.io.read_vxz(
        f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4
    )
    emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0

    _, pbr_attr = o_voxel.io.read_vxz(
        f"{DATA_ROOT}/pbr_voxels_256/{sha}.vxz", num_threads=4
    )
    pbr_feats = torch.cat([
        pbr_attr["base_color"].float() / 255.0 * 2.0 - 1.0,
        pbr_attr["metallic"].float()   / 255.0 * 2.0 - 1.0,
        pbr_attr["roughness"].float()  / 255.0 * 2.0 - 1.0,
        pbr_attr["alpha"].float()      / 255.0 * 2.0 - 1.0,
    ], dim=-1)

    row = dict(sha=sha, n_voxels=coords.shape[0])

    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0 = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    gt_emit = x0.replace(emit_feats)
    row["xy_gt"], row["xz_gt"] = max_proj(gt_emit)

    for tag in RUNS:
        run = loaded_runs[tag]
        _, y = reconstruct(run["encoder"], run["decoder"], run["pbr_encoder"],
                           coords, emit_feats, pbr_feats)
        l1 = (gt_emit.feats - y.feats).abs().mean().item()
        xy_rec, xz_rec = max_proj(y)
        row[f"l1_{tag}"] = l1
        row[f"xy_{tag}"] = xy_rec
        row[f"xz_{tag}"] = xz_rec

    results.append(row)
    l1_str = " | ".join(f"{tag}: L1={row[f'l1_{tag}']:.4f}" for tag in RUNS)
    print(f"{l1_str}  voxels={row['n_voxels']:,}")
    torch.cuda.empty_cache()

# ── Plot: 10 rows × 6 cols (GT XY/XZ, 500ep XY/XZ, 2000ep XY/XZ) ───────────
n = len(results)
tags = list(RUNS.keys())
ncols = 2 + 2 * len(tags)  # GT(XY,XZ) + per-run(XY,XZ)
fig, axes = plt.subplots(n, ncols, figsize=(3 * ncols, 3 * n))

mean_l1 = {tag: np.mean([r[f"l1_{tag}"] for r in results]) for tag in tags}
title_parts = " vs ".join(f"{loaded_runs[t]['label']} (mean L1={mean_l1[t]:.4f})" for t in tags)
fig.suptitle(
    f"Emission VAE Epoch Comparison — {n} samples\n{title_parts}",
    fontsize=13, fontweight="bold", y=1.0,
)

col_titles = ["GT XY", "GT XZ"]
for tag in tags:
    col_titles += [f"{loaded_runs[tag]['label']} XY", f"{loaded_runs[tag]['label']} XZ"]
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=9, fontweight="bold")

for i, r in enumerate(results):
    imgs = [r["xy_gt"], r["xz_gt"]]
    for tag in tags:
        imgs += [r[f"xy_{tag}"], r[f"xz_{tag}"]]

    vmax = max(img.max() for img in imgs)
    vmax = max(vmax, 1e-3)
    for j, img in enumerate(imgs):
        axes[i, j].imshow(img, cmap="hot", vmin=0, vmax=vmax, origin="lower")

    l1_label = " / ".join(f"{r[f'l1_{tag}']:.4f}" for tag in tags)
    axes[i, 0].set_ylabel(
        f"{r['sha'][:8]}...\nL1: {l1_label}\n{r['n_voxels']:,}v",
        fontsize=7, rotation=0, ha="right", va="center",
    )

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
plt.close()
