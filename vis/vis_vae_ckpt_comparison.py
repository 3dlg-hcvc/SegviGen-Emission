#!/usr/bin/env python3
"""
Compare emission VAE reconstructions across specific checkpoints:
  - 500 epochs  (step  1000) : outputs/emission_vae_500ep/ckpts/last.ckpt
  - 2000 epochs (step  4000) : outputs/emission_vae_10sample_10kep/ckpts/stepstep=0004000.ckpt
  - 5000 epochs (step 10000) : outputs/emission_vae_10sample_10kep/ckpts/stepstep=0010000.ckpt
  - 10000 epochs (step 20000): outputs/emission_vae_10sample_10kep/ckpts/stepstep=0020000.ckpt

Layout: 10 rows (one per sample) x N*2+2 cols (GT XY/XZ + Rec XY/XZ per ckpt)

Usage:
    conda run -n trellis2 python data_toolkit/vis_vae_ckpt_comparison.py
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
import matplotlib.gridspec as gridspec

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.modules import sparse as sp
import trellis2.models as trellis_models
from trellis2.trainers.builders import build_models

# ── Checkpoints to compare ────────────────────────────────────────────────────
BASE = "/localhome/dya78/code/TRELLIS.2-lightning/outputs"

CKPTS = [
    {
        "label": "500 ep\n(step 1k)",
        "run_dir": f"{BASE}/emission_vae_500ep",
        "ckpt":    "last.ckpt",
    },
    {
        "label": "2000 ep\n(step 4k)",
        "run_dir": f"{BASE}/emission_vae_10sample_10kep",
        "ckpt":    "stepstep=0004000.ckpt",
    },
    {
        "label": "5000 ep\n(step 10k)",
        "run_dir": f"{BASE}/emission_vae_10sample_10kep",
        "ckpt":    "stepstep=0010000.ckpt",
    },
    {
        "label": "10000 ep\n(step 20k)",
        "run_dir": f"{BASE}/emission_vae_10sample_10kep",
        "ckpt":    "stepstep=0020000.ckpt",
    },
]

DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
OUT_PNG   = "/localhome/dya78/code/TRELLIS.2-lightning/visualizations/emission_vae_ckpt_comparison.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_run(run_dir, ckpt_name):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    models   = build_models(cfg["models"])
    encoder  = models["encoder"]
    decoder  = models["decoder"]
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)
    print(f"  {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    encoder.load_state_dict({k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")})
    decoder.load_state_dict({k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")})
    encoder.to(device).eval()
    decoder.to(device).eval()
    pbr_name = cfg["trainer"]["args"].get("pretrained_pbr_encoder")
    pbr_enc  = trellis_models.from_pretrained(pbr_name).eval().float().to(device) if pbr_name else None
    return encoder, decoder, pbr_enc


def max_proj(sparse_t, res=256):
    c   = sparse_t.coords[:, 1:].long()
    mag = (sparse_t.feats * 0.5 + 0.5).norm(dim=-1)
    xy  = torch.zeros(res, res)
    xz  = torch.zeros(res, res)
    for i in range(c.shape[0]):
        xi, yi, zi = c[i,0].item(), c[i,1].item(), c[i,2].item()
        v = mag[i].item()
        if v > xy[yi, xi]: xy[yi, xi] = v
        if v > xz[zi, xi]: xz[zi, xi] = v
    return xy.numpy(), xz.numpy()


@torch.no_grad()
def reconstruct(encoder, decoder, pbr_encoder, coords, emit_feats, pbr_feats):
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined  = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0        = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    x0_emit   = x0.replace(emit_feats)
    x0_gpu    = x0.to(device)
    z_emit    = encoder(x0_emit.to(device))
    if pbr_encoder is not None:
        z_pbr  = pbr_encoder(x0_gpu.replace(x0_gpu.feats[:, 3:]))
        z_comb = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
    else:
        z_comb = z_emit
    return decoder(z_comb).to("cpu")


# ── Load all checkpoints ──────────────────────────────────────────────────────
print("Loading checkpoints...")
loaded = []
pbr_enc_shared = None  # reuse the frozen PBR encoder across all runs
for info in CKPTS:
    print(f"\n{info['label'].replace(chr(10), ' ')}:")
    enc, dec, pbr_enc = load_run(info["run_dir"], info["ckpt"])
    if pbr_enc_shared is None and pbr_enc is not None:
        pbr_enc_shared = pbr_enc
    loaded.append(dict(label=info["label"], encoder=enc, decoder=dec))

# ── Discover samples ──────────────────────────────────────────────────────────
df      = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
SAMPLES = list(df.index)
print(f"\nFound {len(SAMPLES)} samples")

# ── Reconstruct ───────────────────────────────────────────────────────────────
results = []
for i, sha in enumerate(SAMPLES):
    print(f"  [{i+1}/{len(SAMPLES)}] {sha[:16]}...", end=" ", flush=True)

    coords, emit_attr = o_voxel.io.read_vxz(f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
    emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0

    _, pbr_attr = o_voxel.io.read_vxz(f"{DATA_ROOT}/pbr_voxels_256/{sha}.vxz", num_threads=4)
    pbr_feats = torch.cat([
        pbr_attr["base_color"].float() / 255.0 * 2.0 - 1.0,
        pbr_attr["metallic"].float()   / 255.0 * 2.0 - 1.0,
        pbr_attr["roughness"].float()  / 255.0 * 2.0 - 1.0,
        pbr_attr["alpha"].float()      / 255.0 * 2.0 - 1.0,
    ], dim=-1)

    # GT
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined  = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0        = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    gt_sparse = x0.replace(emit_feats)
    xy_gt, xz_gt = max_proj(gt_sparse)

    row = dict(sha=sha, n_voxels=coords.shape[0], xy_gt=xy_gt, xz_gt=xz_gt, recs=[])
    l1_parts = []
    for run in loaded:
        y    = reconstruct(run["encoder"], run["decoder"], pbr_enc_shared, coords, emit_feats, pbr_feats)
        l1   = (gt_sparse.feats - y.feats).abs().mean().item()
        xy_r, xz_r = max_proj(y)
        row["recs"].append(dict(l1=l1, xy=xy_r, xz=xz_r))
        l1_parts.append(f"L1={l1:.4f}")
        torch.cuda.empty_cache()

    results.append(row)
    print("  ".join(l1_parts))

# ── Plot ──────────────────────────────────────────────────────────────────────
n_samples = len(results)
n_ckpts   = len(loaded)
# Columns: GT XY | GT XZ | [Rec XY | Rec XZ] x n_ckpts
ncols = 2 + 2 * n_ckpts
nrows = n_samples

fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.8 * nrows))

# Title
mean_l1s = [np.mean([r["recs"][ci]["l1"] for r in results]) for ci in range(n_ckpts)]
title = "Emission VAE — Checkpoint Comparison (10 samples)\n"
title += "  |  ".join(f"{loaded[ci]['label'].replace(chr(10), ' ')}: mean L1={mean_l1s[ci]:.4f}" for ci in range(n_ckpts))
fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

# Column headers (row 0 only)
col_labels = ["GT XY", "GT XZ"]
for run in loaded:
    lbl = run["label"].replace("\n", " ")
    col_labels += [f"{lbl} XY", f"{lbl} XZ"]
for c, lbl in enumerate(col_labels):
    axes[0, c].set_title(lbl, fontsize=8, fontweight="bold")

# Fill in each row
for i, r in enumerate(results):
    all_imgs = [r["xy_gt"], r["xz_gt"]] + [img for rec in r["recs"] for img in (rec["xy"], rec["xz"])]
    vmax = max(img.max() for img in all_imgs)
    vmax = max(vmax, 1e-3)

    for c, img in enumerate(all_imgs):
        axes[i, c].imshow(img, cmap="hot", vmin=0, vmax=vmax, origin="lower")

    l1_str = " / ".join(f"{rec['l1']:.4f}" for rec in r["recs"])
    axes[i, 0].set_ylabel(
        f"{r['sha'][:8]}...\n{r['n_voxels']:,}v\nL1: {l1_str}",
        fontsize=6, rotation=0, ha="right", va="center", labelpad=60,
    )

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
plt.close()
