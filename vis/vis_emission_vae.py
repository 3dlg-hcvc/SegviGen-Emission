#!/usr/bin/env python3
"""
Visualize the emission VAE overfit for ALL training samples.
Shows GT vs reconstructed max-projections (XY, XZ) side by side.

Option B architecture: emission encoder + frozen PBR encoder → concat latents → decoder.
"""
import sys, os, types, glob, csv
sys.path.insert(0, "/localhome/dya78/code/TRELLIS.2-lightning")
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.modules import sparse as sp

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
RUN_DIR   = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit"
OUT       = "/localhome/dya78/code/TRELLIS.2-lightning/visualizations/emission_vae_10sample_vis.png"

# ── Discover all samples ─────────────────────────────────────────────────────
df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
SAMPLES = list(df.index)
print(f"Found {len(SAMPLES)} samples")

# ── Load models ──────────────────────────────────────────────────────────────
import yaml
import trellis2.models as trellis_models
from trellis2.trainers.builders import build_models

cfg_path = f"{RUN_DIR}/config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

models = build_models(cfg["models"])
encoder = models["encoder"]
decoder = models["decoder"]

CKPT = f"{RUN_DIR}/ckpts/last.ckpt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ckpt["state_dict"]

enc_state = {k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")}
dec_state = {k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")}
encoder.load_state_dict(enc_state)
decoder.load_state_dict(dec_state)

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device).eval()
decoder.to(device).eval()

pbr_enc_name = cfg["trainer"]["args"].get("pretrained_pbr_encoder")
pbr_encoder = trellis_models.from_pretrained(pbr_enc_name).eval().float().to(device) if pbr_enc_name else None
print(f"PBR encoder: {'loaded' if pbr_encoder else 'none'}")

# ── Read step count ──────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(f"{RUN_DIR}/logs/version_*/metrics.csv"))
total_steps = "?"
if csv_files:
    with open(csv_files[-1]) as f:
        for row in csv.DictReader(f):
            total_steps = row.get("step", total_steps)

# ── max-projection helper ─────────────────────────────────────────────────────
def max_proj(sparse_t, res=256):
    """Return max-luminance projections along XY, XZ, YZ planes (vectorized)."""
    c   = sparse_t.coords[:, 1:].long().cpu()   # (N, 3) — x, y, z
    mag = (sparse_t.feats.cpu() * 0.5 + 0.5).norm(dim=-1)  # (N,)
    xi, yi, zi = c[:, 0], c[:, 1], c[:, 2]

    xy = torch.zeros(res, res)
    xy.index_put_((yi, xi), mag, accumulate=False)
    # use scatter to take max properly
    xy_flat = torch.zeros(res * res)
    xy_flat.scatter_reduce_(0, (yi * res + xi), mag, reduce="amax", include_self=True)
    xy = xy_flat.view(res, res)

    xz_flat = torch.zeros(res * res)
    xz_flat.scatter_reduce_(0, (zi * res + xi), mag, reduce="amax", include_self=True)
    xz = xz_flat.view(res, res)

    yz_flat = torch.zeros(res * res)
    yz_flat.scatter_reduce_(0, (zi * res + yi), mag, reduce="amax", include_self=True)
    yz = yz_flat.view(res, res)

    return xy.numpy(), xz.numpy(), yz.numpy()

# ── Reconstruct all samples ──────────────────────────────────────────────────
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

    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0 = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    x0_emit = x0.replace(emit_feats)

    with torch.no_grad():
        x0_gpu = x0.to(device)
        z_emit = encoder(x0_emit.to(device))
        if pbr_encoder is not None:
            x_pbr = x0_gpu.replace(x0_gpu.feats[:, 3:])
            z_pbr = pbr_encoder(x_pbr)
            z_combined = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
        else:
            z_combined = z_emit
        y = decoder(z_combined).to("cpu")

    x0_emit = x0_emit.to("cpu")
    l1 = (x0_emit.feats - y.feats).abs().mean().item()
    xy_gt, xz_gt, yz_gt = max_proj(x0_emit)
    xy_rec, xz_rec, yz_rec = max_proj(y)
    results.append(dict(sha=sha, l1=l1,
                        xy_gt=xy_gt, xz_gt=xz_gt, yz_gt=yz_gt,
                        xy_rec=xy_rec, xz_rec=xz_rec, yz_rec=yz_rec,
                        n_voxels=x0_emit.feats.shape[0]))
    print(f"L1={l1:.4f}  voxels={x0_emit.feats.shape[0]:,}")
    torch.cuda.empty_cache()

# ── Plot: 10 rows (one per sample) × 6 cols (GT/Rec for XY, XZ, YZ) ────────
n = len(results)
fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
mean_l1 = np.mean([r["l1"] for r in results])
fig.suptitle(
    f"Emission VAE (Option B) Overfit — All {n} samples\n"
    f"{total_steps} steps, lr=0.001   mean L1 = {mean_l1:.4f}",
    fontsize=14, fontweight="bold", y=1.0,
)

col_titles = ["GT  XY", "Rec XY", "GT  XZ", "Rec XZ", "GT  YZ", "Rec YZ"]
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=10, fontweight="bold")

for i, r in enumerate(results):
    pairs = [
        (r["xy_gt"], r["xy_rec"]),
        (r["xz_gt"], r["xz_rec"]),
        (r["yz_gt"], r["yz_rec"]),
    ]
    for j, (gt, rec) in enumerate(pairs):
        vmax = max(gt.max(), rec.max(), 1e-3)
        axes[i, 2*j].imshow(gt, cmap="hot", vmin=0, vmax=vmax, origin="lower")
        axes[i, 2*j+1].imshow(rec, cmap="hot", vmin=0, vmax=vmax, origin="lower")

    axes[i, 0].set_ylabel(f"{r['sha'][:8]}...\nL1={r['l1']:.4f}\n{r['n_voxels']:,}v",
                           fontsize=7, rotation=0, ha="right", va="center")

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT}")
plt.close()
