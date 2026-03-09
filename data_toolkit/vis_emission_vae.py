#!/usr/bin/env python3
"""
Visualize the emission VAE overfit: input vs. reconstructed emission voxels.
Produces a side-by-side figure with max-projection views along XY, XZ, YZ axes.
"""
import sys, os, types
sys.path.insert(0, "/localhome/dya78/code/TRELLIS.2-lightning")
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# stub GPU-only o_voxel submodules
for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

# ── load the single training sample ──────────────────────────────────────────
SAMPLE = "0007deb6d96c4474b80faa5aa3888760"
coords, attr = o_voxel.io.read_vxz(
    f"/tmp/lightgen_test/emission_voxels_256/{SAMPLE}.vxz", num_threads=4
)
feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0  # [N,3] in [-1,1]

from trellis2.modules import sparse as sp
batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
x0 = sp.SparseTensor(feats, torch.cat([batch_col, coords], dim=-1))

# ── load checkpoint & reconstruct ────────────────────────────────────────────
import yaml
from trellis2.trainers.builders import build_models

cfg_path = "/tmp/emission_vae_overfit/config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

models = build_models(cfg["models"])
encoder = models["encoder"]
decoder = models["decoder"]

CKPT = "/tmp/emission_vae_overfit/ckpts/last.ckpt"
ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt["state_dict"]

enc_state = {k[len("models.encoder."):]: v for k, v in state.items() if k.startswith("models.encoder.")}
dec_state = {k[len("models.decoder."):]: v for k, v in state.items() if k.startswith("models.decoder.")}
encoder.load_state_dict(enc_state)
decoder.load_state_dict(dec_state)

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device).eval()
decoder.to(device).eval()
x0_gpu = x0.to(device)

with torch.no_grad():
    z = encoder(x0_gpu)
    y = decoder(z)

# bring results back to CPU for numpy/matplotlib
x0 = x0_gpu.to("cpu")
y  = y.to("cpu")

# ── max-projection helper ─────────────────────────────────────────────────────
def max_proj(sparse_t, res=256):
    """Return max-luminance projections along XY, XZ, YZ planes."""
    c   = sparse_t.coords[:, 1:].long()
    mag = (sparse_t.feats * 0.5 + 0.5).norm(dim=-1)
    xy = torch.zeros(res, res)
    xz = torch.zeros(res, res)
    yz = torch.zeros(res, res)
    for i in range(c.shape[0]):
        xi, yi, zi = c[i, 0].item(), c[i, 1].item(), c[i, 2].item()
        v = mag[i].item()
        if v > xy[yi, xi]: xy[yi, xi] = v
        if v > xz[zi, xi]: xz[zi, xi] = v
        if v > yz[zi, yi]: yz[zi, yi] = v
    return xy.numpy(), xz.numpy(), yz.numpy()

print("Building projections …")
xy_gt,  xz_gt,  yz_gt  = max_proj(x0)
xy_rec, xz_rec, yz_rec = max_proj(y)

err_feats = (x0.feats - y.feats).norm(dim=-1, keepdim=True).expand(-1, 3)
err_st = sp.SparseTensor(err_feats, x0.coords)
xy_err, xz_err, yz_err = max_proj(err_st)

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
l1 = (x0.feats - y.feats).abs().mean().item()
fig.suptitle(
    "Emission VAE Overfit  |  sample: " + SAMPLE[:8] + "...\n"
    "500 steps, lr=0.001   final L1 loss = {:.4f}".format(l1),
    fontsize=13, fontweight="bold"
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]
view_labels  = ["XY projection", "XZ projection", "YZ projection"]
col_titles   = ["Input (GT)", "Reconstructed", "|Error|", "Overlay (G=GT, R=Rec)"]

for r, (gt, rec, err) in enumerate(
    [(xy_gt, xy_rec, xy_err), (xz_gt, xz_rec, xz_err), (yz_gt, yz_rec, yz_err)]
):
    vmax = max(gt.max(), rec.max(), 1e-3)
    axes[r][0].imshow(gt,  cmap="hot",    vmin=0, vmax=vmax, origin="lower")
    axes[r][1].imshow(rec, cmap="hot",    vmin=0, vmax=vmax, origin="lower")
    axes[r][2].imshow(err, cmap="plasma", vmin=0, origin="lower")
    overlay = np.zeros((*gt.shape, 3))
    overlay[:, :, 1] = gt  / vmax
    overlay[:, :, 0] = rec / vmax
    axes[r][3].imshow(np.clip(overlay, 0, 1), origin="lower")
    axes[r][0].set_ylabel(view_labels[r], fontsize=9)
    for c in range(4):
        axes[r][c].set_xticks([]); axes[r][c].set_yticks([])

for c, t in enumerate(col_titles):
    axes[0][c].set_title(t, fontsize=10, fontweight="bold")

# stats
gt_mag  = (x0.feats * 0.5 + 0.5).norm(dim=-1)
rec_mag = (y.feats  * 0.5 + 0.5).norm(dim=-1)
stats = (
    "Voxels : {:,}\n"
    "GT  emission mean : {:.4f}\n"
    "Rec emission mean : {:.4f}\n"
    "L1 recon error    : {:.4f}"
).format(x0.feats.shape[0], gt_mag.mean().item(), rec_mag.mean().item(), l1)
fig.text(0.02, 0.01, stats, fontsize=9, family="monospace",
         va="bottom", bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

OUT = "/localhome/dya78/code/TRELLIS.2-lightning/emission_vae_overfit_vis.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved:", OUT)
plt.close()
