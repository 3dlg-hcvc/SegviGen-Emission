#!/usr/bin/env python3
"""
Visualize the single-sample emission VAE overfit for 001dd281...
Compares GT vs reconstruction across checkpoints at 2.5k, 5k, 7.5k, 10k steps.
Shows max-luminance projections (XY, XZ, YZ).

Usage:
    conda run -n trellis2 python data_toolkit/vis_emission_vae_1sample.py
"""
import sys, os, types, glob
sys.path.insert(0, "/localhome/dya78/code/TRELLIS.2-lightning")
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
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
SHA       = "0007deb6d96c4474b80faa5aa3888760"
RUN_DIR   = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_1sample_overfit"
OUT_PNG   = "/localhome/dya78/code/TRELLIS.2-lightning/visualizations/emission_vae_1sample_overfit.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load common cfg ───────────────────────────────────────────────────────────
with open(os.path.join(RUN_DIR, "config.yaml")) as f:
    cfg = yaml.safe_load(f)

pbr_enc_name = cfg["trainer"]["args"].get("pretrained_pbr_encoder")
pbr_encoder  = trellis_models.from_pretrained(pbr_enc_name).eval().float().to(device) if pbr_enc_name else None
print(f"PBR encoder: {'loaded' if pbr_encoder else 'none'}")

# ── Discover checkpoints ─────────────────────────────────────────────────────
ckpt_dir = os.path.join(RUN_DIR, "ckpts")
step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "stepstep=*.ckpt")))
last_ckpt  = os.path.join(ckpt_dir, "last.ckpt")

ckpts = []
for p in step_ckpts:
    step = int(os.path.basename(p).replace("stepstep=", "").replace(".ckpt", ""))
    ckpts.append((step, p))
if os.path.exists(last_ckpt):
    # Add last only if it isn't already covered by a step ckpt
    ckpt_steps = {s for s, _ in ckpts}
    last_state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    last_step  = int(last_state.get("global_step", 0))
    if last_step not in ckpt_steps:
        ckpts.append((last_step, last_ckpt))
ckpts.sort(key=lambda x: x[0])
print(f"Checkpoints found: {[(s, os.path.basename(p)) for s, p in ckpts]}")

# ── Load GT ───────────────────────────────────────────────────────────────────
coords, emit_attr = o_voxel.io.read_vxz(
    f"{DATA_ROOT}/emission_voxels_256/{SHA}.vxz", num_threads=4
)
emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0

_, pbr_attr = o_voxel.io.read_vxz(
    f"{DATA_ROOT}/pbr_voxels_256/{SHA}.vxz", num_threads=4
)
pbr_feats = torch.cat([
    pbr_attr["base_color"].float() / 255.0 * 2.0 - 1.0,
    pbr_attr["metallic"].float()   / 255.0 * 2.0 - 1.0,
    pbr_attr["roughness"].float()  / 255.0 * 2.0 - 1.0,
    pbr_attr["alpha"].float()      / 255.0 * 2.0 - 1.0,
], dim=-1)

batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
combined  = torch.cat([emit_feats, pbr_feats], dim=-1)
x0        = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
x0_emit   = x0.replace(emit_feats)
print(f"Voxels: {coords.shape[0]:,}  Nonzero emission: {(emit_feats.norm(dim=-1) > 0).sum().item():,}")


def max_proj(sparse_t, res=256):
    c   = sparse_t.coords[:, 1:].long()
    mag = (sparse_t.feats * 0.5 + 0.5).norm(dim=-1)
    xy  = torch.zeros(res, res)
    xz  = torch.zeros(res, res)
    yz  = torch.zeros(res, res)
    for i in range(c.shape[0]):
        xi, yi, zi = c[i, 0].item(), c[i, 1].item(), c[i, 2].item()
        v = mag[i].item()
        if v > xy[yi, xi]: xy[yi, xi] = v
        if v > xz[zi, xi]: xz[zi, xi] = v
        if v > yz[zi, yi]: yz[zi, yi] = v
    return xy.numpy(), xz.numpy(), yz.numpy()


xy_gt, xz_gt, yz_gt = max_proj(x0_emit)

# ── Reconstruct at each checkpoint ───────────────────────────────────────────
results = []
for step, ckpt_path in ckpts:
    print(f"  step {step:>6}: {os.path.basename(ckpt_path)}", end=" ... ", flush=True)
    models  = build_models(cfg["models"])
    encoder = models["encoder"]
    decoder = models["decoder"]

    state     = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    enc_state = {k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")}
    dec_state = {k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.to(device).eval()
    decoder.to(device).eval()

    with torch.no_grad():
        x0_gpu = x0.to(device)
        z_emit = encoder(x0_emit.to(device))
        if pbr_encoder is not None:
            x_pbr    = x0_gpu.replace(x0_gpu.feats[:, 3:])
            z_pbr    = pbr_encoder(x_pbr)
            z_comb   = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
        else:
            z_comb = z_emit
        y = decoder(z_comb).to("cpu")

    l1 = (x0_emit.feats - y.feats).abs().mean().item()
    xy_rec, xz_rec, yz_rec = max_proj(y)
    results.append(dict(step=step, l1=l1, xy=xy_rec, xz=xz_rec, yz=yz_rec))
    print(f"L1={l1:.5f}")
    torch.cuda.empty_cache()

# ── Plot: 1 GT row + N checkpoint rows, 6 cols (XY/XZ/YZ × GT/Rec) ───────────
n     = len(results)
nrows = 1 + n
ncols = 6
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))

vmax = max(
    xy_gt.max(), xz_gt.max(), yz_gt.max(),
    *(max(r["xy"].max(), r["xz"].max(), r["yz"].max()) for r in results),
    1e-3,
)

fig.suptitle(
    f"Single-sample overfit — {SHA[:16]}...\n"
    f"{coords.shape[0]:,} total voxels, "
    f"{(emit_feats.norm(dim=-1) > 0).sum().item():,} non-zero emission voxels",
    fontsize=13, fontweight="bold", y=1.0,
)

col_titles = ["GT  XY", "GT  XZ", "GT  YZ", "Rec XY", "Rec XZ", "Rec YZ"]
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=10, fontweight="bold")

# GT row
for c, img in enumerate([xy_gt, xz_gt, yz_gt]):
    axes[0, c].imshow(img, cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[0, c + 3].imshow(np.zeros_like(img), cmap="hot", vmin=0, vmax=vmax, origin="lower")
axes[0, 0].set_ylabel("Ground Truth", fontsize=9, rotation=0, ha="right", va="center")

# Per-checkpoint rows
for i, r in enumerate(results):
    row = i + 1
    for c, img in enumerate([xy_gt, xz_gt, yz_gt]):
        axes[row, c].imshow(img, cmap="hot", vmin=0, vmax=vmax, origin="lower")
    for c, img in enumerate([r["xy"], r["xz"], r["yz"]]):
        axes[row, c + 3].imshow(img, cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[row, 0].set_ylabel(
        f"step {r['step']:,}\nL1={r['l1']:.5f}",
        fontsize=8, rotation=0, ha="right", va="center",
    )

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
plt.close()
