#!/usr/bin/env python3
"""
End-to-end test: emission VAE reconstruction + pretrained PBR encoder latent extraction.

Loads one sample, runs:
  1. Emission VAE encode → decode → reconstruction error
  2. Pretrained PBR encoder → material latent (proof of concept for Option B conditioning)
  3. Produces a side-by-side visualization
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import yaml

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

import trellis2.models as models
import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models

torch.set_grad_enabled(False)

SAMPLE = "0007deb6d96c4474b80faa5aa3888760"
DATA_ROOT = "/tmp/lightgen_test"
EMISSION_CKPT = "/tmp/emission_vae_optB_overfit/ckpts/stepstep=0001000.ckpt"
EMISSION_CFG = "/tmp/emission_vae_optB_overfit/config.yaml"
OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "emission_vae_optB_test.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── 1. Load emission + PBR voxels ────────────────────────────────────────────
print("\n=== Loading voxels ===")
coords_e, attr_e = o_voxel.io.read_vxz(
    os.path.join(DATA_ROOT, f"emission_voxels_256/{SAMPLE}.vxz"), num_threads=4
)
emit_feats = attr_e["emissive"].float() / 255.0 * 2.0 - 1.0  # [N, 3] in [-1,1]
batch_col = torch.zeros(coords_e.shape[0], 1, dtype=coords_e.dtype)
x_emit = sp.SparseTensor(emit_feats, torch.cat([batch_col, coords_e], dim=-1))
print(f"  Emission voxels: {emit_feats.shape[0]:,} voxels, {emit_feats.shape[1]} channels")

coords_p, attr_p = o_voxel.io.read_vxz(
    os.path.join(DATA_ROOT, f"pbr_voxels_256/{SAMPLE}.vxz"), num_threads=4
)
pbr_attrs = ["base_color", "metallic", "roughness", "alpha"]
pbr_feats = torch.cat([attr_p[a].float() / 255.0 * 2.0 - 1.0 for a in pbr_attrs], dim=-1)
batch_col_p = torch.zeros(coords_p.shape[0], 1, dtype=coords_p.dtype)
x_pbr = sp.SparseTensor(pbr_feats, torch.cat([batch_col_p, coords_p], dim=-1))
print(f"  PBR voxels:      {pbr_feats.shape[0]:,} voxels, {pbr_feats.shape[1]} channels")


# ── 2. Emission VAE encode → decode ──────────────────────────────────────────
print("\n=== Emission VAE ===")
with open(EMISSION_CFG) as f:
    cfg = yaml.safe_load(f)
emit_models = build_models(cfg["models"])
emit_enc = emit_models["encoder"]
emit_dec = emit_models["decoder"]

ckpt = torch.load(EMISSION_CKPT, map_location="cpu")
state = ckpt["state_dict"]
enc_state = {k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")}
dec_state = {k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")}
emit_enc.load_state_dict(enc_state)
emit_dec.load_state_dict(dec_state)
emit_enc.to(device).eval()
emit_dec.to(device).eval()

# ── 2b. Pretrained PBR encoder → material latent ────────────────────────────
print("\n=== Pretrained PBR encoder ===")
pbr_encoder = models.from_pretrained(
    "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
).eval().float().to(device)
z_pbr = pbr_encoder(x_pbr.to(device))
print(f"  PBR material latent: {z_pbr.feats.shape} (coords: {z_pbr.coords.shape})")
print(f"  Latent stats:  mean={z_pbr.feats.mean().item():.4f}  std={z_pbr.feats.std().item():.4f}")

# ── 2c. Emission encode → concat with material latent → decode ──────────────
print("\n=== Emission VAE (Option B) ===")
x_emit_only = x_emit.to(device).replace(x_emit.to(device).feats[:, :3])
z_emit = emit_enc(x_emit_only)
print(f"  Emission latent:  {z_emit.feats.shape} (coords: {z_emit.coords.shape})")
print(f"  PBR latent:       {z_pbr.feats.shape} (coords: {z_pbr.coords.shape})")
assert z_emit.coords.shape == z_pbr.coords.shape, (
    f"Latent coord mismatch: emit {z_emit.coords.shape} vs pbr {z_pbr.coords.shape}"
)

z_combined = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
print(f"  Combined latent:  {z_combined.feats.shape}")
y_emit = emit_dec(z_combined)
print(f"  Reconstruction:   {y_emit.feats.shape}")

target = x_emit.to(device).feats[:, :3]
l1 = (target - y_emit.feats).abs().mean().item()
print(f"  L1 recon error:   {l1:.6f}")

# move back to CPU
x_emit_cpu = x_emit
y_emit_cpu = y_emit.to("cpu")
z_emit_cpu = z_emit.to("cpu")
z_pbr_cpu = z_pbr.to("cpu")

del emit_enc, emit_dec, pbr_encoder
torch.cuda.empty_cache()


# ── 4. Visualization ─────────────────────────────────────────────────────────
print("\n=== Generating visualization ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def max_proj_rgb(sparse_t, res=256):
    """Max-luminance projection keeping RGB color, along 3 axes."""
    c = sparse_t.coords[:, 1:].long()
    rgb = (sparse_t.feats * 0.5 + 0.5).clamp(0, 1)
    mag = rgb.norm(dim=-1)

    projs = []
    for dim_pair, size in [((1, 0), (res, res)), ((2, 0), (res, res)), ((2, 1), (res, res))]:
        img = torch.zeros(*size, 3)
        depth = torch.full(size, -1.0)
        for i in range(c.shape[0]):
            r, col = c[i, dim_pair[0]].item(), c[i, dim_pair[1]].item()
            v = mag[i].item()
            if v > depth[r, col].item():
                depth[r, col] = v
                img[r, col] = rgb[i]
        projs.append(img.numpy())
    return projs  # [XY, XZ, YZ]


def max_proj_scalar(feats_2d, coords, res=256):
    """Max-value projection of a scalar (e.g. latent channel mean)."""
    c = coords[:, 1:].long()
    vals = feats_2d.float()
    projs = []
    for dim_pair in [(1, 0), (2, 0), (2, 1)]:
        img = torch.zeros(res, res)
        for i in range(c.shape[0]):
            r, col = c[i, dim_pair[0]].item(), c[i, dim_pair[1]].item()
            v = vals[i].item()
            if abs(v) > abs(img[r, col].item()):
                img[r, col] = v
        projs.append(img.numpy())
    return projs


gt_projs = max_proj_rgb(x_emit_cpu)
rec_projs = max_proj_rgb(y_emit_cpu)

# Latent visualization: show mean across channels at latent resolution
lat_res = int(z_emit_cpu.coords[:, 1:].max().item()) + 1
emit_lat_projs = max_proj_scalar(z_emit_cpu.feats.mean(dim=-1), z_emit_cpu.coords, lat_res)

pbr_lat_res = int(z_pbr_cpu.coords[:, 1:].max().item()) + 1
pbr_lat_projs = max_proj_scalar(z_pbr_cpu.feats.mean(dim=-1), z_pbr_cpu.coords, pbr_lat_res)

fig = plt.figure(figsize=(20, 12))
fig.suptitle(
    f"Emission VAE E2E Test  |  sample: {SAMPLE[:12]}...\n"
    f"L1 recon = {l1:.6f}  |  emit latent: {z_emit_cpu.feats.shape}  |  "
    f"PBR latent: {z_pbr_cpu.feats.shape}",
    fontsize=12, fontweight="bold",
)

gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)
axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(3)]
view_labels = ["XY", "XZ", "YZ"]
col_titles = ["GT Emission", "Reconstructed", "|Error|", "Emission Latent\n(mean over C)", "PBR Material Latent\n(mean over C)"]

for r in range(3):
    gt = gt_projs[r]
    rec = rec_projs[r]
    err = np.linalg.norm(gt - rec, axis=-1)

    axes[r][0].imshow(gt, origin="lower")
    axes[r][1].imshow(rec, origin="lower")
    axes[r][2].imshow(err, cmap="hot", vmin=0, origin="lower")

    vabs = max(abs(emit_lat_projs[r].min()), abs(emit_lat_projs[r].max()), 1e-6)
    axes[r][3].imshow(emit_lat_projs[r], cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")

    vabs2 = max(abs(pbr_lat_projs[r].min()), abs(pbr_lat_projs[r].max()), 1e-6)
    axes[r][4].imshow(pbr_lat_projs[r], cmap="RdBu_r", vmin=-vabs2, vmax=vabs2, origin="lower")

    axes[r][0].set_ylabel(view_labels[r], fontsize=11, fontweight="bold")
    for c in range(5):
        axes[r][c].set_xticks([])
        axes[r][c].set_yticks([])

for c, t in enumerate(col_titles):
    axes[0][c].set_title(t, fontsize=10, fontweight="bold")

stats_text = (
    f"Emission voxels : {x_emit_cpu.feats.shape[0]:,}\n"
    f"Emission latent : {z_emit_cpu.feats.shape} at res ~{lat_res}\n"
    f"PBR latent      : {z_pbr_cpu.feats.shape} at res ~{pbr_lat_res}\n"
    f"L1 recon error  : {l1:.6f}"
)
fig.text(0.01, 0.01, stats_text, fontsize=9, family="monospace",
         va="bottom", bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
plt.close()
print("Done!")
