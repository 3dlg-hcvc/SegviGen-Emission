#!/usr/bin/env python3
"""
Test the pretrained PBR VAE (encoder + decoder) on one sample.
Visualizes each PBR channel separately: base_color, metallic, roughness, alpha.
Also shows the emission VAE (Option B) reconstruction for comparison.
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
OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "vae_channel_vis.png")

device = "cuda"


# ── 1. Load voxels ───────────────────────────────────────────────────────────
print("=== Loading voxels ===")
coords_e, attr_e = o_voxel.io.read_vxz(
    os.path.join(DATA_ROOT, f"emission_voxels_256/{SAMPLE}.vxz"), num_threads=4
)
emit_feats = attr_e["emissive"].float() / 255.0 * 2.0 - 1.0
batch_col = torch.zeros(coords_e.shape[0], 1, dtype=coords_e.dtype)
x_emit = sp.SparseTensor(emit_feats, torch.cat([batch_col, coords_e], dim=-1))

coords_p, attr_p = o_voxel.io.read_vxz(
    os.path.join(DATA_ROOT, f"pbr_voxels_256/{SAMPLE}.vxz"), num_threads=4
)
pbr_keys = ["base_color", "metallic", "roughness", "alpha"]
pbr_feats = torch.cat([attr_p[a].float() / 255.0 * 2.0 - 1.0 for a in pbr_keys], dim=-1)
batch_col_p = torch.zeros(coords_p.shape[0], 1, dtype=coords_p.dtype)
x_pbr = sp.SparseTensor(pbr_feats, torch.cat([batch_col_p, coords_p], dim=-1))
print(f"  Emission: {emit_feats.shape[0]:,} voxels, {emit_feats.shape[1]} ch")
print(f"  PBR:      {pbr_feats.shape[0]:,} voxels, {pbr_feats.shape[1]} ch")


# ── 2. PBR VAE encode → decode ──────────────────────────────────────────────
print("\n=== Pretrained PBR VAE ===")
pbr_encoder = models.from_pretrained(
    "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
).eval().float().to(device)
pbr_decoder = models.from_pretrained(
    "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
).eval().float().to(device)

x_pbr_gpu = x_pbr.to(device)
z_pbr = pbr_encoder(x_pbr_gpu)
y_pbr = pbr_decoder(z_pbr)
print(f"  PBR latent:       {z_pbr.feats.shape}")
print(f"  PBR recon:        {y_pbr.feats.shape}")

gt_f = x_pbr_gpu.feats
rc_f = y_pbr.feats
pbr_ch_l1 = {}
for name, sl in [("base_color", slice(0,3)), ("metallic", slice(3,4)),
                 ("roughness", slice(4,5)), ("alpha", slice(5,6))]:
    pbr_ch_l1[name] = (gt_f[:, sl] - rc_f[:, sl]).abs().mean().item()
pbr_l1 = (gt_f - rc_f).abs().mean().item()
print(f"  PBR L1 (overall):   {pbr_l1:.6f}")
for name, v in pbr_ch_l1.items():
    print(f"  PBR L1 ({name:>10s}): {v:.6f}")

x_pbr_cpu = x_pbr
y_pbr_cpu = y_pbr.to("cpu")
z_pbr_for_emit = z_pbr  # keep on GPU for emission decoder

del pbr_encoder, pbr_decoder
torch.cuda.empty_cache()


# ── 3. Emission VAE (Option B) encode → decode ──────────────────────────────
print("\n=== Emission VAE (Option B) ===")
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

x_emit_only = x_emit.to(device)
z_emit = emit_enc(x_emit_only)
z_combined = z_emit.replace(torch.cat([z_emit.feats, z_pbr_for_emit.feats], dim=-1))
y_emit = emit_dec(z_combined)

gt_e = x_emit_only.feats
rc_e = y_emit.feats
emit_ch_l1 = {}
for name, sl in [("R", slice(0,1)), ("G", slice(1,2)), ("B", slice(2,3))]:
    emit_ch_l1[name] = (gt_e[:, sl] - rc_e[:, sl]).abs().mean().item()
emit_l1 = (gt_e - rc_e).abs().mean().item()
print(f"  Emission latent:  {z_emit.feats.shape}")
print(f"  Combined latent:  {z_combined.feats.shape}")
print(f"  Emission recon:   {y_emit.feats.shape}")
print(f"  Emission L1 (overall): {emit_l1:.6f}")
for name, v in emit_ch_l1.items():
    print(f"  Emission L1 ({name}):      {v:.6f}")

x_emit_cpu = x_emit
y_emit_cpu = y_emit.to("cpu")

del emit_enc, emit_dec, z_pbr_for_emit
torch.cuda.empty_cache()


# ── 4. Visualization ─────────────────────────────────────────────────────────
print("\n=== Generating visualization ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def max_proj_channels(sparse_t, channel_slice, res=256, to_01=True):
    """Max-intensity projection for specific channels, returns [res, res, C] for 3 axes."""
    c = sparse_t.coords[:, 1:].long()
    feats = sparse_t.feats[:, channel_slice]
    if to_01:
        feats = feats * 0.5 + 0.5
    feats = feats.clamp(0, 1)
    C = feats.shape[1]

    projs = []
    for keep_dims in [(1, 0), (2, 0), (2, 1)]:
        img = torch.zeros(res, res, C)
        depth = torch.full((res, res), -1.0)
        mag = feats.norm(dim=-1) if C > 1 else feats[:, 0]
        for i in range(c.shape[0]):
            r, col = c[i, keep_dims[0]].item(), c[i, keep_dims[1]].item()
            v = mag[i].item()
            if v > depth[r, col].item():
                depth[r, col] = v
                img[r, col] = feats[i]
        projs.append(img.numpy())
    return projs


# Channel layout for PBR
pbr_layout = {
    "base_color": slice(0, 3),
    "metallic":   slice(3, 4),
    "roughness":  slice(4, 5),
    "alpha":      slice(5, 6),
}

channels = [
    ("Base Color",  pbr_layout["base_color"],  x_pbr_cpu, y_pbr_cpu, True,  pbr_ch_l1["base_color"]),
    ("Metallic",    pbr_layout["metallic"],    x_pbr_cpu, y_pbr_cpu, False, pbr_ch_l1["metallic"]),
    ("Roughness",   pbr_layout["roughness"],   x_pbr_cpu, y_pbr_cpu, False, pbr_ch_l1["roughness"]),
    ("Alpha",       pbr_layout["alpha"],       x_pbr_cpu, y_pbr_cpu, False, pbr_ch_l1["alpha"]),
    ("Emission",    slice(0, 3),               x_emit_cpu, y_emit_cpu, True, emit_l1),
]

n_cols = len(channels) * 2  # GT + Recon per channel
n_rows = 3  # XY, XZ, YZ

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2 + 2.5))

pbr_detail = "  |  ".join(f"{k}: {v:.4f}" for k, v in pbr_ch_l1.items())
emit_detail = "  |  ".join(f"{k}: {v:.4f}" for k, v in emit_ch_l1.items())
fig.suptitle(
    f"PBR VAE + Emission VAE (Option B)  |  sample: {SAMPLE[:12]}...\n"
    f"PBR L1 overall={pbr_l1:.4f}  [{pbr_detail}]\n"
    f"Emission L1 overall={emit_l1:.4f}  [{emit_detail}]",
    fontsize=11, fontweight="bold", y=0.99,
)
view_labels = ["XY", "XZ", "YZ"]

for ci, (name, ch_slice, gt_st, rec_st, is_rgb, ch_l1) in enumerate(channels):
    gt_projs = max_proj_channels(gt_st, ch_slice)
    rec_projs = max_proj_channels(rec_st, ch_slice)

    col_gt = ci * 2
    col_rec = ci * 2 + 1

    for row in range(3):
        gt_img = gt_projs[row]
        rec_img = rec_projs[row]

        if is_rgb:
            axes[row][col_gt].imshow(gt_img, origin="lower")
            axes[row][col_rec].imshow(rec_img, origin="lower")
        else:
            axes[row][col_gt].imshow(gt_img[..., 0], origin="lower", cmap="gray", vmin=0, vmax=1)
            axes[row][col_rec].imshow(rec_img[..., 0], origin="lower", cmap="gray", vmin=0, vmax=1)

        axes[row][col_gt].set_xticks([])
        axes[row][col_gt].set_yticks([])
        axes[row][col_rec].set_xticks([])
        axes[row][col_rec].set_yticks([])

        if ci == 0:
            axes[row][col_gt].set_ylabel(view_labels[row], fontsize=11, fontweight="bold")

    axes[0][col_gt].set_title(f"{name}\nGT", fontsize=9, fontweight="bold")
    axes[0][col_rec].set_title(f"{name}\nRecon (L1={ch_l1:.4f})", fontsize=8, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
plt.close()
print("Done!")
