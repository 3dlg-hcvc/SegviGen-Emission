#!/usr/bin/env python3
"""
Test emission DiT (full parity): sample from trained flow model with
shape+PBR concat conditioning and DINOv3 image cross-attention,
decode with emission VAE, compare against GT, and visualize.
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import yaml
from PIL import Image

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

import trellis2.models as models
import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor

torch.set_grad_enabled(False)

SAMPLE = "0007deb6d96c4474b80faa5aa3888760"
DATA_ROOT = "data/lightgen_test"

DIT_CKPT = "/tmp/emission_dit_full_overfit/ckpts/last.ckpt"
DIT_CFG = "/tmp/emission_dit_full_overfit/config.yaml"

EMIT_VAE_CKPT = "/tmp/emission_vae_optB_overfit/ckpts/stepstep=0001000.ckpt"
EMIT_VAE_CFG = "/tmp/emission_vae_optB_overfit/config.yaml"

THUMBNAIL = os.path.join(DATA_ROOT, f"thumbnails/{SAMPLE}.png")

OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "emission_dit_full_test.png")
OUT_LATENT_PNG = os.path.join(os.path.dirname(__file__), "..", "emission_dit_full_latent_test.png")
device = "cuda"


def _load_norm(cfg_dict, key):
    n = cfg_dict.get(key)
    if n is None:
        return None, None
    return (torch.tensor(n["mean"], dtype=torch.float32),
            torch.tensor(n["std"], dtype=torch.float32))


def normalize(feats, mean, std):
    if mean is None:
        return feats
    return (feats - mean.to(feats.device)) / std.to(feats.device)


def denormalize(feats, mean, std):
    if mean is None:
        return feats
    return feats * std.to(feats.device) + mean.to(feats.device)


# ── 1. Load config & normalization stats ──────────────────────────────────────
print("=== Loading config ===")
with open(DIT_CFG) as f:
    dit_cfg = yaml.safe_load(f)

ds_args = dit_cfg["dataset"]["args"]
emit_mean, emit_std = _load_norm(ds_args, "emission_normalization")
pbr_mean, pbr_std = _load_norm(ds_args, "pbr_normalization")
shape_mean, shape_std = _load_norm(ds_args, "shape_normalization")
print(f"  Normalization loaded for emission, PBR, shape")


# ── 2. Load GT voxels ────────────────────────────────────────────────────────
print("\n=== Loading GT voxels ===")
coords_e, attr_e = o_voxel.io.read_vxz(
    os.path.join(DATA_ROOT, f"emission_voxels_256/{SAMPLE}.vxz"), num_threads=4
)
gt_emit = attr_e["emissive"].float() / 255.0 * 2.0 - 1.0
batch_col = torch.zeros(coords_e.shape[0], 1, dtype=coords_e.dtype)
x_emit_gt = sp.SparseTensor(gt_emit, torch.cat([batch_col, coords_e], dim=-1))
print(f"  GT emission voxels: {gt_emit.shape}")


# ── 3. Load pre-encoded latents (raw, unnormalized) ─────────────────────────
print("\n=== Loading pre-encoded latents ===")

def load_latent_npz(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))

z_emit_raw = load_latent_npz(os.path.join(DATA_ROOT, "emission_latents/emission_vae_optB_overfit", f"{SAMPLE}.npz"))
z_pbr_raw = load_latent_npz(os.path.join(DATA_ROOT, "pbr_latents/tex_enc_next_dc_f16c32_fp16_256", f"{SAMPLE}.npz"))
z_shape_raw = load_latent_npz(os.path.join(DATA_ROOT, "shape_latents/shape_enc_next_dc_f16c32_fp16_256", f"{SAMPLE}.npz"))

print(f"  Emission latent: {z_emit_raw.feats.shape}")
print(f"  PBR latent:      {z_pbr_raw.feats.shape}")
print(f"  Shape latent:    {z_shape_raw.feats.shape}")

z_emit_norm = z_emit_raw.replace(normalize(z_emit_raw.feats, emit_mean, emit_std)).to(device)
z_pbr_norm = z_pbr_raw.replace(normalize(z_pbr_raw.feats, pbr_mean, pbr_std)).to(device)
z_shape_norm = z_shape_raw.replace(normalize(z_shape_raw.feats, shape_mean, shape_std)).to(device)

concat_cond = z_emit_norm.replace(
    torch.cat([z_shape_norm.feats, z_pbr_norm.feats], dim=-1)
)
print(f"  concat_cond (shape+PBR): {concat_cond.feats.shape}")


# ── 4. Load DINOv3 and encode thumbnail ──────────────────────────────────────
print("\n=== Encoding thumbnail with DINOv3 ===")
dino_cfg = dit_cfg["trainer"]["args"]["image_cond_model"]["args"]
dino = DinoV3FeatureExtractor(**dino_cfg)
dino.cuda()

image = Image.open(THUMBNAIL).convert("RGB")
image_tensor = torch.from_numpy(
    np.array(image.resize((dino_cfg["image_size"], dino_cfg["image_size"]), Image.LANCZOS))
    .astype(np.float32) / 255.0
).permute(2, 0, 1).unsqueeze(0).to(device)
from torchvision import transforms
dino_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_norm = dino_transform(image_tensor)
image_features = dino.extract_features(image_norm.to(dino.model.embeddings.patch_embeddings.weight.dtype))
neg_cond = torch.zeros_like(image_features)
print(f"  DINOv3 features: {image_features.shape}")

del dino
torch.cuda.empty_cache()


# ── 5. Load DiT model ────────────────────────────────────────────────────────
print("\n=== Loading DiT model ===")
dit_models = build_models(dit_cfg["models"])
dit = dit_models["denoiser"]

ckpt = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
state = ckpt["state_dict"]
dit_state = {k.removeprefix("models.denoiser."): v
             for k, v in state.items() if k.startswith("models.denoiser.")}
dit.load_state_dict(dit_state)
dit.to(device).eval()
print(f"  DiT loaded: {sum(p.numel() for p in dit.parameters()):,} params")


# ── 6. Sample from DiT with CFG ──────────────────────────────────────────────
print("\n=== Sampling from DiT (CFG) ===")
sigma_min = dit_cfg["trainer"]["args"]["sigma_min"]
sampler = FlowEulerCfgSampler(sigma_min=sigma_min)

noise = z_emit_norm.replace(torch.randn_like(z_emit_norm.feats))

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    result = sampler.sample(
        dit, noise,
        cond=image_features,
        neg_cond=neg_cond,
        concat_cond=concat_cond,
        steps=50,
        guidance_strength=3.0,
        verbose=True,
    )
z_emit_sampled_norm = result.samples

z_emit_sampled_raw = z_emit_sampled_norm.replace(
    denormalize(z_emit_sampled_norm.feats, emit_mean.to(device), emit_std.to(device))
)

latent_l1 = (z_emit_raw.feats.to(device) - z_emit_sampled_raw.feats).abs().mean().item()
print(f"  Sampled emission latent: {z_emit_sampled_raw.feats.shape}")
print(f"  Latent L1 (GT vs sampled, raw): {latent_l1:.6f}")


# ── 7. Decode with emission VAE decoder ──────────────────────────────────────
print("\n=== Decoding with emission VAE decoder ===")
with open(EMIT_VAE_CFG) as f:
    vae_cfg = yaml.safe_load(f)
vae_models = build_models(vae_cfg["models"])
emit_dec = vae_models["decoder"]
emit_enc = vae_models["encoder"]

vae_ckpt = torch.load(EMIT_VAE_CKPT, map_location="cpu", weights_only=False)
vae_state = vae_ckpt["state_dict"]
dec_state = {k.removeprefix("models.decoder."): v for k, v in vae_state.items() if k.startswith("models.decoder.")}
enc_state = {k.removeprefix("models.encoder."): v for k, v in vae_state.items() if k.startswith("models.encoder.")}
emit_dec.load_state_dict(dec_state)
emit_enc.load_state_dict(enc_state)
emit_dec.to(device).eval()
emit_enc.to(device).eval()

x_emit_gpu = x_emit_gt.to(device)
z_enc = emit_enc(x_emit_gpu)

z_enc_combined = z_enc.replace(torch.cat([z_enc.feats, z_pbr_raw.feats.to(device)], dim=-1))
y_vae_only = emit_dec(z_enc_combined)

z_gt_with_cache = z_enc.replace(torch.cat([z_emit_raw.feats.to(device), z_pbr_raw.feats.to(device)], dim=-1))
y_gt_recon = emit_dec(z_gt_with_cache)

z_dit_with_cache = z_enc.replace(torch.cat([z_emit_sampled_raw.feats, z_pbr_raw.feats.to(device)], dim=-1))
y_dit = emit_dec(z_dit_with_cache)

dit_l1 = (x_emit_gpu.feats - y_dit.feats).abs().mean().item()
gt_recon_l1 = (x_emit_gpu.feats - y_gt_recon.feats).abs().mean().item()
vae_only_l1 = (x_emit_gpu.feats - y_vae_only.feats).abs().mean().item()

dit_ch_l1 = {}
gt_ch_l1 = {}
vae_ch_l1 = {}
for name, sl in [("R", slice(0, 1)), ("G", slice(1, 2)), ("B", slice(2, 3))]:
    dit_ch_l1[name] = (x_emit_gpu.feats[:, sl] - y_dit.feats[:, sl]).abs().mean().item()
    gt_ch_l1[name] = (x_emit_gpu.feats[:, sl] - y_gt_recon.feats[:, sl]).abs().mean().item()
    vae_ch_l1[name] = (x_emit_gpu.feats[:, sl] - y_vae_only.feats[:, sl]).abs().mean().item()

print(f"  VAE-only L1:             {vae_only_l1:.6f}  ({vae_ch_l1})")
print(f"  GT-latent decode L1:     {gt_recon_l1:.6f}  ({gt_ch_l1})")
print(f"  DiT-sampled decode L1:   {dit_l1:.6f}  ({dit_ch_l1})")


# ── 8. Visualization ─────────────────────────────────────────────────────────
print("\n=== Generating visualization ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def max_proj(sparse_t, ch_slice, res=256):
    c = sparse_t.coords[:, 1:].long().cpu()
    feats = (sparse_t.feats[:, ch_slice].cpu() * 0.5 + 0.5).clamp(0, 1)
    C = feats.shape[1]
    projs = []
    for keep_dims in [(1, 0), (2, 0), (2, 1)]:
        img = torch.zeros(res, res, C)
        depth = torch.full((res, res), -1.0)
        mag = feats.norm(dim=-1) if C > 1 else feats[:, 0]
        for i in range(c.shape[0]):
            r, col = c[i, keep_dims[0]].item(), c[i, keep_dims[1]].item()
            if mag[i].item() > depth[r, col].item():
                depth[r, col] = mag[i].item()
                img[r, col] = feats[i]
        projs.append(img.numpy())
    return projs


x_cpu = x_emit_gt
y_vae_cpu = y_vae_only.to("cpu")
y_gt_cpu = y_gt_recon.to("cpu")
y_dit_cpu = y_dit.to("cpu")

columns = [
    ("GT Emission", x_cpu),
    ("VAE Recon", y_vae_cpu),
    ("GT-Latent Decode", y_gt_cpu),
    ("DiT Sampled (CFG)", y_dit_cpu),
]
n_cols = len(columns)
n_rows = 3
view_labels = ["XY", "XZ", "YZ"]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3 + 2))
fig.suptitle(
    f"Emission DiT Full Parity Test  |  sample: {SAMPLE[:12]}...\n"
    f"VAE L1={vae_only_l1:.4f}  |  GT-Latent L1={gt_recon_l1:.4f}  |  DiT L1={dit_l1:.4f}",
    fontsize=12, fontweight="bold", y=0.98,
)

for ci, (title, sparse_t) in enumerate(columns):
    projs = max_proj(sparse_t, slice(0, 3))
    for row in range(3):
        axes[row][ci].imshow(projs[row], origin="lower")
        axes[row][ci].set_xticks([])
        axes[row][ci].set_yticks([])
        if ci == 0:
            axes[row][ci].set_ylabel(view_labels[row], fontsize=11, fontweight="bold")
    axes[0][ci].set_title(title, fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT_PNG}")
plt.close()


# ── 9. Latent-space visualization ─────────────────────────────────────────────
print("\n=== Generating latent visualization ===")

latent_res = 16


def latent_max_proj(sparse_t, res=16):
    c = sparse_t.coords[:, 1:].long().cpu()
    feats = sparse_t.feats.cpu()
    intensity = feats.abs().mean(dim=-1)
    projs = []
    for keep_dims in [(1, 0), (2, 0), (2, 1)]:
        img = torch.zeros(res, res)
        for i in range(c.shape[0]):
            r, col = c[i, keep_dims[0]].item(), c[i, keep_dims[1]].item()
            if r < res and col < res and intensity[i].item() > img[r, col].item():
                img[r, col] = intensity[i].item()
        projs.append(img.numpy())
    return projs


def latent_channel_proj(sparse_t, ch_indices, res=16):
    c = sparse_t.coords[:, 1:].long().cpu()
    feats = sparse_t.feats[:, ch_indices].cpu()
    lo, hi = feats.min(), feats.max()
    feats_norm = (feats - lo) / (hi - lo + 1e-8)
    projs = []
    for keep_dims in [(1, 0), (2, 0), (2, 1)]:
        img = torch.zeros(res, res, len(ch_indices))
        depth = torch.full((res, res), -1.0)
        mag = feats_norm.norm(dim=-1)
        for i in range(c.shape[0]):
            r, col = c[i, keep_dims[0]].item(), c[i, keep_dims[1]].item()
            if r < res and col < res and mag[i].item() > depth[r, col].item():
                depth[r, col] = mag[i].item()
                img[r, col] = feats_norm[i]
        projs.append(img.numpy())
    return projs


z_gt_cpu = z_emit_raw.cpu()
z_dit_cpu = z_emit_sampled_raw.cpu()
z_diff = z_gt_cpu.replace((z_gt_cpu.feats - z_dit_cpu.feats).abs())

per_ch_l1 = (z_gt_cpu.feats - z_dit_cpu.feats).abs().mean(dim=0)
per_ch_corr = []
for ch in range(32):
    g = z_gt_cpu.feats[:, ch]
    d = z_dit_cpu.feats[:, ch]
    corr = torch.corrcoef(torch.stack([g, d]))[0, 1].item()
    per_ch_corr.append(corr)
per_ch_corr = np.array(per_ch_corr)

fig2 = plt.figure(figsize=(18, 14))
gs = fig2.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
view_labels = ["XY", "XZ", "YZ"]

fig2.suptitle(
    f"Emission DiT Full Parity — Latent Comparison  |  sample: {SAMPLE[:12]}...\n"
    f"Latent L1={latent_l1:.4f}  |  Mean channel corr={per_ch_corr.mean():.4f}",
    fontsize=13, fontweight="bold", y=0.99,
)

lat_columns = [
    ("GT Latent", z_gt_cpu),
    ("DiT Latent", z_dit_cpu),
    ("|Difference|", z_diff),
]
for ci, (title, st) in enumerate(lat_columns):
    projs = latent_max_proj(st, latent_res)
    for row in range(3):
        ax = fig2.add_subplot(gs[row, ci])
        im = ax.imshow(projs[row], origin="lower", cmap="inferno",
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel(view_labels[row], fontsize=11, fontweight="bold")
        if row == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for row, (title, st) in enumerate([
    ("GT ch[0:3]", z_gt_cpu),
    ("DiT ch[0:3]", z_dit_cpu),
    ("Diff ch[0:3]", z_diff),
]):
    projs = latent_channel_proj(st, [0, 1, 2], latent_res)
    ax = fig2.add_subplot(gs[row, 3])
    ax.imshow(projs[0], origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if row == 0:
        ax.set_title("ch[0,1,2] as RGB", fontsize=11, fontweight="bold")
    ax.set_ylabel(title, fontsize=9)

ax_bar = fig2.add_subplot(gs[3, :2])
ax_bar.bar(range(32), per_ch_l1.numpy(), color="steelblue", edgecolor="navy", linewidth=0.5)
ax_bar.set_xlabel("Latent channel", fontsize=10)
ax_bar.set_ylabel("L1 error", fontsize=10)
ax_bar.set_title("Per-channel L1 (GT vs DiT)", fontsize=11, fontweight="bold")
ax_bar.set_xlim(-0.5, 31.5)

ax_corr = fig2.add_subplot(gs[3, 2:])
colors = ["green" if c > 0.9 else "orange" if c > 0.5 else "red" for c in per_ch_corr]
ax_corr.bar(range(32), per_ch_corr, color=colors, edgecolor="darkgray", linewidth=0.5)
ax_corr.set_xlabel("Latent channel", fontsize=10)
ax_corr.set_ylabel("Correlation", fontsize=10)
ax_corr.set_title("Per-channel correlation (GT vs DiT)", fontsize=11, fontweight="bold")
ax_corr.set_xlim(-0.5, 31.5)
ax_corr.set_ylim(-0.1, 1.05)
ax_corr.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, linewidth=0.8)

fig2.savefig(OUT_LATENT_PNG, dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT_LATENT_PNG}")
plt.close()

print("\nDone!")
