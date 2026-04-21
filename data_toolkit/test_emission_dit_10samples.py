#!/usr/bin/env python3
"""
Test the 10-sample emission DiT: sample all 10 training samples, decode,
and compare against GT. Produces a summary grid.
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
from torchvision import transforms

torch.set_grad_enabled(False)

DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
DIT_CKPT  = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_dit_10sample_overfit/ckpts/last.ckpt"
DIT_CFG   = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_dit_10sample_overfit/config.yaml"
EMIT_VAE_CKPT = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit/ckpts/last.ckpt"
EMIT_VAE_CFG  = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit/config.yaml"
OUT_PNG   = os.path.join(os.path.dirname(__file__), "..", "emission_dit_10sample_test.png")
device = "cuda"

import pandas as pd
df = pd.read_parquet(f'{DATA_ROOT}/df_10samples.parquet')
SAMPLES = list(df["sha256"] if "sha256" in df.columns else df.index)[:10]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))

def normalize(feats, mean, std):
    return (feats - mean.to(feats.device)) / std.to(feats.device)

def denormalize(feats, mean, std):
    return feats * std.to(feats.device) + mean.to(feats.device)

def max_proj_rgb(sparse_t, res=256):
    c = sparse_t.coords[:, 1:].long().cpu()
    feats = (sparse_t.feats[:, :3].cpu() * 0.5 + 0.5).clamp(0, 1)
    img = torch.zeros(res, res, 3)
    depth = torch.full((res, res), -1.0)
    mag = feats.norm(dim=-1)
    for i in range(c.shape[0]):
        r, col = c[i, 1].item(), c[i, 0].item()
        if r < res and col < res and mag[i].item() > depth[r, col].item():
            depth[r, col] = mag[i].item()
            img[r, col] = feats[i]
    return img.numpy()


# ── Load normalization stats ──────────────────────────────────────────────────
with open(DIT_CFG) as f:
    dit_cfg = yaml.safe_load(f)
ds_args = dit_cfg["dataset"]["args"]

def _load_norm(key):
    n = ds_args.get(key)
    if n is None: return None, None
    return torch.tensor(n["mean"]), torch.tensor(n["std"])

emit_mean, emit_std = _load_norm("emission_normalization")
pbr_mean, pbr_std   = _load_norm("pbr_normalization")
shape_mean, shape_std = _load_norm("shape_normalization")


# ── Load DINOv3 ──────────────────────────────────────────────────────────────
print("Loading DINOv3...")
dino_cfg = dit_cfg["trainer"]["args"]["image_cond_model"]["args"]
dino = DinoV3FeatureExtractor(**dino_cfg)
dino.cuda()
dino_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def encode_image(sha):
    img = Image.open(f"{DATA_ROOT}/thumbnails/{sha}.png").convert("RGB")
    t = torch.from_numpy(np.array(img.resize((dino_cfg["image_size"],)*2, Image.LANCZOS))
        .astype(np.float32)/255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    t = dino_transform(t).to(dino.model.embeddings.patch_embeddings.weight.dtype)
    return dino.extract_features(t)


# ── Load DiT ─────────────────────────────────────────────────────────────────
print("Loading DiT...")
dit_models = build_models(dit_cfg["models"])
dit = dit_models["denoiser"]
ckpt = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
dit_state = {k.removeprefix("models.denoiser."): v
             for k, v in ckpt["state_dict"].items() if k.startswith("models.denoiser.")}
dit.load_state_dict(dit_state)
dit.to(device).eval()
print(f"  DiT: {sum(p.numel() for p in dit.parameters()):,} params")

sigma_min = dit_cfg["trainer"]["args"]["sigma_min"]
sampler = FlowEulerCfgSampler(sigma_min=sigma_min)


# ── Load VAE ─────────────────────────────────────────────────────────────────
print("Loading VAE...")
with open(EMIT_VAE_CFG) as f:
    vae_cfg = yaml.safe_load(f)
vae_models = build_models(vae_cfg["models"])
emit_enc = vae_models["encoder"]
emit_dec = vae_models["decoder"]
vae_ckpt = torch.load(EMIT_VAE_CKPT, map_location="cpu", weights_only=False)
vae_state = vae_ckpt["state_dict"]
emit_enc.load_state_dict({k.removeprefix("models.encoder."): v
    for k,v in vae_state.items() if k.startswith("models.encoder.")})
emit_dec.load_state_dict({k.removeprefix("models.decoder."): v
    for k,v in vae_state.items() if k.startswith("models.decoder.")})
emit_enc.to(device).eval()
emit_dec.to(device).eval()


# ── Per-sample inference ──────────────────────────────────────────────────────
print("\nRunning per-sample inference...")
results = []

for sha in SAMPLES:
    print(f"  {sha[:20]}...", end=" ", flush=True)

    # Load GT voxels
    coords, attr = o_voxel.io.read_vxz(
        f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
    x_gt = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(device)

    # Load latents
    z_emit_raw  = load_latent(f"{DATA_ROOT}/emission_latents/emission_vae_10sample_overfit/{sha}.npz")
    z_pbr_raw   = load_latent(f"{DATA_ROOT}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{sha}.npz")
    z_shape_raw = load_latent(f"{DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/{sha}.npz")

    z_emit_n  = z_emit_raw.replace(normalize(z_emit_raw.feats, emit_mean, emit_std)).to(device)
    z_pbr_n   = z_pbr_raw.replace(normalize(z_pbr_raw.feats, pbr_mean, pbr_std)).to(device)
    z_shape_n = z_shape_raw.replace(normalize(z_shape_raw.feats, shape_mean, shape_std)).to(device)

    concat_cond = z_emit_n.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))
    img_feat = encode_image(sha)
    neg_cond = torch.zeros_like(img_feat)

    # Sample
    noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = sampler.sample(dit, noise,
            cond=img_feat, neg_cond=neg_cond, concat_cond=concat_cond,
            steps=50, guidance_strength=3.0, verbose=False)
    z_dit_n = result.samples
    z_dit_raw = z_dit_n.replace(denormalize(z_dit_n.feats, emit_mean.to(device), emit_std.to(device)))

    # Decode
    z_enc = emit_enc(x_gt)
    z_pbr_feats = z_pbr_raw.feats.to(device)
    z_gt_dec  = emit_dec(z_enc.replace(torch.cat([z_emit_raw.feats.to(device), z_pbr_feats], dim=-1)))
    z_dit_dec = emit_dec(z_enc.replace(torch.cat([z_dit_raw.feats, z_pbr_feats], dim=-1)))

    vae_l1 = (x_gt.feats - z_gt_dec.feats).abs().mean().item()
    dit_l1 = (x_gt.feats - z_dit_dec.feats).abs().mean().item()
    lat_l1 = (z_emit_raw.feats.to(device) - z_dit_raw.feats).abs().mean().item()

    results.append({
        "sha": sha,
        "gt":  x_gt.cpu(),
        "dit": z_dit_dec.cpu(),
        "vae_l1": vae_l1,
        "dit_l1": dit_l1,
        "lat_l1": lat_l1,
    })
    print(f"VAE L1={vae_l1:.4f}  DiT L1={dit_l1:.4f}  Lat L1={lat_l1:.4f}")

    torch.cuda.empty_cache()


# ── Visualization ─────────────────────────────────────────────────────────────
print("\nGenerating visualization...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = len(results)
fig, axes = plt.subplots(N, 3, figsize=(9, N * 2.5 + 1))
fig.suptitle("Emission DiT — 10-sample overfit test\n"
             "Columns: GT | DiT decode | Latent L1 per channel",
             fontsize=12, fontweight="bold")

for i, r in enumerate(results):
    sha_short = r["sha"][:12] + "..."
    gt_img  = max_proj_rgb(r["gt"],  res=256)
    dit_img = max_proj_rgb(r["dit"], res=256)

    axes[i][0].imshow(gt_img,  origin="lower")
    axes[i][0].set_title(f"{sha_short} GT", fontsize=7)
    axes[i][1].imshow(dit_img, origin="lower")
    axes[i][1].set_title(f"DiT (L1={r['dit_l1']:.3f})", fontsize=7)

    # Per-channel latent L1
    z_gt_feats  = load_latent(f"{DATA_ROOT}/emission_latents/emission_vae_10sample_overfit/{r['sha']}.npz").feats
    z_dit_feats = results[i]["dit"].feats   # decoded, not latent space — use lat_l1 from raw latents

    # Re-load cached emit lat for per-channel breakdown
    z_e = np.load(f"{DATA_ROOT}/emission_latents/emission_vae_10sample_overfit/{r['sha']}.npz")["feats"]
    # We stored z_dit_raw implicitly; approximate per-channel from overall lat_l1
    axes[i][2].bar(range(3), [
        (r["gt"].feats[:, ch] - r["dit"].feats[:, ch]).abs().mean().item()
        for ch in range(3)
    ], color=["red", "green", "blue"])
    axes[i][2].set_title(f"RGB L1 (VAE base={r['vae_l1']:.3f})", fontsize=7)
    axes[i][2].set_xticks([0, 1, 2])
    axes[i][2].set_xticklabels(["R", "G", "B"], fontsize=8)
    axes[i][2].set_ylim(0, max(0.02, r["dit_l1"] * 1.3))

    for ax in axes[i]:
        ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
plt.close()

print("\n=== Summary ===")
print(f"{'SHA':<24} {'VAE L1':>8} {'DiT L1':>8} {'Lat L1':>8}")
print("-" * 52)
for r in results:
    print(f"{r['sha'][:22]}  {r['vae_l1']:>8.4f} {r['dit_l1']:>8.4f} {r['lat_l1']:>8.4f}")

print("\nDone!")
