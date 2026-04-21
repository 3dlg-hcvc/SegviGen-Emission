#!/usr/bin/env python3
"""
Encode 10-sample emission voxels into latents using the *finetuned PBR VAE*.

Packs emission RGB into the PBR 6-channel format (same as training),
then saves 32-ch latents to emission_latents/emission_pbr_finetune_10sample/.
Also prints the normalization statistics for the DiT config.

Usage:
    conda run -n trellis2 python data_toolkit/encode_emission_pbrfinetune_10samples.py
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch, numpy as np, yaml, pandas as pd

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models

torch.set_grad_enabled(False)

DATA_ROOT   = "/local-scratch/localhome/dya78/lightgen_test"
VAE_CKPT    = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_pbr_finetune_10sample/ckpts/last.ckpt"
VAE_CFG     = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_pbr_finetune_10sample/config.yaml"
LATENT_NAME = "emission_pbr_finetune_10sample"
RESOLUTION  = 256

PARQUET   = os.path.join(DATA_ROOT, "df_10samples.parquet")
VOXEL_DIR = os.path.join(DATA_ROOT, f"emission_voxels_{RESOLUTION}")
SAVE_DIR  = os.path.join(DATA_ROOT, "emission_latents", LATENT_NAME)

# Padding used by EmissionPbrFinetuneTrainer
PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA = -1.0, 1.0, 1.0

device = "cuda"


def pack_emission_as_pbr(x: sp.SparseTensor) -> sp.SparseTensor:
    """Pack emission (3ch) into PBR 6-channel: [emit_rgb, -1, 1, 1]."""
    emit = x.feats[:, :3]
    n = emit.shape[0]
    pad = torch.tensor([[PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA]],
                       device=emit.device, dtype=emit.dtype).expand(n, -1)
    return x.replace(torch.cat([emit, pad], dim=-1))


print("Loading finetuned PBR VAE encoder...")
with open(VAE_CFG) as f:
    vae_cfg = yaml.safe_load(f)
vae_models = build_models(vae_cfg["models"])
encoder = vae_models["encoder"]
ckpt = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
enc_state = {k.removeprefix("models.encoder."): v
             for k, v in ckpt["state_dict"].items() if k.startswith("models.encoder.")}
encoder.load_state_dict(enc_state)
encoder.to(device).eval()
print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")

df = pd.read_parquet(PARQUET)
sha_list = list(df["sha256"] if "sha256" in df.columns else df.index)
print(f"  Samples: {len(sha_list)}")

os.makedirs(SAVE_DIR, exist_ok=True)
all_feats = []

for sha in sha_list:
    vxz_path = os.path.join(VOXEL_DIR, f"{sha}.vxz")
    if not os.path.exists(vxz_path):
        print(f"  [skip] {sha}: vxz not found")
        continue

    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
    emit_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    x = sp.SparseTensor(emit_feats, torch.cat([bc, coords], dim=-1)).to(device)

    x_packed = pack_emission_as_pbr(x)  # 6-channel
    z = encoder(x_packed)
    torch.cuda.synchronize()

    z_feats  = z.feats.cpu().numpy().astype(np.float32)
    z_coords = z.coords[:, 1:].cpu().numpy().astype(np.uint8)

    np.savez_compressed(os.path.join(SAVE_DIR, f"{sha}.npz"),
                        feats=z_feats, coords=z_coords)
    all_feats.append(z.feats.cpu())
    print(f"  {sha[:20]}... → {z_feats.shape[0]} tokens, {z_feats.shape[1]}ch")

print(f"\nSaved {len(all_feats)} latent files to {SAVE_DIR}")

# Normalization stats
all_f = torch.cat(all_feats, dim=0)
mean, std = all_f.mean(dim=0), all_f.std(dim=0)
print("\n=== Emission normalization stats (copy into DiT config) ===")
print("    emission_normalization:")
print(f"      mean: [{', '.join(f'{v:.6f}' for v in mean.tolist())}]")
print(f"      std: [{', '.join(f'{v:.6f}' for v in std.tolist())}]")
print(f"\n  Total tokens: {all_f.shape[0]}, channels: {all_f.shape[1]}")
print(f"  Feat range: [{all_f.min():.4f}, {all_f.max():.4f}]")
