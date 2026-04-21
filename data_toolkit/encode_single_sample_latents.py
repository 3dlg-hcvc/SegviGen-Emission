#!/usr/bin/env python3
"""
Encode emission + PBR latents for a single sample, producing .npz files
compatible with LightGenSLatEmissionLatent dataset.

Output format (per .npz):
    feats:  float32 [N_tokens, C_latent]
    coords: uint8   [N_tokens, 3]
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

torch.set_grad_enabled(False)

SAMPLE = "0007deb6d96c4474b80faa5aa3888760"
DATA_ROOT = "data/lightgen_test"
EMISSION_VAE_CKPT = "/tmp/emission_vae_optB_overfit/ckpts/stepstep=0001000.ckpt"
EMISSION_VAE_CFG = "/tmp/emission_vae_optB_overfit/config.yaml"

EMIT_LATENT_NAME = "emission_vae_optB_overfit"
PBR_LATENT_NAME = "tex_enc_next_dc_f16c32_fp16_256"

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

print(f"  Emission voxels: {emit_feats.shape}")
print(f"  PBR voxels:      {pbr_feats.shape}")

# ── 2. Encode PBR latent ─────────────────────────────────────────────────────
print("\n=== Encoding PBR latent ===")
pbr_encoder = models.from_pretrained(
    "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
).eval().float().to(device)

z_pbr = pbr_encoder(x_pbr.to(device))
print(f"  PBR latent: {z_pbr.feats.shape}")

pbr_save_dir = os.path.join(DATA_ROOT, "pbr_latents", PBR_LATENT_NAME)
os.makedirs(pbr_save_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(pbr_save_dir, f"{SAMPLE}.npz"),
    feats=z_pbr.feats.cpu().numpy().astype(np.float32),
    coords=z_pbr.coords[:, 1:].cpu().numpy().astype(np.uint8),
)
print(f"  Saved: {pbr_save_dir}/{SAMPLE}.npz")

del pbr_encoder
torch.cuda.empty_cache()

# ── 3. Encode emission latent ────────────────────────────────────────────────
print("\n=== Encoding emission latent ===")
with open(EMISSION_VAE_CFG) as f:
    cfg = yaml.safe_load(f)

from trellis2.trainers.builders import build_models
emit_models = build_models(cfg["models"])
emit_enc = emit_models["encoder"]

ckpt = torch.load(EMISSION_VAE_CKPT, map_location="cpu")
state = ckpt["state_dict"]
enc_state = {k.removeprefix("models.encoder."): v
             for k, v in state.items() if k.startswith("models.encoder.")}
emit_enc.load_state_dict(enc_state)
emit_enc.to(device).eval()

z_emit = emit_enc(x_emit.to(device))
print(f"  Emission latent: {z_emit.feats.shape}")

emit_save_dir = os.path.join(DATA_ROOT, "emission_latents", EMIT_LATENT_NAME)
os.makedirs(emit_save_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(emit_save_dir, f"{SAMPLE}.npz"),
    feats=z_emit.feats.cpu().numpy().astype(np.float32),
    coords=z_emit.coords[:, 1:].cpu().numpy().astype(np.uint8),
)
print(f"  Saved: {emit_save_dir}/{SAMPLE}.npz")

# ── 4. Verify ────────────────────────────────────────────────────────────────
print("\n=== Verification ===")
print(f"  Emission latent coords match PBR latent coords: "
      f"{torch.equal(z_emit.coords, z_pbr.coords)}")
print(f"  Emission latent shape: {z_emit.feats.shape}")
print(f"  PBR latent shape:      {z_pbr.feats.shape}")
print(f"  Num tokens:            {z_emit.feats.shape[0]}")
print("\nDone!")
