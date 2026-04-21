#!/usr/bin/env python3
"""Diagnostic: check if DiT output decode differs from GT decode."""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch, numpy as np, yaml
for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from torchvision import transforms
from PIL import Image

torch.set_grad_enabled(False)
device = "cuda"
DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
DIT_CKPT = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_dit_10sample_overfit/ckpts/last.ckpt"
DIT_CFG = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_dit_10sample_overfit/config.yaml"
VAE_CKPT = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit/ckpts/last.ckpt"
VAE_CFG = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit/config.yaml"

SHA = "00192b5a4a3249c79141c8dccaad2947"

def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))

with open(DIT_CFG) as f: dit_cfg = yaml.safe_load(f)
ds = dit_cfg["dataset"]["args"]
emit_mean = torch.tensor(ds["emission_normalization"]["mean"])
emit_std  = torch.tensor(ds["emission_normalization"]["std"])
pbr_mean  = torch.tensor(ds["pbr_normalization"]["mean"])
pbr_std   = torch.tensor(ds["pbr_normalization"]["std"])
shape_mean = torch.tensor(ds["shape_normalization"]["mean"])
shape_std  = torch.tensor(ds["shape_normalization"]["std"])

# Load DINOv3
dino_cfg = dit_cfg["trainer"]["args"]["image_cond_model"]["args"]
dino = DinoV3FeatureExtractor(**dino_cfg); dino.cuda()
dino_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def enc_img(sha):
    img = Image.open(f"{DATA_ROOT}/thumbnails/{sha}.png").convert("RGB")
    t = torch.from_numpy(np.array(img.resize((dino_cfg["image_size"],)*2, Image.LANCZOS))
        .astype(np.float32)/255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    t = dino_tr(t).to(dino.model.embeddings.patch_embeddings.weight.dtype)
    return dino.extract_features(t)

# Load DiT
dit_models = build_models(dit_cfg["models"])
dit = dit_models["denoiser"]
ckpt = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
dit_state = {k.removeprefix("models.denoiser."): v for k, v in ckpt["state_dict"].items() if k.startswith("models.denoiser.")}
dit.load_state_dict(dit_state); dit.to(device).eval()
sampler = FlowEulerCfgSampler(sigma_min=dit_cfg["trainer"]["args"]["sigma_min"])

# Load VAE
with open(VAE_CFG) as f: vae_cfg = yaml.safe_load(f)
vae_models = build_models(vae_cfg["models"])
enc, dec = vae_models["encoder"], vae_models["decoder"]
vae_ckpt = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
enc.load_state_dict({k.removeprefix("models.encoder."): v for k,v in vae_ckpt["state_dict"].items() if k.startswith("models.encoder.")})
dec.load_state_dict({k.removeprefix("models.decoder."): v for k,v in vae_ckpt["state_dict"].items() if k.startswith("models.decoder.")})
enc.to(device).eval(); dec.to(device).eval()

# GT voxels + latents
coords, attr = o_voxel.io.read_vxz(f"{DATA_ROOT}/emission_voxels_256/{SHA}.vxz", num_threads=4)
bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
x_gt = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(device)

z_emit_raw = load_latent(f"{DATA_ROOT}/emission_latents/emission_vae_10sample_overfit/{SHA}.npz")
z_pbr_raw  = load_latent(f"{DATA_ROOT}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{SHA}.npz")
z_shape_raw = load_latent(f"{DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/{SHA}.npz")

def norm(f, m, s): return (f - m.to(f.device)) / s.to(f.device)
def denorm(f, m, s): return f * s.to(f.device) + m.to(f.device)

z_emit_n = z_emit_raw.replace(norm(z_emit_raw.feats, emit_mean, emit_std)).to(device)
z_pbr_n  = z_pbr_raw.replace(norm(z_pbr_raw.feats, pbr_mean, pbr_std)).to(device)
z_shape_n = z_shape_raw.replace(norm(z_shape_raw.feats, shape_mean, shape_std)).to(device)

concat_cond = z_emit_n.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))
img_feat = enc_img(SHA)
neg_cond = torch.zeros_like(img_feat)

noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    result = sampler.sample(dit, noise, cond=img_feat, neg_cond=neg_cond,
                            concat_cond=concat_cond, steps=50, guidance_strength=3.0, verbose=False)
z_dit_n = result.samples
z_dit_raw = z_dit_n.replace(denorm(z_dit_n.feats, emit_mean.to(device), emit_std.to(device)))

# Re-encode GT
z_enc = enc(x_gt)
z_pbr_feats = z_pbr_raw.feats.to(device)

# Diagnostic prints
print(f"z_enc      : coords={z_enc.coords.shape}, feats={z_enc.feats.shape}")
print(f"z_emit_raw : coords={z_emit_raw.coords.shape}, feats={z_emit_raw.feats.shape}")
print(f"z_dit_raw  : coords={z_dit_raw.coords.shape}, feats={z_dit_raw.feats.shape}")
print(f"z_pbr_raw  : coords={z_pbr_raw.coords.shape}, feats={z_pbr_raw.feats.shape}")

# Check coord equality
print(f"\nz_enc.coords == z_emit_raw.coords: {torch.equal(z_enc.coords[:,1:].cpu(), z_emit_raw.coords[:,1:].cpu())}")
print(f"z_enc.coords == z_dit_raw.coords:  {torch.equal(z_enc.coords[:,1:].cpu(), z_dit_raw.coords[:,1:].cpu())}")

# Check feats differences
print(f"\n|z_enc.feats - z_emit_raw.feats| mean = {(z_enc.feats.cpu() - z_emit_raw.feats.cpu()).abs().mean():.4f}")
print(f"|z_dit_raw.feats - z_emit_raw.feats| mean = {(z_dit_raw.feats.cpu() - z_emit_raw.feats.cpu()).abs().mean():.4f}")

# Decode with GT emit vs DiT emit
z_gt_dec  = dec(z_enc.replace(torch.cat([z_emit_raw.feats.to(device), z_pbr_feats], dim=-1)))
z_dit_dec = dec(z_enc.replace(torch.cat([z_dit_raw.feats, z_pbr_feats], dim=-1)))

print(f"\nz_gt_dec.feats  : shape={z_gt_dec.feats.shape}")
print(f"z_dit_dec.feats : shape={z_dit_dec.feats.shape}")
print(f"|z_gt_dec - z_dit_dec| mean = {(z_gt_dec.feats - z_dit_dec.feats).abs().mean():.6f}")
print(f"|z_gt_dec - z_dit_dec| max  = {(z_gt_dec.feats - z_dit_dec.feats).abs().max():.6f}")
print(f"\nvae_l1 (x_gt vs z_gt_dec)  = {(x_gt.feats - z_gt_dec.feats).abs().mean():.6f}")
print(f"dit_l1 (x_gt vs z_dit_dec) = {(x_gt.feats - z_dit_dec.feats).abs().mean():.6f}")

# Sanity: also decode with a totally different emission (random)
z_rand = torch.randn_like(z_emit_raw.feats).to(device) * 2.0
z_rand_dec = dec(z_enc.replace(torch.cat([z_rand, z_pbr_feats], dim=-1)))
print(f"\n|z_gt_dec - z_rand_dec| mean (should be large) = {(z_gt_dec.feats - z_rand_dec.feats).abs().mean():.6f}")
