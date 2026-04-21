#!/usr/bin/env python3
"""Check emission voxel distribution + VAE reconstruction on one sample."""
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

torch.set_grad_enabled(False)
device = "cuda"
DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
VAE_CKPT = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit/ckpts/last.ckpt"
VAE_CFG = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs/emission_vae_10sample_overfit/config.yaml"

with open(VAE_CFG) as f: vae_cfg = yaml.safe_load(f)
vae_models = build_models(vae_cfg["models"])
enc, dec = vae_models["encoder"], vae_models["decoder"]
ckpt = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
enc.load_state_dict({k.removeprefix("models.encoder."): v for k,v in ckpt["state_dict"].items() if k.startswith("models.encoder.")})
dec.load_state_dict({k.removeprefix("models.decoder."): v for k,v in ckpt["state_dict"].items() if k.startswith("models.decoder.")})
enc.to(device).eval(); dec.to(device).eval()

def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))

import pandas as pd
df = pd.read_parquet(f'{DATA_ROOT}/df_10samples.parquet')
SAMPLES = list(df["sha256"] if "sha256" in df.columns else df.index)[:10]

print(f"{'SHA':<22} {'N_vox':>6} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'frac>0':>7} {'vae_L1':>8}")
print("-" * 85)

for sha in SAMPLES:
    coords, attr = o_voxel.io.read_vxz(f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0  # [-1, 1]
    x_gt = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(device)

    z_pbr = load_latent(f"{DATA_ROOT}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{sha}.npz")

    z_enc = enc(x_gt)
    z_cat = torch.cat([z_enc.feats, z_pbr.feats.to(device)], dim=-1)
    x_rec = dec(z_enc.replace(z_cat))

    # GT >= -0.9 means emissive > ~13/255 (non-trivial emission)
    frac_emit = (gt_feats.max(dim=-1).values > -0.5).float().mean().item()
    vae_l1 = (x_gt.feats - x_rec.feats).abs().mean().item()

    print(f"{sha[:20]:<22} {coords.shape[0]:>6} {gt_feats.mean():>7.3f} {gt_feats.std():>7.3f} "
          f"{gt_feats.min():>7.3f} {gt_feats.max():>7.3f} {frac_emit:>7.2%} {vae_l1:>8.4f}")

    # Check: what L1 does the "output -1" (all dark) baseline give?
    all_minus1 = torch.full_like(x_gt.feats, -1.0)
    zero_l1 = (x_gt.feats - all_minus1).abs().mean().item()
    print(f"  ▸ L1 if decoder always outputs -1.0 (all dark): {zero_l1:.4f}")
