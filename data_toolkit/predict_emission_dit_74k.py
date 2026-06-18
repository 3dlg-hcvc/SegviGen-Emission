#!/usr/bin/env python3
"""Quick prediction viz for the (possibly partially-trained) 74k emission DiT.

Latents-only — no raw voxels needed. For a few val samples:
  - DiT-sample an emission latent (CFG, conditioned on the DINOv3 image feature +
    the shape latent via skip_pbr_cond concat),
  - decode the DiT sample AND the GT emission latent through the VAE decoder,
  - save a max-projection PNG: [input image | GT (VAE-decode of GT latent) | DiT pred].

The GT column is the VAE reconstruction of the ground-truth emission (decode of the
precomputed GT latent) — i.e. the target the DiT is trying to match — so this shows
"how close is the DiT to the achievable target" without touching raw voxels.

Usage (defaults point at the fir paths):
  python data_toolkit/predict_emission_dit_74k.py \
    --dit_ckpt outputs/emission_dit_74k/ckpts/step0018000-0.3872.ckpt \
    --vae_ckpt outputs/emission_pbr_vae_ft_74k/ckpts/step0016816-0.0095.ckpt \
    --out emission_dit_74k_pred.png
"""
from __future__ import annotations
import argparse, json, os, sys, types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml
from PIL import Image

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
import o_voxel  # noqa: F401  (kept for parity / sparse io registration)

import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from torchvision import transforms

torch.set_grad_enabled(False)
DEVICE = "cuda"


def p(*a):  # flush-through logger
    print(*a, flush=True)


def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))


def normalize(f, m, s):   return (f - m.to(f.device)) / s.to(f.device)
def denormalize(f, m, s): return f * s.to(f.device) + m.to(f.device)


def max_proj_rgb(sparse_t, res=256):
    c = sparse_t.coords[:, 1:].long().cpu()
    feats = (sparse_t.feats[:, :3].cpu() * 0.5 + 0.5).clamp(0, 1)
    img = torch.zeros(res, res, 3)
    depth = torch.full((res, res), -1.0)
    mag = feats.norm(dim=-1)
    for i in range(c.shape[0]):
        r, col = c[i, 1].item(), c[i, 0].item()
        if 0 <= r < res and 0 <= col < res and mag[i].item() > depth[r, col].item():
            depth[r, col] = mag[i].item()
            img[r, col] = feats[i]
    return img.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit_ckpt", required=True)
    ap.add_argument("--dit_cfg", default=None, help="defaults to <dit_ckpt>/../../config.yaml")
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--vae_cfg", default=None, help="defaults to <vae_ckpt>/../../config.yaml")
    ap.add_argument("--data_root", default="/scratch/dya78/lightgen/lightgen_74k")
    ap.add_argument("--emission_latent_name", default="emission_pbr_vae_ft_74k")
    ap.add_argument("--shape_latent_name", default="shape_enc_next_dc_f16c32_fp16_256")
    ap.add_argument("--split_json", default=None, help="default <data_root>/data_splits_emissive_74k_stratified_voxonly.json")
    ap.add_argument("--parquet", default=None, help="default <data_root>/df_SomgProc_final.parquet")
    ap.add_argument("--thumbnail_dir", default=None, help="default <data_root>/thumbnails")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="emission_dit_74k_pred.png")
    args = ap.parse_args()

    dit_cfg_path = args.dit_cfg or os.path.join(os.path.dirname(args.dit_ckpt), "..", "config.yaml")
    vae_cfg_path = args.vae_cfg or os.path.join(os.path.dirname(args.vae_ckpt), "..", "config.yaml")
    split_json = args.split_json or os.path.join(args.data_root, "data_splits_emissive_74k_stratified_voxonly.json")
    parquet = args.parquet or os.path.join(args.data_root, "df_SomgProc_final.parquet")
    thumb_dir = args.thumbnail_dir or os.path.join(args.data_root, "thumbnails")

    # ---- pick samples ----
    import pandas as pd
    df = pd.read_parquet(parquet, columns=["success"])
    df = df[df["success"] == True] if "success" in df.columns else df
    shas = [str(x) for x in df.index]
    split = json.load(open(split_json))
    idx = split[args.split]["indices"] if isinstance(split[args.split], dict) else split[args.split]
    sel = [shas[i] for i in idx[: args.n]]
    p(f"Samples: {len(sel)} from split={args.split!r}")

    # ---- configs / norm ----
    dit_cfg = yaml.safe_load(open(dit_cfg_path))
    ds = dit_cfg["dataset"]["args"]
    emit_mean = torch.tensor(ds["emission_normalization"]["mean"]); emit_std = torch.tensor(ds["emission_normalization"]["std"])
    shape_mean = torch.tensor(ds["shape_normalization"]["mean"]);   shape_std = torch.tensor(ds["shape_normalization"]["std"])

    # ---- DINOv3 ----
    p("Loading DINOv3...")
    dino_cfg = dit_cfg["trainer"]["args"]["image_cond_model"]["args"]
    dino = DinoV3FeatureExtractor(**dino_cfg); dino.cuda()
    dino_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def encode_image(sha):
        img = Image.open(os.path.join(thumb_dir, f"{sha}.png")).convert("RGB")
        t = torch.from_numpy(np.array(img.resize((dino_cfg["image_size"],) * 2, Image.LANCZOS)).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        t = dino_tr(t).to(dino.model.embeddings.patch_embeddings.weight.dtype)
        return img, dino.extract_features(t)

    # ---- DiT ----
    p(f"Loading DiT: {args.dit_ckpt}")
    dit = build_models(dit_cfg["models"])["denoiser"]
    ck = torch.load(args.dit_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    dit.load_state_dict({k.removeprefix("models.denoiser."): v for k, v in ck.items() if k.startswith("models.denoiser.")})
    dit.to(DEVICE).eval()
    sampler = FlowEulerCfgSampler(sigma_min=dit_cfg["trainer"]["args"]["sigma_min"])

    # ---- VAE decoder ----
    p(f"Loading VAE decoder: {args.vae_ckpt}")
    vae_cfg = yaml.safe_load(open(vae_cfg_path))
    dec = build_models(vae_cfg["models"])["decoder"]
    vck = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    dec.load_state_dict({k.removeprefix("models.decoder."): v for k, v in vck.items() if k.startswith("models.decoder.")})
    dec.to(DEVICE).eval()

    # ---- per-sample ----
    rows = []
    for sha in sel:
        p(f"  {sha[:16]}...", )
        z_emit = load_latent(os.path.join(args.data_root, "emission_latents", args.emission_latent_name, f"{sha}.npz"))
        z_shape = load_latent(os.path.join(args.data_root, "shape_latents", args.shape_latent_name, f"{sha}.npz"))
        z_emit_n = z_emit.replace(normalize(z_emit.feats, emit_mean, emit_std)).to(DEVICE)
        z_shape_n = z_shape.replace(normalize(z_shape.feats, shape_mean, shape_std)).to(DEVICE)
        concat_cond = z_emit_n.replace(z_shape_n.feats)  # skip_pbr_cond: shape-only
        pil_img, img_feat = encode_image(sha)
        neg = torch.zeros_like(img_feat)
        noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = sampler.sample(dit, noise, cond=img_feat, neg_cond=neg, concat_cond=concat_cond,
                                 steps=args.steps, guidance_strength=args.guidance, verbose=False)
        z_dit = res.samples.replace(denormalize(res.samples.feats, emit_mean.to(DEVICE), emit_std.to(DEVICE)))
        gt_dec = dec(z_emit.to(DEVICE))           # VAE-recon of GT (the target)
        dit_dec = dec(z_dit)                      # DiT prediction
        l1 = (gt_dec.feats[:, :3] - dit_dec.feats[:, :3]).abs().mean().item()
        rows.append((sha, np.array(pil_img.resize((256, 256))) / 255.0,
                     max_proj_rgb(gt_dec.replace(gt_dec.feats[:, :3])),
                     max_proj_rgb(dit_dec.replace(dit_dec.feats[:, :3])), l1))
        p(f"    L1(GT_recon, DiT)={l1:.4f}")
        torch.cuda.empty_cache()

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = len(rows)
    fig, axes = plt.subplots(N, 3, figsize=(9, N * 3 + 0.5), squeeze=False)
    fig.suptitle(f"Emission DiT 74k — {os.path.basename(args.dit_ckpt)}  (CFG={args.guidance}, steps={args.steps})\n"
                 f"cols: input image | GT (VAE-decode) | DiT prediction", fontsize=11, fontweight="bold")
    for i, (sha, im, gt, pred, l1) in enumerate(rows):
        axes[i][0].imshow(im); axes[i][0].set_title(f"{sha[:12]} input", fontsize=8)
        axes[i][1].imshow(gt, origin="lower"); axes[i][1].set_title("GT (VAE-decode)", fontsize=8)
        axes[i][2].imshow(pred, origin="lower"); axes[i][2].set_title(f"DiT pred (L1={l1:.3f})", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    p(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
