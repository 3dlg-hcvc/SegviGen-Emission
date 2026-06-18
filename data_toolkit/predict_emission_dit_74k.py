#!/usr/bin/env python3
"""Quick prediction viz for the (possibly partially-trained) 74k emission DiT.

For a few samples: encode the GT emission voxels (this populates the VAE's
subdivision cache, which the decoder needs), then for each sample render
  [ input image | GT (VAE recon) | DiT prediction ]
where the DiT prediction is a CFG flow-matching sample conditioned on the DINOv3
image feature + the shape latent (skip_pbr_cond concat), decoded through the same
encoder structure. Per-sample L1 vs the VAE-recon target is printed.

Needs raw emission voxels (DATA_ROOT/emission_voxels_256/{sha}.vxz) + latents +
thumbnails. Arg-driven; defaults point at the fir paths.
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
import o_voxel

import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from torchvision import transforms

torch.set_grad_enabled(False)
DEVICE = "cuda"
PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA = -1.0, 1.0, 1.0


def p(*a): print(*a, flush=True)
def normalize(f, m, s):   return (f - m.to(f.device)) / s.to(f.device)
def denormalize(f, m, s): return f * s.to(f.device) + m.to(f.device)


def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))


def pack_emission_as_pbr(x):
    emit = x.feats[:, :3]
    pad = torch.tensor([[PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA]],
                       device=emit.device, dtype=emit.dtype).expand(emit.shape[0], -1)
    return x.replace(torch.cat([emit, pad], dim=-1))


def max_proj_rgb(sparse_t, res=256):
    c = sparse_t.coords[:, 1:].long().cpu()
    feats = (sparse_t.feats[:, :3].cpu() * 0.5 + 0.5).clamp(0, 1)
    img = torch.zeros(res, res, 3); depth = torch.full((res, res), -1.0)
    mag = feats.norm(dim=-1)
    for i in range(c.shape[0]):
        r, col = c[i, 1].item(), c[i, 0].item()
        if 0 <= r < res and 0 <= col < res and mag[i].item() > depth[r, col].item():
            depth[r, col] = mag[i].item(); img[r, col] = feats[i]
    return img.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit_ckpt", required=True)
    ap.add_argument("--dit_cfg", default=None)
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--vae_cfg", default=None)
    ap.add_argument("--data_root", default="/scratch/dya78/lightgen/lightgen_74k")
    ap.add_argument("--emission_latent_name", default="emission_pbr_vae_ft_74k")
    ap.add_argument("--shape_latent_name", default="shape_enc_next_dc_f16c32_fp16_256")
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--thumbnail_dir", default=None)
    ap.add_argument("--shas", default=None, help="comma-separated shas (overrides split/parquet)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="emission_dit_74k_pred.png")
    args = ap.parse_args()

    dit_cfg_path = args.dit_cfg or os.path.join(os.path.dirname(args.dit_ckpt), "..", "config.yaml")
    vae_cfg_path = args.vae_cfg or os.path.join(os.path.dirname(args.vae_ckpt), "..", "config.yaml")
    split_json = args.split_json or os.path.join(args.data_root, "data_splits_emissive_74k_stratified_voxonly.json")
    parquet = args.parquet or os.path.join(args.data_root, "df_SomgProc_final.parquet")
    thumb_dir = args.thumbnail_dir or os.path.join(args.data_root, "thumbnails")
    vox_dir = os.path.join(args.data_root, f"emission_voxels_{args.resolution}")

    # ---- pick samples ----
    if args.shas:
        sel = [s for s in args.shas.split(",") if s][: args.n]
    else:
        import pandas as pd
        df = pd.read_parquet(parquet, columns=["success"])
        df = df[df["success"] == True] if "success" in df.columns else df
        shas = [str(x) for x in df.index]
        sp_ = json.load(open(split_json))[args.split]
        idx = sp_["indices"] if isinstance(sp_, dict) else sp_
        sel = [shas[i] for i in idx[: args.n]]
    p(f"Samples: {len(sel)}")

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

    # ---- VAE encoder + decoder ----
    p(f"Loading VAE: {args.vae_ckpt}")
    vae = build_models(yaml.safe_load(open(vae_cfg_path))["models"])
    enc, dec = vae["encoder"], vae["decoder"]
    vck = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    enc.load_state_dict({k.removeprefix("models.encoder."): v for k, v in vck.items() if k.startswith("models.encoder.")})
    dec.load_state_dict({k.removeprefix("models.decoder."): v for k, v in vck.items() if k.startswith("models.decoder.")})
    enc.to(DEVICE).eval(); dec.to(DEVICE).eval()

    rows = []
    for sha in sel:
        p(f"  {sha[:16]}...")
        # encode GT voxels -> z_enc (populates the decoder's subdivision cache)
        coords, attr = o_voxel.io.read_vxz(os.path.join(vox_dir, f"{sha}.vxz"), num_threads=4)
        bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
        x_gt = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(DEVICE)
        z_enc = enc(pack_emission_as_pbr(x_gt))

        z_shape = load_latent(os.path.join(args.data_root, "shape_latents", args.shape_latent_name, f"{sha}.npz"))
        z_emit_n = z_enc.replace(normalize(z_enc.feats, emit_mean, emit_std))
        z_shape_n = z_enc.replace(normalize(z_shape.feats.to(DEVICE), shape_mean, shape_std))
        concat_cond = z_emit_n.replace(z_shape_n.feats)
        _, img_feat = encode_image(sha); neg = torch.zeros_like(img_feat)
        noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = sampler.sample(dit, noise, cond=img_feat, neg_cond=neg, concat_cond=concat_cond,
                                 steps=args.steps, guidance_strength=args.guidance, verbose=False)
        z_dit = z_enc.replace(denormalize(res.samples.feats, emit_mean.to(DEVICE), emit_std.to(DEVICE)))

        gt_dec = dec(z_enc)                 # GT VAE recon (z_enc has GT feats + cache)
        dit_dec = dec(z_dit)                # DiT prediction (same coords/cache, predicted feats)
        l1 = (gt_dec.feats[:, :3] - dit_dec.feats[:, :3]).abs().mean().item()
        pil, _ = encode_image(sha)
        rows.append((sha, np.array(pil.resize((256, 256))) / 255.0,
                     max_proj_rgb(gt_dec.replace(gt_dec.feats[:, :3])),
                     max_proj_rgb(dit_dec.replace(dit_dec.feats[:, :3])), l1))
        p(f"    L1(GT_recon, DiT)={l1:.4f}")
        torch.cuda.empty_cache()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = len(rows)
    fig, axes = plt.subplots(N, 3, figsize=(9, N * 3 + 0.5), squeeze=False)
    fig.suptitle(f"Emission DiT 74k — {os.path.basename(args.dit_ckpt)}  (CFG={args.guidance}, steps={args.steps})\n"
                 f"cols: input image | GT (VAE recon) | DiT prediction", fontsize=11, fontweight="bold")
    for i, (sha, im, gt, pred, l1) in enumerate(rows):
        axes[i][0].imshow(im); axes[i][0].set_title(f"{sha[:12]} input", fontsize=8)
        axes[i][1].imshow(gt, origin="lower"); axes[i][1].set_title("GT (VAE recon)", fontsize=8)
        axes[i][2].imshow(pred, origin="lower"); axes[i][2].set_title(f"DiT pred (L1={l1:.3f})", fontsize=8)
        for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    p(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
