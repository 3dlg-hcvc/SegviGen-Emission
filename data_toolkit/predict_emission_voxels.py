#!/usr/bin/env python3
"""Emission DiT inference: runs the emission DiT and saves per-shape predicted
emission voxels (coords, pred, gt_recon, gt_emission) as npz files under
<out_dir>/pred_voxels/. With --gt_only, saves only coords+gt_emission (CPU,
no model or DiT config needed).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, types

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


def sample_seed(sha: str, seed: int) -> int:
    """Per-shape RNG seed for `(sha, seed)`. Order-, batch- and skip-independent.

    VERBATIM COPY of evaluation/newdata_eval/seedutil.py in the parent lightgen repo. Duplicated
    because this submodule runs on cluster nodes where the parent is not checked out. Keep the
    copies byte-identical. The value is the FULL unsigned 64-bit prefix, unmasked -- masking to
    int63 gives a different RNG stream for the same nominal seed.
    """
    return int.from_bytes(hashlib.sha256(f"{sha}:{seed}".encode()).digest()[:8], "big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit_ckpt", required=True)
    ap.add_argument("--dit_cfg", default=None)
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--vae_cfg", default=None)
    ap.add_argument("--data_root", default="/scratch/dya78/lightgen/lightgen_74k")
    ap.add_argument("--emission_latent_name", default="emission_pbr_vae_ft_74k")
    ap.add_argument("--shape_latent_name", default="shape_enc_next_dc_f16c32_fp16_256")
    ap.add_argument("--pbr_latent_name", default="tex_enc_next_dc_f16c32_fp16_256")
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--thumbnail_dir", default=None)
    ap.add_argument("--shas", default=None, help="comma-separated shas (overrides split/parquet)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=str, default="6")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="outputs/emission_dit_74k/eval")
    ap.add_argument("--gt_only", action="store_true",
                    help="Only read GT voxels and save coords+gt_emission (no model, CPU).")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed. Per-shape noise is derived from sha256(sha:seed), so "
                         "it is independent of --shas order, batching and skipped shapes.")
    ap.add_argument("--pred_only", action="store_true",
                    help="write only coords+pred, skipping the gt_recon decode. Use for seeds "
                         "01..04: gt_recon/gt_emission are noise-independent, so seed00's copy "
                         "is the one the ceiling rows are scored from. Do NOT use for seed00 -- "
                         "build_model_gallery.py and the UV viz/bridge tools index those keys "
                         "with no guard and raise KeyError.")
    args = ap.parse_args()

    dit_cfg_path = args.dit_cfg or os.path.join(os.path.dirname(args.dit_ckpt), "..", "config.yaml")
    vae_cfg_path = args.vae_cfg or os.path.join(os.path.dirname(args.vae_ckpt), "..", "config.yaml")
    split_json = args.split_json or os.path.join(args.data_root, "data_splits_emissive_74k_stratified_voxonly.json")
    parquet = args.parquet or os.path.join(args.data_root, "df_SomgProc_final.parquet")
    thumb_dir = args.thumbnail_dir or os.path.join(args.data_root, "thumbnails")
    vox_dir = os.path.join(args.data_root, f"emission_voxels_{args.resolution}")

    # ---- pick samples ----
    if args.shas:
        shas_list = [s for s in args.shas.split(",") if s]
        n_sel = len(shas_list) if str(args.n) == "all" else int(args.n)
        sel = shas_list[:n_sel]
    else:
        import pandas as pd
        df = pd.read_parquet(parquet, columns=["success"])
        df = df[df["success"] == True] if "success" in df.columns else df
        shas = [str(x) for x in df.index]
        sp_ = json.load(open(split_json))[args.split]
        idx = sp_["indices"] if isinstance(sp_, dict) else sp_
        n_sel = len(idx) if str(args.n) == "all" else int(args.n)
        sel = [shas[i] for i in idx[:n_sel]]
    p(f"Samples: {len(sel)}")

    if not args.gt_only:
        # ---- configs / norm ----
        dit_cfg = yaml.safe_load(open(dit_cfg_path))
        ds = dit_cfg["dataset"]["args"]
        emit_mean = torch.tensor(ds["emission_normalization"]["mean"]); emit_std = torch.tensor(ds["emission_normalization"]["std"])
        shape_mean = torch.tensor(ds["shape_normalization"]["mean"]);   shape_std = torch.tensor(ds["shape_normalization"]["std"])
        # pbrcond variant (2026-08-09 conditioning parity): albedo-slot latents
        # WITH the PBR latent channel-concatenated per token, exactly like the
        # twostream predictor's cond assembly. Config-driven; the missing-key
        # default mirrors LightGenSLatEmissionLatent (skip_pbr_cond=False), and
        # the in_channels cross-check refuses to guess for any legacy layout
        # (e.g. the archived in=64 pbr-only-cond overfit config) instead of
        # silently assembling the wrong conditioning.
        use_pbr_cond = not ds.get("skip_pbr_cond", False)
        in_ch = dit_cfg["models"]["denoiser"]["args"]["in_channels"]
        expected_in = 96 if use_pbr_cond else 64
        assert in_ch == expected_in, (
            f"in_channels={in_ch} inconsistent with skip_pbr_cond={ds.get('skip_pbr_cond')} "
            f"(expected {expected_in}); unsupported/legacy cond layout — refusing to guess")
        if use_pbr_cond:
            pbr_mean = torch.tensor(ds["pbr_normalization"]["mean"]); pbr_std = torch.tensor(ds["pbr_normalization"]["std"])

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
        # mmap: the ~20 GB Lightning ckpt (model+optimizer) would otherwise be
        # materialized in RAM and OOM-kill the process on a busy workstation;
        # lazy-mapping means only the tensors load_state_dict copies get read.
        ck = torch.load(args.dit_ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
        dit.load_state_dict({k.removeprefix("models.denoiser."): v for k, v in ck.items() if k.startswith("models.denoiser.")})
        dit.to(DEVICE).eval()
        sampler = FlowEulerCfgSampler(sigma_min=dit_cfg["trainer"]["args"]["sigma_min"])

        # ---- VAE encoder + decoder ----
        p(f"Loading VAE: {args.vae_ckpt}")
        vae = build_models(yaml.safe_load(open(vae_cfg_path))["models"])
        enc, dec = vae["encoder"], vae["decoder"]
        vck = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
        enc.load_state_dict({k.removeprefix("models.encoder."): v for k, v in vck.items() if k.startswith("models.encoder.")})
        dec.load_state_dict({k.removeprefix("models.decoder."): v for k, v in vck.items() if k.startswith("models.decoder.")})
        enc.to(DEVICE).eval(); dec.to(DEVICE).eval()

    out_vox_dir = os.path.join(args.out_dir, "pred_voxels")
    os.makedirs(out_vox_dir, exist_ok=True)

    def to01(feats3):
        return (feats3[:, :3] * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()

    for k, sha in enumerate(sel):
        p(f"[{k+1}/{len(sel)}] {sha[:16]}...")
        coords, attr = o_voxel.io.read_vxz(os.path.join(vox_dir, f"{sha}.vxz"), num_threads=4)
        coords_np = coords.int().cpu().numpy()
        gt_emission01 = (attr["emissive"].float() / 255.0).clamp(0, 1).cpu().numpy()  # [N,3] in [0,1]

        if args.gt_only:
            np.savez(os.path.join(out_vox_dir, f"{sha}.npz"),
                     coords=coords_np, gt_emission=gt_emission01)
            continue

        bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
        x_gt = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(DEVICE)
        z_enc = enc(pack_emission_as_pbr(x_gt))

        z_shape = load_latent(os.path.join(args.data_root, "shape_latents", args.shape_latent_name, f"{sha}.npz"))
        z_emit_n = z_enc.replace(normalize(z_enc.feats, emit_mean, emit_std))
        z_shape_n = z_enc.replace(normalize(z_shape.feats.to(DEVICE), shape_mean, shape_std))
        if use_pbr_cond:
            # cond order [shape | pbr] -> DiT input [x_t | shape | pbr] = 96ch,
            # mirroring predict_emission_voxels_twostream.py and the training
            # dataset's concat_cond construction (incl. its coord-equality
            # assert — equal row counts with a different token order would
            # silently pair PBR features with the wrong tokens).
            z_pbr = load_latent(os.path.join(args.data_root, "pbr_latents", args.pbr_latent_name, f"{sha}.npz")).to(DEVICE)
            assert torch.equal(z_enc.coords, z_pbr.coords) and torch.equal(z_enc.coords, z_shape.coords.to(DEVICE)), \
                f"latent coord mismatch {sha}: enc={tuple(z_enc.coords.shape)} pbr={tuple(z_pbr.coords.shape)} shape={tuple(z_shape.coords.shape)}"
            z_pbr_n = z_enc.replace(normalize(z_pbr.feats, pbr_mean, pbr_std))
            concat_cond = z_emit_n.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))
        else:
            concat_cond = z_emit_n.replace(z_shape_n.feats)
        _, img_feat = encode_image(sha); neg = torch.zeros_like(img_feat)
        # GLOBAL seed, not a torch.Generator. The initial noise is the only randomness in this
        # lane (FlowEuler is a deterministic ODE, the VAE encodes with sample_posterior=False,
        # and trellis2/ has no dropout), so a CUDA generator would work here -- but TEXGen and
        # Hunyuan3D cannot use one, and one uniform mechanism across all four lanes is what
        # stops a later tidy-up from silently breaking reproducibility. See
        # docs/superpowers/specs/2026-08-15-multi-seed-evaluation-design.md §2.2.
        torch.manual_seed(sample_seed(sha, args.seed))
        noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = sampler.sample(dit, noise, cond=img_feat, neg_cond=neg, concat_cond=concat_cond,
                                 steps=args.steps, guidance_strength=args.guidance, verbose=False)
        z_dit = z_enc.replace(denormalize(res.samples.feats, emit_mean.to(DEVICE), emit_std.to(DEVICE)))

        dit_dec = dec(z_dit)
        if args.pred_only:
            # Keep the coord assert: it is the ONLY gate that the decoder output aligns
            # element-wise with coords_np.
            assert torch.equal(dit_dec.coords[:, 1:], coords.to(dit_dec.coords.device)), \
                f"coord order mismatch {sha}: dit_dec={tuple(dit_dec.coords.shape)} raw={tuple(coords.shape)}"
            np.savez(os.path.join(out_vox_dir, f"{sha}.npz"),
                     coords=coords_np,
                     pred=to01(dit_dec.feats))
        else:
            gt_dec = dec(z_enc)
            # shared-coords gate: decode must align in element order, not just count
            assert torch.equal(gt_dec.coords, dit_dec.coords) and \
                   torch.equal(gt_dec.coords[:, 1:], coords.to(gt_dec.coords.device)), \
                f"coord order mismatch {sha}: gt_dec={tuple(gt_dec.coords.shape)} dit_dec={tuple(dit_dec.coords.shape)} raw={tuple(coords.shape)}"

            np.savez(os.path.join(out_vox_dir, f"{sha}.npz"),
                     coords=coords_np,
                     pred=to01(dit_dec.feats),
                     gt_recon=to01(gt_dec.feats),
                     gt_emission=gt_emission01)
        torch.cuda.empty_cache()

    p(f"\nDONE -> {out_vox_dir}  ({len(sel)} shapes, gt_only={args.gt_only})")


if __name__ == "__main__":
    main()
