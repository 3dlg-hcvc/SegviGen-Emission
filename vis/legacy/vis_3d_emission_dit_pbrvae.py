#!/usr/bin/env python3
"""
3D visualizer for the emission DiT overfit (PBR-VAE pipeline).

Single HTML with a dropdown to switch between samples.
Per-sample 3 side-by-side plots: GT | VAE reconstruction | DiT → VAE decode.

Usage:
    conda run -n trellis2 python vis/vis_3d_emission_dit_pbrvae.py
"""
import sys, os, types, argparse
sys.path.insert(0, "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning")
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import pandas as pd
import yaml
from PIL import Image

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.modules import sparse as sp
from trellis2.trainers.builders import build_models
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from torchvision import transforms

import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.set_grad_enabled(False)

DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
BASE_DIR  = "/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/outputs"
DIT_DIR   = f"{BASE_DIR}/emission_dit_10sample_overfit_pbrvae"
VAE_DIR   = f"{BASE_DIR}/emission_pbr_finetune_10sample"

PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA = -1.0, 1.0, 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_POINTS = 60000


def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))


def normalize(f, m, s):   return (f - m.to(f.device)) / s.to(f.device)
def denormalize(f, m, s): return f * s.to(f.device) + m.to(f.device)


def pack_emission_as_pbr(x: sp.SparseTensor) -> sp.SparseTensor:
    emit = x.feats[:, :3]
    n = emit.shape[0]
    pad = torch.tensor([[PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA]],
                       device=emit.device, dtype=emit.dtype).expand(n, -1)
    return x.replace(torch.cat([emit, pad], dim=-1))


def emission_to_rgb(feats_norm):
    linear = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    rgb = (srgb.clamp(0, 1) * 255).byte().numpy()
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def make_3d_scatter(coords, feats_norm, name, visible=True):
    c = coords.numpy()
    mag = (feats_norm * 0.5 + 0.5).norm(dim=-1).numpy()
    colors = emission_to_rgb(feats_norm)
    return go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors, opacity=1.0, line=dict(width=0)),
        name=name,
        text=[f"emit={m:.3f}" for m in mag],
        hoverinfo="text+name",
        visible=visible,
    )


def subsample(idx, max_n=MAX_POINTS):
    if idx.shape[0] > max_n:
        perm = torch.randperm(idx.shape[0])[:max_n]
        return idx[perm]
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--out", type=str,
                        default="/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/visualizations/emission_dit_pbrvae_3d.html")
    args = parser.parse_args()

    # Config + norms
    with open(f"{DIT_DIR}/config.yaml") as f: dit_cfg = yaml.safe_load(f)
    ds = dit_cfg["dataset"]["args"]
    emit_mean  = torch.tensor(ds["emission_normalization"]["mean"])
    emit_std   = torch.tensor(ds["emission_normalization"]["std"])
    pbr_mean   = torch.tensor(ds["pbr_normalization"]["mean"])
    pbr_std    = torch.tensor(ds["pbr_normalization"]["std"])
    shape_mean = torch.tensor(ds["shape_normalization"]["mean"])
    shape_std  = torch.tensor(ds["shape_normalization"]["std"])

    # DINOv3
    print("Loading DINOv3...")
    dino_cfg = dit_cfg["trainer"]["args"]["image_cond_model"]["args"]
    dino = DinoV3FeatureExtractor(**dino_cfg); dino.cuda()
    dino_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # DiT
    print("Loading DiT...")
    dit_models = build_models(dit_cfg["models"])
    dit = dit_models["denoiser"]
    dit_state = {k.removeprefix("models.denoiser."): v
                 for k, v in torch.load(f"{DIT_DIR}/ckpts/last.ckpt",
                                        map_location="cpu", weights_only=False)["state_dict"].items()
                 if k.startswith("models.denoiser.")}
    dit.load_state_dict(dit_state); dit.to(device).eval()
    sampler = FlowEulerCfgSampler(sigma_min=dit_cfg["trainer"]["args"]["sigma_min"])

    # Finetuned PBR VAE
    print("Loading VAE...")
    with open(f"{VAE_DIR}/config.yaml") as f: vae_cfg = yaml.safe_load(f)
    vae_models = build_models(vae_cfg["models"])
    enc, dec = vae_models["encoder"], vae_models["decoder"]
    vae_state = torch.load(f"{VAE_DIR}/ckpts/last.ckpt",
                           map_location="cpu", weights_only=False)["state_dict"]
    enc.load_state_dict({k.removeprefix("models.encoder."): v for k,v in vae_state.items() if k.startswith("models.encoder.")})
    dec.load_state_dict({k.removeprefix("models.decoder."): v for k,v in vae_state.items() if k.startswith("models.decoder.")})
    enc.to(device).eval(); dec.to(device).eval()

    df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
    samples = list(df["sha256"] if "sha256" in df.columns else df.index)

    all_traces = []
    sample_info = []
    TRACES_PER_SAMPLE = 3

    for si, sha in enumerate(samples):
        print(f"\n[{si}] {sha[:20]}...")

        coords, attr = o_voxel.io.read_vxz(f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
        bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
        x_gt = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(device)

        z_emit_raw  = load_latent(f"{DATA_ROOT}/emission_latents/emission_pbr_finetune_10sample/{sha}.npz")
        z_pbr_raw   = load_latent(f"{DATA_ROOT}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{sha}.npz")
        z_shape_raw = load_latent(f"{DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/{sha}.npz")

        z_emit_n  = z_emit_raw.replace(normalize(z_emit_raw.feats,  emit_mean,  emit_std)).to(device)
        z_pbr_n   = z_pbr_raw.replace(normalize(z_pbr_raw.feats,    pbr_mean,   pbr_std)).to(device)
        z_shape_n = z_shape_raw.replace(normalize(z_shape_raw.feats, shape_mean, shape_std)).to(device)

        concat_cond = z_emit_n.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))

        img = Image.open(f"{DATA_ROOT}/thumbnails/{sha}.png").convert("RGB")
        t = torch.from_numpy(np.array(img.resize((dino_cfg["image_size"],) * 2, Image.LANCZOS))
            .astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        t = dino_tr(t).to(dino.model.embeddings.patch_embeddings.weight.dtype)
        img_feat = dino.extract_features(t)
        neg_cond = torch.zeros_like(img_feat)

        # Sample DiT
        noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = sampler.sample(dit, noise, cond=img_feat, neg_cond=neg_cond,
                                    concat_cond=concat_cond, steps=50,
                                    guidance_strength=3.0, verbose=False)
        z_dit_n = result.samples
        z_dit_raw = z_dit_n.replace(denormalize(z_dit_n.feats, emit_mean.to(device), emit_std.to(device)))

        # Decode (finetuned PBR VAE) — run encoder first for subdivision cache
        x_packed = pack_emission_as_pbr(x_gt)
        z_enc = enc(x_packed)
        z_gt_dec  = dec(z_enc.replace(z_emit_raw.feats.to(device)))
        z_dit_dec = dec(z_enc.replace(z_dit_raw.feats))

        emit_gt_pred  = z_gt_dec.feats[:, :3].cpu()
        emit_dit_pred = z_dit_dec.feats[:, :3].cpu()

        vae_l1 = (x_gt.feats.cpu() - emit_gt_pred).abs().mean().item()
        dit_l1 = (x_gt.feats.cpu() - emit_dit_pred).abs().mean().item()
        lat_l1 = (z_emit_raw.feats.to(device) - z_dit_raw.feats).abs().mean().item()
        print(f"  VAE L1={vae_l1:.4f}  DiT L1={dit_l1:.4f}  Lat L1={lat_l1:.4f}")

        visible = (si == 0)
        voxel_coords = coords.long()

        gt_mag = (gt_feats * 0.5 + 0.5).norm(dim=-1)
        gt_idx = subsample(torch.where(gt_mag > args.threshold)[0])

        vae_mag = (emit_gt_pred * 0.5 + 0.5).norm(dim=-1)
        vae_idx = subsample(torch.where(vae_mag > args.threshold)[0])

        dit_mag = (emit_dit_pred * 0.5 + 0.5).norm(dim=-1)
        dit_idx = subsample(torch.where(dit_mag > args.threshold)[0])

        gt_trace  = make_3d_scatter(voxel_coords[gt_idx],  gt_feats[gt_idx],  "GT",  visible)
        vae_trace = make_3d_scatter(voxel_coords[vae_idx], emit_gt_pred[vae_idx], f"VAE rec (L1={vae_l1:.4f})", visible)
        dit_trace = make_3d_scatter(voxel_coords[dit_idx], emit_dit_pred[dit_idx], f"DiT sample (L1={dit_l1:.4f})", visible)

        gt_trace.update(scene="scene")
        vae_trace.update(scene="scene2")
        dit_trace.update(scene="scene3")

        all_traces.extend([gt_trace, vae_trace, dit_trace])
        sample_info.append({
            "sha": sha,
            "vae_l1": vae_l1,
            "dit_l1": dit_l1,
            "lat_l1": lat_l1,
            "total_vox": coords.shape[0],
        })
        torch.cuda.empty_cache()

    # Build figure
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=["Ground Truth", "VAE Reconstruction", "DiT Sample → VAE Decode"],
        horizontal_spacing=0.01,
    )
    for tr in all_traces:
        col = {"scene": 1, "scene2": 2, "scene3": 3}[tr.scene]
        tr.scene = None
        fig.add_trace(tr, row=1, col=col)

    axis_cfg = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    for i in range(1, 4):
        key = "scene" if i == 1 else f"scene{i}"
        fig.layout[key].update(
            xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor="rgb(20,20,20)",
        )

    buttons = []
    for si, info in enumerate(sample_info):
        vis = [False] * len(all_traces)
        for k in range(TRACES_PER_SAMPLE):
            vis[si * TRACES_PER_SAMPLE + k] = True
        title = (f"Emission DiT (PBR-VAE) — Sample {si} ({info['sha'][:12]}...)<br>"
                 f"<sub>VAE L1={info['vae_l1']:.4f} | DiT L1={info['dit_l1']:.4f} | "
                 f"Lat L1={info['lat_l1']:.4f} | {info['total_vox']:,} GT voxels</sub>")
        buttons.append(dict(
            method="update",
            label=f"[{si}] {info['sha'][:12]}",
            args=[{"visible": vis}, {"title.text": title}],
        ))

    init = sample_info[0]
    init_title = (f"Emission DiT (PBR-VAE) — Sample 0 ({init['sha'][:12]}...)<br>"
                  f"<sub>VAE L1={init['vae_l1']:.4f} | DiT L1={init['dit_l1']:.4f} | "
                  f"Lat L1={init['lat_l1']:.4f} | {init['total_vox']:,} GT voxels</sub>")

    fig.update_layout(
        title=init_title,
        paper_bgcolor="rgb(20,20,20)",
        font=dict(color="white"),
        height=700,
        width=1600,
        showlegend=False,
        updatemenus=[dict(
            active=0, buttons=buttons, direction="down",
            x=0.01, y=1.15, xanchor="left", yanchor="top",
            bgcolor="rgb(40,40,40)",
            bordercolor="rgb(80,80,80)",
            font=dict(color="white"),
        )],
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
