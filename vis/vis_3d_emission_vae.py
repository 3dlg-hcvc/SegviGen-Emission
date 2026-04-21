#!/usr/bin/env python3
"""
Interactive 3D visualizer for emission VAE reconstructions.

Generates a standalone HTML file with side-by-side GT vs Reconstructed
3D scatter plots of emission voxels, colored by emission intensity.

Usage:
    conda run -n trellis2 python data_toolkit/vis_3d_emission_vae.py [options]

    # All samples at step 20k (default):
    conda run -n trellis2 python data_toolkit/vis_3d_emission_vae.py

    # Specific sample + checkpoint:
    conda run -n trellis2 python data_toolkit/vis_3d_emission_vae.py \
        --sample 7 --run_dir outputs/emission_vae_10sample_10kep \
        --ckpt stepstep=0020000.ckpt

    # Compare multiple checkpoints:
    conda run -n trellis2 python data_toolkit/vis_3d_emission_vae.py --all_ckpts
"""
import sys, os, types, argparse
sys.path.insert(0, "/localhome/dya78/code/TRELLIS.2-lightning")
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import pandas as pd
import yaml

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.modules import sparse as sp
import trellis2.models as trellis_models
from trellis2.trainers.builders import build_models

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
GLB_ROOT  = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
BASE_DIR  = "/localhome/dya78/code/TRELLIS.2-lightning/outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_POINTS = 80000


def get_glb_path(sha):
    """Resolve GLB path for a sample."""
    df_full = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
    glb_rel = df_full.loc[sha, "glb_1k_path"]
    return os.path.join(GLB_ROOT, glb_rel)


def load_run(run_dir, ckpt_name):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    models  = build_models(cfg["models"])
    encoder = models["encoder"]
    decoder = models["decoder"]
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)
    print(f"  Loading: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    encoder.load_state_dict({k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")})
    decoder.load_state_dict({k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")})
    encoder.to(device).eval()
    decoder.to(device).eval()
    pbr_name = cfg["trainer"]["args"].get("pretrained_pbr_encoder")
    pbr_enc  = trellis_models.from_pretrained(pbr_name).eval().float().to(device) if pbr_name else None
    return encoder, decoder, pbr_enc


@torch.no_grad()
def reconstruct(encoder, decoder, pbr_encoder, coords, emit_feats, pbr_feats):
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined  = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0        = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    x0_emit   = x0.replace(emit_feats)
    x0_gpu    = x0.to(device)
    z_emit    = encoder(x0_emit.to(device))
    if pbr_encoder is not None:
        z_pbr  = pbr_encoder(x0_gpu.replace(x0_gpu.feats[:, 3:]))
        z_comb = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
    else:
        z_comb = z_emit
    return decoder(z_comb).to("cpu")


def emission_to_rgb(feats_norm):
    """Convert [-1,1] emission features (linear space) to sRGB [0,255] for plotly.

    The voxelized emission is in linear color space (Blender extracts textures
    in linear, and the voxelizer stores factor*texture as linear float → uint8).
    CSS rgb() is interpreted as sRGB, so we apply the standard gamma curve.
    """
    linear = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    rgb = (srgb.clamp(0, 1) * 255).byte().numpy()
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def make_3d_scatter(coords, feats_norm, name, subsample_idx=None):
    """Create a plotly Scatter3d trace from sparse voxel data."""
    c = coords.numpy()
    f = feats_norm

    if subsample_idx is not None:
        c = c[subsample_idx]
        f = f[subsample_idx]

    mag = (f * 0.5 + 0.5).norm(dim=-1).numpy()
    colors = emission_to_rgb(f)

    return go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=colors,
            opacity=1.0,
            line=dict(width=0),
        ),
        name=name,
        text=[f"emit={m:.3f}" for m in mag],
        hoverinfo="text+name",
    )


def build_figure_for_sample(sha, coords, emit_feats, pbr_feats, runs, threshold=0.01):
    """
    Build a plotly figure with GT + one or more reconstruction subplots.

    runs: list of (label, encoder, decoder, pbr_encoder)
    """
    n_plots = 1 + len(runs)
    titles  = ["Ground Truth"] + [r[0] for r in runs]
    fig     = make_subplots(
        rows=1, cols=n_plots,
        specs=[[{"type": "scene"}] * n_plots],
        subplot_titles=titles,
        horizontal_spacing=0.02,
    )

    voxel_coords = coords.long()
    emit_mag = (emit_feats * 0.5 + 0.5).norm(dim=-1)
    nz_mask  = emit_mag > threshold

    # Subsample if too many points
    nz_idx = torch.where(nz_mask)[0]
    if nz_idx.shape[0] > MAX_POINTS:
        perm   = torch.randperm(nz_idx.shape[0])[:MAX_POINTS]
        nz_idx = nz_idx[perm]
        print(f"  Subsampled GT non-zero voxels: {nz_mask.sum().item():,} → {MAX_POINTS:,}")

    gt_trace = make_3d_scatter(voxel_coords[nz_idx], emit_feats[nz_idx], "GT")
    fig.add_trace(gt_trace, row=1, col=1)

    for ci, (label, enc, dec, pbr_enc) in enumerate(runs, start=2):
        y = reconstruct(enc, dec, pbr_enc, coords, emit_feats, pbr_feats)
        rec_feats = y.feats.cpu()

        # Show only voxels where reconstruction predicts non-zero emission
        rec_mag  = (rec_feats * 0.5 + 0.5).norm(dim=-1)
        show_mask = rec_mag > threshold
        show_idx  = torch.where(show_mask)[0]
        if show_idx.shape[0] > MAX_POINTS:
            perm     = torch.randperm(show_idx.shape[0])[:MAX_POINTS]
            show_idx = show_idx[perm]

        rec_trace = make_3d_scatter(voxel_coords[show_idx], rec_feats[show_idx], label)
        fig.add_trace(rec_trace, row=1, col=ci)
        l1 = (emit_feats[nz_idx] - rec_feats[nz_idx]).abs().mean().item()
        print(f"  {label}: L1(emission voxels)={l1:.5f}")
        torch.cuda.empty_cache()

    axis_cfg = dict(
        showbackground=False,
        showgrid=False, zeroline=False, showticklabels=False,
        title="",
    )
    for ci in range(1, n_plots + 1):
        scene_name = f"scene{ci}" if ci > 1 else "scene"
        fig.layout[scene_name].update(
            xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        )

    fig.update_layout(
        title=f"Emission VAE 3D — {sha[:16]}...<br>"
              f"<sub>{coords.shape[0]:,} total voxels, {nz_mask.sum().item():,} non-zero emission</sub>",
        paper_bgcolor="rgb(20,20,20)",
        font=dict(color="white"),
        height=700,
        width=450 * n_plots,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         showlegend=False,
    )
    return fig


def build_combined_html(sha, plotly_html, glb_path, out_path):
    """
    Build a combined HTML page with a three.js GLB viewer on the left
    and the plotly voxel comparison on the right.
    """
    import base64
    glb_b64 = ""
    if os.path.exists(glb_path):
        with open(glb_path, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"  Embedded GLB: {os.path.basename(glb_path)} ({os.path.getsize(glb_path)/1e6:.1f} MB)")
    else:
        print(f"  WARNING: GLB not found at {glb_path}")

    plotly_div = plotly_html.replace("<!DOCTYPE html>", "").replace("<html>", "").replace("</html>", "")
    plotly_div = plotly_div.replace("<head>", "").replace("</head>", "").replace("<body>", "").replace("</body>", "")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Emission VAE 3D — {sha[:16]}...</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #111; color: #fff; font-family: Arial, sans-serif; overflow-x: auto; }}
  .header {{ padding: 10px 20px; background: #1a1a1a; border-bottom: 1px solid #333; }}
  .header h2 {{ font-size: 16px; }}
  .header .sub {{ font-size: 12px; color: #999; }}
  .container {{ display: flex; flex-direction: row; width: 100%; min-height: calc(100vh - 50px); }}
  .glb-panel {{ width: 400px; min-width: 400px; border-right: 2px solid #333; position: relative; }}
  .glb-panel .label {{ position: absolute; top: 8px; left: 10px; z-index: 10; font-size: 13px;
                       font-weight: bold; background: rgba(0,0,0,0.6); padding: 3px 8px; border-radius: 4px; }}
  #glb-canvas {{ width: 100%; height: 100%; display: block; }}
  .plotly-panel {{ flex: 1; overflow: auto; }}
</style>
</head>
<body>
<div class="header">
  <h2>Emission VAE 3D Viewer — {sha[:16]}...</h2>
  <div class="sub">Left: GLB emission only (black = no emission) &nbsp;|&nbsp; Right: GT + Reconstructed emission voxels (interactive)</div>
</div>
<div class="container">
  <div class="glb-panel">
    <div class="label">GLB Emission Only</div>
    <canvas id="glb-canvas"></canvas>
  </div>
  <div class="plotly-panel" id="plotly-container">
    {plotly_div}
  </div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const canvas = document.getElementById('glb-canvas');
const panel = canvas.parentElement;
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x111111);
renderer.toneMapping = THREE.NoToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2, 1.5, 2);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0, 0, 0);

const glbData = "{glb_b64}";
if (glbData) {{
  const binary = Uint8Array.from(atob(glbData), c => c.charCodeAt(0));
  const blob = new Blob([binary], {{ type: 'model/gltf-binary' }});
  const url = URL.createObjectURL(blob);
  new GLTFLoader().load(url, (gltf) => {{
    const model = gltf.scene;

    // Strip to emission only: black base color, keep emissive texture/color
    model.traverse((child) => {{
      if (!child.isMesh) return;
      const mat = child.material;
      if (!mat) return;
      const hasEmissive = (mat.emissiveMap !== null && mat.emissiveMap !== undefined)
        || (mat.emissive && (mat.emissive.r > 0 || mat.emissive.g > 0 || mat.emissive.b > 0));

      // Zero out base color so only emission shows
      mat.color = new THREE.Color(0x000000);
      mat.map = null;
      mat.metalness = 0;
      mat.roughness = 1;
      mat.metalnessMap = null;
      mat.roughnessMap = null;
      mat.normalMap = null;
      mat.aoMap = null;

      if (hasEmissive) {{
        mat.emissiveIntensity = 1.0;
        // Keep emissive texture in sRGB so colors match the raw voxel data
        if (mat.emissiveMap) {{
          mat.emissiveMap.colorSpace = THREE.SRGBColorSpace;
        }}
      }} else {{
        mat.emissive = new THREE.Color(0x000000);
        mat.emissiveMap = null;
        mat.emissiveIntensity = 0;
      }}
      mat.needsUpdate = true;
    }});

    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    model.position.sub(center);
    const scale = 2.0 / size;
    model.scale.setScalar(scale);
    scene.add(model);
    controls.target.set(0, 0, 0);
    camera.position.set(1.5, 1.0, 1.5);
    controls.update();
  }});
}}

function resize() {{
  const w = panel.clientWidth;
  const h = panel.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
resize();
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe(panel);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample index (0-9). Default: all samples.")
    parser.add_argument("--run_dir", type=str,
                        default=f"{BASE_DIR}/emission_vae_10sample_10kep")
    parser.add_argument("--ckpt", type=str, default="stepstep=0020000.ckpt")
    parser.add_argument("--all_ckpts", action="store_true",
                        help="Compare 500ep, 2kep, 5kep, 10kep for --sample")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Min emission magnitude to display")
    parser.add_argument("--no_glb", action="store_true",
                        help="Skip GLB embedding (voxel-only output)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Override output filename prefix (default: emission_vae_3d)")
    args = parser.parse_args()

    df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
    samples = list(df.index)

    if args.sample is not None:
        sample_indices = [args.sample]
    else:
        sample_indices = list(range(len(samples)))

    if args.all_ckpts:
        run_configs = [
            ("500 ep",   f"{BASE_DIR}/emission_vae_500ep",          "last.ckpt"),
            ("2000 ep",  f"{BASE_DIR}/emission_vae_10sample_10kep", "stepstep=0004000.ckpt"),
            ("5000 ep",  f"{BASE_DIR}/emission_vae_10sample_10kep", "stepstep=0010000.ckpt"),
            ("10000 ep", f"{BASE_DIR}/emission_vae_10sample_10kep", "stepstep=0020000.ckpt"),
        ]
    else:
        step_str = args.ckpt.replace("stepstep=", "").replace(".ckpt", "").replace("last", "last")
        run_configs = [(f"step {step_str}", args.run_dir, args.ckpt)]

    print("Loading models...")
    runs = []
    pbr_enc_shared = None
    for label, rd, ck in run_configs:
        enc, dec, pbr_enc = load_run(rd, ck)
        if pbr_enc_shared is None and pbr_enc is not None:
            pbr_enc_shared = pbr_enc
        runs.append((label, enc, dec, pbr_enc_shared))

    for si in sample_indices:
        sha = samples[si]
        print(f"\nSample [{si}] {sha[:16]}...")

        coords, emit_attr = o_voxel.io.read_vxz(
            f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
        emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0

        _, pbr_attr = o_voxel.io.read_vxz(
            f"{DATA_ROOT}/pbr_voxels_256/{sha}.vxz", num_threads=4)
        pbr_feats = torch.cat([
            pbr_attr["base_color"].float() / 255.0 * 2.0 - 1.0,
            pbr_attr["metallic"].float()   / 255.0 * 2.0 - 1.0,
            pbr_attr["roughness"].float()  / 255.0 * 2.0 - 1.0,
            pbr_attr["alpha"].float()      / 255.0 * 2.0 - 1.0,
        ], dim=-1)

        fig = build_figure_for_sample(sha, coords, emit_feats, pbr_feats, runs, args.threshold)

        suffix = "all_ckpts" if args.all_ckpts else "3d"
        prefix = args.output_prefix or "emission_vae_3d"
        out_path = args.out or f"/localhome/dya78/code/TRELLIS.2-lightning/visualizations/{prefix}_{sha[:8]}_{suffix}.html"

        if args.no_glb:
            fig.write_html(out_path)
        else:
            glb_path = get_glb_path(sha)
            plotly_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            build_combined_html(sha, plotly_html, glb_path, out_path)

        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
