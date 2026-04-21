#!/usr/bin/env python3
"""
Three-way comparison: original GLB emission | voxelized emission | roundtrip GLB.
Outputs a standalone HTML with three interactive 3D panels.
"""
import sys, os, types, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import numpy as np
import pandas as pd

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

import plotly.graph_objects as go

DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
GLB_ROOT  = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
ROUNDTRIP_DIR = os.path.join(os.path.dirname(__file__), "..", "debug_outputs", "roundtrip_glbs")
MAX_POINTS = 100000


def emission_to_rgb(feats_norm):
    linear = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    rgb = (srgb.clamp(0, 1) * 255).byte().numpy()
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def embed_glb(path):
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    return ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=3,
                        help="Sample index (0-9)")
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
    sha = list(df.index)[args.sample]
    glb_rel = df.loc[sha, "glb_1k_path"]
    glb_path = os.path.join(GLB_ROOT, glb_rel)
    roundtrip_path = os.path.join(ROUNDTRIP_DIR, f"{sha[:8]}_roundtrip.glb")
    print(f"Sample [{args.sample}] {sha}")
    print(f"  Original GLB: {glb_path}")
    print(f"  Roundtrip GLB: {roundtrip_path}")

    # Voxelized emission data
    coords, emit_attr = o_voxel.io.read_vxz(
        f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
    emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0
    print(f"  Voxels: {coords.shape[0]:,}")

    emit_mag = (emit_feats * 0.5 + 0.5).norm(dim=-1)
    nz_mask = emit_mag > args.threshold
    nz_idx = torch.where(nz_mask)[0]
    print(f"  Non-zero emission voxels: {nz_idx.shape[0]:,}")

    if nz_idx.shape[0] > MAX_POINTS:
        perm = torch.randperm(nz_idx.shape[0])[:MAX_POINTS]
        nz_idx = nz_idx[perm]

    c = coords[nz_idx].long().numpy()
    f = emit_feats[nz_idx]
    colors = emission_to_rgb(f)
    mag = (f * 0.5 + 0.5).norm(dim=-1).numpy()

    fig = go.Figure(data=[go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers",
        marker=dict(size=3, color=colors, opacity=1.0, line=dict(width=0)),
        text=[f"emit={m:.3f}" for m in mag],
        hoverinfo="text",
    )])
    axis_cfg = dict(
        showbackground=True, backgroundcolor="rgb(10,10,10)",
        showgrid=False, zeroline=False, showticklabels=False, title="",
    )
    fig.update_layout(
        scene=dict(xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg, aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
        paper_bgcolor="rgb(20,20,20)", font=dict(color="white"),
        height=700, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{nz_mask.sum().item():,} non-zero / {coords.shape[0]:,}",
                   font=dict(size=11)),
    )
    plotly_html = fig.to_html(full_html=True, include_plotlyjs="cdn")

    glb_b64 = embed_glb(glb_path)
    if glb_b64:
        print(f"  Embedded original GLB ({os.path.getsize(glb_path)/1e6:.1f} MB)")

    rt_b64 = embed_glb(roundtrip_path)
    if rt_b64:
        print(f"  Embedded roundtrip GLB ({os.path.getsize(roundtrip_path)/1e6:.1f} MB)")
    else:
        print(f"  WARNING: roundtrip GLB not found at {roundtrip_path}")

    plotly_div = (plotly_html
        .replace("<!DOCTYPE html>", "").replace("<html>", "").replace("</html>", "")
        .replace("<head>", "").replace("</head>", "")
        .replace("<body>", "").replace("</body>", ""))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Emission Pipeline — {sha[:16]}...</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #111; color: #fff; font-family: Arial, sans-serif; }}
  .header {{ padding: 10px 20px; background: #1a1a1a; border-bottom: 1px solid #333; }}
  .header h2 {{ font-size: 16px; }}
  .header .sub {{ font-size: 12px; color: #999; }}
  .container {{ display: flex; width: 100%; height: calc(100vh - 50px); }}
  .panel {{ flex: 1; position: relative; overflow: hidden; }}
  .panel .label {{ position: absolute; top: 8px; left: 10px; z-index: 10; font-size: 13px;
                   font-weight: bold; background: rgba(0,0,0,0.7); padding: 4px 10px; border-radius: 4px; }}
  .divider {{ width: 2px; background: #333; flex-shrink: 0; }}
  canvas.glb-view {{ width: 100%; height: 100%; display: block; }}
</style>
</head>
<body>
<div class="header">
  <h2>Emission Pipeline — {sha[:16]}...</h2>
  <div class="sub">Left: Original GLB (emission only) &nbsp;|&nbsp; Center: Voxelized emission &nbsp;|&nbsp; Right top: Roundtrip emission only &nbsp;|&nbsp; Right bottom: Roundtrip full PBR</div>
</div>
<div class="container">
  <div class="panel" id="panel-orig">
    <div class="label">Original GLB (emission only)</div>
    <canvas id="canvas-orig" class="glb-view"></canvas>
  </div>
  <div class="divider"></div>
  <div class="panel" id="panel-voxel">
    <div class="label">Voxelized</div>
    {plotly_div}
  </div>
  <div class="divider"></div>
  <div class="panel" id="panel-rt" style="display:flex;flex-direction:column;">
    <div style="flex:1;position:relative;">
      <div class="label">Roundtrip (emission only)</div>
      <canvas id="canvas-rt-emit" class="glb-view"></canvas>
    </div>
    <div style="height:2px;background:#333;flex-shrink:0;"></div>
    <div style="flex:1;position:relative;">
      <div class="label">Roundtrip (full PBR)</div>
      <canvas id="canvas-rt-full" class="glb-view"></canvas>
    </div>
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

function makeViewer(canvasId, glbB64, emissionOnly) {{
  const canvas = document.getElementById(canvasId);
  const panel = canvas.parentElement;
  const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x111111);
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  if (!emissionOnly) {{
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(2, 3, 2);
    scene.add(dirLight);
  }}
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(2, 1.5, 2);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.target.set(0, 0, 0);

  if (glbB64) {{
    const binary = Uint8Array.from(atob(glbB64), c => c.charCodeAt(0));
    const blob = new Blob([binary], {{ type: 'model/gltf-binary' }});
    const url = URL.createObjectURL(blob);
    new GLTFLoader().load(url, (gltf) => {{
      const model = gltf.scene;
      if (emissionOnly) {{
        model.traverse((child) => {{
          if (!child.isMesh) return;
          const mat = child.material;
          if (!mat) return;
          const hasEmissive = (mat.emissiveMap !== null && mat.emissiveMap !== undefined)
            || (mat.emissive && (mat.emissive.r > 0 || mat.emissive.g > 0 || mat.emissive.b > 0));
          mat.color = new THREE.Color(0x000000);
          mat.map = null; mat.metalness = 0; mat.roughness = 1;
          mat.metalnessMap = null; mat.roughnessMap = null;
          mat.normalMap = null; mat.aoMap = null;
          if (hasEmissive) {{
            mat.emissiveIntensity = 1.0;
            if (mat.emissiveMap) mat.emissiveMap.colorSpace = THREE.SRGBColorSpace;
          }} else {{
            mat.emissive = new THREE.Color(0x000000);
            mat.emissiveMap = null; mat.emissiveIntensity = 0;
          }}
          mat.needsUpdate = true;
        }});
      }}
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      model.position.sub(center);
      model.scale.setScalar(2.0 / size);
      scene.add(model);
      controls.target.set(0, 0, 0);
      camera.position.set(1.5, 1.0, 1.5);
      controls.update();
    }});
  }}

  function resize() {{
    const w = panel.clientWidth, h = panel.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }}
  resize();
  window.addEventListener('resize', resize);
  new ResizeObserver(resize).observe(panel);

  return {{ scene, camera, controls, renderer, resize }};
}}

const origViewer   = makeViewer('canvas-orig',    "{glb_b64}", true);
const rtEmitViewer = makeViewer('canvas-rt-emit', "{rt_b64}",  true);
const rtFullViewer = makeViewer('canvas-rt-full', "{rt_b64}",  false);

(function animate() {{
  requestAnimationFrame(animate);
  origViewer.controls.update();
  origViewer.renderer.render(origViewer.scene, origViewer.camera);
  rtEmitViewer.controls.update();
  rtEmitViewer.renderer.render(rtEmitViewer.scene, rtEmitViewer.camera);
  rtFullViewer.controls.update();
  rtFullViewer.renderer.render(rtFullViewer.scene, rtFullViewer.camera);
}})();
</script>
</body>
</html>"""

    out_path = args.out or f"/localhome/dya78/code/TRELLIS.2-lightning/visualizations/voxel_vs_glb_{sha[:8]}.html"
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
