#!/usr/bin/env python3
"""
Interactive 3D visualizer: finetuned PBR VAE on emission data.

Generates a SINGLE standalone HTML with a sample dropdown and 2 synced panels:
  GT  |  Finetuned PBR VAE Recon

Usage:
    conda run -n trellis2 python vis/vis_3d_pbr_finetune.py
"""
import sys, os, types, argparse, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import yaml
import numpy as np
import pandas as pd

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.trainers.builders import build_models
import trellis2.modules.sparse as sp

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "emission_pbr_finetune_10sample")
DATA_ROOT = "data/lightgen_test"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Padding values matching EmissionPbrFinetuneTrainer
PAD_METALLIC = -1.0
PAD_ROUGHNESS = 1.0
PAD_ALPHA = 1.0


def emission_to_srgb(feats_norm):
    linear = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    return (srgb.clamp(0, 1) * 255).byte()


def voxels_to_json(coords, feats, threshold, max_points):
    mag = (feats * 0.5 + 0.5).norm(dim=-1)
    mask = mag > threshold
    idx = torch.where(mask)[0]
    if idx.shape[0] > max_points:
        perm = torch.randperm(idx.shape[0])[:max_points]
        idx = idx[perm]
    c = coords[idx].long().numpy()
    rgb = emission_to_srgb(feats[idx]).numpy()
    return {
        "xyz": c.flatten().tolist(),
        "rgb": rgb.flatten().tolist(),
        "n": int(idx.shape[0]),
        "total": int(coords.shape[0]),
    }


@torch.no_grad()
def reconstruct_finetuned(encoder, decoder, coords, emit_feats):
    """Pack emission as PBR (matching EmissionPbrFinetuneTrainer) and decode."""
    n = emit_feats.shape[0]
    pad = torch.tensor([[PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA]]).expand(n, -1)
    packed = torch.cat([emit_feats, pad], dim=-1)  # [N, 6]
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    x = sp.SparseTensor(packed, torch.cat([batch_col, coords], dim=-1))
    z = encoder(x.to(device))
    y = decoder(z)
    return y.replace(y.feats[:, :3]).to("cpu")


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Finetuned PBR VAE on Emission — 3D</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }}
  .header {{
    padding: 12px 20px; background: #141414; border-bottom: 1px solid #2a2a2a;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
  }}
  .header h2 {{ font-size: 15px; font-weight: 600; }}
  .header .stats {{ font-size: 12px; color: #888; }}
  .controls {{
    padding: 8px 20px; background: #111; border-bottom: 1px solid #222;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  .controls label {{ font-size: 13px; color: #aaa; }}
  .controls select {{
    background: #1e1e1e; color: #e0e0e0; border: 1px solid #333; border-radius: 4px;
    padding: 4px 8px; font-size: 13px;
  }}
  .badge {{ border-radius: 4px; padding: 4px 10px; font-size: 13px; }}
  .l1-badge {{ background: #1a2a1a; border: 1px solid #2a4a2a; color: #7c7; }}
  .info-badge {{ background: #1a1a2a; border: 1px solid #2a2a4a; color: #99f; }}
  .viewer-row {{ display: flex; width: 100%; height: calc(100vh - 90px); }}
  .viewer-panel {{ flex: 1; position: relative; border-right: 1px solid #222; }}
  .viewer-panel:last-child {{ border-right: none; }}
  .panel-label {{
    position: absolute; top: 8px; left: 10px; z-index: 10;
    font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 4px;
    background: rgba(0,0,0,0.7); pointer-events: none;
  }}
  .panel-label.gt {{ color: #8cf; }}
  .panel-label.rec {{ color: #fc8; }}
  canvas {{ display: block; width: 100%; height: 100%; }}
</style>
</head>
<body>
<div class="header">
  <h2>Finetuned PBR VAE on Emission (20k steps, lr=1e-4)</h2>
  <div class="stats">{stats}</div>
</div>
<div class="controls">
  <label>Sample:</label>
  <select id="sample-select">{sample_options}</select>
  <span class="badge l1-badge" id="l1-display"></span>
  <span class="badge info-badge" id="info-display"></span>
</div>
<div class="viewer-row">
  <div class="viewer-panel" id="gt-panel">
    <div class="panel-label gt">Ground Truth</div>
    <canvas id="gt-canvas"></canvas>
  </div>
  <div class="viewer-panel" id="rec-panel">
    <div class="panel-label rec">Finetuned PBR VAE Recon</div>
    <canvas id="rec-canvas"></canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {{
  var DATA = {data_json};

  function buildPointCloud(d, norm) {{
    var n = d.n, xyz = d.xyz, rgb = d.rgb;
    var cx = norm.center[0], cy = norm.center[1], cz = norm.center[2], s = norm.scale;
    var pos = new Float32Array(n * 3);
    var col = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) {{
      pos[i*3]   = (xyz[i*3]   - cx) / s;
      pos[i*3+1] = (xyz[i*3+2] - cz) / s;
      pos[i*3+2] = (xyz[i*3+1] - cy) / s;
      col[i*3]   = rgb[i*3]   / 255;
      col[i*3+1] = rgb[i*3+1] / 255;
      col[i*3+2] = rgb[i*3+2] / 255;
    }}
    var geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('color',    new THREE.BufferAttribute(col, 3));
    var mat = new THREE.PointsMaterial({{ size: {point_size}, vertexColors: true, sizeAttenuation: true }});
    return new THREE.Points(geom, mat);
  }}

  function disposeCloud(cloud) {{
    if (!cloud) return;
    cloud.geometry.dispose();
    cloud.material.dispose();
  }}

  function createViewer(canvasId, panelId) {{
    var canvas = document.getElementById(canvasId);
    var panel  = document.getElementById(panelId);
    var renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x0a0a0a);
    var scene  = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    var controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    function resize() {{
      var w = panel.clientWidth, h = panel.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }}
    resize();
    window.addEventListener('resize', resize);
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(resize).observe(panel);
    (function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }})();
    return {{ scene: scene, camera: camera, controls: controls }};
  }}

  var gtV  = createViewer('gt-canvas',  'gt-panel');
  var recV = createViewer('rec-canvas', 'rec-panel');

  var syncing = false;
  function sync(src, dst) {{
    src.controls.addEventListener('change', function() {{
      if (syncing) return;
      syncing = true;
      dst.camera.position.copy(src.camera.position);
      dst.camera.quaternion.copy(src.camera.quaternion);
      dst.controls.target.copy(src.controls.target);
      dst.controls.update();
      syncing = false;
    }});
  }}
  sync(gtV, recV);
  sync(recV, gtV);

  function frameCameraOn(cloud) {{
    var box = new THREE.Box3().setFromObject(cloud);
    var center = box.getCenter(new THREE.Vector3());
    var sz = box.getSize(new THREE.Vector3()).length();
    [gtV, recV].forEach(function(v) {{
      v.controls.target.copy(center);
      v.camera.position.set(center.x + sz * 0.8, center.y + sz * 0.6, center.z + sz * 0.8);
      v.controls.update();
    }});
  }}

  var curGtCloud = null, curRecCloud = null;
  var sampleSel = document.getElementById('sample-select');
  var l1d   = document.getElementById('l1-display');
  var infod = document.getElementById('info-display');

  function loadSample(sha) {{
    var sd = DATA.samples[sha];
    if (curGtCloud)  {{ gtV.scene.remove(curGtCloud);   disposeCloud(curGtCloud); }}
    if (curRecCloud) {{ recV.scene.remove(curRecCloud); disposeCloud(curRecCloud); }}
    curGtCloud  = buildPointCloud(sd.gt, sd.norm);
    curRecCloud = buildPointCloud(sd.rec, sd.norm);
    gtV.scene.add(curGtCloud);
    recV.scene.add(curRecCloud);
    frameCameraOn(curGtCloud);
    l1d.textContent   = 'L1 = ' + sd.l1.toFixed(5);
    infod.textContent = sd.gt.n + ' emission voxels / ' + sd.gt.total + ' total';
  }}

  sampleSel.addEventListener('change', function() {{ loadSample(sampleSel.value); }});
  loadSample(sampleSel.value);
}})();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "ckpts", "last.ckpt"))
    parser.add_argument("--config", type=str, default=os.path.join(CKPT_DIR, "config.yaml"))
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--point_size", type=float, default=0.015)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Load config and build models
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Building models from config...")
    mdls = build_models(cfg["models"])
    encoder = mdls["encoder"]
    decoder = mdls["decoder"]

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"]
    enc_state = {k[len("models.encoder."):]: v for k, v in state.items() if k.startswith("models.encoder.")}
    dec_state = {k[len("models.decoder."):]: v for k, v in state.items() if k.startswith("models.decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.eval().float().to(device)
    decoder.eval().float().to(device)
    print("  Loaded encoder + decoder")

    # Load dataset samples
    df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
    shas = list(df.index)
    print(f"Samples: {len(shas)}")

    all_samples = {}
    l1_vals = []

    for si, sha in enumerate(shas):
        print(f"\n[{si+1}/{len(shas)}] {sha[:16]}...")

        coords, emit_attr = o_voxel.io.read_vxz(
            f"{DATA_ROOT}/emission_voxels_256/{sha}.vxz", num_threads=4)
        emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0

        y = reconstruct_finetuned(encoder, decoder, coords, emit_feats)
        rec_feats = y.feats.cpu()
        l1 = float((emit_feats - rec_feats).abs().mean().item())
        l1_vals.append(l1)
        print(f"  L1={l1:.6f}  ({coords.shape[0]:,} voxels)")

        all_coords_f = coords.float()
        norm_center = (all_coords_f.min(dim=0).values + all_coords_f.max(dim=0).values) / 2.0
        norm_scale = float((all_coords_f.max(dim=0).values - all_coords_f.min(dim=0).values).max().item())
        if norm_scale == 0:
            norm_scale = 1.0

        all_samples[sha] = {
            "gt":  voxels_to_json(coords, emit_feats, args.threshold, args.max_points),
            "rec": voxels_to_json(coords, rec_feats, args.threshold, args.max_points),
            "l1": l1,
            "norm": {"center": norm_center.tolist(), "scale": norm_scale},
        }
        torch.cuda.empty_cache()

    del encoder, decoder
    torch.cuda.empty_cache()

    mean_l1 = np.mean(l1_vals)
    stats = f"{len(shas)} samples &mdash; Mean L1 = {mean_l1:.5f}"

    sample_options = "".join(
        f'<option value="{sha}">[{i}] {sha[:16]}... (L1={all_samples[sha]["l1"]:.4f})</option>'
        for i, sha in enumerate(shas)
    )

    data_blob = json.dumps({
        "samples": all_samples,
        "sample_list": shas,
    }, separators=(",", ":"))

    html = HTML_TEMPLATE.format(
        stats=stats,
        sample_options=sample_options,
        data_json=data_blob,
        point_size=args.point_size,
    )

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "visualizations",
        "pbr_finetune_emission_3d.html",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nSaved: {out_path}  ({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
