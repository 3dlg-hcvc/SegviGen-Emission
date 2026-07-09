#!/usr/bin/env python3
"""
Interactive 3D HTML visualizer for emission VAE overfit results.

Auto-discovers sample SHA, checkpoints, and data paths from a training
output directory. No hardcoded paths — everything is read from config.yaml.

Usage:
    # Default: scan outputs/emission_vae_1sample_overfit
    conda run -n trellis2 python vis/vis_3d_overfit.py

    # Custom output dir:
    conda run -n trellis2 python vis/vis_3d_overfit.py \
        --output_dir outputs/emission_vae_10sample_overfit

    # Pick specific checkpoints:
    conda run -n trellis2 python vis/vis_3d_overfit.py --ckpts last stepstep=0005000

    # Adjust emission threshold and max voxel count:
    conda run -n trellis2 python vis/vis_3d_overfit.py --threshold 0.01 --max_points 50000
"""
import sys, os, types, argparse, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import torch
import pandas as pd
import yaml

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel

from trellis2.modules import sparse as sp
import trellis2.models as trellis_models
from trellis2.trainers.builders import build_models

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

def load_config(output_dir):
    """Load config.yaml from the output directory."""
    cfg_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.yaml in {output_dir}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_sample_shas(cfg):
    """Derive the sample SHA(s) from the dataset config."""
    ds_args = cfg["dataset"]["args"]
    parquet_file = ds_args["parquet_file"]
    df = pd.read_parquet(parquet_file)
    if "success" in df.columns:
        df = df[df["success"] == True]

    samples = list(df.index)
    split_indices = ds_args.get("split_indices")
    if split_indices is not None:
        if isinstance(split_indices, list):
            samples = [samples[i] for i in split_indices]
    return samples


def discover_checkpoints(output_dir, requested=None):
    """Find all checkpoints, return sorted list of (step, filename)."""
    ckpt_dir = os.path.join(output_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No ckpts/ directory in {output_dir}")

    ckpts = []
    for f in sorted(os.listdir(ckpt_dir)):
        if not f.endswith(".ckpt"):
            continue
        if f == "last.ckpt":
            state = torch.load(os.path.join(ckpt_dir, f), map_location="cpu", weights_only=False)
            step = int(state.get("global_step", 0))
            ckpts.append((step, f))
        elif f.startswith("stepstep="):
            step = int(f.replace("stepstep=", "").replace(".ckpt", ""))
            ckpts.append((step, f))

    # Deduplicate by step (prefer named checkpoint over last.ckpt)
    seen = {}
    for step, fname in ckpts:
        if step not in seen or fname != "last.ckpt":
            seen[step] = fname
    ckpts = sorted(seen.items())

    if requested:
        filtered = []
        for step, fname in ckpts:
            if fname in requested or f"step{step}" in requested:
                filtered.append((step, fname))
        if filtered:
            ckpts = filtered

    return ckpts


# ---------------------------------------------------------------------------
# Model loading and reconstruction
# ---------------------------------------------------------------------------

def load_models(output_dir, ckpt_name, pbr_encoder_cache=None):
    """Load encoder + decoder from a checkpoint. Reuse PBR encoder if cached."""
    cfg = load_config(output_dir)
    models = build_models(cfg["models"])
    encoder, decoder = models["encoder"], models["decoder"]

    ckpt_path = os.path.join(output_dir, "ckpts", ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    encoder.load_state_dict(
        {k.removeprefix("models.encoder."): v for k, v in state.items() if k.startswith("models.encoder.")}
    )
    decoder.load_state_dict(
        {k.removeprefix("models.decoder."): v for k, v in state.items() if k.startswith("models.decoder.")}
    )
    encoder.to(device).eval()
    decoder.to(device).eval()

    pbr_enc = pbr_encoder_cache
    if pbr_enc is None:
        pbr_name = cfg["trainer"]["args"].get("pretrained_pbr_encoder")
        if pbr_name:
            pbr_enc = trellis_models.from_pretrained(pbr_name).eval().float().to(device)

    return encoder, decoder, pbr_enc


def load_gt_voxels(cfg, sha):
    """Load GT emission and PBR voxels for a sample."""
    ds_args = cfg["dataset"]["args"]
    data_root = ds_args["data_root"]
    resolution = ds_args.get("resolution", 256)
    pbr_attrs = ds_args.get("pbr_attrs", ["base_color", "metallic", "roughness", "alpha"])

    coords, emit_attr = o_voxel.io.read_vxz(
        os.path.join(data_root, f"emission_voxels_{resolution}", f"{sha}.vxz"), num_threads=4
    )
    emit_feats = emit_attr["emissive"].float() / 255.0 * 2.0 - 1.0

    _, pbr_attr = o_voxel.io.read_vxz(
        os.path.join(data_root, f"pbr_voxels_{resolution}", f"{sha}.vxz"), num_threads=4
    )
    pbr_feats = torch.cat(
        [pbr_attr[a].float() / 255.0 * 2.0 - 1.0 for a in pbr_attrs], dim=-1
    )

    return coords, emit_feats, pbr_feats


@torch.no_grad()
def reconstruct(encoder, decoder, pbr_encoder, coords, emit_feats, pbr_feats):
    """Run emission VAE encode-decode to get reconstruction."""
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    combined = torch.cat([emit_feats, pbr_feats], dim=-1)
    x0 = sp.SparseTensor(combined, torch.cat([batch_col, coords], dim=-1))
    x0_emit = x0.replace(emit_feats)
    x0_gpu = x0.to(device)

    z_emit = encoder(x0_emit.to(device))
    if pbr_encoder is not None:
        z_pbr = pbr_encoder(x0_gpu.replace(x0_gpu.feats[:, emit_feats.shape[1]:]))
        z_comb = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
    else:
        z_comb = z_emit
    return decoder(z_comb).to("cpu")


# ---------------------------------------------------------------------------
# Voxel data -> JSON for embedding in HTML
# ---------------------------------------------------------------------------

def emission_to_srgb(feats_norm):
    """Convert [-1,1] emission to sRGB [0,255]."""
    linear = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    return (srgb.clamp(0, 1) * 255).byte()


def select_gt_indices(feats, threshold, max_points):
    """Pick voxel indices on the GT based on emission magnitude + optional cap."""
    mag = (feats * 0.5 + 0.5).norm(dim=-1)
    idx = torch.where(mag > threshold)[0]
    if idx.shape[0] > max_points:
        perm = torch.randperm(idx.shape[0])[:max_points]
        idx = idx[perm]
    return idx


def voxels_to_json(coords, feats, idx):
    """Serialize the voxels at `idx` (coords + sRGB colors) to a JSON dict."""
    c = coords[idx].long().numpy()
    rgb = emission_to_srgb(feats[idx]).numpy()
    return {
        "xyz": c.flatten().tolist(),
        "rgb": rgb.flatten().tolist(),
        "n": int(idx.shape[0]),
        "total": int(coords.shape[0]),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Emission VAE 3D — {title}</title>
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
  .controls .badge {{
    border-radius: 4px; padding: 4px 10px; font-size: 13px;
  }}
  .controls .l1-badge {{ background: #1a2a1a; border: 1px solid #2a4a2a; color: #7c7; }}
  .controls .info-badge {{ background: #1a1a2a; border: 1px solid #2a2a4a; color: #99f; }}
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
  <h2>Emission VAE 3D — {title}</h2>
  <div class="stats">{stats}</div>
</div>
<div class="controls">
  <label>Sample:</label>
  <select id="sample-select">{sample_options}</select>
  <label>Checkpoint:</label>
  <select id="ckpt-select">{ckpt_options}</select>
  <span class="badge l1-badge" id="l1-display"></span>
  <span class="badge info-badge" id="info-display"></span>
</div>
<div class="viewer-row">
  <div class="viewer-panel" id="gt-panel">
    <div class="panel-label gt">Ground Truth</div>
    <canvas id="gt-canvas"></canvas>
  </div>
  <div class="viewer-panel" id="rec-panel">
    <div class="panel-label rec">Reconstructed</div>
    <canvas id="rec-canvas"></canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {{
  var DATA = {data_json};
  console.log("Samples:", DATA.sample_list, "Checkpoints:", DATA.ckpt_list);

  // Voxel coords are (X, Y, Z) with Z-up; Three.js uses Y-up, so swap Y↔Z.
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

  // Sync cameras
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

  // State
  var curGtCloud = null;
  var curRecCloud = null;
  var curSample = null;
  var sampleSel = document.getElementById('sample-select');
  var ckptSel   = document.getElementById('ckpt-select');
  var l1d       = document.getElementById('l1-display');
  var infod     = document.getElementById('info-display');

  function loadSample(sha) {{
    curSample = sha;
    var sd = DATA.samples[sha];

    // Replace GT
    if (curGtCloud) {{ gtV.scene.remove(curGtCloud); disposeCloud(curGtCloud); }}
    curGtCloud = buildPointCloud(sd.gt, sd.norm);
    gtV.scene.add(curGtCloud);
    frameCameraOn(curGtCloud);

    infod.textContent = sd.gt.n + ' emission voxels / ' + sd.gt.total + ' total';

    // Load current checkpoint
    loadCkpt(ckptSel.value);
  }}

  function loadCkpt(key) {{
    if (!curSample) return;
    var sd = DATA.samples[curSample];
    if (curRecCloud) {{ recV.scene.remove(curRecCloud); disposeCloud(curRecCloud); }}
    curRecCloud = buildPointCloud(sd.checkpoints[key], sd.norm);
    recV.scene.add(curRecCloud);
    l1d.textContent = 'L1 = ' + sd.l1[key].toFixed(5);
  }}

  sampleSel.addEventListener('change', function() {{ loadSample(sampleSel.value); }});
  ckptSel.addEventListener('change', function() {{ loadCkpt(ckptSel.value); }});

  // Initialize: last checkpoint, first sample
  ckptSel.value = ckptSel.options[ckptSel.options.length - 1].value;
  loadSample(sampleSel.value);
}})();
</script>
</body>
</html>"""


def generate_html(output_dir, out_path, threshold, max_points, point_size, ckpts_filter):
    """Main entry: scan output dir, reconstruct all samples, generate single HTML."""
    cfg = load_config(output_dir)
    shas = resolve_sample_shas(cfg)
    ckpts = discover_checkpoints(output_dir, ckpts_filter)

    print(f"Output dir : {output_dir}")
    print(f"Samples    : {len(shas)} — {shas}")
    print(f"Checkpoints: {[(s, f) for s, f in ckpts]}")

    # Load models once per checkpoint, reuse across samples
    model_cache = {}
    pbr_enc_cache = None
    for step, ckpt_name in ckpts:
        enc, dec, pbr_enc_cache = load_models(output_dir, ckpt_name, pbr_enc_cache)
        model_cache[step] = (enc, dec, pbr_enc_cache)

    all_samples = {}
    for si, sha in enumerate(shas):
        coords, emit_feats, pbr_feats = load_gt_voxels(cfg, sha)
        emit_mag = (emit_feats * 0.5 + 0.5).norm(dim=-1)
        n_nonzero = int((emit_mag > threshold).sum().item())

        # Per-sample normalization
        all_coords_f = coords.float()
        norm_center = (all_coords_f.min(dim=0).values + all_coords_f.max(dim=0).values) / 2.0
        norm_scale = float((all_coords_f.max(dim=0).values - all_coords_f.min(dim=0).values).max().item())
        if norm_scale == 0:
            norm_scale = 1.0

        print(f"\n[{si+1}/{len(shas)}] {sha}")
        print(f"  Voxels: {coords.shape[0]:,}  Non-zero emission: {n_nonzero:,}")

        gt_idx = select_gt_indices(emit_feats, threshold, max_points)
        gt_json = voxels_to_json(coords, emit_feats, gt_idx)
        ckpt_data = {}
        l1_data = {}

        for step, ckpt_name in ckpts:
            enc, dec, pbr_enc = model_cache[step]
            print(f"  step {step}...", end=" ", flush=True)
            y = reconstruct(enc, dec, pbr_enc, coords, emit_feats, pbr_feats)
            rec_feats = y.feats.cpu()
            l1 = float((emit_feats - rec_feats).abs().mean().item())
            print(f"L1={l1:.5f}")

            key = f"step_{step}"
            ckpt_data[key] = voxels_to_json(coords, rec_feats, gt_idx)
            l1_data[key] = l1
            torch.cuda.empty_cache()

        all_samples[sha] = {
            "gt": gt_json,
            "checkpoints": ckpt_data,
            "l1": l1_data,
            "norm": {"center": norm_center.tolist(), "scale": norm_scale},
        }

    # Build HTML
    data_blob = json.dumps({
        "samples": all_samples,
        "sample_list": shas,
        "ckpt_list": [f"step_{s}" for s, _ in ckpts],
    }, separators=(",", ":"))

    sample_options = "".join(
        f'<option value="{sha}">[{i}] {sha}</option>'
        for i, sha in enumerate(shas)
    )
    ckpt_options = "".join(
        f'<option value="step_{step}">Step {step:,} ({fname})</option>'
        for step, fname in ckpts
    )
    stats = f"{len(shas)} samples &mdash; {len(ckpts)} checkpoints"

    html = HTML_TEMPLATE.format(
        title=os.path.basename(output_dir),
        stats=stats,
        sample_options=sample_options,
        ckpt_options=ckpt_options,
        data_json=data_blob,
        point_size=point_size,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nSaved: {out_path}  ({len(html)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D HTML comparing GT vs reconstructed emission voxels."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/emission_vae_1sample_overfit",
        help="Training output directory (must contain config.yaml and ckpts/)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output HTML path. Default: visualizations/<run_name>.html",
    )
    parser.add_argument(
        "--ckpts", nargs="*", default=None,
        help="Specific checkpoint filenames to include (default: all)",
    )
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Min emission magnitude to display (default: 0.005)")
    parser.add_argument("--max_points", type=int, default=80000,
                        help="Max voxels per point cloud (default: 80000)")
    parser.add_argument("--point_size", type=float, default=0.015,
                        help="Three.js point size (default: 0.015)")
    args = parser.parse_args()

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "visualizations",
        f"emission_vae_3d_{run_name}.html",
    )

    generate_html(
        output_dir=args.output_dir,
        out_path=out_path,
        threshold=args.threshold,
        max_points=args.max_points,
        point_size=args.point_size,
        ckpts_filter=args.ckpts,
    )


if __name__ == "__main__":
    main()
