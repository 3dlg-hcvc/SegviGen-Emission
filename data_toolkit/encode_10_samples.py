#!/usr/bin/env python3
"""
Full encoding pipeline for 10 samples:
  1. Blender PBR dump (mesh + PBR material textures → pickle)
  2. PBR voxelization + emission voxelization
  3. Dual-grid voxelization (for shape encoder)
  4. PBR latent encoding   (pretrained PBR encoder)
  5. Emission latent encoding (trained emission VAE encoder)
  6. Shape latent encoding  (pretrained shape encoder)

Output root: data/lightgen_test/
"""
import sys, os, types, subprocess, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import numpy as np
import pandas as pd
import torch

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import o_voxel.convert

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET = "/localhome/dya78/code/lightgen/data/baked_uv_local_subset/df_SomgProc_emission_filtered.parquet"
GLB_ROOT = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
THUMB_ROOT = "/localhome/dya78/code/lightgen/data/TexVerse/thumbnails/thumbnails_batch"
OUT_ROOT = "data/lightgen_test"
RESOLUTION = 256
N_SAMPLES = 10

BLENDER = "/localhome/dya78/software/blender-4.2.0-linux-x64/blender"
DUMP_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_script/dump_pbr.py")

EMIT_VAE_CKPT = "/tmp/emission_vae_optB_overfit/ckpts/stepstep=0001000.ckpt"
EMIT_VAE_CFG = "/tmp/emission_vae_optB_overfit/config.yaml"
PBR_ENC_NAME = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
SHAPE_ENC_NAME = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"

EMIT_LAT_NAME = "emission_vae_optB_overfit"
PBR_LAT_NAME = f"tex_enc_next_dc_f16c32_fp16_{RESOLUTION}"
SHAPE_LAT_NAME = f"shape_enc_next_dc_f16c32_fp16_{RESOLUTION}"

device = "cuda"


# ── Load parquet and pick 10 samples ─────────────────────────────────────────
df = pd.read_parquet(PARQUET)
samples = [(sha, df.loc[sha]) for sha in list(df.index)[:N_SAMPLES]]
print(f"Processing {len(samples)} samples")


# ── Create output directories ─────────────────────────────────────────────────
for d in [
    f"{OUT_ROOT}/pbr_dumps",
    f"{OUT_ROOT}/pbr_voxels_{RESOLUTION}",
    f"{OUT_ROOT}/emission_voxels_{RESOLUTION}",
    f"{OUT_ROOT}/thumbnails",
    f"{OUT_ROOT}/pbr_latents/{PBR_LAT_NAME}",
    f"{OUT_ROOT}/emission_latents/{EMIT_LAT_NAME}",
    f"{OUT_ROOT}/shape_latents/{SHAPE_LAT_NAME}",
]:
    os.makedirs(d, exist_ok=True)


# ── Step 1: Blender PBR dump ──────────────────────────────────────────────────
print("\n=== Step 1: Blender PBR dumps ===")
for sha, row in samples:
    dump_path = f"{OUT_ROOT}/pbr_dumps/{sha}.pickle"
    if os.path.exists(dump_path):
        print(f"  [skip] {sha[:16]}... (already exists)")
        continue
    glb_path = os.path.join(GLB_ROOT, row["glb_1k_path"])
    print(f"  Dumping {sha[:16]}... (GLB: {os.path.basename(glb_path)})")
    cmd = [
        BLENDER, "--background",
        "--python", DUMP_SCRIPT,
        "--",
        "--object", glb_path,
        "--output_path", dump_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Blender failed for {sha}")
        print(result.stderr[-1000:])
    else:
        print(f"  OK")


# ── Step 2: Voxelize PBR + emission ──────────────────────────────────────────
print("\n=== Step 2: Voxelization ===")
for sha, row in samples:
    pbr_vxz = f"{OUT_ROOT}/pbr_voxels_{RESOLUTION}/{sha}.vxz"
    emit_vxz = f"{OUT_ROOT}/emission_voxels_{RESOLUTION}/{sha}.vxz"
    if os.path.exists(pbr_vxz) and os.path.exists(emit_vxz):
        print(f"  [skip] {sha[:16]}...")
        continue

    dump_path = f"{OUT_ROOT}/pbr_dumps/{sha}.pickle"
    if not os.path.exists(dump_path):
        print(f"  [skip] {sha[:16]}... no dump")
        continue

    print(f"  Voxelizing {sha[:16]}...")
    with open(dump_path, "rb") as f:
        dump = pickle.load(f)

    # Fix alpha mode
    for mat in dump["materials"]:
        if mat["alphaTexture"] is not None and mat["alphaMode"] == "OPAQUE":
            mat["alphaMode"] = "BLEND"
    dump["materials"].append({
        "baseColorFactor": [0.8, 0.8, 0.8], "alphaFactor": 1.0,
        "metallicFactor": 0.0, "roughnessFactor": 0.5,
        "alphaMode": "OPAQUE", "alphaCutoff": 0.5,
        "baseColorTexture": None, "alphaTexture": None,
        "metallicTexture": None, "roughnessTexture": None,
    })
    dump["objects"] = [obj for obj in dump["objects"]
                       if obj["vertices"].size != 0 and obj["faces"].size != 0]

    try:
        all_verts = np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)
        import torch as _t
        verts_t = _t.from_numpy(all_verts).float()
        center = (verts_t.min(dim=0)[0] + verts_t.max(dim=0)[0]) / 2
        scale = 0.99999 / (verts_t.max(dim=0)[0] - verts_t.min(dim=0)[0]).max()
        for obj in dump["objects"]:
            obj["vertices"] = ((_t.from_numpy(obj["vertices"]).float() - center) * scale).numpy()
            obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1

        coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
            dump, grid_size=RESOLUTION,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            mip_level_offset=0, verbose=False, timing=False,
        )
        attr.pop("normal", None)

        # Emission vxz
        emissive = attr.pop("emissive", None)
        if emissive is not None:
            o_voxel.io.write_vxz(emit_vxz, coord, {"emissive": emissive})
        else:
            o_voxel.io.write_vxz(emit_vxz, coord,
                {"emissive": torch.zeros(coord.shape[0], 3, dtype=torch.uint8)})

        # PBR vxz
        o_voxel.io.write_vxz(pbr_vxz, coord, attr)

        print(f"  OK: {coord.shape[0]} voxels")
    except Exception as e:
        print(f"  ERROR voxelizing {sha}: {e}")
        import traceback; traceback.print_exc()


# ── Step 3: Copy thumbnails ───────────────────────────────────────────────────
print("\n=== Step 3: Copy thumbnails ===")
import shutil
for sha, row in samples:
    src = os.path.join(THUMB_ROOT, row["thumbnail_path"])
    dst = f"{OUT_ROOT}/thumbnails/{sha}.png"
    if os.path.exists(dst):
        continue
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {sha[:16]}...")
    else:
        print(f"  MISSING thumbnail: {src}")


# ── Step 4-6: Encode latents ──────────────────────────────────────────────────
print("\n=== Step 4: Loading encoders ===")

# PBR encoder
pbr_encoder = models.from_pretrained(PBR_ENC_NAME).eval().float().to(device)
print(f"  PBR encoder loaded")

# Emission VAE encoder
import yaml
from trellis2.trainers.builders import build_models
with open(EMIT_VAE_CFG) as f:
    vae_cfg = yaml.safe_load(f)
vae_models = build_models(vae_cfg["models"])
emit_enc = vae_models["encoder"]
vae_ckpt = torch.load(EMIT_VAE_CKPT, map_location="cpu", weights_only=False)
enc_state = {k.removeprefix("models.encoder."): v
             for k, v in vae_ckpt["state_dict"].items()
             if k.startswith("models.encoder.")}
emit_enc.load_state_dict(enc_state)
emit_enc = emit_enc.eval().to(device)
print(f"  Emission encoder loaded")

# Shape encoder
shape_encoder = models.from_pretrained(SHAPE_ENC_NAME).eval().float().to(device)
print(f"  Shape encoder loaded")


print("\n=== Step 5-7: Encoding latents ===")
for sha, row in samples:
    print(f"\n  [{sha[:16]}...]")

    pbr_vxz = f"{OUT_ROOT}/pbr_voxels_{RESOLUTION}/{sha}.vxz"
    emit_vxz = f"{OUT_ROOT}/emission_voxels_{RESOLUTION}/{sha}.vxz"
    pbr_lat_path = f"{OUT_ROOT}/pbr_latents/{PBR_LAT_NAME}/{sha}.npz"
    emit_lat_path = f"{OUT_ROOT}/emission_latents/{EMIT_LAT_NAME}/{sha}.npz"
    shape_lat_path = f"{OUT_ROOT}/shape_latents/{SHAPE_LAT_NAME}/{sha}.npz"

    if not os.path.exists(pbr_vxz) or not os.path.exists(emit_vxz):
        print(f"    [skip] missing vxz files")
        continue

    # ── PBR latent ───────────────────────────────────────────────────────────
    if not os.path.exists(pbr_lat_path):
        try:
            coords, attr = o_voxel.io.read_vxz(pbr_vxz, num_threads=4)
            bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
            base_color = attr["base_color"].float() / 255.0 * 2.0 - 1.0
            metallic   = attr["metallic"].float()   / 255.0 * 2.0 - 1.0
            roughness  = attr["roughness"].float()  / 255.0 * 2.0 - 1.0
            alpha      = attr["alpha"].float()      / 255.0 * 2.0 - 1.0
            feats = torch.cat([base_color, metallic, roughness, alpha], dim=-1)
            x_pbr = sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))
            z_pbr = pbr_encoder(x_pbr.to(device))
            np.savez_compressed(pbr_lat_path,
                feats=z_pbr.feats.cpu().numpy().astype(np.float32),
                coords=z_pbr.coords[:, 1:].cpu().numpy().astype(np.uint8))
            print(f"    PBR latent: {z_pbr.feats.shape}")
        except Exception as e:
            print(f"    ERROR PBR latent: {e}")
    else:
        print(f"    PBR latent: [skip]")

    # ── Emission latent ──────────────────────────────────────────────────────
    if not os.path.exists(emit_lat_path):
        try:
            coords, attr = o_voxel.io.read_vxz(emit_vxz, num_threads=4)
            bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
            emissive = attr["emissive"].float() / 255.0 * 2.0 - 1.0
            x_emit = sp.SparseTensor(emissive, torch.cat([bc, coords], dim=-1))
            z_emit = emit_enc(x_emit.to(device))
            np.savez_compressed(emit_lat_path,
                feats=z_emit.feats.cpu().numpy().astype(np.float32),
                coords=z_emit.coords[:, 1:].cpu().numpy().astype(np.uint8))
            print(f"    Emission latent: {z_emit.feats.shape}")
        except Exception as e:
            print(f"    ERROR emission latent: {e}")
    else:
        print(f"    Emission latent: [skip]")

    # ── Shape latent ─────────────────────────────────────────────────────────
    if not os.path.exists(shape_lat_path):
        try:
            dump_path = f"{OUT_ROOT}/pbr_dumps/{sha}.pickle"
            with open(dump_path, "rb") as f:
                dump = pickle.load(f)

            # Merge mesh
            start = 0
            all_verts, all_faces = [], []
            for obj in dump["objects"]:
                if obj["vertices"].size == 0 or obj["faces"].size == 0:
                    continue
                all_verts.append(obj["vertices"])
                all_faces.append(obj["faces"] + start)
                start += len(obj["vertices"])

            vertices = torch.from_numpy(np.concatenate(all_verts, axis=0)).float()
            faces = torch.from_numpy(np.concatenate(all_faces, axis=0)).long()
            vmin = vertices.min(dim=0)[0]
            vmax = vertices.max(dim=0)[0]
            center = (vmin + vmax) / 2
            scale = 0.99999 / (vmax - vmin).max()
            vertices = (vertices - center) * scale

            # Dual-grid voxelization
            voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices=vertices, faces=faces,
                grid_size=RESOLUTION,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
            )
            dual_vertices = dual_vertices * RESOLUTION - voxel_indices
            dual_vertices = torch.clamp(dual_vertices, 0, 1)

            bc = torch.zeros(voxel_indices.shape[0], 1, dtype=voxel_indices.dtype)
            vert_st = sp.SparseTensor(dual_vertices.float(), torch.cat([bc, voxel_indices], dim=-1))
            inter_st = vert_st.replace(torch.cat([
                intersected[:, 0:1], intersected[:, 1:2], intersected[:, 2:3]
            ], dim=-1).bool())

            z_shape = shape_encoder(vert_st.to(device), inter_st.to(device))
            np.savez_compressed(shape_lat_path,
                feats=z_shape.feats.cpu().numpy().astype(np.float32),
                coords=z_shape.coords[:, 1:].cpu().numpy().astype(np.uint8))
            print(f"    Shape latent: {z_shape.feats.shape}")
        except Exception as e:
            print(f"    ERROR shape latent: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"    Shape latent: [skip]")

    torch.cuda.empty_cache()

print("\n=== Summary ===")
for sha, row in samples:
    pbr_ok = os.path.exists(f"{OUT_ROOT}/pbr_latents/{PBR_LAT_NAME}/{sha}.npz")
    emit_ok = os.path.exists(f"{OUT_ROOT}/emission_latents/{EMIT_LAT_NAME}/{sha}.npz")
    shape_ok = os.path.exists(f"{OUT_ROOT}/shape_latents/{SHAPE_LAT_NAME}/{sha}.npz")
    thumb_ok = os.path.exists(f"{OUT_ROOT}/thumbnails/{sha}.png")
    status = "OK" if (pbr_ok and emit_ok and shape_ok and thumb_ok) else "PARTIAL"
    print(f"  {sha[:20]}...  PBR={pbr_ok} emit={emit_ok} shape={shape_ok} thumb={thumb_ok}  [{status}]")

print("\nDone!")
