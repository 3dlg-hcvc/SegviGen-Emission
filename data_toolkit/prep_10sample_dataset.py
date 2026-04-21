#!/usr/bin/env python3
"""
Prepare the 10-sample emission-VAE training dataset in a persistent location.
Runs steps 1-3 of the full encoding pipeline (Blender dump, voxelization,
parquet subset) — no latent encoding needed for VAE training.

Output root: /local-scratch/localhome/dya78/lightgen_test
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

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET    = "/localhome/dya78/code/lightgen/data/baked_uv_local_subset/df_SomgProc_emission_filtered.parquet"
GLB_ROOT   = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
OUT_ROOT   = "/local-scratch/localhome/dya78/lightgen_test"
RESOLUTION = 256
N_SAMPLES  = 10

BLENDER     = "/localhome/dya78/software/blender-4.2.0-linux-x64/blender"
DUMP_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_script/dump_pbr.py")

# ── Load parquet and pick N samples ──────────────────────────────────────────
df      = pd.read_parquet(PARQUET)
samples = [(sha, df.loc[sha]) for sha in list(df.index)[:N_SAMPLES]]
print(f"Processing {len(samples)} samples  →  {OUT_ROOT}")

# ── Create output directories ─────────────────────────────────────────────────
for d in [
    f"{OUT_ROOT}/pbr_dumps",
    f"{OUT_ROOT}/pbr_voxels_{RESOLUTION}",
    f"{OUT_ROOT}/emission_voxels_{RESOLUTION}",
]:
    os.makedirs(d, exist_ok=True)

# ── Step 1: Blender PBR dump ──────────────────────────────────────────────────
print("\n=== Step 1: Blender PBR dumps ===")
for sha, row in samples:
    dump_path = f"{OUT_ROOT}/pbr_dumps/{sha}.pickle"
    if os.path.exists(dump_path):
        print(f"  [skip] {sha[:16]}...")
        continue
    glb_path = os.path.join(GLB_ROOT, row["glb_1k_path"])
    print(f"  Dumping {sha[:16]}... ({os.path.basename(glb_path)})")
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
        print("  OK")

# ── Step 2: Voxelize PBR + emission ──────────────────────────────────────────
print("\n=== Step 2: Voxelization ===")
for sha, row in samples:
    pbr_vxz  = f"{OUT_ROOT}/pbr_voxels_{RESOLUTION}/{sha}.vxz"
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

    for mat in dump["materials"]:
        if mat["alphaTexture"] is not None and mat["alphaMode"] == "OPAQUE":
            mat["alphaMode"] = "BLEND"
        # Fix emission textures: Blender extracts them in sRGB, but factors
        # are in linear. Convert textures sRGB→linear for consistency.
        # Also zero out all-black textures to avoid factor leak.
        etex = mat.get("emissiveColorTexture")
        if etex is not None:
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(etex["image"]))
            arr = np.array(img)
            if arr.max() == 0:
                mat["emissiveColorFactor"] = [0.0, 0.0, 0.0]
                mat["emissiveColorTexture"] = None
            else:
                # sRGB → linear conversion
                f = arr.astype(np.float64) / 255.0
                linear = np.where(f <= 0.04045, f / 12.92,
                                  ((f + 0.055) / 1.055) ** 2.4)
                linear_u8 = np.clip(linear * 255, 0, 255).astype(np.uint8)
                linear_img = Image.fromarray(linear_u8, mode=img.mode)
                buf = _io.BytesIO()
                linear_img.save(buf, format="PNG")
                etex["image"] = buf.getvalue()
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
        verts_t   = torch.from_numpy(all_verts).float()
        center    = (verts_t.min(dim=0)[0] + verts_t.max(dim=0)[0]) / 2
        scale     = 0.99999 / (verts_t.max(dim=0)[0] - verts_t.min(dim=0)[0]).max()
        for obj in dump["objects"]:
            obj["vertices"] = ((torch.from_numpy(obj["vertices"]).float() - center) * scale).numpy()
            obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1

        coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
            dump, grid_size=RESOLUTION,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            mip_level_offset=0, verbose=False, timing=False,
        )
        attr.pop("normal", None)

        emissive = attr.pop("emissive", None)
        if emissive is not None:
            o_voxel.io.write_vxz(emit_vxz, coord, {"emissive": emissive})
        else:
            o_voxel.io.write_vxz(emit_vxz, coord,
                {"emissive": np.zeros((coord.shape[0], 3), dtype=np.uint8)})

        o_voxel.io.write_vxz(pbr_vxz, coord, attr)
        print(f"  OK: {coord.shape[0]} voxels")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# ── Step 3: Write subset parquet ─────────────────────────────────────────────
print("\n=== Step 3: Writing subset parquet ===")
subset_parquet = f"{OUT_ROOT}/df_10samples.parquet"
sha_list = [sha for sha, _ in samples]
df_sub = df.loc[sha_list]
df_sub.to_parquet(subset_parquet)
print(f"  Saved {len(df_sub)} rows  →  {subset_parquet}")

print("\nDone!")
