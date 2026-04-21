#!/usr/bin/env python3
"""
Compare emission voxels from two paths:
1. Blender dump → blender_dump_to_volumetric_attr (current pipeline)
2. Direct GLB → textured_mesh_to_volumetric_attr (trimesh, no Blender)
"""
import sys, os, types, pickle, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import numpy as np
import torch
import pandas as pd
from PIL import Image

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import o_voxel.convert

DATA_ROOT = "/local-scratch/localhome/dya78/lightgen_test"
GLB_ROOT  = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
SHA = "00192b5a4a3249c79141c8dccaad2947"
RESOLUTION = 256

df = pd.read_parquet(f"{DATA_ROOT}/df_10samples.parquet")
glb_rel = df.loc[SHA, "glb_1k_path"]
glb_path = os.path.join(GLB_ROOT, glb_rel)

# ── Path 1: Current pipeline (Blender dump, with fix) ────────────────────────
print("=" * 60)
print("Path 1: Blender dump → blender_dump_to_volumetric_attr")
print("=" * 60)
dump_path = f"{DATA_ROOT}/pbr_dumps/{SHA}.pickle"
with open(dump_path, "rb") as f:
    dump = pickle.load(f)

for mat in dump["materials"]:
    if mat.get("alphaTexture") is not None and mat.get("alphaMode") == "OPAQUE":
        mat["alphaMode"] = "BLEND"
    etex = mat.get("emissiveColorTexture")
    if etex is not None:
        img = np.array(Image.open(io.BytesIO(etex["image"])))
        if img.max() == 0:
            mat["emissiveColorFactor"] = [0.0, 0.0, 0.0]
            mat["emissiveColorTexture"] = None
dump["materials"].append({
    "baseColorFactor": [0.8, 0.8, 0.8], "alphaFactor": 1.0,
    "metallicFactor": 0.0, "roughnessFactor": 0.5,
    "alphaMode": "OPAQUE", "alphaCutoff": 0.5,
    "baseColorTexture": None, "alphaTexture": None,
    "metallicTexture": None, "roughnessTexture": None,
})
dump["objects"] = [obj for obj in dump["objects"]
                   if obj["vertices"].size != 0 and obj["faces"].size != 0]
all_verts = np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)
verts_t = torch.from_numpy(all_verts).float()
center = (verts_t.min(dim=0)[0] + verts_t.max(dim=0)[0]) / 2
scale = 0.99999 / (verts_t.max(dim=0)[0] - verts_t.min(dim=0)[0]).max()
for obj in dump["objects"]:
    obj["vertices"] = ((torch.from_numpy(obj["vertices"]).float() - center) * scale).numpy()
    obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1

coord1, attr1 = o_voxel.convert.blender_dump_to_volumetric_attr(
    dump, grid_size=RESOLUTION,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    mip_level_offset=0, verbose=False, timing=False,
)
e1 = attr1["emissive"].numpy()
nz1 = (e1.max(axis=1) > 0).sum()
print(f"  Voxels: {e1.shape[0]:,}, non-zero emission: {nz1:,} ({100*nz1/e1.shape[0]:.1f}%)")
print(f"  Emission mean (non-zero): R={e1[e1.max(1)>0,0].mean():.1f} G={e1[e1.max(1)>0,1].mean():.1f} B={e1[e1.max(1)>0,2].mean():.1f}")
# Color diversity (high = noisy)
if nz1 > 0:
    nz_pixels = e1[e1.max(axis=1) > 0].astype(float)
    channel_std = nz_pixels.std(axis=1).mean()
    print(f"  Per-voxel channel std (noise): {channel_std:.1f}")

# ── Path 2: Direct GLB → textured_mesh_to_volumetric_attr ────────────────────
print(f"\n{'='*60}")
print("Path 2: Direct GLB → textured_mesh_to_volumetric_attr")
print("=" * 60)
coord2, attr2 = o_voxel.convert.textured_mesh_to_volumetric_attr(
    glb_path, grid_size=RESOLUTION,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    mip_level_offset=0, verbose=False, timing=False,
)
e2 = attr2["emissive"].numpy()
nz2 = (e2.max(axis=1) > 0).sum()
print(f"  Voxels: {e2.shape[0]:,}, non-zero emission: {nz2:,} ({100*nz2/e2.shape[0]:.1f}%)")
print(f"  Emission mean (non-zero): R={e2[e2.max(1)>0,0].mean():.1f} G={e2[e2.max(1)>0,1].mean():.1f} B={e2[e2.max(1)>0,2].mean():.1f}")
if nz2 > 0:
    nz_pixels2 = e2[e2.max(axis=1) > 0].astype(float)
    channel_std2 = nz_pixels2.std(axis=1).mean()
    print(f"  Per-voxel channel std (noise): {channel_std2:.1f}")

# ── Compare ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Comparison")
print("=" * 60)
print(f"  Path 1 (Blender dump): {nz1:,} non-zero ({100*nz1/e1.shape[0]:.1f}%)")
print(f"  Path 2 (Direct GLB):   {nz2:,} non-zero ({100*nz2/e2.shape[0]:.1f}%)")

# Save both as vxz for visualization comparison
out_dir = "/localhome/dya78/code/TRELLIS.2-lightning/emission_texture_debug"
o_voxel.io.write_vxz(f"{out_dir}/{SHA}_blender.vxz", coord1, {"emissive": torch.from_numpy(e1)})
o_voxel.io.write_vxz(f"{out_dir}/{SHA}_trimesh.vxz", coord2, {"emissive": torch.from_numpy(e2)})
print(f"\n  Saved vxz files for visual comparison in {out_dir}/")
