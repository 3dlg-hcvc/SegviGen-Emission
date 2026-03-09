#!/usr/bin/env python3
"""
Generate dual-grid voxels from PBR dump mesh data and encode shape latent
for a single test sample.

Pipeline:  PBR dump → merge mesh → dual-grid voxelize → shape encoder → .npz
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, TRELLIS2_SITE)

import pickle
import torch
import numpy as np

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import o_voxel.convert

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)

SAMPLE = "0007deb6d96c4474b80faa5aa3888760"
DATA_ROOT = "/tmp/lightgen_test"
PBR_DUMP = os.path.join(DATA_ROOT, f"pbr_dumps/{SAMPLE}.pickle")
RESOLUTION = 256
SHAPE_LATENT_NAME = "shape_enc_next_dc_f16c32_fp16_256"

device = "cuda"

# ── 1. Load and merge mesh from PBR dump ──────────────────────────────────────
print("=== Loading mesh from PBR dump ===")
with open(PBR_DUMP, "rb") as f:
    dump = pickle.load(f)

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
assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), "vertices out of range"

print(f"  Merged mesh: {vertices.shape[0]} verts, {faces.shape[0]} faces")

# ── 2. Generate dual-grid voxels ──────────────────────────────────────────────
print(f"\n=== Generating dual-grid voxels (res={RESOLUTION}) ===")
voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices=vertices,
    faces=faces,
    grid_size=RESOLUTION,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    face_weight=1.0,
    boundary_weight=0.2,
    regularization_weight=1e-2,
)

dual_vertices = dual_vertices * RESOLUTION - voxel_indices
assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3)
dual_vertices = torch.clamp(dual_vertices, 0, 1)

print(f"  Dual-grid voxels: {voxel_indices.shape[0]} voxels")

# ── 3. Build SparseTensor inputs for shape encoder ────────────────────────────
print("\n=== Building shape encoder inputs ===")
vert_feats = dual_vertices.float()
batch_col = torch.zeros(voxel_indices.shape[0], 1, dtype=voxel_indices.dtype)
coords_batched = torch.cat([batch_col, voxel_indices], dim=-1)

vert_st = sp.SparseTensor(vert_feats, coords_batched)

intersected_feats = torch.cat([
    intersected[:, 0:1],
    intersected[:, 1:2],
    intersected[:, 2:3],
], dim=-1).bool()
inter_st = vert_st.replace(intersected_feats)

print(f"  Vertices SparseTensor: feats={vert_st.feats.shape}")
print(f"  Intersected SparseTensor: feats={inter_st.feats.shape}")

# ── 4. Encode with pretrained shape encoder ───────────────────────────────────
print("\n=== Encoding shape latent ===")
shape_encoder = models.from_pretrained(
    "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
).eval().float().to(device)

z_shape = shape_encoder(vert_st.to(device), inter_st.to(device))
print(f"  Shape latent: {z_shape.feats.shape}")

# ── 5. Save .npz ─────────────────────────────────────────────────────────────
save_dir = os.path.join(DATA_ROOT, "shape_latents", SHAPE_LATENT_NAME)
os.makedirs(save_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(save_dir, f"{SAMPLE}.npz"),
    feats=z_shape.feats.cpu().numpy().astype(np.float32),
    coords=z_shape.coords[:, 1:].cpu().numpy().astype(np.uint8),
)
print(f"  Saved: {save_dir}/{SAMPLE}.npz")

del shape_encoder
torch.cuda.empty_cache()

# ── 6. Verify coordinate alignment ───────────────────────────────────────────
print("\n=== Verification ===")
emit_lat = np.load(os.path.join(DATA_ROOT, "emission_latents/emission_vae_optB_overfit", f"{SAMPLE}.npz"))
pbr_lat = np.load(os.path.join(DATA_ROOT, "pbr_latents/tex_enc_next_dc_f16c32_fp16_256", f"{SAMPLE}.npz"))
shape_coords = z_shape.coords[:, 1:].cpu().numpy()
emit_coords = emit_lat["coords"]
pbr_coords = pbr_lat["coords"]

print(f"  Shape latent tokens:    {shape_coords.shape[0]}")
print(f"  Emission latent tokens: {emit_coords.shape[0]}")
print(f"  PBR latent tokens:      {pbr_coords.shape[0]}")
print(f"  Shape coords match PBR: {np.array_equal(shape_coords, pbr_coords)}")
print(f"  Shape coords match emission: {np.array_equal(shape_coords, emit_coords)}")
print(f"  Shape latent channels:  {z_shape.feats.shape[1]}")

print("\nDone!")
