"""
End-to-end smoke test for the emission voxelization pipeline on a single sample.

Runs:
  1. Blender PBR dump (now with emission extraction)
  2. o_voxel voxelization -> pbr_voxels + emission_voxels
  3. Inspection: print emission stats and compare to UV-baked ground truth

Usage:
    cd /localhome/dya78/code/TRELLIS.2-lightning
    python data_toolkit/test_single_emission_sample.py

Optional flags:
    --sample_id UUID    Use a specific sample UUID (default: first train sample)
    --out_dir PATH      Output directory (default: data/lightgen_test)
    --blender PATH      Path to Blender binary
    --resolution INT    Voxel grid resolution (default: 256)
    --no_compare        Skip comparison with UV-baked data
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_ROOT    = Path("/localhome/dya78/code/lightgen/data")
GLB_ROOT     = DATA_ROOT / "TexVerse-1K"
UV_ROOT      = DATA_ROOT / "baked_uv_local_subset"
PARQUET_FILE = UV_ROOT / "df_SomgProc_emission_filtered.parquet"
SPLIT_JSON   = UV_ROOT / "data_splits_emission_filtered.json"
BLENDER_BIN  = "/localhome/dya78/software/blender-4.2.0-linux-x64/blender"
DUMP_SCRIPT  = REPO_ROOT / "data_toolkit" / "blender_script" / "dump_pbr.py"


# ---------------------------------------------------------------------------
def find_sample(sample_id):
    df = pd.read_parquet(PARQUET_FILE)
    if sample_id is None:
        with open(SPLIT_JSON) as f:
            splits = json.load(f)
        first_idx = splits["train"]["indices"][0]
        sample_id = str(list(df.index)[first_idx])
    row = df.loc[sample_id]
    glb_path = GLB_ROOT / row["glb_1k_path"]
    return sample_id, glb_path, row


def run_blender_dump(glb_path, out_pickle, blender_bin):
    out_pickle.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_bin, "--background",
        "--python", str(DUMP_SCRIPT),
        "--",
        "--object",      str(glb_path),
        "--output_path", str(out_pickle),
    ]
    print("\n[1/3] Running Blender dump...")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT (last 3000):", result.stdout[-3000:])
        print("STDERR (last 2000):", result.stderr[-2000:])
        raise RuntimeError(f"Blender dump failed (exit {result.returncode})")
    # Show last 20 lines of blender stdout for [INFO] messages
    lines = result.stdout.strip().splitlines()
    for l in lines[-20:]:
        print("  blender:", l)
    print("  OK - Blender dump complete")


def inspect_dump(pickle_path):
    with open(pickle_path, "rb") as f:
        dump = pickle.load(f)
    print(f"\n  Dump materials ({len(dump['materials'])}):")
    any_emission = False
    for i, mat in enumerate(dump["materials"]):
        ec  = mat.get("emissiveColorFactor", "MISSING")
        es  = mat.get("emissiveStrengthFactor", "MISSING")
        etx = "YES" if mat.get("emissiveColorTexture") else "no"
        print(f"    [{i}] emissiveColor={ec}  strength={es}  texture={etx}")
        if isinstance(ec, list) and max(ec) > 0.001:
            any_emission = True
        if isinstance(es, float) and es > 0.0 and mat.get("emissiveColorTexture"):
            any_emission = True
    if not any_emission:
        print("  NOTE: All emissiveColorFactors are zero/black — this object may have no emission")
        print("        (this is OK; the emissive voxel channel will be all zeros)")
    return dump


def _import_o_voxel():
    """Import o_voxel from the trellis2 conda env.

    o_voxel.__init__ transitively imports triton (via postprocess -> flex_gemm),
    which requires CUDA.  We stub o_voxel.postprocess, o_voxel.rasterize, and
    o_voxel.serialize directly - these are the sub-modules we do not need for
    plain voxelization (convert + io).
    """
    import types

    TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
    if TRELLIS2_SITE not in sys.path:
        sys.path.insert(0, TRELLIS2_SITE)

    # Import torch first so PyTorch shared libs (libc10.so etc.) are in memory.
    import torch  # noqa: F401

    # Stub the GPU sub-modules of o_voxel BEFORE importing o_voxel itself.
    # When o_voxel/__init__.py does "from . import postprocess", Python will
    # find our stub in sys.modules and skip the real (GPU-requiring) code.
    for name in ("o_voxel.postprocess", "o_voxel.rasterize"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    import o_voxel
    return o_voxel


def run_voxelization(dump, uuid, out_dir, resolution):
    o_voxel = _import_o_voxel()

    # Add default fallback material (mirrors voxelize_pbr.py)
    dump = dict(dump)
    dump["materials"] = list(dump["materials"])
    for mat in dump["materials"]:
        if mat.get("alphaTexture") is not None and mat.get("alphaMode") == "OPAQUE":
            mat["alphaMode"] = "BLEND"
    dump["materials"].append({
        "baseColorFactor": [0.8, 0.8, 0.8],
        "alphaFactor": 1.0,
        "metallicFactor": 0.0,
        "roughnessFactor": 0.5,
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "baseColorTexture": None,
        "alphaTexture": None,
        "metallicTexture": None,
        "roughnessTexture": None,
        "emissiveColorFactor": [0.0, 0.0, 0.0],
        "emissiveStrengthFactor": 1.0,
        "emissiveColorTexture": None,
    })
    dump["objects"] = [
        obj for obj in dump["objects"]
        if obj["vertices"].size != 0 and obj["faces"].size != 0
    ]

    # Normalise vertices to [-0.5, 0.5]
    verts_all = np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)
    vmin, vmax = verts_all.min(0), verts_all.max(0)
    center = (vmin + vmax) / 2
    scale  = 0.99999 / (vmax - vmin).max()
    for obj in dump["objects"]:
        obj["vertices"] = ((obj["vertices"] - center) * scale).astype(np.float32)
        obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1
        assert obj["vertices"].min() >= -0.5 and obj["vertices"].max() <= 0.5

    print(f"\n[2/3] Voxelizing at resolution={resolution}...")
    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=True,
        timing=True,
    )
    del attr["normal"]
    print(f"  Attribute keys after voxelization: {list(attr.keys())}")

    # Save emission voxels separately
    emit_dir = out_dir / f"emission_voxels_{resolution}"
    emit_dir.mkdir(parents=True, exist_ok=True)
    emit_path = emit_dir / f"{uuid}.vxz"
    if "emissive" in attr:
        emit_attr = {"emissive": attr.pop("emissive")}
        o_voxel.io.write_vxz(str(emit_path), coord, emit_attr)
        print(f"  OK - Emission voxels -> {emit_path}")
    else:
        print("  WARNING: 'emissive' not produced by o_voxel — writing zeros")
        emit_attr = {"emissive": np.zeros((len(coord), 3), dtype=np.uint8)}
        o_voxel.io.write_vxz(str(emit_path), coord, emit_attr)

    # Save PBR voxels (base_color, metallic, roughness, alpha)
    pbr_dir = out_dir / f"pbr_voxels_{resolution}"
    pbr_dir.mkdir(parents=True, exist_ok=True)
    pbr_path = pbr_dir / f"{uuid}.vxz"
    o_voxel.io.write_vxz(str(pbr_path), coord, attr)
    print(f"  OK - PBR voxels     -> {pbr_path}")

    return emit_path, pbr_path


def inspect_emission(emit_path, row):
    o_voxel = _import_o_voxel()

    print("\n[3/3] Emission voxel inspection...")
    coord, attr = o_voxel.io.read_vxz(str(emit_path), num_threads=1)
    emissive = attr["emissive"].float() / 255.0         # [N, 3], range [0, 1]

    n_total   = len(coord)
    n_emissive = (emissive.max(dim=-1).values > 0.01).sum().item()
    emit_max   = emissive.max().item()
    emit_mean  = (emissive[emissive.max(dim=-1).values > 0.01].mean().item()
                  if n_emissive > 0 else 0.0)

    print(f"  Total voxels            : {n_total:,}")
    print(f"  Emissive voxels (>1%)   : {n_emissive:,}  ({100*n_emissive/max(n_total,1):.1f}%)")
    print(f"  Emission max value      : {emit_max:.4f}")
    print(f"  Emission mean (nonzero) : {emit_mean:.4f}")

    # Compare against parquet stats (from UV pipeline)
    print(f"\n  Parquet reference stats:")
    print(f"    #stat_emission_color_max  = {row.get('#stat_emission_color_max', 'N/A')}")
    print(f"    #stat_emission_color_mean = {row.get('#stat_emission_color_mean', 'N/A')}")


def compare_uv(uuid, row):
    ditem_dir = row.get("ditem_dir", f"{uuid[:3]}-{uuid[3:6]}/{uuid}")
    npz_file  = UV_ROOT / ditem_dir / "somage.npz"
    if not npz_file.exists():
        print(f"\n  (somage.npz not found at {npz_file}, skipping UV comparison)")
        return
    npz = np.load(str(npz_file))
    emit_uv = npz["emission_color"].astype(np.float32) / 255.0  # [H, W, 3]
    mask    = npz["occupancy"].squeeze(-1).astype(bool)
    npz.close()
    vals = emit_uv[mask]
    n_emit = (vals.max(axis=-1) > 0.01).sum()
    n_total = mask.sum()
    print(f"\n  UV-baked comparison (somage.npz):")
    print(f"    Occupied UV pixels      : {n_total:,}")
    print(f"    Emissive UV pixels (>1%): {n_emit:,}  ({100*n_emit/max(n_total,1):.1f}%)")
    print(f"    UV emission max         : {vals.max():.4f}")
    if n_emit > 0:
        print(f"    UV emission mean (nz)   : {vals[vals.max(axis=-1)>0.01].mean():.4f}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id",  default=None)
    parser.add_argument("--out_dir",    default="data/lightgen_test")
    parser.add_argument("--blender",    default=BLENDER_BIN)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--no_compare", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. Identify sample
    uuid, glb_path, row = find_sample(args.sample_id)
    print("=" * 60)
    print(f"Sample UUID : {uuid}")
    print(f"GLB path    : {glb_path}")
    print(f"GLB exists  : {glb_path.exists()}")
    print(f"Output dir  : {out_dir}")
    print("=" * 60)
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    # 1. Blender dump
    pickle_path = out_dir / "pbr_dumps" / f"{uuid}.pickle"
    run_blender_dump(glb_path, pickle_path, args.blender)
    dump = inspect_dump(pickle_path)

    # 2. Voxelisation
    emit_path, pbr_path = run_voxelization(dump, uuid, out_dir, args.resolution)

    # 3. Inspection + comparison
    inspect_emission(emit_path, row)
    if not args.no_compare:
        compare_uv(uuid, row)

    print(f"\n{'='*60}")
    print("Smoke test COMPLETE")
    print(f"  PBR voxels  : {pbr_path}")
    print(f"  Emission    : {emit_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
