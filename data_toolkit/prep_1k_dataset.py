#!/usr/bin/env python3
"""
Prepare the ~1099-sample emission dataset for PBR VAE finetuning.

Runs Blender PBR dump + voxelization for all samples in the emission-filtered
parquet. Skips samples that already have voxel files. Supports parallelization
across cluster nodes via --rank / --world_size.

Output root: /local-scratch/localhome/dya78/lightgen_1k
  ├── df_SomgProc_emission_filtered.parquet   (copied from lightgen repo)
  ├── data_splits_emission_filtered.json      (copied from lightgen repo)
  ├── pbr_dumps/{sha256}.pickle
  ├── pbr_voxels_256/{sha256}.vxz
  └── emission_voxels_256/{sha256}.vxz

Usage:
    # Single machine (slow, ~1099 samples):
    conda run -n trellis2 python data_toolkit/prep_1k_dataset.py

    # Parallel across 4 nodes:
    conda run -n trellis2 python data_toolkit/prep_1k_dataset.py --rank 0 --world_size 4
    conda run -n trellis2 python data_toolkit/prep_1k_dataset.py --rank 1 --world_size 4
    ...
"""
import sys, os, types, subprocess, pickle, shutil, argparse

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
LIGHTGEN_DATA = "/local-scratch/localhome/dya78/code/lightgen/data/baked_uv_local_subset"
PARQUET    = os.path.join(LIGHTGEN_DATA, "df_SomgProc_emission_filtered.parquet")
SPLIT_JSON = os.path.join(LIGHTGEN_DATA, "data_splits_emission_filtered.json")
GLB_ROOT   = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
OUT_ROOT   = "/local-scratch/localhome/dya78/lightgen_1k"
RESOLUTION = 256

BLENDER     = "/localhome/dya78/software/blender-4.2.0-linux-x64/blender"
DUMP_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_script/dump_pbr.py")


def voxelize_sample(sha, row, out_root, resolution):
    """Run Blender dump + voxelization for a single sample. Returns True on success."""
    dump_path = f"{out_root}/pbr_dumps/{sha}.pickle"
    pbr_vxz  = f"{out_root}/pbr_voxels_{resolution}/{sha}.vxz"
    emit_vxz = f"{out_root}/emission_voxels_{resolution}/{sha}.vxz"

    # Skip if both voxel files exist
    if os.path.exists(pbr_vxz) and os.path.exists(emit_vxz):
        return True

    # Step 1: Blender PBR dump
    if not os.path.exists(dump_path):
        glb_path = os.path.join(GLB_ROOT, row["glb_1k_path"])
        if not os.path.exists(glb_path):
            print(f"  SKIP: GLB not found: {glb_path}")
            return False
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
            print(result.stderr[-500:])
            return False

    # Step 2: Voxelize
    if not os.path.exists(dump_path):
        return False

    with open(dump_path, "rb") as f:
        dump = pickle.load(f)

    # Fix materials (same as prep_10sample_dataset.py)
    for mat in dump["materials"]:
        if mat["alphaTexture"] is not None and mat["alphaMode"] == "OPAQUE":
            mat["alphaMode"] = "BLEND"
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
                f_arr = arr.astype(np.float64) / 255.0
                linear = np.where(f_arr <= 0.04045, f_arr / 12.92,
                                  ((f_arr + 0.055) / 1.055) ** 2.4)
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
            dump, grid_size=resolution,
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

        # Delete dump to save disk space (~32MB per sample)
        if os.path.exists(dump_path):
            os.remove(dump_path)

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="Worker rank for parallel processing")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    parser.add_argument("--out_root", type=str, default=OUT_ROOT)
    args = parser.parse_args()

    df = pd.read_parquet(PARQUET)
    all_samples = [(sha, df.loc[sha]) for sha in df.index]

    # Shard samples across workers
    my_samples = all_samples[args.rank::args.world_size]
    print(f"Worker {args.rank}/{args.world_size}: processing {len(my_samples)}/{len(all_samples)} samples → {args.out_root}")

    # Create output directories
    for d in [
        f"{args.out_root}/pbr_dumps",
        f"{args.out_root}/pbr_voxels_{RESOLUTION}",
        f"{args.out_root}/emission_voxels_{RESOLUTION}",
    ]:
        os.makedirs(d, exist_ok=True)

    # Copy metadata files (only rank 0)
    if args.rank == 0:
        for src_name, dst_name in [
            (PARQUET, "df_SomgProc_emission_filtered.parquet"),
            (SPLIT_JSON, "data_splits_emission_filtered.json"),
        ]:
            dst = os.path.join(args.out_root, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(src_name, dst)
                print(f"Copied {dst_name}")

    # Process samples
    success, fail = 0, 0
    for i, (sha, row) in enumerate(my_samples):
        print(f"[{i+1}/{len(my_samples)}] {sha[:16]}...", end=" ")
        ok = voxelize_sample(sha, row, args.out_root, RESOLUTION)
        if ok:
            success += 1
            print("OK")
        else:
            fail += 1

    print(f"\nDone: {success} success, {fail} failed (worker {args.rank})")


if __name__ == "__main__":
    main()
