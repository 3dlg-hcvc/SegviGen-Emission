#!/usr/bin/env python3
"""Stage B-2 — dual_grid_<res>/<sha>.vxz from pbr_dumps/<sha>.pickle.

This is the Blender-pickle path (vertex set matches voxelize_pbr.py's output),
mandatory to keep shape-latent coords aligned with PBR/emission-latent coords.
The trimesh-based shortcut in old `dual_grid_from_glb.py` produced different
sparse coords and broke the DiT's `assert torch.equal(x_0.coords, shape_z.coords)`.

Usage:
    python data_toolkit/prep_74k_dual_grid.py \\
        --root /localscratch/dya78/lightgen_74k_part0 \\
        --rank 0 --world_size 8 --workers 16 --resolution 256
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

# Stub modules that o_voxel imports lazily.
for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import o_voxel.convert
import o_voxel.io


def _dual_grid_from_pickle(pickle_path: str, dual_grid_path: str, resolution: int) -> tuple[bool, str]:
    with open(pickle_path, "rb") as f:
        dump = pickle.load(f)

    objs = [obj for obj in dump["objects"]
            if obj["vertices"].size != 0 and obj["faces"].size != 0]
    if not objs:
        return False, "no geometry in pickle"

    start = 0
    all_v, all_f = [], []
    for obj in objs:
        all_v.append(obj["vertices"])
        all_f.append(obj["faces"] + start)
        start += len(obj["vertices"])

    vertices = torch.from_numpy(np.concatenate(all_v, axis=0)).float()
    faces    = torch.from_numpy(np.concatenate(all_f, axis=0)).long()

    vmin = vertices.min(dim=0)[0]
    vmax = vertices.max(dim=0)[0]
    center = (vmin + vmax) / 2
    scale  = 0.99999 / (vmax - vmin).max()
    vertices = (vertices - center) * scale

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    dual_vertices = dual_vertices * resolution - voxel_indices
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1]
                   + 2 * intersected[:, 1:2]
                   + 4 * intersected[:, 2:3]).type(torch.uint8)

    attr = {"vertices": dual_vertices, "intersected": intersected}
    o_voxel.io.write_vxz(dual_grid_path, voxel_indices, attr)
    return True, f"N={voxel_indices.shape[0]}"


def _process_one(args: tuple[str, str, int]) -> tuple[str, bool, str]:
    sha, root, resolution = args
    pickle_path    = os.path.join(root, "pbr_dumps", f"{sha}.pickle")
    dual_grid_path = os.path.join(root, f"dual_grid_{resolution}", f"{sha}.vxz")
    if os.path.exists(dual_grid_path):
        return sha, True, "skip"
    if not os.path.exists(pickle_path):
        return sha, False, "pickle_missing"
    try:
        ok, info = _dual_grid_from_pickle(pickle_path, dual_grid_path, resolution)
        return sha, ok, info
    except Exception as e:
        return sha, False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.root, "metadata.csv"))
    start = len(df) * args.rank // args.world_size
    end   = len(df) * (args.rank + 1) // args.world_size
    df_rank = df.iloc[start:end]
    if args.limit is not None:
        df_rank = df_rank.head(args.limit)
    print(f"[prep_74k_dual_grid] rank {args.rank}/{args.world_size}: {len(df_rank)} samples")

    out_dir = os.path.join(args.root, f"dual_grid_{args.resolution}")
    os.makedirs(os.path.join(out_dir, "new_records"), exist_ok=True)

    work = [(r["sha256"], args.root, args.resolution) for _, r in df_rank.iterrows()]

    t0 = time.time()
    success = skip = fail = 0
    fail_log: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            sha, ok, msg = fut.result()
            if ok and msg == "skip":
                skip += 1
            elif ok:
                success += 1
            else:
                fail += 1
                fail_log.append((sha, msg))
            if i % 100 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1e-6)
                eta_m = (len(work) - i) / max(rate, 1e-6) / 60
                print(f"[{i:5d}/{len(work)}] ok={success} skip={skip} fail={fail}  "
                      f"rate={rate:.1f}/s  eta={eta_m:.1f}min", flush=True)

    fail_path = os.path.join(out_dir, "new_records", f"part_{args.rank}_failures.txt")
    with open(fail_path, "w") as f:
        for sha, msg in fail_log:
            f.write(f"{sha}\t{msg}\n")
    print(f"[prep_74k_dual_grid] done: ok={success} skip={skip} fail={fail} "
          f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
