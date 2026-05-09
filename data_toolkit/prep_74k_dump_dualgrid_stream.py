#!/usr/bin/env python3
"""Streaming dump + dual_grid for rank 2 backfill recovery.

Per-sample lifecycle in a worker process:
  1. blender --background dump_pbr.py -> /tmp/<sha>.pickle
  2. _dual_grid_from_pickle(pickle) -> dual_grid_<res>/<sha>.vxz
  3. rm pickle
Peak disk stays at ~workers * 30 MB instead of ~9k * 30 MB ≈ 270 GB.

Skips samples that already have dual_grid_<res>/<sha>.vxz.

Usage:
    python data_toolkit/prep_74k_dump_dualgrid_stream.py \\
        --root /localscratch/dya78/lightgen_74k_part2 \\
        --rank 2 --world_size 8 --workers 64 \\
        --blender /localscratch/dya78/blender-4.5.1-linux-x64/blender
"""
from __future__ import annotations
import argparse
import os
import pickle as pkl
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import o_voxel.convert
import o_voxel.io


GLB_ROOT = Path("/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K")
DUMP_SCRIPT = (Path(__file__).parent / "blender_script" / "dump_pbr.py").resolve()


def _dual_grid_from_pickle(pickle_path: str, dual_grid_path: str, resolution: int) -> tuple[bool, str]:
    with open(pickle_path, "rb") as f:
        dump = pkl.load(f)
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
    vmin = vertices.min(dim=0)[0]; vmax = vertices.max(dim=0)[0]
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
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    o_voxel.io.write_vxz(dual_grid_path, voxel_indices, {"vertices": dual_vertices, "intersected": intersected})
    return True, f"N={voxel_indices.shape[0]}"


def _process_one(args: tuple[str, str, str, str, int]) -> tuple[str, bool, str]:
    sha, glb_rel, root, blender, resolution = args
    out_dg = os.path.join(root, f"dual_grid_{resolution}", f"{sha}.vxz")
    if os.path.exists(out_dg):
        return sha, True, "skip-dg"
    glb_path = str(GLB_ROOT / glb_rel)
    if not os.path.exists(glb_path):
        return sha, False, "glb_missing"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_pkl = os.path.join(tmp, f"{sha}.pickle")
        cmd = [blender, "--background", "--python", str(DUMP_SCRIPT), "--",
               "--object", glb_path, "--output_path", tmp_pkl]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return sha, False, "blender timeout"
        if res.returncode != 0 or not os.path.exists(tmp_pkl):
            return sha, False, "blender fail"
        try:
            ok, info = _dual_grid_from_pickle(tmp_pkl, out_dg, resolution)
            return sha, ok, info
        except Exception as e:
            return sha, False, f"dg exc: {type(e).__name__}: {str(e)[:100]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--blender", type=str,
                    default="/localscratch/dya78/blender-4.5.1-linux-x64/blender")
    args = ap.parse_args()

    if not os.path.exists(args.blender):
        print(f"[stream] FATAL: blender missing at {args.blender}", file=sys.stderr)
        return 2

    df = pd.read_csv(os.path.join(args.root, "metadata.csv"))
    s = len(df) * args.rank // args.world_size
    e = len(df) * (args.rank + 1) // args.world_size
    df_rank = df.iloc[s:e]
    print(f"[stream] rank {args.rank}/{args.world_size}: {len(df_rank)} samples, {args.workers} workers")

    os.makedirs(os.path.join(args.root, f"dual_grid_{args.resolution}"), exist_ok=True)
    os.makedirs(os.path.join(args.root, f"dual_grid_{args.resolution}", "new_records"), exist_ok=True)

    work = [(r["sha256"], r["glb_1k_path"], args.root, args.blender, args.resolution)
            for _, r in df_rank.iterrows()]

    t0 = time.time()
    skip = ok = fail = 0
    fail_log: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            sha, ok_flag, msg = fut.result()
            if ok_flag and msg == "skip-dg":
                skip += 1
            elif ok_flag:
                ok += 1
            else:
                fail += 1
                fail_log.append((sha, msg))
            if i % 50 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1e-6)
                eta_h = (len(work) - i) / max(rate, 1e-6) / 3600
                print(f"[{i:5d}/{len(work)}] ok={ok} skip={skip} fail={fail}  "
                      f"rate={rate:.2f}/s  eta={eta_h:.2f}h", flush=True)

    fail_path = os.path.join(args.root, f"dual_grid_{args.resolution}", "new_records",
                             f"part_{args.rank}_stream_failures.txt")
    with open(fail_path, "w") as f:
        for sha, msg in fail_log:
            f.write(f"{sha}\t{msg}\n")
    print(f"[stream] done: ok={ok} skip={skip} fail={fail} in {(time.time()-t0)/60:.1f} min  failures -> {fail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
