#!/usr/bin/env python3
"""
Regenerate data/lightgen_1k/dual_grid_256/*.vxz from Blender pbr_dumps so the
shape latents can be re-encoded with coords aligned to emission/pbr voxels.

The original lightgen-side `dual_grid_from_glb.py` (trimesh shortcut) produces
a different vertex set than Blender's GLTF importer, so the resulting shape
latents have different sparse coords than emission/pbr latents — the DiT's
`assert torch.equal(x_0.coords, shape_z.coords)` assertion fails for ~90% of
samples.

This script restores the original TRELLIS.2 path: Blender pickle →
mesh_to_flexible_dual_grid → dual_grid_256/{sha}.vxz. Skips samples already
processed; logs failures but does not abort. Pickles are kept under
$TMPDIR/dual_grid_pbr_dumps/ during the run and not deleted (re-used across
re-runs).

Usage:
    python data_toolkit/regenerate_1k_dual_grid.py --workers 8

After this finishes, run encode_shape_from_dual_grid.py to rewrite shape
latents from the new dual grids.
"""
from __future__ import annotations
import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

# Stub modules that o_voxel imports lazily.
import types as _types
for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = _types.ModuleType(_m)
import o_voxel
import o_voxel.convert
import o_voxel.io


PARQUET    = "data/lightgen_1k/df_SomgProc_emission_filtered.parquet"
GLB_ROOT   = "/localhome/dya78/code/lightgen/data/TexVerse-1K"
OUT_ROOT   = "data/lightgen_1k"
RESOLUTION = 256

BLENDER     = "/localhome/dya78/software/blender-4.2.0-linux-x64/blender"
DUMP_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_script/dump_pbr.py")
PICKLE_CACHE = os.environ.get("DUAL_GRID_PICKLE_CACHE",
                              "/local-scratch/localhome/dya78/dual_grid_pbr_dumps")


def _blender_dump(glb_path: str, pickle_path: str) -> tuple[bool, str]:
    cmd = [BLENDER, "--background", "--python", DUMP_SCRIPT, "--",
           "--object", glb_path, "--output_path", pickle_path]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        return False, res.stderr[-500:]
    return True, ""


def _dual_grid_from_pickle(pickle_path: str, dual_grid_path: str, resolution: int) -> tuple[bool, str]:
    with open(pickle_path, "rb") as f:
        dump = pickle.load(f)

    objs = [obj for obj in dump["objects"]
            if obj["vertices"].size != 0 and obj["faces"].size != 0]
    if not objs:
        return False, "no geometry in pickle"

    start = 0
    all_verts, all_faces = [], []
    for obj in objs:
        all_verts.append(obj["vertices"])
        all_faces.append(obj["faces"] + start)
        start += len(obj["vertices"])

    vertices = torch.from_numpy(np.concatenate(all_verts, axis=0)).float()
    faces    = torch.from_numpy(np.concatenate(all_faces, axis=0)).long()

    # Same normalization as encode_10_samples.py / prep_1k_dataset.py.
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
    # Match the on-disk packing used by data_toolkit/dual_grid.py and
    # encode_shape_from_dual_grid.py (uint8 quantized vertices + bit-packed
    # intersected mask).
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1]
                   + 2 * intersected[:, 1:2]
                   + 4 * intersected[:, 2:3]).type(torch.uint8)
    attr = {"vertices": dual_vertices, "intersected": intersected}
    o_voxel.io.write_vxz(dual_grid_path, voxel_indices, attr)
    return True, f"N={voxel_indices.shape[0]}"


def _process_one(sha: str, glb_rel: str) -> tuple[str, bool, str]:
    glb_path = os.path.join(GLB_ROOT, glb_rel)
    if not os.path.exists(glb_path):
        return sha, False, f"GLB missing: {glb_path}"

    pickle_path = os.path.join(PICKLE_CACHE, f"{sha}.pickle")
    if not os.path.exists(pickle_path):
        ok, err = _blender_dump(glb_path, pickle_path)
        if not ok:
            return sha, False, f"blender: {err.strip()[-200:]}"

    dual_grid_path = os.path.join(OUT_ROOT, f"dual_grid_{RESOLUTION}", f"{sha}.vxz")
    try:
        ok, info = _dual_grid_from_pickle(pickle_path, dual_grid_path, RESOLUTION)
        if not ok:
            return sha, False, f"dual_grid: {info}"
        return sha, True, info
    except Exception as e:
        return sha, False, f"dual_grid exc: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel Blender workers (default 4; each Blender uses ~2 GB RAM)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N samples (debug)")
    args = ap.parse_args()

    os.makedirs(PICKLE_CACHE, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, f"dual_grid_{RESOLUTION}"), exist_ok=True)

    df = pd.read_parquet(PARQUET)
    samples = list(df.index)
    if args.limit:
        samples = samples[:args.limit]

    work = []
    skipped = 0
    for sha in samples:
        out = os.path.join(OUT_ROOT, f"dual_grid_{RESOLUTION}", f"{sha}.vxz")
        # Always regenerate — the existing 1k dual_grid is broken (trimesh-derived).
        # Re-runs are idempotent; this overwrites.
        glb_rel = df.loc[sha, "glb_1k_path"]
        work.append((sha, glb_rel))

    print(f"Processing {len(work)} samples ({skipped} cached); workers={args.workers}")
    print(f"Pickle cache: {PICKLE_CACHE}")
    print(f"Output: {OUT_ROOT}/dual_grid_{RESOLUTION}/")
    print()

    t0 = time.time()
    success = fail = 0
    fail_log = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, sha, glb_rel): sha for (sha, glb_rel) in work}
        for i, fut in enumerate(as_completed(futs), 1):
            sha = futs[fut]
            try:
                _, ok, msg = fut.result()
            except Exception as e:
                ok, msg = False, f"future exc: {e}"
            if ok:
                success += 1
            else:
                fail += 1
                fail_log.append((sha, msg))
            if i % 10 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(work) - i) / max(rate, 1e-6)
                print(f"[{i:4d}/{len(work)}] ok={success} fail={fail}  rate={rate:.2f}/s  eta={eta/60:.1f}min")

    print(f"\nDone: {success} success, {fail} fail in {(time.time()-t0)/60:.1f} min")
    if fail_log:
        with open("regenerate_1k_dual_grid.failures.txt", "w") as f:
            for sha, msg in fail_log:
                f.write(f"{sha}\t{msg}\n")
        print(f"Failures logged to regenerate_1k_dual_grid.failures.txt")


if __name__ == "__main__":
    main()
