"""dual_grid_256/<uuid>.vxz for the released dataset, on the cells of its PBR voxels.

The release (3dlg-hcvc/LightgenBench) ships PBR and emission voxels but no geometry, and the
shape encoder reads a flexible dual grid built from the mesh. This builds that grid from the
TexVerse GLB with the paper's own routine (data_toolkit/dual_grid_from_glb.process_one: same
weights, same uint8 packing), with ONE change: the mesh comes from the loader the release was
baked with (lightgen data_processing/data_preparation/loader.load_and_merge: every scene-graph
instance, then the bake's tight normalize), not from load_mesh_from_glb, which keeps the first
node of each geometry only.

Gate: the grid's occupied cells must equal the released pbr_voxels.vxz cells exactly (measured
2026-09-19 on 200 release shapes before the run). SegviGenEmissive256 asserts equal latent
coords per sample, so a shape that fails the gate is removed and listed, never kept.

CPU only. Resumable: a shape whose .vxz exists is skipped; a failure is appended to
<out_root>/dual_grid_256/_failed.rank<r>.tsv.

Usage (one rank; the sbatch starts W of them):
    python scripts_segvigen/newdata/build_dual_grid.py \
        --lightgen_root ~/code/lightgen --sid_list <meta>/release_all.txt \
        --release /3dlg-jupiter-project/lightgen/datasets/dataset_release \
        --metadata <TexVerse df_metadata.parquet> --glb_root <TexVerse-1K root> \
        --out_root <work root> --rank 0 --world_size 24
"""
import argparse
import importlib.util
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = 256


def _paper_module():
    spec = importlib.util.spec_from_file_location(
        "dual_grid_from_glb", os.path.join(REPO, "data_toolkit", "dual_grid_from_glb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells(coords) -> np.ndarray:
    c = np.ascontiguousarray(coords.cpu().numpy().astype(np.int64))
    return np.unique(c, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lightgen_root", required=True, help="checkout of the lightgen repo (the bake's loader)")
    ap.add_argument("--sid_list", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--metadata", required=True, help="TexVerse df_metadata.parquet (index = uuid, glb_1k_path)")
    ap.add_argument("--glb_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.expanduser(args.lightgen_root))
    from data_processing.data_preparation.contract import TIGHT_SCALE_NUM
    from data_processing.data_preparation.loader import load_and_merge
    from utils.glb_io import resolve_glb
    import o_voxel

    def bake_mesh(glb_path: str):
        b = load_and_merge(glb_path)
        v = np.asarray(b.vertices, dtype=np.float64)
        center = (v.min(0) + v.max(0)) / 2.0
        ext = float((v.max(0) - v.min(0)).max())
        v = (v - center) * (TIGHT_SCALE_NUM / ext if ext > 0 else 1.0)   # voxels._tight_scene
        return torch.from_numpy(v).float(), torch.from_numpy(np.asarray(b.faces)).long()

    paper = _paper_module()
    paper.load_mesh_from_glb = bake_mesh

    sids = [l.strip() for l in open(args.sid_list) if l.strip()]
    sids = [s for i, s in enumerate(sids) if i % args.world_size == args.rank]
    if args.limit:
        sids = sids[:args.limit]
    meta = pd.read_parquet(args.metadata, columns=["glb_1k_path"])
    out_dir = os.path.join(args.out_root, f"dual_grid_{RES}")
    os.makedirs(out_dir, exist_ok=True)
    failed = os.path.join(out_dir, f"_failed.rank{args.rank}.tsv")
    print(f"[dual_grid] rank {args.rank}/{args.world_size}: {len(sids)} shapes -> {out_dir}", flush=True)

    n = {"ok": 0, "skipped": 0, "mismatch": 0, "error": 0}
    t0 = time.time()
    for i, sid in enumerate(sids):
        out_path = os.path.join(out_dir, f"{sid}.vxz")
        if os.path.exists(out_path):
            n["skipped"] += 1
            continue
        try:
            glb = resolve_glb(sid, args.glb_root, meta)
            r = paper.process_one(sid, glb, out_dir, RES)
            if r["status"] != "ok":
                raise RuntimeError(r.get("error", r["status"]))
            c_dg, _ = o_voxel.io.read(out_path)
            c_pbr, _ = o_voxel.io.read(os.path.join(args.release, sid, "pbr_voxels.vxz"))
            a, b = _cells(c_dg), _cells(c_pbr)
            if a.shape != b.shape or not np.array_equal(a, b):
                os.remove(out_path)
                n["mismatch"] += 1
                with open(failed, "a") as f:
                    f.write(f"{sid}\tmismatch\tdual_grid={len(a)} pbr={len(b)}\n")
                continue
            n["ok"] += 1
        except Exception as e:  # noqa: BLE001
            if os.path.exists(out_path):
                os.remove(out_path)
            n["error"] += 1
            with open(failed, "a") as f:
                f.write(f"{sid}\terror\t{e!r}\n")
            print(f"[dual_grid] ERROR {sid}: {e!r}\n{traceback.format_exc()[-400:]}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"[dual_grid] rank {args.rank}: {i + 1}/{len(sids)} {n} {time.time() - t0:.0f}s", flush=True)
    print(f"[dual_grid] rank {args.rank} DONE: {n} of {len(sids)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
