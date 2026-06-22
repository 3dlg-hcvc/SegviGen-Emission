#!/usr/bin/env python3
"""Resample TRELLIS.2 predicted emission voxels into UV space via the somage
position map, producing a per-shape pred_emission.png that evaluate_texge.py
consumes. Frame map (no rotation): p_voxel = p_somage / 2; voxel grid 256.
Pure numpy + scipy; runs in the `lightgen` conda env."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def voxel_centers(coords: np.ndarray, grid: int = 256) -> np.ndarray:
    """Int voxel coords [N,3] (0..grid-1) -> world-space centers in (-0.5, 0.5)."""
    return -0.5 + (coords.astype(np.float64) + 0.5) / grid


def somage_to_voxel_frame(position: np.ndarray) -> np.ndarray:
    """Somage positions in [-1,1] -> voxel frame [-0.5,0.5] (uniform half-scale)."""
    return position / 2.0


def resample_to_uv(coords, emission, position_map, mask, grid: int = 256,
                   cap_voxels: float = 1.5):
    """Nearest-voxel resample of emission onto the somage UV grid.

    coords:       int [N,3] voxel coords (0..grid-1)
    emission:     float [N,3] in [0,1], aligned to coords
    position_map: float [H,W,3] somage positions in [-1,1]
    mask:         bool [H,W] occupancy
    returns (uv_img [H,W,3] float32 in [0,1], miss_fraction).
    """
    H, W = mask.shape
    uv = np.zeros((H, W, 3), dtype=np.float32)
    sel = mask.astype(bool)
    n = int(sel.sum())
    if n == 0:
        return uv, 0.0
    if len(coords) == 0:
        return uv, 1.0
    centers = voxel_centers(np.asarray(coords), grid)
    tree = cKDTree(centers)
    q = somage_to_voxel_frame(position_map[sel].astype(np.float64))   # [n,3]
    d, idx = tree.query(q, k=1)
    cap = cap_voxels / grid
    vals = np.asarray(emission, dtype=np.float32)[idx]                # [n,3]
    over = d > cap
    vals[over] = 0.0
    uv[sel] = vals
    return uv, float(over.mean())


import os
import argparse
import pandas as pd
import imageio.v2 as imageio

SOMAGE_ROOT_DEFAULT = "/cs/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k"
SRC_PARQUET_DEFAULT = "/local-scratch/localhome/dya78/lightgen_74k_staging/df_SomgProc_final.parquet"


def load_somage_geom(sha, meta_df, somage_root):
    """Return (position_map [H,W,3] float in [-1,1], mask [H,W] bool) for a sha.
    Mirrors evaluate_texge.load_somage_full's decode exactly."""
    ditem_dir = meta_df.loc[sha, "ditem_dir"]
    raw = np.load(os.path.join(somage_root, ditem_dir, "somage.npz"))
    occ = raw["occupancy"]
    mask = (occ[..., 0] if occ.ndim == 3 else occ).astype(bool)
    position_map = raw["position"].astype(np.float32) / 65535.0 * 2 - 1
    return position_map, mask


def build_slim_metadata(src_parquet, shas, out_parquet):
    """Slim subset (ditem_dir, glb_1k_path) indexed by sha — fast per-shape reads."""
    df = pd.read_parquet(src_parquet, columns=["ditem_dir", "glb_1k_path"])
    slim = df.loc[df.index.intersection(shas), ["ditem_dir", "glb_1k_path"]].copy()
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    slim.to_parquet(out_parquet)
    return slim


def bridge_one(npz_path, source, meta_df, somage_root, out_dir, cap_voxels=1.5):
    """Resample one shape's voxels -> UV pred_emission.png. Returns miss-fraction."""
    sha = os.path.splitext(os.path.basename(npz_path))[0]
    d = np.load(npz_path)
    coords = d["coords"]
    emission = np.asarray(d[source], dtype=np.float32)
    position_map, mask = load_somage_geom(sha, meta_df, somage_root)
    uv, miss = resample_to_uv(coords, emission, position_map, mask, cap_voxels=cap_voxels)
    shape_dir = os.path.join(out_dir, sha)
    os.makedirs(shape_dir, exist_ok=True)
    imageio.imwrite(os.path.join(shape_dir, "pred_emission.png"),
                    (np.clip(uv, 0, 1) * 255).astype(np.uint8))
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_voxels_dir", required=True)
    ap.add_argument("--metadata", default=None,
                    help="Slim parquet (ditem_dir, glb_1k_path) indexed by sha. "
                         "If missing, built from --src_parquet over the found shas.")
    ap.add_argument("--src_parquet", default=SRC_PARQUET_DEFAULT)
    ap.add_argument("--somage_root", default=SOMAGE_ROOT_DEFAULT)
    ap.add_argument("--source", default="pred", choices=["pred", "gt_recon", "gt_emission"])
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--shape_ids", nargs="*", default=None)
    ap.add_argument("--cap_voxels", type=float, default=1.5)
    args = ap.parse_args()

    npzs = sorted(f for f in os.listdir(args.pred_voxels_dir) if f.endswith(".npz"))
    shas = [os.path.splitext(f)[0] for f in npzs]
    if args.shape_ids:
        keep = set(args.shape_ids)
        shas = [s for s in shas if s in keep]
        npzs = [f"{s}.npz" for s in shas]

    if args.metadata and os.path.exists(args.metadata):
        meta_df = pd.read_parquet(args.metadata)
    else:
        out_meta = args.metadata or os.path.join(args.out_root, "metadata_slim.parquet")
        meta_df = build_slim_metadata(args.src_parquet, shas, out_meta)
        print(f"Built slim metadata for {len(meta_df)} shas -> {out_meta}")

    missing = [s for s in shas if s not in meta_df.index]
    if missing:
        print(f"  WARNING: {len(missing)} of {len(shas)} shas not in metadata "
              f"— they will error in bridge_one (e.g. {missing[0]})")

    misses = []
    for f in npzs:
        try:
            m = bridge_one(os.path.join(args.pred_voxels_dir, f), args.source,
                           meta_df, args.somage_root, args.out_root, args.cap_voxels)
            misses.append(m)
        except Exception as e:
            print(f"  ERROR {f}: {e}")
    if misses:
        arr = np.array(misses)
        print(f"Bridged {len(misses)} shapes -> {args.out_root}  "
              f"miss-frac mean={arr.mean():.4f} max={arr.max():.4f}")


if __name__ == "__main__":
    main()
