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
