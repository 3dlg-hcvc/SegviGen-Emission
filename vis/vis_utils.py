"""Shared helpers for 3D interactive visualizations.

Used by vis/vis_3d_vae.py and vis/vis_3d_dit.py.

See docs/superpowers/specs/2026-04-22-vis-3d-consolidation-design.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Match the sys.path convention used across existing vis/*.py scripts so the
# helpers work when invoked as `python vis/vis_3d_vae.py ...` from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_TRELLIS2_SITE = "/localhome/dya78/software/miniconda3/envs/trellis2/lib/python3.10/site-packages"
if os.path.isdir(_TRELLIS2_SITE) and _TRELLIS2_SITE not in sys.path:
    sys.path.insert(0, _TRELLIS2_SITE)

# Stub submodules o_voxel tries to import that we don't need here.
for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import o_voxel  # noqa: E402


# =============================================================================
# Sample loading
# =============================================================================

@dataclass
class SampleRecord:
    sha: str
    emission_voxel_path: str
    pbr_voxel_path: str
    glb_path: Optional[str]
    thumbnail_path: Optional[str]
    latents: dict[str, str] = field(default_factory=dict)

    def has_all(self, keys: tuple[str, ...]) -> bool:
        for k in keys:
            v = getattr(self, k, None) if k in {"emission_voxel_path", "pbr_voxel_path", "glb_path", "thumbnail_path"} else self.latents.get(k)
            if not v or not os.path.exists(v):
                return False
        return True


def _resolve_paths(
    data_root: str,
    resolution: int,
    sha: str,
    glb_root: Optional[str],
    glb_rel: Optional[str],
    thumbnail_rel: Optional[str],
    latent_names: Optional[dict[str, str]],
) -> SampleRecord:
    emission_voxel_path = os.path.join(data_root, f"emission_voxels_{resolution}", f"{sha}.vxz")
    pbr_voxel_path = os.path.join(data_root, f"pbr_voxels_{resolution}", f"{sha}.vxz")
    glb_path = os.path.join(glb_root, glb_rel) if (glb_root and glb_rel) else None
    thumbnail_path = (
        os.path.join(data_root, "thumbnails", f"{sha}.png")
        if os.path.exists(os.path.join(data_root, "thumbnails", f"{sha}.png"))
        else (os.path.join(data_root, "thumbnails", thumbnail_rel) if thumbnail_rel else None)
    )
    latents: dict[str, str] = {}
    if latent_names:
        for key, name in latent_names.items():
            latents[key] = os.path.join(data_root, f"{key}_latents", name, f"{sha}.npz")
    return SampleRecord(
        sha=sha,
        emission_voxel_path=emission_voxel_path,
        pbr_voxel_path=pbr_voxel_path,
        glb_path=glb_path,
        thumbnail_path=thumbnail_path,
        latents=latents,
    )


def load_samples(
    data_root: str,
    resolution: int,
    source: str,
    select: str = "all",
    glb_root: Optional[str] = None,
    latent_names: Optional[dict[str, str]] = None,
    require: tuple[str, ...] = ("emission_voxel_path",),
) -> list[SampleRecord]:
    """Load SampleRecords from a parquet, splits JSON, or comma-separated sha list.

    Args:
        data_root: directory holding emission_voxels_<res>/, pbr_voxels_<res>/, etc.
        resolution: voxel grid size (e.g. 256).
        source: parquet path, *.json splits file, or "sha:sha1,sha2,...".
        select: one of "all", "first:<N>", "split:<name>", "sha:<csv>".
        glb_root: directory referenced by the parquet's ``glb_1k_path`` column.
        latent_names: e.g. {"emission": "emission_vae_10sample_overfit"}.
        require: attribute names that must exist on disk to include a record.

    Returns SampleRecords filtered to those where all required paths exist.
    """
    # --- Resolve sha list + per-sha extras from source ---------------------
    if select.startswith("sha:"):
        shas = [s.strip() for s in select[4:].split(",") if s.strip()]
        glb_map: dict[str, str] = {}
        thumb_map: dict[str, str] = {}
    elif source.endswith(".json"):
        with open(source) as f:
            splits = json.load(f)
        split_name = select[6:] if select.startswith("split:") else "train"
        entry = splits.get(split_name, [])
        if isinstance(entry, dict):
            # Format: {"indices": [0, 1, ...]} — resolve via the parquet in data_root.
            idx_list: list[int] = entry.get("indices", [])
            parquet_path = os.path.join(data_root, "df_10samples.parquet")
            _df = pd.read_parquet(parquet_path)
            all_shas = list(_df.index.astype(str))
            shas = [all_shas[i] for i in idx_list if i < len(all_shas)]
        else:
            shas = list(entry)
        glb_map, thumb_map = {}, {}
    else:
        df = pd.read_parquet(source)
        if "success" in df.columns:
            df = df[df["success"] == True]  # noqa: E712
        if df.index.name is None and "sha256" in df.columns:
            df = df.set_index("sha256")
        elif df.index.name != "sha256" and df.index.name != "ditem_id":
            df.index = df.index.astype(str)
        shas = list(df.index.astype(str))
        glb_map = df["glb_1k_path"].to_dict() if "glb_1k_path" in df.columns else {}
        thumb_map = df["thumbnail_path"].to_dict() if "thumbnail_path" in df.columns else {}

    # --- Apply selector over the resolved shas -----------------------------
    if select.startswith("first:"):
        shas = shas[: int(select[6:])]
    elif select == "all" or select.startswith("split:") or select.startswith("sha:"):
        pass
    else:
        raise ValueError(f"unrecognised --samples selector: {select!r}")

    records: list[SampleRecord] = []
    for sha in shas:
        rec = _resolve_paths(
            data_root=data_root,
            resolution=resolution,
            sha=sha,
            glb_root=glb_root,
            glb_rel=glb_map.get(sha),
            thumbnail_rel=thumb_map.get(sha),
            latent_names=latent_names,
        )
        if rec.has_all(require):
            records.append(rec)
    return records
