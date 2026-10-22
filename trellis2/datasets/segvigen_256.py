"""SegviGenEmissive256: SegviGen-Emission's training data on the 256 voxel grid.

Per-shape sources (independent roots, joined by sha256):
    shape latent (concat_cond)     <shape_latent_root>/shape_latents/<shape_latent_name>/<sid>.npz
    pbr/input latent (appearance)  <pbr_latent_root>/pbr_latents/<pbr_latent_name>/<sid>.npz
    emission latent (x_0, target)  <emission_latent_dir>/<sid>.npz
                                    (data_toolkit/encode_emission_mask_latents.py)
    DINOv3 cond                    <a cond_search_dirs entry>/<sid>/cond_pos.pth (or cond.pth),
                                    the first entry that holds it (data_toolkit/build_cond_tree.py);
                                    the split JSON, not these directories, defines split membership

Normalization: shape_normalization / tex_normalization are the stock
TRELLIS.2 pipeline.json stats. The emission-mask latent and the PBR latent
both come from the stock tex encoder, so both use tex_normalization; the
shape latent uses shape_normalization. The stats of another latent space
would silently mis-scale the target.

Every npz's coords are expected on the SAME sparse grid (all three are
downsampled from the same occupied 256^3 voxel set); asserted per sample at
load time, and checked for every shape by data_toolkit/verify_and_pack.py.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor, sparse_cat
from ..utils.data_utils import load_balanced_group_indices

__all__ = ["SegviGenEmissive256"]


# ---------------------------------------------------------------------------
# Helpers: latent loading, collation and the sample list from a parquet index
# ---------------------------------------------------------------------------

def _read_npz_as_sparse(path: str, normalization: Optional[Dict] = None) -> SparseTensor:
    """Load a pre-encoded latent .npz and return a batch-0 SparseTensor."""
    data = np.load(path)
    feats  = torch.from_numpy(data['feats'].copy()).float()
    coords = torch.from_numpy(data['coords'].copy()).int()
    if normalization is not None:
        mean = torch.tensor(normalization['mean']).reshape(1, -1)
        std  = torch.tensor(normalization['std']).reshape(1, -1)
        feats = (feats - mean) / std
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return SparseTensor(feats, torch.cat([batch_col, coords], dim=-1))

def _sparse_collate(batch: List[Dict], split_size: Optional[int] = None):
    """Load-balance and collate a list of sample dicts containing SparseTensors."""
    if split_size is None:
        group_idx = [list(range(len(batch)))]
    else:
        # support both "x" (VAE) and "x_0" (DiT) as the primary sparse tensor key
        primary_key = "x" if "x" in batch[0] else "x_0"
        group_idx = load_balanced_group_indices(
            [b[primary_key].feats.shape[0] for b in batch], split_size
        )

    packs = []
    for group in group_idx:
        sub = [batch[i] for i in group]
        pack: Dict[str, Any] = {}
        for k in sub[0]:
            v0 = sub[0][k]
            if isinstance(v0, SparseTensor):
                pack[k] = sparse_cat([b[k] for b in sub], dim=0)
            elif isinstance(v0, torch.Tensor):
                pack[k] = torch.stack([b[k] for b in sub])
            elif isinstance(v0, list):
                pack[k] = sum([b[k] for b in sub], [])
            else:
                pack[k] = [b[k] for b in sub]
        packs.append(pack)

    return packs[0] if split_size is None else packs

_DEFAULT_SPLIT_JSONS = [
    "data_splits_emission_filtered.json",
    "data_splits_filtered.json",
]

def _build_parquet_samples(
    data_root: str,
    parquet_file: str,
    split: Optional[str],
    min_emission: float,
    split_indices: Optional[Any],
    split_json: Optional[str],
) -> List[Dict[str, str]]:
    """Build a list of {sha256, ...path...} dicts from a parquet index file."""
    df = pd.read_parquet(parquet_file)
    if "success" in df.columns:
        df = df[df["success"] == True]
    emit_col = "#stat_emission_color_max"
    if min_emission > 0.0 and emit_col in df.columns:
        df = df[df[emit_col] >= min_emission]

    # Auto-detect split JSON
    if split is not None and split_indices is None and split_json is None:
        for name in _DEFAULT_SPLIT_JSONS:
            candidate = os.path.join(data_root, name)
            if os.path.exists(candidate):
                split_json = candidate
                break

    # Build flat sample list (before any split filtering)
    samples: List[Dict[str, str]] = []
    for sample_id, row in df.iterrows():
        ditem_dir = row.get("ditem_dir", None)
        if ditem_dir is None:
            sid = str(sample_id)
            ditem_dir = f"{sid[:3]}-{sid[3:6]}/{sid}"
        samples.append({"sha256": str(sample_id), "ditem_dir": ditem_dir})

    # Apply split
    if split is not None and split_json is not None and split_indices is None:
        split_indices = split_json
    if split_indices is not None:
        samples = _apply_split_indices(samples, split_indices, split)

    return samples

def _apply_split_indices(
    samples: List[Dict],
    indices: Any,
    split: Optional[str],
) -> List[Dict]:
    if isinstance(indices, tuple):
        return samples[indices[0] : indices[1]]
    if isinstance(indices, list):
        return [samples[i] for i in indices]
    if isinstance(indices, str):
        with open(indices) as f:
            split_data = json.load(f)
        if split is None:
            idx_list = []
            for v in split_data.values():
                if isinstance(v, list):
                    idx_list.extend(v)
                elif isinstance(v, dict):
                    idx_list.extend(v.get("indices", []))
        else:
            entry = split_data.get(split, [])
            idx_list = entry.get("indices", entry) if isinstance(entry, dict) else entry
        return [samples[i] for i in idx_list]
    return samples


class SegviGenEmissive256(Dataset):
    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        split_json: str,
        split: str,                       # "train" / "val": a key of the split JSON
        shape_latent_root: str,
        shape_latent_name: str,
        pbr_latent_root: str,
        pbr_latent_name: str,
        emission_latent_dir: str,
        cond_search_dirs: List[str],
        shape_normalization: Dict,
        tex_normalization: Dict,
        max_samples: int = 0,
        **_framework_extra,
    ):
        if _framework_extra:
            print(f"[SegviGenEmissive256] ignoring framework-injected args: "
                  f"{sorted(_framework_extra)}")
        self.shape_dir = os.path.join(shape_latent_root, "shape_latents", shape_latent_name)
        self.pbr_dir = os.path.join(pbr_latent_root, "pbr_latents", pbr_latent_name)
        self.emit_dir = emission_latent_dir
        self.cond_search_dirs = list(cond_search_dirs)
        self.sm_norm = {"mean": shape_normalization["mean"], "std": shape_normalization["std"]}
        self.tx_norm = {"mean": tex_normalization["mean"], "std": tex_normalization["std"]}

        base = _build_parquet_samples(data_root, parquet_file, split, 0.0, None, split_json)
        # Admission by directory-listing set intersection: one listdir per root
        # (4 listings total) instead of 4 stats per sid, which is slow on a
        # network file system.
        shape_set = {f[:-4] for f in os.listdir(self.shape_dir) if f.endswith(".npz")}
        pbr_set = {f[:-4] for f in os.listdir(self.pbr_dir) if f.endswith(".npz")}
        emit_set = {f[:-4] for f in os.listdir(self.emit_dir) if f.endswith(".npz")}
        cond_set: set = set()
        for d in self.cond_search_dirs:
            try:
                cond_set |= set(os.listdir(d))
            except FileNotFoundError:
                pass
        # cond membership = the sid dir exists under a cond root; cond_pos.pth
        # inside it is checked by data_toolkit/verify_and_pack.py, and
        # _find_cond_dir re-checks per sample at load time.
        self.sids: List[str] = []
        n_skip = {"shape": 0, "pbr": 0, "emit": 0, "cond": 0}
        for s in base:
            sid = s["sha256"]
            if sid not in shape_set:
                n_skip["shape"] += 1
                continue
            if sid not in pbr_set:
                n_skip["pbr"] += 1
                continue
            if sid not in emit_set:
                n_skip["emit"] += 1
                continue
            if sid not in cond_set:
                n_skip["cond"] += 1
                continue
            self.sids.append(sid)
            if max_samples and len(self.sids) >= max_samples:
                break
        print(f"[SegviGenEmissive256] split={split!r}: {len(base)} in split_json, "
              f"{len(self.sids)} usable (SKIPPED: {n_skip} -- loud by design)")

        # Per-sample token count for BalancedResumableSampler. Prefer the
        # precomputed manifest (data_toolkit/verify_and_pack.py writes
        # <emit_dir>/token_counts.json); opening every npz header at each
        # construction is slow on a network file system.
        manifest = {}
        mpath = os.path.join(self.emit_dir, "token_counts.json")
        if os.path.exists(mpath):
            import json as _json
            with open(mpath) as f:
                manifest = _json.load(f)
        else:
            print(f"[SegviGenEmissive256] WARNING: no token_counts.json in "
                  f"{self.emit_dir}; falling back to per-npz header reads (slow)")
        self.loads: List[int] = []
        for sid in self.sids:
            n = manifest.get(sid)
            if n is None:
                try:
                    with np.load(os.path.join(self.emit_dir, f"{sid}.npz")) as d:
                        n = int(d["coords"].shape[0])
                except Exception:
                    n = 0
            self.loads.append(int(n))

    def _find_cond_dir(self, sid: str) -> Optional[str]:
        for d in self.cond_search_dirs:
            p = os.path.join(d, sid)
            if os.path.exists(os.path.join(p, "cond_pos.pth")) or os.path.exists(os.path.join(p, "cond.pth")):
                return p
        return None

    def __len__(self) -> int:
        return len(self.sids)

    def get_instance(self, sid: str) -> Dict[str, Any]:
        x_0 = _read_npz_as_sparse(os.path.join(self.emit_dir, f"{sid}.npz"), normalization=self.tx_norm)
        pbr = _read_npz_as_sparse(os.path.join(self.pbr_dir, f"{sid}.npz"), normalization=self.tx_norm)
        shp = _read_npz_as_sparse(os.path.join(self.shape_dir, f"{sid}.npz"), normalization=self.sm_norm)

        assert torch.equal(x_0.coords, shp.coords), f"{sid}: emission/shape coords mismatch"
        assert torch.equal(x_0.coords, pbr.coords), f"{sid}: emission/pbr coords mismatch"

        out: Dict[str, Any] = {
            "x_0": x_0,
            "concat_cond": shp,
            "appearance_cond": pbr,
            "sha256": sid,
        }

        d = self._find_cond_dir(sid)
        cp = os.path.join(d, "cond_pos.pth")
        if os.path.exists(cp):
            cond = torch.load(cp, map_location="cpu")
            if isinstance(cond, dict):
                cond = cond["cond"]
        else:
            cond = torch.load(os.path.join(d, "cond.pth"), map_location="cpu")["cond"]
        out["cond"] = cond.float().squeeze(0)   # (1029, 1024)
        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sid = self.sids[index]
        try:
            return self.get_instance(sid)
        except Exception as exc:
            print(f"[SegviGenEmissive256] Error loading {sid}: {exc}")
            raise

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int] = None):
        return _sparse_collate(batch, split_size=split_size)
