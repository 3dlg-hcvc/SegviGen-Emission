"""SegviGen-emissive, 256-native lane.

Owner correction (board lightning_port, 2026-08-26): the from-scratch
fullseg finetune trains on the 256-native voxel grid directly, reusing the
team's shared production shape+pbr latents (Dongchen's latents_v2 tree, the
same ones the emission DiT baseline reads) plus OUR new emission-mask npz
(scripts_segvigen/build_output_tex_latents_256.py, job 8) as the target.
This REPLACES SegviGenEmissiveDirect's dataset_direct/512 lane for the
imminent launch -- do not confuse the two; dataset_direct's own
output_tex_slat.pth is a different (512-resolution, upsampled) artifact.

Per-shape sources (three independent roots, joined by sha256):
    shape latent (concat_cond)     latents_v2/shape_latents/<shape_latent_name>/<sid>.npz
    pbr/input latent (appearance)  latents_v2/pbr_latents/<pbr_latent_name>/<sid>.npz
    emission latent (x_0, target)  OUR build output (emission_latent_dir)/<sid>.npz
    DINOv3 cond                    dataset_direct/{train_72k_agentic,val_72k}/<sid>/cond_pos.pth
                                    (reused verbatim: image conditioning is resolution-
                                    independent, so the 512-lane's precomputed tokens
                                    apply here unchanged -- searched across both split
                                    dirs since this class's split membership is defined
                                    by the _agentic split JSON, not those directory names)

Normalization: shape_normalization / tex_normalization are the SAME shipped
TRELLIS.2 pipeline.json stats segvigen_emissive_direct.yaml uses (NOT the
DiT baseline's regenerated agentic-VAE emission stats) -- both our emission
npz and Dongchen's pbr latent were encoded with the STOCK tex encoder, so
they share the stock tex_normalization; the shape latent is the stock
shape_normalization. Passing the DiT's own emission_normalization here
would silently mis-scale the target: that block belongs to a different
(agentic-VAE) latent space.

Every npz's coords are expected on the SAME sparse grid (all three are
downsampled from the same occupied 256^3 voxel set); asserted per-sample at
load time, and pre-verified at build time for a coord-check sample (the one-off
scripts_segvigen/check_256_coords_vs_shape.py, at tag pre-refactor: 13/13 match on val_72k).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .lightgen_slat import _build_parquet_samples, _read_npz_as_sparse, _sparse_collate

__all__ = ["SegviGenEmissive256"]


class SegviGenEmissive256(Dataset):
    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        split_json: str,
        split: str,                       # "train" / "val" (split-JSON key, NOT a dataset_direct dirname)
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
        # (4 listings total) instead of 4 NFS stats x 36k sids. The per-sid
        # exists() version of this loop cost 30-60 min of dead GPU time at
        # every dataset construction (found live on job 257886, 2026-08-26).
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
        # presence inside it was verified corpus-wide 2026-08-26 (the sidecar
        # fill), and _find_cond_dir still re-checks per sample at load time.
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
        # precomputed manifest (scripts_segvigen/build_token_manifest.py writes
        # <emit_dir>/token_counts.json); opening 36k npz headers over NFS at
        # every construction was the other half of the dead-GPU startup cost.
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
