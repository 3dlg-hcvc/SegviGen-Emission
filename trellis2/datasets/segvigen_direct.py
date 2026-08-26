"""SegviGen-emissive's dataset_direct tree as a lightning-framework dataset.

Reads the SAME per-shape files the standalone trainer
(segvigen_emissive/emissive/train/train_emissive.py) reads:
    shape_slat.pth, input_tex_slat.pth, output_tex_slat.pth   (sparse latents)
    emis_mask.pth                                             (per-voxel weight mask)
    cond_pos.pth / cond.pth                                   (precomputed DINOv3 tokens)
and applies the SAME normalization (TRELLIS.2's shipped pipeline.json stats,
passed in via the config so the run records them).

Batch dict keys (collate handles SparseTensor / tensor / str mixes):
    x_0             target latent  (normalized output_tex_slat)   SparseTensor 32ch
    concat_cond     shape latent   (normalized shape_slat)        SparseTensor 32ch
    appearance_cond input latent   (normalized input_tex_slat)    SparseTensor 32ch
    emis_mask       per-voxel weight mask                         SparseTensor 1ch
    cond            DINOv3 tokens (1029 x 1024)  [embedding mode] dense tensor
                    or an RGBA-preprocessed PIL-ready tensor      [image mode]
    sha256          sample id                                     str

cond_mode:
    'embedding' (default): read the precomputed DINOv3 features (cond_pos.pth if
        present, else the pos half of cond.pth). Pairs with
        SegviGenFlowTrainer's precomputed-cond path; the on-GPU encoder never runs.
    'image': read <thumbnail_dir>/<sha>.png (our PREPROCESSED bg-removed crops,
        not raw thumbnails; pointing this at raw thumbnails would silently change
        the conditioning distribution). 5x lighter on the read path once the
        crops exist on disk.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from .lightgen_slat import _sparse_collate

__all__ = ["SegviGenEmissiveDirect"]


def _norm_pair(d: Optional[Dict]):
    if d is None:
        return None
    return (torch.tensor(d["mean"], dtype=torch.float32)[None],
            torch.tensor(d["std"], dtype=torch.float32)[None])


def _as_sparse(coords: torch.Tensor, feats: torch.Tensor) -> SparseTensor:
    coords = coords.int()
    if coords.shape[1] == 3:   # ensure a batch column for sparse_cat
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    return SparseTensor(feats.float(), coords)


class SegviGenEmissiveDirect(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        shape_normalization: Dict,
        tex_normalization: Dict,
        cond_mode: str = "embedding",
        thumbnail_dir: Optional[str] = None,
        image_size: int = 512,
        require_mask: bool = True,
        max_samples: int = 0,
        **_framework_extra,
    ):
        # train.py injects legacy kwargs (e.g. roots) meant for the standard
        # TRELLIS datasets; this class defines its own contract and ignores
        # them loudly rather than crashing on the injection.
        if _framework_extra:
            print(f"[SegviGenEmissiveDirect] ignoring framework-injected args: "
                  f"{sorted(_framework_extra)}")
        assert cond_mode in ("embedding", "image")
        self.root = os.path.join(dataset_root, split)
        self.cond_mode = cond_mode
        self.thumbnail_dir = thumbnail_dir
        self.image_size = image_size
        self.require_mask = require_mask
        self.sm, self.ss = _norm_pair(shape_normalization)
        self.tm, self.ts = _norm_pair(tex_normalization)

        core = ["shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth"]
        self.sids: List[str] = []
        self.loads: List[int] = []
        n_skipped_mask = n_skipped_cond = 0
        for sid in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, sid)
            if not all(os.path.exists(os.path.join(d, f)) for f in core):
                continue
            if require_mask and not os.path.exists(os.path.join(d, "emis_mask.pth")):
                n_skipped_mask += 1
                continue
            if cond_mode == "embedding" and not (
                os.path.exists(os.path.join(d, "cond_pos.pth"))
                or os.path.exists(os.path.join(d, "cond.pth"))
            ):
                n_skipped_cond += 1
                continue
            if cond_mode == "image":
                if thumbnail_dir is None or not os.path.exists(
                    os.path.join(thumbnail_dir, f"{sid}.png")
                ):
                    n_skipped_cond += 1
                    continue
            self.sids.append(sid)
            mp = os.path.join(d, "meta.json")
            n = 0
            if os.path.exists(mp):
                try:
                    n = int(json.load(open(mp)).get("n_common_coords", 0))
                except Exception:
                    n = 0
            self.loads.append(n)
            if max_samples and len(self.sids) >= max_samples:
                break
        if self.loads and min(self.loads) == 0:
            m = float(np.mean([l for l in self.loads if l > 0]) or 1)
            self.loads = [l if l > 0 else int(m) for l in self.loads]
        print(f"[SegviGenEmissiveDirect] {split}: {len(self.sids)} samples admitted"
              f" (SKIPPED: {n_skipped_mask} missing mask, {n_skipped_cond} missing cond"
              f" -- loud by design, zero expected on a complete split)")

    def __len__(self) -> int:
        return len(self.sids)

    def get_instance(self, sid: str) -> Dict[str, Any]:
        d = os.path.join(self.root, sid)
        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location="cpu")
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location="cpu")
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location="cpu")
        coords = shp["coords"]

        x_0 = _as_sparse(coords, (otx["feats"].float() - self.tm) / self.ts)
        appearance = x_0.replace((itx["feats"].float() - self.tm) / self.ts)
        shape_z = x_0.replace((shp["feats"].float() - self.sm) / self.ss)

        out: Dict[str, Any] = {
            "x_0": x_0,
            "concat_cond": shape_z,
            "appearance_cond": appearance,
            "sha256": sid,
        }

        if self.require_mask:
            m = torch.load(os.path.join(d, "emis_mask.pth"), map_location="cpu")
            m = m.float().reshape(-1, 1)
            assert m.shape[0] == x_0.feats.shape[0], (
                f"{sid}: emis_mask length {m.shape[0]} != {x_0.feats.shape[0]} tokens")
            out["emis_mask"] = x_0.replace(m)

        if self.cond_mode == "embedding":
            cp = os.path.join(d, "cond_pos.pth")
            if os.path.exists(cp):
                cond = torch.load(cp, map_location="cpu")
                if isinstance(cond, dict):
                    cond = cond["cond"]
            else:
                cond = torch.load(os.path.join(d, "cond.pth"), map_location="cpu")["cond"]
            out["cond"] = cond.float().squeeze(0)   # (1029, 1024)
        else:
            from PIL import Image
            image = Image.open(os.path.join(self.thumbnail_dir, f"{sid}.png")).convert("RGBA")
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            alpha = torch.tensor(np.array(image.getchannel(3)), dtype=torch.float32) / 255
            rgb = torch.tensor(np.array(image.convert("RGB")), dtype=torch.float32) / 255
            out["cond"] = (rgb * alpha[..., None]).permute(2, 0, 1)
        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_instance(self.sids[index])

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int] = None):
        return _sparse_collate(batch, split_size=split_size)
