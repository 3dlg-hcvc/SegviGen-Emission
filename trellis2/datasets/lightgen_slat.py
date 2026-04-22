"""
Voxel-space datasets for LightGen: emission generation conditioned on material + geometry.

Pipeline overview
-----------------
GLB files (TexVerse-1K)
  ↓  data_toolkit/dump_pbr.py  (Blender 4.x, now with emission extraction)
pbr_dumps/{sha256}.pickle
  ↓  data_toolkit/voxelize_pbr.py  (o_voxel)
pbr_voxels_{res}/{sha256}.vxz        ← base_color(3) + metallic(1) + roughness(1) + alpha(1)
emission_voxels_{res}/{sha256}.vxz   ← emissive(3)   [same sparse coordinates]
  ↓  data_toolkit/encode_pbr_latent.py
pbr_latents/{name}/{sha256}.npz      ← encoded PBR material latent
  ↓  data_toolkit/encode_emission_latent.py
emission_latents/{name}/{sha256}.npz ← encoded emission latent

Datasets
--------
LightGenSLatEmission
    For training the emission VAE.
    Returns the raw emission voxels (x_0) and optionally the PBR voxels (cond).
    Mirrors SparseVoxelPbrDataset but loads the emission channel instead.

LightGenSLatEmissionLatent
    For training the emission DiT via flow matching.
    Returns pre-encoded emission latent (x_0),
            pre-encoded PBR material latent (concat_cond — concatenated per token),
            and optionally the shape latent (for geometry cross-attention).
    Mirrors SLatPbr with the roles swapped.

Both datasets support two metadata backends:
  1. Standard TRELLIS.2-style: a directory with metadata.csv + sha256-named voxel files.
     Use the ``roots`` argument (same format as StandardDatasetBase).
  2. Parquet-based (TexVerse-1K subset): a parquet index file from baked_uv_local_subset.
     Use the ``data_root`` + ``parquet_file`` arguments.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..modules import sparse as sp
from ..modules.sparse.basic import SparseTensor, sparse_cat
from ..utils.data_utils import load_balanced_group_indices

# o_voxel is a compiled C extension — import lazily so the module loads
# even when the extension hasn't been built yet.
try:
    import o_voxel as _o_voxel
except ImportError:
    _o_voxel = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_vxz_as_sparse(path: str, attr_keys: List[str]) -> SparseTensor:
    """Read a .vxz file and return a batch-0 SparseTensor."""
    if _o_voxel is None:
        raise ImportError(
            "o_voxel is required to load .vxz voxel files. "
            "Build it with: cd o-voxel && pip install -e ."
        )
    coords, attr = _o_voxel.io.read_vxz(path, num_threads=4)
    feats = torch.cat([attr[k] for k in attr_keys], dim=-1).float() / 255.0 * 2.0 - 1.0
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return SparseTensor(feats, torch.cat([batch_col, coords], dim=-1))


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


# ---------------------------------------------------------------------------
# Shared collate function (works for any dict of SparseTensors / Tensors)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parquet-based sample list builder (shared with lightgen_uv.py approach)
# ---------------------------------------------------------------------------

_DEFAULT_SPLIT_JSONS = [
    "data_splits_emission_filtered.json",
    "data_splits_filtered.json",
]

_VXZ_DIR    = "vxz_root"   # placeholder key in roots dict
_LATENT_DIR = "latent_root"


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


# ---------------------------------------------------------------------------
# Dataset 1 — raw emission voxels for VAE training
# ---------------------------------------------------------------------------

class LightGenSLatEmission(Dataset):
    """
    Raw emission + PBR voxels for training the emission VAE.

    Mirrors SparseVoxelPbrDataset: each sample returns a single SparseTensor ``x``
    whose features are ``cat(emission[3], pbr_attrs[C_pbr])`` in [-1, 1].
    The first ``emission_channels`` features are the reconstruction target; the
    remaining PBR features serve as geometry+material context for the encoder.

    Channel layout (default pbr_attrs):
        x[:, 0:3]  — emissive RGB
        x[:, 3:6]  — base_color
        x[:, 6:7]  — metallic
        x[:, 7:8]  — roughness
        x[:, 8:9]  — alpha
        total = 9

    Args:
        data_root      Root of the processed data tree, containing
                       ``emission_voxels_{res}/`` and ``pbr_voxels_{res}/``.
        parquet_file   Path to the parquet index (baked_uv_local_subset style).
        resolution     Voxel grid resolution used during voxelization.
        split          "train" / "val" / "test" / None.
        min_emission   Keep only samples whose max emission stat >= this value.
        split_indices  Override split (tuple / list / JSON path).
        split_json     Explicit path to a split JSON file.
        pbr_attrs      Which PBR attributes to stack after emission.
                       Set to [] to get emission-only (3-channel) input.
    """

    _ATTR_CHANNELS: Dict[str, int] = {
        "base_color": 3, "metallic": 1, "roughness": 1, "alpha": 1
    }
    EMISSION_CHANNELS: int = 3

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        resolution: int = 256,
        split: Optional[str] = "train",
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
        pbr_attrs: List[str] = ("base_color", "metallic", "roughness", "alpha"),
        roots: Optional[Any] = None,  # accepted but ignored (TrellisDataModule injects this)
    ) -> None:
        super().__init__()
        self.data_root   = data_root
        self.resolution  = resolution
        self.pbr_attrs   = list(pbr_attrs)
        self.split       = split
        self.value_range = (-1, 1)

        # channel layout: emission first, then pbr_attrs
        self.emission_channels = self.EMISSION_CHANNELS
        pbr_ch = sum(self._ATTR_CHANNELS[a] for a in self.pbr_attrs)
        self.in_channels = self.emission_channels + pbr_ch

        # layout dict for visualisation / downstream use
        self.layout: Dict[str, slice] = {"emissive": slice(0, self.emission_channels)}
        cur = self.emission_channels
        for a in self.pbr_attrs:
            c = self._ATTR_CHANNELS[a]
            self.layout[a] = slice(cur, cur + c)
            cur += c

        self.samples = _build_parquet_samples(
            data_root, parquet_file, split, min_emission, split_indices, split_json
        )
        # Per-sample "load" for BalancedResumableSampler (and collate split_size).
        # Emission .vxz file size is proportional to voxel count (fixed-byte-per-
        # voxel attrs), so it's a cheap, accurate proxy for per-sample memory.
        # 112x variance in practice — unbalanced assignment to ranks causes OOM.
        self.loads = [
            os.path.getsize(self._emission_vxz(s["sha256"])) for s in self.samples
        ]
        print(
            f"[LightGenSLatEmission] split={split!r}: {len(self.samples)} samples  "
            f"res={resolution}  in_channels={self.in_channels}  "
            f"({self.emission_channels} emission + {pbr_ch} PBR)"
        )

    def _emission_vxz(self, sha256: str) -> str:
        return os.path.join(
            self.data_root, f"emission_voxels_{self.resolution}", f"{sha256}.vxz"
        )

    def _pbr_vxz(self, sha256: str) -> str:
        return os.path.join(
            self.data_root, f"pbr_voxels_{self.resolution}", f"{sha256}.vxz"
        )

    def get_instance(self, sha256: str) -> Dict[str, Any]:
        # emission — always loaded
        coords_e, attr_e = _o_voxel.io.read_vxz(self._emission_vxz(sha256), num_threads=4)
        emit_feats = attr_e["emissive"].float() / 255.0 * 2.0 - 1.0  # [N, 3]

        if self.pbr_attrs:
            # pbr — same coordinates, verified at voxelisation time
            _, attr_p = _o_voxel.io.read_vxz(self._pbr_vxz(sha256), num_threads=4)
            pbr_feats = torch.cat(
                [attr_p[a].float() / 255.0 * 2.0 - 1.0 for a in self.pbr_attrs], dim=-1
            )  # [N, C_pbr]
            feats = torch.cat([emit_feats, pbr_feats], dim=-1)  # [N, 3+C_pbr]
        else:
            feats = emit_feats

        batch_col = torch.zeros(coords_e.shape[0], 1, dtype=coords_e.dtype)
        x = SparseTensor(feats, torch.cat([batch_col, coords_e], dim=-1))
        return {"x": x, "sha256": sha256}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sha256 = self.samples[index]["sha256"]
        try:
            return self.get_instance(sha256)
        except Exception as exc:
            print(f"[LightGenSLatEmission] Error loading {sha256}: {exc}")
            raise

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int] = None):
        return _sparse_collate(batch, split_size)


# ---------------------------------------------------------------------------
# Dataset 2 — pre-encoded emission latents for DiT training
# ---------------------------------------------------------------------------

class LightGenSLatEmissionLatent(Dataset):
    """
    Returns pre-encoded latents for training the emission DiT.

    Each sample contains:
        x_0           SparseTensor  [N, C_emit_lat]   emission latent (noised by flow matching)
        concat_cond   SparseTensor  [N, C_pbr_lat]    PBR material latent (concatenated per-token
                                                       as DiT input; same coordinates as x_0)
        cond          SparseTensor  [M, C_shape_lat]  shape latent (geometry cross-attention),
                                                       only if ``shape_latent_root`` is provided
        sha256        str

    The DiT receives ``cat(x_t, concat_cond)`` as its sequence input and cross-attends to
    ``cond`` (shape) — mirroring the SLatPbr → shape_latent conditioning in TRELLIS.2.

    Args:
        data_root              Root of the processed data tree.
        parquet_file           Path to the parquet index.
        emission_latent_root   Directory containing ``emission_latents/{latent_name}/``.
                               Defaults to ``data_root``.
        pbr_latent_root        Directory containing ``pbr_latents/{latent_name}/``.
                               Defaults to ``data_root``.
        emission_latent_name   Subfolder name for emission latents.
        pbr_latent_name        Subfolder name for PBR latents.
        shape_latent_root      Optional directory for shape latents (geometry conditioning).
        shape_latent_name      Subfolder name for shape latents.
        resolution             Voxel grid resolution (used to look up latent tokens count).
        split                  "train" / "val" / "test" / None.
        min_emission           Keep only samples with max-emission ≥ this value.
        split_indices          Override split selection.
        split_json             Explicit path to split JSON.
        emission_normalization Optional {"mean": [...], "std": [...]} to normalise emission latent.
        pbr_normalization      Optional normalisation for the PBR latent.
        shape_normalization    Optional normalisation for the shape latent.
    """

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        emission_latent_name: str,
        pbr_latent_name: str,
        emission_latent_root: Optional[str] = None,
        pbr_latent_root: Optional[str] = None,
        shape_latent_root: Optional[str] = None,
        shape_latent_name: Optional[str] = None,
        thumbnail_dir: Optional[str] = None,
        image_size: int = 512,
        resolution: int = 256,
        split: Optional[str] = "train",
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
        emission_normalization: Optional[Dict] = None,
        pbr_normalization: Optional[Dict] = None,
        shape_normalization: Optional[Dict] = None,
        dummy_cond_channels: int = 32,
        roots: Optional[Any] = None,  # accepted but ignored
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.resolution = resolution
        self.split = split
        self.dummy_cond_channels = dummy_cond_channels
        self.thumbnail_dir = thumbnail_dir
        self.image_size = image_size

        self.emit_dir   = os.path.join(emission_latent_root or data_root,
                                       "emission_latents", emission_latent_name)
        self.pbr_dir    = os.path.join(pbr_latent_root or data_root,
                                       "pbr_latents",      pbr_latent_name)
        self.shape_dir  = (
            os.path.join(shape_latent_root or data_root, "shape_latents", shape_latent_name)
            if shape_latent_name
            else None
        )

        self.emit_norm  = emission_normalization
        self.pbr_norm   = pbr_normalization
        self.shape_norm = shape_normalization

        self.samples = _build_parquet_samples(
            data_root, parquet_file, split, min_emission, split_indices, split_json
        )

        if self.samples:
            self._validate_first_sample()

        print(
            f"[LightGenSLatEmissionLatent] split={split!r}: {len(self.samples)} samples, "
            f"emit_dir={self.emit_dir}, shape_dir={self.shape_dir}, "
            f"thumbnail_dir={self.thumbnail_dir}"
        )

    def _validate_first_sample(self) -> None:
        sha256 = self.samples[0]["sha256"]
        for d, label in [(self.emit_dir, "emission"), (self.pbr_dir, "PBR material")]:
            p = os.path.join(d, f"{sha256}.npz")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{label} latent not found: {p}\n"
                    f"Run the corresponding encode_*_latent.py script first."
                )

    def get_instance(self, sha256: str) -> Dict[str, Any]:
        x_0 = _read_npz_as_sparse(
            os.path.join(self.emit_dir, f"{sha256}.npz"),
            normalization=self.emit_norm,
        )
        pbr_z = _read_npz_as_sparse(
            os.path.join(self.pbr_dir, f"{sha256}.npz"),
            normalization=self.pbr_norm,
        )
        assert torch.equal(x_0.coords, pbr_z.coords), (
            f"Emission and PBR latent coords mismatch for {sha256}: "
            f"{x_0.coords.shape} vs {pbr_z.coords.shape}"
        )

        # Build concat_cond: shape(32) + PBR(32) = 64ch, or PBR-only (32ch)
        if self.shape_dir is not None:
            shape_path = os.path.join(self.shape_dir, f"{sha256}.npz")
            if os.path.exists(shape_path):
                shape_z = _read_npz_as_sparse(shape_path, normalization=self.shape_norm)
                assert torch.equal(x_0.coords, shape_z.coords), (
                    f"Emission and shape latent coords mismatch for {sha256}"
                )
                concat_cond = x_0.replace(
                    torch.cat([shape_z.feats, pbr_z.feats], dim=-1)
                )
            else:
                concat_cond = pbr_z
        else:
            concat_cond = pbr_z

        out: Dict[str, Any] = {
            "x_0":         x_0,
            "concat_cond": concat_cond,
            "sha256":      sha256,
        }

        # Image conditioning: load thumbnail if available
        if self.thumbnail_dir is not None:
            img_path = os.path.join(self.thumbnail_dir, f"{sha256}.png")
            if os.path.exists(img_path):
                from PIL import Image
                image = Image.open(img_path).convert("RGBA")
                image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
                alpha = torch.tensor(
                    np.array(image.getchannel(3)), dtype=torch.float32
                ) / 255.0
                rgb = torch.tensor(
                    np.array(image.convert("RGB")), dtype=torch.float32
                ).permute(2, 0, 1) / 255.0
                out["cond"] = rgb * alpha.unsqueeze(0)
                return out

        # Fallback: dummy cond for cross-attention
        out["cond"] = torch.zeros(1, self.dummy_cond_channels)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sha256 = self.samples[index]["sha256"]
        try:
            return self.get_instance(sha256)
        except Exception as exc:
            print(f"[LightGenSLatEmissionLatent] Error loading {sha256}: {exc}")
            raise

    @torch.no_grad()
    def visualize_sample(self, sample: dict) -> dict:
        """Max-projection visualization of emission latent channels."""
        x_0 = sample["x_0"]  # SparseTensor [N, C]
        res = self.resolution // 16  # latent resolution

        def _max_proj(sp_tensor: SparseTensor, channels: slice) -> torch.Tensor:
            """Render a batch of max-projection images (B, 3, H, W)."""
            images = []
            for b in range(sp_tensor.shape[0]):
                mask = sp_tensor.coords[:, 0] == b
                coords = sp_tensor.coords[mask, 1:]  # [M, 3]
                feats = sp_tensor.feats[mask, channels]  # [M, C']
                C = feats.shape[-1]
                if C >= 3:
                    rgb = feats[:, :3]
                else:
                    rgb = torch.zeros(feats.shape[0], 3, device=feats.device)
                    rgb[:, :C] = feats
                lo, hi = rgb.min(), rgb.max()
                if hi - lo > 1e-6:
                    rgb = (rgb - lo) / (hi - lo)
                else:
                    rgb = rgb * 0 + 0.5
                img = torch.zeros(res, res, 3, device=feats.device)
                depth = torch.full((res, res), -1.0, device=feats.device)
                mag = rgb.norm(dim=-1)
                c = coords.long()
                for i in range(c.shape[0]):
                    y, x = c[i, 1].item(), c[i, 0].item()
                    if 0 <= y < res and 0 <= x < res and mag[i] > depth[y, x]:
                        depth[y, x] = mag[i]
                        img[y, x] = rgb[i]
                images.append(img.permute(2, 0, 1))  # [3, H, W]
            return torch.stack(images)  # [B, 3, H, W]

        vis = {}
        vis["emission_latent"] = _max_proj(x_0, slice(0, 3))
        if "concat_cond" in sample:
            vis["cond_latent"] = _max_proj(sample["concat_cond"], slice(0, 3))
        if "cond" in sample and sample["cond"].ndim == 4:
            vis["image_cond"] = sample["cond"]  # [B, 3, H, W]
        return vis

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int] = None):
        return _sparse_collate(batch, split_size)
