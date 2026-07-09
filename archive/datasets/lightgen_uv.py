"""
Datasets for LightGen: emission map generation conditioned on material and geometry.

Data source: baked UV maps from TexVerse-1K stored as somage.npz files.

Each somage.npz contains (all at 512×512 resolution):
  - occupancy   : (H, W, 1)  bool        UV occupancy mask
  - position    : (H, W, 3)  uint16      3D vertex positions, range [-2, 2]
  - objnormal   : (H, W, 3)  uint16      Object-space normals, range [-1, 1]
  - color       : (H, W, 3)  uint8       Albedo (base color)
  - metal       : (H, W, 1)  uint8       Metallic value
  - rough       : (H, W, 1)  uint8       Roughness value
  - emission_color : (H, W, 3)  uint8   Emission color (prediction target)

Conditioning stack (11 channels):
    position(3) + normal(3) + albedo(3) + metallic(1) + roughness(1) = 11

Target (3 channels):
    emission_color(3), normalized to [-1, 1]

Two dataset classes are provided:
  LightGenUVEmission
      Raw UV data for VAE training.
      Returns x_0 (emission), cond (geo+mat), mask (occupancy).

  LightGenUVEmissionLatent
      Pre-encoded emission latents + conditioning for DiT flow-matching training.
      Expects encode_lightgen_emission_latent.py to have been run beforehand.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _decode_uint16(data: np.ndarray, lo: float = -2.0, hi: float = 2.0) -> np.ndarray:
    """Decode uint16 → float in [lo, hi]."""
    return (data.astype(np.float32) / 65535.0) * (hi - lo) + lo


def _decode_uint8(data: np.ndarray) -> np.ndarray:
    """Decode uint8 → float in [0, 1]."""
    return data.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Base class shared by both datasets
# ---------------------------------------------------------------------------

class _LightGenUVBase(Dataset):
    """
    Shared infrastructure for LightGen UV datasets.

    Parquet index columns of interest:
      - ditem_dir     : relative path within data_root (e.g. "000-000/<uuid>")
      - success       : bool — whether processing succeeded
      - split         : "train" / "val" / "test"  (present in emission-filtered parquet)
      - #stat_emission_color_max : float — max emission across the sample

    Args:
        data_root      : Root directory of baked_uv_local_subset.
        parquet_file   : Path to the parquet index file.
        split          : One of "train", "val", "test" or None (all data).
        uv_size        : Target (H, W) to resize UV maps.  Default: (256, 256).
        min_emission   : Minimum #stat_emission_color_max to keep a sample.
                         Set to 0.0 to keep all (including non-emissive).
        split_indices  : Override split selection.  Accepts:
                         - tuple (start, end)   → slice
                         - list[int]            → explicit indices
                         - str                  → path to JSON with {"train": [...], ...}
    """

    _NPZ_NAME = "somage.npz"

    # Default split JSON filenames, searched relative to data_root
    _DEFAULT_SPLIT_JSONS = [
        "data_splits_emission_filtered.json",
        "data_splits_filtered.json",
    ]

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        split: Optional[str] = "train",
        uv_size: Tuple[int, int] = (256, 256),
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test", None), f"Invalid split: {split}"
        self.data_root = data_root
        self.split = split
        self.uv_size = uv_size

        # Auto-discover split JSON if not explicitly given
        if split_indices is None and split_json is None and split is not None:
            split_json = self._find_split_json(data_root)

        if split_json is not None and split_indices is None:
            split_indices = split_json  # _apply_split_indices accepts a JSON path

        self.samples = self._build_sample_list(
            parquet_file, split, min_emission, split_indices
        )

    # ------------------------------------------------------------------
    # Sample list construction
    # ------------------------------------------------------------------

    def _find_split_json(self, data_root: str) -> Optional[str]:
        """Return the first matching default split JSON file found under data_root."""
        for name in self._DEFAULT_SPLIT_JSONS:
            path = os.path.join(data_root, name)
            if os.path.exists(path):
                return path
        return None

    def _build_sample_list(
        self,
        parquet_file: str,
        split: Optional[str],
        min_emission: float,
        split_indices: Optional[Any],
    ) -> List[Dict[str, str]]:
        df = pd.read_parquet(parquet_file)

        # Keep only successfully processed rows
        if "success" in df.columns:
            df = df[df["success"] == True]

        # Filter by emission (useful to skip near-zero emission objects)
        emission_col = "#stat_emission_color_max"
        if min_emission > 0.0 and emission_col in df.columns:
            df = df[df[emission_col] >= min_emission]

        # Split selection
        if split is not None and split_indices is None and "split" in df.columns:
            df = df[df["split"] == split]
        
        samples: List[Dict[str, str]] = []
        for sample_id, row in df.iterrows():
            ditem_dir = row.get("ditem_dir", None)
            if ditem_dir is None:
                # Fallback: derive from UUID
                sid = str(sample_id)
                ditem_dir = f"{sid[:3]}-{sid[3:6]}/{sid}"
            samples.append({
                "sample_id": str(sample_id),
                "npz_file": os.path.join(self.data_root, ditem_dir, self._NPZ_NAME),
            })

        # Override with explicit indices if given
        if split_indices is not None:
            samples = self._apply_split_indices(samples, split_indices)

        print(
            f"[LightGen] {self.__class__.__name__} "
            f"split={split!r}: {len(samples)} samples"
        )
        return samples

    def _apply_split_indices(
        self,
        samples: List[Dict[str, str]],
        indices: Any,
    ) -> List[Dict[str, str]]:
        if isinstance(indices, tuple):
            return samples[indices[0] : indices[1]]
        if isinstance(indices, list):
            return [samples[i] for i in indices]
        if isinstance(indices, str):
            # Path to a JSON file.  Supported schemas:
            #   {"train": [0, 1, ...], "val": [...]}               (flat list)
            #   {"train": {"indices": [0, 1, ...]}, "val": {...}}  (dict with "indices" key)
            with open(indices) as f:
                split_data = json.load(f)
            if self.split is None:
                # No split requested — use all available indices across all splits
                idx_list = []
                for v in split_data.values():
                    if isinstance(v, list):
                        idx_list.extend(v)
                    elif isinstance(v, dict):
                        idx_list.extend(v.get("indices", []))
            else:
                entry = split_data.get(self.split, [])
                if isinstance(entry, dict):
                    idx_list = entry.get("indices", [])
                else:
                    idx_list = entry
            return [samples[i] for i in idx_list]
        return samples

    # ------------------------------------------------------------------
    # Low-level NPZ loading
    # ------------------------------------------------------------------

    def _load_npz(self, npz_file: str) -> Dict[str, torch.Tensor]:
        """
        Load a somage.npz file and return decoded torch tensors in [C, H, W] layout.

        Returned keys and shapes (at native 512×512):
          occupancy   [1, H, W]   float32, {0, 1}
          position    [3, H, W]   float32, range [-2, 2]
          normal      [3, H, W]   float32, range [-1, 1]
          albedo      [3, H, W]   float32, range [0, 1]
          metallic    [1, H, W]   float32, range [0, 1]
          roughness   [1, H, W]   float32, range [0, 1]
          emission    [3, H, W]   float32, range [0, 1]  (NOT yet scaled to [-1,1])
        """
        raw = np.load(npz_file)
        arrays = {
            "occupancy": torch.from_numpy(raw["occupancy"].copy()).float(),        # HW1 bool→float
            "position":  torch.from_numpy(_decode_uint16(raw["position"].copy())), # HW3
            "normal":    torch.from_numpy(_decode_uint16(raw["objnormal"].copy(), -1.0, 1.0)),  # HW3
            "albedo":    torch.from_numpy(_decode_uint8(raw["color"].copy())),      # HW3
            "metallic":  torch.from_numpy(_decode_uint8(raw["metal"].copy())),      # HW1
            "roughness": torch.from_numpy(_decode_uint8(raw["rough"].copy())),      # HW1
            "emission":  torch.from_numpy(_decode_uint8(raw["emission_color"].copy())),  # HW3
        }
        raw.close()

        # [H, W, C] → [C, H, W]
        arrays = {k: v.permute(2, 0, 1) for k, v in arrays.items()}

        # Resize to target UV resolution if needed
        th, tw = self.uv_size
        h, w = arrays["occupancy"].shape[1:]
        if (h, w) != (th, tw):
            arrays["occupancy"] = F.interpolate(
                arrays["occupancy"].unsqueeze(0), (th, tw), mode="nearest"
            ).squeeze(0)
            continuous_keys = ["position", "normal", "albedo", "metallic", "roughness", "emission"]
            for k in continuous_keys:
                arrays[k] = F.interpolate(
                    arrays[k].unsqueeze(0), (th, tw), mode="bilinear", align_corners=False
                ).squeeze(0)

        return arrays

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"n={len(self)}, "
            f"uv_size={self.uv_size})"
        )


# ---------------------------------------------------------------------------
# Dataset 1: raw UV data for VAE training
# ---------------------------------------------------------------------------

class LightGenUVEmission(_LightGenUVBase):
    """
    UV-space dataset for training the LightGen emission VAE.

    Each sample exposes:
      x_0       [3, H, W]    Emission map, normalized to [-1, 1].
      cond      [11, H, W]   Geometry + material conditioning channels:
                               position(3) + normal(3) + albedo(3) + metallic(1) + roughness(1)
                             All channels are in their native normalized ranges (no extra scaling).
      mask      [1, H, W]    UV occupancy mask, values in {0, 1}.
      sample_id str          Sample identifier.

    Args:
        data_root       : Root of baked_uv_local_subset/.
        parquet_file    : Path to parquet index.
        split           : "train" / "val" / "test" / None.
        uv_size         : (H, W) to resize UV maps to.  Default: (256, 256).
        min_emission    : Keep only samples with max emission ≥ this value.
        split_indices   : Override split selection (see base class docstring).
        emission_range  : (min, max) clamping for emission before [-1,1] scaling.
                          Use (0, 1) (the default) to leave as-is.
    """

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        split: Optional[str] = "train",
        uv_size: Tuple[int, int] = (256, 256),
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
        emission_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__(data_root, parquet_file, split, uv_size, min_emission, split_indices, split_json)
        self.emission_range = emission_range

    def get_instance(self, npz_file: str, sample_id: str) -> Dict[str, Any]:
        arrays = self._load_npz(npz_file)

        emission = arrays["emission"]
        # Optional clamping before normalizing to [-1, 1]
        e_lo, e_hi = self.emission_range
        if e_lo != 0.0 or e_hi != 1.0:
            emission = emission.clamp(e_lo, e_hi)
            emission = (emission - e_lo) / (e_hi - e_lo)
        # Scale to [-1, 1] (diffusion target convention)
        x_0 = emission * 2.0 - 1.0

        # Conditioning: geo + material channels
        cond = torch.cat([
            arrays["position"],   # [3, H, W]  range [-2, 2]
            arrays["normal"],     # [3, H, W]  range [-1, 1]
            arrays["albedo"],     # [3, H, W]  range [0, 1]
            arrays["metallic"],   # [1, H, W]  range [0, 1]
            arrays["roughness"],  # [1, H, W]  range [0, 1]
        ], dim=0)                 # [11, H, W]

        return {
            "x_0": x_0,              # [3, H, W]  emission target in [-1, 1]
            "cond": cond,            # [11, H, W] conditioning features
            "mask": arrays["occupancy"],  # [1, H, W]
            "sample_id": sample_id,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.samples[index]
        try:
            return self.get_instance(info["npz_file"], info["sample_id"])
        except Exception as exc:
            print(f"[LightGenUVEmission] Error loading {info['sample_id']}: {exc}")
            raise

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stack dense tensors; pass sample_ids as a list."""
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}
        out: Dict[str, Any] = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        return out


# ---------------------------------------------------------------------------
# Dataset 2: pre-encoded latents for DiT flow-matching training
# ---------------------------------------------------------------------------

class LightGenUVEmissionLatent(_LightGenUVBase):
    """
    Dataset for training the LightGen emission DiT using pre-encoded latents.

    Assumes that an encoding script (e.g. encode_lightgen_emission_latent.py) has
    already encoded the emission UV maps with the trained VAE and saved the results
    as <latent_root>/<sample_id>.npz with keys:
        z        : [C_lat, H_lat, W_lat]  float32  encoded emission latent
        (optionally cond_z if the conditioning geo+mat has also been encoded)

    If ``cond_latent_root`` is None (the default), the conditioning is loaded raw
    from somage.npz at the configured uv_size — the DiT cross-attends to the raw
    11-channel geo+mat UV map.  This is the recommended setup for initial experiments.

    If ``cond_latent_root`` is provided, a pre-encoded 2D feature map for the geo+mat
    conditioning is loaded from <cond_latent_root>/<sample_id>.npz (key: ``cond_z``).

    Returns per sample:
        x_0       [C_lat, H_lat, W_lat]       Emission latent (to be noised by flow matching).
        cond      [11, H, W] or [C_cond, ...]  Geo+mat conditioning.
        mask      [1, H, W]                    UV occupancy mask (raw resolution).
        sample_id str
    """

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        latent_root: str,
        split: Optional[str] = "train",
        uv_size: Tuple[int, int] = (256, 256),
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
        cond_latent_root: Optional[str] = None,
        normalization: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        super().__init__(data_root, parquet_file, split, uv_size, min_emission, split_indices, split_json)
        self.latent_root = latent_root
        self.cond_latent_root = cond_latent_root

        # Optional per-channel normalization of the emission latent
        # normalization = {"mean": [...], "std": [...]}
        if normalization is not None:
            self.latent_mean = torch.tensor(normalization["mean"]).float()
            self.latent_std  = torch.tensor(normalization["std"]).float()
        else:
            self.latent_mean = None
            self.latent_std  = None

        # Validate that latent files exist for at least the first sample
        if self.samples:
            first_lat = os.path.join(latent_root, f"{self.samples[0]['sample_id']}.npz")
            if not os.path.exists(first_lat):
                raise FileNotFoundError(
                    f"Emission latent not found: {first_lat}\n"
                    "Run encode_lightgen_emission_latent.py first."
                )

    def _load_latent(self, sample_id: str) -> torch.Tensor:
        path = os.path.join(self.latent_root, f"{sample_id}.npz")
        data = np.load(path)
        z = torch.from_numpy(data["z"].copy()).float()
        data.close()
        if self.latent_mean is not None:
            # Reshape for broadcasting: latent is [C, H, W]
            mean = self.latent_mean.view(-1, 1, 1)
            std  = self.latent_std.view(-1, 1, 1)
            z = (z - mean) / std
        return z

    def _load_cond_latent(self, sample_id: str) -> torch.Tensor:
        path = os.path.join(self.cond_latent_root, f"{sample_id}.npz")
        data = np.load(path)
        cond_z = torch.from_numpy(data["cond_z"].copy()).float()
        data.close()
        return cond_z

    def get_instance(self, npz_file: str, sample_id: str) -> Dict[str, Any]:
        x_0 = self._load_latent(sample_id)

        if self.cond_latent_root is not None:
            cond = self._load_cond_latent(sample_id)
            # Still need mask — load it from the raw NPZ
            arrays = self._load_npz(npz_file)
            mask = arrays["occupancy"]
        else:
            arrays = self._load_npz(npz_file)
            cond = torch.cat([
                arrays["position"],
                arrays["normal"],
                arrays["albedo"],
                arrays["metallic"],
                arrays["roughness"],
            ], dim=0)   # [11, H, W]
            mask = arrays["occupancy"]

        return {
            "x_0": x_0,
            "cond": cond,
            "mask": mask,
            "sample_id": sample_id,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.samples[index]
        try:
            return self.get_instance(info["npz_file"], info["sample_id"])
        except Exception as exc:
            print(f"[LightGenUVEmissionLatent] Error loading {info['sample_id']}: {exc}")
            raise

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}
        out: Dict[str, Any] = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        return out


# ---------------------------------------------------------------------------
# Dataset 3: joint raw loading (VAE input + conditioning together)
#             Useful for debugging / visualisation pipelines
# ---------------------------------------------------------------------------

class LightGenUVFull(_LightGenUVBase):
    """
    Returns all decoded UV-space channels for a sample.

    Useful for inspection, visualisation and data-pipeline debugging.

    Returns per sample:
        position    [3, H, W]
        normal      [3, H, W]
        albedo      [3, H, W]
        metallic    [1, H, W]
        roughness   [1, H, W]
        emission    [3, H, W]   range [0, 1]  (NOT scaled to [-1, 1])
        mask        [1, H, W]
        cond        [11, H, W]  position + normal + albedo + metal + rough
        sample_id   str
    """

    def get_instance(self, npz_file: str, sample_id: str) -> Dict[str, Any]:
        arrays = self._load_npz(npz_file)
        cond = torch.cat([
            arrays["position"],
            arrays["normal"],
            arrays["albedo"],
            arrays["metallic"],
            arrays["roughness"],
        ], dim=0)

        return {
            "position": arrays["position"],
            "normal":   arrays["normal"],
            "albedo":   arrays["albedo"],
            "metallic": arrays["metallic"],
            "roughness": arrays["roughness"],
            "emission": arrays["emission"],
            "mask":     arrays["occupancy"],
            "cond":     cond,
            "sample_id": sample_id,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.samples[index]
        try:
            return self.get_instance(info["npz_file"], info["sample_id"])
        except Exception as exc:
            print(f"[LightGenUVFull] Error loading {info['sample_id']}: {exc}")
            raise

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}
        out: Dict[str, Any] = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        return out
