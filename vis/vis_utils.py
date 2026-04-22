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
    latent_names: Optional[dict[str, str]],
) -> SampleRecord:
    emission_voxel_path = os.path.join(data_root, f"emission_voxels_{resolution}", f"{sha}.vxz")
    pbr_voxel_path = os.path.join(data_root, f"pbr_voxels_{resolution}", f"{sha}.vxz")
    glb_path = os.path.join(glb_root, glb_rel) if (glb_root and glb_rel) else None
    thumbnail_path = os.path.join(data_root, "thumbnails", f"{sha}.png")
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
    parquet_path: Optional[str] = None,
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
        parquet_path: explicit path to the metadata parquet used when ``source``
            is a splits JSON.  If omitted, the parquet is auto-discovered by
            globbing ``<data_root>/df_*.parquet`` (exactly one match required).
        require: attribute names that must exist on disk to include a record.

    Returns SampleRecords filtered to those where all required paths exist.
    """
    # --- Resolve sha list + per-sha extras from source ---------------------
    if select.startswith("sha:"):
        shas = [s.strip() for s in select[4:].split(",") if s.strip()]
        glb_map: dict[str, str] = {}
    elif source.endswith(".json"):
        with open(source) as f:
            splits = json.load(f)
        split_name = select[6:] if select.startswith("split:") else "train"
        entry = splits.get(split_name, [])
        if isinstance(entry, dict):
            # Format: {"indices": [0, 1, ...]} — resolve via the parquet in data_root.
            idx_list: list[int] = entry.get("indices", [])
            if parquet_path is None:
                import glob as _glob
                matches = _glob.glob(os.path.join(data_root, "df_*.parquet"))
                if len(matches) == 1:
                    parquet_path = matches[0]
                else:
                    raise FileNotFoundError(
                        f"could not auto-discover parquet under {data_root!r}; "
                        f"pass parquet_path= (found {len(matches)} matches)"
                    )
            _df = pd.read_parquet(parquet_path)
            all_shas = list(_df.index.astype(str))
            shas = [all_shas[i] for i in idx_list if i < len(all_shas)]
        else:
            raise ValueError(
                f"splits entry for {split_name!r} has unsupported type "
                f"{type(entry).__name__}"
            )
        glb_map: dict[str, str] = {}
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
            latent_names=latent_names,
        )
        if rec.has_all(require):
            records.append(rec)
    return records


# =============================================================================
# Voxel I/O and metrics
# =============================================================================

def read_emission_voxel(path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Read an emission .vxz and return (coords, feats_normed, resolution).

    coords: int32 [N, 3], voxel grid indices.
    feats_normed: float32 [N, 3] in [-1, 1] (linear emission, *not* sRGB).
    resolution: int grid size. Inferred as 2 ** ceil(log2(coords.max()+1))
        if the file header does not expose it.
    """
    coords, attr = o_voxel.io.read_vxz(path, num_threads=4)
    feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
    max_idx = int(coords.max().item()) + 1 if coords.numel() else 1
    # Known resolutions in this project are 256 and 512; pick the smallest
    # power of 2 not below max_idx.
    res = 1
    while res < max_idx:
        res <<= 1
    return coords, feats, res


def emission_to_srgb_u8(feats_norm: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] linear emission to uint8 sRGB.

    Identical curve to vis/vis_3d_overfit.py:184–192.
    """
    linear = (feats_norm * 0.5 + 0.5).clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    return (srgb.clamp(0, 1) * 255).round().byte()


def compute_emission_l1(
    pred_coords: torch.Tensor,
    pred_feats: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_feats: torch.Tensor,
) -> float:
    """Mean L1 between pred and GT emission features at matching voxel positions.

    In all our configs, pred voxels live on exactly the same grid as GT (VAEs and
    DiTs do not change the sparsity pattern — they only change features), so we
    assert equal row counts and compute per-row L1. Both tensors are expected in
    [-1, 1] normalised space.
    """
    if pred_coords.shape[0] != gt_coords.shape[0]:
        raise ValueError(
            f"coord count mismatch: pred {pred_coords.shape[0]} vs gt {gt_coords.shape[0]}"
        )
    return float((pred_feats - gt_feats).abs().mean().item())


# =============================================================================
# Experiment resolution (run_dir or ckpt path → config + ckpt)
# =============================================================================

@dataclass
class ExperimentHandle:
    label: str
    run_dir: str
    config: dict
    ckpt_path: str
    trainer_name: str

    @property
    def data_root(self) -> str:
        return self.config["dataset"]["args"]["data_root"]

    @property
    def resolution(self) -> int:
        return int(self.config["dataset"]["args"].get("resolution", 256))


_STEP_VAL_RE = re.compile(r"step(\d+)-(\d+\.\d+)\.ckpt$")


def _pick_checkpoint(ckpt_dir: str) -> str:
    """Apply the checkpoint-selection rule from the design spec:

    1. If ckpt_dir/last.ckpt exists, use it.
    2. Else, parse files matching ``step{N}-{val_loss:.4f}.ckpt`` and pick
       the smallest val_loss.
    3. Else, scan for any *.ckpt and pick the lexicographically latest, or
       raise FileNotFoundError if none exist.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"no ckpts/ dir at {ckpt_dir}")
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    best = None
    best_val = float("inf")
    for f in os.listdir(ckpt_dir):
        m = _STEP_VAL_RE.search(f)
        if m:
            v = float(m.group(2))
            if v < best_val:
                best_val = v
                best = f
    if best:
        return os.path.join(ckpt_dir, best)
    fallbacks = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt"))
    if fallbacks:
        return os.path.join(ckpt_dir, fallbacks[-1])
    raise FileNotFoundError(f"no checkpoints found in {ckpt_dir}")


def resolve_experiment(label: str, path: str) -> ExperimentHandle:
    """Turn a run_dir or a .ckpt path into an ExperimentHandle."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        run_dir = path
        ckpt_path = _pick_checkpoint(os.path.join(run_dir, "ckpts"))
    elif path.endswith(".ckpt") and os.path.exists(path):
        run_dir = os.path.dirname(os.path.dirname(path))
        ckpt_path = path
    else:
        raise FileNotFoundError(
            f"experiment path {path!r} is neither a run_dir nor an existing .ckpt"
        )
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"no config.yaml at {cfg_path}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    trainer_name = config.get("trainer", {}).get("name", "<unknown>")
    return ExperimentHandle(
        label=label,
        run_dir=run_dir,
        config=config,
        ckpt_path=ckpt_path,
        trainer_name=trainer_name,
    )


# =============================================================================
# VAE loading with trainer-type dispatch (EmissionVaeTrainer vs
# EmissionPbrFinetuneTrainer) — shared by vis_3d_vae.py and vis_3d_dit.py.
# =============================================================================

# Constants mirror EmissionPbrFinetuneTrainer.{PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA}
# in trellis2/trainers/vae/emission_pbr_finetune.py.
_PBR_PAD_METALLIC = -1.0
_PBR_PAD_ROUGHNESS = 1.0
_PBR_PAD_ALPHA = 1.0


def _pack_emission_as_pbr(feats: torch.Tensor) -> torch.Tensor:
    """Pack [N, 3] emission into [N, 6] PBR layout used by the finetune trainer."""
    n = feats.shape[0]
    pad = torch.tensor(
        [[_PBR_PAD_METALLIC, _PBR_PAD_ROUGHNESS, _PBR_PAD_ALPHA]],
        device=feats.device, dtype=feats.dtype,
    ).expand(n, -1)
    return torch.cat([feats, pad], dim=-1)


def _load_state_into_submodule(module: torch.nn.Module, state: dict, prefix: str) -> None:
    sub = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    module.load_state_dict(sub)


def load_vae_from_handle(handle: "ExperimentHandle", device: str = "cuda"):
    """Load a VAE and return a ``decode_emission(coords, emit_feats, pbr_feats_or_none)``
    callable that produces the reconstructed [-1, 1] emission features.

    Branches on ``handle.trainer_name``:
      * ``EmissionVaeTrainer``            — two-stream concat (needs pbr_feats).
      * ``EmissionPbrFinetuneTrainer``    — channel hijack (ignores pbr_feats).

    Any other trainer raises a RuntimeError.
    """
    from trellis2.trainers.builders import build_models
    import trellis2.models as trellis_models
    from trellis2.modules import sparse as sp

    models = build_models(handle.config["models"])
    encoder, decoder = models["encoder"], models["decoder"]
    state = torch.load(handle.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    _load_state_into_submodule(encoder, state, "models.encoder.")
    _load_state_into_submodule(decoder, state, "models.decoder.")
    encoder.to(device).eval()
    decoder.to(device).eval()

    trainer = handle.trainer_name
    if trainer == "EmissionVaeTrainer":
        pbr_name = handle.config["trainer"]["args"].get("pretrained_pbr_encoder")
        pbr_enc = trellis_models.from_pretrained(pbr_name).eval().float().to(device) if pbr_name else None

        @torch.no_grad()
        def decode_emission(coords, emit_feats, pbr_feats):
            if pbr_feats is None:
                raise ValueError("EmissionVaeTrainer requires pbr_feats for the frozen PBR encoder")
            bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
            x_emit = sp.SparseTensor(emit_feats, torch.cat([bc, coords], dim=-1)).to(device)
            z_emit = encoder(x_emit)
            if pbr_enc is not None:
                x_pbr = sp.SparseTensor(pbr_feats, torch.cat([bc, coords], dim=-1)).to(device)
                z_pbr = pbr_enc(x_pbr)
                z = z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1))
            else:
                z = z_emit
            return decoder(z).to("cpu").feats

    elif trainer == "EmissionPbrFinetuneTrainer":
        @torch.no_grad()
        def decode_emission(coords, emit_feats, pbr_feats):
            # Ignore pbr_feats — channel-hijack packs emission as base_color.
            packed = _pack_emission_as_pbr(emit_feats)
            bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
            x = sp.SparseTensor(packed, torch.cat([bc, coords], dim=-1)).to(device)
            z = encoder(x)
            y = decoder(z).to("cpu").feats
            return y[:, :3]  # first 3 channels = emission

    else:
        raise RuntimeError(
            f"unsupported trainer {trainer!r} for VAE visualization; "
            f"expected EmissionVaeTrainer or EmissionPbrFinetuneTrainer"
        )

    return decode_emission
