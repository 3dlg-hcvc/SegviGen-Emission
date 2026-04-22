# 3D Visualization Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ~8 `vis_3d_*.py` scripts under `vis/` with two task-family scripts (`vis_3d_vae.py`, `vis_3d_dit.py`) sharing `vis/_vis_common.py`. Each script renders `GLB | GT emission | experiment_1 | ... | experiment_N` as synced Three.js panels with per-prediction `emission L1` captions.

**Architecture:** One shared helper module with sample loading, voxel I/O, checkpoint resolution, VAE trainer-type dispatch (`EmissionVaeTrainer` vs `EmissionPbrFinetuneTrainer`), Three.js panel builders, and HTML template emission. Two entry scripts consume the helpers; they differ only in whether they run a VAE encode+decode or a DiT sample-then-decode pipeline. GLB mesh and voxels share a canonical `[-0.5, 0.5]^3` frame — GLB gets re-normalized at load time the same way `data_toolkit/voxelize_pbr.py:77–86` did during voxelization; voxel integer coords become `i / res - 0.5 + 0.5 / res`. Per-canvas `OrbitControls` with `change`-event broadcast keeps all panels in sync.

**Tech Stack:** Python 3.10 (conda env `trellis2`), PyTorch, `o_voxel`, `trellis2` (local), Three.js r128 via CDN, GLTFLoader, plain `<select>` dropdown. Tests are assertion-based scripts (no pytest framework — follows `data_toolkit/test_*.py` convention).

**Spec:** `docs/superpowers/specs/2026-04-22-vis-3d-consolidation-design.md`.

**Before starting:** Run `git status` and ensure you're on the `lightgen` branch with a clean working tree. The user does not want cluster edits — all work happens locally.

---

## Task 1: Scaffold `_vis_common.py` and create `vis/legacy/`

**Files:**
- Create: `vis/_vis_common.py`
- Create: `vis/legacy/.gitkeep`

- [ ] **Step 1: Create the legacy directory with a placeholder**

Run: `mkdir -p /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/legacy && touch /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/legacy/.gitkeep`

- [ ] **Step 2: Create the empty helper module with the standard import block and module docstring**

Write `vis/_vis_common.py`:

```python
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
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `conda run -n trellis2 python -c "import sys; sys.path.insert(0, '/local-scratch/localhome/dya78/code/TRELLIS.2-lightning'); import vis._vis_common as _c; print('ok', _c.__file__)"`

Expected: `ok /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/_vis_common.py`

- [ ] **Step 4: Commit**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
git add vis/_vis_common.py vis/legacy/.gitkeep
git commit -m "vis: scaffold _vis_common.py + legacy/ directory"
```

---

## Task 2: `SampleRecord` + `load_samples`

**Files:**
- Modify: `vis/_vis_common.py` (append to end of file)
- Create: `vis/_vis_common_test.py`

- [ ] **Step 1: Append the data model and loader to `vis/_vis_common.py`**

```python
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
        shas = list(splits.get(split_name, []))
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
```

- [ ] **Step 2: Write the assertion-based test**

Write `vis/_vis_common_test.py`:

```python
"""Smoke tests for vis/_vis_common.py — run as a plain script."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vis._vis_common import SampleRecord, load_samples

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lightgen_test"))


def test_load_samples_from_parquet_first_3():
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=os.path.join(DATA_ROOT, "df_10samples.parquet"),
        select="first:3",
        glb_root=None,
        require=("emission_voxel_path",),
    )
    assert len(recs) == 3, f"expected 3, got {len(recs)}"
    for r in recs:
        assert isinstance(r, SampleRecord)
        assert os.path.exists(r.emission_voxel_path), r.emission_voxel_path


def test_load_samples_from_splits_json():
    splits = os.path.join(DATA_ROOT, "data_splits_10sample_as1k.json")
    if not os.path.exists(splits):
        print("skip: splits file missing")
        return
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=splits,
        select="split:val",
        require=("emission_voxel_path",),
    )
    assert len(recs) >= 1, "val split should have records"


def test_load_samples_sha_csv():
    recs_all = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=os.path.join(DATA_ROOT, "df_10samples.parquet"),
        select="all",
    )
    assert len(recs_all) == 10, f"expected 10 samples, got {len(recs_all)}"
    first_two = ",".join(r.sha for r in recs_all[:2])
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source="ignored",
        select=f"sha:{first_two}",
    )
    assert len(recs) == 2, f"expected 2, got {len(recs)}"


if __name__ == "__main__":
    test_load_samples_from_parquet_first_3()
    print("ok: parquet first:3")
    test_load_samples_from_splits_json()
    print("ok: splits json split:val")
    test_load_samples_sha_csv()
    print("ok: sha csv")
    print("all passed")
```

- [ ] **Step 3: Run the test**

Run: `cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning && conda run -n trellis2 python vis/_vis_common_test.py`

Expected output (each line on stdout):
```
ok: parquet first:3
ok: splits json split:val
ok: sha csv
all passed
```

If any test fails, inspect the raised `AssertionError` and the actual path values. Common failures: `data/lightgen_test/` missing some files (verify with `ls data/lightgen_test/`), or parquet column names shifted (verify with the `python3 -c "import pandas as pd; ..."` probe used in the spec-writing step).

- [ ] **Step 4: Commit**

```bash
git add vis/_vis_common.py vis/_vis_common_test.py
git commit -m "vis: SampleRecord + load_samples (parquet/splits/sha sources)"
```

---

## Task 3: Voxel I/O + emission L1

**Files:**
- Modify: `vis/_vis_common.py`
- Modify: `vis/_vis_common_test.py`

- [ ] **Step 1: Append voxel helpers to `vis/_vis_common.py`**

```python
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
    return (srgb.clamp(0, 1) * 255).byte()


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
```

- [ ] **Step 2: Append tests to `vis/_vis_common_test.py`**

Add after the existing imports and before `if __name__ == "__main__":` by editing the file. Insert these new test functions before the main block:

```python
def test_read_emission_voxel():
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=os.path.join(DATA_ROOT, "df_10samples.parquet"),
        select="first:1",
    )
    coords, feats, res = __import__("vis._vis_common", fromlist=["read_emission_voxel"]).read_emission_voxel(recs[0].emission_voxel_path)
    assert coords.ndim == 2 and coords.shape[1] == 3, coords.shape
    assert feats.ndim == 2 and feats.shape[1] == 3, feats.shape
    assert feats.min() >= -1.0 and feats.max() <= 1.0, (feats.min(), feats.max())
    assert res == 256, f"expected 256, got {res}"
    assert coords.shape[0] == feats.shape[0]


def test_emission_to_srgb_u8():
    from vis._vis_common import emission_to_srgb_u8
    import torch
    # black (-1, -1, -1) → 0; white (+1, +1, +1) → 255.
    black = emission_to_srgb_u8(torch.tensor([[-1.0, -1.0, -1.0]]))
    white = emission_to_srgb_u8(torch.tensor([[1.0, 1.0, 1.0]]))
    assert black.tolist() == [[0, 0, 0]], black.tolist()
    assert white.tolist() == [[255, 255, 255]], white.tolist()


def test_compute_emission_l1_zero_for_equal_inputs():
    from vis._vis_common import compute_emission_l1
    import torch
    coords = torch.zeros((4, 3), dtype=torch.int32)
    feats = torch.tensor([[0.1, 0.2, 0.3]] * 4)
    assert compute_emission_l1(coords, feats, coords, feats) == 0.0
```

Update the `if __name__ == "__main__":` block to call the three new tests too:

```python
if __name__ == "__main__":
    test_load_samples_from_parquet_first_3()
    print("ok: parquet first:3")
    test_load_samples_from_splits_json()
    print("ok: splits json split:val")
    test_load_samples_sha_csv()
    print("ok: sha csv")
    test_read_emission_voxel()
    print("ok: read_emission_voxel")
    test_emission_to_srgb_u8()
    print("ok: emission_to_srgb_u8")
    test_compute_emission_l1_zero_for_equal_inputs()
    print("ok: compute_emission_l1 zero")
    print("all passed")
```

- [ ] **Step 3: Run tests**

Run: `cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning && conda run -n trellis2 python vis/_vis_common_test.py`

Expected: all six `ok:` lines followed by `all passed`.

- [ ] **Step 4: Commit**

```bash
git add vis/_vis_common.py vis/_vis_common_test.py
git commit -m "vis: read_emission_voxel + emission_to_srgb_u8 + compute_emission_l1"
```

---

## Task 4: `resolve_experiment` + checkpoint selection

**Files:**
- Modify: `vis/_vis_common.py`
- Modify: `vis/_vis_common_test.py`

- [ ] **Step 1: Append experiment resolution to `vis/_vis_common.py`**

```python
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
```

- [ ] **Step 2: Append the test**

Add to `vis/_vis_common_test.py` (before the `if __name__` block):

```python
def test_resolve_experiment_run_dir():
    from vis._vis_common import resolve_experiment
    # Any existing run dir works — pick the overfit one that's been around longest.
    run_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "emission_vae_10sample_overfit")
    if not os.path.isdir(run_dir):
        print("skip: run_dir missing")
        return
    h = resolve_experiment("vae", run_dir)
    assert h.label == "vae"
    assert os.path.exists(h.ckpt_path), h.ckpt_path
    assert h.trainer_name in ("EmissionVaeTrainer", "EmissionPbrFinetuneTrainer"), h.trainer_name


def test_resolve_experiment_ckpt_file():
    from vis._vis_common import resolve_experiment
    run_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "emission_vae_10sample_overfit")
    if not os.path.isdir(run_dir):
        print("skip: run_dir missing")
        return
    ckpt = os.path.join(run_dir, "ckpts", "last.ckpt")
    if not os.path.exists(ckpt):
        print("skip: last.ckpt missing")
        return
    h = resolve_experiment("vae", ckpt)
    assert h.ckpt_path == os.path.abspath(ckpt)
    assert h.run_dir == os.path.abspath(run_dir)
```

And add calls in the `__main__` block:

```python
    test_resolve_experiment_run_dir()
    print("ok: resolve_experiment run_dir")
    test_resolve_experiment_ckpt_file()
    print("ok: resolve_experiment ckpt file")
```

- [ ] **Step 3: Run tests**

Run: `cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning && conda run -n trellis2 python vis/_vis_common_test.py`

Expected: all existing `ok:` lines plus the two new ones, then `all passed`.

- [ ] **Step 4: Commit**

```bash
git add vis/_vis_common.py vis/_vis_common_test.py
git commit -m "vis: resolve_experiment + ckpt-selection rule (last.ckpt > min val/loss)"
```

---

## Task 5: `load_vae_from_handle` with trainer-type dispatch

**Files:**
- Modify: `vis/_vis_common.py`

- [ ] **Step 1: Append the VAE loader**

```python
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
```

- [ ] **Step 2: Add a tiny integration test**

Append to `vis/_vis_common_test.py`:

```python
def test_load_vae_from_handle_smoke():
    from vis._vis_common import (
        load_vae_from_handle, resolve_experiment, load_samples, read_emission_voxel,
    )
    import torch
    run_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "emission_vae_10sample_overfit")
    if not (torch.cuda.is_available() and os.path.isdir(run_dir)):
        print("skip: cuda or run_dir missing")
        return
    h = resolve_experiment("vae", run_dir)
    decode = load_vae_from_handle(h)

    recs = load_samples(
        data_root=h.data_root,
        resolution=h.resolution,
        source=h.config["dataset"]["args"]["parquet_file"],
        select="first:1",
    )
    coords, emit, _ = read_emission_voxel(recs[0].emission_voxel_path)

    # Load PBR features (needed for the two-stream trainer; harmless otherwise).
    pbr_path = recs[0].pbr_voxel_path
    pbr_feats = None
    if os.path.exists(pbr_path):
        import o_voxel
        _, pbr_attr = o_voxel.io.read_vxz(pbr_path, num_threads=4)
        attrs = ["base_color", "metallic", "roughness", "alpha"]
        pbr_feats = torch.cat([pbr_attr[a].float() / 255.0 * 2.0 - 1.0 for a in attrs], dim=-1)

    y = decode(coords, emit, pbr_feats)
    assert y.shape == emit.shape, f"decoder output {y.shape} vs emit {emit.shape}"
```

Add the call to `__main__`:

```python
    test_load_vae_from_handle_smoke()
    print("ok: load_vae_from_handle smoke")
```

- [ ] **Step 3: Run tests**

Run: `cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning && conda run -n trellis2 python vis/_vis_common_test.py`

Expected: all `ok:` lines plus `ok: load_vae_from_handle smoke` (or `skip: ...` if CUDA/run_dir missing).

- [ ] **Step 4: Commit**

```bash
git add vis/_vis_common.py vis/_vis_common_test.py
git commit -m "vis: load_vae_from_handle with EmissionVae/PbrFinetune dispatch"
```

---

## Task 6: Three.js panel builders + HTML template

**Files:**
- Modify: `vis/_vis_common.py`

- [ ] **Step 1: Append the template module**

Append to `vis/_vis_common.py`:

```python
# =============================================================================
# HTML / Three.js rendering
# =============================================================================

def make_voxel_panel_data(
    coords: torch.Tensor,
    feats_norm: torch.Tensor,
    res: int,
    threshold: float,
    max_points: int,
) -> dict:
    mag = (feats_norm * 0.5 + 0.5).norm(dim=-1)
    idx = torch.where(mag > threshold)[0]
    if idx.shape[0] > max_points:
        perm = torch.randperm(idx.shape[0])[:max_points]
        idx = idx[perm]
    c = coords[idx].long().numpy()
    rgb = emission_to_srgb_u8(feats_norm[idx]).numpy()
    return {
        "xyz": c.flatten().tolist(),
        "rgb": rgb.flatten().tolist(),
        "n": int(idx.shape[0]),
        "res": int(res),
    }


def make_glb_panel_data(glb_path: Optional[str], z_up_to_y_up: bool) -> dict:
    # We serve the GLB as a file:// URL relative to the HTML output path; the
    # entry scripts resolve this to an absolute file URL at write time.
    return {"url": glb_path, "z_up_to_y_up": bool(z_up_to_y_up)}


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0a; color:#e0e0e0; font-family:'Segoe UI',Arial,sans-serif; }}
.header {{ padding:10px 18px; background:#141414; border-bottom:1px solid #2a2a2a; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
.header h2 {{ font-size:14px; font-weight:600; }}
.header select {{ background:#1e1e1e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:4px 8px; font-size:13px; }}
.row {{ display:flex; width:100%; height: calc(100vh - 44px); }}
.cell {{ flex:1; display:flex; flex-direction:column; border-right:1px solid #222; min-width:0; }}
.cell:last-child {{ border-right:none; }}
.cap {{ padding:6px 10px; background:#111; border-bottom:1px solid #222; font-size:12px; color:#8cf; min-height: 26px; }}
.label {{ position:absolute; top:6px; left:10px; background:rgba(0,0,0,0.65); padding:2px 8px; border-radius:3px; font-size:11px; color:#ccc; pointer-events:none; z-index:10; }}
.pane {{ flex:1; position:relative; }}
.pane canvas {{ display:block; width:100%; height:100%; }}
.missing {{ display:flex; align-items:center; justify-content:center; color:#666; font-size:13px; width:100%; height:100%; }}
</style></head><body>
<div class="header">
  <h2>{title}</h2>
  <label for="sample-select">Sample:</label>
  <select id="sample-select">{sample_options}</select>
  <span id="sample-info" style="color:#888; font-size:12px;"></span>
</div>
<div class="row" id="row"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script>
(function() {{
  var DATA = {data_json};
  var POINT_SIZE = {point_size};
  var row = document.getElementById('row');
  var sel = document.getElementById('sample-select');
  var info = document.getElementById('sample-info');

  var panels = [];  // per-cell: {{renderer, scene, camera, controls, content, kind, label}}
  function buildCells(panelSpec) {{
    row.innerHTML = '';
    panels = [];
    panelSpec.forEach(function(p, i) {{
      var cell = document.createElement('div');
      cell.className = 'cell';
      var cap = document.createElement('div');
      cap.className = 'cap';
      cap.id = 'cap-' + i;
      cap.textContent = p.caption || '';
      var pane = document.createElement('div');
      pane.className = 'pane';
      var lab = document.createElement('div');
      lab.className = 'label';
      lab.textContent = p.label || '';
      var cvs = document.createElement('canvas');
      cvs.id = 'cvs-' + i;
      pane.appendChild(lab);
      pane.appendChild(cvs);
      cell.appendChild(cap);
      cell.appendChild(pane);
      row.appendChild(cell);
      var renderer = new THREE.WebGLRenderer({{ canvas: cvs, antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setClearColor(0x0a0a0a);
      var scene = new THREE.Scene();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.0));
      var camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
      camera.position.set(1.2, 1.0, 1.2);
      var controls = new THREE.OrbitControls(camera, cvs);
      controls.enableDamping = true;
      controls.dampingFactor = 0.12;
      var rec = {{ renderer: renderer, scene: scene, camera: camera, controls: controls,
                  content: null, kind: p.kind, label: p.label, pane: pane }};
      panels.push(rec);
      function resize() {{
        var w = pane.clientWidth, h = pane.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }}
      resize();
      window.addEventListener('resize', resize);
      new ResizeObserver(resize).observe(pane);
      (function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }})();
    }});
    // Cross-canvas camera broadcast.
    var syncing = false;
    panels.forEach(function(src) {{
      src.controls.addEventListener('change', function() {{
        if (syncing) return;
        syncing = true;
        panels.forEach(function(dst) {{
          if (dst === src) return;
          dst.camera.position.copy(src.camera.position);
          dst.camera.quaternion.copy(src.camera.quaternion);
          dst.controls.target.copy(src.controls.target);
          dst.controls.update();
        }});
        syncing = false;
      }});
    }});
  }}

  function buildPointCloud(d) {{
    var n = d.n, xyz = d.xyz, rgb = d.rgb, res = d.res;
    var inv = 1.0 / res;
    var half = 0.5;
    var offset = half * inv;  // voxel-cell-center offset
    var pos = new Float32Array(n * 3);
    var col = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) {{
      // voxel_i / res - 0.5 + 0.5/res, then Y/Z swap to Three.js Y-up.
      var x = xyz[i*3]   * inv - half + offset;
      var y = xyz[i*3+1] * inv - half + offset;
      var z = xyz[i*3+2] * inv - half + offset;
      pos[i*3]   = x;
      pos[i*3+1] = z;
      pos[i*3+2] = y;
      col[i*3]   = rgb[i*3]   / 255;
      col[i*3+1] = rgb[i*3+1] / 255;
      col[i*3+2] = rgb[i*3+2] / 255;
    }}
    var geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('color',    new THREE.BufferAttribute(col, 3));
    var mat = new THREE.PointsMaterial({{ size: POINT_SIZE, vertexColors: true, sizeAttenuation: true }});
    return new THREE.Points(geom, mat);
  }}

  // Normalise a GLB mesh to [-0.5, 0.5]^3 using the same bbox/scale rule as
  // data_toolkit/voxelize_pbr.py. Optionally apply Blender-Z-up → Y-up first.
  function normaliseGlb(root, zUpToYUp) {{
    if (zUpToYUp) {{
      // +90° around X: (x, y, z) → (x, z, -y). Applied to each mesh vertex.
      root.traverse(function(o) {{
        if (o.isMesh && o.geometry && o.geometry.attributes.position) {{
          var p = o.geometry.attributes.position.array;
          for (var i = 0; i < p.length; i += 3) {{
            var y = p[i+1], z = p[i+2];
            p[i+1] = z;
            p[i+2] = -y;
          }}
          o.geometry.attributes.position.needsUpdate = true;
          o.geometry.computeBoundingBox();
        }}
      }});
    }}
    var box = new THREE.Box3().setFromObject(root);
    var mn = box.min, mx = box.max;
    var cx = (mn.x + mx.x) * 0.5, cy = (mn.y + mx.y) * 0.5, cz = (mn.z + mx.z) * 0.5;
    var s = 0.99999 / Math.max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z);
    root.traverse(function(o) {{
      if (o.isMesh && o.geometry && o.geometry.attributes.position) {{
        var p = o.geometry.attributes.position.array;
        for (var i = 0; i < p.length; i += 3) {{
          p[i]   = (p[i]   - cx) * s;
          p[i+1] = (p[i+1] - cy) * s;
          p[i+2] = (p[i+2] - cz) * s;
        }}
        o.geometry.attributes.position.needsUpdate = true;
        o.geometry.computeBoundingBox();
      }}
    }});
    // Voxel pipeline writes Y-up coords then we apply Y/Z swap at upload. For
    // the GLB to land in the same viewer-space, apply the same swap on the
    // mesh root after normalisation.
    root.rotation.set(Math.PI / 2, 0, 0);
  }}

  function loadPanel(pIdx, d) {{
    var rec = panels[pIdx];
    if (rec.content) {{
      rec.scene.remove(rec.content);
      if (rec.content.geometry) rec.content.geometry.dispose();
      if (rec.content.material) rec.content.material.dispose();
      rec.content = null;
    }}
    if (rec.kind === 'voxel') {{
      rec.content = buildPointCloud(d);
      rec.scene.add(rec.content);
    }} else if (rec.kind === 'glb') {{
      if (!d.url) {{
        var miss = document.createElement('div');
        miss.className = 'missing';
        miss.textContent = 'GLB unavailable';
        rec.pane.appendChild(miss);
        rec.content = null;
        return;
      }}
      new THREE.GLTFLoader().load(d.url, function(gltf) {{
        normaliseGlb(gltf.scene, d.z_up_to_y_up);
        rec.content = gltf.scene;
        rec.scene.add(rec.content);
      }});
    }}
  }}

  function loadSample(sha) {{
    var s = DATA.samples[sha];
    buildCells(s.panels);
    s.panels.forEach(function(p, i) {{
      loadPanel(i, p.data);
      var cap = document.getElementById('cap-' + i);
      if (cap) cap.textContent = p.caption || '';
    }});
    info.textContent = s.info || '';
  }}

  sel.addEventListener('change', function() {{ loadSample(sel.value); }});
  loadSample(sel.value);
}})();
</script>
</body></html>"""


def render_html(
    samples: list[dict],
    title: str,
    out_path: str,
    point_size: float = 0.008,
) -> None:
    """Write the complete HTML file.

    ``samples`` is a list of per-sample dicts with the structure:
      {"sha": str, "info": str, "panels": [{"kind": str, "data": dict,
                                             "caption": str, "label": str}, ...]}
    """
    data_blob = json.dumps(
        {"samples": {s["sha"]: s for s in samples}},
        separators=(",", ":"),
    )
    sample_options = "".join(
        f'<option value="{s["sha"]}">[{i}] {s["sha"][:14]}</option>'
        for i, s in enumerate(samples)
    )
    html = _HTML_TEMPLATE.format(
        title=title,
        sample_options=sample_options,
        data_json=data_blob,
        point_size=point_size,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path}  ({len(html) / 1e6:.1f} MB)")
```

- [ ] **Step 2: Commit**

```bash
git add vis/_vis_common.py
git commit -m "vis: Three.js panel builders + render_html template"
```

---

## Task 7: `vis/vis_3d_vae.py` entry script

**Files:**
- Create: `vis/vis_3d_vae.py`

- [ ] **Step 1: Write the entry script**

Write `vis/vis_3d_vae.py`:

```python
#!/usr/bin/env python3
"""Unified 3D interactive visualization for emission-VAE reconstruction.

Renders GLB + GT emission + N experiment predictions per sample. All panels
share a Three.js camera. See docs/superpowers/specs/2026-04-22-vis-3d-consolidation-design.md.

Usage:
  python vis/vis_3d_vae.py \\
      --dataset data/lightgen_test/df_10samples.parquet \\
      --samples all \\
      --glb_root /localhome/dya78/code/lightgen/data/TexVerse-1K \\
      --experiment overfit=outputs/emission_vae_10sample_overfit \\
      --out visualizations/vae_overfit.html
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import o_voxel  # noqa: E402
import torch    # noqa: E402

from vis._vis_common import (  # noqa: E402
    ExperimentHandle, compute_emission_l1, load_samples,
    load_vae_from_handle, make_glb_panel_data, make_voxel_panel_data,
    read_emission_voxel, render_html, resolve_experiment,
)


def _load_pbr_feats(path: str) -> torch.Tensor | None:
    if not os.path.exists(path):
        return None
    attrs = ["base_color", "metallic", "roughness", "alpha"]
    _, pbr_attr = o_voxel.io.read_vxz(path, num_threads=4)
    return torch.cat([pbr_attr[a].float() / 255.0 * 2.0 - 1.0 for a in attrs], dim=-1)


def _parse_experiments(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--experiment expects name=path, got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="parquet or splits JSON path")
    ap.add_argument("--samples", default="all",
                    help="all | first:N | split:<name> | sha:<csv>")
    ap.add_argument("--glb_root", default=None,
                    help="root for parquet's glb_1k_path; omit → GLB panel missing")
    ap.add_argument("--experiment", action="append", default=[],
                    help="repeatable name=run_dir_or_ckpt")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument("--threshold", type=float, default=0.005,
                    help="min emission magnitude to render")
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--point_size", type=float, default=0.012)
    args = ap.parse_args()

    experiments: list[ExperimentHandle] = [
        resolve_experiment(name, path) for name, path in _parse_experiments(args.experiment)
    ]
    # When there are experiments, resolution/data_root come from the first one's
    # config. When there are none (data-only mode), infer from the parquet/default.
    if experiments:
        data_root = experiments[0].data_root
        resolution = experiments[0].resolution
    else:
        data_root = os.path.dirname(os.path.abspath(args.dataset))
        resolution = 256

    records = load_samples(
        data_root=data_root,
        resolution=resolution,
        source=args.dataset,
        select=args.samples,
        glb_root=args.glb_root,
        require=("emission_voxel_path",),
    )
    if not records:
        raise SystemExit(f"no samples matched --samples={args.samples!r}")

    # Load VAEs once; reuse across samples.
    decoders = [(h, load_vae_from_handle(h)) for h in experiments]

    out_abs = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_abs)

    samples_out: list[dict] = []
    for rec in records:
        coords, emit, res = read_emission_voxel(rec.emission_voxel_path)
        pbr = _load_pbr_feats(rec.pbr_voxel_path) if any(
            h.trainer_name == "EmissionVaeTrainer" for h in experiments
        ) else None

        panels: list[dict] = []
        # GLB panel (or placeholder)
        glb_rel_url = os.path.relpath(rec.glb_path, out_dir) if rec.glb_path else None
        panels.append({
            "kind": "glb",
            "data": make_glb_panel_data(glb_rel_url, z_up_to_y_up=False),
            "caption": "",
            "label": "GLB",
        })
        # GT
        panels.append({
            "kind": "voxel",
            "data": make_voxel_panel_data(coords, emit, res, args.threshold, args.max_points),
            "caption": f"{coords.shape[0]:,} voxels",
            "label": "GT emission",
        })
        # Experiments
        for h, decode in decoders:
            pred = decode(coords, emit, pbr)
            l1 = compute_emission_l1(coords, pred, coords, emit)
            panels.append({
                "kind": "voxel",
                "data": make_voxel_panel_data(coords, pred, res, args.threshold, args.max_points),
                "caption": f"emission L1: {l1:.4f}",
                "label": h.label,
            })
            torch.cuda.empty_cache()
        samples_out.append({
            "sha": rec.sha,
            "info": f"{coords.shape[0]:,} voxels",
            "panels": panels,
        })

    render_html(
        samples=samples_out,
        title=f"Emission VAE 3D — {len(experiments)} experiment(s)",
        out_path=args.out,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax with a dry import**

Run: `cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning && conda run -n trellis2 python -c "import importlib.util, os; p='/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/vis_3d_vae.py'; spec=importlib.util.spec_from_file_location('v', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"`

Expected: `ok` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add vis/vis_3d_vae.py
git commit -m "vis: vis_3d_vae.py entry script"
```

---

## Task 8: Smoke-test `vis_3d_vae.py` end-to-end and have Codex review the result

**Files:**
- Generate: `visualizations/_smoke_vae.html` (gitignored by `visualizations/` rule)

- [ ] **Step 1: Run the script on the 10-sample overfit run**

Run:

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
mkdir -p log
TS=$(date +%Y%m%d-%H%M%S)
conda run -n trellis2 python vis/vis_3d_vae.py \
    --dataset data/lightgen_test/df_10samples.parquet \
    --samples first:3 \
    --experiment overfit=outputs/emission_vae_10sample_overfit \
    --out visualizations/_smoke_vae.html 2>&1 | tee "log/${TS}-smoke_vae.log"
```

Expected: final line prints `Saved: visualizations/_smoke_vae.html  (<MB size>)`. No stack trace.

- [ ] **Step 2: Quick file-level sanity checks**

Run: `wc -l visualizations/_smoke_vae.html && grep -c 'OrbitControls' visualizations/_smoke_vae.html && grep -c '"n":' visualizations/_smoke_vae.html`

Expected: a large line count (hundreds of KB → a single long line is fine), `1` for OrbitControls (single CDN include), and `>= 6` for `"n":` (3 samples × [GT + prediction] ≥ 6 voxel-panel entries).

- [ ] **Step 3: Ask Codex to review the generated helper module and the entry script**

Use the MCP Codex tool with this prompt (paste via your normal agent invocation):

> Review the following files for correctness, focusing on: (a) the voxel-cell centre-offset math in `buildPointCloud` matches the `i / res - 0.5 + 0.5 / res` rule from the design spec; (b) the GLB normalization in `normaliseGlb` correctly re-applies the voxelization transform from `data_toolkit/voxelize_pbr.py:77–86`; (c) `load_vae_from_handle` correctly reproduces the encode/decode path of `trellis2/trainers/vae/emission_vae.py` and `emission_pbr_finetune.py`; (d) `_pick_checkpoint` follows the "last.ckpt > min val/loss > lexicographic-last" rule.
> Files to review:
> - `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/_vis_common.py`
> - `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/vis_3d_vae.py`
> Reference files to cross-check against:
> - `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/data_toolkit/voxelize_pbr.py`
> - `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/trellis2/trainers/vae/emission_vae.py`
> - `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/trellis2/trainers/vae/emission_pbr_finetune.py`
> Report only real bugs or inconsistencies. Under 500 words.

Inspect Codex's findings; validate each one by reading the cited code. For each confirmed bug, add a follow-up fix step here and address it before proceeding. For each rejected finding, briefly note why the code is correct. Do not accept or reject without reading the referenced lines.

- [ ] **Step 4: Open the generated HTML locally and verify visual alignment**

Run: `realpath visualizations/_smoke_vae.html`

Then open that path in a browser (or ask the user to). Checks:
- The GLB panel renders a recognisable mesh (not empty, not a sliver).
- The GT-emission and prediction voxels line up visually with the GLB shape (same orientation, same approximate bbox).
- Rotating any panel rotates the other two in sync.
- Switching the sample dropdown swaps all panels and preserves the camera pose.

Record the outcome in the commit message. If the GLB is misaligned with the voxels, the most likely cause is `z_up_to_y_up=False` should be `True` for this dataset — toggle the flag in `vis/vis_3d_vae.py` and re-run step 1.

- [ ] **Step 5: Commit** (even if you kept `z_up_to_y_up=False`, commit the verified state)

```bash
git add vis/ log/ || true
git commit -m "vis: smoke-test vis_3d_vae.py on 10-sample overfit run" --allow-empty
```

---

## Task 9: `vis/vis_3d_dit.py` entry script

**Files:**
- Create: `vis/vis_3d_dit.py`

- [ ] **Step 1: Write the entry script**

Write `vis/vis_3d_dit.py`:

```python
#!/usr/bin/env python3
"""Unified 3D interactive visualization for DiT → VAE-decode emission.

Usage:
  python vis/vis_3d_dit.py \\
      --dataset data/lightgen_test/df_10samples.parquet \\
      --samples all \\
      --glb_root /localhome/dya78/code/lightgen/data/TexVerse-1K \\
      --experiment overfit=outputs/emission_dit_10sample_overfit:outputs/emission_vae_10sample_overfit \\
      --out visualizations/dit_overfit.html
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np   # noqa: E402
import torch         # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

import o_voxel  # noqa: E402

from vis._vis_common import (  # noqa: E402
    compute_emission_l1, load_samples, make_glb_panel_data,
    make_voxel_panel_data, read_emission_voxel, render_html, resolve_experiment,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_latent(path: str):
    from trellis2.modules import sparse as sp
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))


def _parse_experiments(specs: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for s in specs:
        if "=" not in s or ":" not in s.split("=", 1)[1]:
            raise ValueError(f"--experiment expects name=dit_path:vae_path, got {s!r}")
        name, pair = s.split("=", 1)
        dit_path, vae_path = pair.split(":", 1)
        out.append((name.strip(), dit_path.strip(), vae_path.strip()))
    return out


def _build_dit(dit_handle, sigma_min: float):
    from trellis2.trainers.builders import build_models
    models = build_models(dit_handle.config["models"])
    dit = models["denoiser"]
    state = torch.load(dit_handle.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    dit.load_state_dict({
        k.removeprefix("models.denoiser."): v for k, v in state.items()
        if k.startswith("models.denoiser.")
    })
    dit.to(DEVICE).eval()
    from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
    return dit, FlowEulerCfgSampler(sigma_min=sigma_min)


def _build_dino(dino_args):
    from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
    dino = DinoV3FeatureExtractor(**dino_args)
    dino.cuda()
    return dino


def _build_vae_decoder_from_latent(vae_handle):
    """Build the VAE decoder alone and return a callable
    ``decode_from_latent(z_dit_raw, z_pbr_raw) -> [N, 3] cpu tensor`` that
    branches on the VAE trainer family. Loaded once per experiment.
    """
    from trellis2.trainers.builders import build_models
    models = build_models(vae_handle.config["models"])
    dec = models["decoder"]
    state = torch.load(vae_handle.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    dec.load_state_dict({
        k.removeprefix("models.decoder."): v for k, v in state.items()
        if k.startswith("models.decoder.")
    })
    dec.to(DEVICE).eval()

    trainer = vae_handle.trainer_name
    if trainer == "EmissionVaeTrainer":
        @torch.no_grad()
        def decode_from_latent(z_dit_raw, z_pbr_raw):
            z_joint = z_dit_raw.replace(torch.cat([z_dit_raw.feats, z_pbr_raw.feats.to(DEVICE)], dim=-1))
            return dec(z_joint).to("cpu").feats
    elif trainer == "EmissionPbrFinetuneTrainer":
        @torch.no_grad()
        def decode_from_latent(z_dit_raw, z_pbr_raw):
            return dec(z_dit_raw).to("cpu").feats[:, :3]
    else:
        raise RuntimeError(f"unsupported VAE trainer for DiT decode: {trainer!r}")
    return decode_from_latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--samples", default="all")
    ap.add_argument("--glb_root", default=None)
    ap.add_argument("--experiment", action="append", required=True,
                    help="repeatable name=dit_path:vae_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--point_size", type=float, default=0.012)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance_strength", type=float, default=3.0)
    args = ap.parse_args()

    specs = _parse_experiments(args.experiment)
    dit_handles = [(n, resolve_experiment(n + "-dit", dp), resolve_experiment(n + "-vae", vp))
                   for n, dp, vp in specs]

    # Dataset metadata comes from the first DiT's config (all experiments must
    # target the same dataset for a sensible HTML).
    dit_cfg = dit_handles[0][1].config
    data_root = dit_cfg["dataset"]["args"]["data_root"]
    resolution = int(dit_cfg["dataset"]["args"].get("resolution", 256))

    records = load_samples(
        data_root=data_root,
        resolution=resolution,
        source=args.dataset,
        select=args.samples,
        glb_root=args.glb_root,
        require=("emission_voxel_path",),
    )
    if not records:
        raise SystemExit(f"no samples matched --samples={args.samples!r}")

    # Per-experiment: build DiT + VAE latent-decoder + sampler + DINO once.
    built = []
    for name, dit_h, vae_h in dit_handles:
        dit, sampler = _build_dit(dit_h, sigma_min=dit_h.config["trainer"]["args"]["sigma_min"])
        decode_from_latent = _build_vae_decoder_from_latent(vae_h)
        dino_args = dit_h.config["trainer"]["args"]["image_cond_model"]["args"]
        dino = _build_dino(dino_args)
        built.append({
            "name": name, "dit": dit, "sampler": sampler,
            "decode_from_latent": decode_from_latent,
            "dino": dino, "dit_cfg": dit_h.config,
        })

    dino_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    out_abs = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_abs)

    samples_out = []
    for rec in records:
        coords, gt_emit, res = read_emission_voxel(rec.emission_voxel_path)

        panels = []
        glb_rel_url = os.path.relpath(rec.glb_path, out_dir) if rec.glb_path else None
        panels.append({"kind": "glb", "data": make_glb_panel_data(glb_rel_url, False),
                       "caption": "", "label": "GLB"})
        panels.append({"kind": "voxel",
                       "data": make_voxel_panel_data(coords, gt_emit, res, args.threshold, args.max_points),
                       "caption": f"{coords.shape[0]:,} voxels", "label": "GT emission"})

        for exp in built:
            ds = exp["dit_cfg"]["dataset"]["args"]
            emit_mean = torch.tensor(ds["emission_normalization"]["mean"])
            emit_std  = torch.tensor(ds["emission_normalization"]["std"])
            pbr_mean  = torch.tensor(ds["pbr_normalization"]["mean"])
            pbr_std   = torch.tensor(ds["pbr_normalization"]["std"])
            shape_mean = torch.tensor(ds["shape_normalization"]["mean"])
            shape_std  = torch.tensor(ds["shape_normalization"]["std"])

            emit_latent_name = ds["emission_latent_name"]
            pbr_latent_name = ds["pbr_latent_name"]
            shape_latent_name = ds["shape_latent_name"]

            z_emit_raw  = _load_latent(f"{data_root}/emission_latents/{emit_latent_name}/{rec.sha}.npz")
            z_pbr_raw   = _load_latent(f"{data_root}/pbr_latents/{pbr_latent_name}/{rec.sha}.npz")
            z_shape_raw = _load_latent(f"{data_root}/shape_latents/{shape_latent_name}/{rec.sha}.npz")

            def _norm(st, mu, sd): return st.replace((st.feats - mu.to(st.feats.device)) / sd.to(st.feats.device))
            def _denorm(st, mu, sd): return st.replace(st.feats * sd.to(st.feats.device) + mu.to(st.feats.device))

            z_emit_n  = _norm(z_emit_raw,  emit_mean,  emit_std).to(DEVICE)
            z_pbr_n   = _norm(z_pbr_raw,   pbr_mean,   pbr_std).to(DEVICE)
            z_shape_n = _norm(z_shape_raw, shape_mean, shape_std).to(DEVICE)
            concat_cond = z_emit_n.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))

            if not rec.thumbnail_path or not os.path.exists(rec.thumbnail_path):
                thumb = os.path.join(data_root, "thumbnails", f"{rec.sha}.png")
            else:
                thumb = rec.thumbnail_path
            img = Image.open(thumb).convert("RGB")
            sz = exp["dino"].image_size if hasattr(exp["dino"], "image_size") else \
                 exp["dit_cfg"]["trainer"]["args"]["image_cond_model"]["args"]["image_size"]
            arr = np.array(img.resize((sz, sz), Image.LANCZOS)).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            t = dino_tr(t).to(exp["dino"].model.embeddings.patch_embeddings.weight.dtype)
            img_feat = exp["dino"].extract_features(t)
            neg_cond = torch.zeros_like(img_feat)

            noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                res_s = exp["sampler"].sample(
                    exp["dit"], noise, cond=img_feat, neg_cond=neg_cond,
                    concat_cond=concat_cond, steps=args.steps,
                    guidance_strength=args.guidance_strength, verbose=False,
                )
            z_dit_n = res_s.samples
            z_dit_raw = z_dit_n.replace(z_dit_n.feats * emit_std.to(DEVICE) + emit_mean.to(DEVICE))

            pred = exp["decode_from_latent"](z_dit_raw, z_pbr_raw)
            l1 = compute_emission_l1(coords, pred, coords, gt_emit)
            panels.append({
                "kind": "voxel",
                "data": make_voxel_panel_data(coords, pred, res, args.threshold, args.max_points),
                "caption": f"emission L1: {l1:.4f}",
                "label": exp["name"],
            })
            torch.cuda.empty_cache()

        samples_out.append({"sha": rec.sha, "info": f"{coords.shape[0]:,} voxels", "panels": panels})

    render_html(
        samples=samples_out,
        title=f"Emission DiT 3D — {len(built)} experiment(s)",
        out_path=args.out,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
```

Note: the decode branch in the DiT main loop intentionally does not call `load_vae_from_handle` — that helper expects voxel-space features as input, but DiT gives us a latent. The explicit inline decode is the right shape for the DiT pipeline.

- [ ] **Step 2: Verify syntax**

Run: `cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning && conda run -n trellis2 python -c "import importlib.util; p='/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/vis_3d_dit.py'; spec=importlib.util.spec_from_file_location('v', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"`

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add vis/vis_3d_dit.py
git commit -m "vis: vis_3d_dit.py entry script (DiT sample → VAE decode)"
```

---

## Task 10: Smoke-test `vis_3d_dit.py` end-to-end and have Codex review

**Files:**
- Generate: `visualizations/_smoke_dit.html`

- [ ] **Step 1: Run the script on the 10-sample overfit DiT + its matching VAE**

Run:

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
TS=$(date +%Y%m%d-%H%M%S)
conda run -n trellis2 python vis/vis_3d_dit.py \
    --dataset data/lightgen_test/df_10samples.parquet \
    --samples first:3 \
    --experiment overfit=outputs/emission_dit_10sample_overfit:outputs/emission_vae_10sample_overfit \
    --out visualizations/_smoke_dit.html 2>&1 | tee "log/${TS}-smoke_dit.log"
```

Expected: `Saved: visualizations/_smoke_dit.html`. No stack trace.

- [ ] **Step 2: Sanity check**

Run: `wc -l visualizations/_smoke_dit.html && grep -c '"n":' visualizations/_smoke_dit.html`

Expected: HTML written, `>= 9` panel-data entries (3 samples × [GT + prediction] × some).

- [ ] **Step 3: Codex review of the DiT pipeline**

Ask Codex with this prompt:

> Review `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/vis_3d_dit.py` for correctness.
>  Focus areas:
>   (a) Does the latent-normalization flow (normalize → DiT sample → denormalize → VAE decode) match the reference `/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/vis/legacy/vis_3d_emission_dit.py` ?
>   (b) Is the DiT decode correctly branching on the VAE's `trainer_name` (concat vs channel-hijack) — cross-check against `vis/legacy/vis_3d_emission_dit.py` and `vis/legacy/vis_3d_emission_dit_pbrvae.py` ?
>   (c) Do the latent-name lookups (`emission_latent_name`, `pbr_latent_name`, `shape_latent_name`) match the DiT config schema under `configs/gen/emission_dit_10sample_overfit.yaml` ?
> Report only real bugs. Under 500 words.

Note: for (a) and (b) the reference files will be named `vis/legacy/vis_3d_emission_dit*.py` after Task 11 moves them. If you run Codex before Task 11, they still live at `vis/vis_3d_emission_dit*.py`. Update the paths in the prompt accordingly.

Apply any confirmed fixes with new edit steps; reject false positives with a noted reason.

- [ ] **Step 4: Open the HTML and verify**

Same check as Task 8 Step 4, but now with three voxel panels per sample (GLB + GT + DiT-decoded prediction). The DiT prediction should look similar to the VAE reconstruction in the 10-sample overfit case (both are trained on the same 10 samples).

- [ ] **Step 5: Commit**

```bash
git add vis/ log/ || true
git commit -m "vis: smoke-test vis_3d_dit.py on 10-sample overfit DiT + VAE" --allow-empty
```

---

## Task 11: Move legacy scripts and update CLAUDE.md

**Files:**
- Move: `vis/vis_3d_*.py` → `vis/legacy/` (eight files)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Move the legacy scripts**

Run:

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
git mv vis/vis_3d_emission_vae.py             vis/legacy/
git mv vis/vis_3d_emission_dit.py             vis/legacy/
git mv vis/vis_3d_emission_dit_pbrvae.py      vis/legacy/
git mv vis/vis_3d_overfit.py                  vis/legacy/
git mv vis/vis_3d_pbr_finetune.py             vis/legacy/
git mv vis/vis_3d_pbr_vae_on_emission.py      vis/legacy/
git mv vis/vis_3d_pbr_vae_val.py              vis/legacy/
git mv vis/vis_voxel_vs_glb.py                vis/legacy/
```

Verify: `ls vis/` should now show `_vis_common.py`, `_vis_common_test.py`, `vis_3d_vae.py`, `vis_3d_dit.py`, `legacy/`, plus the 2D-PNG scripts (`compare_*.py`, `vis_emission_vae*.py`, `vis_vae_ckpt_comparison.py`).

- [ ] **Step 2: Update CLAUDE.md "Data Paths" section**

Find the section `## Data Paths` (around line 72 in the current `CLAUDE.md`). Edit the visualization scripts bullet:

Old:
```
- **Visualization scripts:** `vis/` — all visualization scripts live here. Use these as templates for new visualizations.
```

New:
```
- **Visualization scripts:** `vis/` — unified 3D interactive: `vis/vis_3d_vae.py` (VAE recon) and `vis/vis_3d_dit.py` (DiT → VAE decode). Both share helpers in `vis/_vis_common.py`. Legacy per-case scripts live in `vis/legacy/` for reference.
```

Edit via `Edit` tool using the exact old/new strings above.

- [ ] **Step 3: Commit**

```bash
git add vis/ CLAUDE.md
git commit -m "vis: move per-case scripts to vis/legacy/; CLAUDE.md points at unified pair"
```

- [ ] **Step 4: Final verification**

Run: `ls vis/ vis/legacy/ && echo '---' && git log --oneline -12`

Expected: `vis/` contains the unified pair + 2D scripts; `vis/legacy/` contains the eight moved scripts; git log shows ~11 clean task commits in order.

---

## Spec coverage self-review

| Spec requirement | Task |
|---|---|
| Shared `_vis_common.py` with sample loader, voxel reader, frame computation, Three.js builders, HTML template, trainer-type resolver | Tasks 1–6 |
| `load_samples` covers parquet + splits JSON + sha list with `(data_root, resolution, sha)` path resolution | Task 2 |
| `read_emission_voxel` + sRGB gamma | Task 3 |
| `compute_emission_l1` caption metric | Task 3 |
| `resolve_experiment` with `last.ckpt > min val/loss > error` rule | Task 4 |
| `load_vae_from_handle` dispatches on `EmissionVaeTrainer` vs `EmissionPbrFinetuneTrainer` | Task 5 |
| Three.js `buildPointCloud` with voxel-centre offset `i/res - 0.5 + 0.5/res` | Task 6 |
| GLB re-normalization (same formula as voxelize_pbr.py) | Task 6 |
| Per-canvas `OrbitControls` with change-event broadcast (N panels, not just 2) | Task 6 |
| `emission L1` caption above each prediction panel | Task 6 + Tasks 7, 9 |
| GLB renders full PBR (just loads the GLB; no base-colour stripping) | Task 6 |
| Sample `<select>` preserves camera pose across switches | Task 6 |
| `vis_3d_vae.py` CLI: `--dataset --samples --experiment --out` | Task 7 |
| Zero-experiments data-only mode (`GLB | GT`) | Task 7 (experiments list empty → no extra panels) |
| `vis_3d_dit.py` CLI with required `name=dit:vae` | Task 9 |
| DiT VAE decode branches on VAE trainer family | Task 9 |
| Smoke test reproduces at least one legacy script's output | Tasks 8, 10 |
| Legacy scripts moved to `vis/legacy/` | Task 11 |
| `CLAUDE.md` updated | Task 11 |
| Codex cross-check of the implementation | Tasks 8, 10 |

No spec requirements are missing a task.
