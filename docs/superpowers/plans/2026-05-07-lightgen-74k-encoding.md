# LightGen 74k Encoding Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize PBR voxels, emission voxels, dual-grids, PBR latents and shape latents for all 73,472 samples of the 74k emissive subset, with outputs tar-bundled per shard on `/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/`.

**Architecture:** Eight shards (rank 0..7 of world_size=8). Each shard = two sbatch jobs chained via `--dependency=afterok`: (Job N-AB) Blender PBR dump → voxelize_pbr → dual_grid on a CPU-heavy GPU node, tar-and-upload `voxels/partN.tar`; (Job N-C) pull voxels tar, encode PBR latent + shape latent on a GPU, tar-and-upload `latents/partN.tar`. Blender is self-installed under `/localscratch/dya78/` per node. Dual-grid is derived from the same Blender pickle as the PBR/emission voxels — never from the GLB via trimesh — to preserve the `assert torch.equal(x_0.coords, shape_z.coords)` invariant.

**Tech Stack:** Python 3.10 (conda env `trellis2`); PyTorch 2.6.0 + CUDA 12.4; `o_voxel` (C++ ext); Blender 4.5.1 (self-installed); Slurm (star2 cluster); zsh in sbatch bodies, bash for local launchers; rsync for tar transport. No new test framework — verifications run scripts with `--limit N` against the existing local 1k tree.

**Spec:** `docs/superpowers/specs/2026-05-07-lightgen-74k-encoding-design.md`.

**Before starting:**

* Confirm clean working tree on branch `lightgen`. The plan touches only new files, but don't get tangled with the existing untracked artifacts (`data_toolkit/regenerate_1k_dual_grid.py`, etc.).
* Plan runs locally for Phases 1–5 (no cluster needed yet). Phase 6 onwards uses star2 via `ssh star2 sbatch`.
* Per [feedback memory](feedback_no_cluster_edits.md), never edit code on the cluster; edit locally → push → cluster pulls in sbatch body.
* Per [feedback memory](feedback_tee_logs.md), tee every non-trivial command into `log/${TS}-<task>.log`.

---

## Phase 1 — Foundation: dataset module + bootstrap

### Task 1: Add the `lightgen_74k` dataset module

**Files:**
- Create: `data_toolkit/datasets/lightgen_74k.py`

The in-repo `voxelize_pbr.py` discovers its dataset module via `importlib.import_module(f'datasets.{sys.argv[1]}')`. We need a minimal module so we can call `python data_toolkit/voxelize_pbr.py lightgen_74k --root ... `.

- [ ] **Step 1: Write the module**

```python
"""Minimal dataset module for the 74k emissive subset.

Used by data_toolkit/voxelize_pbr.py (and similar scripts) which dynamically
import `datasets.<name>` from sys.argv[1]. The 74k subset's GLBs already live
on /3dlg-falas, so download/metadata-fetch helpers from `common` are not
needed here — only `foreach_instance` is exposed.
"""
from __future__ import annotations
import argparse

from .common import foreach_instance


def add_args(parser: argparse.ArgumentParser):
    """No dataset-specific CLI flags; the 74k metadata.csv is already on disk."""
    pass


__all__ = ["add_args", "foreach_instance"]
```

- [ ] **Step 2: Verify import path resolves**

Run from repo root:
```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
python -c "import sys; sys.path.insert(0, 'data_toolkit'); import datasets.lightgen_74k as m; m.add_args(__import__('argparse').ArgumentParser()); print('OK', m.foreach_instance)"
```
Expected: `OK <function foreach_instance at 0x...>`.

- [ ] **Step 3: Commit**

```bash
git add data_toolkit/datasets/lightgen_74k.py
git commit -m "data_toolkit: add lightgen_74k dataset module (foreach_instance shim)"
```

---

### Task 2: Write the bootstrap script

**Files:**
- Create: `data_toolkit/prep_74k_bootstrap.py`

Reads the 854k parquet, applies `success==True`, indexes via the 74k pinned split, joins to grab `glb_1k_path`, writes `metadata.csv`, copies parquet + split JSON to jupyter.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""One-shot bootstrap for the 74k encoding pipeline.

Reads:
  /local-scratch/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet
  /local-scratch/localhome/dya78/code/lightgen/data_processing/annotation/data_splits_emissive_74k_pinned.json

Writes (rsync to jupyter):
  /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/metadata.csv
  /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/df_SomgProc_final.parquet
  /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/data_splits_emissive_74k_pinned.json

The split JSON's positional indices reference the success-filtered view of the
parquet (see lightgen/data_processing/create_splits_74k_pinned.py for how the
indices were generated).

Usage:
    python data_toolkit/prep_74k_bootstrap.py            # full run
    python data_toolkit/prep_74k_bootstrap.py --dry-run  # print counts only
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PARQUET_SRC = Path("/local-scratch/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet")
SPLIT_SRC   = Path("/local-scratch/localhome/dya78/code/lightgen/data_processing/annotation/data_splits_emissive_74k_pinned.json")
JUP_BASE    = Path("/3dlg-jupiter-project/lightgen/dataset/lightgen_74k")
JUP_BASE_VIA_STAR2 = "/3dlg-jupiter-project/lightgen/dataset/lightgen_74k"


def build_metadata(dry_run: bool) -> pd.DataFrame:
    print(f"[bootstrap] reading parquet: {PARQUET_SRC}")
    df = pd.read_parquet(PARQUET_SRC)
    print(f"  parquet rows (raw): {len(df)}")
    if "success" not in df.columns:
        raise SystemExit("parquet missing 'success' column — split was built against success-filtered df")
    big = df[df["success"] == True].reset_index()  # ditem_id becomes a column
    print(f"  parquet rows (success==True): {len(big)}")

    print(f"[bootstrap] reading split: {SPLIT_SRC}")
    split = json.load(open(SPLIT_SRC))
    sizes = {k: split[k]["count"] for k in ("train", "val", "test")}
    print(f"  split counts: {sizes}  (sum={sum(sizes.values())})")

    rows = []
    for split_name in ("train", "val", "test"):
        for pos in split[split_name]["indices"]:
            r = big.iloc[pos]
            rows.append({
                "sha256":      r["ditem_id"],
                "glb_1k_path": r["glb_1k_path"],
                "split":       split_name,
            })
    meta = pd.DataFrame(rows)
    assert meta["sha256"].is_unique, "sha256 collision across splits — check pinning logic"
    print(f"[bootstrap] metadata.csv rows: {len(meta)}")
    print(meta.head(3).to_string())
    return meta


def rsync_to_jupyter(local_paths: list[Path], dry_run: bool) -> None:
    JUP_BASE.mkdir(parents=True, exist_ok=True)
    (JUP_BASE / "voxels").mkdir(exist_ok=True)
    (JUP_BASE / "latents").mkdir(exist_ok=True)
    (JUP_BASE / "logs").mkdir(exist_ok=True)
    for p in local_paths:
        dst = JUP_BASE / p.name
        print(f"[bootstrap] rsync {p}  ->  {dst}")
        if dry_run:
            continue
        cmd = ["rsync", "-av", str(p), str(dst)]
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute and print counts; skip rsync")
    args = ap.parse_args()

    meta = build_metadata(args.dry_run)
    if args.dry_run:
        print("[bootstrap] dry-run: skipping rsync")
        return 0

    workdir = Path("/tmp/prep_74k_bootstrap")
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "metadata.csv"
    meta.to_csv(csv_path, index=False)
    print(f"[bootstrap] wrote {csv_path}")

    parquet_local = workdir / PARQUET_SRC.name
    split_local   = workdir / SPLIT_SRC.name
    shutil.copy(PARQUET_SRC, parquet_local)
    shutil.copy(SPLIT_SRC, split_local)

    rsync_to_jupyter([csv_path, parquet_local, split_local], dry_run=False)
    print("[bootstrap] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry-run to verify counts**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
python data_toolkit/prep_74k_bootstrap.py --dry-run
```

Expected output (counts):
```
  parquet rows (raw): 854287
  parquet rows (success==True): 824858
  split counts: {'train': 73251, 'val': 112, 'test': 109}  (sum=73472)
[bootstrap] metadata.csv rows: 73472
```

- [ ] **Step 3: Commit (script only — bootstrap not yet run)**

```bash
git add data_toolkit/prep_74k_bootstrap.py
git commit -m "data_toolkit: prep_74k_bootstrap.py — build metadata.csv + ship parquet/split to jupyter"
```

---

### Task 3: Run bootstrap to populate jupyter

- [ ] **Step 1: Run the real bootstrap**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
mkdir -p log
TS=$(date +%Y%m%d-%H%M%S)
python data_toolkit/prep_74k_bootstrap.py 2>&1 | tee "log/${TS}-prep_74k_bootstrap.log"
```

Expected: `[bootstrap] done.` at the end; rsync output shows three files transferred.

- [ ] **Step 2: Verify jupyter contents on star2**

```bash
ssh star2 'ls -la /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/'
ssh star2 'wc -l /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/metadata.csv'
```

Expected: directory listing shows `metadata.csv`, `df_SomgProc_final.parquet`, `data_splits_emissive_74k_pinned.json`, plus empty subdirs `voxels/`, `latents/`, `logs/`. Wc reports `73473` (header + 73472 data rows).

- [ ] **Step 3: No commit (bootstrap is data, not code)**

---

## Phase 2 — Stage A wrapper

### Task 4: Write `prep_74k_dump.py`

**Files:**
- Create: `data_toolkit/prep_74k_dump.py`

Sharded Blender PBR dump. Mirrors the multiprocessing pattern in `regenerate_1k_dual_grid.py`. Reads metadata.csv, takes rank-N slice, spawns Blender per sample.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Stage A — Blender PBR dump for the 74k encoding pipeline.

Reads <root>/metadata.csv, takes the rank-N slice, and runs:
    blender --background --python blender_script/dump_pbr.py -- \
        --object <glb> --output_path <root>/pbr_dumps/<sha>.pickle
in parallel via ProcessPoolExecutor. Skip-if-exists: pickles already in
pbr_dumps/ are not re-dumped. Failures are appended to
pbr_dumps/new_records/part_<rank>_failures.txt; the script exits 0 even on
per-sample failures (the caller decides whether to retry).

Usage (typical, inside an sbatch body):
    python data_toolkit/prep_74k_dump.py \\
        --root /localscratch/dya78/lightgen_74k_part0 \\
        --rank 0 --world_size 8 --workers 8 \\
        --blender /localscratch/dya78/blender-4.5.1-linux-x64/blender
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


GLB_ROOT = Path("/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K")
DUMP_SCRIPT = (Path(__file__).parent / "blender_script" / "dump_pbr.py").resolve()


def _blender_dump(blender: str, glb_path: str, pickle_path: str) -> tuple[bool, str]:
    cmd = [blender, "--background", "--python", str(DUMP_SCRIPT), "--",
           "--object", glb_path, "--output_path", pickle_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "blender timeout (600s)"
    if res.returncode != 0 or not os.path.exists(pickle_path):
        return False, (res.stderr or res.stdout)[-500:].strip()
    return True, ""


def _process_one(args: tuple[str, str, str, str]) -> tuple[str, bool, str]:
    sha, glb_rel, root, blender = args
    glb_path = str(GLB_ROOT / glb_rel)
    out_path = os.path.join(root, "pbr_dumps", f"{sha}.pickle")
    if os.path.exists(out_path):
        return sha, True, "skip"
    if not os.path.exists(glb_path):
        return sha, False, f"glb_missing: {glb_path}"

    with tempfile.TemporaryDirectory(dir=os.path.join(root, "pbr_dumps")) as tmp:
        tmp_pickle = os.path.join(tmp, f"{sha}.pickle")
        ok, err = _blender_dump(blender, glb_path, tmp_pickle)
        if not ok:
            return sha, False, err
        shutil.move(tmp_pickle, out_path)
    return sha, True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blender", type=str,
                    default="/localscratch/dya78/blender-4.5.1-linux-x64/blender")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N samples (smoke test)")
    args = ap.parse_args()

    if not os.path.exists(args.blender):
        print(f"[prep_74k_dump] FATAL: blender not found at {args.blender}", file=sys.stderr)
        return 2

    csv_path = os.path.join(args.root, "metadata.csv")
    df = pd.read_csv(csv_path)
    print(f"[prep_74k_dump] metadata rows: {len(df)}  rank={args.rank}/{args.world_size}")

    start = len(df) * args.rank // args.world_size
    end   = len(df) * (args.rank + 1) // args.world_size
    df_rank = df.iloc[start:end]
    if args.limit is not None:
        df_rank = df_rank.head(args.limit)
    print(f"[prep_74k_dump] this rank: {len(df_rank)} samples (slice {start}..{end})")

    out_dir = os.path.join(args.root, "pbr_dumps")
    os.makedirs(os.path.join(out_dir, "new_records"), exist_ok=True)

    work = [(r["sha256"], r["glb_1k_path"], args.root, args.blender)
            for _, r in df_rank.iterrows()]

    t0 = time.time()
    success = skip = fail = 0
    fail_log: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            sha, ok, msg = fut.result()
            if ok and msg == "skip":
                skip += 1
            elif ok:
                success += 1
            else:
                fail += 1
                fail_log.append((sha, msg))
            if i % 25 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1e-6)
                eta_h = (len(work) - i) / max(rate, 1e-6) / 3600
                print(f"[{i:5d}/{len(work)}] ok={success} skip={skip} fail={fail}  "
                      f"rate={rate:.2f}/s  eta={eta_h:.2f}h", flush=True)

    fail_path = os.path.join(out_dir, "new_records", f"part_{args.rank}_failures.txt")
    with open(fail_path, "w") as f:
        for sha, msg in fail_log:
            f.write(f"{sha}\t{msg}\n")
    print(f"[prep_74k_dump] done: ok={success} skip={skip} fail={fail} "
          f"in {(time.time()-t0)/60:.1f} min  failures -> {fail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-verify the script's argparse + slicing logic (no Blender call)**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
python data_toolkit/prep_74k_dump.py --help | head -5
```

Expected: argparse usage banner with the seven flags above; no exception.

- [ ] **Step 3: Commit**

```bash
git add data_toolkit/prep_74k_dump.py
git commit -m "data_toolkit: prep_74k_dump.py — sharded Blender PBR dump for 74k subset"
```

---

## Phase 3 — Stage B-2 wrapper

### Task 5: Write `prep_74k_dual_grid.py`

**Files:**
- Create: `data_toolkit/prep_74k_dual_grid.py`

Mirror of `regenerate_1k_dual_grid.py:_dual_grid_from_pickle` but driven by `metadata.csv` and sharded.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Stage B-2 — dual_grid_<res>/<sha>.vxz from pbr_dumps/<sha>.pickle.

This is the Blender-pickle path (vertex set matches voxelize_pbr.py's output),
mandatory to keep shape-latent coords aligned with PBR/emission-latent coords.
The trimesh-based shortcut in old `dual_grid_from_glb.py` produced different
sparse coords and broke the DiT's `assert torch.equal(x_0.coords, shape_z.coords)`.

Usage:
    python data_toolkit/prep_74k_dual_grid.py \\
        --root /localscratch/dya78/lightgen_74k_part0 \\
        --rank 0 --world_size 8 --workers 16 --resolution 256
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

# Stub modules that o_voxel imports lazily.
for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel
import o_voxel.convert
import o_voxel.io


def _dual_grid_from_pickle(pickle_path: str, dual_grid_path: str, resolution: int) -> tuple[bool, str]:
    with open(pickle_path, "rb") as f:
        dump = pickle.load(f)

    objs = [obj for obj in dump["objects"]
            if obj["vertices"].size != 0 and obj["faces"].size != 0]
    if not objs:
        return False, "no geometry in pickle"

    start = 0
    all_v, all_f = [], []
    for obj in objs:
        all_v.append(obj["vertices"])
        all_f.append(obj["faces"] + start)
        start += len(obj["vertices"])

    vertices = torch.from_numpy(np.concatenate(all_v, axis=0)).float()
    faces    = torch.from_numpy(np.concatenate(all_f, axis=0)).long()

    vmin = vertices.min(dim=0)[0]
    vmax = vertices.max(dim=0)[0]
    center = (vmin + vmax) / 2
    scale  = 0.99999 / (vmax - vmin).max()
    vertices = (vertices - center) * scale

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    dual_vertices = dual_vertices * resolution - voxel_indices
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1]
                   + 2 * intersected[:, 1:2]
                   + 4 * intersected[:, 2:3]).type(torch.uint8)

    attr = {"vertices": dual_vertices, "intersected": intersected}
    o_voxel.io.write_vxz(dual_grid_path, voxel_indices, attr)
    return True, f"N={voxel_indices.shape[0]}"


def _process_one(args: tuple[str, str, int]) -> tuple[str, bool, str]:
    sha, root, resolution = args
    pickle_path    = os.path.join(root, "pbr_dumps", f"{sha}.pickle")
    dual_grid_path = os.path.join(root, f"dual_grid_{resolution}", f"{sha}.vxz")
    if os.path.exists(dual_grid_path):
        return sha, True, "skip"
    if not os.path.exists(pickle_path):
        return sha, False, "pickle_missing"
    try:
        ok, info = _dual_grid_from_pickle(pickle_path, dual_grid_path, resolution)
        return sha, ok, info
    except Exception as e:
        return sha, False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.root, "metadata.csv"))
    start = len(df) * args.rank // args.world_size
    end   = len(df) * (args.rank + 1) // args.world_size
    df_rank = df.iloc[start:end]
    if args.limit is not None:
        df_rank = df_rank.head(args.limit)
    print(f"[prep_74k_dual_grid] rank {args.rank}/{args.world_size}: {len(df_rank)} samples")

    out_dir = os.path.join(args.root, f"dual_grid_{args.resolution}")
    os.makedirs(os.path.join(out_dir, "new_records"), exist_ok=True)

    work = [(r["sha256"], args.root, args.resolution) for _, r in df_rank.iterrows()]

    t0 = time.time()
    success = skip = fail = 0
    fail_log: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            sha, ok, msg = fut.result()
            if ok and msg == "skip":
                skip += 1
            elif ok:
                success += 1
            else:
                fail += 1
                fail_log.append((sha, msg))
            if i % 100 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1e-6)
                eta_m = (len(work) - i) / max(rate, 1e-6) / 60
                print(f"[{i:5d}/{len(work)}] ok={success} skip={skip} fail={fail}  "
                      f"rate={rate:.1f}/s  eta={eta_m:.1f}min", flush=True)

    fail_path = os.path.join(out_dir, "new_records", f"part_{args.rank}_failures.txt")
    with open(fail_path, "w") as f:
        for sha, msg in fail_log:
            f.write(f"{sha}\t{msg}\n")
    print(f"[prep_74k_dual_grid] done: ok={success} skip={skip} fail={fail} "
          f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-verify against an existing 1k pickle**

The 1k pipeline already produced pickles at
`/local-scratch/localhome/dya78/lightgen_1k/pbr_dumps/`. Reuse one for a
single-sample local smoke test.

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
mkdir -p /tmp/dg74k_smoke/pbr_dumps /tmp/dg74k_smoke/dual_grid_256
SHA=$(ls /local-scratch/localhome/dya78/lightgen_1k/pbr_dumps/ | grep '\.pickle$' | head -n1 | sed 's/\.pickle$//')
cp /local-scratch/localhome/dya78/lightgen_1k/pbr_dumps/${SHA}.pickle /tmp/dg74k_smoke/pbr_dumps/
echo "sha256,glb_1k_path,split" > /tmp/dg74k_smoke/metadata.csv
echo "${SHA},UNUSED,train"     >> /tmp/dg74k_smoke/metadata.csv
python data_toolkit/prep_74k_dual_grid.py --root /tmp/dg74k_smoke --resolution 256 --workers 1
ls -la /tmp/dg74k_smoke/dual_grid_256/${SHA}.vxz
```

Expected: log line `ok=1 skip=0 fail=0` and a `<sha>.vxz` file ~200KB.

- [ ] **Step 3: Commit**

```bash
git add data_toolkit/prep_74k_dual_grid.py
git commit -m "data_toolkit: prep_74k_dual_grid.py — sharded dual_grid from Blender pickle"
```

---

## Phase 4 — Convenience scripts

### Task 6: Write `extract_74k_tars.sh`

**Files:**
- Create: `data_toolkit/extract_74k_tars.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Extract all per-shard tars (voxels + latents) from jupyter into a unified
# lightgen_74k tree on the caller's local disk. Idempotent — re-running is safe.
set -euo pipefail

JUP_BASE=${1:-/3dlg-jupiter-project/lightgen/dataset/lightgen_74k}
DST=${2:-/localscratch/dya78/lightgen_74k}

mkdir -p "${DST}"
echo "[extract_74k_tars] copying scaffold from ${JUP_BASE} -> ${DST}"
for f in df_SomgProc_final.parquet data_splits_emissive_74k_pinned.json metadata.csv; do
    rsync -av "${JUP_BASE}/${f}" "${DST}/"
done

echo "[extract_74k_tars] extracting voxels tars"
for tar in "${JUP_BASE}/voxels"/part*.tar; do
    [ -e "$tar" ] || { echo "  skip (no match): $tar"; continue; }
    echo "  extracting $(basename "$tar")"
    tar -xf "$tar" -C "${DST}"
done

echo "[extract_74k_tars] extracting latents tars"
for tar in "${JUP_BASE}/latents"/part*.tar; do
    [ -e "$tar" ] || { echo "  skip (no match): $tar"; continue; }
    echo "  extracting $(basename "$tar")"
    tar -xf "$tar" -C "${DST}"
done

echo "[extract_74k_tars] done. Tree at: ${DST}"
ls "${DST}"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/data_toolkit/extract_74k_tars.sh
```

- [ ] **Step 3: Commit**

```bash
git add data_toolkit/extract_74k_tars.sh
git commit -m "data_toolkit: extract_74k_tars.sh — fan-in 16 jupyter tars into a unified tree"
```

---

### Task 7: Write `check_74k_progress.py`

**Files:**
- Create: `data_toolkit/check_74k_progress.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Read each per-shard tar on jupyter, count members per artifact dir, and
report per-shard completion vs the expected sha list from metadata.csv.

Run from any host that can see /3dlg-jupiter-project/.

Usage:
    python data_toolkit/check_74k_progress.py
"""
from __future__ import annotations
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

JUP = Path("/3dlg-jupiter-project/lightgen/dataset/lightgen_74k")
ARTIFACTS = ["pbr_voxels_256", "emission_voxels_256", "dual_grid_256",
             "pbr_latents/tex_enc_next_dc_f16c32_fp16_256",
             "shape_latents/shape_enc_next_dc_f16c32_fp16_256"]


def _shard_expected(meta: pd.DataFrame, rank: int, world: int = 8) -> set[str]:
    s = len(meta) * rank // world
    e = len(meta) * (rank + 1) // world
    return set(meta.iloc[s:e]["sha256"])


def _count_in_tar(tar_path: Path) -> dict[str, set[str]]:
    """Returns {artifact_subdir: {sha256, ...}} for every entry in the tar."""
    out: dict[str, set[str]] = defaultdict(set)
    if not tar_path.exists():
        return out
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            for art in ARTIFACTS:
                if m.name.startswith(art + "/"):
                    base = os.path.basename(m.name)
                    sha = base.rsplit(".", 1)[0]
                    out[art].add(sha)
                    break
    return out


def main() -> int:
    meta = pd.read_csv(JUP / "metadata.csv")
    print(f"[check_74k_progress] total samples: {len(meta)}")
    print()

    print(f"{'shard':>5}  {'expected':>9}  {'pbr_vox':>7}  {'em_vox':>7}  {'dual':>7}  {'pbr_lat':>7}  {'shape_lat':>9}")
    grand = {a: 0 for a in ARTIFACTS}
    for rank in range(8):
        exp = _shard_expected(meta, rank)
        v_tar = JUP / "voxels"  / f"part{rank}_voxels.tar"
        l_tar = JUP / "latents" / f"part{rank}_latents.tar"
        v_have = _count_in_tar(v_tar)
        l_have = _count_in_tar(l_tar)
        have = {**v_have, **l_have}
        cells = []
        for art in ARTIFACTS:
            n = len(have.get(art, set()) & exp)
            grand[art] += n
            cells.append(n)
        print(f"{rank:>5}  {len(exp):>9}  " + "  ".join(f"{c:>7}" for c in cells[:4]) + f"  {cells[4]:>9}")

    print()
    print(f"{'TOTAL':>5}  {len(meta):>9}  " + "  ".join(f"{grand[a]:>7}" for a in ARTIFACTS[:4]) + f"  {grand[ARTIFACTS[4]]:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-verify it imports + runs against the (so-far-empty) jupyter dir**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
ssh star2 'cd /localscratch/dya78/TRELLIS.2-lightning && conda activate trellis2 && python data_toolkit/check_74k_progress.py'
```

Expected: total `73472`; all per-shard cells `0` (nothing produced yet); `expected` per shard varies between 9183 and 9184.

- [ ] **Step 3: Commit**

```bash
git add data_toolkit/check_74k_progress.py
git commit -m "data_toolkit: check_74k_progress.py — per-shard tar inventory check"
```

---

## Phase 5 — Cluster launch infrastructure

### Task 8: Write `prep_74k_install_blender.sh`

**Files:**
- Create: `script/star2/prep_74k_install_blender.sh`

Idempotent self-install of Blender 4.5.1 + Pillow under `/localscratch/dya78/`. Sourced (not executed) by the AB job body.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Idempotent Blender install on the current node's /localscratch.
# Sourced from script/star2/prep_74k_stage_ab.sh; sets BLENDER_PATH for the
# caller. No sudo required.
set -euo pipefail

BLENDER_VERSION="4.5.1"
BLENDER_DIR="/localscratch/dya78/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_PATH="${BLENDER_DIR}/blender"
TARBALL_URL="https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
TARBALL_LOCAL="/localscratch/dya78/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

if [[ -x "${BLENDER_PATH}" ]]; then
    echo "[install_blender] already present: ${BLENDER_PATH}"
    export BLENDER_PATH
    return 0
fi

echo "[install_blender] downloading Blender ${BLENDER_VERSION} -> ${TARBALL_LOCAL}"
mkdir -p /localscratch/dya78
wget -q --show-progress -O "${TARBALL_LOCAL}" "${TARBALL_URL}"

echo "[install_blender] extracting -> /localscratch/dya78/"
tar -xf "${TARBALL_LOCAL}" -C /localscratch/dya78/
rm -f "${TARBALL_LOCAL}"

if [[ ! -x "${BLENDER_PATH}" ]]; then
    echo "[install_blender] FATAL: ${BLENDER_PATH} not present after extraction" >&2
    return 1
fi

echo "[install_blender] installing Pillow into Blender's bundled Python"
PILLOW_SCRIPT="$(pwd)/data_toolkit/blender_script/install_pillow.py"
"${BLENDER_PATH}" -b --python "${PILLOW_SCRIPT}"

echo "[install_blender] done. BLENDER_PATH=${BLENDER_PATH}"
export BLENDER_PATH
```

- [ ] **Step 2: Mark executable**

```bash
chmod +x /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/script/star2/prep_74k_install_blender.sh
```

- [ ] **Step 3: Commit**

```bash
git add script/star2/prep_74k_install_blender.sh
git commit -m "script/star2: prep_74k_install_blender.sh — idempotent Blender 4.5.1 install"
```

---

### Task 9: Write `prep_74k_stage_ab.sh`

**Files:**
- Create: `script/star2/prep_74k_stage_ab.sh`

Local launcher that emits + submits one AB sbatch job (one shard) and prints the resulting jobid on stdout (last token, for `prep_74k_chain.sh` to capture).

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# Submit one AB sbatch job for shard ${RANK}/${WORLD_SIZE}.
# Outputs jobid (single integer) on the last line of stdout for chain capture.
# Stage A: Blender PBR dump. Stage B: voxelize_pbr + dual_grid.
set -euo pipefail

RANK="${1:?usage: $0 RANK WORLD_SIZE [LIMIT]}"
WORLD_SIZE="${2:?usage: $0 RANK WORLD_SIZE [LIMIT]}"
LIMIT_FLAG=""
if [[ -n "${3:-}" ]]; then
    LIMIT_FLAG="--limit ${3}"
fi

JOB_NAME="prep_74k_AB_r${RANK}"
echo "Submitting ${JOB_NAME} (rank ${RANK}/${WORLD_SIZE}, limit='${LIMIT_FLAG}')"

OUT=$(ssh star2 "sbatch --parsable" <<EOF
#!/bin/zsh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00
#SBATCH --output=/localscratch/dya78/${JOB_NAME}.out
#SBATCH -J ${JOB_NAME}

set -e
source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
git pull origin lightgen
conda activate trellis2

LOCAL=/localscratch/dya78/lightgen_74k_part${RANK}
JUP=/3dlg-jupiter-project/lightgen/dataset/lightgen_74k
mkdir -p \${LOCAL}/log
TS=\$(date +%Y%m%d-%H%M%S)
LOG=\${LOCAL}/log/\${TS}-prep_74k_AB_rank${RANK}.log

# Idempotent Blender install (sets BLENDER_PATH).
. script/star2/prep_74k_install_blender.sh

rsync -av \${JUP}/metadata.csv                            \${LOCAL}/
rsync -av \${JUP}/df_SomgProc_final.parquet               \${LOCAL}/
rsync -av \${JUP}/data_splits_emissive_74k_pinned.json    \${LOCAL}/

(
  set -e
  python data_toolkit/prep_74k_dump.py \\
      --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} \\
      --workers 8 --blender \${BLENDER_PATH} ${LIMIT_FLAG}
  python data_toolkit/voxelize_pbr.py lightgen_74k \\
      --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} \\
      --max_workers 16 --resolution 256
  python data_toolkit/prep_74k_dual_grid.py \\
      --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} \\
      --workers 16 --resolution 256 ${LIMIT_FLAG}
) 2>&1 | tee "\${LOG}"
AB_EXIT=\${pipestatus[1]}

if [[ \${AB_EXIT} -eq 0 ]]; then
  cd \${LOCAL}
  tar -cf part${RANK}_voxels.tar pbr_voxels_256 emission_voxels_256 dual_grid_256
  rsync -av part${RANK}_voxels.tar \${JUP}/voxels/
  rsync -av \${LOG} \${JUP}/logs/
  rm -rf pbr_dumps
fi

exit \${AB_EXIT}
EOF
)

JOBID=$(echo "${OUT}" | tail -n1)
echo "submitted ${JOB_NAME} jobid=${JOBID}"
echo "${JOBID}"
```

- [ ] **Step 2: Mark executable**

```bash
chmod +x /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/script/star2/prep_74k_stage_ab.sh
```

- [ ] **Step 3: Commit**

```bash
git add script/star2/prep_74k_stage_ab.sh
git commit -m "script/star2: prep_74k_stage_ab.sh — Stage A+B sbatch launcher"
```

---

### Task 10: Write `prep_74k_stage_c.sh`

**Files:**
- Create: `script/star2/prep_74k_stage_c.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# Submit one C sbatch job for shard ${RANK}/${WORLD_SIZE}, depending on the
# AB job whose id is given as the third arg (or "none" to skip dependency).
# Prints the C jobid on the last line of stdout.
set -euo pipefail

RANK="${1:?usage: $0 RANK WORLD_SIZE AB_JOBID}"
WORLD_SIZE="${2:?usage: $0 RANK WORLD_SIZE AB_JOBID}"
AB_JOBID="${3:?usage: $0 RANK WORLD_SIZE AB_JOBID  (or 'none')}"

DEP_FLAG=""
if [[ "${AB_JOBID}" != "none" ]]; then
    DEP_FLAG="--dependency=afterok:${AB_JOBID}"
fi

JOB_NAME="prep_74k_C_r${RANK}"
echo "Submitting ${JOB_NAME} (rank ${RANK}/${WORLD_SIZE}, dep=${AB_JOBID})"

OUT=$(ssh star2 "sbatch ${DEP_FLAG} --parsable" <<EOF
#!/bin/zsh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00
#SBATCH --output=/localscratch/dya78/${JOB_NAME}.out
#SBATCH -J ${JOB_NAME}

set -e
source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
git pull origin lightgen
conda activate trellis2

LOCAL=/localscratch/dya78/lightgen_74k_part${RANK}
JUP=/3dlg-jupiter-project/lightgen/dataset/lightgen_74k
mkdir -p \${LOCAL}/log
TS=\$(date +%Y%m%d-%H%M%S)
LOG=\${LOCAL}/log/\${TS}-prep_74k_C_rank${RANK}.log

(
  set -e
  rsync -av \${JUP}/metadata.csv                            \${LOCAL}/
  rsync -av \${JUP}/df_SomgProc_final.parquet               \${LOCAL}/
  rsync -av \${JUP}/voxels/part${RANK}_voxels.tar           \${LOCAL}/
  cd \${LOCAL} && tar -xf part${RANK}_voxels.tar && cd -

  python data_toolkit/encode_pbr_latent.py \\
      --root \${LOCAL} --rank 0 --world_size 1 --resolution 256
  python data_toolkit/encode_shape_from_dual_grid.py \\
      --data_root \${LOCAL} \\
      --parquet \${LOCAL}/df_SomgProc_final.parquet \\
      --rank 0 --world_size 1 --resolution 256
) 2>&1 | tee "\${LOG}"
C_EXIT=\${pipestatus[1]}

if [[ \${C_EXIT} -eq 0 ]]; then
  cd \${LOCAL}
  tar -cf part${RANK}_latents.tar pbr_latents shape_latents
  rsync -av part${RANK}_latents.tar \${JUP}/latents/
  rsync -av \${LOG} \${JUP}/logs/
fi

exit \${C_EXIT}
EOF
)

JOBID=$(echo "${OUT}" | tail -n1)
echo "submitted ${JOB_NAME} jobid=${JOBID}"
echo "${JOBID}"
```

- [ ] **Step 2: Mark executable**

```bash
chmod +x /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/script/star2/prep_74k_stage_c.sh
```

- [ ] **Step 3: Commit**

```bash
git add script/star2/prep_74k_stage_c.sh
git commit -m "script/star2: prep_74k_stage_c.sh — Stage C sbatch launcher (afterok)"
```

---

### Task 11: Write `prep_74k_smoke.sh`

**Files:**
- Create: `script/star2/prep_74k_smoke.sh`

End-to-end smoke pilot: submit one AB job (rank=0, world_size=8) with `LIMIT=50`, then one C job depending on it.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Smoke pilot: one AB+C pair on shard 0 of world_size=8, limited to 50 samples.
# Use this to validate Blender install, GLB visibility, and the tar handoff
# *before* fan-out via prep_74k_chain.sh.
set -euo pipefail

cd "$(dirname "$0")/../.."

RANK=0
WORLD_SIZE=8
LIMIT=50

AB_JOBID=$(./script/star2/prep_74k_stage_ab.sh ${RANK} ${WORLD_SIZE} ${LIMIT} | tail -n1)
echo "AB jobid: ${AB_JOBID}"

C_JOBID=$(./script/star2/prep_74k_stage_c.sh ${RANK} ${WORLD_SIZE} ${AB_JOBID} | tail -n1)
echo "C jobid:  ${C_JOBID}"

echo
echo "Smoke pilot submitted:"
echo "  AB: ${AB_JOBID}"
echo "  C : ${C_JOBID}  (depends on AB)"
echo
echo "Watch with:  ssh star2 squeue -u dya78"
```

- [ ] **Step 2: Mark executable + commit**

```bash
chmod +x /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/script/star2/prep_74k_smoke.sh
git add script/star2/prep_74k_smoke.sh
git commit -m "script/star2: prep_74k_smoke.sh — single-shard AB+C pair on 50 samples"
```

---

### Task 12: Write `prep_74k_chain.sh`

**Files:**
- Create: `script/star2/prep_74k_chain.sh`

- [ ] **Step 1: Write the chain orchestrator**

```bash
#!/usr/bin/env bash
# Submit all 16 sbatch jobs (8 AB + 8 C) for the full 74k fan-out.
# AB jobs are independent across shards; each C job depends only on its own AB.
set -euo pipefail

cd "$(dirname "$0")/../.."

WORLD_SIZE=8
echo "shard  AB_jobid    C_jobid"
echo "-----  ---------   ---------"
for RANK in $(seq 0 $((WORLD_SIZE-1))); do
    AB=$(./script/star2/prep_74k_stage_ab.sh ${RANK} ${WORLD_SIZE} | tail -n1)
    C=$(./script/star2/prep_74k_stage_c.sh ${RANK} ${WORLD_SIZE} ${AB} | tail -n1)
    printf "%5d  %-9s   %-9s\n" "${RANK}" "${AB}" "${C}"
done
```

- [ ] **Step 2: Mark executable + commit**

```bash
chmod +x /local-scratch/localhome/dya78/code/TRELLIS.2-lightning/script/star2/prep_74k_chain.sh
git add script/star2/prep_74k_chain.sh
git commit -m "script/star2: prep_74k_chain.sh — submit all 16 jobs for 74k fan-out"
```

---

### Task 13: Push branch to origin so star2 can pull

- [ ] **Step 1: Push**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
git push origin lightgen
```

Expected: branch `lightgen` updated on origin with all the new commits from this plan.

- [ ] **Step 2: Sanity-check star2 pull works**

```bash
ssh star2 'cd /localscratch/dya78/TRELLIS.2-lightning && git fetch origin lightgen && git log --oneline origin/lightgen -5'
```

Expected: top 5 commits on origin/lightgen include the plan's commits in reverse order.

---

## Phase 6 — Smoke pilot

### Task 14: Run the smoke pilot end-to-end

This is the **integration test** for the whole pipeline (no Python unit tests
exist for the prep flow). The smoke pilot validates Blender install,
`/3dlg-falas` GLB visibility from compute nodes, the per-stage scripts, the
tar+rsync handoff, and the afterok chaining — all on a 50-sample subset that
finishes in ~30 minutes wallclock.

- [ ] **Step 1: Submit**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
TS=$(date +%Y%m%d-%H%M%S)
./script/star2/prep_74k_smoke.sh 2>&1 | tee "log/${TS}-prep_74k_smoke_submit.log"
```

Expected: `AB jobid` and `C jobid` printed; both are integer Slurm job IDs.

- [ ] **Step 2: Wait for AB to start, peek at the live log**

Poll squeue every ~5 minutes (per [memory](feedback_check_frequency.md): 10 min cadence is fine; faster polling is wasteful). When AB transitions from `PD` → `R`, peek at its output:

```bash
ssh star2 'squeue -u dya78 -o "%.10i %.20j %.2t %.10M %.20R"'
ssh star2 'tail -50 /localscratch/dya78/prep_74k_AB_r0.out'
```

Expected progression in the AB log:
1. `[install_blender] downloading Blender 4.5.1 -> ...` (only on first run on this node; subsequent runs print "already present")
2. `[install_blender] done. BLENDER_PATH=...`
3. `[prep_74k_dump]` line showing `metadata rows: 73472  rank=0/8`
4. `[prep_74k_dump]` showing `this rank: 50 samples`
5. `[prep_74k_dump] done: ok=50 skip=0 fail=0` after a few minutes
6. `[voxelize_pbr]` lines from the existing script
7. `[prep_74k_dual_grid] done: ok=50 ...`
8. `tar -cf part0_voxels.tar ...`
9. `rsync ...`

- [ ] **Step 3: Verify voxels tar landed on jupyter**

After AB finishes (`squeue` no longer shows it):

```bash
ssh star2 'ls -la /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/'
ssh star2 'tar -tf /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/part0_voxels.tar | head -10'
ssh star2 'tar -tf /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/part0_voxels.tar | wc -l'
```

Expected: `part0_voxels.tar` exists; tar contents include `pbr_voxels_256/<sha>.vxz`, `emission_voxels_256/<sha>.vxz`, `dual_grid_256/<sha>.vxz`. Member count is **150** (50 × 3 artifacts).

- [ ] **Step 4: Wait for C to start, watch the encode log**

```bash
ssh star2 'squeue -u dya78'
ssh star2 'tail -50 /localscratch/dya78/prep_74k_C_r0.out'
```

Expected: rsync of voxels tar; tar -xf; `encode_pbr_latent.py` log lines (`Processing 50 objects ...`); then `encode_shape_from_dual_grid.py`; finally `tar -cf part0_latents.tar` + rsync.

- [ ] **Step 5: Verify latents tar landed on jupyter**

```bash
ssh star2 'ls -la /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/'
ssh star2 'tar -tf /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/part0_latents.tar | head'
ssh star2 'tar -tf /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/part0_latents.tar | wc -l'
```

Expected: `part0_latents.tar` exists; member count is **100** (50 × 2 latent kinds).

- [ ] **Step 6: Run the progress checker**

```bash
ssh star2 'cd /localscratch/dya78/TRELLIS.2-lightning && conda activate trellis2 && python data_toolkit/check_74k_progress.py'
```

Expected: shard 0's row shows `pbr_vox=50 em_vox=50 dual=50 pbr_lat=50 shape_lat=50`; all other shards show `0`.

- [ ] **Step 7: If smoke pilot fails, triage before fan-out**

Common failures and fixes:
* **`glb_missing` for many samples** → mount issue on compute node; `ls /3dlg-falas/...` from a fresh `srun --pty` to confirm; tweak `prep_74k_dump.py`'s `GLB_ROOT` if path differs.
* **Blender wget blocked** → swap `TARBALL_URL` for an alternative blender.org mirror in `prep_74k_install_blender.sh`.
* **Pillow install hangs** → known issue if `/localscratch` is full; check `df -h /localscratch` from inside the AB job.
* **`exit 1` in zsh `pipestatus` reading** → wrong shell; verify `#!/bin/zsh` shebang is on the SBATCH body's first line and `source ~/.zshrc` ran (per [CLAUDE.md cluster launch scripts](CLAUDE.md)).

After fixes, re-run smoke pilot from Step 1. The previous AB job's `pbr_dumps` are gone (rm -rf at end), so it's a clean re-run; the existing `voxels/part0_voxels.tar` will be **overwritten** by the second run's rsync — that's fine because we're still on the smoke 50-sample slice.

- [ ] **Step 8: Once smoke is green, clean up before fan-out**

Delete the smoke tars on jupyter so the real fan-out doesn't fight a half-populated `voxels/` dir:

```bash
ssh star2 'rm -f /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/part0_voxels.tar'
ssh star2 'rm -f /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/part0_latents.tar'
ssh star2 'ls /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/  /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/'
```

Expected: both directories empty.

---

## Phase 7 — Fan-out

### Task 15: Submit the 16-job chain

- [ ] **Step 1: Run chain orchestrator**

```bash
cd /local-scratch/localhome/dya78/code/TRELLIS.2-lightning
TS=$(date +%Y%m%d-%H%M%S)
./script/star2/prep_74k_chain.sh 2>&1 | tee "log/${TS}-prep_74k_chain_submit.log"
```

Expected: 8 lines of `shard / AB_jobid / C_jobid`. Save this log — it's the master ID list for monitoring.

- [ ] **Step 2: Initial squeue check**

```bash
ssh star2 'squeue -u dya78 -o "%.10i %.20j %.2t %.10M %.10l %.20R"'
```

Expected: 16 jobs listed. AB jobs may be `PD` (pending) or `R` (running). C jobs are `PD` with `Dependency` reason.

- [ ] **Step 3: Set up periodic monitoring**

Per [memory](feedback_check_frequency.md), poll every ~10 minutes, not faster:

```bash
ssh star2 'squeue -u dya78 | head'
ssh star2 'ls /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/  /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/'
ssh star2 'cd /localscratch/dya78/TRELLIS.2-lightning && conda activate trellis2 && python data_toolkit/check_74k_progress.py'
```

Expected timeline (best case):
* T0..T+19h: AB jobs running, no tars visible.
* T+19h..T+22h: AB jobs finish in waves, voxels tars accumulate.
* T+19h..T+23h: C jobs unblock as their AB finishes; latents tars appear.
* T+23h: all 8 latents tars visible; fan-out done.

If any AB job exceeds **40 hours** wallclock, peek at its `.out` log on `/localscratch/dya78/prep_74k_AB_r<N>.out`. Common cause: a particular GLB's Blender dump is hung — it'll eventually hit the 600s subprocess timeout and continue, so wait.

---

### Task 16: Final verification

- [ ] **Step 1: All 16 tars present**

```bash
ssh star2 'ls /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/  | wc -l'
ssh star2 'ls /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/latents/ | wc -l'
```

Expected: both report `8`.

- [ ] **Step 2: Final progress report**

```bash
ssh star2 'cd /localscratch/dya78/TRELLIS.2-lightning && conda activate trellis2 && python data_toolkit/check_74k_progress.py'
```

Expected: every per-shard row's cells are within ~1-2% of the `expected` column. Total cells should each be ≥ 73,472 × 95% = 69,799.

- [ ] **Step 3: Triage low-completion shards**

If any artifact column for any shard is < 95% of expected, find the failing shas:

```bash
ssh star2 'tar -tf /3dlg-jupiter-project/lightgen/dataset/lightgen_74k/voxels/part<N>_voxels.tar | grep pbr_voxels_256/ | sed "s|^pbr_voxels_256/||;s|\.vxz$||" > /tmp/have.txt'
# Compare against the expected sha list for that shard via metadata.csv slice.
```

If failures are concentrated in a single shard with a recoverable error
(e.g., transient Slurm/network failure), resubmit just that shard's AB+C
pair via `prep_74k_stage_ab.sh <RANK> 8` then `prep_74k_stage_c.sh <RANK> 8 <AB_jobid>`.

If failures are scattered across shards (likely Blender failing on
specific malformed GLBs), accept the dropouts — they will reproduce
deterministically and aren't pipeline bugs.

- [ ] **Step 4: Append "Data Paths" note in CLAUDE.md**

Edit `CLAUDE.md`'s "Data Paths" subsection to add a bullet referencing the new layout. Insert below the existing `**Data root:** data/lightgen_test/ ...` paragraph:

```markdown
- **74k subset (jupyter):** `/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/` (or `/cs/3dlg-jupiter-project/...` from the jupyter side). Sixteen tars (8 voxels + 8 latents) plus `metadata.csv` + parquet + split JSON. Extract for training via `data_toolkit/extract_74k_tars.sh <jupyter_base> <local_dst>` — the resulting tree mirrors `data/lightgen_1k/` so existing dataset configs port over with one path swap.
```

- [ ] **Step 5: Commit the doc update**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md — note jupyter 74k dataset layout + extraction workflow"
git push origin lightgen
```

---

## Self-review notes

* **Spec coverage**:
  - § 2.1 invariant → Task 5 (uses pickle path, never trimesh).
  - § 2.2 Blender install → Task 8.
  - § 2.3 tar handoff → Tasks 9 (AB tar+rsync), 10 (C tar+rsync), 6 (extract).
  - § 2.4 pickle deletion → Task 9 (AB job's `rm -rf pbr_dumps` after upload).
  - § 2.5 sharding/topology → Tasks 9, 10, 11, 12.
  - § 4 metadata.csv schema → Task 2.
  - § 5.1 new scripts → Tasks 1, 2, 4, 5, 6, 7.
  - § 5.2 reused → Task 9 invokes voxelize_pbr.py and Task 10 invokes the encoders, no edits.
  - § 5.3 launch scripts → Tasks 8–12.
  - § 6.3 smoke pilot → Task 14.
  - § 6.4 fan-out → Task 15.
  - § 7 failure handling → Task 14 step 7 + Task 16 step 3.
  - § 8 extraction → Task 6, plus Task 16 Step 4 doc note.
* **Placeholder scan**: no `TBD`, no `add appropriate error handling`, no
  unspecified imports.
* **Type / signature consistency**:
  `prep_74k_dump.py:_process_one` returns `(sha, bool, msg)`; the loop
  unpacks the same shape. Same for `prep_74k_dual_grid.py`. The `.sh`
  launchers all take `RANK WORLD_SIZE [LIMIT/AB_JOBID]` consistently.

---

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-05-07-lightgen-74k-encoding.md`. The user
has activated auto-mode, so we proceed directly to inline execution
(no per-task subagent dispatch). Tasks 1–13 run locally without any
cluster interaction; Task 14 (smoke) requires waiting on Slurm and
should be the first natural checkpoint. Task 15 (fan-out) is the
second checkpoint — once the 16-job chain is live, the rest is
monitoring + final verification.
