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
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PARQUET_SRC = Path("/local-scratch/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet")
SPLIT_SRC   = Path("/local-scratch/localhome/dya78/code/lightgen/data_processing/annotation/data_splits_emissive_74k_pinned.json")


def _resolve_jupyter_base() -> Path:
    """The jupyter NFS mount appears under different paths on different hosts:
      * star2 cluster: /3dlg-jupiter-project/...
      * local cs-3dlg-23 (and other CS workstations): /cs/3dlg-jupiter-project/...
    Pick whichever mount actually exists."""
    candidates = [
        Path("/3dlg-jupiter-project/lightgen/dataset/lightgen_74k"),
        Path("/cs/3dlg-jupiter-project/lightgen/dataset/lightgen_74k"),
    ]
    for p in candidates:
        if p.parent.parent.parent.exists():  # /<...>/lightgen exists?
            return p
    raise SystemExit(
        "Could not find the jupyter mount. Tried:\n  " +
        "\n  ".join(str(c) for c in candidates))


JUP_BASE = _resolve_jupyter_base()


def build_metadata() -> pd.DataFrame:
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


def rsync_to_jupyter(local_paths: list[Path]) -> None:
    JUP_BASE.mkdir(parents=True, exist_ok=True)
    (JUP_BASE / "voxels").mkdir(exist_ok=True)
    (JUP_BASE / "latents").mkdir(exist_ok=True)
    (JUP_BASE / "logs").mkdir(exist_ok=True)
    for p in local_paths:
        dst = JUP_BASE / p.name
        print(f"[bootstrap] rsync {p}  ->  {dst}")
        cmd = ["rsync", "-av", str(p), str(dst)]
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute and print counts; skip rsync")
    args = ap.parse_args()

    meta = build_metadata()
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

    rsync_to_jupyter([csv_path, parquet_local, split_local])
    print("[bootstrap] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
