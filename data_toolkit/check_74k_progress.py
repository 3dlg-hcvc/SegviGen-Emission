#!/usr/bin/env python3
"""Read each per-shard tar on jupyter, count members per artifact dir, and
report per-shard completion vs the expected sha list from metadata.csv.

Run from any host that can see the jupyter mount (`/3dlg-jupiter-project/...`
on star2; `/cs/3dlg-jupiter-project/...` on cs-3dlg-23 and other CS workstations).

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


def _resolve_jupyter_base() -> Path:
    candidates = [
        Path("/3dlg-jupiter-project/lightgen/dataset/lightgen_74k"),
        Path("/cs/3dlg-jupiter-project/lightgen/dataset/lightgen_74k"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(
        "Could not find the jupyter mount. Tried:\n  " +
        "\n  ".join(str(c) for c in candidates))


JUP = _resolve_jupyter_base()
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
    print(f"[check_74k_progress] jupyter base: {JUP}")
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
