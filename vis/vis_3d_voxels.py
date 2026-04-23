#!/usr/bin/env python3
"""Compare N sets of emission voxel files side-by-side with an optional GLB panel.

Usage:
  python vis/vis_3d_voxels.py \\
      --voxel before=data/lightgen_test/emission_voxels_256 \\
      --voxel after=/tmp/dump_verify/emission_voxels_256_fixed \\
      --samples sha:0007deb6d96c4474b80faa5aa3888760,000b9fd47d6d4f7db7b2f5022d1ae9aa \\
      --glb_root /localhome/dya78/code/lightgen/data/TexVerse-1K \\
      --roundtrip_glb_dir debug_outputs/roundtrip_glbs \\
      --dataset data/lightgen_test/df_10samples.parquet \\
      --out visualizations/voxel_compare.html

Each --voxel NAME=DIR supplies a directory containing <sha>.vxz emission voxels.
For each sample, a row of panels is emitted: [GLB] [voxel source 1] [voxel source 2] ...

This script does not load any model; it is a pure data comparison. Use it to
diagnose pipeline changes that affect voxel geometry (normalization fixes,
different grid resolutions, different dump scripts, etc.).
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vis.vis_utils import (  # noqa: E402
    load_samples, make_glb_panel_data, make_voxel_panel_data,
    read_emission_voxel, render_html,
)


def _parse_voxel_sources(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--voxel expects NAME=DIR, got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def _stage_glb(glb_path: str, assets_dir: str) -> str:
    os.makedirs(assets_dir, exist_ok=True)
    basename = os.path.basename(glb_path)
    link_path = os.path.join(assets_dir, basename)
    if not os.path.lexists(link_path):
        os.symlink(os.path.abspath(glb_path), link_path)
    parent = os.path.dirname(assets_dir)
    assets_root = os.path.basename(parent)
    return f"{assets_root}/{os.path.basename(assets_dir)}/{basename}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", action="append", required=True,
                    help="repeatable NAME=DIR where DIR holds <sha>.vxz emission voxels")
    ap.add_argument("--samples", default="all",
                    help="all | first:N | split:<name> | sha:<csv>")
    ap.add_argument("--dataset", default=None,
                    help="parquet or splits JSON; needed for anything except --samples sha:...")
    ap.add_argument("--data_root", default=None,
                    help="override; default = parent(--dataset)")
    ap.add_argument("--glb_root", default=None,
                    help="root for parquet's glb_1k_path column")
    ap.add_argument("--roundtrip_glb_dir", default=None,
                    help="prefer <sha[:8]>_roundtrip.glb from this dir over --glb_root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--point_size", type=float, default=0.012)
    args = ap.parse_args()

    sources = _parse_voxel_sources(args.voxel)

    if args.samples.startswith("sha:"):
        data_root = args.data_root or (
            os.path.dirname(os.path.abspath(args.dataset)) if args.dataset else None
        )
        if data_root is None:
            data_root = os.path.dirname(os.path.abspath(sources[0][1]))
        shas = [s.strip() for s in args.samples[4:].split(",") if s.strip()]

        glb_map = {}
        if args.dataset and args.dataset.endswith(".parquet") and args.glb_root:
            import pandas as pd
            df = pd.read_parquet(args.dataset)
            if df.index.name != "sha256" and "sha256" in df.columns:
                df = df.set_index("sha256")
            df.index = df.index.astype(str)
            glb_map = df["glb_1k_path"].to_dict() if "glb_1k_path" in df.columns else {}

        class _R:
            pass
        records = []
        for sha in shas:
            r = _R()
            r.sha = sha
            r.glb_path = (
                os.path.join(args.glb_root, glb_map[sha])
                if (args.glb_root and sha in glb_map) else None
            )
            records.append(r)
    else:
        if args.dataset is None:
            raise SystemExit("--dataset is required unless --samples sha:...")
        data_root = args.data_root or os.path.dirname(os.path.abspath(args.dataset))
        records = load_samples(
            data_root=data_root,
            resolution=256,
            source=args.dataset,
            select=args.samples,
            glb_root=args.glb_root,
            require=(),
        )

    if not records:
        raise SystemExit(f"no samples matched --samples={args.samples!r}")

    out_abs = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_abs)
    glb_assets_dir = os.path.join(out_dir, "assets", "glbs")

    samples_out: list[dict] = []
    for rec in records:
        # Resolve the GLB (roundtrip preferred over 1K).
        chosen_glb = None
        if args.roundtrip_glb_dir:
            rt_path = os.path.join(args.roundtrip_glb_dir, f"{rec.sha[:8]}_roundtrip.glb")
            if os.path.exists(rt_path):
                chosen_glb = rt_path
        if chosen_glb is None and rec.glb_path and os.path.exists(rec.glb_path):
            chosen_glb = rec.glb_path
        glb_url = _stage_glb(chosen_glb, glb_assets_dir) if chosen_glb else None

        panels: list[dict] = [{
            "kind": "glb",
            "data": make_glb_panel_data(glb_url, z_up_to_y_up=False),
            "caption": "",
            "label": "GLB",
        }]

        for label, vxz_dir in sources:
            vxz_path = os.path.join(vxz_dir, f"{rec.sha}.vxz")
            if not os.path.exists(vxz_path):
                panels.append({
                    "kind": "voxel",
                    "data": {"xyz": [], "rgb": [], "n": 0, "res": 256},
                    "caption": "(missing)",
                    "label": label,
                })
                continue
            coords, feats, res = read_emission_voxel(vxz_path)
            cmin = coords.min(dim=0).values.tolist()
            cmax = coords.max(dim=0).values.tolist()
            panels.append({
                "kind": "voxel",
                "data": make_voxel_panel_data(
                    coords, feats, res, args.threshold, args.max_points,
                ),
                "caption": (
                    f"{coords.shape[0]:,} voxels  "
                    f"bbox [{cmin[0]},{cmin[1]},{cmin[2]}]→[{cmax[0]},{cmax[1]},{cmax[2]}]"
                ),
                "label": label,
            })

        samples_out.append({
            "sha": rec.sha,
            "info": f"{len(sources)} voxel source(s)",
            "panels": panels,
        })

    render_html(
        samples=samples_out,
        title=f"Voxel comparison — {len(sources)} source(s)",
        out_path=args.out,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
