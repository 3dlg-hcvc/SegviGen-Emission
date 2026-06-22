#!/usr/bin/env python3
"""Stage a split's emission .vxz into <data_root>/emission_voxels_<res>/ from the
part*_voxels.tar bundles.

The 74k fir data root ships raw voxels only as tars (each bundles
`pbr_voxels_256/` + `emission_voxels_256/`); training uses pre-encoded latents so
the voxels are never unpacked. The emission-DiT evaluator
(`predict_emission_voxels.py`) needs the raw emission `.vxz` to seed the decoder's
subdivision cache, so extract just the split we evaluate.

Streams each tar once (sequential read), extracting only `emission_voxels_<res>/`
members whose sha is in the split. Sha selection mirrors
`predict_emission_voxels.py` exactly (success==True rows, then split indices).
"""
from __future__ import annotations
import argparse, json, os, tarfile
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Holds df_SomgProc_final.parquet + the split json; dest is <data_root>/emission_voxels_<res>/")
    ap.add_argument("--tar_dir", required=True, help="Dir with part*_voxels.tar")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    split_json = args.split_json or os.path.join(args.data_root, "data_splits_emissive_74k_stratified_voxonly.json")
    parquet = args.parquet or os.path.join(args.data_root, "df_SomgProc_final.parquet")

    # sha selection — identical to predict_emission_voxels.py
    df = pd.read_parquet(parquet, columns=["success"])
    df = df[df["success"] == True] if "success" in df.columns else df
    shas = [str(x) for x in df.index]
    sp = json.load(open(split_json))[args.split]
    idx = sp["indices"] if isinstance(sp, dict) else sp
    want = {shas[i] for i in idx}
    print(f"want {len(want)} '{args.split}' shas", flush=True)

    prefix = f"emission_voxels_{args.res}/"
    dest_dir = os.path.join(args.data_root, f"emission_voxels_{args.res}")
    os.makedirs(dest_dir, exist_ok=True)
    already = {f[:-4] for f in os.listdir(dest_dir) if f.endswith(".vxz")}
    got = set(already & want)
    if got:
        print(f"  {len(got)} already present, skipping those", flush=True)

    tars = sorted(f for f in os.listdir(args.tar_dir) if f.endswith(".tar"))
    for t in tars:
        if len(got) == len(want):
            break
        with tarfile.open(os.path.join(args.tar_dir, t), "r") as tf:
            for m in tf:
                if not (m.name.startswith(prefix) and m.name.endswith(".vxz")):
                    continue
                sha = m.name[len(prefix):-4]
                if sha in want and sha not in got:
                    tf.extract(m, args.data_root)   # -> <data_root>/emission_voxels_<res>/<sha>.vxz
                    got.add(sha)
        print(f"  after {t}: {len(got)}/{len(want)}", flush=True)

    missing = want - got
    print(f"DONE: extracted {len(got)}/{len(want)} -> {dest_dir}", flush=True)
    if missing:
        print(f"  MISSING {len(missing)} (e.g. {list(missing)[:5]})", flush=True)


if __name__ == "__main__":
    main()
