"""Index files of the released dataset (3dlg-hcvc/LightgenBench) for SegviGenEmissive256.

The dataset class reads a parquet indexed by shape id and a split json of row positions
(trellis2/datasets/segvigen_256.py, _build_parquet_samples). The release ships
splits.json = {"train": [uuid, ...], "val": [...], "test": [...]}; this writes

    <out>/release_index.parquet   one row per shape, index = uuid, column `split`
    <out>/release_splits.json     {"train": [row, ...], "val": [...], "test": [...]}
    <out>/release_<split>.txt     the uuids, one per line
    <out>/release_all.txt

and, with --link_voxels, the flat symlink trees the paper's encoders read:

    <work>/pbr_voxels_256/<uuid>.vxz      -> <release>/<uuid>/pbr_voxels.vxz
    <work>/emission_voxels_256/<uuid>.vxz -> <release>/<uuid>/emission_voxels.vxz

Row order is train, val, test, each in the order of splits.json (sorted there).

Usage:
    python scripts_segvigen/newdata/build_metadata.py \
        --release /3dlg-jupiter-project/lightgen/datasets/dataset_release \
        --out /3dlg-jupiter-project/lightgen/datasets/dataset_release_latents/meta \
        --link_voxels /3dlg-jupiter-project/lightgen/datasets/dataset_release_latents
"""
import argparse
import json
import os

import pandas as pd

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="dataset_release root (holds splits.json and <uuid>/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--link_voxels", default=None, help="work root that gets the two flat symlink trees")
    ap.add_argument("--link_release", default=None,
                    help="release root as the links should spell it (default: --release). The workstation "
                         "sees jupiter under /cs/..., Solar without the prefix; links made on one dangle on the other")
    args = ap.parse_args()

    with open(os.path.join(args.release, "splits.json")) as f:
        splits = json.load(f)
    assert set(splits) == set(SPLITS), f"unexpected split keys: {sorted(splits)}"
    uuids, split_of = [], []
    for s in SPLITS:
        uuids += splits[s]
        split_of += [s] * len(splits[s])
    assert len(uuids) == len(set(uuids)), "a uuid sits in two splits"

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame({"split": split_of}, index=pd.Index(uuids, name="sha256"))
    df.to_parquet(os.path.join(args.out, "release_index.parquet"))
    rows = {s: [i for i, t in enumerate(split_of) if t == s] for s in SPLITS}
    with open(os.path.join(args.out, "release_splits.json"), "w") as f:
        json.dump(rows, f)
    for s in SPLITS:
        with open(os.path.join(args.out, f"release_{s}.txt"), "w") as f:
            f.write("\n".join(splits[s]) + "\n")
    with open(os.path.join(args.out, "release_all.txt"), "w") as f:
        f.write("\n".join(uuids) + "\n")
    print(f"[metadata] {len(uuids)} shapes: " + ", ".join(f"{s} {len(rows[s])}" for s in SPLITS))

    if args.link_voxels:
        for kind, src in (("pbr_voxels_256", "pbr_voxels.vxz"), ("emission_voxels_256", "emission_voxels.vxz")):
            d = os.path.join(args.link_voxels, kind)
            os.makedirs(d, exist_ok=True)
            n = 0
            for u in uuids:
                dst = os.path.join(d, f"{u}.vxz")
                if not os.path.islink(dst):
                    os.symlink(os.path.join(args.link_release or args.release, u, src), dst)
                    n += 1
            print(f"[metadata] {kind}: {n} new links under {d}")


if __name__ == "__main__":
    main()
