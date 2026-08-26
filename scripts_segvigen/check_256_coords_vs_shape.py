"""Sanity check for the 256 target-latent build: our output tex latent's
coords must match Dongchen's shared shape latent's coords for the same sid
(both are downsampled from the same occupied 256^3 voxel set, so the sparse
latent grid they land on should be identical -- the dataset asserts this at
train time via get_common_coords; this catches a mismatch at BUILD time
instead, which is much cheaper to diagnose).

Usage:
    python scripts_segvigen/check_256_coords_vs_shape.py --split val_72k \\
        --world_size 32 --rank 0 [--limit 12]
"""
import argparse
import os

import numpy as np

DATASET_DIRECT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
SHAPE_DIR = ("/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2/shape_latents/"
             "shape_enc_next_dc_f16c32_fp16_256")
OUT_DIR = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/"
           "latents_256/output_tex_latents/segvigen_emissive_mask_256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = check every id in this rank's shard")
    args = ap.parse_args()

    sids = sorted(os.listdir(os.path.join(DATASET_DIRECT, args.split)))
    my_ids = [sid for i, sid in enumerate(sids) if i % args.world_size == args.rank]
    if args.limit:
        my_ids = my_ids[: args.limit]
    print(f"[check256] checking {len(my_ids)} sids")

    n_ok = n_bad = n_missing = 0
    for sid in my_ids:
        op = os.path.join(OUT_DIR, f"{sid}.npz")
        sp = os.path.join(SHAPE_DIR, f"{sid}.npz")
        if not (os.path.exists(op) and os.path.exists(sp)):
            n_missing += 1
            print(f"[check256] MISSING {sid}: out={os.path.exists(op)} shape={os.path.exists(sp)}")
            continue
        o = np.load(op)
        s = np.load(sp)
        ok = o["coords"].shape == s["coords"].shape and np.array_equal(o["coords"], s["coords"])
        print(f"[check256] {sid}: coords_match={ok} our_N={o['coords'].shape[0]} "
              f"shape_N={s['coords'].shape[0]} our_feats={o['feats'].shape}")
        n_ok += int(ok)
        n_bad += int(not ok)

    print(f"[check256] RESULT: {n_ok} match, {n_bad} mismatch, {n_missing} missing (of {len(my_ids)})")


if __name__ == "__main__":
    main()
