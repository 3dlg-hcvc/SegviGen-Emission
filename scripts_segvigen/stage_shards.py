"""Blackwell prep (job 5 amendment): tar-shard dataset_direct into slim
per-shard archives for node-local staging on venus-05/19 once their
Blackwell maintenance drain lifts. Does NOT touch venus-05/19 -- it only
reads the jupiter dataset and writes tars to the falas outputs tree; the
actual extraction to /localscratch/xya120/segvigen_data happens later, on
the target node, once it is safe to land a job there.

Per sample, keeps only what training actually reads: the three slat
tensors, the emission mask, the precomputed conditioning sidecar, and the
metadata json. Deliberately drops cond.pth (cond_pos.pth is half the bytes
and is what the dataset reader prefers -- see
trellis2/datasets/segvigen_direct.py) and both .vxz voxel files (not read
by the direct-slat training path at all). This roughly halves the
per-sample footprint versus a straight copy.

Requires cond_pos.pth to already exist for every sample it stages (built by
emissive/train/build_cond_pos.py, run separately over the full splits) --
refuses to fall back to cond.pth silently, since that would defeat the
point of the slim shard and silently double its size.

Usage:
    python scripts_segvigen/stage_shards.py \\
        --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \\
        --split train_72k_agentic \\
        --out_dir /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards \\
        --shard_idx 0 --num_shards 32
"""
import argparse
import json
import os
import tarfile

KEEP_FILES = [
    "input_tex_slat.pth",
    "output_tex_slat.pth",
    "shape_slat.pth",
    "emis_mask.pth",
    "cond_pos.pth",
    "meta.json",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset_direct root")
    ap.add_argument("--split", required=True, help="split name under --dataset")
    ap.add_argument("--out_dir", required=True, help="output dir for shard tars")
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--require_cond_pos", action="store_true", default=True)
    ap.add_argument("--allow_missing_cond_pos", dest="require_cond_pos", action="store_false",
                     help="fall back to cond.pth for samples missing cond_pos.pth (defeats the size "
                          "win for those samples; only use if job 7 has not finished yet)")
    args = ap.parse_args()

    split_dir = os.path.join(args.dataset, args.split)
    sample_ids = sorted(os.listdir(split_dir))
    my_ids = [sid for i, sid in enumerate(sample_ids) if i % args.num_shards == args.shard_idx]

    os.makedirs(os.path.join(args.out_dir, args.split), exist_ok=True)
    tar_path = os.path.join(args.out_dir, args.split, f"shard_{args.shard_idx:04d}.tar")
    tmp_path = tar_path + f".tmp{os.getpid()}"

    n_written = n_missing_cond_pos = n_skipped_missing_file = 0
    with tarfile.open(tmp_path, "w") as tf:
        for sid in my_ids:
            sdir = os.path.join(split_dir, sid)
            files_to_add = list(KEEP_FILES)
            cond_pos_path = os.path.join(sdir, "cond_pos.pth")
            if not os.path.exists(cond_pos_path):
                n_missing_cond_pos += 1
                if args.require_cond_pos:
                    print(f"[stage_shards] SKIP {sid}: no cond_pos.pth (job 7 incomplete for this sample)")
                    continue
                files_to_add = [f for f in files_to_add if f != "cond_pos.pth"] + ["cond.pth"]

            ok = True
            for fname in files_to_add:
                fpath = os.path.join(sdir, fname)
                if not os.path.exists(fpath):
                    print(f"[stage_shards] SKIP {sid}: missing {fname}")
                    ok = False
                    break
            if not ok:
                n_skipped_missing_file += 1
                continue

            for fname in files_to_add:
                tf.add(os.path.join(sdir, fname), arcname=os.path.join(sid, fname))
            n_written += 1

    os.replace(tmp_path, tar_path)
    size_mb = os.path.getsize(tar_path) / 1e6
    print(f"[stage_shards] {args.split} shard {args.shard_idx}/{args.num_shards}: "
          f"{n_written} samples written, {n_missing_cond_pos} missing cond_pos.pth, "
          f"{n_skipped_missing_file} skipped for another missing file "
          f"-> {tar_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
