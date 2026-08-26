"""Relocate + halve the precomputed DINOv3 conditioning tensors: jupiter fp32
cond_pos.pth -> falas fp16 cond_pos.pth, one flat directory keyed by sid so
the existing cond_search_dirs mechanism (SegviGenEmissiveDirect,
SegviGenEmissive256) works unchanged, just with a new search dir prepended.

Root cause (board lightning_port, 2026-08-26, team-lead's measurement):
training at ~6s/step with GPUs ~47% idle, gated on data; 90%+ of the bytes
are cond_pos.pth (4.2MB/sample) served from jupiter, whose NFS read ceiling
(~60MB/s server-wide) is saturated by the 6-GPU job's dataloaders (13MB/s
per-stream in a 16-way benchmark vs falas's 200 npz in 65ms in the same
test). Fix: move cond onto falas, and halve it (fp32->fp16) since the
reader already upcasts with .float() before use -- AMP casts to bf16 for
compute anyway, so fp16 storage loses nothing the training run would keep.

Contract (verified against trellis2/datasets/segvigen_direct.py and
segvigen_256.py's get_instance, both handle it identically):
    cond_pos.pth is EITHER a bare tensor OR a dict with key "cond".
    Fallback when cond_pos.pth is absent: cond.pth, always a dict with key
    "cond" (the reader does not squeeze/cast anything else out of it here).
This script preserves whichever shape it finds (bare tensor -> bare tensor,
dict -> dict with the same key), casting only the floating tensor's dtype
to float16. No other tensor structure is invented; the reader's own
cond.float().squeeze(0) call handles the fp16->fp32 upcast unchanged, so no
reader patch should be needed (verify this in the check step).

Idempotent: skips any sid whose destination file already exists.

Usage (one array task):
    python scripts_segvigen/convert_cond_fp16.py --rank 0 --world_size 16
"""
import argparse
import os
import time

import torch

DATASET_DIRECT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
SPLITS = ["train_72k_agentic", "val_72k"]
DEST_ROOT = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_fp16"


def _to_fp16(x):
    if torch.is_tensor(x):
        return x.half()
    return x


def convert_one(sid: str, src_dir: str) -> str:
    dest_dir = os.path.join(DEST_ROOT, sid)
    dest_path = os.path.join(dest_dir, "cond_pos.pth")
    if os.path.exists(dest_path):
        return "cached"

    cp = os.path.join(src_dir, "cond_pos.pth")
    cf = os.path.join(src_dir, "cond.pth")
    if os.path.exists(cp):
        obj = torch.load(cp, map_location="cpu")
    elif os.path.exists(cf):
        obj = torch.load(cf, map_location="cpu")
    else:
        return "missing_source"

    if isinstance(obj, dict):
        obj = {k: _to_fp16(v) for k, v in obj.items()}
    else:
        obj = _to_fp16(obj)

    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = dest_path + f".tmp{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, dest_path)  # atomic: a killed job never leaves a half-written file
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(DEST_ROOT, exist_ok=True)

    all_items = []
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIRECT, split)
        for sid in sorted(os.listdir(split_dir)):
            all_items.append((sid, os.path.join(split_dir, sid)))
    my_items = [it for i, it in enumerate(all_items) if i % args.world_size == args.rank]
    print(f"[convert_cond_fp16] rank {args.rank}/{args.world_size}: "
          f"{len(my_items)} of {len(all_items)} sids (both splits combined)")

    t0 = time.time()
    counts = {"built": 0, "cached": 0, "missing_source": 0}
    for i, (sid, src_dir) in enumerate(my_items):
        try:
            result = convert_one(sid, src_dir)
            counts[result] += 1
        except Exception as exc:
            print(f"[convert_cond_fp16] ERROR {sid}: {exc!r}")
            counts["missing_source"] += 1
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"[convert_cond_fp16] rank {args.rank}: {i + 1}/{len(my_items)} "
                  f"({counts}) {dt:.0f}s elapsed")
    print(f"[convert_cond_fp16] rank {args.rank} DONE: {counts} of {len(my_items)} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
