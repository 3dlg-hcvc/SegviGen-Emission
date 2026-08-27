"""Relocate + halve the precomputed DINOv3 conditioning tensors: jupiter fp32
cond_pos.pth -> falas 16-bit cond_pos.pth, one flat directory keyed by sid so
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

DTYPE (owner ruling 2026-08-26, after measurement -- read before changing):
The DEFAULT IS NOW bfloat16, and any FUTURE conversion should use it.
Reason: the trainer runs mix_precision_mode=amp / mix_precision_dtype=
bfloat16, so autocast casts cond to bf16 inside every forward regardless of
how we store it. Storing bf16 is therefore BIT-IDENTICAL to storing fp32,
at the same 2.1MB/shape as fp16 -- measured, not assumed (job 258059, 20
shapes): predicted-velocity cosine(fp32, bf16) = 1.00000000 and
|loss_fp32 - loss_bf16| = 0.0 EXACTLY on every shape.

float16 stays selectable via --dtype float16 because THE CURRENTLY STAGED
SET IS fp16: /3dlg-falas/.../outputs/cond_fp16 (36,613 sids, built by job
257957, staged to venus-05 local disk) was written before this measurement
existed. The owner ruled to KEEP it rather than spend ~1h20m reconverting
plus a re-stage, because the measured fp16 effect is immaterial:
    stored tensors   max abs err 0.0078-0.0081, 0 of 1,053,696 elements
                     per shape outside fp16-ULP tolerance (job 258014)
    forward pass     cosine(fp32, fp16) mean 0.99989, min 0.99904;
                     |loss_fp32 - loss_fp16| mean 1.4e-4, max 3.6e-4 on a
                     loss of ~1.02, i.e. 0.014% relative (job 258059)
For scale, in that same run loss varies 0.057 BETWEEN SHAPES and the live
6-GPU training's step-to-step loss swings 0.28-0.47, so the fp16 effect is
~160x below shape-to-shape spread and ~1000x below minibatch gradient
noise. The fp16 residual is a double-rounding artifact: fp16 keeps 11
mantissa bits vs bf16's 8, so fp32->fp16->bf16 differs from fp32->bf16 only
where fp16 rounding crosses a bf16 boundary.

The directory name "cond_fp16" is kept as-is for the existing set so no
config/reader path has to change; it names that artifact, not this
script's default. When writing a NEW bf16 set, pass --dest_root to a new
directory rather than mixing dtypes inside one tree.

Idempotent: skips any sid whose destination file already exists. NOTE this
means a dtype change will NOT rewrite existing files -- point --dest_root
at a fresh directory instead.

Usage (one array task):
    # future conversions (bf16, the default)
    python scripts_segvigen/convert_cond_fp16.py --rank 0 --world_size 16 \
        --dest_root /3dlg-falas/.../outputs/cond_bf16
    # reproduce the currently staged set
    python scripts_segvigen/convert_cond_fp16.py --rank 0 --world_size 16 \
        --dtype float16
"""
import argparse
import os
import time

import torch

DATASET_DIRECT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
SPLITS = ["train_72k_agentic", "val_72k"]
DEST_ROOT = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_fp16"


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
DEFAULT_DTYPE = "bfloat16"   # see the DTYPE note in the module docstring


def _cast(x, dtype):
    if torch.is_tensor(x) and x.is_floating_point():
        return x.to(dtype)
    return x


def convert_one(sid: str, src_dir: str, dtype=torch.bfloat16, dest_root=None) -> str:
    dest_root = dest_root or DEST_ROOT
    dest_dir = os.path.join(dest_root, sid)
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
        obj = {k: _cast(v, dtype) for k, v in obj.items()}
    else:
        obj = _cast(obj, dtype)

    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = dest_path + f".tmp{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, dest_path)  # atomic: a killed job never leaves a half-written file
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default=DEFAULT_DTYPE,
                    help="storage dtype (default bfloat16: bit-identical under the "
                         "trainer's bf16 autocast; float16 kept selectable, see docstring)")
    ap.add_argument("--dest_root", default=DEST_ROOT,
                    help="output root (override when writing a NEW set so an existing "
                         "one is not silently mixed with a different dtype)")
    args = ap.parse_args()
    dtype = DTYPES[args.dtype]
    dest_root = args.dest_root
    print(f"[convert_cond_fp16] dtype={args.dtype} dest_root={dest_root}")

    os.makedirs(dest_root, exist_ok=True)

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
            result = convert_one(sid, src_dir, dtype=dtype, dest_root=dest_root)
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
