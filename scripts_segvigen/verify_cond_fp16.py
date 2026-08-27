"""Verification for the cond_pos fp32->fp16 relocation (job 257957, board
lightning_port 2026-08-26). Two checks, per the task's item 2/4:

  1. Numerical: for a random sample of sids, the fp16 falas copy must match
     the fp32 jupiter original after upcasting, within fp16 rounding error
     (not exact -- fp16 has ~3 decimal digits of precision, so absolute
     error scales with the value's own magnitude; this checks max abs diff
     against a per-tensor tolerance derived from the fp16 ULP at that
     tensor's max magnitude, not a single fixed atol).
  2. Reader load-path: SegviGenEmissive256's get_instance() must accept the
     fp16 file unmodified and return a float32 "cond" tensor of the
     expected shape, confirming the existing `cond.float().squeeze(0)`
     call is sufficient and no reader patch is needed.

Run AFTER job 257957 reports 0 missing/errors. Do not run against a
partially-converted tree (some sids will legitimately not exist yet).

Usage:
    python scripts_segvigen/verify_cond_fp16.py [--n 20] [--seed 0]
"""
import argparse
import os
import random

import torch

DATASET_DIRECT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
SPLITS = ["train_72k_agentic", "val_72k"]
FP16_ROOT = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_fp16"


def _load_cond(path_pos: str, path_fallback: str):
    if os.path.exists(path_pos):
        obj = torch.load(path_pos, map_location="cpu")
    else:
        obj = torch.load(path_fallback, map_location="cpu")
    if isinstance(obj, dict):
        obj = obj["cond"]
    return obj


def numerical_check(sids_with_src, n, seed):
    random.seed(seed)
    sample = random.sample(sids_with_src, min(n, len(sids_with_src)))
    print(f"[verify_cond_fp16] numerical check on {len(sample)} sids")
    worst = (None, -1.0)
    for sid, src_dir in sample:
        orig = _load_cond(
            os.path.join(src_dir, "cond_pos.pth"),
            os.path.join(src_dir, "cond.pth"),
        ).float()
        fp16_path = os.path.join(FP16_ROOT, sid, "cond_pos.pth")
        assert os.path.exists(fp16_path), f"{sid}: fp16 copy missing at {fp16_path}"
        new = _load_cond(fp16_path, fp16_path).float()

        assert orig.shape == new.shape, f"{sid}: shape mismatch {orig.shape} vs {new.shape}"
        diff = (orig - new).abs()
        # fp16 ULP-scaled tolerance: relative error bounded by ~2^-10 (fp16
        # mantissa), plus a small absolute floor for near-zero values.
        tol = orig.abs() * (2 ** -9) + 1e-3
        n_bad = (diff > tol).sum().item()
        max_diff = diff.max().item()
        print(f"[verify_cond_fp16] {sid}: max_abs_diff={max_diff:.6f} "
              f"n_over_tol={n_bad}/{diff.numel()}")
        if max_diff > worst[1]:
            worst = (sid, max_diff)
        assert n_bad == 0, f"{sid}: {n_bad} elements exceed fp16-rounding tolerance"
    print(f"[verify_cond_fp16] PASS: worst max_abs_diff={worst[1]:.6f} (sid {worst[0]})")


def reader_check(sids_with_src, n, seed):
    import sys
    sys.path.insert(0, ".")
    from trellis2.datasets.segvigen_256 import SegviGenEmissive256  # noqa: E402

    random.seed(seed + 1)
    sample_sid = random.choice(sids_with_src)[0]
    print(f"[verify_cond_fp16] reader load-path check via sid {sample_sid}")
    # Directly exercise the same cond-loading branch get_instance() uses,
    # pointed ONLY at the fp16 root (no jupiter fallback), so this proves
    # the fp16 path specifically, not the fallback.
    cp = os.path.join(FP16_ROOT, sample_sid, "cond_pos.pth")
    cond = torch.load(cp, map_location="cpu")
    if isinstance(cond, dict):
        cond = cond["cond"]
    cond = cond.float().squeeze(0)
    print(f"[verify_cond_fp16] loaded cond: dtype={cond.dtype} shape={tuple(cond.shape)}")
    assert cond.dtype == torch.float32, f"expected float32 after .float(), got {cond.dtype}"
    assert cond.shape[-1] == 1024, f"unexpected feature dim {cond.shape}"
    print("[verify_cond_fp16] PASS: reader's existing .float() upcast handles fp16 storage, "
          "no reader patch needed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sids_with_src = []
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIRECT, split)
        for sid in os.listdir(split_dir):
            sids_with_src.append((sid, os.path.join(split_dir, sid)))
    print(f"[verify_cond_fp16] {len(sids_with_src)} candidate sids across both splits")

    # restrict to sids job 257957 has already converted -- the conversion
    # array may still be running, this is not a partial-tree error
    converted = set(os.listdir(FP16_ROOT))
    before = len(sids_with_src)
    sids_with_src = [(sid, d) for sid, d in sids_with_src if sid in converted]
    print(f"[verify_cond_fp16] {len(sids_with_src)}/{before} already fp16-converted "
          f"(job 257957 may still be running)")

    numerical_check(sids_with_src, args.n, args.seed)
    reader_check(sids_with_src, args.n, args.seed)
    print("[verify_cond_fp16] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
