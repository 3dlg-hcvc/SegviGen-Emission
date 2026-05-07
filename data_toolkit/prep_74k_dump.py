#!/usr/bin/env python3
"""Stage A — Blender PBR dump for the 74k encoding pipeline.

Reads <root>/metadata.csv, takes the rank-N slice, and runs:
    blender --background --python blender_script/dump_pbr.py -- \\
        --object <glb> --output_path <root>/pbr_dumps/<sha>.pickle
in parallel via ProcessPoolExecutor. Skip-if-exists: pickles already in
pbr_dumps/ are not re-dumped. Failures are appended to
pbr_dumps/new_records/part_<rank>_failures.txt; the script exits 0 even on
per-sample failures (the caller decides whether to retry).

Usage (typical, inside an sbatch body):
    python data_toolkit/prep_74k_dump.py \\
        --root /localscratch/dya78/lightgen_74k_part0 \\
        --rank 0 --world_size 8 --workers 8 \\
        --blender /localscratch/dya78/blender-4.5.1-linux-x64/blender
"""
from __future__ import annotations
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


GLB_ROOT = Path("/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K")
DUMP_SCRIPT = (Path(__file__).parent / "blender_script" / "dump_pbr.py").resolve()


def _blender_dump(blender: str, glb_path: str, pickle_path: str) -> tuple[bool, str]:
    cmd = [blender, "--background", "--python", str(DUMP_SCRIPT), "--",
           "--object", glb_path, "--output_path", pickle_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "blender timeout (600s)"
    if res.returncode != 0 or not os.path.exists(pickle_path):
        return False, (res.stderr or res.stdout)[-500:].strip()
    return True, ""


def _process_one(args: tuple[str, str, str, str]) -> tuple[str, bool, str]:
    sha, glb_rel, root, blender = args
    glb_path = str(GLB_ROOT / glb_rel)
    out_path = os.path.join(root, "pbr_dumps", f"{sha}.pickle")
    if os.path.exists(out_path):
        return sha, True, "skip"
    if not os.path.exists(glb_path):
        return sha, False, f"glb_missing: {glb_path}"

    with tempfile.TemporaryDirectory(dir=os.path.join(root, "pbr_dumps")) as tmp:
        tmp_pickle = os.path.join(tmp, f"{sha}.pickle")
        ok, err = _blender_dump(blender, glb_path, tmp_pickle)
        if not ok:
            return sha, False, err
        shutil.move(tmp_pickle, out_path)
    return sha, True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blender", type=str,
                    default="/localscratch/dya78/blender-4.5.1-linux-x64/blender")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N samples (smoke test)")
    args = ap.parse_args()

    if not os.path.exists(args.blender):
        print(f"[prep_74k_dump] FATAL: blender not found at {args.blender}", file=sys.stderr)
        return 2

    csv_path = os.path.join(args.root, "metadata.csv")
    df = pd.read_csv(csv_path)
    print(f"[prep_74k_dump] metadata rows: {len(df)}  rank={args.rank}/{args.world_size}")

    start = len(df) * args.rank // args.world_size
    end   = len(df) * (args.rank + 1) // args.world_size
    df_rank = df.iloc[start:end]
    if args.limit is not None:
        df_rank = df_rank.head(args.limit)
    print(f"[prep_74k_dump] this rank: {len(df_rank)} samples (slice {start}..{end})")

    out_dir = os.path.join(args.root, "pbr_dumps")
    os.makedirs(os.path.join(out_dir, "new_records"), exist_ok=True)

    work = [(r["sha256"], r["glb_1k_path"], args.root, args.blender)
            for _, r in df_rank.iterrows()]

    t0 = time.time()
    success = skip = fail = 0
    fail_log: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            sha, ok, msg = fut.result()
            if ok and msg == "skip":
                skip += 1
            elif ok:
                success += 1
            else:
                fail += 1
                fail_log.append((sha, msg))
            if i % 25 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1e-6)
                eta_h = (len(work) - i) / max(rate, 1e-6) / 3600
                print(f"[{i:5d}/{len(work)}] ok={success} skip={skip} fail={fail}  "
                      f"rate={rate:.2f}/s  eta={eta_h:.2f}h", flush=True)

    fail_path = os.path.join(out_dir, "new_records", f"part_{args.rank}_failures.txt")
    with open(fail_path, "w") as f:
        for sha, msg in fail_log:
            f.write(f"{sha}\t{msg}\n")

    # Write a pbr_dumps/metadata.csv with `pbr_dumped` flags so the downstream
    # voxelize_pbr.py merge sees this column. Existing rows (from prior ranks)
    # are preserved by writing a per-rank fragment that voxelize_pbr.py merges.
    fragment_path = os.path.join(out_dir, "new_records", f"part_{args.rank}.csv")
    rows = []
    for _, r in df_rank.iterrows():
        sha = r["sha256"]
        pickle_path = os.path.join(out_dir, f"{sha}.pickle")
        rows.append({"sha256": sha, "pbr_dumped": os.path.exists(pickle_path)})
    pd.DataFrame.from_records(rows).to_csv(fragment_path, index=False)

    # Also concatenate any existing fragments into pbr_dumps/metadata.csv so
    # voxelize_pbr.py's `metadata.combine_first` finds it. Idempotent: re-runs
    # rebuild from fragments. World-size>1 callers can re-run after all ranks
    # finish to produce a fully merged metadata.csv.
    frags = sorted(glob.glob(os.path.join(out_dir, "new_records", "part_*.csv")))
    if frags:
        merged = pd.concat([pd.read_csv(p) for p in frags], ignore_index=True)
        merged = merged.drop_duplicates(subset="sha256", keep="last")
        merged.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    print(f"[prep_74k_dump] done: ok={success} skip={skip} fail={fail} "
          f"in {(time.time()-t0)/60:.1f} min  failures -> {fail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
