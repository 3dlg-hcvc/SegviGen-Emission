"""Merge one checkpoint's shard results into a single record.

Each shard of a checkpoint is an independent single-GPU array task that scored
its own stride of the validation split and wrote one json. This joins them.

It runs as a dependency job with afterANY, not afterok, so it still runs when a
shard dies. A missing shard is then reported as missing coverage rather than
quietly shrinking the denominator: the record carries how many shards were
planned, how many arrived, and which shapes went unscored, so a point on the
curve can never be read as a full evaluation when it was not one.

Usage:
    python scripts_segvigen/live_eval_merge.py --store <store> --step 4530 \
        [--n_shards 8] [--array_job 12345] [--started_at <unix seconds>]
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=None, help="how many shards were planned")
    ap.add_argument("--array_job", default=None)
    ap.add_argument("--started_at", type=float, default=None,
                    help="unix time the array was submitted, for honest wall time")
    ap.add_argument("--backfilled", action="store_true",
                    help="this checkpoint was skipped earlier and is being filled in late")
    ap.add_argument("--keep_shards", action="store_true")
    ap.add_argument("--wait_s", type=int, default=120,
                    help="wait this long for the expected shard files to appear before "
                         "merging. A shard writes its json seconds before its job ends, and "
                         "the shared filer does not always show it to this node instantly; "
                         "without the wait a merge can call a complete run partial")
    args = ap.parse_args()

    store = args.store
    shard_dir = os.path.join(store, "shards", f"step{args.step:07d}")
    rec_path = os.path.join(store, "records", f"step{args.step:07d}.json")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)

    if args.n_shards:
        deadline = time.time() + args.wait_s
        while time.time() < deadline:
            if len(glob.glob(os.path.join(shard_dir, "shard*.json"))) >= args.n_shards:
                break
            time.sleep(5)

    shards = []
    for p in sorted(glob.glob(os.path.join(shard_dir, "shard*.json"))):
        try:
            with open(p) as f:
                shards.append(json.load(f))
        except (ValueError, OSError) as e:
            print(f"[merge] could not read {p}: {e}", flush=True)
    planned = args.n_shards or (shards[0]["n_shards"] if shards else 0)
    arrived = sorted(s["shard"] for s in shards if s.get("shard") is not None)
    missing = [k for k in range(planned) if k not in arrived]

    per_shape, seen = [], set()
    for sh in shards:
        for row in sh["per_shape"]:
            if row["sid"] in seen:
                continue
            seen.add(row["sid"])
            per_shape.append(row)
    per_shape.sort(key=lambda r: r["sid"])
    scored = [r for r in per_shape if r["status"] == "ok"]

    n_total = shards[0].get("n_shapes_total") if shards else None
    # The selection number excludes shapes with no emissive ground truth: on those
    # IoU compares two empty sets and scores a perfect 1.0 whenever the model
    # paints nothing, which would pay a model for collapsing.
    emitting = [r for r in scored if r.get("gt_frac", 0) > 0]
    dark = [r for r in scored if r.get("gt_frac", 0) == 0]
    dark_draws = [f for r in dark for f in r["pred_frac_per_draw"]]

    per_shard = [{"shard": s.get("shard"), "n_scored": s.get("n_scored"),
                  "elapsed_s": s.get("elapsed_s"), "node": s.get("node"),
                  "slurm_job": s.get("slurm_job")} for s in shards]
    elapsed = [s["elapsed_s"] for s in shards if s.get("elapsed_s")]

    record = {
        "step": args.step,
        "status": "ok" if emitting else "failed",
        "backfilled": bool(args.backfilled),
        "val_loss": next((s.get("val_loss") for s in shards if s.get("val_loss") is not None), None),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "array_job": args.array_job,
        "n_shards": planned,
        "n_shards_completed": len(shards),
        "missing_shards": missing,
        "n_shapes_total": n_total,
        "n_scored": len(scored),
        "n_emitting": len(emitting),
        "n_dark_gt": len(dark),
        "draws": shards[0]["draws"] if shards else None,
        "threshold": shards[0].get("threshold") if shards else None,
        "resolution": shards[0].get("resolution") if shards else None,
        "mean_iou": float(np.mean([r["iou_mean"] for r in emitting])) if emitting else None,
        "mean_iou_all_shapes": float(np.mean([r["iou_mean"] for r in scored])) if scored else None,
        "median_iou": float(np.median([r["iou_mean"] for r in emitting])) if emitting else None,
        "dark_kept_dark": (float(sum(1 for f in dark_draws if f == 0.0) / len(dark_draws))
                           if dark_draws else None),
        "mean_pred_frac": float(np.mean([r["pred_frac_mean"] for r in scored])) if scored else None,
        "n_all_zero_draws": int(sum(1 for r in scored for f in r["pred_frac_per_draw"] if f == 0.0)),
        "n_all_on_draws": int(sum(1 for r in scored for f in r["pred_frac_per_draw"] if f == 1.0)),
        "shard_elapsed_s": per_shard,
        "slowest_shard_s": max(elapsed) if elapsed else None,
        "wall_s": (time.time() - args.started_at) if args.started_at else None,
        "per_shape": per_shape,
    }
    atomic_write_json(record, rec_path)

    print(f"[merge] step {args.step}: {len(shards)}/{planned} shards, {len(scored)} shapes scored "
          f"({len(emitting)} emitting, {len(dark)} with dark ground truth)", flush=True)
    if missing:
        print(f"[merge] MISSING shards {missing}; this point covers only part of the split and "
              "says so in the record", flush=True)
    print(f"[merge] IoU over emitting shapes {record['mean_iou']}, unsplit "
          f"{record['mean_iou_all_shapes']}", flush=True)
    if record["wall_s"] and record["slowest_shard_s"]:
        print(f"[merge] wall {record['wall_s'] / 60:.1f} min, slowest shard "
              f"{record['slowest_shard_s'] / 60:.1f} min", flush=True)
    print(f"[merge] wrote {rec_path}", flush=True)

    if not args.keep_shards and not missing:
        for p in glob.glob(os.path.join(shard_dir, "shard*.json")):
            os.remove(p)
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass

    bridged = os.path.join(store, "bridged", f"step{args.step:07d}.pt")
    if os.path.exists(bridged) and not missing:
        try:
            os.remove(bridged)
            print(f"[merge] removed the bridged checkpoint {bridged}", flush=True)
        except OSError as e:
            print(f"[merge] could not remove {bridged}: {e}", flush=True)


if __name__ == "__main__":
    main()
