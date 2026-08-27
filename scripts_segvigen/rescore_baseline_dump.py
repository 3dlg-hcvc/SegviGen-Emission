"""Rescore a saved five-draw prediction dump under the live loop's own metric.

Why this exists: every reference number this project quotes was computed before
the emissive-slice convention existed, and most of them survive only as
aggregates. A checkpoint whose per-shape predictions were saved can be rescored
exactly, under the same rule the live curve uses, so the baseline line and the
curve cannot mix conventions. A checkpoint whose per-shape data is gone cannot,
and is marked not comparable rather than plotted.

The dump format is dump_pred_voxels_eval5.py's: one directory per draw, each
holding {sid}.npz with coords, pred (the predicted emission RGB) and gt_recon
(the same shape's ground-truth emission decoded through the same VAE). IoU is
computed exactly as live_eval_one.py computes it: brightness mean over RGB,
threshold 0.5, per draw, then averaged over draws.

Usage:
    python scripts_segvigen/rescore_baseline_dump.py \
        --dump /path/to/eval5_samples_ep8 --label "warm start, epoch 8" \
        --out /path/to/live_eval/baselines/ep8_warm.json
"""
import argparse
import json
import os
import time

import numpy as np

THR = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()

    draw_dirs = [os.path.join(args.dump, f"draw{k}") for k in range(args.draws)]
    missing = [d for d in draw_dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"missing draw directories: {missing}")
    sids = sorted(f[:-4] for f in os.listdir(draw_dirs[0]) if f.endswith(".npz"))
    print(f"[rescore] {len(sids)} shapes x {args.draws} draws from {args.dump}", flush=True)

    per_shape, t0 = [], time.time()
    for i, sid in enumerate(sids):
        ious, fracs, gt_frac, ok = [], [], None, True
        for d in draw_dirs:
            p = os.path.join(d, f"{sid}.npz")
            if not os.path.exists(p):
                ok = False
                break
            z = np.load(p)
            gt = z["gt_recon"].astype(np.float32).mean(-1) > THR
            pr = z["pred"].astype(np.float32).mean(-1) > THR
            inter = int((gt & pr).sum())
            union = int((gt | pr).sum())
            # Same convention as every other script here: an empty union scores
            # 1.0. That is exactly the artifact the slices exist to isolate, so
            # it is reproduced rather than quietly patched.
            ious.append(inter / union if union > 0 else 1.0)
            fracs.append(float(pr.mean()))
            gt_frac = float(gt.mean())
        if not ok:
            per_shape.append({"sid": sid, "status": "skipped", "reason": "a draw is missing"})
            continue
        per_shape.append({
            "sid": sid, "status": "ok", "gt_frac": gt_frac,
            "iou_per_draw": [float(v) for v in ious],
            "iou_mean": float(np.mean(ious)), "iou_std": float(np.std(ious)),
            "pred_frac_per_draw": fracs, "pred_frac_mean": float(np.mean(fracs)),
        })
        if (i + 1) % 50 == 0:
            print(f"[rescore] {i + 1}/{len(sids)} shapes, {time.time() - t0:.0f}s", flush=True)

    scored = [s for s in per_shape if s["status"] == "ok"]
    rec = {
        "label": args.label, "kind": "baseline", "dump": args.dump,
        "threshold": THR, "draws": args.draws,
        "rescored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_scored": len(scored),
        "mean_iou_all_shapes": float(np.mean([s["iou_mean"] for s in scored])) if scored else None,
        "per_shape": per_shape,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, args.out)
    print(f"[rescore] {len(scored)} shapes, unsplit mean {rec['mean_iou_all_shapes']:.4f}, "
          f"{time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
