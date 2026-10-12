"""Compare two mask dump dirs (`<dir>/<sid>.npz`, keys `coords` int16 [N, 3], `pred` [N, 3]) with the
bars of the 2026-09-16 design for a new inference path (refactor step 10): per shape, with the mask
`max(pred) > 1/255`, on = mask voxels, m = max(on_a, on_b):
  large masks (m >= 100): IoU >= 0.95 and |on_a - on_b| <= 0.05 m
  small masks (m < 100):  at most 5 mask voxels differ (|A xor B| <= 5)
  every shape:            coordinate sets differ by at most 0.5 % of their union
With --identical the bar is byte identity instead. A non-finite `pred` fails the shape. Also reports byte
identity and the mean |d| of `pred` on the shared voxels. An empty or repeated shapes file is refused. Exit 0
when every shape passes, 1 otherwise, 2 when refused.

    python compare_masks.py --a <reference dir> --b <dir> --shapes <file> [--identical]
"""
import argparse
import sys

import numpy as np

THR = 1.0 / 255


def keyed(coords):
    c = coords.astype(np.int64)
    return (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--identical", action="store_true", help="the bar is byte identity")
    args = ap.parse_args()
    sids = [s.strip() for s in open(args.shapes) if s.strip()]
    if not sids or len(set(sids)) != len(sids):
        print("VERDICT REFUSED: empty or repeated shapes file"); sys.exit(2)
    n_fail = n_same = 0
    ious = []
    for sid in sids:
        try:
            A, B = np.load(f"{args.a}/{sid}.npz"), np.load(f"{args.b}/{sid}.npz")
        except FileNotFoundError:
            print(f"{sid} MISSING"); n_fail += 1; continue
        same = np.array_equal(A["coords"], B["coords"]) and np.array_equal(A["pred"], B["pred"])
        finite = bool(np.isfinite(A["pred"]).all() and np.isfinite(B["pred"]).all())
        n_same += same
        ka, kb = keyed(A["coords"]), keyed(B["coords"])
        union = np.union1d(ka, kb)
        coord_diff = len(np.setxor1d(ka, kb)) / max(len(union), 1)
        ma = ka[A["pred"].astype(np.float32).max(1) > THR]
        mb = kb[B["pred"].astype(np.float32).max(1) > THR]
        inter, uni = len(np.intersect1d(ma, mb)), len(np.union1d(ma, mb))
        iou = inter / uni if uni else 1.0
        xor = len(np.setxor1d(ma, mb))
        m = max(len(ma), len(mb))
        shared, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        mean_d = float(np.abs(A["pred"][ia].astype(np.float32) - B["pred"][ib].astype(np.float32)).mean()) if len(shared) else 0.0
        if m >= 100:
            ok = iou >= 0.95 and abs(len(ma) - len(mb)) <= 0.05 * m
        else:
            ok = xor <= 5
        ok = ok and coord_diff <= 0.005 and finite and (same or not args.identical)
        n_fail += not ok
        ious.append(iou)
        print(f"{sid} {'pass' if ok else 'FAIL'} same={same}{'' if finite else ' NONFINITE'} on={len(ma)}/{len(mb)} iou={iou:.4f} xor={xor} "
              f"coord_diff={coord_diff:.4%} mean|d|={mean_d:.2e}")
    print(f"shapes={len(sids)} byte_identical={n_same} fail={n_fail} mean_iou={np.mean(ious) if ious else float('nan'):.4f} "
          f"VERDICT {'PASS' if n_fail == 0 else 'FAIL'}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
