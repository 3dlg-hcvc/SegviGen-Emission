"""Compare two live-gate output dirs (`live_gate.py`), per shape and key.

Per key: byte identity, mean and max absolute difference. A shape is IDENTICAL when every key is
byte-identical; WITHIN_NOISE when every max |d| is at most the noise bound (`--noise`, read from
a noise comparison's JSON: the max |d| per key over two unchanged-code runs); else DIFFERENT.
NaN on one side, a missing shape, a missing key or a coordinate mismatch is a failure. An empty or
repeated shapes file is refused.

Exit 0 when every shape is IDENTICAL or WITHIN_NOISE, 1 otherwise, 2 when refused.
"""
import argparse
import json
import os
import sys

import numpy as np

KEYS = ("fwd_t25", "fwd_t50", "fwd_t75", "sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--noise", default=None, help="JSON with a per-key max |d| bound")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    sids = [s.strip() for s in open(args.shapes) if s.strip()]
    if not sids or len(set(sids)) != len(sids):
        print("VERDICT REFUSED: empty or repeated shapes file"); sys.exit(2)
    bound = {k: 0.0 for k in KEYS}
    if args.noise:
        bound.update(json.load(open(args.noise))["max_abs"])

    counts = {"IDENTICAL": 0, "WITHIN_NOISE": 0, "DIFFERENT": 0, "MISSING": 0, "NAN": 0}
    max_abs = {k: 0.0 for k in KEYS}
    rows = []
    for sid in sids:
        pa, pb = os.path.join(args.a, f"{sid}.npz"), os.path.join(args.b, f"{sid}.npz")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            counts["MISSING"] += 1; rows.append((sid, "MISSING", {})); continue
        A, B = np.load(pa), np.load(pb)
        if not np.array_equal(A["coords"], B["coords"]):
            counts["DIFFERENT"] += 1; rows.append((sid, "DIFFERENT(coords)", {})); continue
        verdict, per = "IDENTICAL", {}
        for k in KEYS:
            if k not in A or k not in B:
                verdict = "MISSING"; break
            a, b = A[k], B[k]
            if a.shape != b.shape:
                verdict = "DIFFERENT"; break
            if np.isnan(a).any() or np.isnan(b).any():
                verdict = "NAN"; break
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            m, mx = float(d.mean()), float(d.max())
            per[k] = (bool(np.array_equal(a, b)), m, mx)
            max_abs[k] = max(max_abs[k], mx)
            if not per[k][0]:
                if mx <= bound[k] and verdict == "IDENTICAL":
                    verdict = "WITHIN_NOISE"
                elif mx > bound[k]:
                    verdict = "DIFFERENT"
        counts[verdict] += 1
        rows.append((sid, verdict, per))

    for sid, v, per in rows:
        if v != "IDENTICAL":
            print(sid, v, {k: f"same={s} mean={m:.3e} max={x:.3e}" for k, (s, m, x) in per.items()})
    print("max_abs", {k: f"{v:.3e}" for k, v in max_abs.items()})
    ok = counts["DIFFERENT"] == counts["MISSING"] == counts["NAN"] == 0
    print(f"same={counts['IDENTICAL']} within_noise={counts['WITHIN_NOISE']} different={counts['DIFFERENT']} "
          f"missing={counts['MISSING']} nan={counts['NAN']} VERDICT {'PASS' if ok else 'FAIL'}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"counts": counts, "max_abs": max_abs, "pass": ok}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
