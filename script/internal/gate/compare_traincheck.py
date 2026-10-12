"""Compare two training-check outputs (`train_check.sh`): record.jsonl line by line, summary.jsonl file by file.

Ids, steps, epochs, keys and counts must be equal. Numbers (loss, lr, gradient norm, weight sums, val
metrics, EMA and optimizer sums) are IDENTICAL when equal, WITHIN_NOISE when every |d| is at most the
bound in `--noise` (the max |d| per field from a noise comparison's JSON), else DIFFERENT. A NaN on either
side, a missing line or an empty record fails. Exit 0 on PASS, 1 on FAIL, 2 when refused.
"""
import argparse
import json
import math
import os
import sys

NUM_FIELDS = ("loss", "lr", "grad_norm", "weight_sum", "state_dict_sum", "optimizer_sum")


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()
    bound = json.load(open(args.noise))["max_abs"] if args.noise else {}

    problems, max_abs, n_num, n_same = [], {}, 0, 0
    for name in ("record.jsonl", "summary.jsonl"):
        A, B = load(os.path.join(args.a, name)), load(os.path.join(args.b, name))
        if not A or not B:
            print(f"VERDICT REFUSED: empty {name}"); sys.exit(2)
        if len(A) != len(B):
            problems.append(f"{name}: {len(A)} vs {len(B)} lines")
        for i, (ra, rb) in enumerate(zip(A, B)):
            fa, fb = flatten(ra), flatten(rb)
            if set(fa) != set(fb):
                problems.append(f"{name}:{i} fields differ {sorted(set(fa) ^ set(fb))[:4]}"); continue
            for k in fa:
                va, vb = fa[k], fb[k]
                field = k.split(".")[-1] if k.split(".")[-1] in NUM_FIELDS else k.split(".")[0]
                if isinstance(va, float) or isinstance(vb, float):
                    if va is None or vb is None or math.isnan(va) or math.isnan(vb):
                        problems.append(f"{name}:{i} {k} NaN/None {va} {vb}"); continue
                    n_num += 1
                    d = abs(va - vb)
                    max_abs[field] = max(max_abs.get(field, 0.0), d)
                    if d == 0:
                        n_same += 1
                    elif d > bound.get(field, 0.0):
                        problems.append(f"{name}:{i} {k} {va!r} vs {vb!r} (|d|={d:.3e})")
                elif va != vb:
                    problems.append(f"{name}:{i} {k} {va!r} vs {vb!r}")
    for p in problems[:30]:
        print(p)
    ok = not problems
    verdict = "PASS" if ok else "FAIL"
    kind = "IDENTICAL" if ok and n_same == n_num else ("WITHIN_NOISE" if ok else "DIFFERENT")
    print("max_abs", {k: f"{v:.3e}" for k, v in sorted(max_abs.items())})
    print(f"numbers={n_num} equal={n_same} problems={len(problems)} {kind} VERDICT {verdict}")
    if args.json_out:
        json.dump({"max_abs": max_abs, "pass": ok, "problems": len(problems)}, open(args.json_out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
