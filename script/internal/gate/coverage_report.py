"""Coverage run report (step 9): which of OUR lines never ran.

    python coverage_report.py --repo <worktree> --rev <commit> --base 98c2548 --tree_name <name> \
        --json run1.json [run2.json ...] > report.txt

Each --json is `coverage json` output of one run over an extracted tree whose top directory is
<tree_name>; paths are mapped to repo-relative ones by what follows "/<tree_name>/". The executed lines
of all runs are unioned. "Our lines" are the lines `git diff -U0 <base> <rev>` adds to each Python file
(upstream's own lines are D9's and not reported). Per file: our executable lines that never ran, as
ranges, and files of ours no run imported. "Never ran" is not "dead": see the plan's step 9.
"""
import argparse
import json
import re
import subprocess
from collections import defaultdict


def our_lines(repo, base, rev):
    files = subprocess.check_output(["git", "-C", repo, "diff", "--name-only", "--diff-filter=AM", base, rev],
                                    text=True).split()
    out = {}
    for f in files:
        if not f.endswith(".py") or f.startswith(("o-voxel/", "script/internal/gate/")):
            continue
        diff = subprocess.check_output(["git", "-C", repo, "diff", "-U0", base, rev, "--", f], text=True)
        lines = set()
        for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            lines.update(range(start, start + count))
        out[f] = lines
    return out


def ranges(nums):
    nums = sorted(nums)
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--base", default="98c2548")
    ap.add_argument("--tree_name", required=True)
    ap.add_argument("--json", nargs="+", required=True)
    args = ap.parse_args()

    executed, statements = defaultdict(set), defaultdict(set)
    for path in args.json:
        data = json.load(open(path))
        for f, rec in data["files"].items():
            key = f"/{args.tree_name}/"
            if key in f:
                rel = f.split(key, 1)[1]
            elif not f.startswith("/"):
                rel = f          # coverage writes paths relative to the directory the report ran in
            else:
                continue
            executed[rel] |= set(rec["executed_lines"])
            statements[rel] |= set(rec["executed_lines"]) | set(rec["missing_lines"])

    ours = our_lines(args.repo, args.base, args.rev)
    total_ours = total_run = 0
    print(f"coverage of our lines at {args.rev} (vs {args.base}), {len(args.json)} runs\n")
    not_imported = []
    for f in sorted(ours):
        if f not in statements:
            not_imported.append(f)
            continue
        mine = ours[f] & statements[f]
        miss = mine - executed[f]
        total_ours += len(mine)
        total_run += len(mine) - len(miss)
        print(f"{f}: {len(mine) - len(miss)}/{len(mine)} of our executable lines ran"
              + (f"; never: {ranges(miss)}" if miss else ""))
    print(f"\nnever imported by any run: {', '.join(not_imported) if not_imported else 'none'}")
    print(f"total: {total_run}/{total_ours} of our executable lines ran in the imported files")


if __name__ == "__main__":
    main()
