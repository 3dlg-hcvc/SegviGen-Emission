"""Dongchen's evaluation slices, applied to the live training curve.

Adopted 2026-08-26 (owner). The live loop had invented its own rule for the
zero-emission problem: drop every shape whose ground truth has no emissive
voxel, because IoU compares two empty sets there and scores a perfect 1.0
whenever the model also paints nothing, which pays a model for collapsing. The
team's evaluation framework already solves that problem by SLICING, and it
documents the same artifact (evaluation/pointsampled/metrics.py:18). Two rules
for one thing is one rule too many, so the curve uses HIS slice definitions and
his shipped membership lists rather than a parallel invention.

The slices, from docs/evaluation.md's table:

  full            every scored shape, comparable across generations
  nonzero_clean   full minus zero minus copy, the heuristic filter's own rule
  agentic_clean   nonzero_clean minus what the agentic stage also rejects
  zero_only       ground-truth emission is entirely zero
  copy_only       ground-truth emission matches albedo by the copy rule

`agent` in the map is a REJECT list, verified here rather than assumed: on his
usable.txt, usable - zero - copy - agent reproduces his shipped
val_agentic_clean.txt exactly, 183 shapes, no difference either way.

SCOPE. The slices are defined over his usable.txt (381 shapes). His published
tables sit on 371 of those, the rest dropped by `align_fp`, a glb-to-voxel
alignment gate computed from a mesh and a surface sample. This lane never builds
a glb, so that gate is neither reproducible nor applicable here; our `full` is
therefore the 381 and is NOT the same set as his published `full`. Shapes we
score that are outside usable.txt are counted and excluded from every slice, so
no slice silently grows.

THE ARTIFACT DOES NOT VANISH WITH THE ZERO SLICE, and this module measures what
is left. His zero list is computed from the source ground truth; our ground
truth is that emission decoded through the texture VAE at 256 and thresholded,
so a faint or tiny emissive region can survive his rule and still decode to
nothing here. Those shapes sit inside nonzero_clean carrying the same empty-union
1.0. Every slice therefore reports how many of its shapes have empty decoded
ground truth, and its mean both with and without them.
"""
import json
import os

SLICE_ORDER = ["full", "nonzero_clean", "agentic_clean", "zero_only", "copy_only"]
SLICE_BLURB = {
    "full": "every shape in the shared evaluation set",
    "nonzero_clean": "minus the zero-emission and copy-rule shapes",
    "agentic_clean": "minus what the agentic stage also rejects",
    "zero_only": "ground truth is entirely dark",
    "copy_only": "ground truth matches albedo by the copy rule",
}
TEAM_EVAL_SETS = ("/cs/3dlg-falas/project/omages/lightgen/lightgen_repo/"
                  "evaluation/newdata_eval/eval_sets")


def _read_list(path):
    with open(path) as f:
        return set(l.strip() for l in f if l.strip())


def load_defs(store=None, eval_sets=None):
    """Slice membership sets. Reads the store's pinned copy when there is one,
    so a page rebuild does not silently change meaning because someone pulled
    the team repo; falls back to the team repo itself."""
    src = None
    if store:
        pinned = os.path.join(store, "slices")
        if os.path.exists(os.path.join(pinned, "eval_slice_map.json")):
            src = pinned
    if src is None:
        src = eval_sets or TEAM_EVAL_SETS
    m = json.load(open(os.path.join(src, "eval_slice_map.json")))
    usable = _read_list(os.path.join(src, "usable.txt"))
    zero = set(m["zero"]) & usable
    copy = set(m["copy"]) & usable
    agent = set(m["agent"]) & usable
    defs = {
        "source": src,
        "usable": usable,
        "full": set(usable),
        "zero_only": zero,
        "copy_only": copy,
        "nonzero_clean": usable - zero - copy,
        "agentic_clean": usable - zero - copy - agent,
    }
    # Verify against his shipped list rather than trusting the arithmetic.
    shipped = os.path.join(src, "val_agentic_clean.txt")
    if os.path.exists(shipped):
        want = _read_list(shipped)
        defs["agentic_clean_matches_shipped"] = (want == defs["agentic_clean"])
        if not defs["agentic_clean_matches_shipped"]:
            defs["agentic_clean_diff"] = {
                "in_file_not_computed": sorted(want - defs["agentic_clean"])[:10],
                "computed_not_in_file": sorted(defs["agentic_clean"] - want)[:10]}
    return defs


def pin_defs(store, eval_sets=None, git_sha=None):
    """Copy the slice definitions into the store, so what a rebuild means does
    not drift with someone else's checkout."""
    import shutil
    src = eval_sets or TEAM_EVAL_SETS
    dst = os.path.join(store, "slices")
    os.makedirs(dst, exist_ok=True)
    for name in ("eval_slice_map.json", "usable.txt", "val_agentic_clean.txt"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, name))
    meta = {"pinned_from": src, "git_sha": git_sha,
            "pinned_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(dst, "PINNED.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return dst


def _stats(rows):
    if not rows:
        return None
    import statistics as st
    vals = sorted(r["iou_mean"] for r in rows)
    draws = [f for r in rows for f in r["pred_frac_per_draw"]]
    return {
        "n": len(rows),
        "mean_iou": sum(vals) / len(vals),
        "median_iou": st.median(vals),
        "n_zero_scoring": sum(1 for v in vals if v == 0.0),
        "dark_draw_rate": (sum(1 for f in draws if f == 0.0) / len(draws)) if draws else None,
    }


def slice_stats(per_shape, defs):
    """Per-slice statistics for one checkpoint, plus the residual empty-decoded
    ground-truth count inside each slice and the mean without those shapes."""
    ok = [r for r in per_shape if r.get("status") == "ok"]
    by_sid = {r["sid"]: r for r in ok}
    out = {"n_scored": len(ok),
           "n_outside_usable": len([s for s in by_sid if s not in defs["usable"]])}
    for name in SLICE_ORDER:
        rows = [by_sid[s] for s in defs[name] if s in by_sid]
        stats = _stats(rows)
        if stats is None:
            out[name] = None
            continue
        empty_gt = [r for r in rows if r.get("gt_frac", 0) == 0]
        stats["n_empty_decoded_gt"] = len(empty_gt)
        if empty_gt and name != "zero_only":
            kept = [r for r in rows if r.get("gt_frac", 0) > 0]
            stats["mean_iou_excluding_empty_decoded"] = (
                sum(r["iou_mean"] for r in kept) / len(kept) if kept else None)
            stats["n_excluding_empty_decoded"] = len(kept)
        stats["n_missing_from_this_run"] = len(defs[name]) - len(rows)
        out[name] = stats
    return out
