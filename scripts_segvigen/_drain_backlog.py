"""One-off: submit a full sharded evaluation for every checkpoint in the mirror
that has no record yet. Owner directive, 2026-08-26: drain the backlog at once
rather than waiting for the loop's one-at-a-time slot to work through it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from submit_eval_chain import submit_chain, DEFAULT_STORE   # noqa: E402
from live_eval_driver import mirror_snapshots, DEFAULT_MIRROR, record_path  # noqa: E402

store = DEFAULT_STORE
rec_dir = os.path.join(store, "records")
markers = {}
mp = os.path.join(store, "best_markers.json")
if os.path.exists(mp):
    markers = json.load(open(mp))

snaps = mirror_snapshots(DEFAULT_MIRROR)
todo = [s for s in sorted(snaps) if not os.path.exists(record_path(rec_dir, s, "quick"))]
print(f"mirror holds {len(snaps)} checkpoint(s); {len(todo)} without a record: {todo}")
for step in todo:
    vl = markers.get(str(step), {}).get("val_loss")
    r = submit_chain(snaps[step], step, store=store, n_shards=8, val_loss=vl, quiet=True)
    print("step {:>6}  bridge {}  array {}  merge {}".format(
        step, r["bridge"], r["array"], r["merge"]))
