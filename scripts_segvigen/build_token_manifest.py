"""One-off: token-count manifest for the 256 emission npz tree.

Writes <emit_dir>/token_counts.json mapping sid -> N (coords rows), so
SegviGenEmissive256 can feed BalancedResumableSampler without opening
36k npz files over NFS at every dataset construction.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

emit_dir = sys.argv[1]
sids = [f[:-4] for f in os.listdir(emit_dir) if f.endswith(".npz")]
def count(sid):
    with np.load(os.path.join(emit_dir, f"{sid}.npz")) as d:
        return sid, int(d["coords"].shape[0])
out = {}
with ProcessPoolExecutor(max_workers=24) as ex:
    for i, (sid, n) in enumerate(ex.map(count, sids, chunksize=64)):
        out[sid] = n
        if i % 5000 == 0: print(f"{i}/{len(sids)}", flush=True)
tmp = os.path.join(emit_dir, "token_counts.json.tmp")
with open(tmp, "w") as f: json.dump(out, f)
os.replace(tmp, os.path.join(emit_dir, "token_counts.json"))
print(f"wrote {len(out)} entries")
