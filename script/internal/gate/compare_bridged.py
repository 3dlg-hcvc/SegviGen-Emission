"""Compare two bridged SegviGen files ({"state_dict": {"gen3dseg.flow_model.<key>": t}}): same keys, equal tensors."""
import sys

import torch

a = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
b = torch.load(sys.argv[2], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
if set(a) != set(b):
    print(f"BRIDGE DIFFERENT keys: {len(set(a) ^ set(b))} differ"); sys.exit(1)
bad = [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]
print(f"BRIDGE {'IDENTICAL' if not bad else 'DIFFERENT'} keys={len(a)} differing_tensors={len(bad)} {bad[:3]}")
sys.exit(1 if bad else 0)
