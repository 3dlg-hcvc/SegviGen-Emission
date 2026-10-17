"""Strip the standalone lane's key prefix off a SegviGen checkpoint so the
framework's finetune loader can match it.

Standalone checkpoints prefix every key with gen3dseg.flow_model. (wrapper attr
inside the gen3dseg. save prefix). The framework's extractor accepts a payload
of the form {"denoiser": <bare state dict>}, so that is what this writes.
Verifies 640/640 keys and prints any residue instead of writing silently.
"""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")
if "state_dict" in sd:
    sd = sd["state_dict"]
PREFIXES = ("gen3dseg.flow_model.", "flow_model.")
out = {}
for k, v in sd.items():
    for p in PREFIXES:
        if k.startswith(p):
            out[k[len(p):]] = v
            break
    else:
        print(f"UNMATCHED KEY (left out): {k}")
print(f"{len(out)} keys rekeyed of {len(sd)} in source")
assert len(out) == len(sd), "unmatched keys above; refusing to write a partial rekey"
torch.save({"denoiser": out}, dst)
print(f"wrote {dst}")
