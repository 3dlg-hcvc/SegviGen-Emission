"""Round-trip check for the two rekey tools that bridge standalone SegviGen
checkpoints and the lightning framework.

rekey_full_seg.py (forward): standalone {"state_dict": {"gen3dseg.flow_model.<key>": t}}
                              -> bare {"denoiser": {<key>: t}}   (framework finetune_ckpt input)
unrekey_for_eval.py (backward): bare {"denoiser": {<key>: t}}
                              -> standalone {"state_dict": {"gen3dseg.flow_model.<key>": t}}

If these two are truly inverses, running a real standalone checkpoint through
forward-then-backward must reproduce the original tensors byte-for-byte (same
dtype, same values -- torch.equal, not an approximate compare). This script
runs the actual CLI tools as subprocesses (not reimplemented logic) so it
exercises exactly what a user invoking them back to back would get.

Usage:
    python scripts_segvigen/round_trip_check.py [SRC_CKPT]

SRC_CKPT defaults to the full_seg checkpoint the port initializes from:
    /3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/
      snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
"""
import subprocess
import sys
import tempfile
import os

import torch

DEFAULT_SRC = (
    "/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/"
    "snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt"
)
HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"[round_trip] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.exists(src):
        print(f"[round_trip] FAIL: source checkpoint not found: {src}")
        return 1

    with tempfile.TemporaryDirectory() as td:
        bare_path = os.path.join(td, "bare_rekey.pt")
        back_path = os.path.join(td, "standalone_roundtrip.pt")

        run([sys.executable, os.path.join(HERE, "rekey_full_seg.py"), src, bare_path])
        run([sys.executable, os.path.join(HERE, "unrekey_for_eval.py"), bare_path, back_path])

        print("[round_trip] loading original and round-tripped checkpoints for comparison")
        orig = torch.load(src, map_location="cpu")
        orig_sd = orig["state_dict"] if "state_dict" in orig else orig
        back = torch.load(back_path, map_location="cpu")
        back_sd = back["state_dict"]

        orig_keys = set(orig_sd.keys())
        back_keys = set(back_sd.keys())
        if orig_keys != back_keys:
            missing = orig_keys - back_keys
            extra = back_keys - orig_keys
            print(f"[round_trip] FAIL: key sets differ. missing={len(missing)} extra={len(extra)}")
            for k in list(missing)[:10]:
                print(f"  missing: {k}")
            for k in list(extra)[:10]:
                print(f"  extra: {k}")
            return 1

        n_checked = 0
        n_mismatch = 0
        for k in sorted(orig_keys):
            a, b = orig_sd[k], back_sd[k]
            if not (torch.equal(a, b) and a.dtype == b.dtype):
                n_mismatch += 1
                print(f"[round_trip] MISMATCH at {k}: dtype {a.dtype} vs {b.dtype}, "
                      f"equal={torch.equal(a, b) if a.shape == b.shape else 'shape differs'}")
            n_checked += 1

        print(f"[round_trip] checked {n_checked} keys, {n_mismatch} mismatches")
        if n_mismatch == 0 and n_checked == len(orig_keys):
            print(f"[round_trip] PASS: {n_checked}/{n_checked} tensors byte-identical after "
                  "forward rekey_full_seg.py -> backward unrekey_for_eval.py")
            return 0
        print("[round_trip] FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
