#!/usr/bin/env python3
"""Patch a DiT config's emission_normalization stats from a
normalization_stats.json written by encode_emission_pbrfinetune_dataset.py.

Usage:
    python data_toolkit/patch_dit_config_stats.py \
        --config configs/gen/emission_dit_1k_probe.yaml \
        --stats data/lightgen_1k/emission_latents/emission_pbr_finetune_1k/normalization_stats.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


def format_list(vals):
    return "[" + ", ".join(f"{v:.6f}" for v in vals) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stats", required=True)
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())
    new_mean = format_list(stats["mean"])
    new_std = format_list(stats["std"])

    txt = Path(args.config).read_text()
    # Replace the first emission_normalization block's mean/std lines.
    # The YAML here has the two lines directly under emission_normalization.
    block_re = re.compile(
        r"(emission_normalization:\s*\n\s*mean:\s*)\[[^\]]*\](\s*\n\s*std:\s*)\[[^\]]*\]",
        re.MULTILINE,
    )
    new_txt, n = block_re.subn(rf"\1{new_mean}\2{new_std}", txt, count=1)
    if n == 0:
        print("ERROR: could not find emission_normalization block to patch", file=sys.stderr)
        sys.exit(1)
    Path(args.config).write_text(new_txt)
    print(f"Patched {args.config}: emission_normalization mean/std updated")
    print(f"  n_samples={stats.get('n_samples')} channels={stats['channels']} "
          f"total_tokens={stats['total_tokens']}")


if __name__ == "__main__":
    main()
