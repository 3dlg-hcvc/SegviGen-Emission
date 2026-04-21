#!/usr/bin/env python3
"""Compute per-channel normalization stats for YAML config."""
import numpy as np
import yaml

SAMPLE = "0007deb6d96c4474b80faa5aa3888760"
ROOT = "data/lightgen_test"

result = {}
for label, path in [
    ("emission", f"{ROOT}/emission_latents/emission_vae_optB_overfit/{SAMPLE}.npz"),
    ("pbr",      f"{ROOT}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{SAMPLE}.npz"),
    ("shape",    f"{ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/{SAMPLE}.npz"),
]:
    d = np.load(path)
    feats = d["feats"]
    mean = feats.mean(axis=0)
    std  = feats.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)
    key = f"{label}_normalization"
    result[key] = {
        "mean": [round(float(v), 6) for v in mean],
        "std":  [round(float(v), 6) for v in std],
    }
    print(f"\n{label}: shape={feats.shape}, mean_range=[{mean.min():.4f}, {mean.max():.4f}], std_range=[{std.min():.4f}, {std.max():.4f}]")

print("\n\n# YAML output for config:\n")
print(yaml.dump(result, default_flow_style=False, sort_keys=False))
