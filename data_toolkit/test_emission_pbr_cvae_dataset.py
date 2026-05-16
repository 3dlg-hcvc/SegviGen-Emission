"""Smoke: LightGenSLatEmission with pbr_latent_name returns a pbr_latent
SparseTensor at the downsampled grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from trellis2.datasets.lightgen_slat import LightGenSLatEmission


ds = LightGenSLatEmission(
    data_root="data/lightgen_test",
    parquet_file="data/lightgen_test/df_10samples.parquet",
    resolution=256,
    split="train",
    split_json="data/lightgen_test/data_splits_10sample_as1k.json",
    pbr_attrs=[],  # we don't need PBR voxels — using pre-encoded latents
    pbr_latent_name="tex_enc_next_dc_f16c32_fp16_256",
)

sample = ds[0]
failures = []

# 1) x is emission-only (3 channels)
if sample["x"].feats.shape[1] != 3:
    failures.append(f"x.feats has {sample['x'].feats.shape[1]} channels, expected 3 (emission-only)")

# 2) pbr_latent is present and 32-channel
if "pbr_latent" not in sample:
    failures.append("pbr_latent missing from sample dict")
elif sample["pbr_latent"].feats.shape[1] != 32:
    failures.append(f"pbr_latent.feats has {sample['pbr_latent'].feats.shape[1]} channels, expected 32")

# 3) Latent grid is smaller than the voxel grid (downsampled).
if "pbr_latent" in sample and sample["pbr_latent"].coords.shape[0] >= sample["x"].coords.shape[0]:
    failures.append(
        f"Latent should be downsampled vs voxel grid: "
        f"latent M={sample['pbr_latent'].coords.shape[0]} vs voxel N={sample['x'].coords.shape[0]}"
    )

if failures:
    print("FAIL")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("PASS: dataset returns pbr_latent at downsampled grid.")
