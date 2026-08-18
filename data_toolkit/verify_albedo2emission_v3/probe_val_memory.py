"""Measure peak VRAM for the v3 VAE at the config's TRAIN and VAL batch sizes.

Why this exists: the 100-step probe never reached a validation epoch (71,646
train samples / global batch 64 = ~1,120 steps per epoch), so the one inherited
setting that had never run on this hardware went untested --
val_batch_size_per_gpu: 32, which comes from venus-19's 96 GB cards while fir's
H100s have 80 GB.

Dropping it to 16 is NOT a free fix: val/loss is a per-batch voxel mean reduced
with sample-count weighting (basic.py:187-203) and save_top_k:1 selects on it, so
the two chains must validate at the same grouping to stay comparable. Hence
measure rather than assume.

Worst case on purpose: the heaviest samples by voxel count, not a random draw.
Sparse-voxel per-sample memory varies ~100x, and BalancedResumableSampler groups
by load, so the heaviest group is what decides whether a run OOMs.

Usage:
    python data_toolkit/verify_albedo2emission_v3/probe_val_memory.py \
        --config configs/scvae/emission_pbr_vae_ft_74k_v3.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import yaml
from safetensors.torch import load_file

from trellis2.datasets.lightgen_slat import LightGenSLatEmission
from trellis2.trainers.builders import build_models
from trellis2.trainers.vae.emission_pbr_finetune import EmissionPbrFinetuneTrainer

GB = 1024 ** 3


def build(cfg, device):
    models = build_models(cfg["models"])
    ft = cfg["trainer"]["args"]["finetune_ckpt"]
    for name in ("encoder", "decoder"):
        models[name].load_state_dict(load_file(ft[name]), strict=True)
        models[name].to(device).train()
    t = EmissionPbrFinetuneTrainer.__new__(EmissionPbrFinetuneTrainer)
    t.loss_type = cfg["trainer"]["args"].get("loss_type", "l1")
    t.lambda_kl = cfg["trainer"]["args"].get("lambda_kl", 1e-7)
    t.emission_channels = cfg["trainer"]["args"].get("emission_channels", 3)
    t.training_models = models
    return t, models


def heaviest(ds, n):
    order = sorted(range(len(ds)), key=lambda i: ds.loads[i], reverse=True)
    return order[:n]


def run(tag, t, ds, idx, device, backward):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = ds.collate_fn([ds[i] for i in idx])
    x = batch["x"].to(device)
    pbr = batch["pbr_latent"].to(device) if "pbr_latent" in batch else None
    nvox = x.feats.shape[0]
    ctx = torch.enable_grad() if backward else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.float16):
        terms, _ = t.training_losses(x, pbr_latent=pbr)
        if backward:
            terms["loss"].backward()
    peak = torch.cuda.max_memory_allocated() / GB
    total = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"{tag:34s} n={len(idx):3d} voxels={nvox:9,d} "
          f"peak={peak:6.2f} GB / {total:.0f} GB ({100*peak/total:.0f}%)"
          f"{'  [fwd+bwd]' if backward else '  [no_grad]'}")
    for m in t.training_models.values():
        m.zero_grad(set_to_none=True)
    del batch, x, pbr, terms
    torch.cuda.empty_cache()
    return peak, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scvae/emission_pbr_vae_ft_74k_v3.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tr_bs = 16  # what the fir launcher overrides to
    va_bs = cfg["trainer"]["args"]["val_batch_size_per_gpu"]
    print(f"config train BS (launcher override) = {tr_bs}, val BS = {va_bs}\n")

    t, _ = build(cfg, args.device)

    va_args = dict(cfg["dataset"]["args"]); va_args["split"] = "val"
    ds_val = LightGenSLatEmission(**va_args)
    ds_tr = LightGenSLatEmission(**cfg["dataset"]["args"])

    peaks = {}
    peaks["train"] = run(f"TRAIN  bs={tr_bs} heaviest", t, ds_tr, heaviest(ds_tr, tr_bs), args.device, True)
    peaks["val"] = run(f"VAL    bs={va_bs} heaviest", t, ds_val, heaviest(ds_val, va_bs), args.device, False)
    peaks["val16"] = run(f"VAL    bs=16 heaviest (fallback)", t, ds_val, heaviest(ds_val, 16), args.device, False)

    vp, total = peaks["val"]
    print()
    headroom = 100 * (1 - vp / total)
    if vp / total < 0.85:
        print(f"PASS: val bs={va_bs} peaks at {100*vp/total:.0f}% ({headroom:.0f}% headroom) — keep it, "
              f"the two chains stay commensurable")
        return 0
    print(f"FAIL: val bs={va_bs} peaks at {100*vp/total:.0f}% — too close to the edge. "
          f"Raise with Dongchen before dropping it; val/loss comparability is at stake.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
