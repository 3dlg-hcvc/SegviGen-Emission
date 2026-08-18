"""Measure peak VRAM for the v3 VAE on the validation batches that will ACTUALLY occur.

Why this exists: the 100-step probe never reached a validation epoch (~1,120
steps per epoch), so val_batch_size_per_gpu: 32 -- inherited from venus-19's
96 GB cards -- has never run on fir's 80 GB H100s.

Dropping it to 16 is not a free fix: val/loss is a per-batch voxel mean reduced
with sample-count weighting (basic.py:187-203) and save_top_k:1 selects on it,
so both chains must validate at the same grouping to stay comparable.

WHAT THIS MEASURES, and what an earlier version got wrong. A first attempt fed
the model the globally heaviest N samples and OOM'd -- but that batch cannot
occur. The val loader is built with shuffle=False, no sampler and drop_last=False
(trainers/data.py:114-129), so under DDP Lightning injects
DistributedSampler(shuffle=False) and rank r deterministically receives
indices[r::world_size], chunked by val_batch_size. Those exact groups are
reproduced here, and the heaviest few are the ones measured.

Usage:
    python data_toolkit/verify_albedo2emission_v3/probe_val_memory.py \
        --config configs/scvae/emission_pbr_vae_ft_74k_v3.yaml --world-size 4 --top 3
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


def real_val_groups(n_samples, world_size, batch_size):
    """Reproduce DistributedSampler(shuffle=False) + DataLoader chunking."""
    per_rank = -(-n_samples // world_size)          # ceil, DistributedSampler pads
    total = per_rank * world_size
    idx = list(range(n_samples)) + list(range(total - n_samples))   # pad by wrap
    groups = []
    for r in range(world_size):
        mine = idx[r::world_size]
        for s in range(0, len(mine), batch_size):   # drop_last=False
            groups.append((r, mine[s:s + batch_size]))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scvae/emission_pbr_vae_ft_74k_v3.yaml")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    va_bs = cfg["trainer"]["args"]["val_batch_size_per_gpu"]
    print(f"val_batch_size_per_gpu = {va_bs}, world_size = {args.world_size}\n")

    models = build_models(cfg["models"])
    ft = cfg["trainer"]["args"]["finetune_ckpt"]
    for name in ("encoder", "decoder"):
        models[name].load_state_dict(load_file(ft[name]), strict=True)
        models[name].to(args.device).eval()
    t = EmissionPbrFinetuneTrainer.__new__(EmissionPbrFinetuneTrainer)
    t.loss_type = cfg["trainer"]["args"].get("loss_type", "l1")
    t.lambda_kl = cfg["trainer"]["args"].get("lambda_kl", 1e-7)
    t.emission_channels = cfg["trainer"]["args"].get("emission_channels", 3)
    t.training_models = models

    va_args = dict(cfg["dataset"]["args"]); va_args["split"] = "val"
    ds = LightGenSLatEmission(**va_args)

    groups = real_val_groups(len(ds), args.world_size, va_bs)
    weighted = sorted(groups, key=lambda g: sum(ds.loads[i] for i in g[1]), reverse=True)
    print(f"{len(groups)} real validation batches across {args.world_size} ranks; "
          f"sizes {sorted({len(g[1]) for g in groups})}")
    print(f"measuring the {args.top} heaviest\n")

    total = torch.cuda.get_device_properties(0).total_memory / GB
    worst = 0.0
    for rank, gidx in weighted[:args.top]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        batch = ds.collate_fn([ds[i] for i in gidx])
        x = batch["x"].to(args.device)
        pbr = batch["pbr_latent"].to(args.device) if "pbr_latent" in batch else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t.training_losses(x, pbr_latent=pbr)
        peak = torch.cuda.max_memory_allocated() / GB
        worst = max(worst, peak)
        print(f"  rank {rank}  n={len(gidx):2d}  voxels={x.feats.shape[0]:9,d}  "
              f"peak={peak:6.2f} GB / {total:.0f} GB ({100*peak/total:.0f}%)")
        del batch, x, pbr
        torch.cuda.empty_cache()

    print()
    pct = 100 * worst / total
    if pct < 85:
        print(f"PASS: worst real validation batch peaks at {pct:.0f}% "
              f"({total - worst:.1f} GB headroom) — keep val_batch_size_per_gpu={va_bs}")
        return 0
    print(f"FAIL: worst real validation batch peaks at {pct:.0f}% — too close to the edge.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
