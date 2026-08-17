"""Assert that the albedo->emission v3 config differs from the pbr->emission v3
config in exactly the twelve intended fields, and in nothing else.

The albedo v3 config is DERIVED from the other chain's config on purpose, so
that every recipe field the two chains must share -- split, step budget, LR
schedule, batch sizes, KL weight, loss, EMA, sampler, checkpoint policy -- is
inherited verbatim instead of reconciled by hand. This check is what keeps that
property true: an accidental edit to a shared field silently re-introduces a
confound into a head-to-head that exists to isolate one variable.

Re-run after ANY edit to either config.

Usage:
    conda run --no-capture-output -n trellis2 \
        python data_toolkit/verify_albedo2emission_v3/verify_config_diff.py
"""
from __future__ import annotations

import argparse
import sys

import yaml

PBR = "configs/scvae/emission_vae_pbr2emission_74k_v3.yaml"
ALB = "configs/scvae/emission_pbr_vae_ft_74k_v3.yaml"

# The chain identity and its direct consequences. Nothing else may differ.
EXPECTED = {
    "models.encoder.args.in_channels",       # 3 -> 6, base_color-slot packing
    "models.decoder.args.out_channels",      # 3 -> 6, loss takes [:, :3]
    "trainer.name",                          # EmissionVaeTrainer -> EmissionPbrFinetuneTrainer
    "dataset.args.pbr_attrs[0]",             # 4 attrs -> [] (the pack overwrites 3:6 anyway)
    "dataset.args.pbr_attrs[1]",
    "dataset.args.pbr_attrs[2]",
    "dataset.args.pbr_attrs[3]",
    "dataset.args.pbr_latent_name",          # added: supplies pbr_latent
    "trainer.args.pretrained_pbr_encoder",   # deleted: no live encoder on this chain
    "trainer.args.finetune_ckpt.encoder",
    "trainer.args.finetune_ckpt.decoder",
    "trainer.args.wandb_name",
}

# Fields whose whole purpose is to be shared. Listed explicitly so a future edit
# that drops one is a test failure rather than a silent confound.
MUST_TRANSFER = [
    "dataset.args.data_root", "dataset.args.parquet_file", "dataset.args.split_json",
    "dataset.args.resolution", "dataset.args.split",
    "trainer.args.max_steps", "trainer.args.batch_size_per_gpu", "trainer.args.batch_split",
    "trainer.args.val_batch_size_per_gpu", "trainer.args.val_every_epochs",
    "trainer.args.optimizer.name", "trainer.args.optimizer.args.lr",
    "trainer.args.optimizer.args.weight_decay",
    "trainer.args.lr_scheduler.name", "trainer.args.lr_scheduler.args.T_max",
    "trainer.args.lr_scheduler.args.eta_min",
    "trainer.args.ema_rate[0]", "trainer.args.fp16_mode", "trainer.args.fp16_scale_growth",
    "trainer.args.grad_clip.name", "trainer.args.grad_clip.args.max_norm",
    "trainer.args.grad_clip.args.clip_percentile",
    "trainer.args.loss_type", "trainer.args.lambda_kl", "trainer.args.emission_channels",
    "trainer.args.save_top_k", "trainer.args.monitor_metric", "trainer.args.monitor_mode",
    "trainer.args.use_balanced_sampler", "trainer.args.dataloader_num_workers",
    "trainer.args.val_split_name", "trainer.args.num_sanity_val_steps",
    "trainer.args.wandb_enable", "trainer.args.wandb_project",
    "models.encoder.args.latent_channels", "models.decoder.args.latent_channels",
    "models.decoder.args.pred_subdiv",
]


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{p}[{i}]"))
    else:
        out[p] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbr", default=PBR)
    ap.add_argument("--albedo", default=ALB)
    args = ap.parse_args()

    A = flat(yaml.safe_load(open(args.pbr)))
    B = flat(yaml.safe_load(open(args.albedo)))
    diff = sorted(k for k in set(A) | set(B) if A.get(k, "<absent>") != B.get(k, "<absent>"))

    print(f"{len(set(A) | set(B))} leaf fields, {len(diff)} differ")
    for k in diff:
        print(f"  {k}: {A.get(k, '<absent>')!r} -> {B.get(k, '<absent>')!r}")

    failures = []
    extra = set(diff) - EXPECTED
    missing = EXPECTED - set(diff)
    if extra:
        failures.append(f"UNEXPECTED difference(s) — a shared recipe field drifted: {sorted(extra)}")
    if missing:
        failures.append(f"expected difference(s) not applied: {sorted(missing)}")
    for k in MUST_TRANSFER:
        if k not in A:
            failures.append(f"{k} missing from the pbr config — update this list")
        elif A.get(k) != B.get(k):
            failures.append(f"{k} did NOT transfer: {A.get(k)!r} vs {B.get(k)!r}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(f"PASS: exactly {len(EXPECTED)} intended differences; "
          f"all {len(MUST_TRANSFER)} shared recipe fields transferred verbatim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
