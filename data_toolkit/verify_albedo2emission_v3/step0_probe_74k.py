"""Step-0 probe against the REAL 74k data and the REAL v3 config.

The local step0_probe.py proves the bake is a no-op on 1k voxels with
hand-built modules. This one closes the remaining gap: it goes through
build_models on configs/scvae/emission_pbr_vae_ft_74k_v3.yaml, applies the
finetune_ckpt the way the trainer does, builds LightGenSLatEmission on the
production split, and runs EmissionPbrFinetuneTrainer.training_losses once.

It answers the single question a 35,800-step run must not be spent to answer:
does from_latent actually carry the pretrained weights, or did it cold-start?

    ~0.005   -> genuine finetune, proceed
    ~1.774   -> from_latent is at xavier init; STOP

Cheap on purpose: 1 GPU, one batch, no optimizer, no training loop -- so it
queues in minutes where the full 4-GPU probe waits hours.

Usage (on fir, inside the venv):
    python data_toolkit/verify_albedo2emission_v3/step0_probe_74k.py \
        --config configs/scvae/emission_pbr_vae_ft_74k_v3.yaml --batch 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import yaml
from easydict import EasyDict as edict

from trellis2.datasets.lightgen_slat import LightGenSLatEmission
from trellis2.trainers.builders import build_models
from trellis2.trainers.vae.emission_pbr_finetune import EmissionPbrFinetuneTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scvae/emission_pbr_vae_ft_74k_v3.yaml")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    print(f"config: {args.config}")
    print(f"  encoder in_channels  : {cfg['models']['encoder']['args']['in_channels']}")
    print(f"  decoder out_channels : {cfg['models']['decoder']['args']['out_channels']}")
    print(f"  decoder latent_ch    : {cfg['models']['decoder']['args']['latent_channels']}")
    print(f"  trainer              : {cfg['trainer']['name']}")
    print(f"  pbr_latent_name      : {cfg['dataset']['args'].get('pbr_latent_name')}")
    print(f"  pbr_attrs            : {cfg['dataset']['args'].get('pbr_attrs')}")

    # ---- models + the baked init, loaded strictly ----
    models = build_models(cfg["models"])
    ft = cfg["trainer"]["args"]["finetune_ckpt"]
    from safetensors.torch import load_file
    for name in ("encoder", "decoder"):
        sd = load_file(ft[name])
        missing, unexpected = models[name].load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (name, missing, unexpected)
        print(f"  {name}: {len(sd)} tensors loaded, 0 missing / 0 unexpected")
    W = models["decoder"].from_latent.weight
    print(f"  from_latent.weight {tuple(W.shape)}  "
          f"left-half norm {W[:, :32].norm().item():.4f}  "
          f"right-half absmax {W[:, 32:].abs().max().item():.6f}")
    assert W.shape[1] == 64
    for m in models.values():
        m.to(args.device).train()

    # ---- the production dataset, unmodified ----
    ds_args = dict(cfg["dataset"]["args"])
    ds = LightGenSLatEmission(**ds_args)
    print(f"dataset: {len(ds)} samples")

    batch = ds.collate_fn([ds[i] for i in range(args.batch)])
    keys = sorted(k for k in batch if k != "sha256")
    print(f"batch keys: {keys}")
    assert "pbr_latent" in batch, "dataset did NOT supply pbr_latent — check pbr_latent_name"

    def to_dev(v):
        return v.to(args.device) if hasattr(v, "to") else v
    x = to_dev(batch["x"])
    pbr = to_dev(batch["pbr_latent"])
    print(f"  x        {tuple(x.feats.shape)}")
    print(f"  pbr_lat  {tuple(pbr.feats.shape)}  dtype {pbr.feats.dtype}")

    # ---- one forward through the real trainer method ----
    t = EmissionPbrFinetuneTrainer.__new__(EmissionPbrFinetuneTrainer)
    t.loss_type = cfg["trainer"]["args"].get("loss_type", "l1")
    t.lambda_kl = cfg["trainer"]["args"].get("lambda_kl", 1e-7)
    t.emission_channels = cfg["trainer"]["args"].get("emission_channels", 3)
    t.training_models = models

    torch.manual_seed(0)
    with torch.no_grad():
        terms, _ = t.training_losses(x, pbr_latent=pbr)
    out = {k: float(v) for k, v in terms.items()}
    print(json.dumps({k: out[k] for k in ("loss", "l1", "kl") if k in out}, indent=2))

    l1 = out["l1"]
    REF, COLD = 0.004878106061369181, 1.774
    print(f"\nstep-0 train/l1 = {l1:.8f}")
    print(f"  albedo->emission pretrained reference : {REF:.8f}")
    print(f"  cold-start signature to reject        : ~{COLD}")
    ok = l1 < 0.05
    print("PASS: genuine finetune — from_latent carries the pretrained weights" if ok
          else "FAIL: looks like a cold start — DO NOT launch the full run")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
