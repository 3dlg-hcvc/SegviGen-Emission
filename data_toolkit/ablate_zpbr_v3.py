#!/usr/bin/env python
"""Ablate the z_pbr conditioning stream of the pbr->emission VAE.

The pbr->emission chain exists for exactly one reason: its decoder consumes
cat([z_emission, z_pbr]) and can therefore see PBR, where the albedo->emission
decoder cannot. After v3 equalized architecture, initialization and budget, the
two chains score 0.0055 vs 0.0056 — indistinguishable — and the z_pbr half of
from_latent.weight only ever grew to 2.11% of the emission half's norm.

Norm is not contribution, so this measures contribution directly: run the val
split three ways and compare reconstruction L1.

  normal   decoder(cat(z_emission, z_pbr))            the trained model
  zeroed   decoder(cat(z_emission, 0))                z_pbr removed
  shuffled decoder(cat(z_emission, roll(z_pbr)))      z_pbr present but WRONG

`shuffled` is the sharper test. Zeroing is in-distribution for a stream that was
initialized at zero, so a model that ignores z_pbr and a model that merely has a
small-but-real dependence can both look flat. Pairing each shape's emission with
another shape's PBR cannot be shrugged off: if the decoder uses PBR at all, a
mismatched one must hurt.

Deterministic: sample_posterior=False, so the encoder's mean is used and the
three arms differ only in z_pbr.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import yaml

from trellis2.trainers.builders import build_models
from trellis2.datasets import LightGenSLatEmission
from trellis2.modules import sparse as sp


def load_pbr_encoder(path: str, device):
    """The frozen conditioning encoder, loaded exactly as the trainer does
    (emission_vae.py:85-91): from_pretrained, .float().eval(), grads off."""
    import trellis2.models as trellis_models
    enc = trellis_models.from_pretrained(path)
    enc.float().eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pbr-encoder", default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole split")
    a = ap.parse_args()

    device = "cuda"
    cfg = yaml.safe_load(open(a.config))

    models = build_models(cfg["models"])
    sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    for name, m in models.items():
        pre = f"models.{name}."
        own = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        m.load_state_dict(own, strict=True)   # strict: a silent partial load here would
        m.float().eval().to(device)           # invalidate the whole measurement
        print(f"[load] {name}: {len(own)} tensors")

    pbr_enc = load_pbr_encoder(a.pbr_encoder, device)

    dargs = dict(cfg["dataset"]["args"])
    dargs["split"] = a.split
    ds = LightGenSLatEmission(**dargs)
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))
    print(f"[data] {a.split}: {len(ds)} samples, using {n}")

    EMI = cfg["trainer"]["args"].get("emission_channels", 3)
    tot = {"normal": 0.0, "zeroed": 0.0, "shuffled": 0.0}
    cnt = 0

    with torch.no_grad():
        for i in range(n):
            x = ds[i]["x"].to(device)   # get_instance returns {"x": SparseTensor, "sha256": ...}
            feats = x.feats
            x_emit = x.replace(feats[:, :EMI])
            x_pbr = x.replace(feats[:, EMI:])

            z, mean, logvar = models["encoder"](x_emit, sample_posterior=False, return_raw=True)
            z_pbr = pbr_enc(x_pbr)

            target = feats[:, :EMI]
            variants = {
                "normal": z_pbr.feats,
                "zeroed": torch.zeros_like(z_pbr.feats),
                # roll by half the token count: every token is paired with a
                # different token's PBR, within the same shape's occupancy so
                # the sparse structure stays valid.
                "shuffled": torch.roll(z_pbr.feats, shifts=max(1, z_pbr.feats.shape[0] // 2), dims=0),
            }
            for k, zp in variants.items():
                y = models["decoder"](z.replace(torch.cat([z.feats, zp], dim=-1)))
                tot[k] += (target - y.feats).abs().mean().item()
            cnt += 1
            if cnt % 50 == 0:
                print(f"  [{cnt}/{n}] " + "  ".join(f"{k}={tot[k]/cnt:.6f}" for k in tot))

    print("\n===== z_pbr ablation, %s split, n=%d =====" % (a.split, cnt))
    base = tot["normal"] / cnt
    for k in ("normal", "zeroed", "shuffled"):
        v = tot[k] / cnt
        delta = (v - base) / base * 100 if base else float("nan")
        print(f"  {k:9s} L1 = {v:.6f}   ({delta:+.2f}% vs normal)")


if __name__ == "__main__":
    main()
