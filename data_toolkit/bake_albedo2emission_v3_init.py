#!/usr/bin/env python
"""Bake the albedo->emission _v3 VAE initial weights.

NOT A DATA BAKE. "Bake" here means pre-resolving pretrained MODEL WEIGHTS into
the shapes the v3 architecture expects, and writing them out as safetensors.
It touches no dataset, no voxels, no atlas -- the emission bake that produced
lightgen_74k_newbake is a different thing entirely and is not involved. Named
for symmetry with its sibling, data_toolkit/bake_pbr2emission_v3_init.py.

WHY THIS EXISTS
---------------
v3 is the v2 albedo->emission architecture with `decoder.latent_channels`
32 -> 64, so the decoder consumes cat([z_emission, z_pbr]).  Exactly one
parameter tensor changes shape:

    decoder.from_latent.weight    (1024, 32)  ->  (1024, 64)

BasicTrainer._load_weights_for_model drops shape-mismatched tensors rather
than adapting them (trellis2/trainers/basic.py:388-394) -- it prints one
warning and moves on.  That would leave the decoder's sole latent gateway at
xavier init on the whole information path, cold-starting a model that
otherwise inherits 100% of its weights.  That exact failure is what
invalidated the shipped pbr->emission _v2 run.  So it is resolved here,
offline, and the trainer then performs a clean load through UNMODIFIED code.

Contrast with bake_pbr2emission_v3_init.py, which resolves FIVE tensors: that
chain also narrows the encoder input (6 -> 3, with the constant pad folded
into the bias) and the decoder output (6 -> 3).  This chain changes neither --
riding the pretrained PBR VAE's base_color slot at full 6-channel I/O is the
albedo chain's identity -- so the encoder is copied through untouched, bit for
bit, and only the decoder's latent projection is widened.

THE MAPPING
-----------
decoder.from_latent : SparseLinear is nn.Linear (modules/sparse/linear.py:10),
    so weight is (out_features, in_features) and axis 1 is the latent channel.
    The trainer builds cat([z_emission, z_pbr]) with EMISSION FIRST, so the
    pretrained 32 columns belong at 0:32.  Columns 32:64 are the new PBR
    conditioning stream and are ZEROED, which makes the decoder at step 0
    exactly the pretrained function of the emission latent while leaving the
    PBR path fully trainable (its gradient is dL/dh (x) z_pbr, nonzero from
    step 1 even though the weight is 0).
    from_latent.bias transfers verbatim and stays calibrated because the
    zeroed block contributes nothing at init.

OPT-IN SURFACE
--------------
Only the two `finetune_ckpt` paths of
configs/scvae/emission_pbr_vae_ft_74k_v3.yaml point at this output.  Every
other config keeps its hub short-id or its own baked directory, so no other
run's init can be affected.  There is no flag to set wrong.

Usage:
    conda run --no-capture-output -n trellis2 \
        python data_toolkit/bake_albedo2emission_v3_init.py \
            --out data/ckpt_init/albedo2emission_v3
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SNAP = ("/localhome/dya78/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts")
LAT = 32  # pretrained latent width; v3's decoder takes 2 * LAT


def bake_decoder(sd: dict) -> dict:
    """Widen from_latent.weight to 2*LAT input channels, zeroing the new half.

    Arithmetic in fp32, stored back in the source dtype, matching
    bake_pbr2emission_v3_init.py: the pretrained files are fp16 throughout and
    a mixed-dtype checkpoint is a needless trap.
    """
    out = dict(sd)
    W = sd["from_latent.weight"]
    assert W.shape[1] == LAT, (
        f"expected from_latent.weight (_, {LAT}), got {tuple(W.shape)}")
    new = torch.zeros(W.shape[0], 2 * LAT, dtype=torch.float32)
    new[:, :LAT] = W.float()  # emission latent first; cols 32:64 (z_pbr) stay zero
    out["from_latent.weight"] = new.contiguous().to(W.dtype)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ckpt_init/albedo2emission_v3")
    ap.add_argument("--snap", default=os.environ.get("TRELLIS2_SNAP", SNAP))
    args = ap.parse_args()

    enc_src = os.path.join(args.snap, "tex_enc_next_dc_f16c32_fp16.safetensors")
    dec_src = os.path.join(args.snap, "tex_dec_next_dc_f16c32_fp16.safetensors")
    enc_sd = load_file(enc_src)
    dec_sd = load_file(dec_src)

    W_before = tuple(dec_sd["from_latent.weight"].shape)
    baked_dec = bake_decoder(dec_sd)
    W_after = tuple(baked_dec["from_latent.weight"].shape)

    os.makedirs(args.out, exist_ok=True)
    # The encoder is copied through untouched. Written out rather than symlinked
    # so the config's two finetune_ckpt paths sit in one directory and one rsync
    # (or one regeneration in a job body) moves the whole init.
    save_file(enc_sd, os.path.join(args.out, "encoder.safetensors"))
    save_file(baked_dec, os.path.join(args.out, "decoder.safetensors"))

    print(f"encoder: {len(enc_sd)} tensors copied verbatim from {enc_src}")
    print(f"decoder: {len(dec_sd)} tensors; from_latent.weight {W_before} -> {W_after}, "
          f"columns {LAT}:{2 * LAT} zeroed")
    print(f"wrote {args.out}/{{encoder,decoder}}.safetensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
