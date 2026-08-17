"""Tensor-level check of the baked albedo->emission v3 initial weights.

Correctness for this bake is unusually crisp: the encoder must be bit-identical
to pretrained, and the decoder must differ in exactly one tensor, whose left
half is pretrained and whose right half is zero. Anything else is a bug.

Contrast with the pbr->emission chain, whose bake resolves FIVE tensors (it
narrows encoder input 6->3 with a bias fold, and decoder output 6->3). This
chain changes neither I/O width -- the base_color slot packing is its identity
-- so the encoder passes through untouched.

Usage:
    conda run --no-capture-output -n trellis2 \
        python data_toolkit/verify_albedo2emission_v3/verify_identity.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import load_file

SNAP = ("/localhome/dya78/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts")
LAT = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baked", default="data/ckpt_init/albedo2emission_v3")
    ap.add_argument("--snap", default=os.environ.get("TRELLIS2_SNAP", SNAP))
    args = ap.parse_args()

    pre_enc = load_file(os.path.join(args.snap, "tex_enc_next_dc_f16c32_fp16.safetensors"))
    pre_dec = load_file(os.path.join(args.snap, "tex_dec_next_dc_f16c32_fp16.safetensors"))
    new_enc = load_file(os.path.join(args.baked, "encoder.safetensors"))
    new_dec = load_file(os.path.join(args.baked, "decoder.safetensors"))

    failures = []

    # ---- Encoder: bit-identical, key set included ----
    if set(pre_enc) != set(new_enc):
        failures.append(f"encoder key set differs: "
                        f"+{sorted(set(new_enc) - set(pre_enc))} "
                        f"-{sorted(set(pre_enc) - set(new_enc))}")
    for k in sorted(set(pre_enc) & set(new_enc)):
        if pre_enc[k].shape != new_enc[k].shape:
            failures.append(f"encoder.{k} shape {tuple(new_enc[k].shape)} "
                            f"!= pretrained {tuple(pre_enc[k].shape)}")
        elif not torch.equal(pre_enc[k], new_enc[k]):
            failures.append(f"encoder.{k} is not bit-identical to pretrained")

    # ---- Decoder: exactly one tensor may differ ----
    if set(pre_dec) != set(new_dec):
        failures.append(f"decoder key set differs: "
                        f"+{sorted(set(new_dec) - set(pre_dec))} "
                        f"-{sorted(set(pre_dec) - set(new_dec))}")
    changed = [k for k in sorted(set(pre_dec) & set(new_dec))
               if pre_dec[k].shape != new_dec[k].shape
               or not torch.equal(pre_dec[k], new_dec[k])]
    if changed != ["from_latent.weight"]:
        failures.append(f"expected exactly ['from_latent.weight'] to change, got {changed}")

    W_old = pre_dec["from_latent.weight"]
    W_new = new_dec["from_latent.weight"]
    if tuple(W_new.shape) != (W_old.shape[0], 2 * LAT):
        failures.append(f"from_latent.weight shape {tuple(W_new.shape)} "
                        f"!= {(W_old.shape[0], 2 * LAT)}")
    else:
        if not torch.equal(W_new[:, :LAT], W_old):
            failures.append("from_latent.weight[:, :32] != pretrained")
        if W_new[:, LAT:].abs().max().item() != 0.0:
            failures.append("from_latent.weight[:, 32:] is not all-zero")
    if W_new.dtype != W_old.dtype:
        failures.append(f"from_latent.weight dtype {W_new.dtype} != {W_old.dtype}")

    n_enc, n_dec = len(new_enc), len(new_dec)
    print(f"encoder tensors: {n_enc}   decoder tensors: {n_dec}   total: {n_enc + n_dec}")
    print(f"from_latent.weight: {tuple(W_old.shape)} -> {tuple(W_new.shape)}  dtype {W_new.dtype}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: encoder bit-identical; decoder differs in from_latent.weight only "
          "(left half pretrained, right half zero)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
