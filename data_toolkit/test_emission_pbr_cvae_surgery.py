"""Smoke: surgical init of the conditional emission VAE matches the pretrained
weights bit-exactly on shared modules and follows the slice-copy / zero-pad
rules on the three surgical projection layers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from trellis2 import models as trellis_models
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
from trellis2.utils.model_surgery import surgically_init_emission_pbr_cvae


# Match emission_pbr_vae_finetune_1k.yaml's encoder / decoder shape, but with
# the surgical I/O sizes (3-ch in, 64-ch from_latent, 3-ch out).
ENC_COMMON = dict(
    model_channels=[64, 128, 256, 512, 1024],
    latent_channels=32,
    num_blocks=[0, 4, 8, 16, 4],
    block_type=["SparseConvNeXtBlock3d"] * 5,
    block_args=[{"use_checkpoint": False}] * 5,
    use_fp16=False,
)

enc = SparseUnetVaeEncoder(
    in_channels=3,
    down_block_type=["SparseResBlockS2C3d"] * 4,
    **ENC_COMMON,
)

DEC_COMMON = dict(
    model_channels=[1024, 512, 256, 128, 64],
    latent_channels=64,
    num_blocks=[4, 16, 8, 4, 0],
    block_type=["SparseConvNeXtBlock3d"] * 5,
    block_args=[{"use_checkpoint": False}] * 5,
    use_fp16=False,
    pred_subdiv=False,
)
dec = SparseUnetVaeDecoder(
    out_channels=3,
    up_block_type=["SparseResBlockC2S3d"] * 4,
    **DEC_COMMON,
)

ENC_CKPT = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
DEC_CKPT = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"

surgically_init_emission_pbr_cvae(enc, dec, ENC_CKPT, DEC_CKPT)

# Load pretrained as reference
pre_enc = trellis_models.from_pretrained(ENC_CKPT).float().eval()
pre_dec = trellis_models.from_pretrained(DEC_CKPT).float().eval()

failures = []

# 1) Interior block weights are bit-identical to pretrained.
for (k_new, v_new), (k_pre, v_pre) in zip(
    enc.blocks.named_parameters(), pre_enc.blocks.named_parameters()
):
    if not torch.equal(v_new, v_pre):
        failures.append(f"enc.blocks.{k_new} differs from pretrained")
        break
for (k_new, v_new), (k_pre, v_pre) in zip(
    dec.blocks.named_parameters(), pre_dec.blocks.named_parameters()
):
    if not torch.equal(v_new, v_pre):
        failures.append(f"dec.blocks.{k_new} differs from pretrained")
        break

# 2) encoder.input_layer.weight[:, 0:3] == pretrained.input_layer.weight[:, 0:3]
#    encoder.input_layer.bias == pretrained.input_layer.bias
if not torch.equal(enc.input_layer.weight, pre_enc.input_layer.weight[:, 0:3]):
    failures.append("enc.input_layer weight not sliced from pretrained")
if not torch.equal(enc.input_layer.bias, pre_enc.input_layer.bias):
    failures.append("enc.input_layer bias not copied from pretrained")

# 3) decoder.from_latent.weight[:, 0:32] == pretrained.from_latent.weight
#    decoder.from_latent.weight[:, 32:64] == 0
#    decoder.from_latent.bias == pretrained.from_latent.bias
if not torch.equal(dec.from_latent.weight[:, 0:32], pre_dec.from_latent.weight):
    failures.append("dec.from_latent[:, 0:32] != pretrained from_latent weight")
if not torch.all(dec.from_latent.weight[:, 32:64] == 0):
    failures.append("dec.from_latent[:, 32:64] is not zero-init")
if not torch.equal(dec.from_latent.bias, pre_dec.from_latent.bias):
    failures.append("dec.from_latent bias not copied from pretrained")

# 4) decoder.output_layer.weight[0:3, :] == pretrained.output_layer.weight[0:3, :]
#    decoder.output_layer.bias == pretrained.output_layer.bias[0:3]
if not torch.equal(dec.output_layer.weight, pre_dec.output_layer.weight[0:3, :]):
    failures.append("dec.output_layer weight not sliced from pretrained")
if not torch.equal(dec.output_layer.bias, pre_dec.output_layer.bias[0:3]):
    failures.append("dec.output_layer bias not sliced from pretrained")

if failures:
    print("FAIL")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("PASS: surgery init produces the expected weights.")
