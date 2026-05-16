"""Model-surgery helpers for converting a pretrained PBR VAE into other
emission/PBR VAE variants."""
from typing import Tuple
import torch
from .. import models as trellis_models


def surgically_init_emission_pbr_cvae(
    encoder,
    decoder,
    enc_pretrained_path: str,
    dec_pretrained_path: str,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """In-place init `encoder` / `decoder` from the pretrained PBR VAE.

    Expected target shapes:
        encoder.input_layer   :  3 -> model_channels[0]
        encoder.to_latent     :  unchanged from pretrained
        decoder.from_latent   :  64 -> model_channels[0]   (32 emission + 32 PBR cond)
        decoder.output_layer  :  model_channels[-1] -> 3

    All `blocks.*`, `to_subdiv`, and `to_latent` modules are copied 1:1 from
    pretrained. The three surgical layers above are warm-initialised:
        - encoder.input_layer.weight  := pretrained.input_layer.weight[:, 0:3]
        - encoder.input_layer.bias    := pretrained.input_layer.bias
        - decoder.from_latent.weight  := [pretrained.from_latent.weight, 0]
                                          (cols 32:64 are zero-init)
        - decoder.from_latent.bias    := pretrained.from_latent.bias
        - decoder.output_layer.weight := pretrained.output_layer.weight[0:3, :]
        - decoder.output_layer.bias   := pretrained.output_layer.bias[0:3]
    """
    pre_enc = trellis_models.from_pretrained(enc_pretrained_path).float().eval()
    pre_dec = trellis_models.from_pretrained(dec_pretrained_path).float().eval()

    # ---- Encoder ----
    enc_sd = pre_enc.state_dict()

    # Surgical input layer: take only the first 3 input columns.
    in_w = enc_sd["input_layer.weight"]
    in_b = enc_sd["input_layer.bias"]
    assert in_w.shape[1] == 6, (
        f"Expected pretrained encoder input_layer.weight shape [_, 6], got {tuple(in_w.shape)}"
    )
    enc_sd["input_layer.weight"] = in_w[:, 0:3].contiguous().clone()
    enc_sd["input_layer.bias"] = in_b.clone()

    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Encoder surgery had missing/unexpected keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    # ---- Decoder ----
    dec_sd = pre_dec.state_dict()

    # Surgical from_latent: zero-pad cols 32:64.
    fl_w = dec_sd["from_latent.weight"]  # shape [model_channels[0], 32]
    fl_b = dec_sd["from_latent.bias"]
    assert fl_w.shape[1] == 32, (
        f"Expected pretrained decoder from_latent.weight shape [_, 32], got {tuple(fl_w.shape)}"
    )
    new_fl_w = torch.zeros(fl_w.shape[0], 64, dtype=fl_w.dtype)
    new_fl_w[:, 0:32] = fl_w
    dec_sd["from_latent.weight"] = new_fl_w.contiguous()
    dec_sd["from_latent.bias"] = fl_b.clone()

    # Surgical output_layer: take only the first 3 output rows.
    out_w = dec_sd["output_layer.weight"]  # shape [6, model_channels[-1]]
    out_b = dec_sd["output_layer.bias"]
    assert out_w.shape[0] == 6, (
        f"Expected pretrained decoder output_layer.weight shape [6, _], got {tuple(out_w.shape)}"
    )
    dec_sd["output_layer.weight"] = out_w[0:3, :].contiguous().clone()
    dec_sd["output_layer.bias"] = out_b[0:3].contiguous().clone()

    missing, unexpected = decoder.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Decoder surgery had missing/unexpected keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return encoder, decoder
