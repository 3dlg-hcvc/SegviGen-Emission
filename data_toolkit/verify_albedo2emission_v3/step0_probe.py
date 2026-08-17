"""Step-0 equivalence probe: albedo->emission v3 (decoder latent 64, new half
zeroed) must decode IDENTICALLY to v2 (decoder latent 32) at initialization.

W_new @ cat([z_e, z_p]) == W_old @ z_e + 0 @ z_p == W_old @ z_e, so the two
decoders are the same function of the emission latent at step 0.  The only
permissible difference is floating-point accumulation order, which is why the
gate is a tight tolerance rather than torch.equal.

This is a strictly stronger gate than comparing mean L1 against the 0.004878
reference: two different functions can share a mean and still differ per voxel.

No training.  One fixed batch of real 1k voxels, one seed, two decoders.

Usage:
    conda run --no-capture-output -n trellis2 \
        python data_toolkit/verify_albedo2emission_v3/step0_probe.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2")

import numpy as np
import torch
from safetensors.torch import load_file

import o_voxel as _o_voxel
from trellis2.modules import sparse as sp
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder

ROOT = "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2/data/lightgen_1k"
BAKED = "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2/data/ckpt_init/albedo2emission_v3"
SNAP = ("/localhome/dya78/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
DEV = "cuda"
PAD = torch.tensor([-1.0, 1.0, 1.0])  # PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA

COMMON_ENC = dict(in_channels=6, model_channels=[64, 128, 256, 512, 1024], latent_channels=32,
                  num_blocks=[0, 4, 8, 16, 4],
                  block_type=["SparseConvNeXtBlock3d"] * 5,
                  down_block_type=["SparseResBlockS2C3d"] * 4,
                  block_args=[{}] * 5, use_fp16=False)
COMMON_DEC = dict(out_channels=6, model_channels=[1024, 512, 256, 128, 64],
                  num_blocks=[4, 16, 8, 4, 0],
                  block_type=["SparseConvNeXtBlock3d"] * 5,
                  up_block_type=["SparseResBlockC2S3d"] * 4,
                  block_args=[{}] * 5, use_fp16=False, pred_subdiv=False)


def load_batch(n_shapes=4, max_vox=60000):
    """Shapes that have BOTH an emission voxel file and a pre-encoded PBR latent."""
    lat_dir = os.path.join(ROOT, "pbr_latents", "tex_enc_next_dc_f16c32_fp16_256")
    feats, coords, lat_feats, lat_coords = [], [], [], []
    bi = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "emission_voxels_256", "*.vxz"))):
        sha = os.path.basename(f)[:-4]
        npz = os.path.join(lat_dir, f"{sha}.npz")
        if not os.path.exists(npz):
            continue
        c, a = _o_voxel.io.read_vxz(f, num_threads=4)
        if c.shape[0] > max_vox:
            continue
        e = a["emissive"].float() / 255.0 * 2.0 - 1.0            # [N, 3]
        pack = torch.cat([e, PAD.expand(e.shape[0], -1)], -1)     # [N, 6]
        d = np.load(npz)
        lf = torch.from_numpy(d["feats"].copy()).float()          # [M, 32]
        lc = torch.from_numpy(d["coords"].copy()).int()           # [M, 3] or [M, 4]
        if lc.shape[1] == 4:
            lc = lc[:, 1:]
        feats.append(pack)
        coords.append(torch.cat([torch.full((c.shape[0], 1), bi, dtype=c.dtype), c], -1))
        lat_feats.append(lf)
        lat_coords.append(torch.cat([torch.full((lc.shape[0], 1), bi, dtype=lc.dtype), lc], -1))
        bi += 1
        if bi == n_shapes:
            break
    assert bi == n_shapes, f"only found {bi} usable shapes, wanted {n_shapes}"
    x = sp.SparseTensor(torch.cat(feats).to(DEV), torch.cat(coords).to(DEV))
    z_pbr = sp.SparseTensor(torch.cat(lat_feats).to(DEV), torch.cat(lat_coords).to(DEV))
    return x, z_pbr


@torch.no_grad()
def main() -> int:
    torch.manual_seed(0)
    x, z_pbr = load_batch()
    target = x.feats[:, :3]

    enc = SparseUnetVaeEncoder(**COMMON_ENC).float().eval().to(DEV)
    enc.load_state_dict(load_file(os.path.join(SNAP, "tex_enc_next_dc_f16c32_fp16.safetensors")))

    dec_v2 = SparseUnetVaeDecoder(latent_channels=32, **COMMON_DEC).float().eval().to(DEV)
    dec_v2.load_state_dict(load_file(os.path.join(SNAP, "tex_dec_next_dc_f16c32_fp16.safetensors")))

    dec_v3 = SparseUnetVaeDecoder(latent_channels=64, **COMMON_DEC).float().eval().to(DEV)
    dec_v3.load_state_dict(load_file(os.path.join(BAKED, "decoder.safetensors")))

    # sample_posterior defaults to False -> deterministic, so the two decoders
    # see the identical latent.
    z = enc(x)
    assert torch.equal(z.coords, z_pbr.coords), (
        f"latent/pbr coord mismatch: {tuple(z.coords.shape)} vs {tuple(z_pbr.coords.shape)}. "
        "The PBR latent must share the emission latent's coordinates.")

    y_v2 = dec_v2(z)
    y_v3 = dec_v3(z.replace(torch.cat([z.feats, z_pbr.feats], dim=-1)))

    res = {
        "n_shapes": 4,
        "n_voxels": int(x.feats.shape[0]),
        "n_latent_tokens": int(z.feats.shape[0]),
        "v2_l1": (target - y_v2.feats[:, :3]).abs().mean().item(),
        "v3_l1": (target - y_v3.feats[:, :3]).abs().mean().item(),
        "max_abs_diff": (y_v2.feats - y_v3.feats).abs().max().item(),
        "l1_delta": abs((target - y_v2.feats[:, :3]).abs().mean().item()
                        - (target - y_v3.feats[:, :3]).abs().mean().item()),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))

    ok = res["max_abs_diff"] < 1e-5 and res["l1_delta"] < 1e-7
    print("PASS: v3 decodes identically to v2 at step 0" if ok
          else "FAIL: v3 is not a step-0 no-op — the bake is wrong")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
