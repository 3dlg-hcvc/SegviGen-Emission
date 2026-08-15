"""Step-0 forward-only probe: what train/l1 does the pbr->emission v3 VAE log at
init under different weight-transfer policies?

No training. One fixed batch, four policies, same data, same seed.
"""
import os, sys, glob, gc, json, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2")

import torch, numpy as np
from safetensors.torch import load_file
import o_voxel as _o_voxel
from trellis2.modules import sparse as sp
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder

SNAP = "/localhome/dya78/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts"
ROOT = "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2/data/lightgen_1k"
DEV = "cuda"
PAD = torch.tensor([-1.0, 1.0, 1.0])          # PAD_METALLIC, PAD_ROUGHNESS, PAD_ALPHA

COMMON_ENC = dict(model_channels=[64,128,256,512,1024], latent_channels=32,
                  num_blocks=[0,4,8,16,4],
                  block_type=["SparseConvNeXtBlock3d"]*5,
                  down_block_type=["SparseResBlockS2C3d"]*4,
                  block_args=[{}]*5, use_fp16=False)
COMMON_DEC = dict(model_channels=[1024,512,256,128,64],
                  num_blocks=[4,16,8,4,0],
                  block_type=["SparseConvNeXtBlock3d"]*5,
                  up_block_type=["SparseResBlockC2S3d"]*4,
                  block_args=[{}]*5, use_fp16=False, pred_subdiv=False)


def load_batch(n_shapes, max_vox):
    files = sorted(glob.glob(os.path.join(ROOT, "emission_voxels_256", "*.vxz")))
    feats, coords = [], []
    picked = []
    bi = 0
    for f in files:
        sha = os.path.basename(f)[:-4]
        p = os.path.join(ROOT, "pbr_voxels_256", sha + ".vxz")
        if not os.path.exists(p):
            continue
        c, a = _o_voxel.io.read_vxz(f, num_threads=4)
        if c.shape[0] > max_vox:
            continue
        _, ap = _o_voxel.io.read_vxz(p, num_threads=4)
        e = a["emissive"].float()/255.0*2.0 - 1.0
        pb = torch.cat([ap[k].float()/255.0*2.0 - 1.0
                        for k in ("base_color","metallic","roughness","alpha")], -1)
        feats.append(torch.cat([e, pb], -1))
        coords.append(torch.cat([torch.full((c.shape[0],1), bi, dtype=c.dtype), c], -1))
        picked.append((sha, c.shape[0]))
        bi += 1
        if bi >= n_shapes:
            break
    print("batch:", picked, "total voxels:", sum(v for _, v in picked))
    x = sp.SparseTensor(torch.cat(feats,0).to(DEV), torch.cat(coords,0).to(DEV))
    return x


def build_v3(policy, enc_sd, dec_sd):
    """policy in {skip, patched, patched_nofold, patched_pbrcols}"""
    enc = SparseUnetVaeEncoder(in_channels=3, **COMMON_ENC)
    dec = SparseUnetVaeDecoder(out_channels=3, latent_channels=64, **COMMON_DEC)
    e_f = {k: v for k, v in enc_sd.items() if k in enc.state_dict()
           and tuple(v.shape) == tuple(enc.state_dict()[k].shape)}
    d_f = {k: v for k, v in dec_sd.items() if k in dec.state_dict()
           and tuple(v.shape) == tuple(dec.state_dict()[k].shape)}
    enc.load_state_dict(e_f, strict=False)
    dec.load_state_dict(d_f, strict=False)
    skipped_e = [k for k in enc_sd if k not in e_f]
    skipped_d = [k for k in dec_sd if k not in d_f]
    print(f"  [{policy}] shape-compatible: enc {len(e_f)}/{len(enc_sd)}, dec {len(d_f)}/{len(dec_sd)}; "
          f"skipped enc={skipped_e} dec={skipped_d}")

    if policy != "skip":
        W = enc_sd["input_layer.weight"].float()   # (64,6)
        b = enc_sd["input_layer.bias"].float()     # (64,)
        with torch.no_grad():
            enc.input_layer.weight.copy_(W[:, 0:3])
            if policy == "patched_nofold":
                enc.input_layer.bias.copy_(b)
            else:
                enc.input_layer.bias.copy_(b + W[:, 3:6] @ PAD)
            Wf = dec_sd["from_latent.weight"].float()   # (1024,32)
            dec.from_latent.weight.zero_()
            if policy == "patched_pbrcols":
                dec.from_latent.weight[:, 32:64].copy_(Wf)   # align with z_pbr
            else:
                dec.from_latent.weight[:, 0:32].copy_(Wf)    # align with z_emission
            dec.from_latent.bias.copy_(dec_sd["from_latent.bias"].float())
            dec.output_layer.weight.copy_(dec_sd["output_layer.weight"].float()[0:3])
            dec.output_layer.bias.copy_(dec_sd["output_layer.bias"].float()[0:3])
    return enc.float().eval().to(DEV), dec.float().eval().to(DEV)


@torch.no_grad()
def run_v3(enc, dec, pbr_enc, x, seed):
    torch.manual_seed(seed)
    x_emit = x.replace(x.feats[:, :3])
    z, mean, logvar = enc(x_emit, sample_posterior=True, return_raw=True)
    z_pbr = pbr_enc(x.replace(x.feats[:, 3:]))
    z_comb = z.replace(torch.cat([z.feats, z_pbr.feats], -1))
    y = dec(z_comb)
    tgt = x.feats[:, :3]
    return (tgt - y.feats).abs().mean().item(), y.feats


@torch.no_grad()
def run_albedo_ref(enc6, dec6, x, seed):
    torch.manual_seed(seed)
    emission = x.feats[:, :3]
    pad = PAD.to(x.feats.device).to(x.feats.dtype).unsqueeze(0).expand(emission.shape[0], -1)
    xp = x.replace(torch.cat([emission, pad], -1))
    z, mean, logvar = enc6(xp, sample_posterior=True, return_raw=True)
    y = dec6(z)
    return (emission - y.feats[:, :3]).abs().mean().item(), y.feats[:, :3]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_vox", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    enc_sd = {k: v.float() for k, v in load_file(f"{SNAP}/tex_enc_next_dc_f16c32_fp16.safetensors").items()}
    dec_sd = {k: v.float() for k, v in load_file(f"{SNAP}/tex_dec_next_dc_f16c32_fp16.safetensors").items()}
    x = load_batch(args.n, args.max_vox)
    tgt = x.feats[:, :3]
    print("target emission: mean=%.4f  frac all-(-1)=%.3f" %
          (tgt.mean().item(), (tgt <= -1+1e-6).all(-1).float().mean().item()))
    print("trivial: predict -1 -> L1 %.5f ; predict 0 -> L1 %.5f ; predict base_color -> L1 %.5f" %
          ((tgt+1).abs().mean().item(), tgt.abs().mean().item(),
           (tgt - x.feats[:, 3:6]).abs().mean().item()))

    # frozen pretrained PBR encoder (the conditioning path) -- identical to trainer
    pbr_enc = SparseUnetVaeEncoder(in_channels=6, **COMMON_ENC)
    pbr_enc.load_state_dict({k: v for k, v in enc_sd.items()}, strict=True)
    pbr_enc = pbr_enc.float().eval().to(DEV)

    results = {}
    # reference: albedo->emission chain, full pretrained load
    enc6 = SparseUnetVaeEncoder(in_channels=6, **COMMON_ENC); enc6.load_state_dict(enc_sd, strict=True)
    dec6 = SparseUnetVaeDecoder(out_channels=6, latent_channels=32, **COMMON_DEC); dec6.load_state_dict(dec_sd, strict=True)
    enc6, dec6 = enc6.float().eval().to(DEV), dec6.float().eval().to(DEV)
    l1, _ = run_albedo_ref(enc6, dec6, x, args.seed)
    results["albedo->emission reference (full pretrained, 6ch)"] = l1
    print(f"\n>> albedo->emission reference: train/l1 = {l1:.5f}")
    del enc6, dec6; gc.collect(); torch.cuda.empty_cache()

    for policy in ["skip", "patched", "patched_nofold", "patched_pbrcols"]:
        enc, dec = build_v3(policy, enc_sd, dec_sd)
        l1, _ = run_v3(enc, dec, pbr_enc, x, args.seed)
        results[policy] = l1
        print(f">> v3 policy={policy}: train/l1 = {l1:.5f}")
        del enc, dec; gc.collect(); torch.cuda.empty_cache()

    print("\n===== SUMMARY (step-0 train/l1, forward only) =====")
    for k, v in results.items():
        print(f"  {k:52s} {v:.5f}")
    json.dump(results, open("/tmp/dya_probe/results.json", "w"), indent=2)
