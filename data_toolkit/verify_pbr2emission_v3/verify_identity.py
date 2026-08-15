"""Confirm: patched-transfer v3 == albedo->emission reference, as a function, at init."""
import os, sys, glob, gc, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2")
import torch
from safetensors.torch import load_file
sys.path.insert(0, "/tmp/dya_probe")
from step0_probe import (SNAP, PAD, COMMON_ENC, COMMON_DEC, load_batch, build_v3,
                         run_v3, run_albedo_ref, DEV)
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder

t0 = time.time()
enc_sd = {k: v.float() for k, v in load_file(f"{SNAP}/tex_enc_next_dc_f16c32_fp16.safetensors").items()}
dec_sd = {k: v.float() for k, v in load_file(f"{SNAP}/tex_dec_next_dc_f16c32_fp16.safetensors").items()}

for n, mv in [(6, 25000), (4, 60000)]:
    x = load_batch(n, mv)
    tgt = x.feats[:, :3]
    print("  target: mean=%.4f frac all-(-1)=%.3f  | trivial(-1) L1=%.5f" %
          (tgt.mean().item(), (tgt <= -1+1e-6).all(-1).float().mean().item(),
           (tgt+1).abs().mean().item()))
    pbr_enc = SparseUnetVaeEncoder(in_channels=6, **COMMON_ENC)
    pbr_enc.load_state_dict(enc_sd, strict=True); pbr_enc = pbr_enc.float().eval().to(DEV)
    enc6 = SparseUnetVaeEncoder(in_channels=6, **COMMON_ENC); enc6.load_state_dict(enc_sd, strict=True)
    dec6 = SparseUnetVaeDecoder(out_channels=6, latent_channels=32, **COMMON_DEC); dec6.load_state_dict(dec_sd, strict=True)
    enc6, dec6 = enc6.float().eval().to(DEV), dec6.float().eval().to(DEV)
    encP, decP = build_v3("patched", enc_sd, dec_sd)
    for seed in (0, 1, 2):
        l1r, yr = run_albedo_ref(enc6, dec6, x, seed)
        l1p, yp = run_v3(encP, decP, pbr_enc, x, seed)
        print(f"  n={n} seed={seed}: ref L1={l1r:.8f}  patched L1={l1p:.8f}  "
              f"|dL1|={abs(l1r-l1p):.2e}  max|y_ref - y_patched|={(yr-yp).abs().max().item():.3e}")
    del enc6, dec6, encP, decP, pbr_enc, x; gc.collect(); torch.cuda.empty_cache()
print("wall %.1f s" % (time.time()-t0))
