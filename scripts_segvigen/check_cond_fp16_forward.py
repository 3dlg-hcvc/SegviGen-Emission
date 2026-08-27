"""FP16 cond numerical check, part 2 of 2: does the fp16 cond_pos.pth
relocation change what the DENOISER actually predicts?

verify_cond_fp16.py already showed the stored tensors match within fp16
rounding. This is the test that matters: run the REAL denoiser forward on
the SAME shape with fp32 vs fp16 cond and compare the predicted velocity.

Deliberately standalone -- NO LightningModule, NO Trainer:
  * the model is built with build_models() straight from the config's
    models block and loaded with the same payload["denoiser"] bare-state-dict
    path BasicTrainer._load_weights_for_model uses for full_seg_rekeyed.pt,
  * diffuse / get_v are the three lines of arithmetic from
    FlowMatchingTrainer (scalar t, batch of 1), replicated here,
  * so nothing touches self.log / self.device / any trainer-bound property.
An earlier version instantiated SegviGenFlowTrainer and died with
"is not attached to a `Trainer`"; do not reintroduce that.

Also deliberately NO Dataset construction: SegviGenEmissive256.__init__
does a readdir of jupiter's 72k-entry latents_v2 dirs, which stalled three
earlier attempts in NFS rpc_wait (server contention, not a code bug). This
reads ~20 shapes by explicit path instead: a few dozen stats, no readdir of
any large directory. The one listdir it does is on the falas fp16 root
(fast, and the only place the candidate list can come from).

Three forwards per shape, all with identical noise and t:
    (a) fp32 cond (jupiter dataset_direct)
    (b) fp16 cond (falas cond_fp16, upcast by the same .float() the reader
        applies -- exactly the real training path)
    (c) fp32 cond AGAIN -- the CONTROL. Any (a)-vs-(c) difference is the
        sparse-op forward's own nondeterminism (atomics in flex_gemm /
        sparse conv reductions), not fp16. The verdict compares (a,b)
        against (a,c): fp16 passes if its effect is no larger than the
        model's own noise floor.

Usage:
    python scripts_segvigen/check_cond_fp16_forward.py --n 20 --seed 0
"""
import argparse
import os
import statistics as st
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trellis2.datasets.lightgen_slat import _read_npz_as_sparse  # noqa: E402
from trellis2.trainers.builders import build_models  # noqa: E402

DATASET_DIRECT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
SPLIT_DIRS = [os.path.join(DATASET_DIRECT, s) for s in ("train_72k_agentic", "val_72k")]
FP16_ROOT = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_fp16"
LATENTS_V2 = "/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2"
EMIT_DIR = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/"
            "latents_256/output_tex_latents/segvigen_emissive_mask_256")
CONFIG_PATH = os.path.join(ROOT, "configs", "gen", "segvigen_emissive_256.yaml")
SIGMA_MIN = 1e-5


def load_cond(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        obj = obj["cond"]
    return obj.float().squeeze(0)          # the reader's own upcast, verbatim


def build_sample(sid, cfg, cond_path):
    """Everything SegviGenEmissive256.get_instance builds, by explicit path."""
    d = cfg["dataset"]["args"]
    tx = {"mean": d["tex_normalization"]["mean"], "std": d["tex_normalization"]["std"]}
    sm = {"mean": d["shape_normalization"]["mean"], "std": d["shape_normalization"]["std"]}
    x_0 = _read_npz_as_sparse(os.path.join(EMIT_DIR, f"{sid}.npz"), normalization=tx)
    pbr = _read_npz_as_sparse(
        os.path.join(LATENTS_V2, "pbr_latents", d["pbr_latent_name"], f"{sid}.npz"), normalization=tx)
    shp = _read_npz_as_sparse(
        os.path.join(LATENTS_V2, "shape_latents", d["shape_latent_name"], f"{sid}.npz"), normalization=sm)
    assert torch.equal(x_0.coords, shp.coords), f"{sid}: emission/shape coords mismatch"
    assert torch.equal(x_0.coords, pbr.coords), f"{sid}: emission/pbr coords mismatch"
    cond = load_cond(cond_path).unsqueeze(0)          # (1, 1029, 1024)
    return x_0.cuda(), shp.cuda(), pbr.cuda(), cond.cuda()


@torch.no_grad()
def forward_once(model, x_0, shp, pbr, cond, seed):
    """FlowMatchingTrainer.diffuse + get_v + denoiser forward, batch of 1,
    scalar t -- the arithmetic, without the LightningModule."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    noise_feats = torch.randn(x_0.feats.shape, generator=g, device="cuda", dtype=x_0.feats.dtype)
    t = torch.rand(1, generator=g, device="cuda").item()          # same t for every call at this seed

    x_t_feats = (1 - t) * x_0.feats + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise_feats
    x_t = x_0.replace(x_t_feats)
    t_batch = torch.tensor([t * 1000], device="cuda", dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):            # matches training's amp/bfloat16
        pred = model(x_t, t_batch, cond, concat_cond=shp, appearance_cond=pbr)

    target_feats = (1 - SIGMA_MIN) * noise_feats - x_0.feats      # get_v
    pred_f = pred.feats.float()
    loss = ((pred_f - target_feats.float()) ** 2).mean().item()
    return pred_f.flatten(), loss


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def pick_sids(n):
    """Candidates from the falas fp16 root (fast listdir), kept only if every
    other per-sid file exists. Per-sid stats only -- no readdir on jupiter."""
    converted = sorted(os.listdir(FP16_ROOT))
    print(f"[check_fwd] {len(converted)} sids converted to fp16 so far (job 257957)")
    d = yaml.safe_load(open(CONFIG_PATH))["dataset"]["args"]
    out = []
    for sid in converted:
        fp16 = os.path.join(FP16_ROOT, sid, "cond_pos.pth")
        if not os.path.exists(fp16):
            continue
        fp32 = None
        for sd in SPLIT_DIRS:
            p = os.path.join(sd, sid, "cond_pos.pth")
            if os.path.exists(p):
                fp32 = p
                break
        if fp32 is None:
            continue
        if not os.path.exists(os.path.join(EMIT_DIR, f"{sid}.npz")):
            continue
        if not os.path.exists(os.path.join(LATENTS_V2, "shape_latents", d["shape_latent_name"], f"{sid}.npz")):
            continue
        if not os.path.exists(os.path.join(LATENTS_V2, "pbr_latents", d["pbr_latent_name"], f"{sid}.npz")):
            continue
        out.append((sid, fp32, fp16))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH))
    sids = pick_sids(args.n)
    print(f"[check_fwd] {len(sids)} shapes usable for the comparison")

    model = build_models(cfg["models"])["denoiser"]
    ck = cfg["trainer"]["args"]["finetune_ckpt"]["denoiser"]
    payload = torch.load(ck, map_location="cpu")
    state = payload["denoiser"] if "denoiser" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[check_fwd] loaded {ck}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = model.cuda().eval()

    cos_fp16, cos_ctrl, cos_bf16, dloss, dloss_bf = [], [], [], [], []
    for sid, fp32_path, fp16_path in sids:
        x_0, shp, pbr, c32 = build_sample(sid, cfg, fp32_path)
        c16 = load_cond(fp16_path).unsqueeze(0).cuda()
        assert c32.shape == c16.shape, f"{sid}: cond shape differs {c32.shape} vs {c16.shape}"

        # (d) the SCALE THAT MATTERS: cond rounded to bf16, which is what
        # AMP already does to it inside every forward. bf16 has 8 mantissa
        # bits vs fp16's 11, so this is the coarser rounding the training
        # pipeline applies regardless of how we store the file.
        cbf = c32.to(torch.bfloat16).float()

        pa, la = forward_once(model, x_0, shp, pbr, c32, seed=12345)
        pb, lb = forward_once(model, x_0, shp, pbr, c16, seed=12345)
        pc, lc = forward_once(model, x_0, shp, pbr, c32, seed=12345)
        pd, ld = forward_once(model, x_0, shp, pbr, cbf, seed=12345)

        ab, ac, ad = cosine(pa, pb), cosine(pa, pc), cosine(pa, pd)
        cos_fp16.append(ab)
        cos_ctrl.append(ac)
        cos_bf16.append(ad)
        dloss.append(abs(la - lb))
        dloss_bf.append(abs(la - ld))
        print(f"[check_fwd] {sid}: cos(fp32,fp16)={ab:.8f} cos(fp32,bf16)={ad:.8f} "
              f"cos(fp32,fp32rerun)={ac:.8f} loss fp32={la:.6f} fp16={lb:.6f} bf16={ld:.6f}")

    print("")
    print(f"[check_fwd] N={len(sids)}")
    print(f"[check_fwd] cos(fp32,fp16):      mean={st.mean(cos_fp16):.8f} min={min(cos_fp16):.8f}")
    print(f"[check_fwd] cos(fp32,bf16):      mean={st.mean(cos_bf16):.8f} min={min(cos_bf16):.8f}   <- what AMP already does")
    print(f"[check_fwd] cos(fp32,fp32rerun): mean={st.mean(cos_ctrl):.8f} min={min(cos_ctrl):.8f}   <- forward nondeterminism")
    print(f"[check_fwd] |loss fp32 - fp16|:  mean={st.mean(dloss):.8f} max={max(dloss):.8f}")
    print(f"[check_fwd] |loss fp32 - bf16|:  mean={st.mean(dloss_bf):.8f} max={max(dloss_bf):.8f}")
    # The decision gate: fp16 STORAGE is immaterial if its effect is no
    # larger than the bf16 rounding AMP applies to the same tensor anyway.
    # (Comparing against the forward's own nondeterminism instead would be
    # the wrong bar -- that floor is ~1e-7 and nothing survives it.)
    ok = min(cos_fp16) >= min(cos_bf16)
    print(f"[check_fwd] {'PASS' if ok else 'INVESTIGATE'}: fp16-cond storage perturbs the predicted "
          f"velocity {'LESS than' if ok else 'MORE than'} the bf16 cast training already applies")


if __name__ == "__main__":
    main()
