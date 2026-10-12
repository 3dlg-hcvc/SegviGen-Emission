"""Dump SegviGen-256 predictions as npz the shared point-sampled evaluator can score.

The live training run (`segvigen_256_bw_6gpu`, job 258105) is 256-native: it was
trained on our own `trellis2_bw/latents_v2` 256 shape/pbr latents plus the emission
mask npz at resolution 256. Feeding it `dataset_direct`'s 512-upsampled conditioning
is a real train/eval mismatch, so `dump_pred_voxels_eval5.py` (which is built around
dataset_direct at 512) is the wrong entry point for this lane.

This script reuses the owner's input roots and model construction verbatim by
importing `live_eval_one`, and differs from it in exactly one way: instead of
computing a voxel IoU inline against the decoded GT, it writes the prediction to
`<out_dir>/draw<k>/<sid>.npz` with keys

    coords  int16   [N, 3]  voxel indices on the 256 grid
    pred    float16 [N, 3]  predicted emission MASK, nominal [0, 1]

`pred` is [N, 3] but it is NOT a colour field. This lane trains on a binarised
mask painted white/black into the base_color slot of the stock TRELLIS.2 tex
encoder (`build_output_tex_latents_256.py`: `is_emis = (emissive > 0).any(-1)`,
`out_base_color = white if is_emis else black`) -- the same channel-hijack
convention the 512 ep8 lane uses. Measured on 33 test shapes the three channels
are the same signal: corr(R,G)=0.999, corr(R,B)=0.995, mean absolute channel
spread 0.027. So it must be RENDERED as albedo x mask (`--pred_masks`), never as
`--pred_rgb`, and it must be DESCRIBED as mask-predicting, not colour-predicting.

which is the format `evaluation.newdata_eval.batch_resample_voxels` produces and
`evaluation.pointsampled.pipeline --voxel_pred_root` consumes. Scoring then runs
through the unmodified shared evaluator, so this lane's number is comparable with
every other lane's. No resampling step: the predictions are already on the 256 grid.
"""
import argparse
import os
import sys
import time

# Gate copy: PROBE is the tree under test (SEGVIGEN_PROBE); everything else is the parent's file
# ../evaluation/newdata_eval/dump_segvigen256_voxels.py unchanged.
PROBE = os.environ.get("SEGVIGEN_PROBE") or "/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe"
if not os.path.isdir(PROBE):                      # workstation mount
    PROBE = "/cs" + PROBE
sys.path.insert(0, os.path.join(PROBE, "scripts_segvigen"))

import numpy as np
import torch

import live_eval_one as leo                        # input roots, load_cond, RESOLUTION

# The owner's COND_DIRS covers cond_fp16, train_72k_agentic and val_72k -- his loop only
# ever evaluates validation shapes. Ours also runs the TEST split and the picked56 study
# set, whose conditioning lives in dataset_direct/test_72k, so without this every test
# shape resolves to "no conditioning file" and is skipped: 0 of 204 on test and exactly
# the 34 val members of picked56 on the study set (2026-08-27). find_cond_dir re-reads
# this list on every call, so appending is enough.
_TEST_COND = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/test_72k"
if not os.path.isdir(_TEST_COND):
    _TEST_COND = "/cs" + _TEST_COND
if _TEST_COND not in leo.COND_DIRS:
    leo.COND_DIRS.append(_TEST_COND)
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg
from eval_emissive import load_eval_models
from collections import OrderedDict


def resolve(p):
    return p if os.path.exists(p) else ("/cs" + p if os.path.exists("/cs" + p) else p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridged_ckpt", required=True, help="output of unrekey_for_eval.py")
    ap.add_argument("--sids_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    # Checkpoints trained on the released dataset read the release's own inputs
    # (jupiter datasets/dataset_release_latents/), not the owner's 2026-07 roots.
    ap.add_argument("--shape_latent_dir", default=None, help="replaces live_eval_one.SHAPE_LATENT_DIR")
    ap.add_argument("--pbr_latent_dir", default=None, help="replaces live_eval_one.PBR_LATENT_DIR")
    ap.add_argument("--cond_dir", default=None,
                    help="the ONLY conditioning root searched (<dir>/<sid>/cond_pos.pth), replacing COND_DIRS")
    args = ap.parse_args()
    if args.shape_latent_dir:
        leo.SHAPE_LATENT_DIR = args.shape_latent_dir
    if args.pbr_latent_dir:
        leo.PBR_LATENT_DIR = args.pbr_latent_dir
    if args.cond_dir:
        leo.COND_DIRS[:] = [args.cond_dir]
    print(f"[dump] inputs: shape={leo.SHAPE_LATENT_DIR} pbr={leo.PBR_LATENT_DIR} cond={leo.COND_DIRS}", flush=True)

    device = "cuda"
    t0 = time.time()
    sids = [s.strip() for s in open(args.sids_file) if s.strip()]
    n_total = len(sids)
    sids = sids[args.shard::args.n_shards]
    print(f"[dump] shard {args.shard}/{args.n_shards}: {len(sids)} of {n_total} shapes, "
          f"{args.draws} draw(s), {args.steps} sampler steps", flush=True)

    flow = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.bridged_ckpt, map_location=device, weights_only=False)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"[dump] REFUSING: state dict did not load cleanly "
                         f"({len(missing)} missing, {len(unexpected)} unexpected)")
    gen.eval()

    md = load_eval_models(device)
    tex_decoder, shape_decoder, sampler = md["tex_decoder"], md["shape_decoder"], md["sampler"]
    sm, ss, tm, ts = md["sm"], md["ss"], md["tm"], md["ts"]
    sp_params = dict(md["pipeline_args"]["tex_slat_sampler"]["params"])
    sp_params["steps"] = args.steps
    shape_decoder.set_resolution(leo.RESOLUTION)
    t_loaded = time.time()
    print(f"[dump] MODELS_READY {t_loaded - t0:.1f}s", flush=True)

    for k in range(args.draws):
        os.makedirs(os.path.join(args.out_dir, f"draw{k}"), exist_ok=True)

    n_ok = n_skip = 0
    t_infer0 = time.time()
    for i, sid in enumerate(sids):
        shp_p = resolve(os.path.join(leo.SHAPE_LATENT_DIR, f"{sid}.npz"))
        pbr_p = resolve(os.path.join(leo.PBR_LATENT_DIR, f"{sid}.npz"))
        # NOT emission: the owner's loop decodes the GT emission latent to score
        # inline, but our GT comes from the atlas via the shared evaluator, so this
        # lane needs no emission latent. Requiring one would skip every test shape
        # (the 256 emission npz exist for train+val only, 0 of the 204 test shapes).
        cond, cond_src = leo.load_cond(sid, device)
        missing_inputs = [p for p in (shp_p, pbr_p) if not os.path.exists(p)]
        if missing_inputs or cond is None:
            n_skip += 1
            print(f"[dump] {sid} SKIPPED (missing inputs or conditioning)", flush=True)
            continue

        shp_raw, pbr_raw = np.load(shp_p), np.load(pbr_p)
        assert np.array_equal(shp_raw["coords"], pbr_raw["coords"]), f"{sid}: shape/pbr coord mismatch"
        coords = torch.from_numpy(shp_raw["coords"].copy()).int().to(device)
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=device), coords], -1)

        shp_feats = torch.from_numpy(shp_raw["feats"].copy()).float().to(device)
        pbr_feats = torch.from_numpy(pbr_raw["feats"].copy()).float().to(device)
        shp_n = sp.SparseTensor((shp_feats - sm) / ss, coords)
        itx_n = sp.SparseTensor((pbr_feats - tm) / ts, coords)
        cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}

        with torch.no_grad():
            _, subs = shape_decoder(sp.SparseTensor(shp_feats, coords), return_subs=True)

        # Per-shape seeds fixed by shape id, identical to the owner's loop, so a
        # checkpoint is always asked the same question about a given shape.
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)
        for k in range(args.draws):
            torch.manual_seed(base_seed + k)
            np.random.seed(args.seed)
            with torch.no_grad():
                noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
                out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
                pred_vox = tex_decoder(out * ts + tm, guide_subs=subs) * 0.5 + 0.5
            vc = pred_vox.coords[:, 1:].cpu().numpy().astype(np.int16)
            pv = pred_vox.feats[:, :3].float().cpu().numpy().astype(np.float16)
            dst = os.path.join(args.out_dir, f"draw{k}", f"{sid}.npz")
            np.savez(dst + ".tmp.npz", coords=vc, pred=pv)
            os.replace(dst + ".tmp.npz", dst)
        n_ok += 1
        print(f"[dump] {i+1}/{len(sids)} {sid} n_vox={vc.shape[0]} "
              f"elapsed={time.time() - t_infer0:.0f}s", flush=True)

    t_end = time.time()
    print(f"[dump] INFER_DONE {t_end - t_infer0:.1f}s for {n_ok} shapes "
          f"({(t_end - t_infer0) / max(n_ok, 1):.2f}s/shape-draw-set, {n_skip} skipped)", flush=True)
    print(f"DUMP_DONE shard={args.shard} n_ok={n_ok} n_skip={n_skip} "
          f"load_s={t_loaded - t0:.1f} infer_s={t_end - t_infer0:.1f} total_s={t_end - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
