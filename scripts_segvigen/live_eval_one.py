"""Evaluate ONE training checkpoint on the owner's first 10 shapes, 5 draws each.

This is the worker of the live evaluation loop (board live_eval_loop). The
driver (live_eval_driver.py) notices a new checkpoint, stages it onto falas, and
submits one of these per checkpoint. This script does the whole per-checkpoint
job in a single process, on one A40:

  1. bridge the lightning checkpoint into the standalone format the evaluation
     machinery loads (scripts_segvigen/unrekey_for_eval.py's logic, imported,
     not re-derived),
  2. sample 5 draws for each of the 10 shapes with per-shape seeds fixed by
     shape id, so every checkpoint is asked the SAME 50 questions and the curve
     over checkpoints is a controlled comparison rather than a reseeded one,
  3. score voxel IoU at 0.5 against the ground-truth emission each shape's own
     target latent decodes to,
  4. paint one thumbnail per draw plus a geometry and a ground-truth panel per
     shape,
  5. write one record file, which the page builder reads.

The 256-native lane, not dataset_direct: this checkpoint was trained on
Dongchen's 256 shape/pbr latents plus our own emission npz at resolution 256.
Feeding it dataset_direct's 512-upsampled conditioning would be a real
train/eval mismatch and could make a healthy checkpoint look like noise. Same
reasoning, same three data roots, and the same stock normalisation statistics as
scripts_segvigen/sanity_256_ckpt.py, which proved this lane end to end on job
258112.

Usage (GPU node, trellis2 env; the sbatch wrapper is live_eval_one.sbatch):
    python scripts_segvigen/live_eval_one.py \
        --lightning_ckpt /3dlg-falas/.../live_eval/staging/step0002265-0.3219.ckpt \
        --step 2265 --val_loss 0.3219 \
        --store /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict

SEGVIGEN = "/3dlg-falas/project/omages/lightgen/segvigen_emissive"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(SEGVIGEN, "emissive", "eval"))
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import numpy as np
import torch
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg
from eval_emissive import load_eval_models   # decoders + shipped norm stats + sampler

import live_eval_thumbs as thumbs
from unrekey_for_eval import unrekey

RESOLUTION = 256              # native, NOT dataset_direct's 512
THR = 0.5
SHAPE_LATENT_DIR = ("/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2/"
                    "shape_latents/shape_enc_next_dc_f16c32_fp16_256")
PBR_LATENT_DIR = ("/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2/"
                  "pbr_latents/tex_enc_next_dc_f16c32_fp16_256")
EMISSION_NPZ_DIR = (f"{SEGVIGEN}/outputs/latents_256/output_tex_latents/"
                    "segvigen_emissive_mask_256")
COND_DIRS = [
    f"{SEGVIGEN}/outputs/cond_fp16",
    "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/train_72k_agentic",
    "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/val_72k",
]
DEFAULT_SIDS_FILE = (f"{SEGVIGEN}/xgconsole/web/_preview/sample56_latest/"
                     "dataset_filtering_selected_56.txt")
VAL_SPLIT_DIR = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/val_72k"

# Shipped TRELLIS.2 pipeline.json stock statistics, the same values
# segvigen_emissive_256.yaml trains with, copied verbatim.
SHAPE_NORM = {
    "mean": [0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252, 1.518149, -0.933218,
             -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
             0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
             -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944, -3.186688, 1.578665],
    "std": [5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237, 5.020802, 5.444004,
            5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
            4.805265, 5.124013, 5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
            5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691],
}


def find_cond_dir(sid):
    for d in COND_DIRS:
        p = os.path.join(d, sid)
        if os.path.exists(os.path.join(p, "cond_pos.pth")) or os.path.exists(os.path.join(p, "cond.pth")):
            return p
    return None


def load_cond(sid, device):
    d = find_cond_dir(sid)
    if d is None:
        return None, None
    cp = os.path.join(d, "cond_pos.pth")
    path = cp if os.path.exists(cp) else os.path.join(d, "cond.pth")
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        obj = obj["cond"]
    obj = obj.float()
    obj = obj.squeeze(0) if obj.dim() == 3 else obj
    return obj.unsqueeze(0), d


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lightning_ckpt", required=True)
    ap.add_argument("--step", type=int, required=True, help="optimizer step, from the filename")
    ap.add_argument("--val_loss", type=float, default=None, help="val loss, from the filename")
    ap.add_argument("--store", required=True, help="live_eval store root on falas")
    ap.add_argument("--sids_file", default=DEFAULT_SIDS_FILE)
    ap.add_argument("--sids_dir", default=None,
                    help="score every shape in this directory instead of --sids_file; this is "
                         "how the full validation tier is pointed at val_72k")
    ap.add_argument("--n", type=int, default=10,
                    help="how many shapes, taken from the top of the list; 0 means all of them")
    ap.add_argument("--tier", default="quick", choices=["quick", "full_val"],
                    help="quick: the ten-shape screen that runs on every epoch. full_val: the "
                         "whole validation split, which is what checkpoint selection is judged on")
    ap.add_argument("--no_thumbs", action="store_true",
                    help="score without painting panels; the full tier uses this, since two "
                         "thousand panels per checkpoint would swamp the page and the disk")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--steps", type=int, default=12, help="sampler steps per draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thumb_px", type=int, default=192)
    ap.add_argument("--ema", default=None, help="pull this EMA rate instead of the raw weights")
    ap.add_argument("--keep_ckpt", action="store_true",
                    help="keep the checkpoint after a successful evaluation. The snapshot "
                         "lane always passes this: a snapshot of last.ckpt is the ONLY copy "
                         "of that epoch, because the training run overwrites its own last "
                         "pointer and keeps no per-epoch file of its own")
    args = ap.parse_args()
    device = "cuda"

    store = args.store
    rec_dir = os.path.join(store, "records")
    img_dir = os.path.join(store, "img")
    step_img_dir = os.path.join(img_dir, f"step{args.step:07d}")
    dirs = [rec_dir, img_dir] + ([] if args.no_thumbs else [step_img_dir])
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    if args.sids_dir:
        sids_all = sorted(d for d in os.listdir(args.sids_dir)
                          if os.path.isdir(os.path.join(args.sids_dir, d)))
        source = args.sids_dir
    else:
        sids_all = [l.strip() for l in open(args.sids_file) if l.strip()]
        source = args.sids_file
    sids = sids_all if args.n <= 0 else sids_all[: args.n]
    print(f"[live_eval] checkpoint step {args.step}, tier {args.tier}, {len(sids)} shapes "
          f"x {args.draws} draws, from {source}", flush=True)

    t_start = time.time()
    # The bridged denoiser is a 2.6GB intermediate that only this process reads.
    # It goes on the node's own disk, never on falas: a write-then-read of it
    # over NFS costs minutes per checkpoint and this loop runs every 25 minutes.
    scratch = os.environ.get("SLURM_TMPDIR") or "/tmp"
    bridged = os.path.join(scratch, f"live_eval_bridged_step{args.step:07d}.pt")
    unrekey(args.lightning_ckpt, bridged, ema_rate=args.ema)

    flow = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(bridged, map_location=device, weights_only=False)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print(f"[live_eval] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    if missing or unexpected:
        # A checkpoint whose weights do not fully land is not a checkpoint we can
        # report a number for: fail loudly rather than publish a plausible IoU.
        raise SystemExit(f"[live_eval] REFUSING to score: state dict did not load cleanly "
                         f"({len(missing)} missing, {len(unexpected)} unexpected)")
    gen.eval()
    os.remove(bridged)

    md = load_eval_models(device)
    tex_decoder, shape_decoder, sampler = md["tex_decoder"], md["shape_decoder"], md["sampler"]
    sm, ss, tm, ts = md["sm"], md["ss"], md["tm"], md["ts"]
    sp_params = dict(md["pipeline_args"]["tex_slat_sampler"]["params"])
    sp_params["steps"] = args.steps
    shape_decoder.set_resolution(RESOLUTION)
    t_loaded = time.time()
    print(f"[live_eval] models ready in {t_loaded - t_start:.0f}s", flush=True)

    per_shape = []
    for sid in sids:
        shp_p = os.path.join(SHAPE_LATENT_DIR, f"{sid}.npz")
        pbr_p = os.path.join(PBR_LATENT_DIR, f"{sid}.npz")
        emit_p = os.path.join(EMISSION_NPZ_DIR, f"{sid}.npz")
        missing_inputs = [p for p in (shp_p, pbr_p, emit_p) if not os.path.exists(p)]
        cond, cond_src = load_cond(sid, device)
        if missing_inputs or cond is None:
            why = ("missing " + ", ".join(os.path.basename(os.path.dirname(p)) for p in missing_inputs)
                   if missing_inputs else "no conditioning file")
            per_shape.append({"sid": sid, "status": "skipped", "reason": why})
            print(f"[live_eval] {sid} SKIPPED: {why}", flush=True)
            continue

        shp_raw, pbr_raw, emit_raw = np.load(shp_p), np.load(pbr_p), np.load(emit_p)
        assert np.array_equal(shp_raw["coords"], pbr_raw["coords"]), f"{sid}: shape/pbr coord mismatch"
        assert np.array_equal(shp_raw["coords"], emit_raw["coords"]), f"{sid}: shape/emission coord mismatch"
        coords = torch.from_numpy(shp_raw["coords"].copy()).int().to(device)
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=device), coords], -1)

        shp_feats = torch.from_numpy(shp_raw["feats"].copy()).float().to(device)
        pbr_feats = torch.from_numpy(pbr_raw["feats"].copy()).float().to(device)
        emit_feats = torch.from_numpy(emit_raw["feats"].copy()).float().to(device)
        shp_n = sp.SparseTensor((shp_feats - sm) / ss, coords)
        itx_n = sp.SparseTensor((pbr_feats - tm) / ts, coords)
        cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}

        with torch.no_grad():
            _, subs = shape_decoder(sp.SparseTensor(shp_feats, coords), return_subs=True)
            gt_vox = tex_decoder(sp.SparseTensor(emit_feats, coords), guide_subs=subs) * 0.5 + 0.5
        gt_e = gt_vox.feats[:, :3].mean(-1) > THR
        vox_coords = gt_vox.coords[:, 1:].cpu().numpy().astype(np.int32)
        gt_e_np = gt_e.cpu().numpy()

        # One projection per shape, shared by every panel of that shape and by
        # every checkpoint that ever evaluates it, so cells stay comparable and
        # seven panels cost barely more than one.
        view = None
        if not args.no_thumbs:
            view = thumbs.ShapeView(vox_coords, size=args.thumb_px)
            for name, mask in (("geom", None), ("gt", gt_e_np)):
                p = os.path.join(img_dir, f"{sid}_{name}.png")
                if not os.path.exists(p):
                    thumbs.save_png(view.render(mask), p)

        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)
        ious, fracs, seeds = [], [], []
        for k in range(args.draws):
            seed_k = base_seed + k
            seeds.append(seed_k)
            torch.manual_seed(seed_k)
            np.random.seed(args.seed)
            with torch.no_grad():
                noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
                out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
                pred_vox = tex_decoder(out * ts + tm, guide_subs=subs) * 0.5 + 0.5
            pe = pred_vox.feats[:, :3].mean(-1) > THR
            inter = (pe & gt_e).sum().item()
            union = (pe | gt_e).sum().item()
            ious.append(inter / union if union > 0 else 1.0)
            fracs.append(float(pe.float().mean().item()))
            if view is not None:
                thumbs.save_png(view.render(pe.cpu().numpy()),
                                os.path.join(step_img_dir, f"{sid}_d{k}.png"))

        per_shape.append({
            "sid": sid, "status": "ok",
            "gt_frac": float(gt_e.float().mean().item()),
            "n_voxels": int(vox_coords.shape[0]),
            "seeds": seeds,
            "iou_per_draw": [float(v) for v in ious],
            "iou_mean": float(np.mean(ious)), "iou_std": float(np.std(ious)),
            "pred_frac_per_draw": fracs, "pred_frac_mean": float(np.mean(fracs)),
            "cond_src": os.path.basename(os.path.dirname(cond_src)),
        })
        print(f"[live_eval] {sid} gt_frac={per_shape[-1]['gt_frac']:.4f} "
              f"iou={np.mean(ious):.4f}+-{np.std(ious):.4f} "
              f"pred_frac={np.mean(fracs):.4f} elapsed={time.time() - t_loaded:.0f}s", flush=True)

    scored = [s for s in per_shape if s["status"] == "ok"]
    record = {
        "step": args.step,
        "tier": args.tier,
        "val_loss": args.val_loss,
        "status": "ok" if scored else "failed",
        "ckpt_name": os.path.basename(args.lightning_ckpt),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "ema": args.ema,
        "n_shapes": len(sids), "n_scored": len(scored), "draws": args.draws,
        "sampler_steps": args.steps, "threshold": THR, "resolution": RESOLUTION,
        "mean_iou": float(np.mean([s["iou_mean"] for s in scored])) if scored else None,
        "mean_pred_frac": float(np.mean([s["pred_frac_mean"] for s in scored])) if scored else None,
        "n_all_zero_draws": int(sum(1 for s in scored for f in s["pred_frac_per_draw"] if f == 0.0)),
        "n_all_on_draws": int(sum(1 for s in scored for f in s["pred_frac_per_draw"] if f == 1.0)),
        "elapsed_s": time.time() - t_start,
        "per_shape": per_shape,
    }
    suffix = "" if args.tier == "quick" else f"_{args.tier}"
    atomic_write_json(record, os.path.join(rec_dir, f"step{args.step:07d}{suffix}.json"))
    print(f"[live_eval] step {args.step} ({args.tier}): mean IoU {record['mean_iou']} over "
          f"{len(scored)} shapes, {record['elapsed_s']:.0f}s", flush=True)

    if not args.keep_ckpt and scored:
        try:
            os.remove(args.lightning_ckpt)
            print(f"[live_eval] removed the staged checkpoint {args.lightning_ckpt}", flush=True)
        except OSError as e:
            print(f"[live_eval] could not remove the staged checkpoint: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
