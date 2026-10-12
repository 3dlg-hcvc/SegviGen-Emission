"""Predict emission masks with SegviGen-Emission on this repo's own code.

    python data_toolkit/predict_emission_mask_voxels.py --run_config <config.yaml> --ckpt <checkpoint> \
        --sids_file <file> --out_dir <dir> --shape_latent_dir <dir> --pbr_latent_dir <dir> \
        --cond_dirs <dir>[,<dir> ...] [--draws 1] [--seed 0] [--steps 12] [--shard i --n_shards n] \
        [--ma_out_dir <dir>]

The denoiser comes from the run's `config.yaml` through this repo's registries (`SegviGenSLatFlowModel`)
and loads a Lightning checkpoint (`models.denoiser.*`) or a bridged file (`gen3dseg.flow_model.*`, what
`scripts_segvigen/unrekey_for_eval.py` writes) strictly. Per shape: the shape and PBR latents of the
256 grid (`<dir>/<sid>.npz`, normalized with the stats of TRELLIS.2-4B's `pipeline.json`), the DINOv3
tokens of the first `--cond_dirs` entry holding `<sid>/cond_pos.pth` (or `cond.pth`), noise from
`torch.manual_seed(seed * 1000003 + int(sid[:8], 16) % 100000 + k)`, `--steps` Euler steps with the
shipped texture sampler (`pipeline.json` `tex_slat_sampler`: guidance 1.0, interval [0.6, 0.9],
rescale_t 3.0), and the stock TRELLIS.2-4B texture decoder guided by the shape decoder's subdivisions
at resolution 256. Output `<out_dir>/draw<k>/<sid>.npz`: `coords` int16 [N, 3] (the 256 grid), `pred`
float16 [N, 3], the predicted emission mask (three equal channels, nominal [0, 1]), scored by the
parent repo's `evaluation/pointsampled/pipeline.py --voxel_pred_root`. With `--ma_out_dir`, also
`<ma_out_dir>/draw<k>/<sid>.npz` = mask x the asset's albedo (the PBR latent through the same decoder
and subdivisions, clamped to [0, 1]), for MAE and PSNR.

This is the published chain (the parent's `dump_segvigen256_voxels.py` with the original SegviGen
repo's `Gen3DSeg`, `Sampler`, `load_eval_models`) on this repo's classes. The decoders run as that
repo's vendored copy ran them (`decode_as_published`); on an A40 in the env of the published dumps, this
script reproduces them byte for byte (`script/internal/gate/`).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PIPELINE_REPO = "microsoft/TRELLIS.2-4B"
TEX_DECODER = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
SHAPE_DECODER = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
RESOLUTION = 256


def denoiser_state(path):
    """Bare denoiser state dict from a Lightning checkpoint or a bridged file."""
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = payload.get("state_dict", payload)
    for prefix in ("models.denoiser.", "gen3dseg.flow_model."):
        bare = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if bare:
            return bare
    raise SystemExit(f"[predict] REFUSING: no denoiser keys (models.denoiser.* or gen3dseg.flow_model.*) in {path}")


def _c2s_forward_published(self, x, subdiv=None):
    # SparseResBlockC2S3d._forward as the original SegviGen repo vendors it, verbatim: a voxel whose eight
    # subdivision logits are all <= 0 gets no child (this package's block gives it one, the arg-max).
    if self.pred_subdiv:
        subdiv = self.to_subdiv(x)
    h = x.replace(self.norm1(x.feats))
    h = h.replace(F.silu(h.feats))
    h = self.conv1(h)
    subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
    h = self.updown(h, subdiv_binarized)
    x = self.updown(x, subdiv_binarized)
    h = h.replace(self.norm2(h.feats))
    h = h.replace(F.silu(h.feats))
    h = self.conv2(h)
    h = h + self.skip_connection(x)
    if self.pred_subdiv:
        return h, subdiv
    else:
        return h


def decode_as_published(decoder):
    """Make one stock TRELLIS.2-4B decoder run as the published chain's vendored copy ran it: the blocks stay fp16
    (as the checkpoint loads them) and take the `from_latent` output cast to fp16 (the copy's `h.type(self.dtype)`;
    this package's decoder keeps the latent's fp32, which its fp16 subdivision head rejects), and the upsample
    blocks give no child to a voxel without a positive subdivision logit. Only this instance changes, and only its
    `forward` (its `upsample_features` casts back to the latent's dtype after every block and fails)."""
    import types
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseResBlockC2S3d
    decoder.blocks[0][0].register_forward_pre_hook(lambda _, args: (args[0].type(decoder.dtype),) + tuple(args[1:]))
    for block in decoder.modules():
        if isinstance(block, SparseResBlockC2S3d):
            block._forward = types.MethodType(_c2s_forward_published, block)
    return decoder


def load_cond(sid, cond_dirs, device):
    for d in cond_dirs:
        p = os.path.join(d, sid)
        cp = os.path.join(p, "cond_pos.pth")
        path = cp if os.path.exists(cp) else os.path.join(p, "cond.pth")
        if os.path.exists(path):
            obj = torch.load(path, map_location=device, weights_only=False)
            if isinstance(obj, dict):
                obj = obj["cond"]
            obj = obj.float()
            obj = obj.squeeze(0) if obj.dim() == 3 else obj
            return obj.unsqueeze(0)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run_config", required=True, help="the run's config.yaml (its models.denoiser block)")
    ap.add_argument("--ckpt", required=True, help="Lightning checkpoint or bridged file")
    ap.add_argument("--sids_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shape_latent_dir", required=True, help="<dir>/<sid>.npz, shape latents of the 256 grid")
    ap.add_argument("--pbr_latent_dir", required=True, help="<dir>/<sid>.npz, PBR latents of the 256 grid")
    ap.add_argument("--cond_dirs", required=True, help="comma-separated, searched in order for <sid>/cond_pos.pth")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--ma_out_dir", default=None, help="also write mask x albedo here")
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        raise SystemExit(f"[predict] REFUSING: --shard {args.shard} is not in [0, --n_shards {args.n_shards})")
    for flag in ("shape_latent_dir", "pbr_latent_dir"):
        if not os.path.isdir(getattr(args, flag)):
            raise SystemExit(f"[predict] REFUSING: --{flag} {getattr(args, flag)} is not a directory")
    cond_dirs = [d for d in args.cond_dirs.split(",") if d]
    if not any(os.path.isdir(d) for d in cond_dirs):
        raise SystemExit(f"[predict] REFUSING: none of --cond_dirs exists: {cond_dirs}")

    from huggingface_hub import hf_hub_download

    import trellis2.modules.sparse as sp
    from trellis2 import models
    from trellis2.pipelines import samplers
    from trellis2.trainers.builders import build_models

    device = "cuda"
    t0 = time.time()
    sids = [s.strip() for s in open(args.sids_file) if s.strip()]
    n_total = len(sids)
    sids = sids[args.shard::args.n_shards]

    cfg = yaml.safe_load(open(args.run_config))
    den = build_models({"denoiser": cfg["models"]["denoiser"]})["denoiser"]
    res = den.load_state_dict(denoiser_state(args.ckpt), strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit(f"[predict] REFUSING: the checkpoint does not load cleanly "
                         f"({len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys)")
    den = den.to(device).eval()

    pa = json.load(open(hf_hub_download(repo_id=PIPELINE_REPO, filename="pipeline.json")))["args"]
    sm = torch.tensor(pa["shape_slat_normalization"]["mean"])[None].to(device)
    ss = torch.tensor(pa["shape_slat_normalization"]["std"])[None].to(device)
    tm = torch.tensor(pa["tex_slat_normalization"]["mean"])[None].to(device)
    ts = torch.tensor(pa["tex_slat_normalization"]["std"])[None].to(device)
    sampler_cfg = pa["tex_slat_sampler"]
    sampler = getattr(samplers, sampler_cfg["name"])(**sampler_cfg["args"])
    sp_params = dict(sampler_cfg["params"])
    sp_params["steps"] = args.steps
    tex_decoder = models.from_pretrained(TEX_DECODER).to(device).eval()
    shape_decoder = models.from_pretrained(SHAPE_DECODER).to(device).eval()
    shape_decoder.set_resolution(RESOLUTION)
    decode_as_published(tex_decoder)
    decode_as_published(shape_decoder)
    for k in range(args.draws):
        os.makedirs(os.path.join(args.out_dir, f"draw{k}"), exist_ok=True)
        if args.ma_out_dir:
            os.makedirs(os.path.join(args.ma_out_dir, f"draw{k}"), exist_ok=True)
    t_ready = time.time()
    print(f"[predict] READY {t_ready - t0:.1f}s, shard {args.shard}/{args.n_shards}: {len(sids)} of {n_total} shapes, "
          f"{args.draws} draw(s), {args.steps} steps, sampler {sampler_cfg['name']} {sp_params}", flush=True)

    def save(dirname, k, sid, coords, feats):
        dst = os.path.join(dirname, f"draw{k}", f"{sid}.npz")
        np.savez(dst + ".tmp.npz", coords=coords, pred=feats)
        os.replace(dst + ".tmp.npz", dst)

    n_ok = n_skip = 0
    for i, sid in enumerate(sids):
        shp_p = os.path.join(args.shape_latent_dir, f"{sid}.npz")
        pbr_p = os.path.join(args.pbr_latent_dir, f"{sid}.npz")
        cond = load_cond(sid, cond_dirs, device)
        if not (os.path.exists(shp_p) and os.path.exists(pbr_p)) or cond is None:
            n_skip += 1
            print(f"[predict] {sid} SKIPPED (missing latents or conditioning)", flush=True)
            continue
        shp_raw, pbr_raw = np.load(shp_p), np.load(pbr_p)
        assert np.array_equal(shp_raw["coords"], pbr_raw["coords"]), f"{sid}: shape/pbr coord mismatch"
        coords = torch.from_numpy(shp_raw["coords"].copy()).int().to(device)
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=device), coords], -1)
        shp_feats = torch.from_numpy(shp_raw["feats"].copy()).float().to(device)
        pbr_feats = torch.from_numpy(pbr_raw["feats"].copy()).float().to(device)
        shp_n = sp.SparseTensor((shp_feats - sm) / ss, coords)
        pbr_n = sp.SparseTensor((pbr_feats - tm) / ts, coords)
        with torch.no_grad():
            _, subs = shape_decoder(sp.SparseTensor(shp_feats, coords), return_subs=True)
            if args.ma_out_dir:
                alb = tex_decoder(sp.SparseTensor(pbr_feats, coords), guide_subs=subs) * 0.5 + 0.5
                alb_c = alb.coords[:, 1:].cpu().numpy().astype(np.int16)
                alb_f = alb.feats[:, :3].clamp(0, 1).float().cpu().numpy()

        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)
        for k in range(args.draws):
            torch.manual_seed(base_seed + k)
            np.random.seed(args.seed)
            with torch.no_grad():
                noise = sp.SparseTensor(torch.randn_like(pbr_n.feats), coords)
                out = sampler.sample(den, noise, cond=cond, neg_cond=torch.zeros_like(cond), verbose=False,
                                     concat_cond=shp_n, appearance_cond=pbr_n, **sp_params).samples
                pred = tex_decoder(out * ts + tm, guide_subs=subs) * 0.5 + 0.5
            vc = pred.coords[:, 1:].cpu().numpy().astype(np.int16)
            pv = pred.feats[:, :3].float().cpu().numpy()
            save(args.out_dir, k, sid, vc, pv.astype(np.float16))
            if args.ma_out_dir:
                assert np.array_equal(vc, alb_c), f"{sid}: mask and albedo decoded on different voxels"
                save(args.ma_out_dir, k, sid, vc, np.clip(pv.astype(np.float16).astype(np.float32) * alb_f, 0.0, 1.0).astype(np.float16))
        n_ok += 1
        print(f"[predict] {i + 1}/{len(sids)} {sid} n_vox={vc.shape[0]} elapsed={time.time() - t_ready:.0f}s", flush=True)

    print(f"PREDICT_DONE shard={args.shard} n_ok={n_ok} n_skip={n_skip} load_s={t_ready - t0:.1f} "
          f"infer_s={time.time() - t_ready:.1f}", flush=True)
    if n_skip:
        raise SystemExit(f"[predict] {n_skip} shape(s) SKIPPED for missing inputs; their outputs are absent or stale")


if __name__ == "__main__":
    main()
