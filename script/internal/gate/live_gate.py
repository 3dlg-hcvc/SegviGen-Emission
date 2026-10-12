"""Live-code gate: run this tree's own SegviGen denoiser on fixed shapes and save what it computes.

The published dumps never ran this repo's `trellis2/` (they went through the original SegviGen
repo's wrapper), so this gate is what checks a refactor step: it builds the denoiser from a run's
`config.yaml` through the tree's registries, loads a checkpoint into it strictly, reads each shape
through the tree's `SegviGenEmissive256` and its collate, and saves per shape

    fwd_t25, fwd_t50, fwd_t75   denoiser output at fixed noise and t in {0.25, 0.5, 0.75}
    sample                      the latent after `--steps` Euler steps with the shipped sampler
                                values (guidance 1.0, interval [0.6, 0.9], rescale_t 3.0), before decoding
    coords                      the shape's latent coordinates

to `<out>/<sid>.npz`, plus `<out>/meta.json`. Compare two runs with `compare_live.py`.

Run from anywhere; `--tree` is the root of the tree under test (put first on sys.path).
"""
import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

SIGMA_MIN = 1e-5
T_FWD = (0.25, 0.5, 0.75)


def load_denoiser_state(path):
    """Bare denoiser state dict from a Lightning checkpoint or a bridged file."""
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = payload["state_dict"] if "state_dict" in payload else payload
    for prefix in ("models.denoiser.", "gen3dseg.flow_model."):
        bare = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if bare:
            return bare, prefix
    if "denoiser" in payload:
        return payload["denoiser"], "denoiser"
    raise SystemExit(f"[live_gate] REFUSING: no denoiser keys in {path}")


def git_rev(tree):
    rev_file = os.path.join(tree, "TREE_REV")
    if os.path.exists(rev_file):
        return open(rev_file).read().strip()
    try:
        return subprocess.check_output(["git", "-C", tree, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "not a git tree"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--run_config", required=True, help="a run's config.yaml (models + dataset blocks)")
    ap.add_argument("--ckpt", required=True, help="Lightning checkpoint or bridged snapshot")
    ap.add_argument("--shapes", required=True, help="file with one sid per line")
    ap.add_argument("--split", required=True, help="split-JSON key the shapes belong to")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--parquet_file", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--shape_latent_root", required=True)
    ap.add_argument("--pbr_latent_root", required=True)
    ap.add_argument("--emission_latent_dir", required=True)
    ap.add_argument("--cond_dirs", required=True, help="comma-separated, searched in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    sys.path.insert(0, tree)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    from trellis2.trainers.builders import build_models, build_dataset
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    import trellis2
    assert os.path.dirname(os.path.abspath(trellis2.__file__)) == os.path.join(tree, "trellis2"), \
        f"trellis2 imported from {trellis2.__file__}, not from the tree under test"

    sids = [s.strip() for s in open(args.shapes) if s.strip()]
    if not sids:
        print("VERDICT EMPTY: no shapes"); sys.exit(2)
    if len(set(sids)) != len(sids):
        print("VERDICT REFUSED: repeated shapes"); sys.exit(2)
    sids = sids[args.shard::args.n_shards]
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(args.run_config))
    t0 = time.time()
    den = build_models({"denoiser": cfg["models"]["denoiser"]})["denoiser"]
    state, prefix = load_denoiser_state(args.ckpt)
    res = den.load_state_dict(state, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit(f"[live_gate] REFUSING: {len(res.missing_keys)} missing, "
                         f"{len(res.unexpected_keys)} unexpected keys (prefix {prefix})")
    den = den.cuda().eval()
    del state

    ds_cfg = copy.deepcopy(cfg["dataset"])
    ds_cfg["args"].update(dict(
        data_root=args.data_root, parquet_file=args.parquet_file, split_json=args.split_json,
        split=args.split, shape_latent_root=args.shape_latent_root, pbr_latent_root=args.pbr_latent_root,
        emission_latent_dir=args.emission_latent_dir, cond_search_dirs=args.cond_dirs.split(",")))
    ds = build_dataset(ds_cfg)
    absent = [s for s in sids if s not in set(ds.sids)]
    if absent:
        raise SystemExit(f"[live_gate] REFUSING: {len(absent)} shapes not admitted by the dataset, "
                         f"first {absent[:3]}")
    t_ready = time.time()
    print(f"[live_gate] READY {t_ready - t0:.1f}s, {len(sids)} shapes, ckpt prefix {prefix}", flush=True)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=SIGMA_MIN)
    for i, sid in enumerate(sids):
        batch = ds.collate_fn([ds.get_instance(sid)])
        x0 = batch["x_0"].cuda()
        cc = batch["concat_cond"].cuda()
        ac = batch["appearance_cond"].cuda()
        cond = batch["cond"].cuda()
        neg = torch.zeros_like(cond)
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)
        out = {"coords": x0.coords.cpu().numpy()}
        with torch.no_grad():
            for t in T_FWD:
                torch.manual_seed(base_seed + int(round(t * 100)))
                noise = torch.randn_like(x0.feats)
                tt = torch.full((1,), t, device="cuda")
                x_t = x0.replace((1 - t) * x0.feats + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise)
                pred = den(x_t, tt * 1000, cond, concat_cond=cc, appearance_cond=ac)
                out[f"fwd_t{int(round(t * 100))}"] = pred.feats.float().cpu().numpy()
            torch.manual_seed(base_seed)
            noise = x0.replace(torch.randn_like(x0.feats))
            res = sampler.sample(den, noise, cond=cond, neg_cond=neg, steps=args.steps, rescale_t=3.0,
                                 guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=(0.6, 0.9),
                                 verbose=False, concat_cond=cc, appearance_cond=ac)
            out["sample"] = res.samples.feats.float().cpu().numpy()
        dst = os.path.join(args.out, f"{sid}.npz")
        np.savez(dst + ".tmp.npz", **out)
        os.replace(dst + ".tmp.npz", dst)
        print(f"[live_gate] {i + 1}/{len(sids)} {sid} n={out['coords'].shape[0]}", flush=True)

    meta = dict(tree=tree, tree_rev=git_rev(tree), run_config=os.path.abspath(args.run_config),
                ckpt=os.path.abspath(args.ckpt), split=args.split, seed=args.seed, steps=args.steps,
                shard=args.shard, n_shards=args.n_shards, n_shapes=len(sids), host=platform.node(),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                attn=os.environ.get("ATTN_BACKEND", "default"), seconds=round(time.time() - t0, 1))
    with open(os.path.join(args.out, f"meta.shard{args.shard}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"LIVE_GATE_DONE n={len(sids)} infer_s={time.time() - t_ready:.1f}", flush=True)


if __name__ == "__main__":
    main()
