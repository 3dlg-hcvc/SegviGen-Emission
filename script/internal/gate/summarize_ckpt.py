"""Summarize a Lightning checkpoint written by train.py as one JSON line: what a resume reads back.

    python summarize_ckpt.py <ckpt> [<ckpt> ...] > summary.jsonl
"""
import json
import os
import sys

import torch


def _tsum(obj):
    if torch.is_tensor(obj):
        return float(obj.detach().double().sum()) if obj.is_floating_point() else float(obj.sum())
    if isinstance(obj, dict):
        return sum(_tsum(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tsum(v) for v in obj)
    return 0.0


def _plain(obj):
    if torch.is_tensor(obj):
        return obj.tolist() if obj.numel() <= 16 else f"tensor{tuple(obj.shape)} sum={_tsum(obj):.6e}"
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj] if len(obj) <= 16 else f"list[{len(obj)}] sum={_tsum(obj):.6e}"
    return obj if isinstance(obj, (int, float, str, bool, type(None))) else repr(obj)


for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("state_dict", {})
    ema = (ck.get("ema") or {}).get("state", {})
    opt = ck.get("optimizer_states") or []
    out = dict(
        file=os.path.basename(path), keys=sorted(ck.keys()),
        global_step=ck.get("global_step"), epoch=ck.get("epoch"),
        state_dict_keys=len(sd), state_dict_sum=_tsum(sd),
        state_dict_prefixes=sorted({k.split(".")[0] + "." + k.split(".")[1] for k in sd if k.count(".") >= 1}),
        ema={k: _tsum(v) for k, v in ema.items()},
        optimizer_sum=_tsum([o.get("state", {}) for o in opt]),
        lr_schedulers=_plain(ck.get("lr_schedulers")),
        sampler_state=_plain(ck.get("sampler_state")), elastic_controller=_plain(ck.get("elastic_controller")),
        callbacks=sorted(str(k) for k in (ck.get("callbacks") or {})),
        other={k: _plain(ck[k]) for k in ck if k not in (
            "state_dict", "ema", "optimizer_states", "lr_schedulers", "sampler_state", "elastic_controller",
            "callbacks", "loops", "global_step", "epoch")},
    )
    print(json.dumps(out), flush=True)
