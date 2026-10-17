"""DINOv3 thumbnail conditioning for the released dataset: <out>/<uuid>/cond_pos.pth.

Same artifact as the paper run's cond_fp16 tree (its convert_cond_fp16.py, at tag pre-refactor): one bare
float16 tensor (1, 1029, 1024) per shape, the DINOv3 ViT-L/16 tokens of the TexVerse thumbnail after
SegviGen's own preprocessing. The conditioning depends on the thumbnail only, not on the bake, so a
shape the old runs already embedded keeps its bytes. Per shape, first route that applies:

  1. reuse    <paper_fp16_root>/<uuid>/cond_pos.pth exists            -> byte copy
  2. convert  <dataset_direct>/<split>/<uuid>/{cond_pos,cond}.pth      -> the fp32 tensor cast to fp16
  3. build    neither                                                  -> img_to_cond on
              <release>/<uuid>/thumbnail.png (a byte copy of the TexVerse thumbnail); if SegviGen's
              preprocessing raises, its repair variant (img_to_cond_repair)

Routes 1 and 2 need the paper run's trees (--paper_fp16_root, --dataset_direct); without them every shape
takes route 3. Route 3 runs SegviGen's own preprocessing, copied verbatim into thumbnail_cond.py beside
this file, on this repo's BiRefNet and DinoV3FeatureExtractor. Never writes zeros: a shape with no usable
thumbnail is listed and left out.

Resumable (an existing output is skipped), atomic writes, a JSON report per rank.

Usage:
    python data_toolkit/build_cond_tree.py --sid_list <meta>/release_all.txt \
        --release <dataset_release> --out <latents root>/segvigen_cond_fp16 \
        [--paper_fp16_root <dir> --dataset_direct <dir>] [--routes reuse,convert | --routes build]
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

import torch

DIRECT_SPLITS = ("train_72k", "val_72k", "test_72k")
SHAPE = (1, 1029, 1024)


def _as_fp16(obj) -> torch.Tensor:
    t = obj["cond"] if isinstance(obj, dict) else obj
    assert torch.is_tensor(t) and tuple(t.shape) == SHAPE, f"unexpected cond {getattr(t, 'shape', type(t))}"
    assert torch.isfinite(t).all(), "non-finite cond"
    return t.to(torch.float16).contiguous()


def _save(t: torch.Tensor, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + f".tmp{os.getpid()}"
    torch.save(t, tmp)
    os.replace(tmp, dest)


class Builder:
    """SegviGen's own thumbnail -> cond path (thumbnail_cond.py), loaded on first use."""

    def __init__(self):
        self.ready = False

    def _load(self):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))   # this repo's trellis2
        sys.path.insert(0, here)
        from thumbnail_cond import img_to_cond, img_to_cond_repair
        from trellis2.pipelines.rembg import BiRefNet
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        self.img_to_cond, self.img_to_cond_repair = img_to_cond, img_to_cond_repair
        self.rembg = BiRefNet(model_name="briaai/RMBG-2.0")
        self.rembg.cuda()
        self.dino = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
        self.dino.cuda()
        self.ready = True

    def __call__(self, thumb: str, scratch: str):
        if not self.ready:
            self._load()
        used = "img_to_cond"
        try:
            self.img_to_cond(self.rembg, self.dino, thumb, scratch)
        except Exception as e:  # noqa: BLE001
            branch = self.img_to_cond_repair(self.rembg, self.dino, thumb, scratch)
            used = f"repair:{branch} after {e!r}"[:200]
        t = _as_fp16(torch.load(scratch, map_location="cpu"))
        os.remove(scratch)
        return t, used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sid_list", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper_fp16_root", default=None, help="the paper run's fp16 conditioning tree (route 1)")
    ap.add_argument("--dataset_direct", default=None, help="the paper's dataset_direct tree (route 2)")
    ap.add_argument("--routes", default="reuse,convert,build")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    routes = set(args.routes.split(","))
    assert routes <= {"reuse", "convert", "build"}, routes

    sids = [l.strip() for l in open(args.sid_list) if l.strip()]
    sids = [s for i, s in enumerate(sids) if i % args.world_size == args.rank]
    if args.limit:
        sids = sids[:args.limit]
    direct = {}   # every route needs it: build must not redo a shape that convert owns
    for sp in (DIRECT_SPLITS if args.dataset_direct else ()):   # one listing per split, not a stat per shape
        try:
            for name in os.listdir(os.path.join(args.dataset_direct, sp)):
                direct.setdefault(name, sp)
        except FileNotFoundError:
            print(f"[cond] no {sp} under {args.dataset_direct}", flush=True)
    builder = Builder() if "build" in routes else None

    n = {"cached": 0, "reuse": 0, "convert": 0, "build": 0, "left": 0, "failed": 0}
    repaired, failed, no_thumb = {}, [], []
    t0 = time.time()
    for i, sid in enumerate(sids):
        dest = os.path.join(args.out, sid, "cond_pos.pth")
        if os.path.exists(dest):
            n["cached"] += 1
            continue
        try:
            # the route is decided by the files that exist, then run only if this call was asked for it
            paper = os.path.join(args.paper_fp16_root, sid, "cond_pos.pth") if args.paper_fp16_root else None
            src = None
            if sid in direct:
                d = os.path.join(args.dataset_direct, direct[sid], sid)
                src = next((q for q in (os.path.join(d, "cond_pos.pth"), os.path.join(d, "cond.pth"))
                            if os.path.exists(q)), None)
            route = "reuse" if paper and os.path.exists(paper) else "convert" if src else "build"
            if route not in routes:
                n["left"] += 1
            elif route == "reuse":
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + f".tmp{os.getpid()}"
                shutil.copyfile(paper, tmp)
                os.replace(tmp, dest)
                n["reuse"] += 1
            elif route == "convert":
                _save(_as_fp16(torch.load(src, map_location="cpu")), dest)
                n["convert"] += 1
            else:
                thumb = os.path.join(args.release, sid, "thumbnail.png")
                if not os.path.exists(thumb):
                    no_thumb.append(sid)
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                t, used = builder(thumb, dest + f".build{os.getpid()}.pth")
                if used != "img_to_cond":
                    repaired[sid] = used
                _save(t, dest)
                n["build"] += 1
        except Exception as e:  # noqa: BLE001
            n["failed"] += 1
            failed.append({"sid": sid, "error": repr(e)[:300]})
            print(f"[cond] FAIL {sid}: {e!r}\n{traceback.format_exc()[-300:]}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"[cond] rank {args.rank}: {i + 1}/{len(sids)} {n} {time.time() - t0:.0f}s", flush=True)

    print(f"[cond] rank {args.rank} DONE: {n} no_thumb={len(no_thumb)} repaired={len(repaired)} "
          f"of {len(sids)} in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{'-'.join(sorted(routes))}.rank{args.rank}of{args.world_size}"
    with open(os.path.join(args.out, f"_report.{tag}.json"), "w") as f:
        json.dump({"counts": n, "no_thumb": no_thumb, "repaired": repaired, "failed": failed}, f, indent=1)


if __name__ == "__main__":
    main()
