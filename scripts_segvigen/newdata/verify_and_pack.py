"""Check the released dataset's SegviGen inputs shape by shape, then pack them for the clusters.

verify   For every shape id: the shape, pbr and emission-mask latents exist, hold finite
         (N, 32) feats on IDENTICAL ordered coords (SegviGenEmissive256.get_instance asserts that
         per sample and a failed assert kills the run), and cond_pos.pth is a finite float16
         (1, 1029, 1024). Writes
             <root>/meta/usable.txt, <root>/meta/dropped.tsv  (shape id, reason)
             <root>/segvigen_emission_mask_npz/token_counts.json   (tokens per usable shape, the
                                                  BalancedResumableSampler's load; dataset reads it)
             <root>/meta/release_splits_usable.json   the split rows, unusable shapes removed
pack     Tars of the usable shapes under <root>/tars/ (few large files: the clusters cap inodes),
         member paths as the training job unpacks them:
             latents.tar            meta/ + shape_latents/ + pbr_latents/ + emission_npz/
             cond_<k>of<n>.tar      cond_fp16/<uuid>/cond_pos.pth
         and SHA256SUMS over them.

Usage:
    python scripts_segvigen/newdata/verify_and_pack.py verify --root <latents root> --workers 16
    python scripts_segvigen/newdata/verify_and_pack.py pack   --root <latents root> --cond_shards 8
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SHAPE_NAME = "shape_enc_next_dc_f16c32_fp16_256"
PBR_NAME = "tex_enc_next_dc_f16c32_fp16_256"
EMIT_DIR = "segvigen_emission_mask_npz"
COND_DIR = "segvigen_cond_fp16"
COND_SHAPE = (1, 1029, 1024)


def _paths(root, sid):
    return {
        "shape": os.path.join(root, "shape_latents", SHAPE_NAME, f"{sid}.npz"),
        "pbr": os.path.join(root, "pbr_latents", PBR_NAME, f"{sid}.npz"),
        "emit": os.path.join(root, EMIT_DIR, f"{sid}.npz"),
        "cond": os.path.join(root, COND_DIR, sid, "cond_pos.pth"),
    }


def _check(job):
    root, sid = job
    import torch
    p = _paths(root, sid)
    for k, q in p.items():
        if not os.path.exists(q):
            return sid, f"missing_{k}", 0
    try:
        ref = None
        for k in ("shape", "pbr", "emit"):
            with np.load(p[k]) as d:
                feats, coords = d["feats"], d["coords"]
            if feats.ndim != 2 or feats.shape[1] != 32 or feats.shape[0] != coords.shape[0] or coords.shape[1] != 3:
                return sid, f"bad_shape_{k}:{feats.shape}/{coords.shape}", 0
            if not np.isfinite(feats).all():
                return sid, f"nonfinite_{k}", 0
            if ref is None:
                ref = coords
            elif coords.shape != ref.shape or not np.array_equal(coords, ref):
                return sid, f"coords_differ_{k}", 0
        c = torch.load(p["cond"], map_location="cpu")
        c = c["cond"] if isinstance(c, dict) else c
        if tuple(c.shape) != COND_SHAPE or c.dtype != torch.float16 or not torch.isfinite(c).all():
            return sid, f"bad_cond:{tuple(c.shape)}/{c.dtype}", 0
        return sid, "ok", int(ref.shape[0])
    except Exception as e:  # noqa: BLE001
        return sid, f"error:{e!r}"[:200], 0


def verify(args):
    meta = os.path.join(args.root, "meta")
    sids = [l.strip() for l in open(os.path.join(meta, "release_all.txt")) if l.strip()]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_check, [(args.root, s) for s in sids], chunksize=64))
    ok = {s: n for s, st, n in res if st == "ok"}
    with open(os.path.join(meta, "usable.txt"), "w") as f:
        f.write("\n".join(s for s in sids if s in ok) + "\n")
    with open(os.path.join(meta, "dropped.tsv"), "w") as f:
        f.writelines(f"{s}\t{st}\n" for s, st, _ in res if st != "ok")
    with open(os.path.join(args.root, EMIT_DIR, "token_counts.json"), "w") as f:
        json.dump(ok, f)
    rows = json.load(open(os.path.join(meta, "release_splits.json")))
    usable_rows = {k: [i for i in v if sids[i] in ok] for k, v in rows.items()}
    with open(os.path.join(meta, "release_splits_usable.json"), "w") as f:
        json.dump(usable_rows, f)
    reasons = {}
    for _, st, _ in res:
        key = st.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"[verify] {len(ok)}/{len(sids)} usable; " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    print("[verify] per split: " + ", ".join(f"{k} {len(usable_rows[k])}/{len(rows[k])}" for k in rows))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _pack_cond(job):
    root, out, sids = job
    tmp = out + f".tmp{os.getpid()}"
    with tarfile.open(tmp, "w") as t:
        for s in sids:
            t.add(_paths(root, s)["cond"], arcname=f"cond_fp16/{s}/cond_pos.pth")
    os.replace(tmp, out)
    return out


def pack(args):
    meta = os.path.join(args.root, "meta")
    sids = [l.strip() for l in open(os.path.join(meta, "usable.txt")) if l.strip()]
    tars = os.path.join(args.root, "tars")
    os.makedirs(tars, exist_ok=True)
    lat = os.path.join(tars, "latents.tar")
    if not os.path.exists(lat):
        tmp = lat + f".tmp{os.getpid()}"
        with tarfile.open(tmp, "w") as t:
            for name in ("release_index.parquet", "release_splits_usable.json", "usable.txt", "dropped.tsv"):
                t.add(os.path.join(meta, name), arcname=f"meta/{name}")
            t.add(os.path.join(args.root, EMIT_DIR, "token_counts.json"), arcname="emission_npz/token_counts.json")
            for s in sids:
                p = _paths(args.root, s)
                t.add(p["shape"], arcname=f"shape_latents/{SHAPE_NAME}/{s}.npz")
                t.add(p["pbr"], arcname=f"pbr_latents/{PBR_NAME}/{s}.npz")
                t.add(p["emit"], arcname=f"emission_npz/{s}.npz")
        os.replace(tmp, lat)
    n = args.cond_shards
    jobs = []
    for k in range(n):
        out = os.path.join(tars, f"cond_{k:02d}of{n:02d}.tar")
        if not os.path.exists(out):
            jobs.append((args.root, out, sids[k::n]))
    with ProcessPoolExecutor(min(args.workers, max(1, len(jobs)))) as ex:
        for out in ex.map(_pack_cond, jobs):
            print("[pack]", out, flush=True)
    names = sorted(f for f in os.listdir(tars) if f.endswith(".tar"))
    with ProcessPoolExecutor(min(args.workers, len(names))) as ex:
        sums = list(ex.map(_sha256, [os.path.join(tars, f) for f in names]))
    with open(os.path.join(tars, "SHA256SUMS"), "w") as f:
        f.writelines(f"{h}  {name}\n" for h, name in zip(sums, names))
    total = sum(os.path.getsize(os.path.join(tars, f)) for f in names)
    print(f"[pack] {len(names)} tars, {total / 1e9:.1f} GB, {len(sids)} shapes -> {tars}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["verify", "pack"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cond_shards", type=int, default=8)
    args = ap.parse_args()
    {"verify": verify, "pack": pack}[args.cmd](args)


if __name__ == "__main__":
    main()
