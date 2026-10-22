"""Check the released dataset's SegviGen inputs shape by shape, then pack them for a training machine.

verify   For every shape id: the shape, pbr and emission-mask latents exist, hold finite
         (N, 32) feats on IDENTICAL ordered coords (SegviGenEmissive256.get_instance asserts that
         per sample and a failed assert kills the run), and cond_pos.pth is a finite float16
         (1, 1029, 1024). Writes
             <root>/meta/usable.txt, <root>/meta/dropped.tsv  (shape id, reason)
             <root>/segvigen_emission_mask_npz/token_counts.json   (tokens per usable shape, the
                                                  BalancedResumableSampler's load; dataset reads it)
             <root>/meta/release_splits_usable.json   the split rows, unusable shapes removed
pack     Tars of the usable shapes under <root>/tars/ (few large files, for file systems that cap inodes),
         member paths as the training job unpacks them:
             latents.tar            meta/ + shape_latents/ + pbr_latents/ + emission_npz/
             cond_<k>of<n>.tar      cond_fp16/<uuid>/cond_pos.pth
         and SHA256SUMS over them.

Usage:
    python data_toolkit/verify_and_pack.py verify --root <latents root> --workers 16
    python data_toolkit/verify_and_pack.py pack   --root <latents root> --cond_shards 8
"""
import argparse
import hashlib
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
            if (feats.ndim != 2 or coords.ndim != 2 or feats.shape[1] != 32 or coords.shape[1] != 3
                    or feats.shape[0] != coords.shape[0] or feats.shape[0] == 0):
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


def _latents_tar(args, meta, sids, lat):
    tmp = lat + f".tmp{os.getpid()}"
    with tarfile.open(tmp, "w") as t:
        for name in ("release_index.parquet", "release_splits_usable.json", "usable.txt", "dropped.tsv"):
            t.add(os.path.join(meta, name), arcname=f"meta/{name}")
        t.add(os.path.join(args.root, EMIT_DIR, "token_counts.json"), arcname="emission_npz/token_counts.json")
        for s_ in sids:
            p_ = _paths(args.root, s_)
            t.add(p_["shape"], arcname=f"shape_latents/{SHAPE_NAME}/{s_}.npz")
            t.add(p_["pbr"], arcname=f"pbr_latents/{PBR_NAME}/{s_}.npz")
            t.add(p_["emit"], arcname=f"emission_npz/{s_}.npz")
    os.replace(tmp, lat)


def pack_incremental(args):
    """After a repair: the cond tars of the previous pack stay (they may already be on the training
    machine), the conditioning of shapes that became usable since goes into ONE extra tar, latents.tar is
    rebuilt whole."""
    meta = os.path.join(args.root, "meta")
    tars = os.path.join(args.root, "tars")
    sids = [l.strip() for l in open(os.path.join(meta, "usable.txt")) if l.strip()]
    prev = {l.strip() for l in open(args.incremental_from) if l.strip()}
    gone = prev - set(sids)
    if gone:
        raise SystemExit(f"[pack] {len(gone)} shapes of the previous pack are no longer usable; do a full pack")
    new = [s for s in sids if s not in prev]
    old_sums = {}
    for line in open(os.path.join(tars, "SHA256SUMS")):
        h, name = line.split()
        old_sums[name] = h
    _latents_tar(args, meta, sids, os.path.join(tars, "latents.tar"))
    extra = os.path.join(tars, f"cond_extra_{len(new):05d}.tar")
    for f in os.listdir(tars):
        if f.startswith("cond_extra_"):
            os.remove(os.path.join(tars, f))
    if new:
        _pack_cond((args.root, extra, new))
    names = sorted(f for f in os.listdir(tars) if f.endswith(".tar"))
    with open(os.path.join(tars, "SHA256SUMS"), "w") as f:
        for name in names:
            keep = name.startswith("cond_") and not name.startswith("cond_extra_") and name in old_sums
            f.write(f"{old_sums[name] if keep else _sha256(os.path.join(tars, name))}  {name}\n")
    print(f"[pack] incremental: {len(new)} newly usable shapes, {len(sids)} in all, {len(names)} tars -> {tars}")


def pack(args):
    if args.incremental_from:
        return pack_incremental(args)
    meta = os.path.join(args.root, "meta")
    sids = [l.strip() for l in open(os.path.join(meta, "usable.txt")) if l.strip()]
    tars = os.path.join(args.root, "tars")
    os.makedirs(tars, exist_ok=True)
    # one generation at a time: tars of an older verify (or another --cond_shards) must not sit beside
    # the new ones, the training job unpacks every tar SHA256SUMS names
    stale = [f for f in os.listdir(tars) if not f.startswith(".")]
    if stale and not args.overwrite:
        raise SystemExit(f"[pack] {tars} holds {len(stale)} files; pass --overwrite to rebuild them all")
    for f in stale:
        os.remove(os.path.join(tars, f))
    _latents_tar(args, meta, sids, os.path.join(tars, "latents.tar"))
    n = args.cond_shards
    jobs = []
    for k in range(n):
        jobs.append((args.root, os.path.join(tars, f"cond_{k:02d}of{n:02d}.tar"), sids[k::n]))
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
    ap.add_argument("--overwrite", action="store_true", help="pack: delete the tars already there first")
    ap.add_argument("--incremental_from", default=None,
                    help="pack: the usable.txt of the pack already in tars/; keeps its cond tars, see pack_incremental")
    args = ap.parse_args()
    {"verify": verify, "pack": pack}[args.cmd](args)


if __name__ == "__main__":
    main()
