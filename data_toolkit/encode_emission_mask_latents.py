"""SegviGen-Emission's target latents: the emission mask of each shape through the stock texture encoder.

The mask is written as a PBR texture: white = emissive / black = not in base_color, metallic 0,
roughness 255, alpha 255, emission binarized at > 1/255 (any nonzero channel).

Per shape, from the released dataset's layout (3dlg-hcvc/LightgenBench):
  1. read <release_root>/<sid>/pbr_voxels.vxz and emission_voxels.vxz (256 grid); refuse a shape whose
     two coordinate sets differ.
  2. is_emis = (emissive > 0).any(dim=-1)
  3. out_base_color = white where is_emis else black; out_metallic 0, out_roughness 255, out_alpha 255
     (the OUT_*_U8 constants).
  4. attr = cat([out_base_color, out_metallic, out_roughness, out_alpha]) / 255 * 2 - 1.
  5. encode with the stock tex_enc_next_dc_f16c32_fp16 encoder (the encoder of the shape and PBR latents)
     -> sparse latent (feats, coords).
  6. save {feats, coords} as <out_root>/<sid>.npz, the schema of the shape and PBR latents.

No normalization statistics are computed: training reuses the shipped TRELLIS.2-4B pipeline.json texture
statistics. A shape whose output already exists is skipped, so a killed array task resumes.

Usage (one array task):
  python data_toolkit/encode_emission_mask_latents.py --sid_list <file> --release_root <dataset_release> \
      --out_root <latents root>/segvigen_emission_mask_npz --rank 0 --world_size 10
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "train.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate the lightning repo root above {__file__}")
    ROOT = parent
sys.path.insert(0, ROOT)

import o_voxel  # noqa: E402
import trellis2.modules.sparse as sp  # noqa: E402
from trellis2 import models  # noqa: E402

TRELLIS = "microsoft/TRELLIS.2-4B/ckpts"

# the constant channels of the convention above
OUT_METALLIC_U8 = 0
OUT_ROUGHNESS_U8 = 255
OUT_ALPHA_U8 = 255


def build_one(sid: str, tex_encoder, release_root: str, out_root: str) -> str:
    """Returns a short status string; raises on unexpected failure."""
    out_path = os.path.join(out_root, f"{sid}.npz")
    if os.path.exists(out_path):
        return "cached"

    pbr_vxz = os.path.join(release_root, sid, "pbr_voxels.vxz")
    emis_vxz = os.path.join(release_root, sid, "emission_voxels.vxz")
    if not os.path.exists(pbr_vxz):
        return f"SKIP missing pbr_voxels.vxz: {pbr_vxz}"
    if not os.path.exists(emis_vxz):
        return f"SKIP missing emission_voxels.vxz: {emis_vxz}"

    coords_pbr, pbr = o_voxel.io.read(pbr_vxz)
    coords_emis, emis = o_voxel.io.read(emis_vxz)
    if not torch.equal(coords_pbr, coords_emis):
        return f"SKIP coords mismatch between pbr and emission voxels for {sid}"

    metallic = pbr["metallic"]
    roughness = pbr["roughness"]
    alpha = pbr["alpha"]
    emissive = emis["emissive"]

    is_emis = (emissive > 0).any(dim=1)
    out_base_color = (is_emis.unsqueeze(1).expand(-1, 3).to(torch.uint8)) * 255
    out_metallic = torch.full_like(metallic, OUT_METALLIC_U8)
    out_roughness = torch.full_like(roughness, OUT_ROUGHNESS_U8)
    out_alpha = torch.full_like(alpha, OUT_ALPHA_U8)

    attr = torch.cat(
        [out_base_color.float(), out_metallic.float(), out_roughness.float(), out_alpha.float()],
        dim=-1,
    ) / 255.0 * 2 - 1

    coords_batched = torch.cat(
        [torch.zeros(coords_pbr.shape[0], 1, dtype=torch.int32), coords_pbr], dim=1
    ).cuda()
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr.cuda(), coords_batched))

    os.makedirs(out_root, exist_ok=True)
    tmp_path = out_path + f".tmp{os.getpid()}.npz"  # must end in .npz or np.savez appends it, breaking os.replace
    feats_np = tex_slat.feats.detach().cpu().numpy()
    coords_np = tex_slat.coords[:, 1:].detach().cpu().numpy()  # drop batch column: (N, 3)
    import numpy as np
    np.savez(tmp_path, feats=feats_np.astype("float32"), coords=coords_np.astype("uint8"))
    os.replace(tmp_path, out_path)
    return "built"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sid_list", required=True, help="one shape id per line")
    ap.add_argument("--release_root", required=True, help="the released dataset's root (<root>/<sid>/*.vxz)")
    ap.add_argument("--out_root", required=True, help="where <sid>.npz goes")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world_size", type=int, required=True)
    args = ap.parse_args()

    all_ids = [l.strip() for l in open(args.sid_list) if l.strip()]
    src = args.sid_list
    my_ids = [sid for i, sid in enumerate(all_ids) if i % args.world_size == args.rank]
    print(f"[build256] rank {args.rank}/{args.world_size}: {len(my_ids)} of {len(all_ids)} sids in {src}",
          flush=True)

    # .float() forces fp32 weights to match our fp32 input feats -- the
    # pretrained ckpt is fp16 per its name, and Triton's tl.dot in flex_gemm
    # asserts matching dtypes (same fix as encode_pbr_latent.py and
    # encode_shape_from_dual_grid.py).
    tex_encoder = models.from_pretrained(f"{TRELLIS}/tex_enc_next_dc_f16c32_fp16").eval().float().cuda()

    n_built = n_cached = n_skipped = n_error = 0
    for i, sid in enumerate(my_ids):
        try:
            status = build_one(sid, tex_encoder, args.release_root, args.out_root)
        except Exception as e:
            n_error += 1
            print(f"[build256] ERROR {sid}: {e!r}", flush=True)
            continue
        if status == "built":
            n_built += 1
        elif status == "cached":
            n_cached += 1
        else:
            n_skipped += 1
            print(f"[build256] {status}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"[build256] rank {args.rank}: {i + 1}/{len(my_ids)} "
                  f"(built={n_built} cached={n_cached} skipped={n_skipped} error={n_error})", flush=True)

    print(f"[build256] rank {args.rank} DONE: built={n_built} cached={n_cached} "
          f"skipped={n_skipped} error={n_error} of {len(my_ids)}", flush=True)


if __name__ == "__main__":
    main()
