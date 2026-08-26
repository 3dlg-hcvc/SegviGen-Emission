"""256-resolution target-latent build for the SegviGen 256 pilot (job 8/GO
2026-08-26). Mirrors the 512 lane's build_dataset_direct.py channel-hijack
convention EXACTLY (segvigen_emissive/emissive/data_prep/build_dataset_direct.py):
white=emissive / black=non in base_color, metallic=0, roughness=255, alpha=255,
emission binarized at >1/255 (any nonzero channel) -- but skips the 512 lane's
dual-grid geometry re-extraction and 256->512 upsample entirely, because at
256 we use Dongchen's native pbr_voxels_256/emission_voxels_256 coords
directly (no upsampling needed) and only need the TEX encoder (shape/input-tex
latents at 256 are shared, already built by Dongchen -- see latents_v2, job 8).

Per shape:
  1. read <sid>/pbr_voxels_256/{sid}.vxz and <sid>/emission_voxels_256/{sid}.vxz
     from the team's read-only voxel tree (see the OVOX_ROOT note below for
     why this is the real path and not the two symlink chains that dangle
     on compute nodes) -- asserts identical coords between the two, same as
     build_dataset_direct.py.
  2. is_emis = (emissive > 0).any(dim=-1)  (binarize >1/255, any channel)
  3. out_base_color = white where is_emis else black; out_metallic=0,
     out_roughness=255, out_alpha=255 (OUT_*_U8 constants, byte-identical to
     the 512 lane's).
  4. attr = cat([out_base_color, out_metallic, out_roughness, out_alpha]) / 255
     * 2 - 1  (same normalization vxz_to_latent_slat.py uses).
  5. encode with the stock tex_enc_next_dc_f16c32_fp16 encoder (the SAME
     pretrained encoder the 512 lane and Dongchen's shared 256 latents use)
     -> sparse latent (feats, coords).
  6. save {feats, coords} as .npz (NOT .pth) -- matches the latents_v2 schema
     Dongchen's shape/tex latents use, so a 256 config can point
     shape_latent_root/pbr_latent_root/emission_latent_root at a mix of
     latents_v2 (shared) and this output (ours) with one reader.

NO normalization-stats computation: the 256 lane reuses the shipped
TRELLIS.2-4B pipeline.json tex stats verbatim, same as 512 (owner-set
2026-08-26).

Per-item result cache: skips any sid whose output npz already exists, so a
killed/resubmitted array resumes instead of redoing finished work.

Usage (one array task):
  python scripts_segvigen/build_output_tex_latents_256.py \\
      --split train_72k_agentic --rank 0 --world_size 10
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

OVOX_ROOT = "/3dlg-jupiter-project/lightgen/datasets/uv_voxel_74k"
# NOTE (2026-08-26), two dead ends before landing here, both worth recording:
#   1. build_dataset_direct.py's own OVOX_ROOT
#      (uv_voxel_pipeline/out_uv_voxel_74k/<sid>/pbr_voxels_256/) resolves
#      on the WORKSTATION but is a symlink whose absolute target is baked in
#      WITH the /cs/ prefix (uv_voxel_pipeline/out_uv_voxel_74k ->
#      /cs/3dlg-jupiter-project/lightgen/datasets/uv_voxel_74k) -- that
#      prefix exists only on the workstation, so the symlink dangles on
#      every compute node (confirmed via readlink -f + ls on
#      cs-venus-07/13: ENOENT, not a mount gap -- /3dlg-jupiter-project
#      itself mounts fine).
#   2. trellis2_bw/lightgen_74k_newbake/pbr_voxels_256/<sid>.vxz is ALSO
#      just a symlink back into the same out_uv_voxel_74k tree, so it
#      inherits the same dangling-on-compute-nodes problem.
# The real, compute-node-safe path is the symlink's actual target, one
# level shallower than build_dataset_direct.py's per-sid layout implies:
#   <OVOX_ROOT>/<sid>/pbr_voxels_256/<sid>.vxz
#   <OVOX_ROOT>/<sid>/emission_voxels_256/<sid>.vxz
# (verified present for the val_72k sids that looked "missing" through
# both symlink chains above).
DATASET_DIRECT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
TRELLIS = "microsoft/TRELLIS.2-4B/ckpts"
OUT_ROOT = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/"
            "latents_256/output_tex_latents/segvigen_emissive_mask_256")

# channel-hijack constants -- byte-identical to build_dataset_direct.py's
# OUT_METALLIC_U8 / OUT_ROUGHNESS_U8 / OUT_ALPHA_U8
OUT_METALLIC_U8 = 0
OUT_ROUGHNESS_U8 = 255
OUT_ALPHA_U8 = 255


def build_one(sid: str, tex_encoder) -> str:
    """Returns a short status string; raises on unexpected failure."""
    out_path = os.path.join(OUT_ROOT, f"{sid}.npz")
    if os.path.exists(out_path):
        return "cached"

    pbr_vxz = os.path.join(OVOX_ROOT, sid, "pbr_voxels_256", f"{sid}.vxz")
    emis_vxz = os.path.join(OVOX_ROOT, sid, "emission_voxels_256", f"{sid}.vxz")
    if not os.path.exists(pbr_vxz):
        return f"SKIP missing pbr_voxels_256: {pbr_vxz}"
    if not os.path.exists(emis_vxz):
        return f"SKIP missing emission_voxels_256: {emis_vxz}"

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

    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp_path = out_path + f".tmp{os.getpid()}.npz"  # must end in .npz or np.savez appends it, breaking os.replace
    feats_np = tex_slat.feats.detach().cpu().numpy()
    coords_np = tex_slat.coords[:, 1:].detach().cpu().numpy()  # drop batch column, match latents_v2 (N,3)
    import numpy as np
    np.savez(tmp_path, feats=feats_np.astype("float32"), coords=coords_np.astype("uint8"))
    os.replace(tmp_path, out_path)
    return "built"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train_72k_agentic", "val_72k"])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world_size", type=int, required=True)
    args = ap.parse_args()

    split_dir = os.path.join(DATASET_DIRECT, args.split)
    all_ids = sorted(os.listdir(split_dir))
    my_ids = [sid for i, sid in enumerate(all_ids) if i % args.world_size == args.rank]
    print(f"[build256] rank {args.rank}/{args.world_size}: {len(my_ids)} of {len(all_ids)} sids in {args.split}",
          flush=True)

    # .float() forces fp32 weights to match our fp32 input feats -- the
    # pretrained ckpt is fp16 per its name, and Triton's tl.dot in flex_gemm
    # asserts matching dtypes (same fix as encode_pbr_latent.py and
    # encode_shape_from_dual_grid.py, commit d842eb6).
    tex_encoder = models.from_pretrained(f"{TRELLIS}/tex_enc_next_dc_f16c32_fp16").eval().float().cuda()

    n_built = n_cached = n_skipped = n_error = 0
    for i, sid in enumerate(my_ids):
        try:
            status = build_one(sid, tex_encoder)
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
