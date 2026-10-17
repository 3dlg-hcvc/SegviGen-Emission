#!/bin/bash
# SegviGen-Emission's training inputs from the released LightgenBench dataset (Hugging Face 3dlg-hcvc/LightgenBench)
# and TexVerse:
#
#   bash script/lightgenbench/prepare_inputs.sh <release_dir> <texverse_1k> <lightgen_root> <out_dir> [workers]
#
#   release_dir        the release unpacked: splits.json and <uuid>/{pbr_voxels,emission_voxels}.vxz, thumbnail.png
#   texverse_1k        a TexVerse-1K download (Hugging Face YiboZhang2001/TexVerse-1K: glbs/glbs_1k/<group>/<uuid>_1024.glb)
#   lightgen_root      a checkout of the LightGen repo: the dual grid reads each GLB with the loader the release was
#                      baked with (data_processing/data_preparation/)
#   out_dir            gets meta/, dual_grid_256/, shape_latents/, pbr_latents/, segvigen_emission_mask_npz/,
#                      segvigen_cond_fp16/ (what train.sh reads)
#
# Needs one GPU and an environment per CLAUDE.md § Environments (the conditioning step needs a transformers that loads
# BiRefNet); PYTHON selects the interpreter (default python). Every stage skips what already exists, so a stopped run
# resumes. For the whole release, run the per-shape stages on several GPUs: each takes --rank/--world_size.
set -euo pipefail
REL=$(readlink -f "${1:?release_dir}"); GLB=$(readlink -f "${2:?texverse_1k}")
LG=$(readlink -f "${3:?lightgen_root}"); OUT=${4:?out_dir}; W=${5:-8}
PY=${PYTHON:-python}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
cd "$REPO"
say() { echo "[prepare] $(date +%T) $*"; }
say "metadata";          $PY data_toolkit/build_lightgenbench_index.py --release "$REL" --out "$OUT/meta" --link_voxels "$OUT"
say "dual grid";         $PY data_toolkit/dual_grid_from_glb.py --parquet "$OUT/meta/release_index.parquet" --glb_root "$GLB" \
                           --out_root "$OUT" --loader lightgenbench --lightgen_repo "$LG" --check_cells \
                           || say "some shapes have no grid: $OUT/dual_grid_256/failed_rank0.txt (verify leaves them out)"
say "shape latents";     $PY data_toolkit/encode_shape_from_dual_grid.py --data_root "$OUT" --parquet "$OUT/meta/release_index.parquet" --resolution 256
say "PBR latents";       $PY data_toolkit/encode_pbr_from_voxels.py --data_root "$OUT" --parquet "$OUT/meta/release_index.parquet" --resolution 256
say "emission masks";    $PY data_toolkit/encode_emission_mask_latents.py --sid_list "$OUT/meta/release_all.txt" --release_root "$REL" \
                           --out_root "$OUT/segvigen_emission_mask_npz" --rank 0 --world_size 1
say "conditioning";      $PY data_toolkit/build_cond_tree.py --sid_list "$OUT/meta/release_all.txt" --release "$REL" \
                           --out "$OUT/segvigen_cond_fp16" --routes build
say "verify";            $PY data_toolkit/verify_and_pack.py verify --root "$OUT" --workers "$W"
say "done: $OUT"
