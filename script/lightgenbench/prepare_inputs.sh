#!/bin/bash
# SegviGen-Emission's training inputs from the released LightgenBench dataset (Hugging Face 3dlg-hcvc/LightgenBench)
# and TexVerse:
#
#   bash script/lightgenbench/prepare_inputs.sh <release_dir> <texverse_metadata> <texverse_glb_root> <lightgen_root> <out_dir> [workers]
#
#   release_dir        the release unpacked: splits.json and <uuid>/{pbr_voxels,emission_voxels}.vxz, thumbnail.png
#   texverse_metadata  TexVerse's df_metadata.parquet (index = uuid, column glb_1k_path)
#   texverse_glb_root  the TexVerse-1K GLB root the metadata's paths are relative to
#   lightgen_root      a checkout of the LightGen repo: the dual grid reads each GLB with the loader the release was
#                      baked with (data_processing/data_preparation/)
#   out_dir            gets meta/, dual_grid_256/, shape_latents/, pbr_latents/, segvigen_emission_mask_npz/,
#                      segvigen_cond_fp16/ (what train.sh reads)
#
# Needs one GPU and an environment per CLAUDE.md § Environments (the conditioning step needs a transformers that loads
# BiRefNet); PYTHON selects the interpreter (default python). Every stage skips what already exists, so a stopped run
# resumes. For the whole release, run the per-shape stages on several GPUs: each takes --rank/--world_size.
set -euo pipefail
REL=$(readlink -f "${1:?release_dir}"); META=${2:?texverse_metadata}; GLB=${3:?texverse_glb_root}
LG=$(readlink -f "${4:?lightgen_root}"); OUT=${5:?out_dir}; W=${6:-8}
PY=${PYTHON:-python}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
cd "$REPO"
say() { echo "[prepare] $(date +%T) $*"; }
say "metadata";          $PY scripts_segvigen/newdata/build_metadata.py --release "$REL" --out "$OUT/meta" --link_voxels "$OUT"
say "dual grid";         $PY scripts_segvigen/newdata/build_dual_grid.py --lightgen_root "$LG" --sid_list "$OUT/meta/release_all.txt" \
                           --release "$REL" --metadata "$META" --glb_root "$GLB" --out_root "$OUT"
say "shape latents";     $PY data_toolkit/encode_shape_from_dual_grid.py --data_root "$OUT" --parquet "$OUT/meta/release_index.parquet" --resolution 256
say "PBR latents";       $PY data_toolkit/encode_pbr_from_voxels.py --data_root "$OUT" --parquet "$OUT/meta/release_index.parquet" --resolution 256
say "emission masks";    $PY scripts_segvigen/build_output_tex_latents_256.py --sid_list "$OUT/meta/release_all.txt" --release_root "$REL" \
                           --out_root "$OUT/segvigen_emission_mask_npz" --rank 0 --world_size 1
say "conditioning";      $PY scripts_segvigen/newdata/build_cond_tree.py --sid_list "$OUT/meta/release_all.txt" --release "$REL" \
                           --out "$OUT/segvigen_cond_fp16" --routes build
say "verify";            $PY scripts_segvigen/newdata/verify_and_pack.py verify --root "$OUT" --workers "$W"
say "done: $OUT"
