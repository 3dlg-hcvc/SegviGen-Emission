#!/bin/bash
# Latents of the released dataset on one Solar GPU: the paper's frozen encoders over one rank slice.
# <stages> is a comma list of: shape (needs dual_grid_256 complete), pbr, emit (the emission-mask target).
#
#   sbatch -w <node> --gres=gpu:<type>:1 --cpus-per-task=6 --mem=48G --time=06:00:00 \
#     --job-name=sv_encode --output=<latents root>/_logs/%x-%j.log \
#     script/internal/star2/newdata_encode.sbatch <code root> <latents root> <stages> <rank> <world_size> [emit sid list]
#
# The shared trellis2 env (torch 2.6 + cu124) does not run on the Blackwell nodes (cs-venus-05/19/21).
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
CODE="${1:?code root}"; ROOT="${2:?latents root}"; STAGES="${3:?stages}"; RANK="${4:?rank}"; WORLD="${5:?world size}"
EMIT_LIST="${6:-$ROOT/meta/release_all.txt}"   # emit slices by index modulo, so a subset (val + test first) needs its own list
PY="$HOME/envs/trellis2/bin/python"
REL=/3dlg-jupiter-project/lightgen/datasets/dataset_release
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Autotune results must be kept: with USE=0 every new tensor size is re-tuned (measured 2026-09-19: 173 s for 6
# shapes against 6 s). AUTOSAVE=0 because several slices share ~/.flex_gemm and would race writing it.
export FLEX_GEMM_USE_AUTOTUNE_CACHE=1 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
cd "$CODE"
echo "[encode] $(date) node=$(hostname) stages=$STAGES rank=$RANK/$WORLD code=$(git rev-parse --short HEAD)"
"$PY" -c "import torch, o_voxel; assert torch.cuda.is_available(); torch.zeros(1).cuda(); print('[encode] gpu ok:', torch.cuda.get_device_name(0))" || exit 4
rc=0
for s in ${STAGES//,/ }; do
  echo "[encode] ---- stage $s $(date)"
  case "$s" in
    shape) "$PY" -u data_toolkit/encode_shape_from_dual_grid.py --data_root "$ROOT" --parquet "$ROOT/meta/release_index.parquet" \
             --resolution 256 --rank "$RANK" --world_size "$WORLD" || rc=1 ;;
    pbr)   "$PY" -u data_toolkit/encode_pbr_from_voxels.py --data_root "$ROOT" --parquet "$ROOT/meta/release_index.parquet" \
             --resolution 256 --rank "$RANK" --world_size "$WORLD" || rc=1 ;;
    emit)  "$PY" -u scripts_segvigen/build_output_tex_latents_256.py --sid_list "$EMIT_LIST" \
             --release_root "$REL" --out_root "$ROOT/segvigen_emission_mask_npz" --rank "$RANK" --world_size "$WORLD" || rc=1 ;;
    *) echo "unknown stage $s"; rc=2 ;;
  esac
done
echo "[encode] $(date) done rc=$rc"
exit $rc
