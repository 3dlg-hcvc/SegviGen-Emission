#!/bin/bash
# dual_grid_256 for the released dataset on a Solar CPU allocation (no GPU: mesh_to_flexible_dual_grid is CPU).
# Starts <workers> rank processes, ranks <rank_offset> .. <rank_offset>+<workers>-1 of <world_size>, so several
# nodes share one rank space. Resumable. Job values are POSITIONAL (never --export on star2):
#
#   sbatch -w <node> --cpus-per-task=<workers> --mem=<8G per worker or more> --time=04:00:00 \
#     --job-name=sv_dualgrid --output=<latents root>/_logs/%x-%j.log \
#     script/internal/star2/newdata_dual_grid.sbatch <code root> <latents root> <workers> <rank_offset> <world_size>
#
# A few TexVerse meshes need hundreds of GB in this routine (TRELLIS.2 74k prep, 2026-05); a rank the kernel
# kills loses the rest of its slice, which a second pass with few workers picks up.
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
CODE="${1:?code root}"; ROOT="${2:?latents root}"; W="${3:?workers}"; OFF="${4:?rank offset}"; WORLD="${5:?world size}"
PY="$HOME/envs/trellis2/bin/python"
LG="$HOME/code/lightgen"
cd "$LG"
GLB="$($PY -c 'from data_processing.data_preparation.config import SOLAR_GLB_ROOT; print(SOLAR_GLB_ROOT)')"
echo "[dual_grid] $(date) node=$(hostname) workers=$W ranks=$OFF..$((OFF+W-1))/$WORLD lightgen=$(git rev-parse --short HEAD) code=$(git -C "$CODE" rev-parse --short HEAD)"
cd "$CODE"
mkdir -p "$ROOT/_logs"
pids=()
for i in $(seq 0 $((W-1))); do
  r=$((OFF+i))
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$PY" -u data_toolkit/dual_grid_from_glb.py --parquet "$ROOT/meta/release_index.parquet" --glb_root "$GLB" \
    --out_root "$ROOT" --loader lightgenbench --lightgen_repo "$LG" --check_cells \
    --rank "$r" --world_size "$WORLD" > "$ROOT/_logs/dual_grid.rank$r.$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
grep -h " done: " "$ROOT"/_logs/dual_grid.rank*."$SLURM_JOB_ID".log | tail -n "$W"
echo "[dual_grid] $(date) rank processes that exited non-zero: $fail of $W (failed shapes: $ROOT/dual_grid_256/failed_rank*.txt)"
exit $(( fail > 0 ))
