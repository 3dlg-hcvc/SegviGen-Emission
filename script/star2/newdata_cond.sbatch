#!/bin/bash
# Thumbnail conditioning of the released dataset, the two I/O routes (reuse = byte copy of the paper run's fp16
# file, convert = fp32 -> fp16) on a Solar CPU allocation, <workers> rank processes. The build route needs a GPU
# and SegviGen's own env; see scripts_segvigen/newdata/build_cond_tree.py.
#
#   sbatch -w <node> --cpus-per-task=<workers> --mem=32G --time=04:00:00 --job-name=sv_cond \
#     --output=<latents root>/_logs/%x-%j.log script/star2/newdata_cond.sbatch <code root> <latents root> <workers>
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
CODE="${1:?code root}"; ROOT="${2:?latents root}"; W="${3:?workers}"
PY="$HOME/envs/trellis2/bin/python"
cd "$CODE"
echo "[cond] $(date) node=$(hostname) workers=$W code=$(git rev-parse --short HEAD)"
pids=()
for r in $(seq 0 $((W-1))); do
  "$PY" -u scripts_segvigen/newdata/build_cond_tree.py --routes reuse,convert \
    --sid_list "$ROOT/meta/release_all.txt" --release /3dlg-jupiter-project/lightgen/datasets/dataset_release \
    --out "$ROOT/segvigen_cond_fp16" \
    --paper_fp16_root /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_fp16 \
    --dataset_direct /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
    --rank "$r" --world_size "$W" > "$ROOT/_logs/cond_io.rank$r.$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
grep -h "DONE" "$ROOT"/_logs/cond_io.rank*."$SLURM_JOB_ID".log
echo "[cond] $(date) rank processes that exited non-zero: $fail of $W"
exit $(( fail > 0 ))
