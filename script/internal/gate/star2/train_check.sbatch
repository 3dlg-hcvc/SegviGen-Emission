#!/bin/bash
# Training check (train_check.sh) on one Solar GPU in the shared ~/envs/trellis2. Checkpoints stay on the
# node's /localscratch and are deleted; record.jsonl, summary.jsonl and the log go to
# $HOME/segvigen_refactor/out/<name>/. Positional values only (never --export on star2).
#
#   sbatch --gres=gpu:l40s:1 -c 6 --mem=96G --time=01:30:00 -J sv_train \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log train_check.sbatch <tree tgz> <out name> fresh|resume|resume_paper
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"; MODE="${3:?mode}"; COV="${4:-}"   # 4th arg "cov": the coverage run (step 9)
H=$HOME/segvigen_refactor; G=$H/gate
LOC=/localscratch/dya78/sv_train_$SLURM_JOB_ID; mkdir -p $LOC/tree
trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
export GATE_PY=$HOME/envs/trellis2/bin/python TMPDIR=$LOC/tmp; mkdir -p $TMPDIR
export TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor HF_HUB_OFFLINE=1
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/flex_gemm.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH 2>/dev/null || true
export SLURM_JOB_NAME=bash
if [ "$COV" = cov ]; then
  mkdir -p $LOC/cov; printf "[run]\nparallel = True\ndata_file = $LOC/cov/.coverage\nsource = $TREE\nomit = $TREE/o-voxel/*\n" > $LOC/covrc
  export COVERAGE_PROCESS_START=$LOC/covrc PYTHONPATH=$H/pydeps
fi
bash $G/train_check.sh $TREE $LOC/out $MODE > $LOC/train_check.log 2>&1
rc=$?
if [ "$COV" = cov ]; then $GATE_PY -m coverage combine --rcfile=$LOC/covrc -q && $GATE_PY -m coverage json --rcfile=$LOC/covrc -q -o $LOC/out/cov.json; fi
mkdir -p $H/out/$NAME; cp $LOC/out/cov.json $H/out/$NAME/ 2>/dev/null; cp $LOC/out/record.jsonl $LOC/out/summary.jsonl $H/out/$NAME/ 2>/dev/null; cp $LOC/train_check.log $H/out/$NAME/
tail -3 $LOC/train_check.log
echo "[sv_train] $(date) node=$(hostname) rc=$rc"
exit $rc
