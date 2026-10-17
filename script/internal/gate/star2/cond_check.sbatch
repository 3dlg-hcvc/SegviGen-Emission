#!/bin/bash
# Conditioning check (step 11c): build_cond_tree.py of a tree under test on one Solar
# GPU in Kevin's jupiter miniforge env (the env of the release's conditioning): routes 1-2 (reuse, convert) on
# cond_io6.txt and route 3 (build) on cond_build5.txt, into $HOME/segvigen_refactor/out/<name>/cond/.
# "old" = a tree whose route 3 imports the original SegviGen tree (the falas deploy), "new" = the port.
#
#   sbatch --gres=gpu:a40:1 -c 4 --mem=32G --time=00:40:00 -J sv_cond \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log cond_check.sbatch <tree tgz> <out name> old|new
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"; WHICH="${3:?old|new}"
H=$HOME/segvigen_refactor; G=$H/gate
J=/3dlg-jupiter-project/lightgen; F=/3dlg-falas/project/omages/lightgen
LOC=/localscratch/dya78/sv_cond_$SLURM_JOB_ID; mkdir -p $LOC/out $LOC/tree
trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
export HF_HOME=$J/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOTS=(--paper_fp16_root $F/segvigen_emissive/outputs/cond_fp16 --dataset_direct $J/segvigen_emissive/dataset_direct)
OLD=(); [ "$WHICH" = old ] && OLD=(--segvigen_repo $F/segvigen_emissive)
echo "[cond_check] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) which=$WHICH tree=$(cat $TREE/TREE_REV 2>/dev/null)"
cd $TREE
CT=data_toolkit/build_cond_tree.py; [ -f $CT ] || CT=scripts_segvigen/newdata/build_cond_tree.py   # before step 15d
fail=0
python3 -u $CT --sid_list $G/cond_io6.txt --release $J/datasets/dataset_release \
  --out $LOC/out/cond --routes reuse,convert "${ROOTS[@]}" > $LOC/out/io.log 2>&1 || fail=$((fail+1))
python3 -u $CT --sid_list $G/cond_build5.txt --release $J/datasets/dataset_release \
  --out $LOC/out/cond --routes build "${ROOTS[@]}" "${OLD[@]}" > $LOC/out/build.log 2>&1 || fail=$((fail+1))
grep -h DONE $LOC/out/io.log $LOC/out/build.log
mkdir -p $H/out/$NAME; cp -r $LOC/out/. $H/out/$NAME/
echo "[cond_check] $(date) done, $fail failed"
exit $(( fail > 0 ))
