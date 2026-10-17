#!/bin/bash
# Release-layer check (step 14): script/lightgenbench/{prepare_inputs,get_init,train}.sh of a tree under test on one
# Solar GPU, on a 4-shape mini release (2 train, 1 val, 1 test: symlinks into jupiter's dataset_release plus a
# splits.json). prepare_inputs and get_init run in Kevin's jupiter miniforge env (its transformers loads BiRefNet),
# train.sh's dry run (--tryrun, one step) in ~/envs/trellis2 (lightgen 2.6). The inputs and the init go to
# $HOME/segvigen_refactor/out/<name>/ for comparison with jupiter's datasets/dataset_release_latents/.
#
#   sbatch --gres=gpu:l40s:1 -c 8 --mem=96G --time=02:00:00 -J sv_lgb \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log lightgenbench_check.sbatch <tree tgz> <out name> <uuid list>
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"; LIST="${3:?uuid list: 2 train, 1 val, 1 test}"
H=$HOME/segvigen_refactor
J=/3dlg-jupiter-project/lightgen; REL=$J/datasets/dataset_release
LOC=/localscratch/dya78/sv_lgb_$SLURM_JOB_ID; mkdir -p $LOC/tree $LOC/mini
trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
mapfile -t U < "$LIST"
for u in "${U[@]}"; do ln -s $REL/$u $LOC/mini/$u; done
printf '{"train": ["%s", "%s"], "val": ["%s"], "test": ["%s"]}\n' "${U[0]}" "${U[1]}" "${U[2]}" "${U[3]}" > $LOC/mini/splits.json
LG=$HOME/code/lightgen
GLB=$(cd $LG && $HOME/envs/trellis2/bin/python -c 'from data_processing.data_preparation.config import SOLAR_GLB_ROOT; print(SOLAR_GLB_ROOT)')
export HF_HOME=$J/hf_cache TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/flex_gemm.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH 2>/dev/null || true
echo "[lgb] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) tree=$(cat $TREE/TREE_REV 2>/dev/null) glb=$GLB"
fail=0
( source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
  PYTHON=python3 bash $TREE/script/lightgenbench/prepare_inputs.sh $LOC/mini "$GLB" $LG $LOC/inputs 4
) > $LOC/prepare.log 2>&1 || fail=$((fail+1))
tail -n 3 $LOC/prepare.log
( source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
  PYTHON=python3 bash $TREE/script/lightgenbench/get_init.sh $LOC/init.pt
) > $LOC/get_init.log 2>&1 || fail=$((fail+1))
tail -n 2 $LOC/get_init.log
PYTHON=$HOME/envs/trellis2/bin/python SLURM_JOB_NAME=bash TMPDIR=$LOC bash $TREE/script/lightgenbench/train.sh \
  $LOC/inputs $LOC/init.pt $LOC/train_out 1 1 -- --tryrun > $LOC/train.log 2>&1 || fail=$((fail+1))
tail -n 3 $LOC/train.log
mkdir -p $H/out/$NAME
cp $LOC/prepare.log $LOC/get_init.log $LOC/train.log $H/out/$NAME/
( cd $LOC/inputs && tar -cf $H/out/$NAME/inputs.tar --exclude='pbr_voxels_256' --exclude='emission_voxels_256' . ) 2>/dev/null
$HOME/envs/trellis2/bin/python - $LOC/init.pt $J/datasets/dataset_release_latents/init/full_seg_rekeyed.pt > $H/out/$NAME/init_check.txt 2>&1 <<'PYEOF'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["denoiser"]
b = torch.load(sys.argv[2], map_location="cpu", mmap=True, weights_only=False)["denoiser"]
bad = [k for k in b if k not in a or a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]
print(f"INIT {'IDENTICAL' if set(a) == set(b) and not bad else 'DIFFERENT'} keys={len(a)}/{len(b)} differing={len(bad)}")
PYEOF
cat $H/out/$NAME/init_check.txt
echo "[lgb] $(date) done, $fail failed"
exit $(( fail > 0 ))
