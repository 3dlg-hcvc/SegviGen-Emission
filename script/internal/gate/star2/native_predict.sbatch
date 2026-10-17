#!/bin/bash
# The native predictor (data_toolkit/predict_emission.py) of a tree under test on one Solar GPU, on
# the published-chain gate's shapes and inputs: 5 paper test shapes (the paper's bridged checkpoint and its
# published inputs) and 20 release test shapes (the release snapshot, staged release inputs), seed 0, 12 steps,
# mask and mask x albedo. env "kevin" = Kevin's jupiter miniforge env (that of every published dump), "ours" =
# ~/envs/trellis2. Outputs go to $HOME/segvigen_refactor/out/<name>/{paper_smoke5,release_test20}{,_ma}/draw0/.
#
#   sbatch --gres=gpu:a40:1 -c 6 --mem=64G --time=01:00:00 -J sv_native \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log native_predict.sbatch <tree tgz> <out name> kevin|ours
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"; ENV="${3:?kevin|ours}"
H=$HOME/segvigen_refactor; G=$H/gate
J=/3dlg-jupiter-project/lightgen; F=/3dlg-falas/project/omages/lightgen; L=$J/datasets/dataset_release_latents
LOC=/localscratch/dya78/sv_native_$SLURM_JOB_ID; mkdir -p $LOC/in/shape $LOC/in/pbr $LOC/in/cond $LOC/out $LOC/tree
trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
if [ "$ENV" = kevin ]; then source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2; PY=python3
else PY=$HOME/envs/trellis2/bin/python; fi
export HF_HOME=$J/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/flex_gemm.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH 2>/dev/null || true
echo "[native] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) env=$ENV tree=$(cat $TREE/TREE_REV 2>/dev/null)"
cd $TREE
fail=0
LV2=$J/trellis2_bw/latents_v2
$PY -u data_toolkit/predict_emission.py --run_config $J/outputs/segvigen_256_bw_6gpu/config.yaml \
  --ckpt $J/outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt --sids_file $G/paper_smoke5.txt \
  --out_dir $LOC/out/paper_smoke5 --ma_out_dir $LOC/out/paper_smoke5_ma \
  --shape_latent_dir $LV2/shape_latents/shape_enc_next_dc_f16c32_fp16_256 --pbr_latent_dir $LV2/pbr_latents/tex_enc_next_dc_f16c32_fp16_256 \
  --cond_dirs $F/segvigen_emissive/outputs/cond_fp16,$J/segvigen_emissive/dataset_direct/train_72k_agentic,$J/segvigen_emissive/dataset_direct/val_72k,$J/segvigen_emissive/dataset_direct/test_72k \
  > $LOC/out/paper_smoke5.log 2>&1 || fail=$((fail+1))
tail -1 $LOC/out/paper_smoke5.log
while read -r s; do
  [ -n "$s" ] || continue
  cp $L/shape_latents/shape_enc_next_dc_f16c32_fp16_256/$s.npz $LOC/in/shape/ && cp $L/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/$s.npz $LOC/in/pbr/ \
    && mkdir -p $LOC/in/cond/$s && cp $L/segvigen_cond_fp16/$s/cond_pos.pth $LOC/in/cond/$s/ || { echo "FATAL: staging $s"; exit 3; }
done < $G/release_test20.txt
$PY -u data_toolkit/predict_emission.py --run_config $J/outputs/segvigen_256_release_4h100/config.yaml \
  --ckpt $J/outputs/segvigen_256_release_4h100/snapshots/step0044722.pt --sids_file $G/release_test20.txt \
  --out_dir $LOC/out/release_test20 --ma_out_dir $LOC/out/release_test20_ma \
  --shape_latent_dir $LOC/in/shape --pbr_latent_dir $LOC/in/pbr --cond_dirs $LOC/in/cond \
  > $LOC/out/release_test20.log 2>&1 || fail=$((fail+1))
tail -1 $LOC/out/release_test20.log
mkdir -p "$H/out/$NAME" && cp -r $LOC/out/. "$H/out/$NAME/" || { echo "FATAL: cannot write $H/out/$NAME"; fail=$((fail+1)); }
echo "[native] $(date) done, $fail failed"
exit $(( fail > 0 ))
