#!/bin/bash
# Step 10's full check: the native predictor (data_toolkit/predict_emission.py) of a tree under test
# remakes one published test-set dump (one run, one seed; masks and mask x albedo) in Kevin's jupiter miniforge env
# on the card that made it, and compares both with the published dumps (compare_masks.py; the bar is identity).
# paper: 204 test shapes, seeds 0-3 on A40, seed 4 on RTX 6000 Ada; release: 200 test shapes, seeds 0-4 on A40.
# Outputs: $HOME/segvigen_refactor/out/<name>/{mask,ma}.txt and the predictor log (the dumps are not kept).
#
#   sbatch --gres=gpu:a40:1 -c 6 --mem=64G --time=00:45:00 -J sv_full \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log native_full.sbatch <tree tgz> <out name> paper|release <seed>
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"; RUN="${3:?paper|release}"; SEED="${4:?seed}"
H=$HOME/segvigen_refactor
J=/3dlg-jupiter-project/lightgen; F=/3dlg-falas/project/omages/lightgen; L=$J/datasets/dataset_release_latents
LOC=/localscratch/dya78/sv_full_$SLURM_JOB_ID; mkdir -p $LOC/out $LOC/tree
trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
GT=$TREE/script/internal/gate
source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
export HF_HOME=$J/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/flex_gemm.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH 2>/dev/null || true
if [ "$RUN" = paper ]; then
  LV2=$J/trellis2_bw/latents_v2; SH=$GT/paper_test204.txt
  PUB=$J/trellis2_bw/eval_outputs/segvigen256_ourval/dump_step0023405_seed${SEED}_test/draw0
  PUB_MA=$J/trellis2_bw/eval_outputs/segvigen256_ourval/dump_step0023405_seed${SEED}_test_ma/draw0
  ARGS=(--run_config $J/outputs/segvigen_256_bw_6gpu/config.yaml --ckpt $J/outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt
        --shape_latent_dir $LV2/shape_latents/shape_enc_next_dc_f16c32_fp16_256 --pbr_latent_dir $LV2/pbr_latents/tex_enc_next_dc_f16c32_fp16_256
        --cond_dirs $F/segvigen_emissive/outputs/cond_fp16,$J/segvigen_emissive/dataset_direct/train_72k_agentic,$J/segvigen_emissive/dataset_direct/val_72k,$J/segvigen_emissive/dataset_direct/test_72k)
else
  SH=$GT/release_test200.txt
  PUB=$J/trellis2_bw/eval_outputs/release_eval/dumps_test/step0044722/seed0${SEED}/draw0
  PUB_MA=$J/trellis2_bw/eval_outputs/release_eval/dumps_test_ma/step0044722/seed0${SEED}/draw0
  ARGS=(--run_config $J/outputs/segvigen_256_release_4h100/config.yaml --ckpt $J/outputs/segvigen_256_release_4h100/snapshots/step0044722.pt
        --shape_latent_dir $L/shape_latents/shape_enc_next_dc_f16c32_fp16_256 --pbr_latent_dir $L/pbr_latents/tex_enc_next_dc_f16c32_fp16_256
        --cond_dirs $L/segvigen_cond_fp16)
fi
echo "[full] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) run=$RUN seed=$SEED tree=$(cat $TREE/TREE_REV 2>/dev/null)"
cd $TREE
fail=0
python3 -u data_toolkit/predict_emission.py "${ARGS[@]}" --sids_file $SH --seed $SEED \
  --out_dir $LOC/out/mask --ma_out_dir $LOC/out/ma > $LOC/predict.log 2>&1 || fail=$((fail+1))
tail -n 1 $LOC/predict.log
mkdir -p "$H/out/$NAME" && cp $LOC/predict.log "$H/out/$NAME/" || { echo "FATAL: cannot write $H/out/$NAME"; exit 4; }
python3 $GT/compare_masks.py --identical --a $PUB --b $LOC/out/mask/draw0 --shapes $SH > "$H/out/$NAME/mask.txt" 2>&1 || fail=$((fail+1))
python3 $GT/compare_masks.py --identical --a $PUB_MA --b $LOC/out/ma/draw0 --shapes $SH > "$H/out/$NAME/ma.txt" 2>&1 || fail=$((fail+1))
echo "mask: $(tail -n 1 "$H/out/$NAME/mask.txt")"; echo "ma:   $(tail -n 1 "$H/out/$NAME/ma.txt")"
echo "[full] $(date) done, $fail failed"
exit $(( fail > 0 ))
