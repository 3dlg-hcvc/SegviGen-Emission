#!/bin/bash
# Repeatability on the smallest test shapes (step 10): on one A40 in Kevin's env, the published chain (gate's
# dump_published.py with PROBE = the tree) twice and the predictor twice, seed 0, on the three smallest test shapes of
# each run plus a control, paper and release; live autotuning logged. Outputs: ~/segvigen_refactor/out/<name>/.
#   sbatch --gres=gpu:a40:1 -c 6 --mem=64G --time=00:40:00 -J sv_diag --output=$HOME/segvigen_refactor/logs/%x-%j.log small_shapes.sbatch <tree tgz> <name>
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ=$1; NAME=$2
H=$HOME/segvigen_refactor; G=$H/gate
J=/3dlg-jupiter-project/lightgen; F=/3dlg-falas/project/omages/lightgen; L=$J/datasets/dataset_release_latents; LV2=$J/trellis2_bw/latents_v2
LOC=/localscratch/dya78/sv_diag_$SLURM_JOB_ID; mkdir -p $LOC/tree $LOC/out; trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
export HF_HOME=$J/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor
export SEGVIGEN_PROBE=$TREE FLEX_GEMM_AUTOTUNER_VERBOSE=1
printf '%s\n' 3c2165d6139d4b6ca44622ec16599730 93e02c9032084a24af3a4daf6fcea7e8 67055b9df3a84965a60f64e241c72cdd 889cdabc54ee40edb0a1ce1fc993d927 > $LOC/paper.txt
printf '%s\n' 3c2165d6139d4b6ca44622ec16599730 93e02c9032084a24af3a4daf6fcea7e8 1f2044e52c804d1aac116f0757f2d6c6 889cdabc54ee40edb0a1ce1fc993d927 > $LOC/release.txt
echo "[diag] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) tree=$(cat $TREE/TREE_REV)"
cd $TREE
fresh_cache() { export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/fg_$1.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH; }
for r in a b; do
  fresh_cache pub_paper_$r; python3 $G/dump_published.py --bridged_ckpt $J/outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt --sids_file $LOC/paper.txt \
    --out_dir $LOC/out/pub_paper_$r --draws 1 --seed 0 --steps 12 > $LOC/out/pub_paper_$r.log 2>&1
  fresh_cache pub_release_$r; python3 $G/dump_published.py --bridged_ckpt $J/outputs/segvigen_256_release_4h100/snapshots/step0044722.pt --sids_file $LOC/release.txt \
    --out_dir $LOC/out/pub_release_$r --draws 1 --seed 0 --steps 12 --shape_latent_dir $L/shape_latents/shape_enc_next_dc_f16c32_fp16_256 \
    --pbr_latent_dir $L/pbr_latents/tex_enc_next_dc_f16c32_fp16_256 --cond_dir $L/segvigen_cond_fp16 > $LOC/out/pub_release_$r.log 2>&1
  fresh_cache nat_paper_$r; python3 data_toolkit/predict_emission_mask_voxels.py --run_config $J/outputs/segvigen_256_bw_6gpu/config.yaml \
    --ckpt $J/outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt --sids_file $LOC/paper.txt --out_dir $LOC/out/nat_paper_$r \
    --shape_latent_dir $LV2/shape_latents/shape_enc_next_dc_f16c32_fp16_256 --pbr_latent_dir $LV2/pbr_latents/tex_enc_next_dc_f16c32_fp16_256 \
    --cond_dirs $F/segvigen_emissive/outputs/cond_fp16,$J/segvigen_emissive/dataset_direct/train_72k_agentic,$J/segvigen_emissive/dataset_direct/val_72k,$J/segvigen_emissive/dataset_direct/test_72k \
    > $LOC/out/nat_paper_$r.log 2>&1
  fresh_cache nat_release_$r; python3 data_toolkit/predict_emission_mask_voxels.py --run_config $J/outputs/segvigen_256_release_4h100/config.yaml \
    --ckpt $J/outputs/segvigen_256_release_4h100/snapshots/step0044722.pt --sids_file $LOC/release.txt --out_dir $LOC/out/nat_release_$r \
    --shape_latent_dir $L/shape_latents/shape_enc_next_dc_f16c32_fp16_256 --pbr_latent_dir $L/pbr_latents/tex_enc_next_dc_f16c32_fp16_256 \
    --cond_dirs $L/segvigen_cond_fp16 > $LOC/out/nat_release_$r.log 2>&1
done
for f in $LOC/out/*.log; do echo "$(basename $f): $(grep -c 'Running autotuning' $f) live autotunes; $(tail -n 1 $f)"; done
mkdir -p $H/out/$NAME && cp -r $LOC/out/. $H/out/$NAME/ && cp $LOC/paper.txt $LOC/release.txt $H/out/$NAME/
echo "[diag] $(date) done"
