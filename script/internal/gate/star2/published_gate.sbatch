#!/bin/bash
# Published-chain gate on one Solar GPU, in Kevin's jupiter miniforge env (the env of every published dump):
#   1. the published dump script with PROBE = the tree under test: 5 paper test shapes (the published inputs)
#      and 20 release test shapes (release inputs, staged), seed 0, 12 steps;
#   2. the tree's unrekey_for_eval.py on the paper snapshot, compared tensor by tensor with the published
#      bridged file;
#   3. the tree's ckpt_step.py on the paper snapshot and on the release last.ckpt.
# Positional values only (never --export on star2). Outputs go to $HOME/segvigen_refactor/out/<name>/.
#
#   sbatch --gres=gpu:a40:1 -c 6 --mem=64G --time=01:30:00 -J sv_pubgate \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log published_gate.sbatch <tree tgz> <out name>
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"
H=$HOME/segvigen_refactor; G=$H/gate
J=/3dlg-jupiter-project/lightgen; F=/3dlg-falas/project/omages/lightgen; L=$J/datasets/dataset_release_latents
PAPER_SNAP=$F/segvigen_emissive/outputs/segvigen_256_bw_6gpu_ckpts_mirror/step0023405_snapshot.ckpt
PAPER_BRIDGED=$J/outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt
REL=$J/outputs/segvigen_256_release_4h100
LOC=/localscratch/dya78/sv_pubgate_$SLURM_JOB_ID; mkdir -p $LOC/in/shape $LOC/in/pbr $LOC/in/cond $LOC/out $LOC/tree
trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
export HF_HOME=$J/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/flex_gemm.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH 2>/dev/null || true
export SEGVIGEN_PROBE=$TREE
echo "[pubgate] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) tree=$(cat $TREE/TREE_REV 2>/dev/null)"
cd $TREE
fail=0
python3 $G/dump_published.py --bridged_ckpt $PAPER_BRIDGED --sids_file $G/paper_smoke5.txt \
  --out_dir $LOC/out/paper_smoke5 --draws 1 --seed 0 --steps 12 > $LOC/out/paper_smoke5.log 2>&1 || fail=$((fail+1))
tail -1 $LOC/out/paper_smoke5.log
while read -r s; do
  [ -n "$s" ] || continue
  cp $L/shape_latents/shape_enc_next_dc_f16c32_fp16_256/$s.npz $LOC/in/shape/ && cp $L/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/$s.npz $LOC/in/pbr/ \
    && mkdir -p $LOC/in/cond/$s && cp $L/segvigen_cond_fp16/$s/cond_pos.pth $LOC/in/cond/$s/ || { echo "FATAL: staging $s"; exit 3; }
done < $G/release_test20.txt
python3 $G/dump_published.py --bridged_ckpt $REL/snapshots/step0044722.pt --sids_file $G/release_test20.txt \
  --out_dir $LOC/out/release_test20 --draws 1 --seed 0 --steps 12 \
  --shape_latent_dir $LOC/in/shape --pbr_latent_dir $LOC/in/pbr --cond_dir $LOC/in/cond > $LOC/out/release_test20.log 2>&1 || fail=$((fail+1))
tail -1 $LOC/out/release_test20.log
python3 $TREE/scripts_segvigen/unrekey_for_eval.py $PAPER_SNAP $LOC/bridged.pt > $LOC/out/unrekey.log 2>&1 || fail=$((fail+1))
python3 $G/compare_bridged.py $LOC/bridged.pt $PAPER_BRIDGED > $LOC/out/bridge_check.txt 2>&1 || fail=$((fail+1))
cat $LOC/out/bridge_check.txt
{ python3 $TREE/scripts_segvigen/ckpt_step.py $PAPER_SNAP; python3 $TREE/scripts_segvigen/ckpt_step.py $REL/ckpts/last.ckpt; } > $LOC/out/ckpt_step.txt 2>&1 || fail=$((fail+1))
cat $LOC/out/ckpt_step.txt
mkdir -p $H/out/$NAME; cp -r $LOC/out/. $H/out/$NAME/
echo "[pubgate] $(date) done, $fail failed"
exit $(( fail > 0 ))
