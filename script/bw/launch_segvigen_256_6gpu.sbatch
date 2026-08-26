#!/bin/bash
# ============================================================================
# DO NOT SUBMIT WHILE cs-venus-05 / cs-venus-19 ARE DRAINING (Mercury
# deployment, as of 2026-08-26). This script is prep only: ready to run once
# the owner confirms a node is back and clears it. Submitting to either node
# now is out of scope for job 5 -- see the standing exclusions.
# ============================================================================
#SBATCH -J segvigen_256_bw_6gpu
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --nodelist=cs-venus-05
##SBATCH --nodelist=cs-venus-19   # swap to this line instead of venus-05 if that's the one cleared first
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=490G
#SBATCH --time=24:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/segvigen_256_bw_6gpu_%j.log
#
# Prereqs before this can run for real:
#   1. script/bw/bootstrap_segvigen_bw.sh has completed successfully on the
#      target node (env, extensions, repo clone -- see its UNVERIFIED note
#      about private-repo GitHub access on this node).
#   2. script/bw/extract_staged_shards.sh has extracted our slim tar shards
#      to /localscratch/xya120/segvigen_data (built by job 5's amendment;
#      those shards themselves depend on job 7's cond_pos.pth sidecars
#      being complete for every sample they include).
#
# BATCH_SIZE_PER_GPU is TBD by the owner (per the ratified Option-A
# conservative recipe: lr fixed, batch set by measured per-GPU-class
# memory). Defaulted here to 2, the same safe fallback measured on a 44GB
# A40 -- the RTX Pro 6000 Blackwell cards have far more VRAM (96GB class),
# so this is likely overly conservative, but no batch probe has run on
# sm_120 yet (no smoke has run on Blackwell at all) to justify raising it.
# Effective batch for this launch = BATCH_SIZE_PER_GPU * 6 (num_gpus) --
# state the real product in any report that uses this launch.
set -x
: "${BATCH_SIZE_PER_GPU:=8}"
D=/localscratch/xya120
REPO=$D/segvigen-lightning
source $D/miniforge3/etc/profile.d/conda.sh
conda activate segvigen-bw
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --- sm_120 facts, all confirmed venus-19-proven in Dongchen's launcher
# (script/star2/emission_dit_pbr2emission_74k_v3_nonzero_nocopy_venus19.sh,
# commit 1dc7824) -- job 10 documentation harvest, read before this launch
# was written:
# flash-attn has no sm_120 support -> fall back to sdpa/xformers.
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
# flex_gemm's autotune file-cache races across ranks on a shared filesystem;
# in-memory-only autotune avoids it.
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
# Lightning's SLURM-coordinated launcher errors (ntasks=1 vs devices=6) when
# it detects a SLURM job name it recognizes; this is the one-task-native-spawn
# trick, NOT --ntasks-per-node=<gpus> (star2 convention).
export SLURM_JOB_NAME=bash
# --- end sm_120 facts ---
cd "$REPO"
# sync the 256 lane files from falas (idempotent; smoke already placed them, this guards a fresh node-local clone)
FALAS=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
cp $FALAS/trellis2/datasets/segvigen_256.py trellis2/datasets/segvigen_256.py
cp $FALAS/configs/gen/segvigen_emissive_256.yaml configs/gen/segvigen_emissive_256.yaml
grep -q SegviGenEmissive256 trellis2/datasets/__init__.py || sed -i "s/    'SegviGenEmissiveDirect': 'segvigen_direct',/    'SegviGenEmissiveDirect': 'segvigen_direct',\n    'SegviGenEmissive256': 'segvigen_256',/" trellis2/datasets/__init__.py
mkdir -p log outputs
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py \
  --config configs/gen/segvigen_emissive_256.yaml \
  --output_dir outputs/segvigen_256_bw_6gpu \
  --ckpt last \
  --num_gpus 6 \
  --override trainer.args.batch_size_per_gpu=${BATCH_SIZE_PER_GPU} \
  --override trainer.args.disable_final_snapshot=true \
  2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-segvigen_bw_6gpu_${SLURM_JOB_ID}.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=${TRAIN_EXIT}"
exit ${TRAIN_EXIT}
