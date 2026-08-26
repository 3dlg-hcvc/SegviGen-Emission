#!/bin/bash
# ============================================================================
# venus-05 is CLEARED and in production since 2026-08-26 (the Mercury-drain
# banner that used to sit here is obsolete; the node runs our training).
# This is the live 256-lane launcher: submit-replacement-then-cancel is the
# restart pattern (submit this while the old job runs, then scancel the old).
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
#   2. All four 256-lane data roots staged locally by
#      script/bw/stage_data_256_local.sh (emission_npz, shape_latents,
#      pbr_latents, cond_fp16 under /localscratch/xya120/segvigen_data_256)
#      -- this launcher's own sync block below self-heals this if the
#      MANIFEST_OK it writes is missing, so a bare submission on a fresh
#      node still works, just spends its first minutes staging instead of
#      training.
#
# BATCH_SIZE_PER_GPU=8 was measured on this node (single-GPU 100-step probe,
# board lightning_port 2026-08-26: BS 4/8/12 all fit inside the 96GB-class
# RTX Pro 6000, BS=8 chosen for headroom under multi-GPU/NCCL overhead on a
# first 24h run). Effective batch for this launch = BATCH_SIZE_PER_GPU * 6
# (num_gpus) = 48 -- state the real product in any report that uses this
# launch if BATCH_SIZE_PER_GPU is overridden.
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

# Self-healing local-stage gate (owner decision 2026-08-26: all four data
# roots now live on this node's local disk, not jupiter/falas directly --
# see segvigen_emissive_256.yaml's header). A missing or short
# MANIFEST_OK means the stage never ran, was interrupted, or landed on a
# fresh node: (re)run it inline rather than let train.py fail cold on a
# missing sid partway through the first epoch.
STAGE_DIR=/localscratch/xya120/segvigen_data_256
STAGE_SCRIPT="$FALAS/script/bw/stage_data_256_local.sh"
if [ ! -f "$STAGE_DIR/MANIFEST_OK" ]; then
  echo "MANIFEST_OK missing at $STAGE_DIR -- staging now (self-heal)"
  bash "$STAGE_SCRIPT"
fi
if [ ! -f "$STAGE_DIR/MANIFEST_OK" ]; then
  echo "FATAL: staging did not produce MANIFEST_OK, refusing to start training"
  exit 1
fi
cat "$STAGE_DIR/MANIFEST_OK"

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
