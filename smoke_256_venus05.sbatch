#!/bin/bash
#SBATCH -J smoke_256_venus05
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_256_venus05_%j.log
# Smoke for the CORRECTED 256-native lane (SegviGenEmissive256 +
# segvigen_emissive_256.yaml): proves the new reader assembles real batches
# from the three-root (shape/pbr latents_v2 + our emission npz) join and
# that a real training step runs -- before any 6-GPU commitment.
set -x
D=/localscratch/xya120
REPO=$D/segvigen-lightning
source $D/miniforge3/etc/profile.d/conda.sh
conda activate segvigen-bw
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
cd "$REPO"
git pull --ff-only origin main
# UNCOMMITTED overlay: SegviGenEmissive256 + segvigen_emissive_256.yaml exist
# only on the falas working tree right now (we never commit; the master
# does). Copy them onto the node-local clone directly for this smoke so it
# is not blocked on a commit+push cycle -- these need a real commit after
# the smoke passes, this copy does not persist across a fresh clone.
FALAS=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
cp "$FALAS/trellis2/datasets/segvigen_256.py" trellis2/datasets/segvigen_256.py
grep -q "SegviGenEmissive256" trellis2/datasets/__init__.py || \
  sed -i "s/    'SegviGenEmissiveDirect': 'segvigen_direct',/    'SegviGenEmissiveDirect': 'segvigen_direct',\n    'SegviGenEmissive256': 'segvigen_256',/" trellis2/datasets/__init__.py
cp "$FALAS/configs/gen/segvigen_emissive_256.yaml" configs/gen/segvigen_emissive_256.yaml
mkdir -p log outputs/smoke_256
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py \
  --config configs/gen/segvigen_emissive_256.yaml \
  --output_dir outputs/smoke_256 \
  --ckpt none \
  --num_gpus 1 \
  --override dataset.args.max_samples=64 \
  --override trainer.args.batch_size_per_gpu=2 \
  --override trainer.args.max_steps=30 \
  --override trainer.args.i_save=10 \
  --override trainer.args.i_log=1 \
  --override trainer.args.batch_log_every=1 \
  --override trainer.args.val_every_epochs=999 \
  --override trainer.args.ckpt_last_every_n_steps=10 \
  --override trainer.args.disable_final_snapshot=true \
  2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-smoke_256_venus05.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=${TRAIN_EXIT}"
ls -la outputs/smoke_256/ckpts/ 2>&1
exit ${TRAIN_EXIT}
