#!/bin/bash
# Item 2(c): 30-step training smoke on venus-05 (sm_120), single GPU, data
# read straight over NFS (no node-local staging -- that's item 4/light
# staging, still pending on the /localscratch space question). Mirrors the
# solar A40/L40S smoke (smoke_segvigen.sbatch) exactly, plus the sm_120 env
# facts harvested from Dongchen's venus-19 launcher.
#SBATCH -J smoke_train_venus05
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_train_venus05_%j.log
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
# sm_120 facts (job 10 harvest, venus-19-proven):
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
cd "$REPO"
git pull --ff-only origin main
mkdir -p log outputs/smoke_venus05
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py \
  --config configs/gen/segvigen_emissive_direct.yaml \
  --output_dir outputs/smoke_venus05 \
  --ckpt none \
  --num_gpus 1 \
  --override dataset.args.max_samples=48 \
  --override trainer.args.batch_size_per_gpu=2 \
  --override trainer.args.max_steps=30 \
  --override trainer.args.i_save=10 \
  --override trainer.args.i_log=1 \
  --override trainer.args.batch_log_every=1 \
  --override trainer.args.val_every_epochs=999 \
  --override trainer.args.ckpt_last_every_n_steps=10 \
  --override trainer.args.disable_final_snapshot=true \
  2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-smoke_train_venus05.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=${TRAIN_EXIT}"
ls -la outputs/smoke_venus05/ckpts/ 2>&1
exit ${TRAIN_EXIT}
