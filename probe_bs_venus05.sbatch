#!/bin/bash
#SBATCH -J probe_bs_venus05
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/probe_bs_venus05_%j.log
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
mkdir -p log outputs/probe_bs
nvidia-smi --query-gpu=name,memory.total --format=csv
for BS in 4 8 12; do
  echo "=== BATCH_SIZE_PER_GPU=$BS ==="
  python train.py \
    --config configs/gen/segvigen_emissive_direct.yaml \
    --output_dir outputs/probe_bs/bs${BS} \
    --ckpt none \
    --num_gpus 1 \
    --override dataset.args.max_samples=64 \
    --override trainer.args.batch_size_per_gpu=${BS} \
    --override trainer.args.max_steps=8 \
    --override trainer.args.i_save=999 \
    --override trainer.args.i_log=1 \
    --override trainer.args.batch_log_every=1 \
    --override trainer.args.val_every_epochs=999 \
    --override trainer.args.disable_final_snapshot=true \
    2>&1 | tail -n 40
  EXIT=${PIPESTATUS[0]}
  echo "BS=$BS EXIT=$EXIT"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
done
