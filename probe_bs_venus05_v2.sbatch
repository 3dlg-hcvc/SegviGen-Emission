#!/bin/bash
#SBATCH -J probe_bs_venus05_v2
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/probe_bs_venus05_v2_%j.log
# Same methodology as the repo's probe_batch.sbatch (commit 1ad4a74: 100
# real steps, i_log=1/batch_log_every=1, skip 10 warmup steps, read
# PEAK_VRAM at on_train_end) but ported to the segvigen-bw sm_120 env
# since probe_batch.sbatch itself is pinned to the trellis2 env and
# explicitly excludes cs-venus-05.
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
mkdir -p log outputs/probe_bs_v2
nvidia-smi --query-gpu=name,memory.total --format=csv
for BS in 4 8; do
  OUTDIR="outputs/probe_bs_v2/bs${BS}"
  mkdir -p "$OUTDIR"
  echo "=== BATCH_SIZE_PER_GPU=$BS (100 steps) ==="
  python train.py \
    --config configs/gen/segvigen_emissive_direct.yaml \
    --output_dir "$OUTDIR" \
    --ckpt none \
    --num_gpus 1 \
    --override dataset.args.max_samples=256 \
    --override trainer.args.batch_size_per_gpu=${BS} \
    --override trainer.args.max_steps=100 \
    --override trainer.args.i_log=1 \
    --override trainer.args.batch_log_every=1 \
    --override trainer.args.val_every_epochs=999 \
    --override trainer.args.ckpt_last_every_n_steps=100000 \
    --override trainer.args.disable_final_snapshot=true \
    --override trainer.args.wandb_enable=false \
    2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-probe_bs${BS}_venus05_${SLURM_JOB_ID}.log
  EXIT=${PIPESTATUS[0]}
  echo "BS=$BS EXIT=$EXIT"
done
