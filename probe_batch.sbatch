#!/bin/bash
#SBATCH -J segvigen_batch_probe
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-17,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/batch_probe_%j.log
# Job 3 (batch probe): 100 real steps at a given batch_size_per_gpu on a
# given GPU class, single GPU, to measure shapes/s and peak VRAM. GRES and
# BATCH_SIZE come from the submission command (sbatch --gres=... --export=BATCH_SIZE=...).
# batch_log_every=1 + i_log=1 gives a BATCH_TIME line every step; the probe
# reads steps 11..100 (skip 10 warmup steps: dataloader worker spin-up,
# CUDA kernel autotune, elastic controller ramp) to compute shapes/s, and
# the PEAK_VRAM line CheckpointTimingCallback prints at on_train_end.
set -x
: "${BATCH_SIZE:?set via sbatch --export=BATCH_SIZE=N}"
PROBE=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export PYTHONPATH="$PROBE/pysite:$PYTHONPATH"
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROBE"
nvidia-smi --query-gpu=name,memory.total --format=csv
OUTDIR="outputs/batch_probe_bs${BATCH_SIZE}_${SLURM_JOB_ID}"
mkdir -p log "$OUTDIR"
python train.py \
  --config configs/gen/segvigen_emissive_direct.yaml \
  --output_dir "$OUTDIR" \
  --ckpt none \
  --num_gpus 1 \
  --override dataset.args.max_samples=256 \
  --override trainer.args.batch_size_per_gpu=${BATCH_SIZE} \
  --override trainer.args.max_steps=100 \
  --override trainer.args.i_log=1 \
  --override trainer.args.batch_log_every=1 \
  --override trainer.args.val_every_epochs=999 \
  --override trainer.args.ckpt_last_every_n_steps=100000 \
  --override trainer.args.disable_final_snapshot=true \
  --override trainer.args.wandb_enable=false \
  2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-batch_probe_bs${BATCH_SIZE}_${SLURM_JOB_ID}.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=${TRAIN_EXIT}"
grep -q "PEAK_VRAM" log/*batch_probe_bs${BATCH_SIZE}_${SLURM_JOB_ID}.log || echo "WARN: no PEAK_VRAM line -- run did not reach on_train_end cleanly (check for OOM/traceback above)"
exit ${TRAIN_EXIT}
