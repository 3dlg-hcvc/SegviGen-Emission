#!/bin/bash
#SBATCH -J lightning_segvigen_smoke
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-15,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_segvigen_%j.log
# First smoke of the SegviGen port: 30 real steps at batch_size_per_gpu 4 on
# our own dataset_direct data, init from the rekeyed full_seg. Pass gates:
# [finetune] reports 0 missing / 0 unexpected keys for denoiser (the loader
# warns instead of failing -- grep the log, do not trust exit code alone),
# loss is finite and comparable to the standalone lane's early values (~0.3),
# a checkpoint is written, and a resume run picks it back up.
set -x
PROBE=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export PYTHONPATH="$PROBE/pysite:$PYTHONPATH"
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROBE"
mkdir -p log outputs/smoke_segvigen
python train.py \
  --config configs/gen/segvigen_emissive_direct.yaml \
  --output_dir outputs/smoke_segvigen \
  --ckpt none \
  --num_gpus 1 \
  --override dataset.args.max_samples=48 \
  --override trainer.args.batch_size_per_gpu=2 \
  --override trainer.args.max_steps=30 \
  --override trainer.args.i_save=10 \
  --override trainer.args.i_log=1 \
  --override trainer.args.val_every_epochs=999 \
  --override trainer.args.ckpt_last_every_n_steps=10 \
  --override trainer.args.disable_final_snapshot=true \
  2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-smoke_segvigen.log
echo "TRAIN_EXIT=${PIPESTATUS[0]}"
ls -la outputs/smoke_segvigen/ckpts/ || true
