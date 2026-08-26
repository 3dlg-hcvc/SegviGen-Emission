#!/bin/bash
#SBATCH -J lightning_segvigen_smoke_resume
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-15,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_segvigen_resume_%j.log
# Second half of the SegviGen smoke: is the checkpoint the first run wrote
# (outputs/smoke_segvigen/ckpts/last.ckpt, step 30) re-loadable? Resume via
# --ckpt last and take a few more steps. Same overrides as the first smoke
# (48-sample subset, batch 2, ckpt_last_every_n_steps 10, final snapshot
# disabled -- SegviGenEmissiveDirect has no visualize_sample yet, see
# lightning_port board notes) plus a higher max_steps so the resumed run
# actually advances past step 30.
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
mkdir -p log
python train.py \
  --config configs/gen/segvigen_emissive_direct.yaml \
  --output_dir outputs/smoke_segvigen \
  --ckpt last \
  --num_gpus 1 \
  --override dataset.args.max_samples=48 \
  --override trainer.args.batch_size_per_gpu=2 \
  --override trainer.args.max_steps=40 \
  --override trainer.args.i_log=1 \
  --override trainer.args.val_every_epochs=999 \
  --override trainer.args.ckpt_last_every_n_steps=10 \
  --override trainer.args.disable_final_snapshot=true \
  2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-smoke_segvigen_resume.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=${TRAIN_EXIT}"
ls -la outputs/smoke_segvigen/ckpts/ || true
exit $TRAIN_EXIT
