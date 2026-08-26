#!/bin/bash
#SBATCH -J lightning_smoke_resume
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_resume_%j.log

# Second half of the smoke: is the checkpoint the first run wrote re-loadable?
# Resume from last.ckpt (step 30) and take 4 more steps.
set -x
PROBE=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2

export PYTHONPATH="$PROBE/pysite:$PYTHONPATH"
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export WANDB_MODE=offline
export WANDB_DIR="$PROBE/outputs/smoke_dit10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROBE"
TS=$(date +%Y%m%d-%H%M%S)
python train.py \
  --config configs/gen/emission_dit_10sample_overfit.yaml \
  --output_dir outputs/smoke_dit10 \
  --ckpt last \
  --num_gpus 1 \
  --override trainer.args.max_steps=34 \
  --override trainer.args.i_save=1000 \
  --override trainer.args.i_log=1 \
  --override trainer.args.i_sample=0 \
  --override trainer.args.wandb_enable=false \
  2>&1 | tee "log/${TS}-smoke_resume.log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit $TRAIN_EXIT
