#!/bin/bash
#SBATCH -J lightning_smoke_dit10
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_dit10_%j.log

# Smoke: does the team's TRELLIS.2-lightning trainer run in our hands?
# Config is Dongchen's own 10-sample emission-DiT overfit config, unmodified;
# only step counts / save cadence are overridden so it finishes in minutes.
# Working copy lives on falas (our project root); Dongchen's latents are read
# where they live, on jupiter, through data/ symlinks. Nothing is copied.
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
mkdir -p log outputs/smoke_dit10
nvidia-smi
python -c "import lightning, torch, wandb; print('lightning', lightning.__version__, 'torch', torch.__version__, 'wandb', wandb.__version__)"

TS=$(date +%Y%m%d-%H%M%S)
python train.py \
  --config configs/gen/emission_dit_10sample_overfit.yaml \
  --output_dir outputs/smoke_dit10 \
  --ckpt none \
  --num_gpus 1 \
  --override trainer.args.max_steps=30 \
  --override trainer.args.i_save=10 \
  --override trainer.args.i_log=1 \
  --override trainer.args.i_sample=0 \
  --override trainer.args.wandb_enable=true \
  --override trainer.args.wandb_project=lightgen-lightning-probe \
  --override trainer.args.wandb_name=lightning_feasibility_smoke \
  2>&1 | tee "log/${TS}-smoke_dit10.log"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "=== ckpts written ==="
ls -la outputs/smoke_dit10/ckpts/ || true
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit $TRAIN_EXIT
