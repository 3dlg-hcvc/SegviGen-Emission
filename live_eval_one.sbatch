#!/bin/bash
#SBATCH -J live_eval_one
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
# Never cs-venus-05: the training run this loop watches holds all six of its
# GPUs. cs-venus-09 (unexplained hang) and cs-venus-19 (Blackwell, wrong torch
# build) are the standing exclusions. That leaves cs-venus-07/13/14 for a40.
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/live_eval_one_%j.log
# One checkpoint of the live evaluation loop: 10 shapes x 5 draws, IoU, thumbnails.
# Submitted by live_eval_driver.py, which sets every variable below.
set -x
: "${LIGHTNING_CKPT:?set by live_eval_driver.py}"
: "${LIVE_EVAL_STEP:?set by live_eval_driver.py}"
: "${LIVE_EVAL_STORE:?set by live_eval_driver.py}"
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
nvidia-smi --query-gpu=name,memory.total --format=csv
VLOSS_ARG=""
if [ -n "${LIVE_EVAL_VAL_LOSS}" ] && [ "${LIVE_EVAL_VAL_LOSS}" != "None" ]; then
  VLOSS_ARG="--val_loss ${LIVE_EVAL_VAL_LOSS}"
fi
python3 scripts_segvigen/live_eval_one.py \
  --lightning_ckpt "${LIGHTNING_CKPT}" \
  --step "${LIVE_EVAL_STEP}" \
  ${VLOSS_ARG} \
  --store "${LIVE_EVAL_STORE}" \
  ${LIVE_EVAL_EXTRA}
EXIT=$?
echo "LIVE_EVAL_ONE_EXIT=${EXIT}"
exit ${EXIT}
