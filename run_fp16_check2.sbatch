#!/bin/bash
#SBATCH -J check_cond_fp16_forward2
#SBATCH --nodelist=cs-venus-13
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/check_cond_fp16_forward2_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export PYTHONPATH="/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/pysite:$PYTHONPATH"
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 -u scripts_segvigen/check_cond_fp16_forward.py --n 20 --seed 0
