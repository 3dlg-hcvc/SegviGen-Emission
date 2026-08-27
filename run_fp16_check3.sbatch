#!/bin/bash
#SBATCH -J check_cond_fp16_forward3
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-07,cs-venus-08,cs-venus-09,cs-venus-13,cs-venus-14,cs-venus-17,cs-venus-19,cs-venus-21
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/check_cond_fp16_forward3_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export PYTHONPATH="/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/pysite:$PYTHONPATH"
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 -u scripts_segvigen/check_cond_fp16_forward.py --n 20 --seed 0
