#!/bin/bash
#SBATCH -J verify_cond_fp16_v2
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-07,cs-venus-08,cs-venus-09,cs-venus-13,cs-venus-14,cs-venus-17,cs-venus-19,cs-venus-21
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/verify_cond_fp16_v2_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export PYTHONPATH="/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/pysite:$PYTHONPATH"
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 -u scripts_segvigen/verify_cond_fp16.py --n 20 --seed 0
