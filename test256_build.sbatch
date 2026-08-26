#!/bin/bash
#SBATCH -J test256_build
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --exclude=cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-17,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/test256_build_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/build_output_tex_latents_256.py --split val_72k --rank 0 --world_size 32
python3 scripts_segvigen/check_256_coords_vs_shape.py --split val_72k --world_size 32 --rank 0
