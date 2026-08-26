#!/bin/bash
#SBATCH -J stage_shards_val
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --exclude=cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards/log/val_%j.log
set -x
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/stage_shards.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --split val_72k \
  --out_dir /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards \
  --shard_idx 0 --num_shards 1
