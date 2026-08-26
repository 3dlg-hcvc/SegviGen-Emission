#!/bin/bash
#SBATCH -J stage_shards_train
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-31
#SBATCH --exclude=cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards/log/train_%a_%A.log
# Blackwell prep (job 5 amendment): build slim tar shards of
# dataset_direct/train_72k_agentic for later node-local staging on
# venus-05/19. Submitted with --dependency=afterok on the cond_pos.pth
# build array (job 7) so every sample this touches already has its
# sidecar -- see scripts_segvigen/stage_shards.py for exactly what's kept.
set -x
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/stage_shards.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --split train_72k_agentic \
  --out_dir /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards \
  --shard_idx ${SLURM_ARRAY_TASK_ID} --num_shards 32
