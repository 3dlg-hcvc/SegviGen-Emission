#!/bin/bash
#SBATCH -J build256_latents
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-9
# cs-venus-02 is Turing (2080_ti) -- flash_attn needs Ampere+, same class of
# failure as job 257282's untyped-gres miss. cs-venus-05/19/21 are Blackwell
# (sm_120), unsupported by our normal trellis2 env. 08/14 have the dead
# /project mount on lab partitions; 09 hangs; 17 is production, left alone.
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-17,cs-venus-19,cs-venus-21
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/build256_%a_%A.log
# Job 8 (256 target-latent build), owner GO 2026-08-26. 10-way array over
# train_72k_agentic (rank 0-9 each get ~3623 of 36226) covers the whole
# split; val_72k (387) is small enough that rank 0 also does the whole
# thing (idempotent, per-item cached, so this is not a race with the other
# 9 ranks -- they never touch val_72k).
set -x
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/build_output_tex_latents_256.py \
  --split train_72k_agentic --rank ${SLURM_ARRAY_TASK_ID} --world_size 10
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  python3 scripts_segvigen/build_output_tex_latents_256.py --split val_72k --rank 0 --world_size 1
fi
