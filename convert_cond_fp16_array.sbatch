#!/bin/bash
#SBATCH -J convert_cond_fp16
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-15
# Absolutely nothing on venus-05 (reserved for the 6-GPU training job);
# venus-09 excluded per standing policy (unexplained hang).
#SBATCH --exclude=cs-venus-05,cs-venus-09
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/convert_cond_fp16_%a_%A.log
# Job: relocate + halve the DINOv3 conditioning tensors off jupiter's
# saturated NFS onto falas as fp16 (board lightning_port, 2026-08-26,
# team-lead's diagnosis: training gated ~47% idle on cond_pos.pth reads,
# jupiter server ceiling ~60MB/s vs falas's 200 npz in 65ms in the same
# test). CPU-only, read/write bound, 16-way split -- more workers would not
# help per the same diagnosis (server ceiling, not client-side concurrency).
set -x
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/convert_cond_fp16.py --rank ${SLURM_ARRAY_TASK_ID} --world_size 16
