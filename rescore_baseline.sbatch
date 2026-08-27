#!/bin/bash
#SBATCH -J rescore_baseline
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/rescore_baseline_%j.log
set -x
: "${DUMP:?}"; : "${LABEL:?}"; : "${OUT:?}"
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/rescore_baseline_dump.py --dump "${DUMP}" --label "${LABEL}" --out "${OUT}"
echo "RESCORE_EXIT=$?"
