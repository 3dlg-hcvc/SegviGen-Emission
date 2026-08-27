#!/bin/bash
#SBATCH -J live_eval_merge
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=20
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/live_eval_merge_%j.log
# Join one checkpoint's shard results into a single record. Runs on afterANY,
# not afterok, so a dead shard still produces a record that SAYS its coverage
# is partial rather than leaving the checkpoint silently unevaluated.
set -x
: "${LIVE_EVAL_STEP:?}"
: "${LIVE_EVAL_STORE:?}"
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
python3 scripts_segvigen/live_eval_merge.py \
  --store "${LIVE_EVAL_STORE}" --step "${LIVE_EVAL_STEP}" \
  --n_shards "${LIVE_EVAL_NSHARDS:-8}" \
  --array_job "${LIVE_EVAL_ARRAY_JOB:-}" \
  --started_at "${LIVE_EVAL_STARTED_AT:-0}" ${LIVE_EVAL_MERGE_EXTRA}
EXIT=$?
echo "LIVE_EVAL_MERGE_EXIT=${EXIT}"
exit ${EXIT}
