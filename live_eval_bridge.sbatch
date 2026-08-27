#!/bin/bash
#SBATCH -J live_eval_bridge
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=30
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/live_eval_bridge_%j.log
# Turn one lightning checkpoint into the standalone format the sampler loads,
# ONCE, before the shards start. Eight shards each unpickling the same 10.5GB
# lightning file would read 84GB off the shared filer per checkpoint and spend
# minutes doing it; they read one 2.6GB file instead. No GPU needed.
set -x
: "${LIGHTNING_CKPT:?}"
: "${LIVE_EVAL_STEP:?}"
: "${LIVE_EVAL_STORE:?}"
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
OUT="${LIVE_EVAL_STORE}/bridged/step$(printf %07d ${LIVE_EVAL_STEP}).pt"
mkdir -p "$(dirname "$OUT")"
if [ -s "$OUT" ]; then echo "already bridged: $OUT"; exit 0; fi
python3 scripts_segvigen/unrekey_for_eval.py "${LIGHTNING_CKPT}" "${OUT}.tmp" && mv "${OUT}.tmp" "${OUT}"
EXIT=$?
echo "LIVE_EVAL_BRIDGE_EXIT=${EXIT}"
exit ${EXIT}
