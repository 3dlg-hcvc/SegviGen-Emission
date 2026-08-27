#!/bin/bash
#SBATCH -J live_eval_shard
#SBATCH --partition=cs-gpu-research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=cs-bd-01,cs-gpu1,cs-gpu3,cs-venus-01,cs-venus-02,cs-venus-03,cs-venus-05,cs-venus-06,cs-venus-08,cs-venus-09,cs-venus-14,cs-venus-17,cs-venus-18,cs-venus-19,cs-venus-20,cs-venus-21
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/live_eval_shard_%A_%a.log
# ONE SHARD of one checkpoint's validation run: an independent single-GPU task.
#
# Submitted as an array (--array=0-7), never as one job asking for eight GPUs.
# A job demanding eight cards at once sits in the queue until eight are free
# simultaneously; an array starts whatever it can and the rest join as cards
# open up, which is also what lets several checkpoints' arrays share the pool.
set -x
: "${LIVE_EVAL_STEP:?}"
: "${LIVE_EVAL_STORE:?}"
: "${LIVE_EVAL_NSHARDS:?}"
: "${BRIDGED_CKPT:?}"
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Wait for the bridge job rather than depending on it. This started as a fix for
# what looked like an afterok race on 2026-08-26; the real cause turned out to be
# cs-bd-01 (see the exclude line above), and the polling was kept anyway because
# it removes a dependency edge for about a minute of one GPU per checkpoint and
# lets the shards do their environment setup while the bridge finishes.
WAITED=0
while [ ! -s "${BRIDGED_CKPT}" ]; do
  if [ "${WAITED}" -ge 1500 ]; then
    echo "the bridged checkpoint never appeared at ${BRIDGED_CKPT} after ${WAITED}s"
    exit 3
  fi
  sleep 10
  WAITED=$(( WAITED + 10 ))
done
echo "bridged checkpoint ready after ${WAITED}s: ${BRIDGED_CKPT}"
STEPDIR="${LIVE_EVAL_STORE}/shards/step$(printf %07d ${LIVE_EVAL_STEP})"
mkdir -p "${STEPDIR}"
VLOSS_ARG=""
if [ -n "${LIVE_EVAL_VAL_LOSS}" ] && [ "${LIVE_EVAL_VAL_LOSS}" != "None" ]; then
  VLOSS_ARG="--val_loss ${LIVE_EVAL_VAL_LOSS}"
fi
python3 scripts_segvigen/live_eval_one.py \
  --bridged_ckpt "${BRIDGED_CKPT}" \
  --step "${LIVE_EVAL_STEP}" ${VLOSS_ARG} \
  --store "${LIVE_EVAL_STORE}" \
  --sids_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/val_72k \
  --shard "${SLURM_ARRAY_TASK_ID}" --n_shards "${LIVE_EVAL_NSHARDS}" \
  --wall_sids_file /3dlg-falas/project/omages/lightgen/segvigen_emissive/xgconsole/web/_preview/sample56_latest/dataset_filtering_selected_56.txt \
  --out "${STEPDIR}/shard${SLURM_ARRAY_TASK_ID}.json" \
  --keep_ckpt ${LIVE_EVAL_EXTRA}
EXIT=$?
echo "LIVE_EVAL_SHARD_EXIT=${EXIT}"
exit ${EXIT}
