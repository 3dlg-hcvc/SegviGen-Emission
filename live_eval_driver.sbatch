#!/bin/bash
#SBATCH -J live_eval_driver
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/live_eval_driver_%j.log
# The standing watcher of the live evaluation loop (board live_eval_loop). It
# holds no GPU and almost no CPU: it reads the training run's last pointer,
# snapshots each new epoch into the mirror, and submits one evaluation at a time.
#
# It chains its own successor when its 12 hours are up, so the loop survives
# sessions. Two things stop it: a file named STOP in the store directory, or a
# crash within the first ten minutes, which is treated as a real failure rather
# than something to retry forever.
set -x
STORE="${LIVE_EVAL_STORE:-/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval}"
MIRROR="${LIVE_EVAL_MIRROR:-/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/segvigen_256_bw_6gpu_ckpts_mirror}"
cd /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe
START=$(date +%s)
/usr/bin/python3 scripts_segvigen/live_eval_driver.py \
  --store "${STORE}" \
  --mirror "${MIRROR}" \
  --max_hours 11.5
EXIT=$?
ELAPSED=$(( $(date +%s) - START ))
echo "LIVE_EVAL_DRIVER_EXIT=${EXIT} ELAPSED=${ELAPSED}s"
if [ -f "${STORE}/STOP" ]; then
  echo "STOP file present; not chaining a successor"
elif [ "${ELAPSED}" -lt 600 ]; then
  echo "the driver exited after only ${ELAPSED}s; treating that as a real failure, not chaining"
else
  sbatch --export=ALL,LIVE_EVAL_STORE="${STORE}",LIVE_EVAL_MIRROR="${MIRROR}" \
    live_eval_driver.sbatch
fi
exit ${EXIT}
