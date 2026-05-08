#!/usr/bin/env bash
# Submit one C sbatch job for shard ${RANK}/${WORLD_SIZE}, depending on the
# AB job whose id is given as the third arg (or "none" to skip dependency).
# Prints the C jobid on the last line of stdout.
set -euo pipefail

RANK="${1:?usage: $0 RANK WORLD_SIZE AB_JOBID}"
WORLD_SIZE="${2:?usage: $0 RANK WORLD_SIZE AB_JOBID}"
AB_JOBID="${3:?usage: $0 RANK WORLD_SIZE AB_JOBID  (or 'none')}"

DEP_FLAG=""
if [[ "${AB_JOBID}" != "none" ]]; then
    DEP_FLAG="--dependency=afterok:${AB_JOBID}"
fi

JOB_NAME="prep_74k_C_r${RANK}"
echo "Submitting ${JOB_NAME} (rank ${RANK}/${WORLD_SIZE}, dep=${AB_JOBID})"

OUT=$(ssh star2 "sbatch ${DEP_FLAG} --parsable" <<EOF
#!/bin/bash
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00
#SBATCH --nodelist=cs-venus-02,cs-venus-07,cs-venus-09,cs-venus-15,cs-venus-16
#SBATCH --output=/localscratch/dya78/${JOB_NAME}.out
#SBATCH -J ${JOB_NAME}

set -e
source ~/.bashrc 2>/dev/null || true
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

REPO=/localscratch/dya78/TRELLIS.2-lightning_part${RANK}
if [[ ! -d \${REPO}/.git ]]; then
  git clone -b lightgen git@github.com:3dlg-hcvc/TRELLIS.2-lightning.git \${REPO}
fi
cd \${REPO}
git fetch origin lightgen
git stash push -u -m "auto-stash \$(date +%s)" 2>/dev/null || true
git reset --hard origin/lightgen

LOCAL=/localscratch/dya78/lightgen_74k_part${RANK}
JUP=/3dlg-jupiter-project/lightgen/dataset/lightgen_74k
mkdir -p \${LOCAL}/log
TS=\$(date +%Y%m%d-%H%M%S)
LOG=\${LOCAL}/log/\${TS}-prep_74k_C_rank${RANK}.log

(
  set -e
  rsync -av \${JUP}/metadata.csv                            \${LOCAL}/
  rsync -av \${JUP}/df_SomgProc_final.parquet               \${LOCAL}/
  rsync -av \${JUP}/voxels/part${RANK}_voxels.tar           \${LOCAL}/
  cd \${LOCAL} && tar -xf part${RANK}_voxels.tar && cd -

  python data_toolkit/encode_pbr_latent.py --root \${LOCAL} --rank 0 --world_size 1 --resolution 256
  python data_toolkit/encode_shape_from_dual_grid.py --data_root \${LOCAL} --parquet \${LOCAL}/df_SomgProc_final.parquet --rank 0 --world_size 1 --resolution 256
) 2>&1 | tee "\${LOG}"
C_EXIT=\${PIPESTATUS[0]}

if [[ \${C_EXIT} -eq 0 ]]; then
  cd \${LOCAL}
  tar -cf part${RANK}_latents.tar pbr_latents shape_latents
  rsync -av part${RANK}_latents.tar \${JUP}/latents/
  rsync -av \${LOG} \${JUP}/logs/
fi

exit \${C_EXIT}
EOF
)

JOBID=$(echo "${OUT}" | tail -n1)
echo "submitted ${JOB_NAME} jobid=${JOBID}"
echo "${JOBID}"
