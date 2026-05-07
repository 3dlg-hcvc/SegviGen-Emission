#!/usr/bin/env bash
# Submit one AB sbatch job for shard ${RANK}/${WORLD_SIZE}.
# Outputs jobid (single integer) on the last line of stdout for chain capture.
# Stage A: Blender PBR dump. Stage B: voxelize_pbr + dual_grid.
set -euo pipefail

RANK="${1:?usage: $0 RANK WORLD_SIZE [LIMIT]}"
WORLD_SIZE="${2:?usage: $0 RANK WORLD_SIZE [LIMIT]}"
LIMIT_FLAG=""
if [[ -n "${3:-}" ]]; then
    LIMIT_FLAG="--limit ${3}"
fi

JOB_NAME="prep_74k_AB_r${RANK}"
echo "Submitting ${JOB_NAME} (rank ${RANK}/${WORLD_SIZE}, limit='${LIMIT_FLAG}')"

OUT=$(ssh star2 "sbatch --parsable" <<EOF
#!/bin/bash
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-12:00
#SBATCH --nodelist=cs-venus-08,cs-venus-15,cs-venus-16,cs-venus-17
#SBATCH --output=/localscratch/dya78/${JOB_NAME}.out
#SBATCH -J ${JOB_NAME}

set -e
source ~/.bashrc 2>/dev/null || true
cd /localscratch/dya78/TRELLIS.2-lightning
git pull origin lightgen
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

LOCAL=/localscratch/dya78/lightgen_74k_part${RANK}
JUP=/3dlg-jupiter-project/lightgen/dataset/lightgen_74k
mkdir -p \${LOCAL}/log
TS=\$(date +%Y%m%d-%H%M%S)
LOG=\${LOCAL}/log/\${TS}-prep_74k_AB_rank${RANK}.log

# Idempotent Blender install (sets BLENDER_PATH).
. script/star2/prep_74k_install_blender.sh

rsync -av \${JUP}/metadata.csv                            \${LOCAL}/
rsync -av \${JUP}/df_SomgProc_final.parquet               \${LOCAL}/
rsync -av \${JUP}/data_splits_emissive_74k_pinned.json    \${LOCAL}/

(
  set -e
  python data_toolkit/prep_74k_dump.py \\
      --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} \\
      --workers 8 --blender \${BLENDER_PATH} ${LIMIT_FLAG}
  python data_toolkit/voxelize_pbr.py lightgen_74k \\
      --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} \\
      --max_workers 16 --resolution 256
  python data_toolkit/prep_74k_dual_grid.py \\
      --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} \\
      --workers 16 --resolution 256 ${LIMIT_FLAG}
) 2>&1 | tee "\${LOG}"
AB_EXIT=\${PIPESTATUS[0]}

if [[ \${AB_EXIT} -eq 0 ]]; then
  cd \${LOCAL}
  tar -cf part${RANK}_voxels.tar pbr_voxels_256 emission_voxels_256 dual_grid_256
  rsync -av part${RANK}_voxels.tar \${JUP}/voxels/
  rsync -av \${LOG} \${JUP}/logs/
  rm -rf pbr_dumps
fi

exit \${AB_EXIT}
EOF
)

JOBID=$(echo "${OUT}" | tail -n1)
echo "submitted ${JOB_NAME} jobid=${JOBID}"
echo "${JOBID}"
