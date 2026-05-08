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
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-12:00
#SBATCH --nodelist=cs-venus-02,cs-venus-07,cs-venus-09,cs-venus-15,cs-venus-16
#SBATCH --output=/localscratch/dya78/${JOB_NAME}.out
#SBATCH -J ${JOB_NAME}

set -e
source ~/.bashrc 2>/dev/null || true
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

# Per-rank repo clone — multiple ranks on the same node can't share one repo
# (concurrent git pull races + uncommitted local changes from prior debugging).
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
LOG=\${LOCAL}/log/\${TS}-prep_74k_AB_rank${RANK}.log

# Idempotent Blender install (sets BLENDER_PATH).
. script/star2/prep_74k_install_blender.sh

rsync -av \${JUP}/metadata.csv                            \${LOCAL}/
rsync -av \${JUP}/df_SomgProc_final.parquet               \${LOCAL}/
rsync -av \${JUP}/data_splits_emissive_74k_pinned.json    \${LOCAL}/

(
  set -e
  python data_toolkit/prep_74k_dump.py --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} --workers 8 --blender \${BLENDER_PATH} ${LIMIT_FLAG}
  # voxelize_pbr.py runs with rank=0 world_size=1 because it post-filters
  # metadata by pbr_dumped==True (only this rank's pickles), and re-slicing
  # that already-sliced set with the parent rank/world drops most samples.
  # prep_74k_dual_grid.py keeps the parent's rank/world (it reads the unfiltered
  # parent metadata.csv and per-sample skips when no pickle exists).
  python data_toolkit/voxelize_pbr.py lightgen_74k --root \${LOCAL} --rank 0 --world_size 1 --max_workers 16 --resolution 256
  python data_toolkit/prep_74k_dual_grid.py --root \${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} --workers 16 --resolution 256 ${LIMIT_FLAG}
  # Merge per-rank new_records/* into <artifact>/metadata.csv so downstream
  # encoders find their expected pbr_voxelized / emission_voxelized flags.
  python -c "import pandas as pd, glob, os; root='\${LOCAL}'
for art in ['pbr_voxels_256','emission_voxels_256','dual_grid_256']:
  parts=sorted(glob.glob(os.path.join(root,art,'new_records','part_*.csv')))
  if not parts: continue
  df=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True).drop_duplicates('sha256',keep='last')
  df.to_csv(os.path.join(root,art,'metadata.csv'),index=False)
  print(f'[merge] {art}: {len(df)} rows')"
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
