#!/bin/bash
# Stage the emission .vxz for a split into the 74k data root, from the part tars.
# CPU-only (pure tar I/O) -> def-msavva_cpu, no GPU. One-time prep before the
# emission-voxel eval GPU job. Persistent into <data_root>/emission_voxels_256/.
#
# Usage:  [SPLIT=test] bash script/fir/stage_test_voxels.sh
NAME="stage_test_voxels"
REPO="/scratch/dya78/TRELLIS.2-lightning"
DATA_ROOT="/scratch/dya78/lightgen/lightgen_74k"
BRANCH="lightgen"
echo "Submitting ${NAME} (split=${SPLIT:-test})"
if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi
"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH -J ${NAME}
#SBATCH --account=def-msavva_cpu
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=%N-%j.out
echo "Node: \$(hostname)  Started: \$(date)"
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH} || { echo "GIT_SYNC_FAILED — aborting"; exit 1; }
source ${REPO}/env/bin/activate
python -u data_toolkit/stage_test_voxels.py \\
    --data_root ${DATA_ROOT} \\
    --tar_dir ${DATA_ROOT}/voxels \\
    --split ${SPLIT:-test}
echo "STAGE_DONE \$(date) -> ${DATA_ROOT}/emission_voxels_256"
EOF
echo "Submitted: ${NAME}"
