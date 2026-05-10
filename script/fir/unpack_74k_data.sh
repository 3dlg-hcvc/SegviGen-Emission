#!/bin/bash
# Unpack the 74k voxel dataset on fir cluster nodes.
# Source: /scratch/dya78/lightgen/lightgen_74k/{voxels/part{0..7}_voxels.tar,*.parquet,*.json}
# Extracts to: ./lightgen_74k/{pbr_voxels_256,emission_voxels_256,dual_grid_256}/
#
# If running inside a SLURM job, stages tars to $SLURM_TMPDIR first.
#
# Usage:
#   bash unpack_74k_data.sh [/path/to/lightgen_74k_source_dir]

set -e

DEFAULT_SRC="/scratch/${USER}/lightgen/lightgen_74k"
SRC_DIR="${1:-${DEFAULT_SRC}}"
TARGET_DIR="."

if [ -n "$SLURM_TMPDIR" ]; then
    echo "=== 74K Voxel Data Unpacking (Fir Cluster, SLURM) ==="
else
    echo "=== 74K Voxel Data Unpacking ==="
fi
echo "Source dir : ${SRC_DIR}"
echo "Target dir : $(pwd)/lightgen_74k"
echo ""

mkdir -p "${TARGET_DIR}/lightgen_74k"
DATA_PATH="${TARGET_DIR}/lightgen_74k"

# Verify all 8 voxel tars present
for n in 0 1 2 3 4 5 6 7; do
    tar_path="${SRC_DIR}/voxels/part${n}_voxels.tar"
    if [ ! -f "${tar_path}" ]; then
        echo "ERROR: missing ${tar_path}"
        exit 1
    fi
done

# Optional staging to local SSD
STAGE_DIR="${SRC_DIR}/voxels"
if [ -n "$SLURM_TMPDIR" ] && [[ "${SRC_DIR}" != ${SLURM_TMPDIR}/* ]]; then
    STAGE_DIR="${SLURM_TMPDIR}/lightgen_74k_voxels_in"
    mkdir -p "${STAGE_DIR}"
    echo "Staging 8 tars to local SSD: ${STAGE_DIR}"
    rsync -ah --info=progress2 "${SRC_DIR}/voxels/" "${STAGE_DIR}/"
    echo "Staging complete"
    echo ""
fi

# Extract all 8 tars into ./lightgen_74k/
echo "Extracting 8 tars into ${DATA_PATH}/ ..."
for n in 0 1 2 3 4 5 6 7; do
    echo "  part${n}..."
    tar -xf "${STAGE_DIR}/part${n}_voxels.tar" -C "${DATA_PATH}"
done
echo "Extraction complete"

# Copy parquet + every split JSON we find (pinned/stratified/voxonly/etc.)
if [ -f "${SRC_DIR}/df_SomgProc_final.parquet" ]; then
    cp "${SRC_DIR}/df_SomgProc_final.parquet" "${DATA_PATH}/"
else
    echo "WARNING: missing df_SomgProc_final.parquet"
fi
shopt -s nullglob
for f in "${SRC_DIR}"/data_splits_emissive_74k_*.json; do
    cp "${f}" "${DATA_PATH}/"
done
shopt -u nullglob

# Verify
echo ""
echo "=== Verifying ==="
EMIT_COUNT=$(ls "${DATA_PATH}/emission_voxels_256/"*.vxz 2>/dev/null | wc -l)
PBR_COUNT=$(ls "${DATA_PATH}/pbr_voxels_256/"*.vxz 2>/dev/null | wc -l)
DG_COUNT=$(ls "${DATA_PATH}/dual_grid_256/"*.vxz 2>/dev/null | wc -l)
DATA_SIZE=$(du -sh "${DATA_PATH}" | cut -f1)

echo "Data directory : ${DATA_PATH} (${DATA_SIZE})"
echo "Emission voxels: ${EMIT_COUNT}"
echo "PBR voxels     : ${PBR_COUNT}"
echo "Dual grids     : ${DG_COUNT}"

for f in "df_SomgProc_final.parquet" "data_splits_emissive_74k_pinned.json"; do
    if [ -f "${DATA_PATH}/${f}" ]; then
        echo "  OK: ${f}"
    else
        echo "  MISSING: ${f}"
    fi
done

# Optional: clean up staged tars to free $SLURM_TMPDIR space (~43 GB)
if [ -n "$SLURM_TMPDIR" ] && [[ "${STAGE_DIR}" == ${SLURM_TMPDIR}/* ]]; then
    echo ""
    echo "Removing staged tars (${STAGE_DIR}) to free local SSD..."
    rm -rf "${STAGE_DIR}"
fi

echo ""
echo "Data ready at: $(cd "${DATA_PATH}" && pwd)"
