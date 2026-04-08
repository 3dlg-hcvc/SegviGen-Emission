#!/bin/bash
# Unpack the 1k voxel dataset on fir cluster nodes.
# Source: /scratch/dya78/lightgen/lightgen_1k_voxels.tar
# Extracts to: ./lightgen_1k/ (relative to current directory)
#
# If running inside a SLURM job, stages the tar to $SLURM_TMPDIR first.
#
# Usage:
#   bash unpack_1k_data.sh [path/to/lightgen_1k_voxels.tar]

set -e

DEFAULT_TAR_SOURCE="/scratch/${USER}/lightgen/lightgen_1k_voxels.tar"
TAR_SOURCE="${1:-${DEFAULT_TAR_SOURCE}}"
TARGET_DIR="."

if [ -n "$SLURM_TMPDIR" ]; then
    echo "=== 1K Voxel Data Unpacking (Fir Cluster, SLURM) ==="
else
    echo "=== 1K Voxel Data Unpacking ==="
fi
echo "Extracting to: $(pwd)"
echo ""

# Locate source tar
if [ ! -f "${TAR_SOURCE}" ]; then
    echo "Error: Archive not found: ${TAR_SOURCE}"
    echo "Usage: $0 [path/to/lightgen_1k_voxels.tar]"
    exit 1
fi

TAR_SIZE=$(du -sh "${TAR_SOURCE}" | cut -f1)
echo "Source archive : ${TAR_SOURCE} (${TAR_SIZE})"

# Stage to local SSD if in SLURM job
if [ -n "$SLURM_TMPDIR" ]; then
    TAR_FILE="${SLURM_TMPDIR}/lightgen_1k_voxels.tar"
    if [[ "${TAR_SOURCE}" != ${SLURM_TMPDIR}/* ]]; then
        echo "Staging to local SSD: ${TAR_FILE}"
        rsync -ahP "${TAR_SOURCE}" "${TAR_FILE}"
        echo "Staging complete"
    else
        TAR_FILE="${TAR_SOURCE}"
    fi
else
    TAR_FILE="${TAR_SOURCE}"
fi

# Extract
echo ""
echo "Extracting..."
tar -xf "${TAR_FILE}" -C "${TARGET_DIR}"
echo "Extraction complete"

# Verify
echo ""
echo "=== Verifying ==="
DATA_PATH="${TARGET_DIR}/lightgen_1k"

if [ -d "${DATA_PATH}" ]; then
    EMIT_COUNT=$(ls "${DATA_PATH}/emission_voxels_256/"*.vxz 2>/dev/null | wc -l)
    PBR_COUNT=$(ls "${DATA_PATH}/pbr_voxels_256/"*.vxz 2>/dev/null | wc -l)
    DATA_SIZE=$(du -sh "${DATA_PATH}" | cut -f1)

    echo "Data directory : ${DATA_PATH} (${DATA_SIZE})"
    echo "Emission voxels: ${EMIT_COUNT}"
    echo "PBR voxels     : ${PBR_COUNT}"

    for f in "df_SomgProc_emission_filtered.parquet" "data_splits_emission_filtered.json"; do
        if [ -f "${DATA_PATH}/${f}" ]; then
            echo "  OK: ${f}"
        else
            echo "  MISSING: ${f}"
        fi
    done
else
    echo "ERROR: ${DATA_PATH} not found after extraction!"
    exit 1
fi

echo ""
echo "Data ready at: $(cd "${DATA_PATH}" && pwd)"
