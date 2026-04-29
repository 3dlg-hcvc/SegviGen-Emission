#!/bin/bash
# Unpack the 1k DiT slice (encoded latents + thumbnails + metadata) on fir.
# Source: /scratch/dya78/lightgen/lightgen_1k_dit.tar
# Extracts to: ./lightgen_1k/ (relative to current directory)
#
# If running inside a SLURM job, stages the tar to $SLURM_TMPDIR first.
#
# Contents (DiT-only — no voxels):
#   lightgen_1k/emission_latents/emission_pbr_finetune_1k/{*.npz, normalization_stats.json}
#   lightgen_1k/shape_latents/shape_enc_next_dc_f16c32_fp16_256/{*.npz, normalization_stats.json}
#   lightgen_1k/thumbnails/*.png
#   lightgen_1k/df_SomgProc_emission_filtered.parquet
#   lightgen_1k/data_splits_emission_filtered.json
#
# Usage:
#   bash unpack_1k_dit_data.sh [path/to/lightgen_1k_dit.tar]

set -e

DEFAULT_TAR_SOURCE="/scratch/${USER}/lightgen/lightgen_1k_dit.tar"
TAR_SOURCE="${1:-${DEFAULT_TAR_SOURCE}}"
TARGET_DIR="."

if [ -n "$SLURM_TMPDIR" ]; then
    echo "=== 1K DiT Slice Unpacking (Fir Cluster, SLURM) ==="
else
    echo "=== 1K DiT Slice Unpacking ==="
fi
echo "Extracting to: $(pwd)"
echo ""

if [ ! -f "${TAR_SOURCE}" ]; then
    echo "Error: Archive not found: ${TAR_SOURCE}"
    echo "Usage: $0 [path/to/lightgen_1k_dit.tar]"
    exit 1
fi

TAR_SIZE=$(du -sh "${TAR_SOURCE}" | cut -f1)
echo "Source archive : ${TAR_SOURCE} (${TAR_SIZE})"

if [ -n "$SLURM_TMPDIR" ]; then
    TAR_FILE="${SLURM_TMPDIR}/lightgen_1k_dit.tar"
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

echo ""
echo "Extracting..."
tar -xf "${TAR_FILE}" -C "${TARGET_DIR}"
echo "Extraction complete"

echo ""
echo "=== Verifying ==="
DATA_PATH="${TARGET_DIR}/lightgen_1k"

if [ -d "${DATA_PATH}" ]; then
    EMIT_LAT_COUNT=$(ls "${DATA_PATH}/emission_latents/emission_pbr_finetune_1k/"*.npz 2>/dev/null | wc -l)
    SHAPE_LAT_COUNT=$(ls "${DATA_PATH}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/"*.npz 2>/dev/null | wc -l)
    THUMB_COUNT=$(ls "${DATA_PATH}/thumbnails/" 2>/dev/null | wc -l)
    DATA_SIZE=$(du -sh "${DATA_PATH}" | cut -f1)

    echo "Data directory       : ${DATA_PATH} (${DATA_SIZE})"
    echo "Emission latents     : ${EMIT_LAT_COUNT}"
    echo "Shape latents        : ${SHAPE_LAT_COUNT}"
    echo "Thumbnails           : ${THUMB_COUNT}"

    for f in "df_SomgProc_emission_filtered.parquet" \
             "data_splits_emission_filtered.json" \
             "emission_latents/emission_pbr_finetune_1k/normalization_stats.json" \
             "shape_latents/shape_enc_next_dc_f16c32_fp16_256/normalization_stats.json"; do
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
