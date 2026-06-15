#!/bin/bash
# Unpack the 74k DiT slice (encoded latents + thumbnails + metadata) on fir.
# Source: /scratch/dya78/lightgen/lightgen_74k_dit.tar
# Extracts to: ./lightgen_74k/ (relative to current directory)
#
# Contents (DiT-only — no voxels):
#   lightgen_74k/emission_latents/emission_pbr_vae_ft_74k/{*.npz, normalization_stats.json}
#   lightgen_74k/shape_latents/shape_enc_next_dc_f16c32_fp16_256/*.npz
#   lightgen_74k/thumbnails/*.png
#   lightgen_74k/df_SomgProc_final.parquet
#   lightgen_74k/data_splits_emissive_74k_stratified_voxonly.json
#
# Usage:  bash unpack_74k_dit_data.sh [path/to/lightgen_74k_dit.tar]
set -e

DEFAULT_TAR_SOURCE="/scratch/${USER}/lightgen/lightgen_74k_dit.tar"
TAR_SOURCE="${1:-${DEFAULT_TAR_SOURCE}}"
TARGET_DIR="."

if [ ! -f "${TAR_SOURCE}" ]; then
    echo "Error: Archive not found: ${TAR_SOURCE}"; exit 1
fi
echo "Source archive : ${TAR_SOURCE} ($(du -sh "${TAR_SOURCE}" | cut -f1))"

if [ -n "$SLURM_TMPDIR" ]; then
    TAR_FILE="${SLURM_TMPDIR}/lightgen_74k_dit.tar"
    if [[ "${TAR_SOURCE}" != ${SLURM_TMPDIR}/* ]]; then
        echo "Staging to local SSD: ${TAR_FILE}"
        rsync -ahP "${TAR_SOURCE}" "${TAR_FILE}"
    else
        TAR_FILE="${TAR_SOURCE}"
    fi
else
    TAR_FILE="${TAR_SOURCE}"
fi

echo "Extracting..."
tar -xf "${TAR_FILE}" -C "${TARGET_DIR}"

DATA_PATH="${TARGET_DIR}/lightgen_74k"
echo "=== Verifying ==="
EMIT=$(ls "${DATA_PATH}/emission_latents/emission_pbr_vae_ft_74k/"*.npz 2>/dev/null | wc -l)
SHAPE=$(ls "${DATA_PATH}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/"*.npz 2>/dev/null | wc -l)
THUMB=$(ls "${DATA_PATH}/thumbnails/" 2>/dev/null | wc -l)
echo "Emission latents : ${EMIT}"
echo "Shape latents    : ${SHAPE}"
echo "Thumbnails       : ${THUMB}"
for f in "df_SomgProc_final.parquet" \
         "data_splits_emissive_74k_stratified_voxonly.json" \
         "emission_latents/emission_pbr_vae_ft_74k/normalization_stats.json"; do
    [ -f "${DATA_PATH}/${f}" ] && echo "  OK: ${f}" || echo "  MISSING: ${f}"
done
echo "Data ready at: $(cd "${DATA_PATH}" && pwd)"
