#!/usr/bin/env bash
# Extract all per-shard tars (voxels + latents) from jupyter into a unified
# lightgen_74k tree on the caller's local disk. Idempotent — re-running is safe.
set -euo pipefail

JUP_BASE=${1:-/3dlg-jupiter-project/lightgen/dataset/lightgen_74k}
DST=${2:-/localscratch/dya78/lightgen_74k}

mkdir -p "${DST}"
echo "[extract_74k_tars] copying scaffold from ${JUP_BASE} -> ${DST}"
for f in df_SomgProc_final.parquet data_splits_emissive_74k_pinned.json metadata.csv; do
    rsync -av "${JUP_BASE}/${f}" "${DST}/"
done

echo "[extract_74k_tars] extracting voxels tars"
for tar in "${JUP_BASE}/voxels"/part*.tar; do
    [ -e "$tar" ] || { echo "  skip (no match): $tar"; continue; }
    echo "  extracting $(basename "$tar")"
    tar -xf "$tar" -C "${DST}"
done

echo "[extract_74k_tars] extracting latents tars"
for tar in "${JUP_BASE}/latents"/part*.tar; do
    [ -e "$tar" ] || { echo "  skip (no match): $tar"; continue; }
    echo "  extracting $(basename "$tar")"
    tar -xf "$tar" -C "${DST}"
done

echo "[extract_74k_tars] done. Tree at: ${DST}"
ls "${DST}"
