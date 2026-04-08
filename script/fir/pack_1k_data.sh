#!/bin/bash
# Pack the 1k voxel dataset into a tar for transfer to fir.
# Excludes pbr_dumps/ (intermediate files not needed for training).
#
# Usage:
#   bash script/fir/pack_1k_data.sh
#
# Then transfer:
#   scp /local-scratch/localhome/dya78/lightgen_1k_voxels.tar fir:/scratch/dya78/lightgen/

set -e

DATA_DIR="/local-scratch/localhome/dya78/lightgen_1k"
TAR_OUT="/local-scratch/localhome/dya78/lightgen_1k_voxels.tar"

echo "Packing ${DATA_DIR} → ${TAR_OUT}"
echo "Excluding: pbr_dumps/"

cd /local-scratch/localhome/dya78/
tar cf "${TAR_OUT}" \
    --exclude='lightgen_1k/pbr_dumps' \
    lightgen_1k/

TAR_SIZE=$(du -sh "${TAR_OUT}" | cut -f1)
echo "Done: ${TAR_OUT} (${TAR_SIZE})"
echo ""
echo "Transfer to fir:"
echo "  scp ${TAR_OUT} fir:/scratch/dya78/lightgen/lightgen_1k_voxels.tar"
