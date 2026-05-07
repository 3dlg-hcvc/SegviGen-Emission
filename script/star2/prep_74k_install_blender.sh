#!/usr/bin/env bash
# Idempotent Blender install on the current node's /localscratch.
# Sourced from script/star2/prep_74k_stage_ab.sh; sets BLENDER_PATH for the
# caller. No sudo required.
set -euo pipefail

BLENDER_VERSION="4.5.1"
BLENDER_DIR="/localscratch/dya78/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_PATH="${BLENDER_DIR}/blender"
TARBALL_URL="https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
TARBALL_LOCAL="/localscratch/dya78/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

if [[ -x "${BLENDER_PATH}" ]]; then
    echo "[install_blender] already present: ${BLENDER_PATH}"
    export BLENDER_PATH
    return 0
fi

# Multiple ranks on the same node can race here; serialize via flock.
LOCK=/localscratch/dya78/.blender_install.lock
mkdir -p /localscratch/dya78
exec 9>"${LOCK}"
flock 9

# Re-check after acquiring the lock (another rank may have just finished).
if [[ -x "${BLENDER_PATH}" ]]; then
    echo "[install_blender] already present (after lock): ${BLENDER_PATH}"
    export BLENDER_PATH
    flock -u 9
    return 0
fi

echo "[install_blender] downloading Blender ${BLENDER_VERSION} -> ${TARBALL_LOCAL}"
wget -q --show-progress -O "${TARBALL_LOCAL}" "${TARBALL_URL}"

echo "[install_blender] extracting -> /localscratch/dya78/"
tar -xf "${TARBALL_LOCAL}" -C /localscratch/dya78/
rm -f "${TARBALL_LOCAL}"

if [[ ! -x "${BLENDER_PATH}" ]]; then
    echo "[install_blender] FATAL: ${BLENDER_PATH} not present after extraction" >&2
    flock -u 9
    return 1
fi

echo "[install_blender] installing Pillow into Blender's bundled Python"
PILLOW_SCRIPT="$(pwd)/data_toolkit/blender_script/install_pillow.py"
"${BLENDER_PATH}" -b --python "${PILLOW_SCRIPT}"

echo "[install_blender] done. BLENDER_PATH=${BLENDER_PATH}"
export BLENDER_PATH
flock -u 9
