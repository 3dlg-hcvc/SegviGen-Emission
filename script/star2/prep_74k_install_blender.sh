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

# Step 1: Blender binary install (node-shared; flock to avoid races).
if [[ ! -x "${BLENDER_PATH}" ]]; then
    LOCK=/localscratch/dya78/.blender_install.lock
    mkdir -p /localscratch/dya78
    exec 9>"${LOCK}"
    flock 9

    if [[ ! -x "${BLENDER_PATH}" ]]; then
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
        echo "[install_blender] also installing Pillow into Blender's bundled Python (one-time)"
        PILLOW_SCRIPT="$(pwd)/data_toolkit/blender_script/install_pillow.py"
        "${BLENDER_PATH}" -b --python "${PILLOW_SCRIPT}"
    else
        echo "[install_blender] already present (after lock): ${BLENDER_PATH}"
    fi
    flock -u 9
else
    echo "[install_blender] already present: ${BLENDER_PATH}"
fi

# Step 2: Per-repo Pillow install (each rank's clone needs its own
# blender_packages/PIL with .so files; .so are gitignored so a fresh clone is
# broken). Idempotent: skip if _imaging.so is already present.
REPO_PIL_SO="$(pwd)/blender_packages/PIL/_imaging.cpython-311-x86_64-linux-gnu.so"
if [[ ! -f "${REPO_PIL_SO}" ]]; then
    echo "[install_blender] installing Pillow into $(pwd)/blender_packages/"
    BLENDER_PY="${BLENDER_DIR}/4.5/python/bin/python3.11"
    if [[ -x "${BLENDER_PY}" ]]; then
        # --upgrade ensures the .so files (typically gitignored) are written
        # alongside the tracked .py files.
        "${BLENDER_PY}" -m pip install --target="$(pwd)/blender_packages" --upgrade Pillow
    else
        echo "[install_blender] WARNING: ${BLENDER_PY} not found; skipping --target install" >&2
    fi
else
    echo "[install_blender] blender_packages PIL already complete"
fi

echo "[install_blender] done. BLENDER_PATH=${BLENDER_PATH}"
export BLENDER_PATH
