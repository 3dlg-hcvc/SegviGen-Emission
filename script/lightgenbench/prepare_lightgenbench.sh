#!/bin/bash
# Download the LightGenBench release (Hugging Face 3dlg-hcvc/LightgenBench) and lay it out the way SegviGen-Emission's
# producers read it.
#
#   bash script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>
#
# <download_dir>  the Hugging Face files SegviGen-Emission uses: splits.json, checksums.sha256 and
#                 data/{train,val,test}/{voxels,dual_grid,thumbnail}/*.tar (about 30 GB). The dataset is gated: accept
#                 its terms on the Hugging Face page and run `hf auth login` first. Files already there are
#                 kept; SKIP_DOWNLOAD=1 uses <download_dir> as it is.
# <data_root>     written: release/<uuid>/{pbr_voxels,emission_voxels,dual_grid}.vxz and thumbnail.png with
#                 release/splits.json (the release's own layout), the index in meta/ and the links pbr_voxels_256/,
#                 emission_voxels_256/, dual_grid_256/ (data_toolkit/build_lightgenbench_index.py). run_chain.sh
#                 builds the rest there.
# Run from anywhere, in the python env of this repo (README.md, Environment Setup); PYTHON selects the interpreter.
set -euo pipefail
DL=$(realpath -m "${1:?download_dir}"); ROOT=$(realpath -m "${2:?data_root}")
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
PY=${PYTHON:-python}
REPO_ID=${REPO_ID:-3dlg-hcvc/LightgenBench}
# the released splits.json: every published number uses this split.
SPLITS_SHA256=${SPLITS_SHA256:-6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e}
WORKERS=${WORKERS:-8}
mkdir -p "$DL" "$ROOT"

if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  DL="$DL" REPO_ID="$REPO_ID" WORKERS="$WORKERS" $PY - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["REPO_ID"], repo_type="dataset", local_dir=os.environ["DL"],
                  allow_patterns=["splits.json", "checksums.sha256", "data/*/voxels/*.tar", "data/*/dual_grid/*.tar",
                                  "data/*/thumbnail/*.tar"],
                  max_workers=int(os.environ["WORKERS"]))
PY
fi

cd "$DL"
echo "$SPLITS_SHA256  splits.json" | sha256sum -c --quiet - || { echo "splits.json is not the released one" >&2; exit 2; }
grep -E '  data/(train|val|test)/(voxels|dual_grid|thumbnail)/[^/]+\.tar$' checksums.sha256 > "$ROOT/.tars.sha256"
[ -s "$ROOT/.tars.sha256" ] || { echo "checksums.sha256 lists no voxel, dual-grid or thumbnail tar" >&2; exit 2; }
echo "[prepare] checking $(wc -l < "$ROOT/.tars.sha256") tars against checksums.sha256"
xargs -P "$WORKERS" -L 1 sh -c 'echo "$0  $1" | sha256sum -c --quiet -' < "$ROOT/.tars.sha256"

echo "[prepare] unpacking into $ROOT/release"
mkdir -p "$ROOT/release"
# Only the tars just checked; members are <uuid>/{emission_voxels.vxz,pbr_voxels.vxz,dual_grid.vxz,thumbnail.png},
# kept as they are.
awk '{print $2}' "$ROOT/.tars.sha256" | xargs -P "$WORKERS" -I{} tar xf {} -C "$ROOT/release"
cp splits.json "$ROOT/release/splits.json"
$PY - "$ROOT/release" <<'PY' || exit 3
import json, os, sys
rel = sys.argv[1]
uuids = [u for v in json.load(open(os.path.join(rel, "splits.json"))).values() for u in v]
missing = [f"{u}/{f}" for u in uuids for f in ("emission_voxels.vxz", "pbr_voxels.vxz", "dual_grid.vxz", "thumbnail.png")
           if not os.path.isfile(os.path.join(rel, u, f))]
print(f"[prepare] {len(uuids)} shapes in splits.json; {len(missing)} files missing" + (f" (first {missing[0]})" if missing else ""), flush=True)
os._exit(1 if missing else 0)
PY
cd "$REPO"
$PY data_toolkit/build_lightgenbench_index.py --release "$ROOT/release" --out "$ROOT/meta" --link_voxels "$ROOT"
echo "[prepare] done: $ROOT"
