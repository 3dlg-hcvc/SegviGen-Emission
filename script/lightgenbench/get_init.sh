#!/bin/bash
# The training's init: SegviGen's full_seg checkpoint (Hugging Face fenghora/SegviGen), rekeyed for train.py.
#
#   bash script/lightgenbench/get_init.sh <out.pt>
#
# PYTHON selects the interpreter (default python); needs huggingface_hub.
set -euo pipefail
OUT=${1:?out.pt}; PY=${PYTHON:-python}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
SRC=$($PY -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("fenghora/SegviGen", "full_seg.ckpt"))')
$PY "$REPO/scripts_segvigen/rekey_full_seg.py" "$SRC" "$OUT"
