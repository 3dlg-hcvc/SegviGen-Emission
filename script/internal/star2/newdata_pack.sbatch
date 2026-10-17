#!/bin/bash
# Verify the release inputs shape by shape, then pack the usable ones into the tars the clusters unpack.
# CPU only; Solar reads jupiter over a faster path than the workstation.
#
#   sbatch -w <node> --cpus-per-task=<workers> --mem=64G --time=04:00:00 --job-name=sv_pack \
#     --output=<latents root>/_logs/%x-%j.log script/internal/star2/newdata_pack.sbatch <code root> <latents root> <workers> [cond shards] [previous usable.txt]
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -euo pipefail
CODE="${1:?code root}"; ROOT="${2:?latents root}"; W="${3:?workers}"; SHARDS="${4:-8}"
PREV="${5:-}"   # usable.txt of the pack already in tars/: keep its cond tars, pack only what became usable since
PY="$HOME/envs/trellis2/bin/python"
cd "$CODE"
echo "[pack] $(date) node=$(hostname) workers=$W code=$(git rev-parse --short HEAD)"
"$PY" -u data_toolkit/verify_and_pack.py verify --root "$ROOT" --workers "$W"
echo "[pack] dropped by reason:"; cut -f2 "$ROOT/meta/dropped.tsv" | cut -d: -f1 | sort | uniq -c || true
if [ -n "$PREV" ]; then
  "$PY" -u data_toolkit/verify_and_pack.py pack --root "$ROOT" --workers "$W" --incremental_from "$PREV"
else
  "$PY" -u data_toolkit/verify_and_pack.py pack --root "$ROOT" --workers "$W" --cond_shards "$SHARDS" --overwrite
fi
ls -la "$ROOT/tars"; echo "[pack] $(date) done"
