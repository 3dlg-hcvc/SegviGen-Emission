#!/bin/bash
# Extract the slim tar shards built by scripts_segvigen/stage_shards.py
# (outputs/staging_shards/{train_72k_agentic,val_72k}/shard_*.tar) into the
# node-local layout dataset_direct.py expects. Run this ON the target node
# (venus-05/19) once its Mercury-deployment drain lifts and the bootstrap
# above has succeeded -- not before.
set -uo pipefail
SRC=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards
DST=/localscratch/xya120/segvigen_data
mkdir -p "$DST/train_72k_agentic" "$DST/val_72k"

echo "=== extracting train_72k_agentic shards ==="
for t in "$SRC"/train_72k_agentic/shard_*.tar; do
  echo "  $t"
  tar xf "$t" -C "$DST/train_72k_agentic"
done
echo "=== extracting val_72k shard ==="
for t in "$SRC"/val_72k/shard_*.tar; do
  echo "  $t"
  tar xf "$t" -C "$DST/val_72k"
done

echo "=== verify ==="
n_train=$(ls "$DST/train_72k_agentic" | wc -l)
n_val=$(ls "$DST/val_72k" | wc -l)
echo "train_72k_agentic: $n_train sample dirs (source split has 36226)"
echo "val_72k: $n_val sample dirs (source split has 387)"
du -sh "$DST"
