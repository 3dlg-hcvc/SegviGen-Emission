#!/bin/bash
#SBATCH -J extract_val_venus05
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/extract_val_venus05_%j.log
# PARTIAL, DELIBERATE: val_72k only (1.9GB), NOT the full extract_staged_shards.sh,
# because /localscratch has only ~150GB free right now vs 175GB of total
# shard tars -- extracting the full set would risk filling a disk shared
# live with other users' jobs. This proves the extraction mechanism end to
# end and stages val_72k for use; train_72k_agentic staging is deferred
# pending disk space, see board note.
set -uo pipefail
set -x
SRC=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/staging_shards
DST=/localscratch/xya120/segvigen_data
mkdir -p "$DST/val_72k"
df -h /localscratch
echo "=== extracting val_72k shard ==="
for t in "$SRC"/val_72k/shard_*.tar; do
  echo "  $t"
  tar xf "$t" -C "$DST/val_72k"
done
echo "=== verify ==="
n_val=$(ls "$DST/val_72k" | wc -l)
echo "val_72k: $n_val sample dirs (source split has 387)"
ls "$DST/val_72k" | head -1 | xargs -I{} ls -la "$DST/val_72k/{}"
du -sh "$DST/val_72k"
df -h /localscratch
