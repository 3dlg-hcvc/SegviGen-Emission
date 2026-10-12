#!/bin/bash
# Predict emission masks on a LightgenBench split with a trained SegviGen-Emission checkpoint, as the release run was
# scored (seeds 0-4, one draw each, 12 Euler steps, the shipped TRELLIS.2 texture sampler):
#
#   bash script/lightgenbench/predict.sh <inputs_dir> <run_config> <checkpoint> <out_dir> [seeds] [split]
#
#   inputs_dir  what prepare_inputs.sh wrote          run_config  the run's config.yaml (train.sh's <output_dir>/config.yaml)
#   checkpoint  a Lightning checkpoint (<output_dir>/ckpts/*.ckpt) or a snapshot (<output_dir>/snapshots/*.pt)
#   seeds       comma-separated, default 0,1,2,3,4    split  test (default) or val
#
# Writes <out_dir>/seed<N>/mask/draw0/<uuid>.npz (the predicted mask on the 256 grid) and <out_dir>/seed<N>/ma/draw0/
# (mask x the asset's albedo), the inputs of the LightGen repo's point-sampled evaluator (`--voxel_pred_root`).
# One GPU; PYTHON selects the interpreter. Existing outputs are overwritten.
set -euo pipefail
IN=$(readlink -f "${1:?inputs_dir}"); CFG=$(readlink -f "${2:?run_config}"); CKPT=$(readlink -f "${3:?checkpoint}")
OUT=${4:?out_dir}; SEEDS=${5:-0,1,2,3,4}; SPLIT=${6:-test}
PY=${PYTHON:-python}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
# the split's usable shapes, as train.py reads them: rows of release_index.parquet (index = uuid)
$PY -c "import json,sys,pandas as pd; u=pd.read_parquet(sys.argv[1]).index; print('\n'.join(u[i] for i in json.load(open(sys.argv[2]))[sys.argv[3]]))" \
  "$IN/meta/release_index.parquet" "$IN/meta/release_splits_usable.json" "$SPLIT" > "$OUT/$SPLIT.txt"
cd "$REPO"
for S in ${SEEDS//,/ }; do
  echo "[predict.sh] $(date +%T) seed $S, $(wc -l < "$OUT/$SPLIT.txt") $SPLIT shapes"
  $PY data_toolkit/predict_emission_mask_voxels.py --run_config "$CFG" --ckpt "$CKPT" --sids_file "$OUT/$SPLIT.txt" \
    --seed "$S" --out_dir "$OUT/seed$S/mask" --ma_out_dir "$OUT/seed$S/ma" \
    --shape_latent_dir "$IN/shape_latents/shape_enc_next_dc_f16c32_fp16_256" \
    --pbr_latent_dir "$IN/pbr_latents/tex_enc_next_dc_f16c32_fp16_256" --cond_dirs "$IN/segvigen_cond_fp16"
done
echo "[predict.sh] $(date +%T) done: $OUT"
