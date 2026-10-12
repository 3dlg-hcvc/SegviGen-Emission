#!/bin/bash
# Train SegviGen-Emission on LightgenBench, the recipe of the release run (configs/gen/segvigen_emissive_256_release.yaml:
# 50,000 steps, AdamW lr 1e-5, effective batch 48), on one node:
#
#   bash script/lightgenbench/train.sh <inputs_dir> <init.pt> <output_dir> [num_gpus] [batch_per_gpu] [-- extra train.py args]
#
#   inputs_dir  what prepare_inputs.sh wrote          init.pt  what get_init.sh wrote
#   num_gpus x batch_per_gpu should stay 48 (the release ran 4 x 12 on H100 80 GB; batch_size_per_gpu sets memory)
#
# Rerunning the same command resumes from <output_dir>/ckpts/last.ckpt. WANDB=1 logs to wandb (project LightGen as
# in the config; set trainer.args.wandb_project with an extra --override). PYTHON selects the interpreter.
set -euo pipefail
IN=$(readlink -f "${1:?inputs_dir}"); INIT=$(readlink -f "${2:?init.pt}"); OUT=${3:?output_dir}
NG=${4:-4}; BS=${5:-12}; shift $(( $# < 5 ? $# : 5 )); [ "${1:-}" = "--" ] && shift
PY=${PYTHON:-python}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
# one epoch in optimizer steps: last.ckpt is refreshed once per epoch, as in the release run
EPOCH_STEPS=$($PY -c "import json,sys; n=len(json.load(open(sys.argv[1]))['train']); print(max(1, n // ($NG * $BS)))" \
  "$IN/meta/release_splits_usable.json")
WANDB_ON=false; [ "${WANDB:-0}" = 1 ] && WANDB_ON=true
cd "$REPO"
exec $PY train.py --config configs/gen/segvigen_emissive_256_release.yaml --output_dir "$OUT" --ckpt last --num_gpus "$NG" \
  --override trainer.args.batch_size_per_gpu="$BS" --override trainer.args.batch_split=1 \
  --override trainer.args.ckpt_last_every_n_steps="$EPOCH_STEPS" --override trainer.args.disable_final_snapshot=true \
  --override trainer.args.wandb_enable=$WANDB_ON --override trainer.args.finetune_ckpt.denoiser="$INIT" \
  --override dataset.args.data_root="$IN/meta" --override dataset.args.parquet_file="$IN/meta/release_index.parquet" \
  --override dataset.args.split_json="$IN/meta/release_splits_usable.json" \
  --override dataset.args.shape_latent_root="$IN" --override dataset.args.pbr_latent_root="$IN" \
  --override dataset.args.emission_latent_dir="$IN/segvigen_emission_mask_npz" \
  --override "dataset.args.cond_search_dirs=[\"$IN/segvigen_cond_fp16\"]" "$@"
