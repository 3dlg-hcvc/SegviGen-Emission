#!/bin/bash
# Training check through the real train.py of the tree under test, one GPU:
#   train_check.sh <tree> <out> fresh|resume|resume_paper
# fresh         from the full_seg init, 16 release train shapes (max_samples), batch 1, 32 steps:
#               two epochs, two validations (16 val shapes), saves at steps 16 and 32
# resume        from the release run's ckpts/last.ckpt (step 49,270) on the full release data, 17 steps,
#               a save every 8 steps
# resume_paper  from the paper snapshot (step 23,405) with the same release data and config, 17 steps:
#               shows the code still loads the paper's full training state
# The run's record.jsonl (sitecustomize.py) and summary.jsonl (summarize_ckpt.py on every saved file)
# go to <out>. Env: GATE_PY (python). Paths resolve with /cs on lab boxes, without it on Solar.
set -euo pipefail
TREE=$(readlink -f "$1"); OUT=$2; MODE=$3
PFX=""; [ -d /3dlg-jupiter-project/lightgen ] || PFX=/cs
J=$PFX/3dlg-jupiter-project/lightgen; F=$PFX/3dlg-falas/project/omages/lightgen
R=$J/datasets/dataset_release_latents; REL=$J/outputs/segvigen_256_release_4h100
GATE=$(dirname "$(readlink -f "$0")")
PY=${GATE_PY:-python}
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
[ -e "$OUT/run" ] && { echo "refusing: $OUT/run exists (a first launch never starts in a non-empty dir)"; exit 2; }
COMMON=(--num_gpus 1
  --override trainer.args.batch_size_per_gpu=1 --override trainer.args.batch_split=1
  --override trainer.args.val_batch_size_per_gpu=1 --override trainer.args.dataloader_num_workers=2
  --override trainer.args.wandb_enable=false --override trainer.args.i_log=1
  --override trainer.args.disable_final_snapshot=true
  --override trainer.args.finetune_ckpt.denoiser=$R/init/full_seg_rekeyed.pt
  --override dataset.args.data_root=$R/meta --override dataset.args.parquet_file=$R/meta/release_index.parquet
  --override dataset.args.split_json=$R/meta/release_splits_usable.json
  --override dataset.args.shape_latent_root=$R --override dataset.args.pbr_latent_root=$R
  --override dataset.args.emission_latent_dir=$R/segvigen_emission_mask_npz
  --override "dataset.args.cond_search_dirs=[\"$R/segvigen_cond_fp16\"]")
case $MODE in
  fresh)  EXTRA=(--ckpt none --override dataset.args.max_samples=16 --override trainer.args.max_steps=32
                 --override trainer.args.ckpt_last_every_n_steps=16) ;;
  resume) EXTRA=(--ckpt $REL/ckpts/last.ckpt --override trainer.args.max_steps=49287
                 --override trainer.args.ckpt_last_every_n_steps=8) ;;
  resume_paper)
          EXTRA=(--ckpt $F/segvigen_emissive/outputs/segvigen_256_bw_6gpu_ckpts_mirror/step0023405_snapshot.ckpt
                 --override trainer.args.max_steps=23422 --override trainer.args.ckpt_last_every_n_steps=8) ;;
  *) echo "usage: $0 <tree> <out> fresh|resume|resume_paper"; exit 2 ;;
esac
export PYTHONPATH=$GATE/traincheck${PYTHONPATH:+:$PYTHONPATH}
export TRAINCHECK_RECORD=$OUT/record.jsonl CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-/tmp}
rm -f "$TRAINCHECK_RECORD"
echo "[train_check] $(date) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tree=$(cat $TREE/TREE_REV 2>/dev/null || git -C $TREE rev-parse HEAD) mode=$MODE"
cd "$TREE"
"$PY" -u train.py --config configs/gen/segvigen_emissive_256_release.yaml --output_dir "$OUT/run" "${COMMON[@]}" "${EXTRA[@]}"
"$PY" "$GATE/summarize_ckpt.py" $(ls "$OUT"/run/ckpts/*.ckpt | sort) > "$OUT/summary.jsonl"
echo "TRAIN_CHECK_DONE mode=$MODE steps=$(grep -c '"kind": "step"' "$OUT/record.jsonl") saves=$(wc -l < "$OUT/summary.jsonl")"
