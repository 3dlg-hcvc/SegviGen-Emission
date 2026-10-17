#!/bin/bash
# SegviGen-Emission on LightgenBench, one stage per call, in the order below:
#
#   TEXVERSE_1K=<dir> bash script/lightgenbench/run_chain.sh latents <data_root>   # dual grid, shape + PBR latents
#   bash script/lightgenbench/run_chain.sh masks <data_root>                        # emission-mask latents
#   bash script/lightgenbench/run_chain.sh cond <data_root>                         # DINOv3 conditioning, then the check
#   bash script/lightgenbench/run_chain.sh train <data_root>                        # the init, then training
#   CKPT=<ckpt> bash script/lightgenbench/run_chain.sh predict <data_root>          # the test split, seeds 0-4
#
# <data_root> is prepare_lightgenbench.sh's output; arguments after it go to train.py (train), such as
# --override trainer.args.dataloader_num_workers=4. The recipe is the release run's: configs/gen/segvigen_emissive_256_release.yaml
# (50,000 steps, AdamW lr 1e-5, init from SegviGen's full_seg), trained on 4 H100 80 GB GPUs with 12 shapes per GPU
# (global 48). The release scored step 44,722, picked on the validation split by IoU; a checkpoint is always named:
# nothing here picks one.
#
# env: NUM_GPUS (4); BS (12; another global batch than the release's 48 is refused unless ANY_BATCH=1); MAX_STEPS;
#      OUTPUTS (outputs: the run lands in <OUTPUTS>/segvigen_emissive_256_release); RESUME=1 to continue a run directory
#      that holds checkpoints (a resume skips the init load, so a first launch must start empty); WORKERS (latents:
#      dual-grid processes; cond: the check's processes; 8); LIGHTGEN_REPO (latents: the lightgen repo, default the
#      directory holding this repo); SEEDS (predict, 0,1,2,3,4); SPLIT (predict, test); PRED_OUT (predict:
#      <OUTPUTS>/predict_<split>); CHAIN_TMPDIR (below); PYTHON (the interpreter, default python: the cond stage needs
#      a transformers that loads BiRefNet, train needs lightning 2.6; README.md, Setup). wandb runs offline unless
#      logged in.
# Each call tees its output to log/<TS>-lightgenbench_<stage>.log. Run from anywhere; it works in the repo root.
set -euo pipefail
STAGE=${1:?latents|masks|cond|train|predict}; ROOT=$(realpath "${2:?data_root}"); shift 2; EXTRA=("$@")
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
abs() { [ -z "${1:-}" ] || realpath -m "$1"; }   # caller paths are relative to where the caller stands
OUTPUTS=$(abs "${OUTPUTS:-$REPO/outputs}"); TEXVERSE_1K=$(abs "${TEXVERSE_1K:-}"); LIGHTGEN_REPO=$(abs "${LIGHTGEN_REPO:-}")
CKPT=$(abs "${CKPT:-}"); PRED_OUT=$(abs "${PRED_OUT:-}"); CHAIN_TMPDIR=$(abs "${CHAIN_TMPDIR:-}")
cd "$REPO"; NUM_GPUS=${NUM_GPUS:-4}; PY=${PYTHON:-python}; W=${WORKERS:-8}
RUN_NAME=segvigen_emissive_256_release
SHAPE_DIR=$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_256; PBR_DIR=$ROOT/pbr_latents/tex_enc_next_dc_f16c32_fp16_256
MASK_DIR=$ROOT/segvigen_emission_mask_npz; COND_DIR=$ROOT/segvigen_cond_fp16
INDEX=$ROOT/meta/release_index.parquet; ALL=$ROOT/meta/release_all.txt; USABLE=$ROOT/meta/release_splits_usable.json
INIT=$REPO/data/ckpt_init/full_seg_rekeyed.pt
mkdir -p log; LOG=log/$(date +%Y%m%d-%H%M%S)-lightgenbench_${STAGE}.log
exec > >(tee -a "$LOG") 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Lightning writes each checkpoint to a temp file in $TMPDIR before moving it into place (11 GB here), and Python
# makes Unix sockets there, whose paths must stay under 108 characters. A small /tmp: CHAIN_TMPDIR=<a short path on a
# disk with room>.
[ -z "$CHAIN_TMPDIR" ] || { mkdir -p "$CHAIN_TMPDIR" && export TMPDIR=$CHAIN_TMPDIR; }
[ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc || export WANDB_MODE=${WANDB_MODE:-offline}

fail() { echo "[chain] FATAL: $*" >&2; exit 2; }
[ "$STAGE" = train ] || [ ${#EXTRA[@]} -eq 0 ] || fail "only the train stage takes arguments after <data_root>"
[ -f "$INDEX" ] && [ -f "$ALL" ] || fail "$ROOT holds no index: run prepare_lightgenbench.sh first"
need() {  # <dir> <name pattern, {u} = the uuid>: every shape of the index has its file there, and every .npz opens whole
  $PY - "$INDEX" "$1" "$2" <<'PY' || fail "$1 is incomplete"
import os, sys, zipfile
import pandas as pd
index, d, pat = sys.argv[1:4]
uuids = [str(u) for u in pd.read_parquet(index).index]
missing, bad = [], []
for u in uuids:
    f = os.path.join(d, pat.replace("{u}", u))
    if not os.path.exists(f):
        missing.append(u)
    elif f.endswith(".npz"):
        try:
            with zipfile.ZipFile(f) as z:
                if z.testzip() is not None:
                    bad.append(u)
        except Exception:
            bad.append(u)
msg = f"[chain] {d}: {len(uuids) - len(missing) - len(bad)} of {len(uuids)}"
if missing:
    msg += f"; {len(missing)} missing (first {missing[0]})"
if bad:
    msg += f"; {len(bad)} unreadable, delete them and rerun the stage: {' '.join(bad[:5])}"
print(msg, flush=True)
os._exit(1 if missing or bad else 0)   # not sys.exit: pyarrow's teardown aborted at exit once (2026-10-12)
PY
}
# GPUs: the caller's CUDA_VISIBLE_DEVICES (a scheduler's allocation), else 0..NUM_GPUS-1
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
per_gpu() {  # <command...>: once per GPU with --rank/--world_size; need() after it decides
  local pids=() r p bad=0
  [ "${#GPUS[@]}" -ge "$NUM_GPUS" ] || fail "NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES lists ${#GPUS[@]}"
  for r in $(seq 0 $((NUM_GPUS - 1))); do CUDA_VISIBLE_DEVICES=${GPUS[$r]} "$@" --rank "$r" --world_size "$NUM_GPUS" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  [ $bad -eq 0 ] || echo "[chain] a rank of ${*:1:3} exited with an error (its log above)"
}
ckpt_cfg() {  # <ckpt>: the run's config.yaml, one level above ckpts/ or snapshots/
  [ -f "$1" ] || fail "no checkpoint $1"; echo "$(dirname "$(dirname "$(realpath "$1")")")/config.yaml"
}
tmp_room() {  # <GB>: the temp folder holds one checkpoint, and its path is short
  local t=${TMPDIR:-/tmp} gb
  [ ${#t} -le 60 ] || fail "TMPDIR $t is ${#t} characters, too long for the Unix sockets made under it: set CHAIN_TMPDIR to a shorter path"
  gb=$(df -BG --output=avail "$t" | tail -1 | tr -dc 0-9)
  [ "$gb" -ge "$1" ] || fail "$t has $gb GB free and a checkpoint passes through it ($1 GB): set CHAIN_TMPDIR to a short path on a disk with room"
}
split_uuids() {  # <split>: the uuids of the split's usable rows, as train.py reads them
  $PY -c 'import json, os, sys, pandas as pd
idx = [str(u) for u in pd.read_parquet(sys.argv[1]).index]
print("\n".join(idx[i] for i in json.load(open(sys.argv[2]))[sys.argv[3]]), flush=True); os._exit(0)' "$INDEX" "$USABLE" "$1"
}

case "$STAGE" in
  latents)
    [ -n "${TEXVERSE_1K:-}" ] || fail "set TEXVERSE_1K to a TexVerse-1K download (glbs/glbs_1k/<group>/<uuid>_1024.glb)"
    need "$ROOT/pbr_voxels_256" "{u}.vxz"
    pids=(); bad=0
    for r in $(seq 0 $((W - 1))); do
      $PY -u data_toolkit/dual_grid_from_glb.py --parquet "$INDEX" --glb_root "$TEXVERSE_1K" --out_root "$ROOT" \
        --loader lightgenbench --check_cells ${LIGHTGEN_REPO:+--lightgen_repo "$LIGHTGEN_REPO"} \
        --rank "$r" --world_size "$W" & pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || bad=1; done
    [ $bad -eq 0 ] || echo "[chain] a dual-grid process failed; failed shapes are in $ROOT/dual_grid_256/failed_rank*.txt"
    need "$ROOT/dual_grid_256" "{u}.vxz"
    per_gpu $PY -u data_toolkit/encode_shape_from_dual_grid.py --data_root "$ROOT" --parquet "$INDEX" --resolution 256
    per_gpu $PY -u data_toolkit/encode_pbr_from_voxels.py --data_root "$ROOT" --parquet "$INDEX" --resolution 256
    need "$SHAPE_DIR" "{u}.npz"; need "$PBR_DIR" "{u}.npz"
    ;;
  masks)
    need "$ROOT/release" "{u}/emission_voxels.vxz"
    per_gpu $PY -u data_toolkit/encode_emission_mask_latents.py --sid_list "$ALL" --release_root "$ROOT/release" --out_root "$MASK_DIR"
    need "$MASK_DIR" "{u}.npz"
    ;;
  cond)
    need "$ROOT/release" "{u}/thumbnail.png"
    per_gpu $PY -u data_toolkit/build_cond_tree.py --sid_list "$ALL" --release "$ROOT/release" --out "$COND_DIR" --routes build
    need "$COND_DIR" "{u}/cond_pos.pth"
    # The check every training input passes; writes meta/release_splits_usable.json and the sampler's token counts.
    need "$SHAPE_DIR" "{u}.npz"; need "$PBR_DIR" "{u}.npz"; need "$MASK_DIR" "{u}.npz"
    $PY -u data_toolkit/verify_and_pack.py verify --root "$ROOT" --workers "$W"
    ;;
  train)
    [ -f "$USABLE" ] || fail "no $USABLE: run the cond stage first (its check writes it)"
    BS=${BS:-12}
    if [ $((NUM_GPUS * BS)) -ne 48 ] && [ "${ANY_BATCH:-0}" != 1 ]; then
      fail "$NUM_GPUS GPUs x $BS = $((NUM_GPUS * BS)); the release's global batch is 48 (ANY_BATCH=1 to run anyway)"
    fi
    if [ ! -f "$INIT" ]; then   # SegviGen's full_seg checkpoint, rekeyed for train.py (gitignored, seconds)
      SRC=$($PY -c 'import os; from huggingface_hub import hf_hub_download
print(hf_hub_download("fenghora/SegviGen", "full_seg.ckpt"), flush=True); os._exit(0)')
      mkdir -p "$(dirname "$INIT")"; $PY data_toolkit/init_segvigen_from_full_seg.py "$SRC" "$INIT.tmp" && mv "$INIT.tmp" "$INIT"
    fi
    OUT=$OUTPUTS/$RUN_NAME; tmp_room 11
    ckpt=last
    if [ "${RESUME:-0}" != 1 ]; then
      ckpt=none   # a fresh run: no resume, so the init load happens
      if compgen -G "$OUT/ckpts/*.ckpt" >/dev/null || compgen -G "$OUT/ckpts/misc_step*.pt" >/dev/null; then
        fail "$OUT holds checkpoints: RESUME=1 continues that run (without the init load); a fresh run needs an empty directory"
      fi
    fi
    # one epoch in optimizer steps: last.ckpt is refreshed once per epoch, as in the release run
    EPOCH_STEPS=$(split_uuids train | grep -c . | awk -v g=$((NUM_GPUS * BS)) '{n = int($1 / g); print (n > 0 ? n : 1)}')
    $PY -u train.py --config configs/gen/segvigen_emissive_256_release.yaml --output_dir "$OUT" --ckpt "$ckpt" \
      --num_gpus "$NUM_GPUS" --override trainer.args.batch_size_per_gpu="$BS" --override trainer.args.batch_split=1 \
      ${MAX_STEPS:+--override trainer.args.max_steps=$MAX_STEPS} \
      --override trainer.args.ckpt_last_every_n_steps="$EPOCH_STEPS" --override trainer.args.disable_final_snapshot=true \
      --override trainer.args.finetune_ckpt.denoiser="$INIT" \
      --override dataset.args.data_root="$ROOT/meta" --override dataset.args.parquet_file="$INDEX" \
      --override dataset.args.split_json="$USABLE" \
      --override dataset.args.shape_latent_root="$ROOT" --override dataset.args.pbr_latent_root="$ROOT" \
      --override dataset.args.emission_latent_dir="$MASK_DIR" \
      --override "dataset.args.cond_search_dirs=[\"$COND_DIR\"]" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  predict)
    [ -n "${CKPT:-}" ] || fail "set CKPT (the release: step 44,722, picked on the validation split)"
    [ -f "$USABLE" ] || fail "no $USABLE: run the cond stage first (its check writes it)"
    SPLIT=${SPLIT:-test}; OUT=${PRED_OUT:-$OUTPUTS/predict_$SPLIT}; mkdir -p "$OUT"
    split_uuids "$SPLIT" > "$OUT/$SPLIT.txt"
    echo "[chain] predicting $(grep -c . "$OUT/$SPLIT.txt") $SPLIT shapes into $OUT"
    for S in $(echo "${SEEDS:-0,1,2,3,4}" | tr , ' '); do
      # <OUT>/seed<S>/mask/draw0/<uuid>.npz: the mask on the 256 grid; .../ma/: mask x the asset's albedo
      $PY -u data_toolkit/predict_emission.py --run_config "$(ckpt_cfg "$CKPT")" --ckpt "$CKPT" --sids_file "$OUT/$SPLIT.txt" \
        --seed "$S" --out_dir "$OUT/seed$S/mask" --ma_out_dir "$OUT/seed$S/ma" \
        --shape_latent_dir "$SHAPE_DIR" --pbr_latent_dir "$PBR_DIR" --cond_dirs "$COND_DIR"
    done
    ;;
  *) fail "unknown stage '$STAGE'" ;;
esac
echo "[chain] $STAGE done; log $LOG"
