#!/bin/bash
# Live-code gate for one checkpoint: run_live.sh paper|release <tree> <out> <shapes> [ckpt]
# Paths resolve with /cs on the workstation and lab boxes, without it on Solar.
# Env overrides: GATE_PY (python), SHARD/N_SHARDS.
set -euo pipefail
WHICH=$1; TREE=$2; OUT=$3; SHAPES=$4; CKPT=${5:-}
PFX=""; [ -d /3dlg-jupiter-project/lightgen ] || PFX=/cs
J=$PFX/3dlg-jupiter-project/lightgen; F=$PFX/3dlg-falas/project/omages/lightgen
GATE=$(dirname "$(readlink -f "$0")")
case $WHICH in
  paper)
    CFG=$J/outputs/segvigen_256_bw_6gpu/config.yaml
    CKPT=${CKPT:-$J/outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt}
    M=$J/trellis2_bw/lightgen_74k_newbake
    ARGS=(--split val --data_root $M --parquet_file $M/df_SomgProc_final.parquet
          --split_json $M/data_splits_emissive_74k_stratified_newbake_vae_agentic.json
          --shape_latent_root $J/trellis2_bw/latents_v2 --pbr_latent_root $J/trellis2_bw/latents_v2
          --emission_latent_dir $F/segvigen_emissive/outputs/latents_256/output_tex_latents/segvigen_emissive_mask_256
          --cond_dirs $F/segvigen_emissive/outputs/cond_fp16,$J/segvigen_emissive/dataset_direct/train_72k_agentic,$J/segvigen_emissive/dataset_direct/val_72k) ;;
  release)
    CFG=$J/outputs/segvigen_256_release_4h100/config.yaml
    CKPT=${CKPT:-$J/outputs/segvigen_256_release_4h100/snapshots/step0044722.pt}
    R=$J/datasets/dataset_release_latents
    ARGS=(--split test --data_root $R/meta --parquet_file $R/meta/release_index.parquet
          --split_json $R/meta/release_splits_usable.json --shape_latent_root $R --pbr_latent_root $R
          --emission_latent_dir $R/segvigen_emission_mask_npz --cond_dirs $R/segvigen_cond_fp16) ;;
  *) echo "usage: $0 paper|release <tree> <out> <shapes> [ckpt]"; exit 2 ;;
esac
exec ${GATE_PY:-python} -u $GATE/live_gate.py --tree "$TREE" --run_config $CFG --ckpt $CKPT \
  --shapes "$SHAPES" --out "$OUT" --shard ${SHARD:-0} --n_shards ${N_SHARDS:-1} "${ARGS[@]}"
