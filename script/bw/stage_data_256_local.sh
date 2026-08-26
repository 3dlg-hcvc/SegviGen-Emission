#!/bin/bash
# Stage all four 256-lane data sources onto venus-05 local disk (owner
# decision, board lightning_port, 2026-08-26: /localscratch is NOT
# periodically wiped there, and the full 256 lane totals ~90GB against
# 184GB free -- moving everything local removes the jupiter NFS ceiling
# that was gating training at ~6s/step with GPUs ~47% idle).
#
# Sources:
#   emission_npz   OUR build, falas, flat <sid>.npz (+ token_counts.json), 2.1GB
#   shape_latents  Dongchen's shared latents_v2, jupiter, ONLY this lane's
#                  36,613 sids (the full corpus is 72,421+ -- do not copy it)
#   pbr_latents    same, jupiter, same subset
#   cond_fp16      job 257957's fp16 relocation, falas, per-sid dirs, 77GB
#
# Run ON venus-05 (paths are node-local). Idempotent: every stage is
# skip-if-already-staged by count, so a killed/resubmitted run only
# finishes what's missing. Writes MANIFEST_OK on full success; the
# launcher gates on this file (see launch_segvigen_256_6gpu.sbatch's
# self-healing sync block) and re-runs this script if it's missing/short.
set -uo pipefail
set -x

DST=/localscratch/xya120/segvigen_data_256
mkdir -p "$DST"

FALAS_EMIT=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/latents_256/output_tex_latents/segvigen_emissive_mask_256
JUPITER_LATENTS=/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2
SHAPE_LATENT_NAME=shape_enc_next_dc_f16c32_fp16_256
PBR_LATENT_NAME=tex_enc_next_dc_f16c32_fp16_256
FALAS_COND=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_fp16
SID_LIST="$DST/_sid_list.txt"

# --- 0. sid list: exactly the sids this lane covers (the emission npz
# tree IS the definition -- it is the smallest of the four sources) ---
if [ ! -s "$SID_LIST" ]; then
    ls "$FALAS_EMIT" | grep '\.npz$' | sed 's/\.npz$//' > "$SID_LIST"
fi
N_SIDS=$(wc -l < "$SID_LIST")
echo "sid list: $N_SIDS sids"

# --- 1. emission_npz/ (2.1GB, small file count, flat copy from falas --
# falas is fast, not the bottleneck this task exists to fix; plain rsync
# is fine here, unlike the jupiter shape/pbr copy below) ---
if [ ! -d "$DST/emission_npz" ] || [ "$(ls "$DST/emission_npz" 2>/dev/null | grep -c '\.npz$')" -lt "$N_SIDS" ]; then
    mkdir -p "$DST/emission_npz"
    rsync -a "$FALAS_EMIT"/ "$DST/emission_npz"/
fi

# --- 2. shape_latents/<name>/ + pbr_latents/<name>/ (ONLY the needed
# subset of jupiter's shared 72k+ corpus, one streamed tar per source
# instead of 36,613 discrete NFS round trips per source -- that per-file
# cost is exactly what saturated jupiter for cond_pos.pth; do not repeat
# it here at 2x the file count). Subdirectory structure kept IDENTICAL to
# the jupiter layout (shape_latents/<latent_name>/<sid>.npz) so the config
# needs only a root-path change, no reader change.
for spec in "shape_latents:$SHAPE_LATENT_NAME" "pbr_latents:$PBR_LATENT_NAME"; do
    kind="${spec%%:*}"; name="${spec##*:}"
    SRC_DIR="$JUPITER_LATENTS/$kind/$name"
    DST_DIR="$DST/$kind/$name"
    if [ ! -d "$DST_DIR" ] || [ "$(ls "$DST_DIR" 2>/dev/null | grep -c '\.npz$')" -lt "$N_SIDS" ]; then
        mkdir -p "$DST_DIR"
        sed 's/$/.npz/' "$SID_LIST" > "$DST/_${kind}_filelist.txt"
        tar -C "$SRC_DIR" -cf - -T "$DST/_${kind}_filelist.txt" | tar -C "$DST_DIR" -xf -
    fi
done

# --- 3. cond_fp16/ (77GB, per-sid dirs, flat copy from falas -- once job
# 257957 has finished converting all sids) ---
if [ ! -d "$DST/cond_fp16" ] || [ "$(ls "$DST/cond_fp16" 2>/dev/null | wc -l)" -lt "$N_SIDS" ]; then
    mkdir -p "$DST/cond_fp16"
    rsync -a "$FALAS_COND"/ "$DST/cond_fp16"/
fi

# --- 4. manifest (launcher's self-heal gate reads this) ---
n_emit=$(ls "$DST/emission_npz" | grep -c '\.npz$')
n_shape=$(ls "$DST/shape_latents/$SHAPE_LATENT_NAME" | grep -c '\.npz$')
n_pbr=$(ls "$DST/pbr_latents/$PBR_LATENT_NAME" | grep -c '\.npz$')
n_cond=$(ls "$DST/cond_fp16" | wc -l)
echo "emission_npz=$n_emit shape_latents=$n_shape pbr_latents=$n_pbr cond_fp16=$n_cond (expected $N_SIDS each)"
if [ "$n_emit" -ge "$N_SIDS" ] && [ "$n_shape" -ge "$N_SIDS" ] && [ "$n_pbr" -ge "$N_SIDS" ] && [ "$n_cond" -ge "$N_SIDS" ]; then
    cat > "$DST/MANIFEST_OK" <<EOF
emission_npz=$n_emit
shape_latents=$n_shape
pbr_latents=$n_pbr
cond_fp16=$n_cond
expected=$N_SIDS
staged_at=$(date -Iseconds)
EOF
    echo "MANIFEST_OK written"
else
    echo "FATAL: staging incomplete (see counts above), NOT writing MANIFEST_OK"
    exit 1
fi
