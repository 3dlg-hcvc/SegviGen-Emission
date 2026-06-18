#!/bin/bash
# Launch the TWO-STREAM emission DiT 74k training on fir (4 H100). Mirror of
# script/fir/emission_dit_74k.sh (channel-hijack baseline), pointed at the
# two-stream VAE's emission latents + production PBR latents (96-ch input).
#
# Prereq: script/fir/encode_twostream_dit_latents_74k.sh has run and populated
#   /scratch/dya78/lightgen/lightgen_74k/emission_latents/emission_vae_twostream_74k/
#   /scratch/dya78/lightgen/lightgen_74k/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/
# (shape_latents + thumbnails + parquet + split already live there).
#
# DATA I/O: for the probe we read latents directly from persistent /scratch
# (Lustre) — fine for ~100 steps. For the full multi-day run, stage to local SSD
# first (build a DiT tar like unpack_74k_dit_data.sh, then set STAGE=1).
#
# Usage:
#   MAX_STEPS=100 WALLTIME=1:00:00 bash script/fir/emission_dit_twostream_74k.sh   # probe
#   bash script/fir/emission_dit_twostream_74k.sh                                   # full run

CONFIG="configs/gen/emission_dit_twostream_74k.yaml"
NAME="emission_dit_twostream_74k"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/${NAME}"
BRANCH="lightgen"
SCRATCH_74K="/scratch/dya78/lightgen/lightgen_74k"
# MAX_STEPS: set for a bounded probe (e.g. MAX_STEPS=100); unset for the real run.
MAX_STEPS="${MAX_STEPS:-}"
# WALLTIME: short (e.g. 1:00:00) for the probe; default for the real run.
WALLTIME="${WALLTIME:-03:00:00}"

echo "Submitting job: ${NAME}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --mem=1024G
#SBATCH --account=rrg-msavva
#SBATCH --requeue
# 25-min lead: the 20 GB last.ckpt save at walltime takes ~10-13 min on fir
# Lustre; a 5-min window gets SIGKILLed mid-write -> 0-byte last.ckpt -> crash
# loop on resume. 1500s gives the save time to finish.
#SBATCH --signal=B:SIGTERM@1500

handle_timeout() {
    echo "Timeout signal received. Stopping training gracefully..."
    if [ ! -z "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        wait \$TRAIN_PID
    fi
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
}
trap 'handle_timeout' SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Started: \$(date)  Node: \$(hostname)"
echo "GPUs: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================"

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd /scratch/dya78/TRELLIS.2-lightning
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

# Probe reads latents straight from persistent /scratch (no SSD staging).
DATA_ROOT=${SCRATCH_74K}

# Fail fast if the encode prereq hasn't produced both latent streams. Use
# find -print -quit (NOT ls *.npz): at 74k a glob expands ~72k filenames onto
# the command line and blows ARG_MAX -> false "missing" -> spurious abort.
for d in "emission_latents/emission_vae_twostream_74k" "pbr_latents/tex_enc_next_dc_f16c32_fp16_256" "shape_latents/shape_enc_next_dc_f16c32_fp16_256"; do
    if [ -z "\$(find \${DATA_ROOT}/\${d} -maxdepth 1 -name '*.npz' -print -quit 2>/dev/null)" ]; then
        echo "ERROR: missing latents at \${DATA_ROOT}/\${d} — run encode_twostream_dit_latents_74k.sh first."; exit 1
    fi
done

mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
set -o pipefail

srun python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
    --override dataset.args.data_root=\${DATA_ROOT} \
    --override dataset.args.parquet_file=\${DATA_ROOT}/df_SomgProc_final.parquet \
    --override dataset.args.split_json=\${DATA_ROOT}/data_splits_emissive_74k_stratified_voxonly.json \
    --override dataset.args.thumbnail_dir=\${DATA_ROOT}/thumbnails \
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo "Training crashed (exit code \$EXIT_CODE) at \$(date). Requeueing job..."
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
fi
echo "Training completed at \$(date)."
EOF

echo "Submitted: ${NAME}"
