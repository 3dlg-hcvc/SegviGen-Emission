#!/bin/bash
# Launch the albedo-slot + PBR-cond parity DiT on fir (4 H100), newbake 72,421
# cohort. Trains the albedo2emission chain's albedo-slot latents WITH the PBR
# latent channel-concatenated (skip_pbr_cond=false, in=96) — the 2026-08-09
# conditioning-parity successor that inherits the "albedo→emission" display
# name. Siblings: emission_dit_albedo2emission_74k_v2.sh (in=64, now the
# "without pbr condition" ablation) and emission_dit_pbr2emission_74k_v2.sh
# (in=96, two-stream latents). Same batch_size_per_gpu for a controlled
# comparison.
#
# Data lives on persistent /scratch at data/lightgen_74k_newbake (the config's
# default repo-relative data_root), so training reads directly from Lustre — no
# $SLURM_TMPDIR staging, so a requeue resumes immediately instead of re-staging.
#
# Usage:
#   MAX_STEPS=100 WALLTIME=1:00:00 bash script/fir/emission_dit_albedo2emission_pbrcond_74k_v2.sh  # probe
#   bash script/fir/emission_dit_albedo2emission_pbrcond_74k_v2.sh                                  # full run

CONFIG="configs/gen/emission_dit_albedo2emission_pbrcond_74k_v2.yaml"
NAME="emission_dit_albedo2emission_pbrcond_74k_v2"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/${NAME}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
WALLTIME="${WALLTIME:-24:00:00}"

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
#SBATCH --account=rrg-msavva_gpu
#SBATCH --requeue
# 25-min lead: the ~20 GB last.ckpt save at walltime takes ~10-13 min on fir
# Lustre; a 5-min window gets SIGKILLed mid-write -> 0-byte last.ckpt -> crash
# loop on resume. 1500s lets the save finish.
#SBATCH --signal=B:SIGTERM@1500

handle_timeout() {
    echo "SIGTERM received. Stopping training gracefully..."
    if [ ! -z "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        wait \$TRAIN_PID
    fi
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
}
trap 'handle_timeout' SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})  restart_count=\${SLURM_RESTART_COUNT:-0}"
echo "Started: \$(date)  Node: \$(hostname)"
echo "GPUs: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================"

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd /scratch/dya78/TRELLIS.2-lightning
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flex_gemm's shared autotune cache on NFS home can be corrupted by concurrent
# ranks (JSONDecodeError on import). Disable read+write -> each rank autotunes
# in-memory (one-time cost, no race).
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
set -o pipefail

mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"

# Read newbake latents directly from persistent /scratch. The config's default
# repo-relative paths (data_root=data/lightgen_74k_newbake, etc.) resolve here
# since cwd is the repo root -> no path overrides needed.
srun python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
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
