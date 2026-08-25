#!/bin/bash
# pbr->emission DiT v3 on the _agentic split (train 36,255) — fir, 4x H100 80 GB.
# Port of script/star2/emission_dit_pbr2emission_74k_v3_nonzero_nocopy_venus19.sh onto the fir
# DiT template script/fir/emission_dit_albedo2emission_pbrcond_74k_v2.sh. Deltas vs venus-19:
# srun + 4 tasks (Lightning's SLURM env requires ntasks == devices on fir), module loads + venv,
# and NO DinoV3 --override: that launcher pointed image_cond_model at a venus-19-local snapshot
# under /localscratch; on fir the model resolves from ~/.cache/huggingface with HF_HUB_OFFLINE=1,
# exactly as the fir albedo-pbrcond DiT already does.
#
# EQUAL OPTIMIZER STEPS, not equal epochs (spec "Equal optimizer steps (standing rule)"):
# max_steps 50,000 at global batch 32 is held from the baseline. 36,255 train shapes / 32 =
# 1,133 steps per epoch, so this run does ~44 epochs against the heuristic run's ~25. Seeing
# each shape more often is the accepted cost of a compute-matched comparison; the defence is
# val-based selection (save_top_k=1 on val/loss) plus reporting the val curve and selected step.
#
# --signal=B:SIGTERM@1500 (not the VAE's 900): the DiT's last.ckpt is ~20 GB and takes 10-13 min
# to write on Lustre. A killed mid-write save leaves a 0-byte last.ckpt that crash-loops every
# resume after it.
#
# Probe first, into a separate output dir (a non-empty run dir flips resume_mode=lightning and
# SKIPS the finetune load silently):
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=1:00:00 bash script/fir/emission_dit_pbr2emission_74k_v3_agentic.sh
# Full run:
#   bash script/fir/emission_dit_pbr2emission_74k_v3_agentic.sh
# Spec: lightgen docs/superpowers/specs/2026-08-22-agentic-filtered-baselines-retrain-design.md

CONFIG="configs/gen/emission_dit_pbr2emission_74k_v3_agentic.yaml"
NAME="emission_dit_pbr2emission_74k_v3_agentic"
TAG="fir"
REPO="/scratch/dya78/TRELLIS.2-lightning"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="${REPO}/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
NUM_GPUS="${NUM_GPUS:-4}"
BS="${BS:-8}"
WALLTIME="${WALLTIME:-24:00:00}"
ACCOUNT="${ACCOUNT:-rrg-msavva_gpu}"
WANDB_NAME="${WANDB_NAME:-dit_pbr->emission_74k_v3_agentic_${TAG}}"
SPLIT_MD5="159492b2b8d104ab63f3d13eeea394d0"
LATENT_NAME="emission_pbr2emission_74k_v3_agentic"

GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
ACTUAL=$((NUM_GPUS * BS))
if [ "${ACTUAL}" -ne "${GLOBAL_BATCH}" ]; then
    echo "REFUSING TO SUBMIT: NUM_GPUS(${NUM_GPUS}) x BS(${BS}) = ${ACTUAL}, expected ${GLOBAL_BATCH}." >&2
    exit 1
fi
echo "Submitting ${NAME}${OUT_SUFFIX}_${TAG}: ${NUM_GPUS} x BS ${BS} = global ${ACTUAL}; walltime ${WALLTIME}; max_steps ${MAX_STEPS:-<config 50000>}"

if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh -o BatchMode=yes fir sbatch); fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}_${TAG}
#SBATCH --gpus-per-node=h100:${NUM_GPUS}
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --mem=1024G
#SBATCH --account=${ACCOUNT}
#SBATCH --requeue
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

# fc10218 is a known sick node (an orphaned PID held 69 GB on GPU 0, 2026-08-20; an unrelated
# vLLM job held 71 GB there on 2026-08-24 and OOM'd our encode): refuse to train on a card that
# is already occupied before we start.
BUSY=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1 > 2048' | wc -l)
if [ "\${BUSY}" -gt 0 ]; then
    echo "FATAL: \${BUSY} GPU(s) on \$(hostname) already hold >2 GB before training; node is sick. Requeueing."
    scontrol requeue \${SLURM_JOB_ID}; sleep 60; exit 0
fi

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
echo "repo at: \$(git log --oneline -1)"
source ${REPO}/env/bin/activate

D=${REPO}/data/lightgen_74k_newbake
# Count WITHOUT a glob: ls "\${dir}"/*.npz expands 72,421 paths into argv and dies with
# "Argument list too long", so the count returns 0 and the gate fails a healthy job.
NEMIT=\$(ls "\${D}/emission_latents/${LATENT_NAME}" | grep -c "\.npz\$")
echo "emission_latents/${LATENT_NAME}: \${NEMIT} (expected 72421)"
[ "\${NEMIT}" -ge 72421 ] || { echo "FATAL: emission latents incomplete"; exit 1; }
[ -f "\${D}/emission_latents/${LATENT_NAME}/normalization_stats.json" ] || { echo "FATAL: normalization_stats.json missing"; exit 1; }
NPBR=\$(ls "\${D}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256" | grep -c "\.npz\$")
echo "pbr_latents: \${NPBR} (expected 72421)"
[ "\${NPBR}" -ge 72421 ] || { echo "FATAL: pbr latents incomplete"; exit 1; }
GOT=\$(md5sum \${D}/data_splits_emissive_74k_stratified_newbake_vae_agentic.json | cut -d' ' -f1)
[ "\${GOT}" = "${SPLIT_MD5}" ] || { echo "FATAL: split md5 \${GOT} != ${SPLIT_MD5}"; exit 1; }
echo "split md5 OK"

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
export WANDB_MODE=offline
set -o pipefail
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}${OUT_SUFFIX}_${TAG}.log"

srun python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus ${NUM_GPUS} \
    --override trainer.args.batch_size_per_gpu=${BS} \
    --override "trainer.args.wandb_name=${WANDB_NAME}" \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
TRAIN_EXIT=\$?
if [ \$TRAIN_EXIT -ne 0 ]; then
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then
        echo "Training crashed (exit \$TRAIN_EXIT) at \$(date). Requeueing..."; scontrol requeue \${SLURM_JOB_ID}; exit 0
    fi
    echo "Training crashed (exit \$TRAIN_EXIT); restart cap reached. NOT requeueing."; exit \$TRAIN_EXIT
fi
echo "Training completed at \$(date)."
exit \$TRAIN_EXIT
EOF
echo "Submitted: ${NAME}${OUT_SUFFIX}_${TAG}"
