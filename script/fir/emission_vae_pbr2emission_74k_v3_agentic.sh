#!/bin/bash
# pbr->emission VAE v3 on the _agentic split (train 36,255) — fir, 4x H100 80 GB.
# Port of script/star2/emission_vae_pbr2emission_74k_v3_nonzero_nocopy_venus19.sh onto the fir
# sbatch template of script/fir/emission_pbr_vae_ft_74k_v3.sh. Deltas vs the venus-19 launcher:
# srun + 4 tasks (Lightning's SLURM env requires ntasks == devices on fir), module loads + venv,
# BS 16 per GPU (80 GB cards; 4x16 = global 64 exactly as venus-19's 4x16), data gates for the
# PBR voxels (staged 2026-08-23, fir held none before) and the split json md5, the init bake
# from fir's HF snapshot, and 24 h walltime segments with --requeue (the SIGTERM trap requeues;
# --ckpt last resumes; a 26 h run needs two segments).
#
# ATTENTION BACKEND: the venus-19 launcher pins ATTN_BACKEND=sdpa and SPARSE_ATTN_BACKEND=xformers
# for Blackwell; fir's env has flash-attn and the fir VAE template this is ported from sets
# neither, so both stay unset here. Numerics differ by site, semantics do not.
#
# HF SNAPSHOT: HF_HOME is deliberately NOT set. bake_pbr2emission_v3_init.py's resolve_hub()
# checks $HF_HOME/hub first and ALWAYS appends ~/.cache/huggingface/hub after it — which is
# where fir's models--microsoft--TRELLIS.2-4B snapshot lives. So leaving HF_HOME unset simply
# resolves straight to fir's cache; the venus-19 value would be a dead root on this site.
#
# Probe first, into a separate output dir (a non-empty dir flips resume_mode and SKIPS the
# finetune load):
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=1:00:00 bash script/fir/emission_vae_pbr2emission_74k_v3_agentic.sh
# Full run:
#   bash script/fir/emission_vae_pbr2emission_74k_v3_agentic.sh
# Spec: lightgen docs/superpowers/specs/2026-08-22-agentic-filtered-baselines-retrain-design.md

CONFIG="configs/scvae/emission_vae_pbr2emission_74k_v3_agentic.yaml"
NAME="emission_vae_pbr2emission_74k_v3_agentic"
TAG="fir"
REPO="/scratch/dya78/TRELLIS.2-lightning"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="${REPO}/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
NUM_GPUS="${NUM_GPUS:-4}"
BS="${BS:-16}"
WALLTIME="${WALLTIME:-24:00:00}"
ACCOUNT="${ACCOUNT:-rrg-msavva_gpu}"
WANDB_NAME="${WANDB_NAME:-vae_pbr->emission_74k_v3_agentic_${TAG}}"
SPLIT_MD5="159492b2b8d104ab63f3d13eeea394d0"

GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
ACTUAL=$((NUM_GPUS * BS))
if [ "${ACTUAL}" -ne "${GLOBAL_BATCH}" ]; then
    echo "REFUSING TO SUBMIT: NUM_GPUS(${NUM_GPUS}) x BS(${BS}) = ${ACTUAL}, expected ${GLOBAL_BATCH}." >&2
    exit 1
fi
echo "Submitting ${NAME}${OUT_SUFFIX}_${TAG}: ${NUM_GPUS} x BS ${BS} = global ${ACTUAL}; walltime ${WALLTIME}; max_steps ${MAX_STEPS:-<config 35800>}"

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
#SBATCH --signal=B:SIGTERM@900

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

# fc10218 is a known sick node (an orphaned PID held 69 GB on GPU 0, 2026-08-20): refuse to
# train on a card that is already occupied before we start.
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
for sub in emission_voxels_256 pbr_voxels_256; do
    [ -d "\${D}/\${sub}" ] || { echo "FATAL: \${D}/\${sub} missing — not staged"; exit 1; }
    N=\$(ls "\${D}/\${sub}" | grep -c "\.vxz\$")
    echo "\${sub}: \${N} (expected 72421)"
    [ "\${N}" -ge 72421 ] || { echo "FATAL: \${sub} incomplete"; exit 1; }
done
GOT=\$(md5sum \${D}/data_splits_emissive_74k_stratified_newbake_vae_agentic.json | cut -d' ' -f1)
[ "\${GOT}" = "${SPLIT_MD5}" ] || { echo "FATAL: split md5 \${GOT} != ${SPLIT_MD5}"; exit 1; }
echo "split md5 OK"

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
export WANDB_MODE=offline
set -o pipefail
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}${OUT_SUFFIX}_${TAG}.log"

# Bake the derived init here (gitignored, deterministic, seconds). The verifier is a hard
# gate: without it the trainer would silently drop the 5 derived tensors and cold-start.
python -u data_toolkit/bake_pbr2emission_v3_init.py \
    --out data/ckpt_init/pbr2emission_v3 \
    --verify-config ${CONFIG} 2>&1 | tee -a "\${LOG}"
[ \${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: init bake failed"; exit 1; }

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
