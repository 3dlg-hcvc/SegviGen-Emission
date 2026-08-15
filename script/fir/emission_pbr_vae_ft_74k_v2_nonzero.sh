#!/bin/bash
# albedo->emission VAE on the EMISSIVE-ONLY training set, fir 4x H100.
#
# The experiment: drop the 6,183 shapes whose baked atlas emission is uniformly
# zero from train (71,646 -> 65,463); val (387) and test (388) pass through
# unchanged, so this run and its baseline select checkpoints under an identical
# criterion and differ only in what they trained on.
#
# TWO STAGES, one launcher. CONFIG is overridable because the baseline's LR curve
# was not one run: constant 1e-4 to 25,800 steps, then a fresh cosine tail to
# 35,800.
#   stage 1:  bash script/fir/emission_pbr_vae_ft_74k_v2_nonzero.sh
#   stage 2:  CONFIG=configs/scvae/emission_pbr_vae_ft_74k_v2_nonzero_cosine.yaml \
#             WALLTIME=8:00:00 bash script/fir/emission_pbr_vae_ft_74k_v2_nonzero.sh
# Stage 2 resumes stage 1's last.ckpt out of the SAME output dir via --ckpt last.
#
# Probe first, into a separate output dir so it cannot leave a last.ckpt where the
# real run would resume it:
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=1:00:00 bash script/fir/emission_pbr_vae_ft_74k_v2_nonzero.sh
#
# DATA. Reads emission_voxels_256/ directly from persistent /scratch, the same
# arrangement the pbrcond DiT uses -- NOT staged into $SLURM_TMPDIR like the older
# script/fir/finetune_pbr_vae_74k.sh. A requeue therefore resumes immediately
# instead of re-extracting 4.7 GB first. This is affordable because the config
# sets pbr_attrs: [], so only the emission voxels have to live on fir: 72,421
# files rather than 144,842, which matters because fir's /scratch inode quota,
# not its disk space, is the binding constraint.
#
# The config's repo-relative paths (data_root=data/lightgen_74k_newbake etc.)
# resolve correctly because cwd is the repo root -> no --override needed.

CONFIG="${CONFIG:-configs/scvae/emission_pbr_vae_ft_74k_v2_nonzero.yaml}"
NAME="emission_pbr_vae_ft_74k_v2_nonzero"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
WALLTIME="${WALLTIME:-12:00:00}"
WANDB_NAME="${WANDB_NAME:-vae_albedo->emission_74k_v2_nonzero}"

echo "Submitting job: ${NAME}${OUT_SUFFIX}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Walltime: ${WALLTIME}   Max steps: ${MAX_STEPS:-<config default>}"

# Submit directly when already on a Slurm submit host (the fir login node — our
# tmux-session workflow); fall back to `ssh fir sbatch` only when launched from
# the local workstation. Running `ssh fir ...` ON fir fails with "Could not
# resolve hostname fir" and silently submits nothing while printing a bogus
# "Submitted" line.
if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
# 4 tasks (one per GPU) is the canonical Lightning+Slurm DDP pattern on fir:
# Lightning's SLURM cluster env asserts ntasks-per-node == devices and crashes at
# startup if they differ. srun spawns the 4 ranks; Lightning picks them up via
# SLURM_PROCID. (This is the opposite of the star2 convention, which uses ONE
# task plus Lightning native spawn — do not copy that pattern here.)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
# Fir H100 nodes have RealMemory=1125G; 4 x 256G stays under the physical max and
# under the 1:12:288 RGU ratio. 1152G is rejected outright.
#SBATCH --mem=1024G
#SBATCH --account=rrg-msavva_gpu
#SBATCH --requeue
# The VAE checkpoint is far smaller than the DiT's 20 GB, but this is still
# Lustre. 900 s of SIGTERM lead lets the save finish; a killed mid-write save
# leaves a 0-byte last.ckpt that crash-loops every resume after it.
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

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd /scratch/dya78/TRELLIS.2-lightning
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
echo "repo at: \$(git log --oneline -1)"
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

# Fail fast and loudly if the voxels are not where the config expects them.
# Without this the run dies one sample into the first epoch with a bare
# FileNotFoundError after the queue wait is already spent.
VOXDIR=/scratch/dya78/TRELLIS.2-lightning/data/lightgen_74k_newbake/emission_voxels_256
if [ ! -d "\${VOXDIR}" ]; then
    echo "FATAL: \${VOXDIR} does not exist. The emission voxels have not been staged to fir."
    exit 1
fi
NVOX=\$(ls "\${VOXDIR}" | wc -l)
echo "emission voxels present: \${NVOX} (expected 72421)"
if [ "\${NVOX}" -lt 72421 ]; then
    echo "FATAL: only \${NVOX} of 72421 voxel files present; staging is incomplete."
    exit 1
fi

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flex_gemm's shared autotune cache on NFS home can be corrupted by concurrent
# ranks (JSONDecodeError on import). Disable read+write -> each rank autotunes
# in-memory (one-time cost, no race).
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
set -o pipefail

mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}${OUT_SUFFIX}.log"

# > >(tee) process substitution, NOT a tee pipe — \$! must be train.py's PID (the
# TERM trap signals it and \$? must be its exit); a backgrounded pipeline would
# hand both to tee instead.
srun python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    --override "trainer.args.wandb_name=${WANDB_NAME}" \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
TRAIN_EXIT=\$?

if [ \$TRAIN_EXIT -ne 0 ]; then
    # Cap requeues: transient faults deserve a retry, deterministic errors must
    # not crash-loop (an uncapped requeue burned 70 restarts on a config error).
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then
        echo "Training crashed (exit code \$TRAIN_EXIT) at \$(date). Requeueing job..."
        scontrol requeue \${SLURM_JOB_ID}
        exit 0
    fi
    echo "Training crashed (exit code \$TRAIN_EXIT) at \$(date); restart cap reached (\${SLURM_RESTART_COUNT}). NOT requeueing."
    exit \$TRAIN_EXIT
fi
echo "Training completed at \$(date)."
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}${OUT_SUFFIX}"
