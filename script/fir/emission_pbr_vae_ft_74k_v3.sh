#!/bin/bash
# albedo->emission VAE v3 -- PBR conditioning at the decoder. fir, 4x H100.
#
# THE CHANGE: decoder.latent_channels 32 -> 64, so it consumes
# cat([z_emission 32, z_pbr 32]). The pbr->emission decoder has had that input
# since its own v3; this closes the last conditioning asymmetry between the two
# chains. After it, they differ only in the emission-latent representation.
#
# Design + plan:
#   docs/superpowers/specs/2026-08-16-trellis2-albedo2emission-vae-pbr-condition-design.md
#   docs/superpowers/plans/2026-08-16-trellis2-albedo2emission-vae-pbr-condition.md
#
# ONE STAGE, one job -- unlike the v2_nonzero launcher this was copied from,
# which drove two (constant LR, then a resumed cosine tail). The v3 config is
# derived from configs/scvae/emission_vae_pbr2emission_74k_v3.yaml and therefore
# inherits ONE full-length CosineAnnealingLR(T_max=35800) plus that chain's
# split, step budget and batch sizes. So there is no CONFIG-swap second stage.
#
# Probe first, into a separate output dir so it cannot leave a last.ckpt where
# the real run would resume it:
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=1:00:00 bash script/fir/emission_pbr_vae_ft_74k_v3.sh
# Then the real run:
#   WALLTIME=<from probe> bash script/fir/emission_pbr_vae_ft_74k_v3.sh
#
# WATCH THE FIRST LOGGED train/l1. It must be ~0.005. A value near 1.774 means
# decoder.from_latent cold-started at xavier init instead of loading the baked
# file -- cancel immediately rather than spend 35,800 steps confirming it. That
# signature is what identified the shipped pbr->emission _v2 as an accidental
# from-scratch train.
#
# DATA. Reads emission_voxels_256/ AND pbr_latents/ directly from persistent
# /scratch. pbr_attrs is [] so pbr_voxels_256/ is never touched; the PBR
# information arrives as the pre-encoded latents the DiT stage already uses, so
# this run stages nothing new and spends no additional inodes. Per-shape
# coverage verified 2026-08-17: 72,421/72,421 on train+val+test.
#
# BATCH SIZE is overridden below, not edited into the config. The config carries
# venus-19's 32; fir's 80 GB H100s take 16, and 4 x 16 holds the same GLOBAL
# batch of 64 that the optimizer actually sees. val_batch_size_per_gpu stays at
# the config's 32 on purpose -- val/loss is reduced with sample-count weighting
# (basic.py:187-203) and save_top_k:1 selects on it, so the two chains must
# validate at the same grouping to stay commensurable. If the probe OOMs ONLY in
# validation, do not quietly drop it; that would make the head-to-head number
# incomparable.
#
# The config's repo-relative paths (data_root=data/lightgen_74k_newbake etc.)
# resolve correctly because cwd is the repo root -> no --override needed.

CONFIG="${CONFIG:-configs/scvae/emission_pbr_vae_ft_74k_v3.yaml}"
NAME="emission_pbr_vae_ft_74k_v3"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
WALLTIME="${WALLTIME:-12:00:00}"
WANDB_NAME="${WANDB_NAME:-vae_albedo->emission_74k_v3}"

# Hold the GLOBAL batch the optimizer sees at 64 -- the value both chains ran.
# The config carries venus-19's 32 per GPU; fir's 80 GB H100s take 16 (BS 32 on
# the newbake long tail OOM'd one outright). Refuse to submit if the product
# drifts, so a future GPU-count change cannot silently alter the optimizer.
NUM_GPUS="${NUM_GPUS:-4}"
BS="${BS:-16}"
GLOBAL_BATCH=$(( NUM_GPUS * BS ))
if [ "${GLOBAL_BATCH}" -ne 64 ]; then
    echo "FATAL: NUM_GPUS(${NUM_GPUS}) x BS(${BS}) = ${GLOBAL_BATCH}, expected 64."
    echo "       Both chains trained at global batch 64; changing it voids the comparison."
    exit 1
fi

echo "Submitting job: ${NAME}${OUT_SUFFIX}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Walltime: ${WALLTIME}   Max steps: ${MAX_STEPS:-<config default>}"
echo "GPUs: ${NUM_GPUS} x BS ${BS} = global batch ${GLOBAL_BATCH}"

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

# v3 conditions the decoder on the pre-encoded PBR latents, so they are now a
# hard dependency of the VAE stage too -- not just of the DiT.
LATDIR=/scratch/dya78/TRELLIS.2-lightning/data/lightgen_74k_newbake/pbr_latents/tex_enc_next_dc_f16c32_fp16_256
if [ ! -d "\${LATDIR}" ]; then
    echo "FATAL: \${LATDIR} does not exist. v3 conditions the decoder on the pre-encoded PBR latents."
    exit 1
fi
# Count WITHOUT a glob. Writing it as ls "\${LATDIR}"/[star].npz looks
# natural and is a trap:
# the shell expands 72,421 paths into the argv and ls dies with "Argument
# list too long", so the count comes back 0 and this guard fails a job that
# had nothing wrong with it. (Cost one 125-minute queue wait, 2026-08-17.)
NLAT=\$(ls "\${LATDIR}" | grep -c "\.npz\$")
echo "pbr latents present: \${NLAT} (expected 72421)"
if [ "\${NLAT}" -lt 72421 ]; then
    echo "FATAL: only \${NLAT} of 72421 PBR latents present; staging is incomplete."
    exit 1
fi

# The baked init is gitignored, so it cannot arrive via git pull -- regenerate it
# here. Deterministic, ~30 s, and the verifier is a hard gate: without it the
# trainer would silently drop the reshaped from_latent tensor and cold-start.
# TRELLIS2_SNAP: the bake script's default points at the workstation cache;
# fir's lives under /home.
export TRELLIS2_SNAP=/home/dya78/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts
python data_toolkit/bake_albedo2emission_v3_init.py --out data/ckpt_init/albedo2emission_v3 || { echo "FATAL: init bake failed"; exit 1; }
python data_toolkit/verify_albedo2emission_v3/verify_identity.py || { echo "FATAL: baked init failed verification"; exit 1; }
python data_toolkit/verify_albedo2emission_v3/verify_config_diff.py || { echo "FATAL: config drifted from the pbr->emission v3 recipe"; exit 1; }

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
    --num_gpus ${NUM_GPUS} \
    --override trainer.args.batch_size_per_gpu=${BS} \
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
