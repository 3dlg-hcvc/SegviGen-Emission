#!/bin/bash
# Launch conditional emission PBR VAE (surgical finetune, overfit on 10 samples) on star2.
#
# Config:  configs/scvae/emission_pbr_cvae_10sample.yaml
# Trainer: EmissionPbrConditionalVaeTrainer
#          encoder: 3->32 (surgical input_layer from base_color cols)
#          decoder: 64->3 (cols [0:32] from pretrained from_latent, cols [32:64] zero-init)
#          stage A (0..1000): freeze interior, surgical layers only
#          stage B (1000..):  unfreeze, interior @ lr=3e-5, surgical @ lr=1e-4
#
# Env vars:
#   NUM_GPUS    number of GPUs (default 4 — matches the tested overfit setup).
#               Changing this requires bumping TRAIN_SIZE below to stay a
#               multiple of (NUM_GPUS * batch_size_per_gpu).
#   MAX_STEPS   optional override for trainer.args.max_steps (probe runs e.g. 100).
#   TAG         suffix on job name / output dir / log file (default = "g${NUM_GPUS}").

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-g${NUM_GPUS}}

CONFIG="configs/scvae/emission_pbr_cvae_10sample.yaml"
NAME="emission_pbr_cvae_10s_${TAG}"

GPU_GRES="gpu:${NUM_GPUS}"
if [[ "${NUM_GPUS}" -ge 2 ]]; then
    GPU_GRES="gpu:l40s:${NUM_GPUS}"
fi

echo "Submitting job: ${NAME}"
echo "  config: ${CONFIG}"
echo "  num_gpus: ${NUM_GPUS}"
if [[ -n "${MAX_STEPS}" ]]; then echo "  max_steps override: ${MAX_STEPS}"; fi

STEPS_OVERRIDE=""
if [[ -n "${MAX_STEPS}" ]]; then
    STEPS_OVERRIDE="--override trainer.args.max_steps=${MAX_STEPS}"
fi

ssh star2 "sbatch" << EOF
#!/bin/zsh
#SBATCH --gres=${GPU_GRES}
#SBATCH --mem=$((64*NUM_GPUS))G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-15
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  NUM_GPUS=${NUM_GPUS}  STEPS_OVERRIDE='${STEPS_OVERRIDE}'"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2
git pull origin lightgen

# Regenerate the oversampled split (data/ is gitignored).
# TRAIN_SIZE=640 = 64 (global batch: 4 GPU * BS=8) * 10 unique → 20 train batches/epoch,
# drop_last drops zero rows. Adjust if NUM_GPUS or batch_size_per_gpu changes.
TRAIN_SIZE=640 python data_toolkit/prep_10sample_as1k.py

# Reduce fragmentation — helps sparse voxel workloads with variable batch memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pwd: \$(pwd)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Background GPU memory monitor — samples every 15s so peak is visible after the run.
( while true; do
    echo "[MEMLOG] \$(date +%s) \$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' ';')"
    sleep 15
  done ) &
MON_PID=\$!

# Tee training output into log/ with a timestamped, config-named filename so the
# run is reproducible later (CLAUDE.md "Run logs" section).
mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG_FILE="log/\${TS}-${NAME}.log"
echo "Logging to: \${LOG_FILE}"

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --num_gpus ${NUM_GPUS} \
    --ckpt last \
    ${STEPS_OVERRIDE} \
    --override trainer.args.wandb_name=${NAME} 2>&1 | tee "\${LOG_FILE}"

TRAIN_EXIT=\${pipestatus[1]}
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}"
