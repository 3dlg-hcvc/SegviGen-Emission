#!/bin/bash
# Launch conditional emission PBR VAE (surgical finetune, 1k samples) on star2.
#
# Config:  configs/scvae/emission_pbr_cvae_1k.yaml
# Trainer: EmissionPbrConditionalVaeTrainer
#          encoder: 3->32 (surgical input_layer from base_color cols)
#          decoder: 64->3 (cols [0:32] from pretrained from_latent, cols [32:64] zero-init)
#          stage A (0..1000): freeze interior, surgical layers only
#          stage B (1000..):  unfreeze, interior @ lr=3e-5, surgical @ lr=1e-4
#
# Apples-to-apples with finetune_pbr_1k.sh (channel-hijack baseline) for the
# same dataset / scale comparison.
#
# Env vars:
#   NUM_GPUS    number of GPUs (default 4).
#   TAG         suffix on job name / output dir / log file (default = "g${NUM_GPUS}").

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
TAG=${TAG:-g${NUM_GPUS}}

CONFIG="configs/scvae/emission_pbr_cvae_1k.yaml"
NAME="emission_pbr_cvae_1k_${TAG}"
DATA_ROOT="/localscratch/dya78/lightgen_1k"

GPU_GRES="gpu:${NUM_GPUS}"
if [[ "${NUM_GPUS}" -ge 2 ]]; then
    GPU_GRES="gpu:l40s:${NUM_GPUS}"
fi

echo "Submitting job: ${NAME}"
echo "  config: ${CONFIG}"
echo "  num_gpus: ${NUM_GPUS}"
echo "  data_root: ${DATA_ROOT}"

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
echo "Config: ${CONFIG}  NUM_GPUS=${NUM_GPUS}  DATA_ROOT=${DATA_ROOT}"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2
git pull origin lightgen

# Wandb auth isn't set up on per-node envs (~/.netrc missing). Offline mode logs
# to disk; 'wandb sync outputs/<run>/wandb/offline-run-*' pushes later.
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pwd: \$(pwd)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Background GPU memory monitor.
( while true; do
    echo "[MEMLOG] \$(date +%s) \$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' ';')"
    sleep 15
  done ) &
MON_PID=\$!

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG_FILE="log/\${TS}-${NAME}.log"
echo "Logging to: \${LOG_FILE}"

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --num_gpus ${NUM_GPUS} \
    --ckpt last \
    --override dataset.args.data_root=${DATA_ROOT} \
    --override dataset.args.parquet_file=${DATA_ROOT}/df_SomgProc_emission_filtered.parquet \
    --override dataset.args.split_json=${DATA_ROOT}/data_splits_emission_filtered.json \
    --override trainer.args.wandb_name=${NAME} 2>&1 | tee "\${LOG_FILE}"

TRAIN_EXIT=\${pipestatus[1]}
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}"
