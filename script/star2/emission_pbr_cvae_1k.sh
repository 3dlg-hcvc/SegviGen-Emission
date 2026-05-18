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
# Targets cs-venus-07 (4x A40, 48 GB) — currently the next 3dlg-hcvc-lab-node
# to free 4 GPUs (~21h ETA, vs cs-venus-15 at 3+ days). Data isn't staged
# locally on cs-venus-07; the sbatch body rsyncs it from the workstation
# (cs-venus-07's pubkey is already in cs-3dlg-23's authorized_keys).
#
# Apples-to-apples with finetune_pbr_1k.sh (channel-hijack 1k baseline).
#
# Env vars:
#   NUM_GPUS    number of GPUs (default 4).
#   NODELIST    SBATCH --nodelist (default cs-venus-07).
#   GPU_TYPE    GPU type for gres (default a40 — A40 has 48 GB which matches
#               cs-venus-07; switch to l40s for cs-venus-15/16/17).
#   TAG         suffix on job name / output dir / log file (default = "g${NUM_GPUS}").

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
NODELIST=${NODELIST:-cs-venus-07}
GPU_TYPE=${GPU_TYPE:-a40}
TAG=${TAG:-g${NUM_GPUS}}

CONFIG="configs/scvae/emission_pbr_cvae_1k.yaml"
NAME="emission_pbr_cvae_1k_${TAG}"
DATA_ROOT="/localscratch/dya78/lightgen_1k"
# Local source we'll pull from if DATA_ROOT is missing on the compute node.
LOCAL_FQDN="cs-3dlg-23.cmpt.sfu.ca"
LOCAL_DATA_PATH="/local-scratch/localhome/dya78/code/TRELLIS.2-lightning/data/lightgen_1k"

echo "Submitting job: ${NAME}"
echo "  config:   ${CONFIG}"
echo "  nodelist: ${NODELIST}"
echo "  gres:     gpu:${GPU_TYPE}:${NUM_GPUS}"
echo "  data:     ${DATA_ROOT} (pulled from ${LOCAL_FQDN} if missing)"

ssh star2 "sbatch" << EOF
#!/bin/zsh
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --mem=$((64*NUM_GPUS))G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=${NODELIST}
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  NUM_GPUS=${NUM_GPUS}  DATA_ROOT=${DATA_ROOT}"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2
git pull origin lightgen

# Stage the 1k dataset on this node's /localscratch if it's missing. cs-venus-07
# (and most other 3dlg nodes) don't have it; cs-venus-16 was the only one when
# this script was retargeted, and it's also locked for ~6 days. The compute
# node's ed25519 pubkey was added to the workstation's authorized_keys during
# the 10sample rsync setup, so this is passwordless.
if [ ! -d "${DATA_ROOT}" ]; then
    echo "[stage] ${DATA_ROOT} missing — pulling from ${LOCAL_FQDN}..."
    mkdir -p \$(dirname "${DATA_ROOT}")
    rsync -avh --info=progress2 \\
        "dya78@${LOCAL_FQDN}:${LOCAL_DATA_PATH}" \\
        \$(dirname "${DATA_ROOT}")/ 2>&1 | tail -5
    echo "[stage] Done. Sizes:"
    du -sh "${DATA_ROOT}"
fi

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

python train.py \\
    --config ${CONFIG} \\
    --output_dir outputs/${NAME} \\
    --num_gpus ${NUM_GPUS} \\
    --ckpt last \\
    --override dataset.args.data_root=${DATA_ROOT} \\
    --override dataset.args.parquet_file=${DATA_ROOT}/df_SomgProc_emission_filtered.parquet \\
    --override dataset.args.split_json=${DATA_ROOT}/data_splits_emission_filtered.json \\
    --override trainer.args.wandb_name=${NAME} 2>&1 | tee "\${LOG_FILE}"

TRAIN_EXIT=\${pipestatus[1]}
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}"
