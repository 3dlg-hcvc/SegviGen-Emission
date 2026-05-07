#!/bin/bash
# Launch emission DiT 1k probe (20000 steps) on star2 — 4 × A100 80GB DDP on cs-venus-08.
#
# Config: configs/gen/emission_dit_1k_probe.yaml
# Data:   data/lightgen_1k/ (873 train / 111 val, encoded latents from the 1k VAE).
# Env:   SLURM_JOB_NAME=bash inside the body so Lightning uses native DDP spawn.

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-v08_g${NUM_GPUS}}

CONFIG="configs/gen/emission_dit_1k_probe.yaml"
NAME="emission_dit_1k_probe_${TAG}"

GPU_GRES="gpu:${NUM_GPUS}"
if [[ "${NUM_GPUS}" -ge 2 ]]; then
    GPU_GRES="gpu:a100:${NUM_GPUS}"
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
#SBATCH --time=5-00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-08
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  NUM_GPUS=${NUM_GPUS}  STEPS_OVERRIDE='${STEPS_OVERRIDE}'"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2
git pull origin lightgen

# Lightning: disable SLURM-coordinated launcher so devices=4 spawns its own processes.
export SLURM_JOB_NAME=bash

# Reduce fragmentation for the sparse voxel workload.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pwd: \$(pwd)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG_FILE="log/\${TS}-${NAME}.log"
echo "Logging to: \${LOG_FILE}"

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --num_gpus ${NUM_GPUS} \
    --ckpt last \
    --override trainer.args.batch_size_per_gpu=24 \
    --override trainer.args.batch_split=1 \
    ${STEPS_OVERRIDE} 2>&1 | tee "\${LOG_FILE}"

TRAIN_EXIT=\${pipestatus[1]}
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}"
