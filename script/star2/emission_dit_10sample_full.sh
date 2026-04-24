#!/bin/bash
# Launch emission DiT 10-sample overfit (6000 steps) on star2 — 4 × L40S DDP.
#
# Config: configs/gen/emission_dit_10sample_overfit_pbrvae_as1k.yaml
# Data:   data/lightgen_test/ (10 samples, oversampled train split TRAIN_SIZE=640
#         regenerated in the job body).
# Env:   SLURM_JOB_NAME=bash inside the body so Lightning uses native DDP spawn
#         instead of SLURM-coordinated world_size=SLURM_NTASKS (which would be 1).

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-g${NUM_GPUS}}

CONFIG="configs/gen/emission_dit_10sample_overfit_pbrvae_as1k.yaml"
NAME="emission_dit_10s_overfit_as1k_${TAG}"

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
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-15
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  NUM_GPUS=${NUM_GPUS}  STEPS_OVERRIDE='${STEPS_OVERRIDE}'"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2
git pull origin lightgen

# Regenerate oversampled split (gitignored); TRAIN_SIZE=640 fits 4 * BS=8 = 32 exactly (20 batches/epoch, drop_last drops 0).
TRAIN_SIZE=640 python data_toolkit/prep_10sample_as1k.py

# Lightning: disable SLURM-coordinated launcher so devices=4 spawns its own processes.
export SLURM_JOB_NAME=bash

# Reduce fragmentation for the sparse voxel workload.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pwd: \$(pwd)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Tee to log/<TS>-<NAME>.log per repo convention.
mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG_FILE="log/\${TS}-${NAME}.log"
echo "Logging to: \${LOG_FILE}"

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --num_gpus ${NUM_GPUS} \
    --ckpt last \
    ${STEPS_OVERRIDE} 2>&1 | tee "\${LOG_FILE}"

TRAIN_EXIT=\${pipestatus[1]}
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}"
