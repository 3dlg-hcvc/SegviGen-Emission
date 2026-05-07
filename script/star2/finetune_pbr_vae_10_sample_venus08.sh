#!/bin/bash
# Launch emission PBR VAE channel-hijack finetune (overfit on 10 samples) on cs-venus-08 (4× A100 80GB).
#
# Same setup as finetune_pbr_vae_10_sample.sh; pinned to cs-venus-08 with --gres=gpu:a100.
#
# Config: configs/scvae/emission_pbr_vae_finetune_10sample.yaml
# Trainer: EmissionPbrFinetuneTrainer (channel-hijack of pretrained PBR VAE)
# Data:   repo-relative `data/lightgen_test/` (10 samples, oversampled split with
#         TRAIN_SIZE=640 = 4 GPUs * BS=16 * 10 unique — regenerated on launch
#         via data_toolkit/prep_10sample_as1k.py so 640 % global_batch == 0
#         and drop_last drops zero rows).
#
# Env vars:
#   NUM_GPUS    number of GPUs (default 4 — matches the tested overfit setup;
#               see CLAUDE.md "Tiny-dataset overfit setup (10 samples + 4 GPUs)").
#               Changing this requires bumping TRAIN_SIZE below to stay a
#               multiple of (NUM_GPUS * batch_size_per_gpu).
#   MAX_STEPS   optional override for trainer.args.max_steps (probe runs e.g. 50).
#   TAG         suffix on job name / output dir / log file (default = "v08_g${NUM_GPUS}").

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-v08_g${NUM_GPUS}}

CONFIG="configs/scvae/emission_pbr_vae_finetune_10sample.yaml"
NAME="emission_pbr_vae_ft_10s_${TAG}"

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
#SBATCH --time=5-15:00
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

# Regenerate the oversampled split (not tracked in git since data/ is gitignored).
# TRAIN_SIZE=640 = 64 (global batch) * 10 (unique samples) → 10 train batches/epoch,
# drop_last drops zero rows.
TRAIN_SIZE=640 python data_toolkit/prep_10sample_as1k.py

# Reduce fragmentation — helps sparse voxel workloads with variable batch memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pwd: \$(pwd)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Background GPU memory monitor — samples every 15s so we can see peak after the run.
( while true; do
    echo "[MEMLOG] \$(date +%s) \$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' ';')"
    sleep 15
  done ) &
MON_PID=\$!

# Tee training output into log/ with a timestamped, config-named filename so the
# run is reproducible later (see CLAUDE.md "Run Logs" section).
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
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
exit \$TRAIN_EXIT
EOF

echo "Submitted: ${NAME}"
