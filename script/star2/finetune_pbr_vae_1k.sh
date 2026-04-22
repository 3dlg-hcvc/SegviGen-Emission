#!/bin/bash
# Launch emission PBR VAE channel-hijack finetune (full 1k subset) on star2 cs-venus-18.
#
# Config: configs/scvae/emission_pbr_vae_finetune_1k.yaml
# Trainer: EmissionPbrFinetuneTrainer (channel-hijack of pretrained PBR VAE)
# Data:   repo-relative `data/lightgen_1k/` (~1099 unique samples, real
#         emission-filtered split — no oversampling needed at 4 GPUs * BS=8).
#
# cs-venus-18 is a theia-lab-owned L40S×4 node, but our 3dlg-hcvc-lab account can
# reach it via the open `cs-gpu-research` partition (verified with sbatch --test-only).
# Do NOT submit via theia-lab-* (not our lab). See CLAUDE.md cluster-access notes.
#
# Env vars:
#   NUM_GPUS    number of GPUs (default 4 — matches the 10-sample sibling).
#   MAX_STEPS   optional override for trainer.args.max_steps (probe runs e.g. 50).
#   TAG         suffix on job name / output dir / log file (default = "g${NUM_GPUS}").

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-g${NUM_GPUS}}

CONFIG="configs/scvae/emission_pbr_vae_finetune_1k.yaml"
NAME="emission_pbr_vae_ft_1k_${TAG}"

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
# default partition default-debug caps at 8h (auto-picked when --partition is
# omitted); cs-gpu-research is the only 7-day partition cs-venus-18 belongs to
# that we are allowed to use (NOT theia-lab-* — not our lab). --account is
# omitted: dya78's sole association is 3dlg-hcvc-lab, so slurm picks it.
#SBATCH --partition=cs-gpu-research
#SBATCH --gres=${GPU_GRES}
#SBATCH --mem=$((64*NUM_GPUS))G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-18
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  NUM_GPUS=${NUM_GPUS}  STEPS_OVERRIDE='${STEPS_OVERRIDE}'"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2
git pull origin lightgen

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
