#!/bin/bash
# Launch emission PBR finetune (1k samples) on star2 cs-venus-16 with 2x L40S.
#
# Env vars:
#   BS              batch_size_per_gpu (default 64). Effective global batch = BS * 2.
#   MAX_STEPS       optional override for trainer.args.max_steps (probe runs set e.g. 30).
#   TAG             suffix added to job name / output dir / log file (default = "bs${BS}").

set -euo pipefail

BS=${BS:-64}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-bs${BS}}

CONFIG="configs/scvae/emission_pbr_finetune_1k.yaml"
NAME="emission_pbr_ft_1k_${TAG}"
DATA_ROOT="/localscratch/dya78/lightgen_1k"

echo "Submitting job: ${NAME}"
echo "  config: ${CONFIG}"
echo "  batch_size_per_gpu: ${BS}  (global = $((BS*2)))"
if [[ -n "${MAX_STEPS}" ]]; then echo "  max_steps override: ${MAX_STEPS}"; fi

STEPS_OVERRIDE=""
if [[ -n "${MAX_STEPS}" ]]; then
    STEPS_OVERRIDE="--override trainer.args.max_steps=${MAX_STEPS}"
fi

ssh star2 "sbatch" << EOF
#!/bin/bash
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=128G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-16
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  BS=${BS}  STEPS_OVERRIDE='${STEPS_OVERRIDE}'"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
conda activate trellis2

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

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --num_gpus 2 \
    --ckpt last \
    --override trainer.args.batch_size_per_gpu=${BS} \
    --override dataset.args.data_root=${DATA_ROOT} \
    --override dataset.args.parquet_file=${DATA_ROOT}/df_SomgProc_emission_filtered.parquet \
    --override dataset.args.split_json=${DATA_ROOT}/data_splits_emission_filtered.json \
    ${STEPS_OVERRIDE}

TRAIN_EXIT=\$?
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
EOF

echo "Submitted: ${NAME}"
