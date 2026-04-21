#!/bin/bash
# Finetune-setup isolation test: run the 1k PBR finetune config unmodified on
# the 10-sample data (each of 10 samples repeated to 873 train rows, matching
# lightgen_1k's train split exactly).
#
# Mirrors finetune_pbr_1k_2gpu.sh byte-for-byte except: CONFIG, NAME, DATA_ROOT,
# parquet_file, split_json. Same bs / fp16 / EMA / grad-clip / 2xL40S.
#
# Env vars:
#   BS              batch_size_per_gpu (default 64). Effective global batch = BS * 2.
#   MAX_STEPS       optional override for trainer.args.max_steps (probe runs set e.g. 30).
#   TAG             suffix added to job name / output dir / log file (default = "bs${BS}").

set -euo pipefail

BS=${BS:-64}
WORKERS=${WORKERS:-}
MAX_STEPS=${MAX_STEPS:-}
TAG=${TAG:-bs${BS}}

CONFIG="configs/scvae/emission_pbr_finetune_10sample_as1k.yaml"
NAME="emission_pbr_ft_10s_as1k_${TAG}"
DATA_ROOT="data/lightgen_test"

echo "Submitting job: ${NAME}"
echo "  config: ${CONFIG}"
echo "  batch_size_per_gpu: ${BS}  (global = $((BS*2)))"
if [[ -n "${MAX_STEPS}" ]]; then echo "  max_steps override: ${MAX_STEPS}"; fi

STEPS_OVERRIDE=""
if [[ -n "${MAX_STEPS}" ]]; then
    STEPS_OVERRIDE="--override trainer.args.max_steps=${MAX_STEPS}"
fi
WORKERS_OVERRIDE=""
if [[ -n "${WORKERS}" ]]; then
    WORKERS_OVERRIDE="--override trainer.args.dataloader_num_workers=${WORKERS}"
fi

ssh star2 "sbatch" << EOF
#!/bin/bash
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=128G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-16
#SBATCH --output=/project/3dlg-hcvc/lightgen/probe/${NAME}.out
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
    --override dataset.args.parquet_file=${DATA_ROOT}/df_10samples.parquet \
    --override dataset.args.split_json=${DATA_ROOT}/data_splits_10sample_as1k.json \
    ${WORKERS_OVERRIDE} \
    ${STEPS_OVERRIDE}

TRAIN_EXIT=\$?
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
EOF

echo "Submitted: ${NAME}"
