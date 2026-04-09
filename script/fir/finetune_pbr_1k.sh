#!/bin/bash
# Launch emission PBR finetune (1k samples) on fir cluster
#
# Usage:
#   bash script/fir/finetune_pbr_1k.sh

CONFIG="configs/scvae/emission_pbr_finetune_1k.yaml"
NAME="emission_pbr_ft_1k"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_finetune_1k"
REPO_URL="git@github.com:3dlg-hcvc/TRELLIS.2-lightning.git"
BRANCH="lightgen"

echo "Submitting job: ${NAME}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

ssh fir "sbatch" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --mem=256G
#SBATCH --account=rrg-msavva
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

handle_timeout() {
    echo "Timeout signal received (5 min remaining). Stopping training gracefully..."
    if [ ! -z "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        echo "Waiting for training to save checkpoint..."
        wait \$TRAIN_PID
        echo "Training process exited. Requeueing job..."
    fi
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
}

trap 'handle_timeout' SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Started: \$(date)"
echo "Node: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================"

module load StdEnv/2023 cuda/12.6
module load gcc
module load opencv
module load arrow/22.0.0
module load sparsehash

# Activate trellis2 conda env
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

# Clone repo to fast local SSD
cd $SLURM_TMPDIR
git clone ${REPO_URL} -b ${BRANCH} --recursive
cd TRELLIS.2-lightning/

# Unpack voxel data to local SSD
bash script/fir/unpack_1k_data.sh

DATA_ROOT=${SLURM_TMPDIR}/TRELLIS.2-lightning/lightgen_1k

echo ""
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Run training in background so SIGTERM trap can fire
python train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --override dataset.args.data_root=${DATA_ROOT} \
    --override dataset.args.parquet_file=${DATA_ROOT}/df_SomgProc_emission_filtered.parquet \
    --override dataset.args.split_json=${DATA_ROOT}/data_splits_emission_filtered.json &
TRAIN_PID=\$!
wait \$TRAIN_PID
EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo "Training crashed (exit code \$EXIT_CODE) at \$(date). Requeueing job..."
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
fi

echo "Training completed at \$(date)."
EOF

echo "Submitted: ${NAME}"
