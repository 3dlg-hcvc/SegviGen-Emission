#!/bin/bash
# Launch emission PBR VAE finetune (74k samples, channel-hijack) on fir cluster.
# Full H100 node: 4 GPU + 48 CPU + 1152 GB RAM.
#
# Usage:
#   bash script/fir/finetune_pbr_vae_74k.sh

CONFIG="configs/scvae/emission_pbr_vae_finetune_74k.yaml"
NAME="emission_pbr_vae_ft_74k"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/${NAME}"
BRANCH="lightgen"

echo "Submitting job: ${NAME}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

ssh fir "sbatch" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:4
# 3-hour window: matches the maintenance-friendly slot that e74k_van fits into.
# Training won't finish in one window (20k steps * ~1s = ~5.5h), so the
# --requeue + SIGTERM handler below checkpoints at the 2h55m mark and the
# job re-enters the queue automatically. Bump back to 24h once the
# maintenance reservation clears.
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
# Fir H100 nodes have RealMemory=1125G (verified via scontrol show node).
# 4 * 256G = 1024G stays under physical max AND under the 1:12:288 RGU ratio
# (mirrors finetune_pbr_1k.sh's 256G/GPU pattern that we know works on fir).
#SBATCH --mem=1024G
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
echo "GPUs: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================"

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

# Pull latest code from origin/${BRANCH} (per CLAUDE.md: never edit on cluster)
cd /scratch/dya78/TRELLIS.2-lightning
git fetch origin ${BRANCH}
git checkout ${BRANCH}
git pull --ff-only origin ${BRANCH}

# Activate trellis2 env
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

# Stage 74k voxels to local SSD
cd \$SLURM_TMPDIR
bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_data.sh

DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k

cd /scratch/dya78/TRELLIS.2-lightning

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG="log/\${TS}-${NAME}.log"

echo ""
echo "Data root: \${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log: \${LOG}"
echo ""

# Reduce VRAM fragmentation (matters at BS=48 on H100; see
# memory project_fir_pbr_vae_bs_sweep.md).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training in background so SIGTERM trap can fire
python train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    --override dataset.args.data_root=\${DATA_ROOT} \
    --override dataset.args.parquet_file=\${DATA_ROOT}/df_SomgProc_final.parquet \
    --override dataset.args.split_json=\${DATA_ROOT}/data_splits_emissive_74k_stratified_voxonly.json \
    2>&1 | tee "\${LOG}" &
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
