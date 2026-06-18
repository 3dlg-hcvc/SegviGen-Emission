#!/bin/bash
# Launch emission PBR finetune (1k samples) on star2 (cs-venus-16)

CONFIG="configs/scvae/emission_pbr_finetune_1k.yaml"
NAME="emission_pbr_ft_1k"
DATA_ROOT="/localscratch/dya78/lightgen_1k"

echo "Submitting job: ${NAME} with config: ${CONFIG}"

ssh star2 "sbatch" << EOF
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-15
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}"

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
git pull origin lightgen
conda activate trellis2

echo "pwd: \$(pwd)"
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --ckpt last \
    --override dataset.args.data_root=${DATA_ROOT} \
    --override dataset.args.parquet_file=${DATA_ROOT}/df_SomgProc_emission_filtered.parquet \
    --override dataset.args.split_json=${DATA_ROOT}/data_splits_emission_filtered.json

echo "Job End: ${NAME}"
EOF

echo "Submitted: ${NAME}"