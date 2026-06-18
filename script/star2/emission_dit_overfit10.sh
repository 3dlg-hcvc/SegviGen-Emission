#!/bin/bash
# Launch emission DiT overfit (10 samples) on star2 — fine-tuned from pretrained PBR DiT

CONFIG="configs/gen/emission_dit_10sample_overfit_pbrvae.yaml"
NAME="emission_dit_10sample_overfit_pbrvae"

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
conda activate trellis2

echo "pwd: \$(pwd)"
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python train.py \
    --config ${CONFIG} \
    --output_dir outputs/${NAME} \
    --ckpt none

echo "Job End: ${NAME}"
EOF

echo "Submitted: ${NAME}"
