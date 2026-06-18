#!/bin/bash
# Quick prediction viz from the (partially-trained) 74k emission DiT best-val ckpt.
# Short single-GPU job -> fits a shared node, backfills without competing with the
# whole-node training. Writes a PNG to the repo root on scratch; pull it after.
#
# Usage:  [DIT_CKPT=...] bash script/fir/predict_emission_dit_74k.sh
NAME="predict_emission_dit_74k"
REPO="/scratch/dya78/TRELLIS.2-lightning"
DIT_CKPT="${DIT_CKPT:-${REPO}/outputs/emission_dit_74k/ckpts/step0018000-0.3872.ckpt}"
VAE_CKPT="${REPO}/outputs/emission_pbr_vae_ft_74k/ckpts/step0016816-0.0095.ckpt"
OUT="${REPO}/emission_dit_74k_pred.png"
BRANCH="lightgen"
echo "Submitting ${NAME}  (DiT=$(basename ${DIT_CKPT}))"
if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi
"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-msavva
#SBATCH --output=%N-%j.out
echo "Node: \$(hostname)  Started: \$(date)"
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
cd ${REPO}
git fetch origin ${BRANCH} >/dev/null 2>&1 && git checkout ${BRANCH} >/dev/null 2>&1 && git pull --ff-only origin ${BRANCH} 2>&1 | tail -1
source ${REPO}/env/bin/activate
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
python -u data_toolkit/predict_emission_dit_74k.py \\
    --dit_ckpt ${DIT_CKPT} \\
    --vae_ckpt ${VAE_CKPT} \\
    --data_root /scratch/dya78/lightgen/lightgen_74k \\
    --split val --n 6 --guidance 3.0 --steps 50 --out ${OUT}
echo "PRED_DONE \$(date) -> ${OUT}"
EOF
echo "Submitted: ${NAME}"
