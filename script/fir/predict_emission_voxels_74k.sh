#!/bin/bash
# Emission-voxel inference for the full 74k test split (or any N/split/guidance/steps).
# Runs on a fir MIG slice (def-msavva_gpu) — lightweight eval, no full H100 needed.
# Writes {sha}.npz files to outputs/emission_dit_74k/eval/pred_voxels/ on scratch.
#
# Usage:  [DIT_CKPT=...] [SPLIT=test] [N=all] [GUIDANCE=3.0] [STEPS=50] \
#           bash script/fir/predict_emission_voxels_74k.sh
NAME="predict_emission_voxels_74k"
REPO="/scratch/dya78/TRELLIS.2-lightning"
DIT_CKPT="${DIT_CKPT:-${REPO}/outputs/emission_dit_74k/ckpts/step0018000-0.3872.ckpt}"
VAE_CKPT="${VAE_CKPT:-${REPO}/outputs/emission_pbr_vae_ft_74k/ckpts/step0016816-0.0095.ckpt}"
BRANCH="lightgen"
echo "Submitting ${NAME}  (DiT=$(basename ${DIT_CKPT}))"
if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi
"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH -J ${NAME}
#SBATCH --account=def-msavva_gpu
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --time=${WALLTIME:-4:00:00}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%N-%j.out
echo "Node: \$(hostname)  Started: \$(date)"
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH} || { echo "GIT_SYNC_FAILED — aborting"; exit 1; }
source ${REPO}/env/bin/activate
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
python -u data_toolkit/predict_emission_voxels.py \\
    --dit_ckpt ${DIT_CKPT} \\
    --vae_ckpt ${VAE_CKPT} \\
    --data_root /scratch/dya78/lightgen/lightgen_74k \\
    --split ${SPLIT:-test} --n ${N:-all} --guidance ${GUIDANCE:-3.0} --steps ${STEPS:-50} \\
    --out_dir ${REPO}/outputs/emission_dit_74k/eval
echo "PRED_VOX_DONE \$(date) -> ${REPO}/outputs/emission_dit_74k/eval/pred_voxels"
EOF
echo "Submitted: ${NAME}"
