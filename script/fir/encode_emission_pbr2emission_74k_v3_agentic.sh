#!/bin/bash
# Encode emission latents for all 72,421 newbake shapes with the _agentic pbr->emission VAE.
# 4 ranks on one 4xH100 node, then --aggregate_stats. Reads voxels from persistent /scratch
# (data/lightgen_74k_newbake); writes emission_latents/<LATENT_NAME>/ beside them.
# Usage: VAE_CKPT=<best>.ckpt bash script/fir/encode_emission_pbr2emission_74k_v3_agentic.sh
NAME="encode_emission_pbr2emission_74k_v3_agentic"
REPO="/scratch/dya78/TRELLIS.2-lightning"
RUN="${REPO}/outputs/emission_vae_pbr2emission_74k_v3_agentic"
VAE_CKPT="${RUN}/ckpts/${VAE_CKPT:?set VAE_CKPT=<best filename>.ckpt}"
VAE_CFG="${RUN}/config.yaml"
LATENT_NAME="emission_pbr2emission_74k_v3_agentic"
ACCOUNT="${ACCOUNT:-rrg-msavva_gpu}"
BRANCH="lightgen"
if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh -o BatchMode=yes fir sbatch); fi
"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=%N-%j.out
#SBATCH --mem=512G
#SBATCH --account=${ACCOUNT}
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
cd ${REPO} && git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
source ${REPO}/env/bin/activate
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0 HF_HUB_OFFLINE=1
[ -f "${VAE_CKPT}" ] || { echo "FATAL: ${VAE_CKPT} missing"; exit 1; }
D=${REPO}/data/lightgen_74k_newbake
NVOX=\$(ls "\${D}/emission_voxels_256" | grep -c "\.vxz\$")
echo "emission_voxels_256: \${NVOX} (expected 72421)"
[ "\${NVOX}" -ge 72421 ] || { echo "FATAL: emission_voxels_256 incomplete"; exit 1; }
set -o pipefail; mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"
{
  pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python -u data_toolkit/encode_emission_twostream_dataset.py \\
      --data_root \${D} --parquet \${D}/df_SomgProc_final.parquet \\
      --vae_ckpt ${VAE_CKPT} --vae_cfg ${VAE_CFG} \\
      --latent_name ${LATENT_NAME} --resolution 256 --rank \$r --world_size 4 &
    pids+=(\$!)
  done
  fail=0; for p in \${pids[@]}; do wait \$p || fail=1; done
  [ \$fail -eq 0 ] || { echo "ERROR: an encode rank failed"; exit 1; }
  python -u data_toolkit/encode_emission_twostream_dataset.py \\
      --data_root \${D} --parquet \${D}/df_SomgProc_final.parquet \\
      --vae_ckpt ${VAE_CKPT} --vae_cfg ${VAE_CFG} \\
      --latent_name ${LATENT_NAME} --resolution 256 --rank 0 --world_size 4 --aggregate_stats || exit 1
  NLAT=\$(ls \${D}/emission_latents/${LATENT_NAME} | grep -c '\.npz\$')
  echo "latents: \${NLAT} (expected 72421)"
  [ "\${NLAT}" -ge 72421 ] || { echo "FATAL: only \${NLAT} of 72421 latents present; encode incomplete"; exit 1; }
  cat \${D}/emission_latents/${LATENT_NAME}/normalization_stats.json
} 2>&1 | tee "\${LOG}"
exit \${PIPESTATUS[0]}
EOF
echo "Submitted: ${NAME}"
