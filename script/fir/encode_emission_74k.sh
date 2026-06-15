#!/bin/bash
# Phase 1: encode 74k emission latents with the finished channel-hijack VAE.
# 4 H100s on one node; each rank encodes a shard to local SSD (shared
# $SLURM_TMPDIR), rank 0 aggregates normalization stats, then we rsync the
# latents + stats back to persistent scratch.
#
# Usage:  bash script/fir/encode_emission_74k.sh

NAME="encode_emission_74k"
VAE_CKPT="/scratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_vae_ft_74k/ckpts/step0004485-0.0097.ckpt"
VAE_CFG="/scratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_vae_ft_74k/config.yaml"
LATENT_NAME="emission_pbr_vae_ft_74k"
SCRATCH_74K="/scratch/dya78/lightgen/lightgen_74k"
BRANCH="lightgen"

echo "Submitting job: ${NAME}"

if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --mem=1024G
#SBATCH --account=rrg-msavva

echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})  Node: \$(hostname)  Started: \$(date)"

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd /scratch/dya78/TRELLIS.2-lightning
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

# Stage 74k voxels to local SSD (extracts to \${SLURM_TMPDIR}/lightgen_74k)
cd \$SLURM_TMPDIR
bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_data.sh
DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k

cd /scratch/dya78/TRELLIS.2-lightning
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"

set -o pipefail
{
  # 4 ranks, one per GPU, writing to the shared local save_dir.
  pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python data_toolkit/encode_emission_pbrfinetune_dataset.py \\
        --data_root \${DATA_ROOT} \\
        --parquet   \${DATA_ROOT}/df_SomgProc_final.parquet \\
        --vae_ckpt  ${VAE_CKPT} \\
        --vae_cfg   ${VAE_CFG} \\
        --latent_name ${LATENT_NAME} \\
        --resolution 256 \\
        --rank \$r --world_size 4 &
    pids+=(\$!)
  done
  # Bare 'wait' returns 0 even if a rank crashed, which would let the aggregate
  # run over a partial shard set and rsync incomplete latents + wrong stats.
  # Wait each rank explicitly and abort before aggregating if any failed.
  rank_fail=0
  for p in \${pids[@]}; do
    wait \$p || rank_fail=1
  done
  if [ \$rank_fail -ne 0 ]; then
    echo "ERROR: one or more encode ranks failed; aborting before aggregate."
    exit 1
  fi
  # Aggregate stats once all shards exist (re-reads every .npz in save_dir).
  python data_toolkit/encode_emission_pbrfinetune_dataset.py \\
      --data_root \${DATA_ROOT} \\
      --parquet   \${DATA_ROOT}/df_SomgProc_final.parquet \\
      --vae_ckpt  ${VAE_CKPT} \\
      --vae_cfg   ${VAE_CFG} \\
      --latent_name ${LATENT_NAME} \\
      --resolution 256 \\
      --rank 0 --world_size 4 --aggregate_stats
} 2>&1 | tee "\${LOG}"
ENC_RC=\${PIPESTATUS[0]}
if [ \$ENC_RC -ne 0 ]; then
  echo "Encode/aggregate failed (rc=\$ENC_RC); NOT rsyncing partial latents."
  exit \$ENC_RC
fi

# Persist latents + stats back to scratch (Lustre); also copy the printed stats.
DST=${SCRATCH_74K}/emission_latents/${LATENT_NAME}
mkdir -p "\${DST}"
rsync -ah --info=progress2 \${DATA_ROOT}/emission_latents/${LATENT_NAME}/ "\${DST}/"

echo "Encode done at \$(date). Latents at \${DST}"
echo "Stats JSON: \${DST}/normalization_stats.json"
EOF

echo "Submitted: ${NAME}"
