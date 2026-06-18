#!/bin/bash
# Encode the two latent streams the two-stream emission DiT needs on 74k:
#   1. emission latents from the TWO-STREAM VAE (step-10000 last-v1.ckpt),
#      via encode_emission_twostream_dataset.py  -> emission_latents/emission_vae_twostream_74k/
#   2. PBR latents from the production tex encoder (the DiT's 96-ch input
#      conditions on these; the baseline channel-hijack DiT used skip_pbr_cond
#      and never needed them), via encode_pbr_latent.py
#      -> pbr_latents/tex_enc_next_dc_f16c32_fp16_256/
#
# Both run on one 4-H100 node: 4 ranks (one per GPU), shared $SLURM_TMPDIR
# save_dir, then aggregate normalization stats and rsync latents+stats back to
# persistent scratch. Voxels are staged ONCE (unpack_74k_data.sh extracts both
# emission_voxels_256 and pbr_voxels_256).
#
# Usage:  bash script/fir/encode_twostream_dit_latents_74k.sh

NAME="encode_twostream_dit_latents_74k"
VAE_CKPT="/scratch/dya78/TRELLIS.2-lightning/outputs/emission_vae_twostream_74k_fir/ckpts/last-v1.ckpt"
VAE_CFG="/scratch/dya78/TRELLIS.2-lightning/outputs/emission_vae_twostream_74k_fir/config.yaml"
EMIT_LATENT_NAME="emission_vae_twostream_74k"
PBR_LATENT_NAME="tex_enc_next_dc_f16c32_fp16_256"   # what encode_pbr_latent.py derives from enc_pretrained+res
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
#SBATCH --time=08:00:00
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

# Stage 74k voxels to local SSD (extracts emission_voxels_256 + pbr_voxels_256 to \${SLURM_TMPDIR}/lightgen_74k)
cd \$SLURM_TMPDIR
bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_data.sh
DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k

cd /scratch/dya78/TRELLIS.2-lightning
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"

# flex_gemm autotune cache on NFS home corrupts under concurrent ranks -> disable.
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
set -o pipefail

{
  echo "=================================================================="
  echo "STREAM 1/2: two-stream emission latents (${EMIT_LATENT_NAME})"
  echo "=================================================================="
  pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python -u data_toolkit/encode_emission_twostream_dataset.py \\
        --data_root \${DATA_ROOT} \\
        --parquet   \${DATA_ROOT}/df_SomgProc_final.parquet \\
        --vae_ckpt  ${VAE_CKPT} \\
        --vae_cfg   ${VAE_CFG} \\
        --latent_name ${EMIT_LATENT_NAME} \\
        --resolution 256 \\
        --rank \$r --world_size 4 &
    pids+=(\$!)
  done
  rank_fail=0
  for p in \${pids[@]}; do wait \$p || rank_fail=1; done
  if [ \$rank_fail -ne 0 ]; then echo "ERROR: emission encode rank failed; abort."; exit 1; fi
  python -u data_toolkit/encode_emission_twostream_dataset.py \\
      --data_root \${DATA_ROOT} --parquet \${DATA_ROOT}/df_SomgProc_final.parquet \\
      --vae_ckpt ${VAE_CKPT} --vae_cfg ${VAE_CFG} \\
      --latent_name ${EMIT_LATENT_NAME} --resolution 256 \\
      --rank 0 --world_size 4 --aggregate_stats || exit 1

  # Persist emission latents immediately (so a later pbr failure doesn't lose them).
  EMIT_DST=${SCRATCH_74K}/emission_latents/${EMIT_LATENT_NAME}
  mkdir -p "\${EMIT_DST}"
  rsync -ah --info=progress2 \${DATA_ROOT}/emission_latents/${EMIT_LATENT_NAME}/ "\${EMIT_DST}/"
  echo "Emission latents persisted to \${EMIT_DST}"

  echo "=================================================================="
  echo "STREAM 2/2: production PBR latents (${PBR_LATENT_NAME})"
  echo "=================================================================="
  # encode_pbr_latent.py is metadata.csv-driven and filters pbr_voxelized==True.
  # Synthesize a minimal metadata.csv from the parquet sha index.
  python -u -c "
import pandas as pd, os
df = pd.read_parquet('\${DATA_ROOT}/df_SomgProc_final.parquet')
out = pd.DataFrame({'sha256': df.index.astype(str)})
out['pbr_voxelized'] = True
out.to_csv(os.path.join('\${DATA_ROOT}', 'metadata.csv'), index=False)
print('wrote metadata.csv with', len(out), 'rows')
"
  pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python -u data_toolkit/encode_pbr_latent.py \\
        --root \${DATA_ROOT} \\
        --pbr_voxel_root \${DATA_ROOT} \\
        --pbr_latent_root \${DATA_ROOT} \\
        --resolution 256 \\
        --enc_pretrained microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16 \\
        --rank \$r --world_size 4 &
    pids+=(\$!)
  done
  rank_fail=0
  for p in \${pids[@]}; do wait \$p || rank_fail=1; done
  if [ \$rank_fail -ne 0 ]; then echo "ERROR: pbr encode rank failed; abort."; exit 1; fi

  # PBR normalization stats (encode_pbr_latent.py doesn't write them) — same
  # mean/std aggregation the emission encoder does (torch.std default is
  # unbiased, so use ddof=1 here to match), written next to the latents.
  python -u -c "
import glob, json, os, numpy as np
d = os.path.join('\${DATA_ROOT}', 'pbr_latents', '${PBR_LATENT_NAME}')
feats = [np.load(p)['feats'] for p in sorted(glob.glob(os.path.join(d, '*.npz')))]
assert feats, 'no pbr latents to aggregate'
f = np.concatenate(feats, 0).astype(np.float64)
mean, std = f.mean(0), f.std(0, ddof=1)
json.dump({'mean': mean.tolist(), 'std': std.tolist(),
           'total_tokens': int(f.shape[0]), 'channels': int(f.shape[1]),
           'feat_min': float(f.min()), 'feat_max': float(f.max()), 'n_samples': len(feats)},
          open(os.path.join(d, 'normalization_stats.json'), 'w'), indent=2)
print('=== PBR normalization stats (paste into DiT config) ===')
print('    pbr_normalization:')
print('      mean: [' + ', '.join(f'{v:.6f}' for v in mean.tolist()) + ']')
print('      std: [' + ', '.join(f'{v:.6f}' for v in std.tolist()) + ']')
" || exit 1

  PBR_DST=${SCRATCH_74K}/pbr_latents/${PBR_LATENT_NAME}
  mkdir -p "\${PBR_DST}"
  rsync -ah --info=progress2 \${DATA_ROOT}/pbr_latents/${PBR_LATENT_NAME}/ "\${PBR_DST}/"
  echo "PBR latents persisted to \${PBR_DST}"
} 2>&1 | tee "\${LOG}"
RC=\${PIPESTATUS[0]}

echo "Encode finished rc=\$RC at \$(date)."
echo "  emission stats: ${SCRATCH_74K}/emission_latents/${EMIT_LATENT_NAME}/normalization_stats.json"
echo "  pbr stats:      ${SCRATCH_74K}/pbr_latents/${PBR_LATENT_NAME}/normalization_stats.json"
exit \$RC
EOF

echo "Submitted: ${NAME}"
