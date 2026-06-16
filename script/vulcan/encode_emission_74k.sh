#!/bin/bash
# Phase 1 (vulcan): encode 74k emission latents with the finished channel-hijack
# VAE. Single vulcan node (4 L40S); each rank encodes a shard to local SSD
# (shared $SLURM_TMPDIR), rank 0 aggregates stats, then rsync latents + stats
# back to persistent scratch. Cloned from script/fir/encode_emission_74k.sh
# with the vulcan deltas: login-node git sync (compute nodes have no outbound
# SSH), cvmfs module source, aip-msavva account, no in-body git pull.
#
# Prereq (Task V0): emission VAE ckpt staged to the vulcan path below.
#
# Usage:  bash script/vulcan/encode_emission_74k.sh

set -u

NAME="encode_emission_74k_vulcan"
REPO="/scratch/dya78/TRELLIS.2-lightning"
VAE_CKPT="${REPO}/outputs/emission_pbr_vae_ft_74k/ckpts/step0004485-0.0097.ckpt"
VAE_CFG="${REPO}/outputs/emission_pbr_vae_ft_74k/config.yaml"
LATENT_NAME="emission_pbr_vae_ft_74k"
SCRATCH_74K="/scratch/dya78/lightgen/lightgen_74k"
BRANCH="lightgen"

# Parallelism: NSHARDS independent single-node jobs, each handling 1/NSHARDS of
# the data (4 GPUs/node). Global rank = SHARD*4 + local_gpu; world_size =
# NSHARDS*4. Each shard writes to local SSD and rsyncs its slice to shared
# scratch (disjoint shas, so concurrent rsync to one dir is safe). The final
# normalization stats are computed separately over scratch by
# data_toolkit/aggregate_emission_stats.py (no GPU). NSHARDS=1 keeps the
# original single-job behaviour incl. the in-body aggregate.
GPUS=4
NSHARDS="${NSHARDS:-1}"
SHARD="${SHARD:-0}"
WORLD=$((NSHARDS * GPUS))
[ "${NSHARDS}" -gt 1 ] && NAME="${NAME}_s${SHARD}"

# Submit from a vulcan login node (sbatch on PATH) or from the workstation.
if command -v sbatch >/dev/null 2>&1; then
    SUBMIT=(sbatch)
    SUBMIT_DESC="local sbatch (already on a Slurm host)"
    run_on_vulcan() { ( cd "${REPO}" && eval "$1" ); }
else
    SUBMIT=(ssh vulcan sbatch)
    SUBMIT_DESC="ssh vulcan sbatch (from workstation)"
    run_on_vulcan() { ssh vulcan "cd ${REPO} && $1"; }
fi

# Sync code on the vulcan LOGIN node — compute nodes have no outbound SSH, so a
# git fetch inside the job body fails ("github.com port 22: Network is
# unreachable"). Sync here; the job body just cd's in.
echo "Syncing ${BRANCH} on vulcan login node..."
run_on_vulcan "git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}" \
    || { echo "FAIL: git sync on vulcan login node (diverged tree, or run from a host without repo+outbound)"; exit 1; }
SYNCED_COMMIT="$(run_on_vulcan 'git rev-parse --short HEAD' | tail -1)"
[ -n "${SYNCED_COMMIT}" ] || { echo "FAIL: could not read synced commit on vulcan"; exit 1; }

echo "Submitting job: ${NAME}"
echo "  Config: encode emission latents -> ${LATENT_NAME}"
echo "  VAE ckpt: ${VAE_CKPT}"
echo "  Submit:   ${SUBMIT_DESC}"
echo "  Commit:   ${SYNCED_COMMIT} (synced on login node)"

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH -J ${NAME}
#SBATCH --account=aip-msavva
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append

echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})  Node: \$(hostname)  Started: \$(date)"

# module is NOT on PATH under #!/bin/bash sbatch (no login profile sourced) —
# source the cvmfs profile first, else "module: command not found".
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

# Code was already synced to ${SYNCED_COMMIT} on the login node before submit —
# vulcan compute nodes have no outbound SSH so we cannot git fetch here.
cd ${REPO} || { echo "FAIL: repo dir missing"; exit 1; }
RUN_COMMIT=\$(git rev-parse --short HEAD)
echo "Repo HEAD: \${RUN_COMMIT} (launcher synced ${SYNCED_COMMIT})"
[ "\${RUN_COMMIT}" = "${SYNCED_COMMIT}" ] || echo "WARN: HEAD drifted from the launcher-synced commit since submission"

source ${REPO}/env/bin/activate \\
    || { echo "FAIL: virtualenv missing — run setup.sh on vulcan first"; exit 1; }

# Stage 74k voxels to local SSD (single node; extracts to \$SLURM_TMPDIR/lightgen_74k)
cd \$SLURM_TMPDIR
bash ${REPO}/script/fir/unpack_74k_data.sh
DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k

cd ${REPO}
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flex_gemm's shared autotune cache (~/.flex_gemm/autotune_cache.json) lives on
# NFS home; FileLock is unreliable there, so concurrent ranks corrupted it
# (JSONDecodeError "Extra data" on import -> all ranks crash). Disable cache
# read+write so each rank autotunes in-memory (one-time startup cost, no race).
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
set -o pipefail
{
  # 4 ranks, one per GPU, writing to the shared local save_dir.
  pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python -u data_toolkit/encode_emission_pbrfinetune_dataset.py \\
        --data_root \${DATA_ROOT} \\
        --parquet   \${DATA_ROOT}/df_SomgProc_final.parquet \\
        --vae_ckpt  ${VAE_CKPT} \\
        --vae_cfg   ${VAE_CFG} \\
        --latent_name ${LATENT_NAME} \\
        --resolution 256 \\
        --rank \$((${SHARD} * ${GPUS} + r)) --world_size ${WORLD} &
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
  # Aggregate stats in-body only for the single-job case. For sharded runs
  # (NSHARDS>1) the shards write disjoint slices to scratch and stats are
  # computed afterwards over scratch by data_toolkit/aggregate_emission_stats.py.
  if [ ${NSHARDS} -eq 1 ]; then
    python -u data_toolkit/encode_emission_pbrfinetune_dataset.py \\
        --data_root \${DATA_ROOT} \\
        --parquet   \${DATA_ROOT}/df_SomgProc_final.parquet \\
        --vae_ckpt  ${VAE_CKPT} \\
        --vae_cfg   ${VAE_CFG} \\
        --latent_name ${LATENT_NAME} \\
        --resolution 256 \\
        --rank 0 --world_size ${WORLD} --aggregate_stats
  fi
} 2>&1 | tee "\${LOG}"
ENC_RC=\${PIPESTATUS[0]}
if [ \$ENC_RC -ne 0 ]; then
  echo "Encode/aggregate failed (rc=\$ENC_RC); NOT rsyncing partial latents."
  exit \$ENC_RC
fi

# Persist latents + stats back to scratch (Lustre).
DST=${SCRATCH_74K}/emission_latents/${LATENT_NAME}
mkdir -p "\${DST}"
rsync -ah --info=progress2 \${DATA_ROOT}/emission_latents/${LATENT_NAME}/ "\${DST}/"

echo "Encode done at \$(date). Latents at \${DST}"
echo "Stats JSON: \${DST}/normalization_stats.json"
EOF

echo "Submitted: ${NAME}"
