#!/bin/bash
# Launch emission PBR VAE finetune (74k samples, channel-hijack) on vulcan cluster.
# Vulcan L40S nodes: 4 x L40S (48 GB) + 48 CPU/node + ~512 GB RAM/node.
#
# Multi-node DDP via sbatch + srun. Each node runs 4 tasks (one per GPU);
# Lightning's SLURM cluster env picks up SLURM_PROCID/SLURM_NODEID and wires
# DDP across all <NODES * 4> ranks. train.py is invoked with --num_gpus 4
# --num_nodes $NODES so build_trainer constructs a DDPStrategy.
#
# Usage:
#   bash script/vulcan/finetune_pbr_vae_74k.sh            # NODES=2 default
#   NODES=4 BS=20 bash script/vulcan/finetune_pbr_vae_74k.sh
#
# Tunable env vars:
#   NODES        number of compute nodes  (default 2)
#   BS           batch_size_per_gpu       (default 20  — L40S 48 GB, untested)
#   WALLTIME     SBATCH --time            (default 12:00:00 — fits b2 partition)
#   NAME_SUFFIX  appended to job name + output dir (default "_${NODES}node")

set -u

NODES="${NODES:-4}"
BS="${BS:-8}"
WALLTIME="${WALLTIME:-12:00:00}"
NAME_SUFFIX="${NAME_SUFFIX:-_${NODES}node}"

# Vulcan-tuned config: L40S BS, sqrt-scaled LR for the 2x larger global batch,
# halved max_steps to match fir's sample-count budget. See header in the yaml.
CONFIG="configs/scvae/emission_pbr_vae_finetune_74k_vulcan.yaml"
NAME="emission_pbr_vae_ft_74k_vulcan${NAME_SUFFIX}"
REPO="/scratch/dya78/TRELLIS.2-lightning"
OUTPUT_DIR="${REPO}/outputs/${NAME}"
BRANCH="lightgen"

# Submit directly if we're already on a Slurm host (sbatch on PATH = running
# on a vulcan login node); otherwise ssh in from the workstation. This lets
# the same launcher run from local OR from inside a vulcan tmux session
# without depending on an ssh ControlMaster being warm. run_on_vulcan() runs
# a command on the vulcan login node either way — used for the git sync below.
if command -v sbatch >/dev/null 2>&1; then
    SUBMIT=(sbatch)
    SUBMIT_DESC="local sbatch (already on a Slurm host)"
    run_on_vulcan() { ( cd "${REPO}" && eval "$1" ); }
else
    SUBMIT=(ssh vulcan sbatch)
    SUBMIT_DESC="ssh vulcan sbatch (from workstation)"
    run_on_vulcan() { ssh vulcan "cd ${REPO} && $1"; }
fi

# Sync code on the vulcan LOGIN node — NOT in the sbatch body. Vulcan compute
# nodes have no outbound SSH, so `git fetch` from inside the job fails with
# "github.com port 22: Network is unreachable". The login node has outbound,
# so sync here and let the job run against the already-checked-out tree.
echo "Syncing ${BRANCH} on vulcan login node..."
run_on_vulcan "git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}" \
    || { echo "FAIL: git sync on vulcan login node (diverged tree, or run from a host without repo+outbound)"; exit 1; }
# `| tail -1` strips any ssh MOTD noise; guard against an empty result so a
# failed rev-parse can't slip an empty commit string into the submitted job.
SYNCED_COMMIT="$(run_on_vulcan 'git rev-parse --short HEAD' | tail -1)"
[ -n "${SYNCED_COMMIT}" ] || { echo "FAIL: could not read synced commit on vulcan"; exit 1; }

echo "Submitting job: ${NAME}"
echo "  Nodes:    ${NODES} (x 4 L40S = $((NODES*4)) GPUs)"
echo "  BS/GPU:   ${BS}   (global BS = $((NODES*4*BS)))"
echo "  Walltime: ${WALLTIME}"
echo "  Config:   ${CONFIG}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Submit:   ${SUBMIT_DESC}"
echo "  Commit:   ${SYNCED_COMMIT} (synced on login node)"

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH -J ${NAME}
#SBATCH --account=aip-msavva
#SBATCH --nodes=${NODES}
#SBATCH --gpus-per-node=l40s:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
# --mem=0 = whole-node memory. L40S nodes on vulcan have ~512 GB; we want it
# all because activations + dataloader workers + 4 ranks compete for RSS.
# Switch to a concrete value (e.g. 480G) if --mem=0 ever gets rejected by
# the QoS on vulcan; do not lowball — see project_fir_pbr_vae_bs_sweep.md
# for the OOM history.
#SBATCH --time=${WALLTIME}
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

# ---- Graceful timeout: at T-5min, catch SIGTERM, kill srun, requeue.
# Trap fires on the launcher node (the batch script process). \$TRAIN_PID
# MUST be the srun PID — not tee's — for kill -TERM to propagate through
# the job step to every rank on every node, so Lightning's on_signal hook
# can checkpoint everywhere. See the process-substitution trick at the
# 'srun ... > >(tee "\$LOG") 2>&1 &' line below; using a '| tee' pipeline
# instead would set \$! to tee's PID and the trap would be a no-op.
handle_timeout() {
    echo "Timeout signal received (5 min remaining). Stopping training gracefully..."
    if [ ! -z "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        echo "Waiting for srun to forward SIGTERM and ranks to checkpoint..."
        wait \$TRAIN_PID
        echo "srun exited. Requeueing job..."
    fi
    scontrol requeue \${SLURM_JOB_ID} || { echo "FATAL: scontrol requeue failed; run is lost"; exit 1; }
    exit 0
}
trap 'handle_timeout' SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Started: \$(date)"
echo "Launcher node: \$(hostname)"
echo "Allocated nodes (\${SLURM_NNODES}): \${SLURM_JOB_NODELIST}"
echo "GPUs per node: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "======================================"

# Modules — mirror fir's working stack; all available on vulcan (verified
# via 'module spider' on vulcan2: arrow/21.0.0, opencv/4.13.0, sparsehash/2.0.4,
# python/3.11.5, cuda/12.6, StdEnv/2023, gcc all present).
#
# IMPORTANT: sbatch runs with #!/bin/bash (NOT login shell), so /etc/profile
# and /cvmfs/.../config/profile/bash.sh aren't auto-sourced and the module
# command is not on PATH. Symptom when missing: "line N: module: command
# not found" in the slurm log, then cv2 (or torch, anything from a module)
# fails to import at training time. Source the cvmfs profile explicitly.
# (No backticks in this comment block: the parent launcher's heredoc is
# unquoted, so backticks would be evaluated by the LOCAL submitter shell.)
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

# Code was already synced to ${SYNCED_COMMIT} on the login node before this
# job was submitted — vulcan compute nodes have no outbound SSH so we cannot
# git fetch here. Just cd in and log what we're running against; warn (don't
# fail) if HEAD drifted from what the launcher synced.
cd /scratch/dya78/TRELLIS.2-lightning || { echo "FAIL: repo dir missing"; exit 1; }
RUN_COMMIT=\$(git rev-parse --short HEAD)
echo "Repo HEAD: \${RUN_COMMIT} (launcher synced ${SYNCED_COMMIT})"
[ "\${RUN_COMMIT}" = "${SYNCED_COMMIT}" ] || echo "WARN: HEAD drifted from the launcher-synced commit since submission"

# Activate virtualenv (built by setup.sh; see project_fir_deployment.md)
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate \\
    || { echo "FAIL: virtualenv missing — run setup.sh on vulcan first"; exit 1; }

# ---- Multi-node data staging
# Each node has its own \$SLURM_TMPDIR (local SSD, same path string everywhere
# but physically distinct). One unpack step per node — srun fans out to all
# nodes, each cd's into its own SLURM_TMPDIR then runs the unpack (which writes
# to './lightgen_74k' relative to cwd).
echo ""
echo "Staging 74k voxels on each of \${SLURM_NNODES} nodes..."
srun --ntasks=\${SLURM_NNODES} --ntasks-per-node=1 \\
    bash -c 'cd "\$SLURM_TMPDIR" && bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_data.sh' \\
    || { echo "FAIL: multi-node data staging — see srun output above";
         scontrol requeue \${SLURM_JOB_ID} || { echo "FATAL: scontrol requeue failed after staging error; run is lost"; exit 1; };
         exit 0; }

DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k

cd /scratch/dya78/TRELLIS.2-lightning

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG="log/\${TS}-${NAME}.log"

echo ""
echo "Data root (per-node): \${DATA_ROOT}"
echo "Output dir:           ${OUTPUT_DIR}"
echo "Log:                  \${LOG}"
echo ""

# Reduce VRAM fragmentation. PYTORCH_CUDA_ALLOC_CONF is the old name;
# PYTORCH_ALLOC_CONF is the new one. Set both for fwd/bwd compatibility.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pipefail so the pipeline's exit reflects python's exit (not tee's).
set -o pipefail

# ---- Multi-node DDP launch
# srun spawns SLURM_NNODES * 4 ranks total (--ntasks-per-node=4 from sbatch).
# Lightning's SLURM cluster env routes them via SLURM_PROCID. We pass
# --num_gpus 4 (per-node devices) and --num_nodes \${SLURM_NNODES} so
# build_trainer constructs DDPStrategy with devices=4, num_nodes=\${NODES}.
# Backticks here would trigger LOCAL command substitution inside the
# unquoted heredoc; everything that should defer to runtime is \\-escaped.
# Use process substitution (> >(tee ...)) so srun is the tail of the
# backgrounded command. With '| tee', \$! would be tee's PID and the SIGTERM
# trap would have no path to the remote ranks. With >(tee ...), \$! is srun's
# PID — kill -TERM \$TRAIN_PID forwards SIGTERM through the srun step to every
# rank, and Lightning's signal handler runs its checkpoint save.
srun python -u train.py \\
    --config ${CONFIG} \\
    --output_dir ${OUTPUT_DIR} \\
    --ckpt last \\
    --num_gpus 4 \\
    --num_nodes \${SLURM_NNODES} \\
    --override trainer.args.batch_size_per_gpu=${BS} \\
    --override dataset.args.data_root=\${DATA_ROOT} \\
    --override dataset.args.parquet_file=\${DATA_ROOT}/df_SomgProc_final.parquet \\
    --override dataset.args.split_json=\${DATA_ROOT}/data_splits_emissive_74k_stratified_voxonly.json \\
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo "Training crashed (exit code \$EXIT_CODE) at \$(date). Requeueing job..."
    scontrol requeue \${SLURM_JOB_ID} || { echo "FATAL: scontrol requeue failed after crash; run is lost"; exit 1; }
    exit 0
fi

echo "Training completed at \$(date)."
EOF

echo "Submitted: ${NAME}"
