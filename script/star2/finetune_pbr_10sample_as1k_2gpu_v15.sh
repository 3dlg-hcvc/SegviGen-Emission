#!/bin/bash
# venus-15 variant of finetune_pbr_10sample_as1k_2gpu.sh.
#
# Differences vs the venus-16 script:
#   - nodelist → cs-venus-15
#   - output path → /project/3dlg-hcvc/lightgen/probe/<NAME>.out (shared NFS, visible
#     from star2 head node for tailing). Training I/O still goes to /localscratch.
#   - Skips `source ~/.zshrc`/`conda activate`: venus-15's per-node home has no
#     conda init. Uses the rsynced standalone env at
#     /localscratch/dya78/miniconda3/envs/trellis2 via direct PATH.
#
# Everything else (training hyperparameters, BS, grad clip, fp16, EMA) is
# identical to the venus-16 variant — so this still runs the unmodified 1k
# config on the 10-sample-as-1k dataset.

set -euo pipefail

BS=${BS:-12}
BATCH_SPLIT=${BATCH_SPLIT:-1}
WORKERS=${WORKERS:-}
MAX_STEPS=${MAX_STEPS:-}
if [[ "${BATCH_SPLIT}" != "1" ]]; then
    TAG=${TAG:-bs${BS}_split${BATCH_SPLIT}}
else
    TAG=${TAG:-bs${BS}}
fi

CONFIG="configs/scvae/emission_pbr_finetune_10sample_as1k.yaml"
NAME="emission_pbr_ft_10s_as1k_v15_${TAG}"
ENV_PREFIX="/localscratch/dya78/miniconda3/envs/trellis2"

# NOTE: dataset paths come from the config as repo-relative paths
# (data/lightgen_test/...). The data lives as real files inside the repo on
# every host. Sync the data dir to the cluster once (or whenever it changes):
#   rsync -av --partial data/lightgen_test/ \
#       star2:/localscratch/dya78/TRELLIS.2-lightning/data/lightgen_test/

echo "Submitting job: ${NAME}"
echo "  config: ${CONFIG}"
echo "  batch_size_per_gpu: ${BS}  (global = $((BS*2)))"
if [[ -n "${MAX_STEPS}" ]]; then echo "  max_steps override: ${MAX_STEPS}"; fi

STEPS_OVERRIDE=""
if [[ -n "${MAX_STEPS}" ]]; then
    STEPS_OVERRIDE="--override trainer.args.max_steps=${MAX_STEPS}"
fi
WORKERS_OVERRIDE=""
if [[ -n "${WORKERS}" ]]; then
    WORKERS_OVERRIDE="--override trainer.args.dataloader_num_workers=${WORKERS}"
fi

ssh star2 "sbatch" << EOF
#!/bin/bash
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=128G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-15
#SBATCH --output=/project/3dlg-hcvc/lightgen/probe/${NAME}.out
#SBATCH -J ${NAME}
#SBATCH -A 3dlg-hcvc-lab

echo "Start: ${NAME}"
echo "Config: ${CONFIG}  BS=${BS}  STEPS_OVERRIDE='${STEPS_OVERRIDE}'"

# venus-15 has no per-node conda init. Use rsynced standalone env directly.
export CONDA_PREFIX="${ENV_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="/localscratch/dya78/extralibs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

# User-site packages (pyarrow, fastparquet, tensorboard, etc.) that live in
# ~/.local on the local workstation but not inside the conda env.
export PYTHONPATH=/localscratch/dya78/extra_site_packages:\${PYTHONPATH:-}

# HF cache was pre-seeded with the tex_enc/tex_dec TRELLIS.2-4B checkpoints.
export HF_HOME=/localscratch/dya78/hf_cache
export HF_HUB_OFFLINE=1

cd /localscratch/dya78/TRELLIS.2-lightning

# Reduce fragmentation — helps sparse voxel workloads with variable batch memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pwd: \$(pwd)"
echo "python: \$(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Background GPU memory monitor.
( while true; do
    echo "[MEMLOG] \$(date +%s) \$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' ';')"
    sleep 15
  done ) &
MON_PID=\$!

python train.py \\
    --config ${CONFIG} \\
    --output_dir outputs/${NAME} \\
    --num_gpus 2 \\
    --ckpt last \\
    --override trainer.args.batch_size_per_gpu=${BS} \\
    --override trainer.args.batch_split=${BATCH_SPLIT} \\
    ${WORKERS_OVERRIDE} \\
    ${STEPS_OVERRIDE}

TRAIN_EXIT=\$?
kill \$MON_PID 2>/dev/null || true
echo "Job End: ${NAME}  exit=\$TRAIN_EXIT"
EOF

echo "Submitted: ${NAME}"
