#!/bin/bash
# SegviGen-Emission, 256-native lane, on the released dataset: 4 x H100, one node, on ANY Alliance cluster.
# The paper run (segvigen_256_bw_6gpu) used 6 x RTX Pro 6000 x batch 8; this keeps its effective batch: 4 x 12 = 48.
# Modeled on TRELLIS.2-lightning script/alliance/emission_dit_74k.sh (per-cluster headers, hedged-submission
# grace minute, SIGTERM -> requeue, data unpacked into $SLURM_TMPDIR, weights from a staged offline cache).
#
# hedge.sh runs a launcher from ~/scratch/TRELLIS.2-lightning (whose env/ this lane shares), so call it as
#   hedge.sh submit <run> fir=../segvigen-lightning/script/alliance/segvigen_256_release.sh ...
# or by hand on a login node:  bash ~/scratch/segvigen-lightning/script/alliance/segvigen_256_release.sh [--dry-run]
#
#   env: MAX_STEPS=<n> bounded probe; OUT_SUFFIX=_probe keeps a probe out of the real run's directory (its
#        last.ckpt and wandb id would otherwise be resumed); WALLTIME=HH:MM:SS (default 24:00:00);
#        BATCH_SIZE_PER_GPU (12), BATCH_SPLIT (1; 2 halves the activation memory at the same batch); CLUSTER=<name>;
#        EPOCH_STEPS_OVERRIDE=<n> (probes only: a short "epoch" so the snapshot sidecar gets exercised)
#
# Staged per cluster under ~/scratch/lightgen/segvigen_release/ (scripts_segvigen/newdata/verify_and_pack.py):
#   tars/latents.tar tars/cond_*.tar tars/SHA256SUMS   full_seg_rekeyed.pt   segvigen-lightning.bundle
set -u
CONFIG="configs/gen/segvigen_emissive_256_release.yaml"; NAME="segvigen_256_release_4h100"; BRANCH="retrain-newdata"
MAX_STEPS="${MAX_STEPS:-}"; WALLTIME="${WALLTIME:-24:00:00}"; DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
BS="${BATCH_SIZE_PER_GPU:-12}"; SPLIT="${BATCH_SPLIT:-1}"
CLUSTER="${CLUSTER:-$(scontrol show config 2>/dev/null | awk -F'= ' '/^ClusterName/{print $2}')}"
CODE="${CODE:-$HOME/scratch/segvigen-lightning}"; ENVREPO="${ENVREPO:-$HOME/scratch/TRELLIS.2-lightning}"
STAGE="$HOME/scratch/lightgen/segvigen_release"; OUTPUT_DIR="$CODE/outputs/${NAME}${OUT_SUFFIX:-}"
MODULES="StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash"
SIGNAL="#SBATCH --signal=B:SIGTERM@1500"; MEM="#SBATCH --mem=1024G"; GPU="#SBATCH --gpus-per-node=h100:4"; TIME="#SBATCH --time=${WALLTIME}"
case "$CLUSTER" in
  fir)       ACCOUNT=rrg-msavva ;;
  nibi)      ACCOUNT=def-msavva_gpu ;;
  rorqual)   ACCOUNT=def-msavva_gpu; MEM="#SBATCH --mem=450G" ;;
  killarney) ACCOUNT=aip-msavva ;;
  tamia)     ACCOUNT=aip-msavva; MEM="#SBATCH --mem=450G"; GPU="#SBATCH --gres=gpu:h100:4" ;;
  grillium|trillium*) ACCOUNT=def-msavva; MEM=""; SIGNAL=""; GPU="#SBATCH --gpus-per-node=4"
             TIME="#SBATCH -t $(( $(echo "$WALLTIME" | awk -F: '{print $1*60+$2}') ))" ;;
  *) echo "unknown cluster '$CLUSTER' — set CLUSTER=…"; exit 2 ;;
esac
for f in tars/latents.tar tars/SHA256SUMS full_seg_rekeyed.pt; do
  [ -f "$STAGE/$f" ] || { echo "staged input missing: $STAGE/$f"; [ $DRY = 1 ] || exit 2; }
done
[ -x "$ENVREPO/env/bin/python" ] || { echo "environment missing: $ENVREPO/env"; [ $DRY = 1 ] || exit 2; }
mkdir -p "$CODE/log"; JOB="$CODE/log/job_${NAME}${OUT_SUFFIX:-}_${CLUSTER}.sh"
cat > "$JOB" <<JOB
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
${GPU}
${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
${MEM}
#SBATCH --account=${ACCOUNT}
#SBATCH --requeue
${SIGNAL}
# 25-min lead before the time limit: a 10 GB last.ckpt on Lustre must finish before SIGKILL.
# The sidecar runs in its own process group so its converter child dies with it (a bare kill of the shell orphans it).
stop_sidecar() { [ -n "\${SIDE_PID:-}" ] && { kill -TERM -- -\$SIDE_PID 2>/dev/null; wait \$SIDE_PID 2>/dev/null; SIDE_PID=""; }; }
handle_timeout() { echo "SIGTERM: stopping training gracefully..."; stop_sidecar; [ -n "\${EPOCH_STEPS:-}" ] && bash script/alliance/snapshot_sidecar.sh ${OUTPUT_DIR} \$EPOCH_STEPS flush >> "\$SIDELOG" 2>&1; [ -n "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null && { kill -TERM \$TRAIN_PID; wait \$TRAIN_PID; }; scontrol requeue \${SLURM_JOB_ID}; exit 0; }
trap 'handle_timeout' SIGTERM
sleep 60   # hedged submission: a job cancelled in its first minute has cost nothing
echo "====================================== Job \${SLURM_JOB_NAME} (\${SLURM_JOB_ID}) cluster ${CLUSTER} node \$(hostname) started \$(date) restart=\${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module load ${MODULES}
source ${ENVREPO}/env/bin/activate
DATA=\$SLURM_TMPDIR/segvigen_release; mkdir -p \$DATA
t0=\$SECONDS
# exactly the tars SHA256SUMS names (a stray tar of an older pack must not be unpacked on top)
awk '{print \$2}' ${STAGE}/tars/SHA256SUMS | xargs -P 9 -I{} tar -xf ${STAGE}/tars/{} -C \$DATA || { echo "FATAL: unpack failed"; exit 1; }
N_USABLE=\$(wc -l < \$DATA/meta/usable.txt); N_COND=\$(ls \$DATA/cond_fp16 | wc -l); N_EMIT=\$(ls \$DATA/emission_npz | grep -c '\.npz\$')
N_TRAIN=\$(python -c "import json; print(len(json.load(open('\$DATA/meta/release_splits_usable.json'))['train']))")
echo "unpacked in \$((SECONDS-t0))s: usable=\$N_USABLE cond=\$N_COND emission_npz=\$N_EMIT train=\$N_TRAIN"
[ "\$N_COND" = "\$N_USABLE" ] && [ "\$N_EMIT" = "\$N_USABLE" ] || { echo "FATAL: unpacked counts differ from usable.txt"; exit 1; }
EPOCH_STEPS=${EPOCH_STEPS_OVERRIDE:-\$(( N_TRAIN / (4 * ${BS}) ))}
cd ${CODE}
mkdir -p log ${OUTPUT_DIR}/snapshots; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}${OUT_SUFFIX:-}-${CLUSTER}.log"; SIDELOG="log/\${TS}-${NAME}${OUT_SUFFIX:-}-${CLUSTER}-sidecar.log"
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
# fir keeps its Hugging Face cache in ~/.cache, the other clusters got theirs staged in ~/scratch/.cache
if [ -d \$HOME/scratch/.cache/huggingface/hub ]; then export HF_HOME=\$HOME/scratch/.cache/huggingface; else export HF_HOME=\$HOME/.cache/huggingface; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; echo "HF_HOME=\$HF_HOME"
if [ -f \$HOME/.netrc ] && curl -sSI -m 8 https://api.wandb.ai >/dev/null 2>&1; then echo "wandb: online"; else export WANDB_MODE=offline; echo "wandb: OFFLINE (sync later from a login node)"; fi
if ! ( : > \$HOME/.wtest.\$\$ ) 2>/dev/null; then export TRITON_CACHE_DIR=\$SLURM_TMPDIR/.triton CUDA_CACHE_PATH=\$SLURM_TMPDIR/.nv MPLCONFIGDIR=\$SLURM_TMPDIR/.mpl WANDB_CACHE_DIR=\$SLURM_TMPDIR/.wandb WANDB_CONFIG_DIR=\$SLURM_TMPDIR/.wandb_cfg XDG_CACHE_HOME=\$SLURM_TMPDIR/.cache; echo "HOME read-only: caches redirected to \$SLURM_TMPDIR"; else rm -f \$HOME/.wtest.\$\$; fi
# one eval-ready weights file per epoch (the paper run kept full 10.5 GB states through an external mirror)
setsid bash script/alliance/snapshot_sidecar.sh ${OUTPUT_DIR} \$EPOCH_STEPS > "\$SIDELOG" 2>&1 &
SIDE_PID=\$!
set -o pipefail
srun python -u train.py \\
    --config ${CONFIG} --output_dir ${OUTPUT_DIR} --ckpt last --num_gpus 4 \\
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \\
    --override trainer.args.batch_size_per_gpu=${BS} \\
    --override trainer.args.batch_split=${SPLIT} \\
    --override trainer.args.ckpt_last_every_n_steps=\${EPOCH_STEPS} \\
    --override trainer.args.disable_final_snapshot=true \\
    --override trainer.args.wandb_name=${NAME}${OUT_SUFFIX:-} \\
    --override trainer.args.finetune_ckpt.denoiser=${STAGE}/full_seg_rekeyed.pt \\
    --override dataset.args.data_root=\${DATA}/meta \\
    --override dataset.args.parquet_file=\${DATA}/meta/release_index.parquet \\
    --override dataset.args.split_json=\${DATA}/meta/release_splits_usable.json \\
    --override dataset.args.shape_latent_root=\${DATA} \\
    --override dataset.args.pbr_latent_root=\${DATA} \\
    --override dataset.args.emission_latent_dir=\${DATA}/emission_npz \\
    --override "dataset.args.cond_search_dirs=[\"\${DATA}/cond_fp16\"]" \\
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID; EXIT_CODE=\$?
stop_sidecar
bash script/alliance/snapshot_sidecar.sh ${OUTPUT_DIR} \$EPOCH_STEPS once >> "\$SIDELOG" 2>&1
if [ \$EXIT_CODE -ne 0 ]; then
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then echo "Training crashed (exit \$EXIT_CODE) at \$(date). Requeueing..."; scontrol requeue \${SLURM_JOB_ID}; exit 0; fi
    echo "Training crashed (exit \$EXIT_CODE); restart cap reached. NOT requeueing."; exit \$EXIT_CODE
fi
echo "Training completed at \$(date). Outputs: ${OUTPUT_DIR} — back them up to the project store from the workstation."
JOB
sed -i '/^$/d' "$JOB"
if [ $DRY = 1 ]; then cat "$JOB"; exit 0; fi
cd -P "$CODE" || exit 2
# Code arrives as a git bundle made on the workstation (the fork's GitHub repo is Kevin's and six of its commits
# were never pushed); never edited here.
B="$STAGE/segvigen-lightning.bundle"
[ -f "$B" ] && { git pull -q --ff-only "$B" "$BRANCH" || echo "WARNING: git pull --ff-only from the bundle failed — submitting the tree as it is"; }
echo "code: $(git rev-parse --short HEAD) on $(git branch --show-current)"
sbatch "$JOB"
