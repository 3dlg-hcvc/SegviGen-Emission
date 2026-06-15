# Emission DiT 74k Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Cluster note:** All fir tasks use the `cluster-access` skill and run inside a tmux session named after the node (per project convention). Never edit code on the cluster — edit locally, `git push origin lightgen`, and the launcher bodies `git pull` from origin. fir paths: repo `/scratch/dya78/TRELLIS.2-lightning`, venv `/scratch/dya78/TRELLIS.2-lightning/env`, 74k source `/scratch/dya78/lightgen/lightgen_74k/`.

**Goal:** Train the emission DiT at 74k scale by retargeting the pretrained `imgshape2tex` texture DiT to predict emission latents (64ch channel-hijack, full weight reuse), on fir (4×H100).

**Architecture:** The DiT mirrors the VAE's channel-hijack: byte-identical to `slat_flow_imgshape2tex_dit_1_3B_512_bf16` (in=64 = 32 noised emission latent + 32 shape latent; image via DinoV3 cross-attn; `skip_pbr_cond=true`; out=32). 100% pretrained-weight reuse including the input projection. Output emission latents decode through the finished `emission_pbr_vae_ft_74k` VAE.

**Tech Stack:** PyTorch Lightning, `ElasticSLatFlowModel` (sparse flow-matching DiT), `LightGenSLatEmissionLatent` dataset, DinoV3 image conditioning, fir Slurm (StdEnv/2023, cuda/12.6, persistent venv).

**Spec:** `docs/superpowers/specs/2026-06-15-emission-dit-74k-design.md`

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `script/fir/encode_emission_74k.sh` | Create | Phase 1: sharded 4-GPU encode of 74k emission latents with the finished VAE; aggregate norm stats; rsync latents back to scratch. |
| `configs/gen/emission_dit_74k.yaml` | Create | The 64ch channel-hijack DiT training config (clone of the 1k probe, 74k paths + correct stats + signed-off hyperparams). |
| `script/fir/unpack_74k_dit_data.sh` | Create | Stage the 74k DiT slice (emission_latents + shape_latents + thumbnails + parquet + split) to `$SLURM_TMPDIR`. |
| `script/fir/emission_dit_74k.sh` | Create | Phase 4: full 74k DiT training launcher (4×H100, srun+Lightning DDP, requeue/SIGTERM). |
| `data_toolkit/test_emission_dit_74k.py` | Create | Phase 5: per-sample eval (GT vs VAE-recon vs DiT-decode) + max-projection PNG. |

No core-code changes: `LightGenSLatEmissionLatent` already exposes `self.loads` (token count) for the balanced sampler, and the finetune loader already reuses the input projection when channels match (64==64), as proven by the 1k probe.

---

## Task 1: Phase 0 — verify fir state (go/no-go gate)

**Files:** none (interactive verification on fir via `cluster-access`).

This gate confirms the four DiT inputs exist on fir and pins the exact VAE checkpoint. Subsequent tasks assume its findings; if any differ, note the correction inline before proceeding.

- [ ] **Step 1: Open an interactive shell on fir**

Per `cluster-access`, from the local workstation open a tmux session and `ssh fir`, then land on the login node (no GPU needed for `ls`).

- [ ] **Step 2: Verify the 74k source tree on scratch**

Run:
```bash
ls -la /scratch/dya78/lightgen/lightgen_74k/
ls /scratch/dya78/lightgen/lightgen_74k/voxels/part{0..7}_voxels.tar
ls /scratch/dya78/lightgen/lightgen_74k/df_SomgProc_final.parquet
ls /scratch/dya78/lightgen/lightgen_74k/data_splits_emissive_74k_stratified_voxonly.json
```
Expected: 8 voxel tars, the parquet, and the `*_voxonly.json` split all present.

- [ ] **Step 3: Verify shape latents + thumbnails exist for 74k**

Run:
```bash
# Shape latents (already produced by 74k prep per CLAUDE.md):
ls /scratch/dya78/lightgen/lightgen_74k/shape_latents/shape_enc_next_dc_f16c32_fp16_256/ | head
ls /scratch/dya78/lightgen/lightgen_74k/shape_latents/shape_enc_next_dc_f16c32_fp16_256/*.npz | wc -l
# Thumbnails (the one DiT input the VAE run never used — confirm dir name + PNGs):
ls /scratch/dya78/lightgen/lightgen_74k/thumbnails/ 2>/dev/null | head
ls /scratch/dya78/lightgen/lightgen_74k/thumbnails/*.png 2>/dev/null | wc -l
```
Expected: ~73k shape `.npz`. For thumbnails — if the directory is absent or empty, STOP and resolve thumbnail generation before continuing (the DiT cannot be image-conditioned without them). Record the actual thumbnail directory name/path for use in Tasks 2/4.

- [ ] **Step 4: Pin the final VAE checkpoint**

Run:
```bash
ls -la /scratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_vae_ft_74k/ckpts/
```
Expected: confirm `step0004485-0.0097.ckpt` (or a newer best-val ckpt if the run progressed past step 4485) and a sibling `config.yaml`. Record the exact filename — it is the `--vae_ckpt` for Task 2.

- [ ] **Step 5: Record findings**

Write the confirmed paths (thumbnail dir, VAE ckpt filename, shape-latent count) as a short comment block at the top of `configs/gen/emission_dit_74k.yaml` when authoring Task 3, so "what data this run used" is unambiguous. No commit in this task.

---

## Task 2: Phase 1 — encode 74k emission latents (the prerequisite)

**Files:**
- Create: `script/fir/encode_emission_74k.sh`

The encoder script `data_toolkit/encode_emission_pbrfinetune_dataset.py` already exists and is shard-aware. This task adds the fir launcher that stages voxels to local SSD, runs 4 ranks (one per H100) writing latents locally, aggregates stats on rank 0, then rsyncs the latents + stats back to persistent scratch.

- [ ] **Step 1: Write the encode launcher**

Create `script/fir/encode_emission_74k.sh`:
```bash
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
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python data_toolkit/encode_emission_pbrfinetune_dataset.py \\
        --data_root \${DATA_ROOT} \\
        --parquet   \${DATA_ROOT}/df_SomgProc_final.parquet \\
        --vae_ckpt  ${VAE_CKPT} \\
        --vae_cfg   ${VAE_CFG} \\
        --latent_name ${LATENT_NAME} \\
        --resolution 256 \\
        --rank \$r --world_size 4 &
  done
  wait
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

# Persist latents + stats back to scratch (Lustre); also copy the printed stats.
DST=${SCRATCH_74K}/emission_latents/${LATENT_NAME}
mkdir -p "\${DST}"
rsync -ah --info=progress2 \${DATA_ROOT}/emission_latents/${LATENT_NAME}/ "\${DST}/"

echo "Encode done at \$(date). Latents at \${DST}"
echo "Stats JSON: \${DST}/normalization_stats.json"
EOF

echo "Submitted: ${NAME}"
```

- [ ] **Step 2: Lint the launcher**

Run: `bash -n script/fir/encode_emission_74k.sh`
Expected: no syntax errors (exit 0).

- [ ] **Step 3: Commit the launcher**

```bash
git add script/fir/encode_emission_74k.sh
git commit -m "feat(emission-dit-74k): fir launcher to encode 74k emission latents"
git push origin lightgen
```

- [ ] **Step 4: Submit the encode job (on fir)**

Per `cluster-access`, from the fir login node: `bash script/fir/encode_emission_74k.sh`. Monitor with `squeue --me` and `tail -f <node>-<jobid>.out`.

- [ ] **Step 5: Verify the encode completed**

Run (on fir):
```bash
ls /scratch/dya78/lightgen/lightgen_74k/emission_latents/emission_pbr_vae_ft_74k/*.npz | wc -l
cat /scratch/dya78/lightgen/lightgen_74k/emission_latents/emission_pbr_vae_ft_74k/normalization_stats.json
```
Expected: latent count ≈ shape-latent count from Task 1 Step 3 (≈73k, modulo the ~1% prep losses). `normalization_stats.json` has 32-length `mean`/`std`, `channels: 32`, and a plausible `n_samples`. **Copy the `mean`/`std` arrays** — they go into Task 3.

---

## Task 3: Author the 74k DiT config

**Files:**
- Create: `configs/gen/emission_dit_74k.yaml`

Clone the structure of `configs/gen/emission_dit_1k_probe.yaml` (already the correct 64ch / `skip_pbr_cond` shape). Three substantive changes: 74k data paths, the **emission_normalization from Task 2**, and **shape_normalization from the base DiT's production stats** (not the 10-sample stats the probe carried).

- [ ] **Step 1: Write the config**

Create `configs/gen/emission_dit_74k.yaml`. Use this exact content, pasting the emission `mean`/`std` arrays from Task 2 Step 5 into the marked block:

```yaml
# Emission DiT — 74k channel-hijack (retarget imgshape2tex texture DiT to emission).
#
# Data confirmed in Phase 0 (Task 1): thumbnails at <fill from Task1 Step3>,
# VAE ckpt <fill from Task1 Step4>, emission latents emission_pbr_vae_ft_74k.
#
# Architecture byte-identical to slat_flow_imgshape2tex_dit_1_3B_512_bf16:
#   in=64 (32 noised emission + 32 shape), out=32, image via DinoV3 cross-attn.
#   skip_pbr_cond=true -> no PBR stream. Full weight reuse incl. input proj.

models:
  denoiser:
    name: ElasticSLatFlowModel
    args:
      resolution: 32
      in_channels: 64
      out_channels: 32
      model_channels: 1536
      cond_channels: 1024
      num_blocks: 30
      num_heads: 12
      mlp_ratio: 5.3334
      pe_mode: rope
      share_mod: true
      initialization: scaled
      qk_rms_norm: true
      qk_rms_norm_cross: true

dataset:
  name: LightGenSLatEmissionLatent
  args:
    data_root: data/lightgen_74k
    parquet_file: data/lightgen_74k/df_SomgProc_final.parquet
    emission_latent_name: emission_pbr_vae_ft_74k
    pbr_latent_name: tex_enc_next_dc_f16c32_fp16_256   # unused (skip_pbr_cond)
    shape_latent_name: shape_enc_next_dc_f16c32_fp16_256
    thumbnail_dir: data/lightgen_74k/thumbnails
    image_size: 512
    resolution: 256
    split: train
    split_json: data/lightgen_74k/data_splits_emissive_74k_stratified_voxonly.json
    skip_pbr_cond: true
    emission_normalization:
      # >>> PASTE mean/std from Task 2 (normalization_stats.json) <<<
      mean: [REPLACE_WITH_32_VALUES_FROM_TASK2]
      std:  [REPLACE_WITH_32_VALUES_FROM_TASK2]
    shape_normalization:
      # Production shape stats from slat_flow_imgshape2tex_dit_1_3B_512_bf16.yaml
      # (shape_slat_normalization) — the shape latents are the production
      # shape_enc_next_dc_f16c32_fp16_256, so these are the correct stats.
      mean: [0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252, 1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667, -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944, -3.186688, 1.578665]
      std: [5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237, 5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013, 5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194, 5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691]

trainer:
  name: ImageConditionedSparseFlowMatchingCFGTrainer
  args:
    max_steps: 50000
    batch_size_per_gpu: 8     # probe in Task 6 may bump to 16
    batch_split: 2
    optimizer:
      name: AdamW
      args:
        lr: 0.0001
        weight_decay: 0.01
        betas: [0.9, 0.95]
        eps: 1.0e-08
    lr_scheduler:
      name: LinearWarmupLRScheduler
      args:
        warmup_steps: 1000
    ema_rate:
    - 0.9999
    t_schedule:
      name: uniform
      args: {}
    sigma_min: 1.0e-05
    p_uncond: 0.1
    mix_precision_mode: amp
    mix_precision_dtype: bfloat16
    elastic:
      name: LinearMemoryController
      args:
        target_ratio: 0.75
        max_mem_ratio_start: 0.5
    grad_clip:
      name: AdaptiveGradClipper
      args:
        max_norm: 1.0
        clip_percentile: 95
    finetune_ckpt:
      denoiser: microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16
    i_log: 50
    i_sample: 5000
    skip_initial_samples: true
    dataloader_num_workers: 8
    use_balanced_sampler: true
    num_sanity_val_steps: 0
    val_split_name: val
    val_every_epochs: 1        # recalibrate after the probe gives steps/sec
    val_batch_size_per_gpu: 4
    save_top_k: 1
    monitor_metric: val/loss
    monitor_mode: min
    wandb_enable: true
    wandb_project: emission-dit
    wandb_name: emission_dit_74k
    image_cond_model:
      name: DinoV3FeatureExtractor
      args:
        model_name: facebook/dinov3-vitl16-pretrain-lvd1689m
        image_size: 512
```

- [ ] **Step 2: Validate the config loads (dry run)**

Run (locally, where `data/lightgen_74k` need not exist — `--tryrun` validates config construction):
```bash
python train.py --config configs/gen/emission_dit_74k.yaml \
  --output_dir /tmp/emission_dit_74k_dryrun --tryrun
```
Expected: config parses, models build (denoiser ~1.3B params reported), no schema/key errors. (If it fails because it tries to read data, instead validate YAML only: `python -c "import yaml; yaml.safe_load(open('configs/gen/emission_dit_74k.yaml'))"` → exit 0.)

- [ ] **Step 3: Confirm no placeholder arrays remain**

Run: `grep -n "REPLACE_WITH" configs/gen/emission_dit_74k.yaml`
Expected: no matches (the emission mean/std were pasted in Step 1).

- [ ] **Step 4: Commit the config**

```bash
git add configs/gen/emission_dit_74k.yaml
git commit -m "feat(emission-dit-74k): 64ch channel-hijack DiT config (74k, prod shape stats)"
git push origin lightgen
```

---

## Task 4: Stage the 74k DiT slice (data packing)

**Files:**
- Create: `script/fir/unpack_74k_dit_data.sh`

The DiT reads latents + thumbnails (not voxels). Mirror `unpack_1k_dit_data.sh`: assemble a single `lightgen_74k_dit.tar` on scratch (emission_latents + shape_latents + thumbnails + parquet + split), and an unpack script that stages it to `$SLURM_TMPDIR`.

- [ ] **Step 1: Build the DiT slice tar (on fir)**

Run (on fir, after Task 2 completed so emission latents exist):
```bash
cd /scratch/dya78/lightgen/lightgen_74k
tar -cf /scratch/dya78/lightgen/lightgen_74k_dit.tar \
    --transform 's,^,lightgen_74k/,' \
    emission_latents/emission_pbr_vae_ft_74k \
    shape_latents/shape_enc_next_dc_f16c32_fp16_256 \
    thumbnails \
    df_SomgProc_final.parquet \
    data_splits_emissive_74k_stratified_voxonly.json
ls -lh /scratch/dya78/lightgen/lightgen_74k_dit.tar
```
Expected: a single tar (tens of GB) whose members are prefixed `lightgen_74k/…`. (Adjust `thumbnails` to the dir name confirmed in Task 1 Step 3 if different.)

- [ ] **Step 2: Write the unpack script**

Create `script/fir/unpack_74k_dit_data.sh`:
```bash
#!/bin/bash
# Unpack the 74k DiT slice (encoded latents + thumbnails + metadata) on fir.
# Source: /scratch/dya78/lightgen/lightgen_74k_dit.tar
# Extracts to: ./lightgen_74k/ (relative to current directory)
#
# Contents (DiT-only — no voxels):
#   lightgen_74k/emission_latents/emission_pbr_vae_ft_74k/{*.npz, normalization_stats.json}
#   lightgen_74k/shape_latents/shape_enc_next_dc_f16c32_fp16_256/*.npz
#   lightgen_74k/thumbnails/*.png
#   lightgen_74k/df_SomgProc_final.parquet
#   lightgen_74k/data_splits_emissive_74k_stratified_voxonly.json
#
# Usage:  bash unpack_74k_dit_data.sh [path/to/lightgen_74k_dit.tar]
set -e

DEFAULT_TAR_SOURCE="/scratch/${USER}/lightgen/lightgen_74k_dit.tar"
TAR_SOURCE="${1:-${DEFAULT_TAR_SOURCE}}"
TARGET_DIR="."

if [ ! -f "${TAR_SOURCE}" ]; then
    echo "Error: Archive not found: ${TAR_SOURCE}"; exit 1
fi
echo "Source archive : ${TAR_SOURCE} ($(du -sh "${TAR_SOURCE}" | cut -f1))"

if [ -n "$SLURM_TMPDIR" ]; then
    TAR_FILE="${SLURM_TMPDIR}/lightgen_74k_dit.tar"
    if [[ "${TAR_SOURCE}" != ${SLURM_TMPDIR}/* ]]; then
        echo "Staging to local SSD: ${TAR_FILE}"
        rsync -ahP "${TAR_SOURCE}" "${TAR_FILE}"
    else
        TAR_FILE="${TAR_SOURCE}"
    fi
else
    TAR_FILE="${TAR_SOURCE}"
fi

echo "Extracting..."
tar -xf "${TAR_FILE}" -C "${TARGET_DIR}"

DATA_PATH="${TARGET_DIR}/lightgen_74k"
echo "=== Verifying ==="
EMIT=$(ls "${DATA_PATH}/emission_latents/emission_pbr_vae_ft_74k/"*.npz 2>/dev/null | wc -l)
SHAPE=$(ls "${DATA_PATH}/shape_latents/shape_enc_next_dc_f16c32_fp16_256/"*.npz 2>/dev/null | wc -l)
THUMB=$(ls "${DATA_PATH}/thumbnails/" 2>/dev/null | wc -l)
echo "Emission latents : ${EMIT}"
echo "Shape latents    : ${SHAPE}"
echo "Thumbnails       : ${THUMB}"
for f in "df_SomgProc_final.parquet" \
         "data_splits_emissive_74k_stratified_voxonly.json" \
         "emission_latents/emission_pbr_vae_ft_74k/normalization_stats.json"; do
    [ -f "${DATA_PATH}/${f}" ] && echo "  OK: ${f}" || echo "  MISSING: ${f}"
done
echo "Data ready at: $(cd "${DATA_PATH}" && pwd)"
```

- [ ] **Step 3: Lint the unpack script**

Run: `bash -n script/fir/unpack_74k_dit_data.sh`
Expected: exit 0.

- [ ] **Step 4: Smoke-test extraction (on fir, outside Slurm)**

Run (on fir, in a scratch temp dir):
```bash
cd /scratch/dya78/tmp_unpack_test 2>/dev/null || mkdir -p /scratch/dya78/tmp_unpack_test && cd /scratch/dya78/tmp_unpack_test
bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_dit_data.sh
```
Expected: prints Emission/Shape/Thumbnail counts all ≈73k and all four `OK:` lines. Then `rm -rf /scratch/dya78/tmp_unpack_test`.

- [ ] **Step 5: Commit the unpack script**

```bash
git add script/fir/unpack_74k_dit_data.sh
git commit -m "feat(emission-dit-74k): fir DiT-slice unpack script"
git push origin lightgen
```

---

## Task 5: Author the 74k DiT training launcher

**Files:**
- Create: `script/fir/emission_dit_74k.sh`

Clone `script/fir/finetune_emission_vae_74k.sh` (the proven 4×H100 DDP launcher with requeue/SIGTERM), swapping the config, the unpack script (DiT slice), and the data-path overrides.

- [ ] **Step 1: Write the training launcher**

Create `script/fir/emission_dit_74k.sh`:
```bash
#!/bin/bash
# Launch emission DiT 74k training on fir (4 H100). Channel-hijack: retarget
# the pretrained imgshape2tex texture DiT to predict emission latents.
#
# Usage:  bash script/fir/emission_dit_74k.sh

CONFIG="configs/gen/emission_dit_74k.yaml"
NAME="emission_dit_74k"
OUTPUT_DIR="/scratch/dya78/TRELLIS.2-lightning/outputs/${NAME}"
BRANCH="lightgen"

echo "Submitting job: ${NAME}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh fir sbatch); fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --mem=1024G
#SBATCH --account=rrg-msavva
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

handle_timeout() {
    echo "Timeout signal received (5 min remaining). Stopping training gracefully..."
    if [ ! -z "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        wait \$TRAIN_PID
    fi
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
}
trap 'handle_timeout' SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Started: \$(date)  Node: \$(hostname)"
echo "GPUs: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================"

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd /scratch/dya78/TRELLIS.2-lightning
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
source /scratch/dya78/TRELLIS.2-lightning/env/bin/activate

# Stage 74k DiT slice (latents + thumbnails) to local SSD.
cd \$SLURM_TMPDIR
bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_dit_data.sh
DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k

cd /scratch/dya78/TRELLIS.2-lightning
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}.log"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -o pipefail

srun python train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    --override dataset.args.data_root=\${DATA_ROOT} \
    --override dataset.args.parquet_file=\${DATA_ROOT}/df_SomgProc_final.parquet \
    --override dataset.args.split_json=\${DATA_ROOT}/data_splits_emissive_74k_stratified_voxonly.json \
    --override dataset.args.thumbnail_dir=\${DATA_ROOT}/thumbnails \
    2>&1 | tee "\${LOG}" &
TRAIN_PID=\$!
wait \$TRAIN_PID
EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo "Training crashed (exit code \$EXIT_CODE) at \$(date). Requeueing job..."
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
fi
echo "Training completed at \$(date)."
EOF

echo "Submitted: ${NAME}"
```

- [ ] **Step 2: Lint the launcher**

Run: `bash -n script/fir/emission_dit_74k.sh`
Expected: exit 0.

- [ ] **Step 3: Commit the launcher**

```bash
git add script/fir/emission_dit_74k.sh
git commit -m "feat(emission-dit-74k): fir 4xH100 DiT training launcher"
git push origin lightgen
```

---

## Task 6: Phase 3 — interactive probe (~100 steps, settle batch size)

**Files:** none (interactive on fir; may edit `batch_size_per_gpu` / `val_every_epochs` in the config afterward).

Per CLAUDE.md, any config that could OOM/skew DDP gets a ~100-step interactive probe before the full sbatch.

- [ ] **Step 1: Allocate an interactive 4×H100 (on fir, in a node-named tmux session)**

Per `cluster-access`, `salloc`/`srun --pty` a 4-GPU H100 node (3h is plenty). Inside the allocation, set `export SLURM_JOB_NAME=bash` so Lightning sees world_size>1 (per memory `feedback_slurm_job_name_bash_for_interactive_ddp`).

- [ ] **Step 2: Stage data + run a 100-step probe**

Run (inside the allocation):
```bash
cd /scratch/dya78/TRELLIS.2-lightning
git pull --ff-only origin lightgen
source env/bin/activate
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
cd $SLURM_TMPDIR && bash /scratch/dya78/TRELLIS.2-lightning/script/fir/unpack_74k_dit_data.sh
DATA_ROOT=$SLURM_TMPDIR/lightgen_74k
cd /scratch/dya78/TRELLIS.2-lightning
export SLURM_JOB_NAME=bash PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p log; TS=$(date +%Y%m%d-%H%M%S)
srun python train.py --config configs/gen/emission_dit_74k.yaml \
  --output_dir outputs/emission_dit_74k_probe --num_gpus 4 --ckpt none \
  --override trainer.args.max_steps=100 \
  --override dataset.args.data_root=$DATA_ROOT \
  --override dataset.args.parquet_file=$DATA_ROOT/df_SomgProc_final.parquet \
  --override dataset.args.split_json=$DATA_ROOT/data_splits_emissive_74k_stratified_voxonly.json \
  --override dataset.args.thumbnail_dir=$DATA_ROOT/thumbnails \
  2>&1 | tee "log/${TS}-probe_dit_74k_bs8.log"
```
Expected: reaches step 100 without OOM; the Lightning progress bar shows batches across all 4 ranks (e.g. `N/N … it/s`); a finite `train/loss` that is decreasing or stable (not NaN). Peak GPU memory well under 80 GB.

- [ ] **Step 3: (Optional) try BS=16**

If Step 2's peak memory leaves headroom, rerun with `--override trainer.args.batch_size_per_gpu=16`. Keep the largest BS that stays safely under ~70 GB peak.

- [ ] **Step 4: Lock the batch size + estimate val cadence**

From the probe's it/s, compute steps/epoch (≈73k / (4×BS)) and a `val_every_epochs` that fires at least once inside the 3h walltime window (per the VAE-run lesson — never let val fail to fire). If BS changed, edit `configs/gen/emission_dit_74k.yaml` accordingly.

- [ ] **Step 5: Commit any config change**

```bash
git add configs/gen/emission_dit_74k.yaml
git commit -m "tune(emission-dit-74k): batch size + val cadence from fir probe"
git push origin lightgen
```
(If nothing changed, skip the commit and note "probe confirmed BS=8 defaults; no change".)

- [ ] **Step 6: Release the allocation**

Exit the interactive shell (`--pty` releases on exit) or `scancel` if stuck; keep the tmux session for the next task.

---

## Task 7: Phase 4 — submit the full 74k DiT run

**Files:** none (submit + monitor on fir).

- [ ] **Step 1: Submit**

Run (on fir): `bash script/fir/emission_dit_74k.sh`
Expected: prints a job id; `squeue --me` shows `emission_dit_74k` queued/running.

- [ ] **Step 2: Confirm first checkpoint cycle**

Monitor `tail -f <node>-<jobid>.out` and the wandb dashboard (project `emission-dit`, run `emission_dit_74k`). Per CLAUDE.md, check logs roughly every ~10 min, not continuously.
Expected: training starts, `train/loss` logged every 50 steps; at the first val (`val_every_epochs`) a `step*-<val>.ckpt` plus `last.ckpt` appear under `outputs/emission_dit_74k/ckpts/`; the requeue/SIGTERM cycle re-enters the queue cleanly at the 2h55m mark and resumes from `last.ckpt`.

- [ ] **Step 3: Track to convergence**

Watch `val/loss` across requeue cycles until it plateaus (or `max_steps=50000`). Record the best-val checkpoint filename for eval. Early-stop if val saturates well before 50k.

---

## Task 8: Phase 5 — evaluate + visualize

**Files:**
- Create: `data_toolkit/test_emission_dit_74k.py`

Clone `data_toolkit/test_emission_dit_1k.py` and repoint it at the 74k artifacts. The body (DinoV3 encode → FlowEulerCfgSampler → VAE decode → per-sample L1 + max-projection PNG) is unchanged; only the header constants change.

- [ ] **Step 1: Create the eval script**

Copy `data_toolkit/test_emission_dit_1k.py` to `data_toolkit/test_emission_dit_74k.py` and replace the constants block (lines ~37–49) with:
```python
DATA_ROOT = "data/lightgen_74k"
DIT_CKPT  = "outputs/emission_dit_74k/ckpts/<BEST_VAL_CKPT_FROM_TASK7>.ckpt"
DIT_CFG   = "outputs/emission_dit_74k/config.yaml"
VAE_CKPT  = "outputs/emission_pbr_vae_ft_74k/ckpts/step0004485-0.0097.ckpt"
VAE_CFG   = "outputs/emission_pbr_vae_ft_74k/config.yaml"
SPLIT_JSON = f"{DATA_ROOT}/data_splits_emissive_74k_stratified_voxonly.json"
PARQUET    = f"{DATA_ROOT}/df_SomgProc_final.parquet"
EMISSION_LATENT_NAME = "emission_pbr_vae_ft_74k"
SHAPE_LATENT_NAME    = "shape_enc_next_dc_f16c32_fp16_256"
N_PER_SPLIT = 5
GUIDANCE = 3.0
SAMPLER_STEPS = 50
OUT_PNG = "emission_dit_74k_test.png"
device = "cuda"
```
Also update the figure `suptitle` string from "1k probe" to "74k".

- [ ] **Step 2: Sanity-check the script parses**

Run: `python -c "import ast; ast.parse(open('data_toolkit/test_emission_dit_74k.py').read())"`
Expected: exit 0 (no syntax error).

- [ ] **Step 3: Run eval on a held-out test/val slice (on fir, 1 GPU)**

This needs the 74k DiT slice + emission voxels available locally on the node. Run inside a single-GPU interactive allocation (or add a tiny eval sbatch mirroring Task 5's staging — staging both `unpack_74k_dit_data.sh` and `unpack_74k_data.sh` so emission voxels are present for the VAE-recon ceiling). Then:
```bash
python data_toolkit/test_emission_dit_74k.py
```
Expected: per-sample table printing `VAE L1`, `DiT L1`, `Lat L1`; `emission_dit_74k_test.png` written. DiT L1 should sit within ~1–3× the VAE-recon ceiling (per the 10-sample report's bar), with bright/dense-emission samples the hardest.

- [ ] **Step 4: 3D visualization**

Run `vis/vis_3d_dit.py` with `--experiment emission_dit_74k=outputs/emission_dit_74k/ckpts/<best>.ckpt` (and the VAE) to produce an interactive HTML; publish to `/project/3dlg-hcvc/lightgen/www/` per the web-publishing convention. Pull the PNG/HTML back to the workstation via rsync from inside the allocation (per memory `feedback_compute_node_rsync_to_local`).

- [ ] **Step 5: Commit the eval script + write the report**

```bash
git add data_toolkit/test_emission_dit_74k.py
git commit -m "feat(emission-dit-74k): held-out eval script (GT vs VAE-recon vs DiT)"
git push origin lightgen
```
Then write `reports/emission_dit_74k_status.md` (aggregate VAE/DiT/Lat L1, hardest samples, links to PNG/HTML), following the existing `reports/*.md` templates.

---

## Self-Review

**Spec coverage:**
- §2 architecture (64ch hijack, full reuse, skip_pbr_cond) → Task 3 config. ✓
- §3 Phase 0 verification → Task 1. ✓
- §3 Phase 1 encode + stats → Task 2. ✓
- §4 config + signed-off hyperparams (lr 1e-4, ~50k, EMA, save_top_k=1, balanced sampler, prod shape stats) → Task 3. ✓
- §5 probe → Task 6. ✓
- §6 fir launcher (module loads, requeue, no-cluster-edits, tee) → Tasks 2/5. ✓
- §7 eval + 3D vis + report → Task 8. ✓
- §9 open items (thumbnail dir, final VAE ckpt, val cadence) → Task 1 Steps 3–4, Task 6 Step 4. ✓
- Data staging (tar slice) — not in spec but required by fir's layout → Task 4. ✓

**Placeholder scan:** The only intentional fill-ins are (a) the emission mean/std arrays, produced by Task 2 and pasted in Task 3 (Step 3 greps to prove none remain), and (b) the best-val DiT ckpt filename, produced by Task 7 and used in Task 8 — both are genuine cross-task data dependencies with the exact producing command shown, not vague TODOs. The `<fill from Task1…>` comment tokens are documentation of confirmed paths, written during Task 3 Step 1.

**Type/name consistency:** `emission_pbr_vae_ft_74k` is used uniformly as both the VAE run name and the emission-latent dir name across Tasks 2/3/4/8. `shape_enc_next_dc_f16c32_fp16_256` and `data_splits_emissive_74k_stratified_voxonly.json` match the VAE config and `unpack_74k_data.sh`. Tar member prefix `lightgen_74k/` (Task 4 Step 1) matches the unpack verification paths (Task 4 Step 2).
