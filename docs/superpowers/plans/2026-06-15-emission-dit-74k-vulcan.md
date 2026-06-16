# Emission DiT 74k on vulcan (L40S) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.
>
> **This is a compute-target port** of `docs/superpowers/plans/2026-06-15-emission-dit-74k.md`. The DiT design, the config (`configs/gen/emission_dit_74k.yaml`), and the eval (`data_toolkit/test_emission_dit_74k.py`) are unchanged. Only the cluster moves fir → vulcan because fir is down. Spec: `docs/superpowers/specs/2026-06-15-emission-dit-74k-design.md`.
>
> **Cluster note:** vulcan is 2FA tmux-only (like fir). Drive a local tmux session named `vulcan` (`tmux send-keys` + `tmux capture-pane`). Never `ssh vulcan` directly from Claude Code. Never edit code on the cluster — edit locally, `git push origin lightgen`; the launcher syncs on the **login node** before submit.

**Goal:** Run the emission DiT 74k channel-hijack training on vulcan (L40S) instead of fir, starting single-node and deciding multi-node after a probe measures L40S it/s.

**Why a separate plan:** vulcan has four hard differences from fir that change the launchers (not the model): no compute-node outbound SSH, `module` needs an explicit cvmfs source, `aip-msavva` account / L40S partitions, and a `> >(tee)` SIGTERM pattern.

---

## Reused as-is (no new files)
- `configs/gen/emission_dit_74k.yaml` — cluster-agnostic; the launcher overrides `data_root`/`parquet_file`/`split_json`/`thumbnail_dir` to `$SLURM_TMPDIR`. BS=8 (L40S 48 GB; matches the 1k probe that ran on star2 L40S).
- `script/fir/unpack_74k_data.sh` — voxel staging (the vulcan VAE launcher already calls it directly).
- `script/fir/unpack_74k_dit_data.sh` — DiT-slice staging (reads `/scratch/$USER/lightgen/lightgen_74k_dit.tar`; path-identical on vulcan).
- `data_toolkit/test_emission_dit_74k.py` — eval.

## New files (authored locally, no vulcan needed)
| File | Responsibility |
|---|---|
| `script/vulcan/encode_emission_74k.sh` | Phase 1 on vulcan: login-node git sync, cvmfs+module, single-node 4×L40S sharded encode, rsync latents+stats back to vulcan scratch. |
| `script/vulcan/emission_dit_74k.sh` | Training launcher cloned from `script/vulcan/finetune_emission_vae_74k.sh` (inherits all 4 vulcan gotchas), pointed at the DiT config + DiT-slice unpack. `NODES=1` default; `NODES`/`BS` env knobs for scale-up. |

## The four vulcan deltas baked into both launchers
1. **No compute-node outbound SSH** → sync code on the **login node** before `sbatch` (`run_on_vulcan "git fetch && checkout && pull"`); the job body only `cd`s in and logs the commit. No in-body `git pull`.
2. **`module` not on PATH** under `#!/bin/bash` sbatch → `source /cvmfs/soft.computecanada.ca/config/profile/bash.sh` before `module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash`.
3. **Account/partition/mem:** `--account=aip-msavva`, `--gpus-per-node=l40s:4`, `--mem=0`, `--time` within `gpubase_bynode_b2` (12 h).
4. **`> >(tee "$LOG") 2>&1 &`** (process substitution, not `| tee`) so `$TRAIN_PID=$!` is srun's PID and the SIGTERM trap can forward to ranks for graceful checkpoint-on-requeue.

---

## Task V0: Phase 0v — verify vulcan state + stage missing artifacts

**Files:** none (interactive on vulcan tmux + local rsync).

The encode (Phase 1) needs the emission VAE ckpt on vulcan; the DiT needs `shape_latents` + `thumbnails`. `log/20260605-*-rsync_voxels_to_vulcan.log` shows voxels reached vulcan Jun 5, but the VAE ckpt is from Jun 9 (almost certainly not yet there) and shape_latents/thumbnails are unconfirmed.

- [ ] **Step 1: Verify env + data on vulcan**

In the `vulcan` tmux session (login node):
```bash
ls -d /scratch/dya78/TRELLIS.2-lightning/env/bin/activate          # venv built?
ls /scratch/dya78/lightgen/lightgen_74k/voxels/part{0..7}_voxels.tar # voxels (Jun 5)
ls /scratch/dya78/lightgen/lightgen_74k/df_SomgProc_final.parquet
ls /scratch/dya78/lightgen/lightgen_74k/data_splits_emissive_74k_stratified_voxonly.json
ls /scratch/dya78/lightgen/lightgen_74k/shape_latents/shape_enc_next_dc_f16c32_fp16_256/*.npz 2>/dev/null | wc -l
ls /scratch/dya78/lightgen/lightgen_74k/thumbnails/*.png 2>/dev/null | wc -l
ls -la /scratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_vae_ft_74k/ckpts/ 2>/dev/null
```
Record which of {venv, shape_latents, thumbnails, VAE ckpt} are present vs. missing.

- [ ] **Step 2: If venv missing — build it**

Follow `cluster-access` → `fir/env-setup.md` recipe (identical on vulcan; account `aip-msavva`): virtualenv at `env/`, `module load gcc arrow opencv`, `pip install --no-index` Alliance wheels + `diffusers`, then project `setup.sh`. Skip if Step 1 shows `env/bin/activate` exists.

- [ ] **Step 3: Stage the emission VAE ckpt to vulcan (from local — fir is down)**

The VAE ckpt + its config live on the local workstation. rsync to vulcan scratch (per memory `feedback_sync_large_files` — large artifacts via rsync, never git):
```bash
# From the local workstation:
rsync -ahP --info=progress2 \
  outputs/emission_pbr_vae_ft_74k/ckpts/step0004485-0.0097.ckpt \
  outputs/emission_pbr_vae_ft_74k/config.yaml \
  vulcan:/scratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_vae_ft_74k/ckpts/
# (config.yaml goes one dir up — adjust the dest so config.yaml lands in
#  outputs/emission_pbr_vae_ft_74k/config.yaml, ckpt in .../ckpts/)
```
Note: vulcan rsync from local goes through the login node (outbound works there). If a 2FA prompt appears, the user types it in the tmux session (never store the password).

- [ ] **Step 4: Stage shape_latents + thumbnails if missing**

If Step 1 shows either absent, rsync them from local (or from wherever the authoritative copy is) into `/scratch/dya78/lightgen/lightgen_74k/{shape_latents,thumbnails}/`. These are required DiT inputs.

- [ ] **Step 5: Verify emission voxels are NON-ZERO**

Same gate as the fir plan (memory `project_74k_zero_emission_root_cause`): spot-check `emissive max > 0` on ~20 staged `.vxz` before encoding. 0/20 ⇒ regenerate voxels first.

---

## Task V1: Author `script/vulcan/encode_emission_74k.sh`

**Files:**
- Create: `script/vulcan/encode_emission_74k.sh`

Clone the *structure* of `script/fir/encode_emission_74k.sh` (sharded 4-GPU encode + per-rank wait + PIPESTATUS guard + rsync-back — keep those correctness fixes) but apply the four vulcan deltas: login-node `run_on_vulcan` git sync before submit, `source /cvmfs/...` before `module load`, `--account=aip-msavva --gpus-per-node=l40s:4 --mem=0`, single node `--nodes=1 --ntasks-per-node=1 --cpus-per-task=48`. No in-body git pull. VAE ckpt/cfg at the vulcan paths staged in Task V0.

- [ ] **Step 1: Write the launcher** (full content provided at authoring time; mirror `finetune_emission_vae_74k.sh`'s login-node-sync preamble + `encode_emission_74k.sh`'s rank loop).
- [ ] **Step 2:** `bash -n script/vulcan/encode_emission_74k.sh` → exit 0.
- [ ] **Step 3:** Render the node-side heredoc body (stub the submitter to `cat`) and `bash -n` it — the outer lint doesn't check inside the heredoc.
- [ ] **Step 4:** Independent Codex review of heredoc escaping (login-node vs node-time vars), rank loop, PIPESTATUS guard.
- [ ] **Step 5:** Commit + `git push origin lightgen`.

---

## Task V2: Author `script/vulcan/emission_dit_74k.sh`

**Files:**
- Create: `script/vulcan/emission_dit_74k.sh`

Clone `script/vulcan/finetune_emission_vae_74k.sh` verbatim, then change: `CONFIG=configs/gen/emission_dit_74k.yaml`; `NAME=emission_dit_74k_vulcan`; default `NODES=1`, `BS=8`; data staging switches from `unpack_74k_data.sh` (voxels) to `script/fir/unpack_74k_dit_data.sh` (DiT slice); add `--override dataset.args.thumbnail_dir=\${DATA_ROOT}/thumbnails` to the srun train command (the VAE launcher omits it; the DiT needs images).

- [ ] **Step 1: Write the launcher** (full content at authoring time).
- [ ] **Step 2:** `bash -n` + render-and-lint the node-side body.
- [ ] **Step 3:** Codex review (focus: the `> >(tee)` SIGTERM pattern survived the edits, thumbnail override present, single-node staging works for NODES=1).
- [ ] **Step 4:** Commit + push.

---

## Task V3: Run Phase 1 encode on vulcan

- [ ] **Step 1:** `bash script/vulcan/encode_emission_74k.sh` (from the `vulcan` tmux login node, or from local).
- [ ] **Step 2:** Monitor `squeue --me` + the `%x-%j.out` log (check ~every 10 min).
- [ ] **Step 3:** Verify: `ls /scratch/dya78/lightgen/lightgen_74k/emission_latents/emission_pbr_vae_ft_74k/*.npz | wc -l` ≈ shape-latent count; `normalization_stats.json` has 32-len mean/std. Copy the mean/std.

---

## Task V4: Finalize config + build DiT-slice tar

- [ ] **Step 1:** Paste the emission `mean`/`std` from Task V3 into `configs/gen/emission_dit_74k.yaml` (the 2 `REPLACE_WITH_32_VALUES_FROM_TASK2` markers). `grep REPLACE_WITH` → none. Commit + push.
- [ ] **Step 2:** Build the DiT-slice tar on vulcan (same command as the fir plan Task 4 Step 1, on vulcan scratch): tar emission_latents + shape_latents + thumbnails + parquet + split into `/scratch/dya78/lightgen/lightgen_74k_dit.tar` with member prefix `lightgen_74k/`.

---

## Task V5: Single-node probe → decide node count

- [ ] **Step 1:** Submit a single-node short run: `NODES=1 bash script/vulcan/emission_dit_74k.sh` with `--override trainer.args.max_steps=<curr+100>` (or an interactive `srun --pty` in the tmux session; `export SLURM_JOB_NAME=bash` for DDP world_size>1 per memory `feedback_slurm_job_name_bash_for_interactive_ddp`).
- [ ] **Step 2:** Confirm no OOM at BS=8 on L40S (48 GB), DDP across 4 ranks, sane `train/loss`, and record **it/s**.
- [ ] **Step 3:** Estimate full-run wall-clock (50k steps / it/s ÷ 4 ranks). **Decide single-node vs multi-node** (`NODES=N`) for the full run; recalibrate `val_every_epochs` to fire within the 12 h `b2` walltime cycle. Commit any config/BS change.

---

## Task V6: Full 74k DiT run on vulcan

- [ ] **Step 1:** `NODES=<decided> bash script/vulcan/emission_dit_74k.sh`.
- [ ] **Step 2:** Track `val/loss` across requeue cycles to plateau / `max_steps`. Record best-val ckpt.

---

## Task V7: Eval + report

Same as the fir plan Task 8: set `DIT_CKPT` in `data_toolkit/test_emission_dit_74k.py` to the best-val ckpt, run eval (1 L40S) + 3D vis, write `reports/emission_dit_74k_status.md`. Note the cluster was vulcan in the report.

---

## Self-Review

**Coverage vs. fir plan:** every fir task has a vulcan analog — Phase 0 verify+gate (V0), encode (V1+V3), config finalize (V4), probe (V5), train (V6), eval (V7). New vs. fir: V0 ckpt/shape/thumbnail staging (vulcan didn't have the fir-trained VAE), and the two vulcan launchers (V1/V2) replacing the fir ones. The DiT-slice tar build moved into V4.

**Vulcan-delta coverage:** all four deltas (login-node sync, cvmfs source, account/partition/mem, process-substitution tee) are enumerated once and referenced by both launcher tasks.

**Open items:** (1) Task V0 must confirm/stage the VAE ckpt, shape_latents, thumbnails — the biggest unknown is whether they're already on vulcan. (2) Node count decided empirically in V5. (3) Non-zero emission-voxel gate carried over from the fir plan.
