# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRELLIS.2-Lightning is a 4B-parameter 3D generative model using sparse voxel representations (O-Voxel) for image-to-3D generation with full PBR materials. The `lightgen` branch extends it to generate emissive textures for 3D objects.

## Environment Setup

```bash
# Full install (conda env "trellis2", Python 3.10, PyTorch 2.6.0 + CUDA 12.4)
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Rebuild just the O-Voxel C++ extension
cd o-voxel && pip install -e . && cd ..
```

## Training Commands

```bash
# Launch training (single example)
python train.py --config configs/scvae/emission_vae_10sample_overfit.yaml \
  --output_dir outputs/emission_vae_10sample_overfit \
  --data_dir '{"lightgen_test": {"base": "..."}}'

# Resume from checkpoint
python train.py --config <CONFIG> --output_dir <DIR> --ckpt last

# Override config values at runtime
python train.py --config <CONFIG> --output_dir <DIR> --override trainer.args.max_steps=5000

# Dry run (validate config without training)
python train.py --config <CONFIG> --output_dir <DIR> --tryrun

# Multi-GPU / multi-node
python train.py --config <CONFIG> --output_dir <DIR> --num_gpus 4 --num_nodes 2 --node_rank 0
```

### Run logging policy

Every training run must produce **two** logs in parallel:

1. **Stdout tee → `log/<TS>-<NAME>.log`** — full reproducibility trace (config dump, GPU info, stack traces). General pattern: `<TS>=$(date +%Y%m%d-%H%M%S)`, `<NAME>` is the config basename (or job tag for cluster runs).

   ```bash
   mkdir -p log
   TS=$(date +%Y%m%d-%H%M%S)
   CFG=<config_basename>          # e.g. matches configs/scvae/<CFG>.yaml
   python train.py --config "configs/scvae/${CFG}.yaml" \
     --output_dir "outputs/${CFG}" 2>&1 | tee "log/${TS}-${CFG}.log"
   ```

2. **Wandb is mandatory.** Every training config must enable wandb so scalar metrics, samples, and the resolved config end up in the dashboard. Set the following under `trainer.args:`:

   ```yaml
   trainer:
     args:
       wandb_enable: true
       wandb_project: <project>          # e.g. "emission-pbr-vae-finetune"
       wandb_name: <run_name>            # default; override per-submission
       # wandb_entity: <team>            # optional
       # wandb_log_every_steps: <N>      # optional, defaults to i_log
   ```

   Per-submission, override `wandb_name` so every job gets its own dashboard row, e.g. `--override trainer.args.wandb_name=${NAME}` in the launch script. The wandb logger is wired in `train.py:build_logger`; if `wandb_enable` is false (or missing) it silently falls back to `CSVLogger` and **no metrics reach wandb** — do not ship a config in that state.

   Authentication on a cluster node: `wandb login <key>` once, or `export WANDB_API_KEY=...`. To run without network: `export WANDB_MODE=offline` and `wandb sync` later.

Apply the `log/${TS}-<name>.log` tee pattern to data-prep scripts too (e.g. `log/${TS}-prep_split.log`) and inside cluster job bodies (see "Cluster launch scripts" below).

#### Logging-interval gotcha (small datasets)

Lightning flushes scalars every `log_every_n_steps` (= `i_log`, default `50` in `train.py`). If `total_train_batches_per_epoch < log_every_n_steps`, the run can finish having flushed **zero** scalars to wandb / the CSV. For tiny / overfit datasets, set `i_log` (or `wandb_log_every_steps`) low enough that at least one flush happens per epoch — typically `i_log: 1` for `<N>sample` smoke runs, `i_log: 10` for `<N>00sample`, `i_log: 50` for `<N>k`+.

### Cluster launch scripts (star2)

SBATCH wrappers live under `script/star2/<task>.sh` — one script per experiment family. They `ssh star2 sbatch <<EOF ... EOF` to submit to the cluster. **The SBATCH job body must use zsh** (the user's login shell on cluster nodes); the outer wrapper that runs `ssh ... sbatch` locally stays bash. Inside the heredoc:

- Shebang: `#!/bin/zsh`.
- `source ~/.zshrc` before `conda activate trellis2`.
- Pipe exit codes: zsh uses 1-indexed lowercase `${pipestatus[1]}` — **not** bash's `${PIPESTATUS[0]}` — when capturing `train.py`'s exit through a `tee` pipe. End the body with `exit \$TRAIN_EXIT` so Slurm records training failures as failures.
- Tee logs into `log/${TS}-${NAME}.log` per the Run Logs convention (the `mkdir -p log` + `TS=$(date +%Y%m%d-%H%M%S)` block lives inside the SBATCH body so the timestamp is the actual run start on the node).
- **Sync the launcher's commit to the cluster.** Never edit source files on the cluster; edit locally, `git push origin <branch>`, and put `git pull origin <branch>` in the sbatch body right after `cd /localscratch/dya78/TRELLIS.2-lightning`. This makes "what code ran" unambiguous and lets multiple collaborators share the cluster clone without racing edits.
- **Regenerate gitignored artifacts in the body.** Files under `data/` (e.g. `data_splits_*.json` produced by `data_toolkit/prep_*.py`) are gitignored and won't arrive via `git pull` — regenerate them in the body so a fresh cluster clone works end-to-end.

Naming convention: `script/star2/<task>.sh` ↔ job name / output dir `<task>[_<TAG>]` ↔ status report `reports/<task>_status.md` (see existing `reports/*.md` for templates). When adding a new launch script, copy the closest existing one under `script/star2/` rather than writing from scratch.

#### Interactive debug allocation (probe before sbatch)

**Decision rule:** any config change that could OOM or skew DDP (new batch size, sampler, dataset) gets a ~100-step interactive probe before the full sbatch. Sparse-voxel runs have rank-skewed memory and sbatch crashes burn hours of wall-clock — a probe pays for itself on the first avoided crash.

Allocation flow (see the cluster-access skill for full tmux/srun rationale; keep tmux **local**, not star2-side):

```bash
# From local workstation — session name = node name
tmux new-session -d -s cs-venus-15 \
  'ssh star2 -t "srun --nodelist=cs-venus-15 --gres=gpu:l40s:4 \
  --cpus-per-task=32 --mem=256G --time=8:00:00 --pty zsh -l"'
tmux a -t cs-venus-15
```

Once attached (inside the compute-node shell):
```bash
cd /localscratch/dya78/TRELLIS.2-lightning && conda activate trellis2
git pull --ff-only origin <branch>
TRAIN_SIZE=<N> python data_toolkit/prep_10sample_as1k.py   # regen gitignored splits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p log; TS=$(date +%Y%m%d-%H%M%S)
python train.py --config configs/scvae/<cfg>.yaml --output_dir outputs/<run> \
  --num_gpus 4 --ckpt last \
  --override trainer.args.batch_size_per_gpu=<probe_bs> \
  --override trainer.args.max_steps=<current_step+100> 2>&1 \
  | tee "log/${TS}-probe_bs<probe_bs>.log"
```

Probe passes if it reaches a full val epoch without OOM and with sane `val/loss`. With e.g. BS=8 + TRAIN_SIZE=640 + `val_every_epochs=5`, 100 steps = 5 epochs = 1 val epoch — exactly right. Then commit the config change and submit the full sbatch. When done, exit the compute shell (`--pty` releases on shell exit), or `scancel <jobid>` if it's stuck, then `tmux kill-session -t cs-venus-15`.

Gotcha: `nvidia-smi` via a second `srun --jobid=<id> --overlap` is scoped to the overlap step's cgroup GPU devices — that subset is what you see, *not* the parent training step's full GRES binding. Don't conclude "only 1 GPU is used" from it; the Lightning progress bar (`N/N ... it/s` where N matches `total_batches/rank × ranks`) is the authoritative DDP-across-all-ranks signal.

## Data Paths

- **Data root:** `data/lightgen_test/` — repo-relative path that works on every host. The 10-sample dataset (parquet, emission voxels, PBR voxels, latents, splits) lives here as real files. The `data/` directory is gitignored. To deploy on a new host, copy/rsync `data/lightgen_test/` along with the repo (e.g. `rsync -av data/ host:/path/to/repo/data/`).
- **74k subset (jupyter):** `/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/` (or `/cs/3dlg-jupiter-project/...` from the jupyter side). Holds `metadata.csv`, `df_SomgProc_final.parquet`, `data_splits_emissive_74k_pinned.json`, plus 8 `voxels/part<N>_voxels.tar` and 8 `latents/part<N>_latents.tar` produced by the prep pipeline (`script/star2/prep_74k_chain.sh`). Extract for training via `bash data_toolkit/extract_74k_tars.sh <jupyter_base> <local_dst>` — the resulting tree mirrors `data/lightgen_1k/` so existing dataset configs port over with one path swap. See `docs/superpowers/specs/2026-05-07-lightgen-74k-encoding-design.md` for the full pipeline design.
- **Visualization scripts:** `vis/` — unified 3D interactive: `vis/vis_3d_vae.py` (VAE reconstruction) and `vis/vis_3d_dit.py` (DiT → VAE decode). Both share helpers in `vis/vis_utils.py` and support `--experiment name=path` (repeatable) for side-by-side method comparison. Legacy per-case scripts live in `vis/legacy/` as reference. The 2D max-projection PNG scripts (`vis_emission_vae*.py`, `vis_vae_ckpt_comparison.py`, `compare_*.py`) remain at `vis/` root.

  **Known issue — GLB↔voxel scale misalignment:** The GLB panel and voxel panels don't bbox-align perfectly. Root cause: Blender's GLTF importer (used by `data_toolkit/blender_script/dump_pbr.py` to produce the pickle the voxelizer consumes) and the browser's GLTFLoader decode the same `.glb` file into geometrically different vertex sets (different vertex counts, different per-axis extents) — Blender applies import-time axis conversion, modifiers, and other transforms not exposed as toggles. The voxels come from Blender's view of the mesh; the browser renders trimesh/GLTFLoader's view. No single transform can reconcile them. Sample 0 happens to match closely (~0.003 on normalized axes); samples with more complex hierarchy show per-axis extent differences of 10–30% of the unit cube. Fix path (not yet done): during data prep, export a GLB from Blender at dump time (via `bpy.ops.export_scene.gltf`) as a sidecar file alongside each `.vxz`, then the viewer loads that "roundtrip" GLB instead of the 1K original — the two share Blender's exact vertex set.
- **Visualization output:** `visualizations/` — generated HTML/PNG files go here.
- **Web publishing:** copy HTML visualizations to `/project/3dlg-hcvc/lightgen/www/` for browser access.

## Data Preparation Pipeline

Run in order before training:
```bash
. ./data_toolkit/setup.sh                                  # setup
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT>  # metadata
python data_toolkit/dump_mesh.py <SUBSET> --root <ROOT>        # extract meshes
python data_toolkit/dump_pbr.py <SUBSET> --root <ROOT>         # extract PBR + emissive textures
python data_toolkit/voxelize_pbr.py ...                        # mesh → O-Voxel
python data_toolkit/encode_pbr_latent.py ...                   # voxel → latent
python data_toolkit/encode_emission_latent.py ...              # emission voxel → latent (lightgen)
```

## Testing

No automated test framework. Ad-hoc test/diagnostic scripts live in `data_toolkit/`:
```bash
python data_toolkit/test_emission_dit_10samples.py   # test DiT inference
python data_toolkit/test_emission_vae_e2e.py         # test VAE encode/decode
python data_toolkit/vis_emission_vae.py              # visualize VAE reconstructions
python data_toolkit/vis_3d_emission_vae.py           # 3D interactive HTML visualization
```

## Architecture

### Two-Stage Generation Pipeline

**Stage 1 — Sparse Compact VAE (SC-VAE):** Encodes 3D voxel attributes into compact latent spaces. Three separate VAEs handle shape, PBR materials, and emission respectively. All share the `SparseUnetVaeEncoder`/`Decoder` architecture (sparse 3D U-Net with residual blocks).

**Stage 2 — Flow Matching DiT:** Generates latents conditioned on input images. Uses `ElasticSLatFlowModel` — a sparse transformer with flow matching objective, DINOv3 image conditioning (cross-attention), and classifier-free guidance.

### Emission (LightGen) Specifics

Two distinct approaches are implemented for the emission VAE — pick the right one when picking a config/trainer.

#### PBR VAE finetune via channel hijack (`EmissionPbrFinetuneTrainer`)

Architecture is **byte-identical** to the production PBR VAE (`tex_{enc,dec}_next_dc_f16c32_fp16`): 6→6 input/output channels, 64→1024 model channels, latent_channels=32. **100% pretrained-weight reuse**, no input/output projection mismatch.

Trick (in `trellis2/trainers/vae/emission_pbr_finetune.py:_pack_emission_as_pbr`): pack emission RGB into the `base_color` slot and fill `metallic / roughness / alpha` with constants `(-1, +1, +1)` in [-1, 1] space. Loss is taken only on the first 3 output channels (emission); the trailing 3 channels of the decoder output are discarded.

I/O note: with the default `pbr_attrs: [base_color, metallic, roughness, alpha]`, `LightGenSLatEmission` reads `pbr_voxels_{res}/{sha}.vxz` for every sample, only for the trainer to overwrite those features with constants. Set `pbr_attrs: []` in the dataset args to skip the read — it's a no-op for the model.

Configs: `configs/scvae/emission_pbr_*.yaml`. Sub-patterns:
- `emission_pbr_vae_finetune_<N>sample.yaml` — small-dataset overfit setup (oversampled split + val + fp16 + EMA + AdaptiveGradClip).
- `emission_pbr_finetune_<N>k.yaml` — full-scale training on the corresponding `<N>k` dataset.

#### Two-stream concat VAE (`EmissionVaeTrainer`)

Two encoders: a trainable emission encoder (3→32-ch latent) plus a frozen pretrained PBR encoder (PBR→32-ch latent). The decoder receives `concat(z_emission, z_pbr)` = 64 channels along the feature axis, so its input projection is **reinitialized** (cannot reuse the PBR decoder's input weights). Output is 3-channel emission.

The emission DiT (downstream stage) similarly takes 96-channel input (32 emission + 32 PBR + 32 shape) and predicts 32-channel emission noise.

Key trainer: `EmissionVaeTrainer` (`trellis2/trainers/vae/emission_vae.py`) — splits input into emission/PBR, encodes emission, freezes PBR encoder via `pretrained_pbr_encoder: ...`, concatenates latents for decoder.

Configs: `configs/scvae/emission_vae_*.yaml` (overfit / arch-variant configs — currently the `*_<N>sample_overfit.yaml`, `*_overfit.yaml`, `*_option*_overfit.yaml` family; new variants follow the same `emission_vae_<descriptor>.yaml` shape).

### Tiny-dataset overfit setup (10 samples + 4 GPUs)

All trainers use `drop_last=True` on the train DataLoader. With `len(dataset) = 10` and any `num_gpus * batch_size_per_gpu > 10`, every rank gets ≤ batch_size samples and produces zero training batches — Lightning prints `Total length of DataLoader across ranks is zero` and `Trainer.fit stopped: No training batches.`

Fix: oversample the train indices via `data_toolkit/prep_10sample_as1k.py`, which writes `data/lightgen_test/data_splits_10sample_as1k.json` with a repeated train list (`TRAIN_SIZE` env var, default 120) and the 10-row val list. In the dataset args:

```yaml
dataset:
  args:
    split: train
    split_json: data/lightgen_test/data_splits_10sample_as1k.json
```

For validation on the same 10 samples, use **only** `val_split_name: val` in the trainer args (and **not** `val_split_file`). Background: `TrellisDataModule` injects `val_split_file` as the kwarg `split_file=...`, but `LightGenSLatEmission.__init__` accepts `split_json` / `split_indices` only — passing `val_split_file` will TypeError. With just `val_split_name: val`, the data module clones the train args (which already include `split_json`) and overrides `split` to `"val"`, which works.

`TRAIN_SIZE` **must** be an exact multiple of `num_gpus * batch_size_per_gpu` so `drop_last=True` drops **zero** rows and every epoch visits every position. A multiple that is merely "clean for step counts" but leaves a tail (e.g. 120 with global batch 64 → 56 dropped, ~47%) is a bug — with rotating shuffle (see `ResumableSampler.set_epoch`, fixed in e6df30c) the dropped rows differ each epoch, but you're still leaving half the intended oversample on the floor per epoch. For 10 unique samples × 4 GPUs × BS=16 (global 64), use `TRAIN_SIZE=640` (each sample visited 64× / epoch, zero drop). Regenerate in the sbatch body on each submission so it's always consistent with the current `num_gpus × batch_size_per_gpu`.

### Load balancing across DDP ranks (BalancedResumableSampler)

Sparse-voxel datasets have ~100× per-sample memory variance (emission `.vxz` files range 5–561 KB in the 10-sample split). Under `ResumableSampler` + DDP, a permutation that concentrates heavy samples on one rank will OOM that rank while the others sit comfortably. This was masked pre-e6df30c because the sampler never actually rotated the shuffle; once it rotated, rank 0 OOMed mid-run.

Fix is `BalancedResumableSampler` (in `trellis2/utils/data_utils.py`), which splits each super-batch (`batch_size * world_size`) into `world_size` equal-size sub-groups via `load_balanced_group_indices(loads, world_size, equal_size=True)`. It requires the dataset to expose `self.loads: List[int]` indexed in the same order as `self.samples`. Enable with `trainer.args.use_balanced_sampler: true`.

**Pick the right load proxy — voxel count, not file size.** Sparse activation memory scales with `coords.shape[0]`, not bytes on disk. `.vxz` files are compressed with variable ratios: in the 10-sample split, bytes/voxel varies 8.5× (0.12–1.02), Pearson correlation between file size and voxel count is only 0.70, and voxel-count range (11.5×) is far narrower than file-size range (101.5×). File-size balancing therefore under-weights high-voxel-count + high-compression samples. For small datasets read `coords.shape[0]` once per sample at `__init__` (via `o_voxel.io.read_vxz`; ~10 ms per sample); for large datasets precompute into the parquet/metadata (sibling `sparse_voxel_pbr.py` uses `num_pbr_voxels`, `structured_latent.py` uses `{latent_key}_tokens`).

- `LightGenSLatEmission` — exposes `self.loads`. First version used `.vxz` file size (fast but only 0.70 correlation with real memory); replace with voxel count for accurate balancing.
- `LightGenSLatEmissionLatent` — does **not** provide `loads` yet. Add it before any balanced DiT run (e.g. `tokens_per_sample` from the latent metadata).
- For any new sparse dataset, add `self.loads` up front using voxel/token count; the cost at init is negligible and it future-proofs DDP scaling.

**Load balancing alone doesn't solve an overprovisioned batch size.** Even with ideal balancing, if all ranks run at the VRAM edge (observed: ~44 GB / 45 GB L40S after rotation), *any* run-to-run memory spike (Triton autotuner, checkpoint recomputation) can OOM a rank. Balancing + right-sized `batch_size_per_gpu` are complementary, not substitutes.

### Key Source Locations

- `trellis2/models/sc_vaes/sparse_unet_vae.py` — VAE encoder/decoder architecture
- `trellis2/models/structured_latent_flow.py` — `SLatFlowModel`, `ElasticSLatFlowModel` (DiT)
- `trellis2/trainers/` — PyTorch Lightning trainer classes (VAE, flow matching, with mixins for image/text conditioning)
- `trellis2/datasets/lightgen_slat.py` — `LightGenSLatEmission` (raw voxels), `LightGenSLatEmissionLatent` (pre-encoded)
- `trellis2/modules/` — Low-level ops: sparse convolution, attention, positional embeddings
- `trellis2/pipelines/` — End-to-end inference pipelines (image-to-3D, texturing)
- `o-voxel/` — C++/CUDA extension for voxelization, serialization (.vxz format), and mesh export

### Configuration System

YAML configs in `configs/` define the full training setup (models, datasets, trainer args, optimizer, scheduler). The trainer is selected via `trainer.name` which maps to a class in `TRAINER_MAP` inside `train.py`. Config values can be overridden at runtime with `--override key.path=value`.

#### Emission VAE config naming convention

All configs live under `configs/scvae/`. Pick trainer by prefix:

| Pattern | Trainer | Approach |
|---|---|---|
| `emission_vae_*.yaml` | `EmissionVaeTrainer` | two-stream concat VAE (trainable emission encoder + frozen PBR encoder) |
| `emission_pbr_*.yaml` | `EmissionPbrFinetuneTrainer` | PBR VAE channel-hijack finetune |

Sub-token rules under `emission_pbr_*`:
- `emission_pbr_vae_finetune_*` — explicit "finetune of the pretrained PBR VAE" (the `_pbr_vae_` token).
- `emission_pbr_finetune_*` — same trainer, larger-scale variants (legacy name without the `_vae_` token).

Dataset-size suffix is part of the file stem: `_<N>sample` for tiny overfit (10, 100, …), `_<N>k` / `_<N>m` for full training. Output dir, job name, and log filename all reuse the same stem.

**Validation and wandb are both mandatory.** Every training config must define:

- A val loop (`val_split_name: <name>` in trainer args, plus a non-zero `limit_val_batches`). Configs without validation are not accepted — for tiny / overfit datasets, validate against the same samples as train (the "did the model memorize?" signal).
- Wandb logging (`wandb_enable: true` + `wandb_project` + `wandb_name`). See "Run logging policy" above for the full block.

When adding a new config, copy the val block **and** the wandb block from any existing `*_<N>sample*.yaml` overfit config in `configs/scvae/` as a template.

#### Checkpoint policy

Two modes, selected per-config in trainer args:

1. **Top-k by validation metric (preferred).** Set `save_top_k: <K>` (e.g. `3`). At the end of every validation epoch, the top K checkpoints by `monitor_metric` (default `val/loss`, `monitor_mode: min`) are kept under `outputs/<run>/ckpts/`, plus `last.ckpt` for resume. Filename format: `step{step:07d}-{val/loss:.4f}.ckpt`. Requires a working val loop. `i_save` is ignored in this mode.
2. **Step interval (legacy).** Set `i_save: <N>` and omit `save_top_k`. A checkpoint is written every N training steps with `save_top_k=-1` (keep all) and `last.ckpt` for resume.

Both modes are wired in `train.py` around the `ModelCheckpoint` construction (search for `save_top_k_cfg`). Convention: any new `configs/scvae/*_<N>sample*.yaml` overfit config uses top-k val mode by default; older / no-val configs (`limit_val_batches: 0`) cannot use top-k mode and must keep `i_save`.

### Sparse Tensor Convention

The codebase uses `SparseTensor` (from `trellis2/modules/sparse/`) throughout — a struct holding `feats` (feature tensor) and `coords` (integer voxel coordinates). Models operate on these rather than dense grids, enabling efficient processing of high-resolution (256-512) voxel data.
