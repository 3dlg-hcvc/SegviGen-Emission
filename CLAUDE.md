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
- Pipe exit codes: zsh uses 1-indexed lowercase `${pipestatus[1]}` — **not** bash's `${PIPESTATUS[0]}` — when capturing `train.py`'s exit through a `tee` pipe.
- Tee logs into `log/${TS}-${NAME}.log` per the Run Logs convention (the `mkdir -p log` + `TS=$(date +%Y%m%d-%H%M%S)` block lives inside the SBATCH body so the timestamp is the actual run start on the node).

Naming convention: `script/star2/<task>.sh` ↔ job name / output dir `<task>[_<TAG>]` ↔ status report `reports/<task>_status.md` (see existing `reports/*.md` for templates). When adding a new launch script, copy the closest existing one under `script/star2/` rather than writing from scratch.

## Data Paths

- **Data root:** `data/lightgen_test/` — repo-relative path that works on every host. The 10-sample dataset (parquet, emission voxels, PBR voxels, latents, splits) lives here as real files. The `data/` directory is gitignored. To deploy on a new host, copy/rsync `data/lightgen_test/` along with the repo (e.g. `rsync -av data/ host:/path/to/repo/data/`).
- **Visualization scripts:** `vis/` — all visualization scripts live here. Use these as templates for new visualizations.
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

Pick `TRAIN_SIZE` to be a multiple of `num_gpus * batch_size_per_gpu` for clean step counts (e.g. `TRAIN_SIZE=128` for 4×8=32 global batch → 4 steps/epoch).

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
