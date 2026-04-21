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

The emission pipeline follows "Option B": the emission VAE decoder receives concatenated emission + PBR latents (64 channels), using a frozen pretrained PBR encoder to provide material context. The emission DiT takes 96-channel input (32 emission + 32 PBR + 32 shape) and predicts 32-channel emission noise.

Key trainer: `EmissionVaeTrainer` (`trellis2/trainers/vae/emission_vae.py`) — splits input into emission/PBR, encodes emission, freezes PBR encoder, concatenates latents for decoder.

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

### Sparse Tensor Convention

The codebase uses `SparseTensor` (from `trellis2/modules/sparse/`) throughout — a struct holding `feats` (feature tensor) and `coords` (integer voxel coordinates). Models operate on these rather than dense grids, enabling efficient processing of high-resolution (256-512) voxel data.
