# Emission PBR Conditional VAE — Design

**Date:** 2026-05-07
**Branch:** `lightgen`
**Status:** Draft, awaiting implementation plan

## Goal

Replace the current "channel-hijack" emission VAE (`EmissionPbrFinetuneTrainer`) with a **conditional VAE** whose encoder ingests emission only and whose decoder is explicitly conditioned on shape + PBR (albedo, metallic, roughness, alpha) latents — mirroring the channel-concat conditioning pattern used by the production PBR DiT and the emission DiT.

The surgery starts from the pretrained PBR VAE (`tex_{enc,dec}_next_dc_f16c32_fp16`) and modifies three projection layers; all interior ConvNeXt and downsample/upsample blocks remain bit-identical to the pretrained model.

### Why a third pathway

Two existing pathways frame the design space:

| Pathway | Encoder I/O | Decoder I/O | Pretrained weight reuse | Conditioning |
| --- | --- | --- | --- | --- |
| Channel-hijack (`EmissionPbrFinetuneTrainer`) | 6 → 32 | 32 → 6 (only first 3 used) | 100% (no surgery) | None — PBR slots padded with constants |
| Two-stream concat (`EmissionVaeTrainer`) | 3 → 32 (smaller arch) | 64 → 3 (`from_latent` re-init) | Limited — fresh 32→512 backbone | PBR latent via live frozen encoder |
| **Conditional (this design)** | **3 → 32 (surgical input_layer)** | **96 → 3 (surgical from_latent + output_layer)** | **High — entire interior + most of input/output layers preserved** | **Shape + PBR latents via channel-concat at `from_latent`** |

The conditional pathway combines the strengths of both:
- It keeps the production-scale 64→1024-channel backbone (vs. the 32→512 Option B fresh init).
- It encodes only emission — no constants polluting the encoder input distribution.
- It exposes shape + PBR as first-class conditioning signals, matching the DiT's input contract.

## Architecture

### Forward pass

```
emission_voxels [N, 3]
        │
        ▼
encoder.input_layer  (SparseLinear 3 → 64, surgical)
        │
        ▼
encoder.blocks.*     (unchanged from pretrained PBR VAE)
        │
        ▼
encoder.to_latent    (unchanged)
        │
        ▼
z_emit [M, 32]   ───┐
                    │
pbr_latent_pre [M, 32]  (loaded from disk)  ───┤
                    │
shape_latent_pre [M, 32] (loaded from disk) ───┤
                    ▼
        cat([z_emit, pbr_z, shape_z], dim=-1)   [M, 96]
                    │
                    ▼
decoder.from_latent  (SparseLinear 96 → 1024, surgical)
                    │
                    ▼
decoder.blocks.*     (unchanged from pretrained PBR VAE)
                    │
                    ▼
decoder.output_layer (SparseLinear 1024 → 3, surgical)
                    │
                    ▼
emission_recon [N, 3]
```

### Surgical layer table

| Layer | Pretrained shape | New shape | Init strategy |
| --- | --- | --- | --- |
| `encoder.input_layer` | `SparseLinear(6, 64)` | `SparseLinear(3, 64)` | Copy `pretrained.weight[:, 0:3]` (base_color cols) and `pretrained.bias`. Same channels emission lives in for the channel-hijack approach, so the encoder starts in a workable regime. |
| `decoder.from_latent` | `SparseLinear(32, 1024)` | `SparseLinear(96, 1024)` | `weight[:, 0:32] = pretrained.weight` (full pretrained), `weight[:, 32:96] = 0`, `bias = pretrained.bias`. At step 0 the decoder behaves **identically** to the pretrained decoder on the emission latent — the conditioning path is a no-op until training opens it. |
| `decoder.output_layer` | `SparseLinear(1024, 6)` | `SparseLinear(1024, 3)` | Copy `pretrained.weight[0:3, :]` (base_color rows) and `pretrained.bias[0:3]`. Decoder still produces a sensible base_color-like output on step 0. |

All other modules (`encoder.blocks.*`, `decoder.blocks.*`, `decoder.to_subdiv` heads inside C2S blocks, `encoder.to_latent`, all `LayerNorm32` and `SparseConv3d` weights inside blocks) load unchanged from the pretrained PBR VAE.

### Coordinate alignment invariant

The emission encoder, the pre-encoded PBR latent, and the pre-encoded shape latent must share the same downsampled `(coords, M)` per sample. This holds because:
- Both encoders are run on the same sparse voxel structure produced by the data prep pipeline (same `coords` at input resolution).
- All three encoders use identical downsample architectures (same number of stages, same downsample ratios), so they emit latents at the same downsampled grid.

The trainer asserts `z_emit.coords == pbr_z.coords == shape_z.coords` at every step, matching the existing assertion pattern in `EmissionVaeTrainer` and `LightGenSLatEmissionLatent`.

### Shape conditioning rationale

Even though shape is implicitly carried by the shared sparse voxel structure, including an explicit shape latent matches the emission DiT's input contract (`concat_cond = cat([shape_z, pbr_z])`). This is the more conservative choice: it gives the VAE the same conditioning signal the downstream DiT uses, so the VAE's reconstruction objective is consistent with the DiT's generation objective.

## Components

### New trainer: `EmissionPbrConditionalVaeTrainer`

File: `trellis2/trainers/vae/emission_pbr_conditional.py`.

Subclass of `BasicTrainer`. Key methods:

- `__init__(loss_type, lambda_kl, emission_channels=3, pbr_latent_channels=32, shape_latent_channels=32, stage_a_steps=1000, lr_interior_stage_b, ...)`
- `prepare_dataloader` — identical to `EmissionVaeTrainer` (uses `BalancedResumableSampler` over `LightGenSLatEmission.loads`).
- `training_losses(x, pbr_latent, shape_latent, **kwargs)` — runs the trainable encoder on `x.feats[:, :3]` (emission), concatenates `[z_emit, pbr_latent, shape_latent]` along the feature axis, runs the trainable decoder, computes L1 on the 3-channel output + `lambda_kl` × KL on the emission posterior.
- `on_train_batch_start` (Lightning hook) — checks `self.global_step == self.stage_a_steps`; on hit calls `_apply_stage_b()` which (a) flips `requires_grad=True` on previously frozen modules and (b) appends a new param group `{'params': interior_params, 'lr': lr_interior_stage_b}` to the optimizer.
- `on_train_start` — if resuming past `stage_a_steps`, invoke `_apply_stage_b()` once before training so the optimizer/grad state matches the saved step.
- `run_snapshot` — mirrors `EmissionVaeTrainer.run_snapshot` (mean-absolute emission reconstruction error on a held-out batch).

No frozen encoder is loaded by the trainer (PBR + shape are pre-encoded on disk).

### New surgery helper

File: `trellis2/utils/model_surgery.py`.

Function: `surgically_init_emission_pbr_cvae(encoder, decoder, pretrained_path) -> Tuple[encoder, decoder]`.

Steps:
1. `trellis_models.from_pretrained(pretrained_path + '/' + enc_name)` → `pretrained_encoder`.
2. `trellis_models.from_pretrained(pretrained_path + '/' + dec_name)` → `pretrained_decoder`.
3. Build a state-dict for `encoder` and `decoder` that:
   - Copies all `blocks.*`, `to_latent.*`, `to_subdiv.*` keys 1:1 from the corresponding pretrained model.
   - Re-shapes the three surgical layers per the init-strategy table above.
4. `encoder.load_state_dict(new_sd_enc, strict=True)`, `decoder.load_state_dict(new_sd_dec, strict=True)`.
5. Return `(encoder, decoder)`.

Invoked from `train.py` when the YAML contains `trainer.args.finetune_ckpt: {pretrained_pbr: <path>}`. The existing `finetune_ckpt: {encoder: ..., decoder: ...}` path used by `EmissionPbrFinetuneTrainer` is untouched.

### Dataset extension: `LightGenSLatEmission`

File: `trellis2/datasets/lightgen_slat.py`.

Add three optional constructor args mirroring `LightGenSLatEmissionLatent`:
- `pbr_latent_name: Optional[str]` — subfolder under `pbr_latents/`.
- `shape_latent_name: Optional[str]` — subfolder under `shape_latents/`.
- `pbr_latent_root`, `shape_latent_root` — optional roots (default to `data_root`).

When both are set, `__getitem__` additionally loads `pbr_latents/{name}/{sha256}.npz` and `shape_latents/{name}/{sha256}.npz` (using the existing `_read_npz_as_sparse` helper) and returns them in the output dict as `pbr_latent` and `shape_latent` (both `SparseTensor`). The trainer's `forward_args` are extended so Lightning's batch dispatch passes them through to `training_losses`.

The collate function is updated to `sparse_cat` the new fields per super-batch, the same way `concat_cond` is handled in `LightGenSLatEmissionLatent.collate_fn`.

**Normalization.** PBR / shape latents on disk are stored normalized (the values used by the DiT). For the VAE we leave them **as-is (normalized)**. Reason: the new `from_latent` cols `[32:96]` are zero-init, so the decoder is invariant to the conditioning distribution at step 0; it learns whatever distribution we feed it during training. Normalized inputs sit in a more numerically stable range, and we avoid duplicating the dataset's normalization stats into the VAE config.

### Configs

**`configs/scvae/emission_pbr_cvae_10sample.yaml`** — 10-sample overfit smoke test.

Mirrors `emission_pbr_vae_finetune_10sample.yaml` for hyperparams (BS=8, lr=1e-4, EMA=0.9999, AdaptiveGradClip, fp16=`inflat_all`, val_every_epochs=1, save_top_k=3, wandb on). Diffs:

- `trainer.name: EmissionPbrConditionalVaeTrainer`
- `trainer.args.finetune_ckpt: {pretrained_pbr: microsoft/TRELLIS.2-4B/ckpts}` (single pointer; the helper resolves both `tex_enc_next_dc_f16c32_fp16` and `tex_dec_next_dc_f16c32_fp16` underneath).
- `trainer.args.stage_a_steps: 1000`
- `trainer.args.lr_interior_stage_b: 3e-5`
- `dataset.args.pbr_latent_name: tex_enc_next_dc_f16c32_fp16_256`
- `dataset.args.shape_latent_name: shape_enc_next_dc_f16c32_fp16_256`
- `wandb_project: emission-pbr-cvae`

**`configs/scvae/emission_pbr_cvae_1k.yaml`** — 1k full run, apples-to-apples with `emission_pbr_vae_finetune_1k.yaml`.

Same diffs as above, plus:
- `dataset.args.parquet_file: data/lightgen_1k/df_SomgProc_emission_filtered.parquet`
- `dataset.args.split_json: data/lightgen_1k/data_splits_emission_filtered.json`
- `max_steps: 40000`, `val_every_epochs: 5`, `i_log: 1` (matches the channel-hijack 1k for direct wandb overlay).

### Cluster launch scripts

- `script/star2/emission_pbr_cvae_10sample.sh` — copy of `script/star2/emission_pbr_vae_finetune_10sample.sh`, with run name / config path swapped. Regenerates `data_splits_10sample_as1k.json` in the sbatch body (`TRAIN_SIZE=640`).
- `script/star2/emission_pbr_cvae_1k.sh` — copy of `script/star2/emission_pbr_vae_finetune_1k.sh`, with run name / config path swapped. Pulls the 1k split from origin per the existing convention.

## Training schedule

Both configs use the same two-stage schedule:

**Stage A — Surgery warmup (steps 0 → `stage_a_steps = 1000`)**

| Item | Value |
| --- | --- |
| Frozen modules | `encoder.blocks.*`, `decoder.blocks.*` (all interior ConvNeXt + downsample/upsample blocks). Pretrained PBR + shape encoders are *not loaded* (we read pre-encoded latents from disk) — there is no "frozen encoder" module in the trainer. |
| Trainable modules | `encoder.input_layer`, `encoder.to_latent`, `decoder.from_latent`, `decoder.output_layer` (plus any `decoder.to_subdiv` heads that live *inside* `blocks.*` — those follow the interior freeze) |
| Optimizer | AdamW, single param group |
| Peak LR | `1e-4` |
| LR schedule | Linear warmup 0 → 1e-4 over the first 500 steps, then constant |
| Loss | `L1(emission) + 1e-7 × KL` (identical to Option B and the channel-hijack baseline) |

**Stage B — Full finetune (steps `stage_a_steps` → `max_steps`)**

| Item | Value |
| --- | --- |
| Frozen modules | None inside the encoder/decoder. (Pre-encoded conditioning latents are inherently fixed.) |
| Trainable modules | Everything |
| Optimizer | AdamW, **two param groups**: surgical layers @ `lr=1e-4`, interior blocks @ `lr=3e-5` |
| LR schedule | Constant for both groups (both already warm; no second warmup) |

**Stage transition mechanism**

Trainer's `on_train_batch_start(batch, batch_idx)`:
```python
if self.global_step == self.stage_a_steps and not self._stage_b_applied:
    self._apply_stage_b()
```

`_apply_stage_b()`:
1. Flip `requires_grad=True` on every parameter in the previously frozen interior modules.
2. Append a new param group `{'params': interior_params, 'lr': self.lr_interior_stage_b}` to the existing optimizer (AdamW preserves moments for the existing group; new params start with fresh moments).
3. Set `self._stage_b_applied = True`.
4. Log `train/stage = B` for the rest of the run.

**Resume from checkpoint.** Lightning restores `self.global_step` via the trainer state dict. The trainer's `on_train_start` hook checks `if self.global_step >= self.stage_a_steps and not self._stage_b_applied: self._apply_stage_b()` so a resume past the boundary lands in Stage B with the right optimizer geometry.

**Wandb logging.** Per step: `train/l1`, `train/kl`, `train/loss`, `train/lr_surgical`, `train/lr_interior` (NaN during Stage A), `train/stage` (`'A'` or `'B'`). Per validation epoch: `val/l1`, `val/loss`. Keys deliberately match the channel-hijack and Option B trainers so all three pathways overlay cleanly in one wandb dashboard.

## Acceptance criteria

### 10-sample smoke run

- `train/loss < 0.01` within 5000 steps. Matches the channel-hijack 10-sample sibling's overfit signal.
- `val/loss` within ~10% of `train/loss` (10-sample val = 10-sample train).
- `train/stage` transitions A → B exactly once, at step 1000.
- No OOM on 4× L40S with `batch_size_per_gpu = 8`.

If `train/loss` plateaus above ~0.05 the surgery + staging is harming convergence relative to the channel-hijack baseline; bisect by (a) disabling Stage A (`stage_a_steps: 0`) on the same dataset and (b) re-running with shape conditioning dropped (`from_latent: 64 → 1024`).

### 1k apples-to-apples comparison

Run `emission_pbr_cvae_1k` and `emission_pbr_vae_finetune_1k` on the same 1k dataset with identical BS, schedule, EMA, grad clip, and val frequency. Plot `val/loss` for both on the same wandb dashboard. The conditional VAE is a clear win if its `val/loss` curve sits below the channel-hijack curve from step ~5000 onward and the ratio widens through the run (i.e., conditioning helps more as both models close the optimization gap).

## What this design does **not** do

- **No CFG / classifier-free guidance** at the VAE level. CFG only matters for the DiT.
- **No live frozen encoder** at training time. PBR + shape latents are read from disk. (If the chosen subset doesn't have pre-encoded PBR/shape latents available, that's a data-prep prerequisite — the DiT already needs them, so they should exist for any subset reaching this stage.)
- **No cross-attention** in the VAE decoder. The decoder is a sparse U-Net; we keep its architecture entirely intact other than the three projection-layer surgeries.
- **No image conditioning** at the VAE level. Image conditioning lives in the DiT (DINOv3 cross-attention). The VAE is an autoencoder; its job is to compress emission given material/structure context.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Stage A trains too few params (only the 3 surgical layers + `to_latent`) and underfits the conditioning path before Stage B opens | `from_latent[:, 32:96]` *is* trainable in Stage A, so the conditioning path can begin to open during Stage A. The zero-init means the conditioning gradient signal is small initially (the cols are at zero, so the loss prefers to first move the warm-started emission slot before scaling up the conditioning slot). Stage A's intent is "let the new I/O regime settle without perturbing the pretrained interior." If Stage A is too short for the conditioning to start meaningfully, val/loss at the A→B boundary will dip sharply once the interior opens — visible as a step on the wandb curve and corrected by lengthening `stage_a_steps`. |
| Resume-past-stage-boundary loses optimizer-state continuity for the surgical layers | The first param group (surgical) is preserved across the stage transition, so its AdamW moments survive resume. Only the newly-added interior group has fresh moments, which is expected behavior for any unfreeze-and-resume pattern. |
| Normalized conditioning latents have a different distribution than the pretrained `from_latent[:, 0:32]` expects | The zero-init of `from_latent[:, 32:96]` means the conditioning has *no* effect at step 0 regardless of its distribution. The decoder learns to use whatever distribution we feed. |
| Data-prep prerequisite: PBR + shape latents must exist for the 10-sample and 1k subsets | The 1k subset's latents are already produced for the DiT (per the existing `emission_dit_*.yaml` configs that consume `pbr_latents/tex_enc_next_dc_f16c32_fp16_256` and `shape_latents/shape_enc_next_dc_f16c32_fp16_256`). The 10-sample subset must be re-encoded if not already present — implementation plan should verify this in step 1. |

## File summary

| Path | Status | Purpose |
| --- | --- | --- |
| `trellis2/trainers/vae/emission_pbr_conditional.py` | New | `EmissionPbrConditionalVaeTrainer` |
| `trellis2/utils/model_surgery.py` | New | `surgically_init_emission_pbr_cvae` |
| `trellis2/datasets/lightgen_slat.py` | Modify | Add optional `pbr_latent_name`, `shape_latent_name` to `LightGenSLatEmission`; extend `collate_fn` |
| `trellis2/trainers/__init__.py` / `train.py` `TRAINER_MAP` | Modify | Register `EmissionPbrConditionalVaeTrainer` |
| `train.py` | Modify | Detect `finetune_ckpt: {pretrained_pbr: <path>}` and call surgery helper before training |
| `configs/scvae/emission_pbr_cvae_10sample.yaml` | New | 10-sample overfit smoke |
| `configs/scvae/emission_pbr_cvae_1k.yaml` | New | 1k apples-to-apples run |
| `script/star2/emission_pbr_cvae_10sample.sh` | New | Cluster launch |
| `script/star2/emission_pbr_cvae_1k.sh` | New | Cluster launch |
