# Emission DiT — 74k Training (Channel-Hijack)

**Date:** 2026-06-15
**Branch:** `lightgen`
**Author:** Claude (with @dya78)

## 1. Problem & goal

The emission **VAE** finetune is done: `emission_pbr_vae_ft_74k` (channel-hijack
of the production PBR VAE — emission RGB packed into the `base_color` slot,
`metallic/roughness/alpha` set to constants). The best-val checkpoint is
`outputs/emission_pbr_vae_ft_74k/ckpts/step0004485-0.0097.ckpt` (val/loss 0.0097).

What remains is **Stage 2**: train the emission **DiT** to generate emission
latents from an input image (+ shape), at full 74k scale, on fir (H100).

Status going in:
- **10-sample DiT overfit:** passed (`reports/emission_dit_10sample_overfit_report.md`)
  — pipeline is correctly wired end-to-end (image → emission latent → VAE decode).
- **1k DiT probe:** never converged — only `step0000046-0.3118.ckpt` exists. No
  usable 1k DiT.
- **74k DiT:** not started. The 74k emission latents have **not** been encoded
  with the finished VAE yet — that is the real prerequisite.

Decision (settled with @dya78): **go straight to 74k** (the 10-sample overfit
already proved the wiring; skip a 1k milestone).

## 2. Approach — DiT channel-hijack (mirror of the VAE)

The VAE retargets the production PBR VAE to emission by reinterpreting the
`base_color` slot. The DiT does the analogous thing: retarget the pretrained
**`slat_flow_imgshape2tex_dit_1_3B_512_bf16`** texture DiT to predict **emission
latents** instead of PBR latents, with **zero architecture change**.

Why this is a pure retarget (not a re-architecture):
- The base texture DiT predicts a single **32ch joint latent** (`out_channels:
  32`) that the VAE decodes into all 6 PBR attributes together. There is no
  per-attribute channel in latent space — so there is nothing to "swap"
  channel-by-channel. We just point the same 32ch-latent predictor at the
  emission-latent distribution.
- The metallic/roughness/alpha "other channels" only exist in the VAE's 6ch
  attribute space (constants in, discarded out). They are baked *inside* the
  32ch emission latent by the encoder; they never surface as DiT inputs/outputs.

### Architecture (byte-identical to the base)

```
64ch input = [ 32ch  noised EMISSION latent ]   ← prediction target
           + [ 32ch  SHAPE latent (concat)  ]   ← geometry conditioning
        ( + image via DinoV3 cross-attention, not concat )
out_channels = 32  → emission latent → emission_pbr_vae_ft_74k decoder → emission voxels
```

- `in_channels: 64`, `skip_pbr_cond: true` — **no PBR-latent stream** (the 32
  PBR channels of the rejected 96ch option are dropped, matching the hijack
  philosophy: don't add a material stream).
- `finetune_ckpt.denoiser: microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16`
  → **100% weight reuse including the input projection** (64==64, no reinit,
  unlike the 10-sample 96ch run which reinitialized the input layer).
- Inference stays a clean **2-stage chain**: `image (+shape) → emission`. No
  dependency on a PBR DiT.

The bet this encodes: emission is inferable from **image + shape** alone, with
no material/PBR latent at inference. Consistent with the VAE story.

## 3. Pipeline (phases)

```
Phase 0  Verify fir state — 74k tree has emission_voxels_256, shape_latents,
         thumbnails/images, parquet, split_json; confirm final VAE ckpt.
Phase 1  Encode 74k emission latents with the finished VAE → normalization stats.   [GPU, sharded]
Phase 2  Author configs/gen/emission_dit_74k.yaml + fir launcher.
Phase 3  Interactive ~100-step probe on fir (settle batch size; sane val/loss; DDP×4).
Phase 4  Full 74k DiT sbatch on fir (4×H100).
Phase 5  Eval + 3D/2D visualization vs GT on the held-out test split.
```

### Phase 0 — verification (on fir)

Confirm, in the fir 74k tree (`<fir_repo>/data/lightgen_74k/`):
- `emission_voxels_256/{sha}.vxz` (VAE-prep output; encoder input for Phase 1)
- `shape_latents/shape_enc_next_dc_f16c32_fp16_256/{sha}.npz` (already present)
- thumbnails/images for DinoV3 conditioning — **confirm directory name + that
  PNGs exist**; this is the one artifact not exercised by VAE training.
- `df_SomgProc_final.parquet`, `data_splits_emissive_74k_stratified_voxonly.json`
- the final VAE ckpt + its sibling `config.yaml`. Confirm
  `step0004485-0.0097.ckpt` is the intended final model (the run targeted
  20k steps but the best-val ckpt is at step 4485 — verify nothing newer/better
  exists on fir).

### Phase 1 — encode 74k emission latents (the prerequisite)

`data_toolkit/encode_emission_pbrfinetune_dataset.py` is already parameterized
and shard-aware (`--rank/--world_size`, `--aggregate_stats` on rank 0). Run it
across the 4 H100s:

```
python data_toolkit/encode_emission_pbrfinetune_dataset.py \
    --data_root data/lightgen_74k \
    --parquet   data/lightgen_74k/df_SomgProc_final.parquet \
    --vae_ckpt  outputs/emission_pbr_vae_ft_74k/ckpts/step0004485-0.0097.ckpt \
    --vae_cfg   outputs/emission_pbr_vae_ft_74k/config.yaml \
    --latent_name emission_pbr_vae_ft_74k \
    --resolution 256 \
    --rank <r> --world_size 4   # + --aggregate_stats on rank 0
```

- Output: `data/lightgen_74k/emission_latents/emission_pbr_vae_ft_74k/{sha}.npz`
  + `normalization_stats.json` (mean/std for the DiT config).
- **Encoder weights: raw (non-EMA)** — what the script already loads, consistent
  with how the VAE reconstruction was validated. (EMA-encode is an optional
  later refinement; if adopted, the inference decoder must use EMA too.)
- Shape latents already exist from VAE prep — **no shape re-encode**.

## 4. Phase 2 — the 74k DiT config

`configs/gen/emission_dit_74k.yaml`, cloned from `emission_dit_1k_probe.yaml`
(already the correct 64ch shape). Changes vs. that probe:

**Model** — unchanged (matches base `imgshape2tex` DiT exactly).

**Dataset** (`LightGenSLatEmissionLatent`, 74k):
- `data_root: data/lightgen_74k`, `parquet_file:
  data/lightgen_74k/df_SomgProc_final.parquet`
- `emission_latent_name: emission_pbr_vae_ft_74k`, `skip_pbr_cond: true`
- `shape_latent_name: shape_enc_next_dc_f16c32_fp16_256`
- `split: train`, `split_json:
  data/lightgen_74k/data_splits_emissive_74k_stratified_voxonly.json` (same
  split the VAE used)
- `thumbnail_dir:` 74k image/thumbnail dir (confirm name in Phase 0)
- `image_size: 512`, `resolution: 256`
- `emission_normalization:` ← Phase 1 `normalization_stats.json`
- **`shape_normalization:` ← the base DiT's production `shape_slat_normalization`**
  (from `slat_flow_imgshape2tex_dit_1_3B_512_bf16.yaml`), **not** the
  10-sample-derived stats the probe carried (that was a probe shortcut).

**Trainer** (`ImageConditionedSparseFlowMatchingCFGTrainer`) — fir/Lustre
lessons from the VAE 74k run baked in:
- `save_top_k: 1` — DiT ckpts are ~15 GB; top-k=3 blocked training 10+ min/val
  on Lustre.
- `val_every_epochs` sized so val fires inside a walltime/requeue cycle (the VAE
  run wasted ~4 days because val never fired within a cycle and `last.ckpt`
  froze). One 74k epoch ≈ 2,300 steps at global batch 32.
- `use_balanced_sampler: true` — `LightGenSLatEmissionLatent` exposes
  `self.loads` (per-sample token count from `coords.shape[0]`), so balanced DDP
  works out of the box. (CLAUDE.md's note that it "does not provide loads yet"
  is stale — code lines 459–473 populate it.)
- `val_split_name: val`, top-k val checkpointing (`monitor_metric: val/loss`,
  `monitor_mode: min`).
- `wandb_enable: true`, `wandb_project: emission-dit`, `wandb_name:
  emission_dit_74k`.
- `image_cond_model: DinoV3FeatureExtractor` (vitl16, image_size 512, frozen).

### Hyperparameter decisions (signed off)

| Decision | Value | Rationale |
|---|---|---|
| Learning rate | `1e-4` + 1k linear warmup | base's rate; PBR→emission target shift is large enough to want the full rate (vs. 5e-5 probe) |
| Training budget | `~50k steps` (~22 epochs @ global batch 32), val-gated top-k, early-stop on plateau | finetune from a strong base needs far fewer than the base's 1M; shorten to ~30k if val saturates |
| EMA | `ema_rate: [0.9999]` | base default; reliably improves sample quality (probe omitted it) |
| Batch size | settle in Phase 3 probe (start 8/H100, try 16) | emission latents are token-sparse → good headroom on 80 GB |
| `p_uncond` | `0.1` | CFG, standard |
| VAE ckpt (encoding) | `step0004485-0.0097.ckpt`, raw (non-EMA) encoder | best-val; consistent with recon validation |

## 5. Phase 3 — interactive probe (fir)

Per the CLAUDE.md decision rule (any change that could OOM/skew DDP gets a
~100-step interactive probe before sbatch): allocate an interactive 4×H100,
`export SLURM_JOB_NAME=bash` (so Lightning sees world_size>1), run ~100 steps at
the candidate batch size. Probe passes if it reaches a full val epoch without
OOM, with sane `val/loss`, and the Lightning progress bar shows DDP across all 4
ranks. Settle BS (8 → try 16), then commit the config and submit the full sbatch.

## 6. Phase 4 — full 74k DiT sbatch (fir)

fir launcher modeled on the existing 74k VAE fir launcher:
- fir env: `module load gcc arrow opencv` + `pip install diffusers` (per
  `project_fir_deployment` memory); not pip wheels for arrow/opencv.
- Edit locally → `git push` → launcher body `git pull` (no cluster edits).
- Regenerate gitignored splits in the body if needed.
- Tee logs to `log/${TS}-emission_dit_74k.log`; requeue + wandb-run-id
  persistence (already in code, commit ce9fe9e).
- `exit $TRAIN_EXIT` so Slurm records training failures.

## 7. Phase 5 — eval & visualization

- Adapt `data_toolkit/test_emission_dit_1k.py` → 74k held-out **test** split:
  per-sample L1 of `decode(DiT_sample)` vs GT, against the VAE-reconstruction
  ceiling (`decode(encode(GT))`), plus latent-space L1.
- `vis/vis_3d_dit.py` for qualitative 3D (DiT → VAE decode), with
  `--experiment` side-by-side vs GT; publish HTML to
  `/project/3dlg-hcvc/lightgen/www/`.
- Watch the hard cases (bright/dense-emission samples) the 10-sample report
  flagged as lagging the VAE ceiling.

## 8. Out of scope

- Two-stream concat VAE / DiT (Option B) — separate thread.
- 96ch PBR-conditioned DiT — explicitly rejected (adds an inference dependency;
  contradicts the hijack philosophy).
- Re-encoding shape/PBR latents — already present in the 74k tree.

## 9. Open items to resolve during implementation

1. **Phase 0 must verify the 74k emission voxels are non-zero** before encoding.
   Per memory `project_74k_zero_emission_root_cause`, a broken cs-venus `o_voxel`
   `_C.so` once produced all-zero `emission_voxels`; encoding those would silently
   train the DiT to all-black. The VAE's val/loss 0.0097 + visualized recon
   suggest the fixed voxels are in place, but verify directly (spot-check
   `emissive max > 0` on a sample) — 0/N non-zero ⇒ regenerate voxels first.
2. **Phase 0 must confirm the 74k thumbnail/image directory** exists with PNGs —
   the only DiT input not exercised by VAE training.
3. **Confirm `step0004485-0.0097.ckpt` is the run's final/best** on fir (run
   targeted 20k steps; best-val is step 4485).
4. **`val_every_epochs` calibration** against the actual fir walltime cycle once
   we know steps/sec from the probe.
