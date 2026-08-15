# `emission_vae_pbr2emission_74k_v3` — status

Retraining the `pbr→emission` VAE so it matches the `albedo→emission` VAE apart from
input/output channels, and is actually finetuned from the pretrained PBR VAE.

Config: `configs/scvae/emission_vae_pbr2emission_74k_v3.yaml`
Launcher: `script/star2/emission_vae_pbr2emission_74k_v3_venus19.sh`

## Why

Audit 2026-08-14. The shipped `_v2` run (`step0154600-0.0088`) was never a finetune:

| | `albedo→emission` | `pbr→emission` `_v2` |
|---|---|---|
| trainable params | 828,601,398 | **128,634,491** (6.44× smaller) |
| `finetune_ckpt` in config | encoder + decoder | **absent** |
| what was pretrained | 284/284 tensors of both files, 0 shape mismatches | only the *frozen* `pretrained_pbr_encoder`, which sits off the emission path |
| emission encoder + decoder at step 0 | pretrained | **random init** |
| first logged `train/l1` | 0.00329 @ step 0 | 0.2957 @ step 9 |
| steps run | 34,800 (cosine tail) | 154,600 (constant LR) |

A missing `finetune_ckpt` key is a **silent no-op** — `_apply_finetune_checkpoint` reads
`getattr(self, "finetune_ckpt", None)` and returns without printing anything
(`trellis2/trainers/basic.py:403-409`). Nothing in the logs said the weights were random.

Adding the key to the `_v2` architecture would have been a measured no-op as well: against the
halved width/depth, only 1 of 284 encoder tensors (`to_latent.bias`, 64 params) and **0 of 284**
decoder tensors were shape-compatible. The architecture had to change too.

## What v3 is

Verified tensor by tensor against the `albedo→emission` config before any GPU time
(both models instantiated from their real config args and diffed):

| | result |
|---|---|
| parameter tensors with identical shape | **564** |
| tensors with different shape | **4** |
| tensors present in only one model | **0** |
| params | 828,633,779 vs 828,601,398 (**+0.0039%**) |

The 4 differences are all I/O boundaries, and all 4 are exactly the tensors the finetune loader
skips:

| tensor | pretrained | v3 |
|---|---|---|
| `encoder.input_layer.weight` | (64, 6) | (64, 3) |
| `decoder.from_latent.weight` | (1024, 32) | (1024, 64) |
| `decoder.output_layer.weight` | (6, 64) | (3, 64) |
| `decoder.output_layer.bias` | (6,) | (3,) |

`decoder.latent_channels` stays 64 deliberately: this decoder consumes
`cat(z_emission, z_pbr)` and can see PBR where the `albedo→emission` decoder cannot. That is the
variable this baseline exists to test.

Budget is equalized too — `max_steps` 160,000 → **35,800** with the `albedo→emission` chain's
`CosineAnnealingLR(T_max=10000, eta_min=1e-6)`. Fixing capacity while leaving a 4.44× step gap
would only trade a capacity confound for a budget one.

## Probe — venus-19 job 244547, 2026-08-15

100 steps, 2 × RTX Pro 6000 SE, `batch_size_per_gpu 32` = global 64 (the baseline's global
batch). `COMPLETED`, exit 0, 23:51 wall.

**Passed:**

1. **The finetune load fires and skips exactly the 4 predicted tensors** — encoder missing
   keys: 1, decoder missing keys: 3, the other 564 loaded.
2. **HF hub short-ids resolve against venus-19's offline cache.** This was a real risk: that
   node's HF cache was relocated to `/localscratch/dya78/hf_home` after the root FS filled.
   `[EmissionVaeTrainer] Loading frozen PBR encoder: microsoft/TRELLIS.2-4B/ckpts/...` confirms it.
3. **VRAM is not a constraint** — 19.4 GB and 15.9 GB of 96 GB per card at BS 32.

**Throughput:** 100 steps in 21:14 = **12.74 s/step** (0.22 it/s). 35,800 steps ⇒ **~5.3 days**
on 2 GPUs. Consistent with the `_v2` run's 1.49 s/step at 128 M once scaled for 6.44× params
plus the frozen 354 M encoder's forward pass.

**The probe exposed a real defect: v3 was cold-starting.** `train/l1`, i_log 10, so roughly
steps 9…99:

```
0.4840  0.1971  0.1309  0.0995  0.0866  0.1016  0.0528  0.0762  0.0790  0.0880
```

> **Two corrections to an earlier version of this section (2026-08-15 audit).**
> It claimed v3 "starts WORSE than from-scratch" by comparing 0.4840 against `_v2`'s 0.2957.
> **Unsupported** — a second from-scratch run of the same architecture logged **0.5071** at the
> same step, so 0.4840 sits inside the spread. It also cited `albedo→emission` at "0.00329 @
> step 0"; that number is **misattributed** to an April 1k-dataset run at `i_log: 1`, batch 8.
> The matched full-transfer anchor is **~0.033 at gstep 9** (two independent runs).
>
> The corrected finding is worse, not better: v3's curve tracks the from-scratch curve almost
> point for point (0.484/0.197/0.131/0.100/0.087 against 0.296/0.176/0.118/0.067/0.076). It was
> **not benefiting from its 99.992% inherited weights at all**.

Cause, established by a step-0 forward probe on real voxels (3 shapes / 49,660 voxels;
`data_toolkit/verify_pbr2emission_v3/`): the finetune loader **skips** shape-mismatched tensors rather than
adapting them (`basic.py:385-394`), and two of the four skipped tensors —
`encoder.input_layer.weight` and `decoder.from_latent.weight` — sit directly **on the
information path**. A random tensor at either end destroys the pretrained function no matter how
good the 564 interior tensors are.

| policy | step-0 `train/l1` |
|---|---|
| `albedo→emission` reference (full pretrained, 6ch) | 0.00488 |
| **v3 with raw hub files (skip)** | **1.77424** |
| **v3 with baked init** | **0.00488** |
| baked but without the `input_layer.bias` fold | 0.00663 |
| `from_latent` written to columns 32:64 instead of 0:32 | 0.67540 |

Trivial baselines on the same batch: predict −1 → 0.06835, predict 0 → 0.94108, predict
base_color → 0.67427. **The skip policy was worse than predicting a constant.** The baked init
matches the reference to 5e-8 — it is not merely close, it is the same function of emission RGB,
plus a zero-weight live-gradient path to `z_pbr`. The wrong-column arm landing within 0.17% of
"predict base_color" is exactly what the mechanism predicts, which is what makes the channel
mapping trustworthy rather than plausible.

## Resolution — baked init, not a trainer patch

`data_toolkit/bake_pbr2emission_v3_init.py` resolves all of it offline and writes checkpoints
whose shapes already match v3, so the trainer does a clean 284/284 load through **unmodified**
code. The config's two `finetune_ckpt` paths are the entire opt-in surface.

This was chosen over an opt-in trainer flag deliberately. Five DiT configs run `in_channels: 96`
against a pretrained `(1536, 64)` input projection whose columns genuinely correspond
(`[x_t | shape]` = 0:64), so a generic "adapt mismatched shapes" option would silently change
the init behind the **shipped 0.3326 (fir) / 0.3224 (venus-19)** numbers. A baked file cannot be
mis-set. It also expresses something a shape-remap flag cannot — the bias fold below.

| tensor | resolution |
|---|---|
| `encoder.input_layer.weight` (64,6)→(64,3) | pretrained columns 0:3 (the base_color slot, where the albedo chain packs emission) |
| `encoder.input_layer.bias` (64,)→(64,) | **a 5th tensor, not one of the mismatched four** — it loads cleanly today, but the dropped columns 3:6 saw the constant pad (−1,+1,+1), a fixed offset worth **1.77×** the entire bias (‖fold‖ 2.0147 vs ‖bias‖ 1.1381). Folded in; omitting it costs 0.00488 → 0.00663 |
| `decoder.from_latent.weight` (1024,32)→(1024,64) | pretrained block into columns **0:32** (emission is first in `cat([z_emission, z_pbr])`, `emission_vae.py:142`); columns 32:64 **zeroed** |
| `decoder.output_layer.weight` (6,64)→(3,64) | pretrained rows 0:3 |
| `decoder.output_layer.bias` (6,)→(3,) | pretrained rows 0:3 |

Independently re-verified after baking: all tensors fp16, exactly those 5 changed, the other 563
bit-identical, `from_latent` left half equal to pretrained and right half all zero,
`from_latent.bias` untouched, and `[verify] PASS` (284/284, 0 mismatches) against the config
built on `meta`.

Zeroing `from_latent[:, 32:64]` starts the **PBR-conditioning stream — the one capability this
baseline exists to test — at exactly zero contribution.** Gradients flow from step 1 (the
first-backward gradient L2 on the zeroed block is 74% of the pretrained block's), but "the
decoder learned to use `z_pbr`" is now an outcome to **verify, not assume**. Track
‖`from_latent.weight[:, 32:]`‖ / ‖`[:, :32]`‖ over training, or ablate `z_pbr` at eval.

## A second, independent defect found in the same pass — the LR schedule never annealed

`CosineAnnealingLR` is periodic with period 2·`T_max`. With `T_max: 10000` over a 35,800-step
run, lr returns to the full 1e-4 at step 20,000 and the run would **end at 6.28e-5 on a rising
limb**. No anneal at all.

`T_max: 10000` came from the `albedo→emission` chain, but there it governed a **~9k-step resumed
tail leg** after a constant-LR plateau, so its cosine stayed inside the first descending
half-period. The number was copied without its context, which inverted its meaning. Fixed to
`T_max: 35800`; verified monotone 1e-4 → 8.21e-5 → 4.14e-5 → 7.27e-6 → 1e-6.

Residual difference from the baseline, recorded rather than hidden: `albedo→emission`'s realized
trajectory was constant-LR then a short cosine tail; v3 runs one smooth full-length cosine.
Reproducing the two-leg trajectory exactly would take two jobs.

## Launch notes

- **Never resume a first launch.** `on_fit_start` returns before `_apply_finetune_checkpoint`
  when `resume_mode` is `legacy`/`lightning` (`basic.py:104-106`), so starting into a non-empty
  output dir silently skips the pretrained init — the same defect `_v2` had, by another route.
  `--ckpt last` is safe only because `resolve_ckpt_path` returns `None` on an empty dir
  (`train.py:640-655`). The launcher's `OUT_SUFFIX` keeps the probe out of the real output dir.
- **The launcher bakes the init in the job body.** `data/` is gitignored, so a cluster clone
  cannot receive the baked files via `git pull`; the bake is deterministic and takes seconds.
  It refuses to start training if `[verify]` does not pass.
- **`GLOBAL_BATCH` guard.** The launcher refuses to submit unless
  `NUM_GPUS × BATCH_SIZE == 64`, the `albedo→emission` baseline's global batch. Changing the GPU
  count without changing the micro-batch is the easiest way to silently invalidate the
  comparison. Default is 4 × 16 on all of venus-19.
- **EMA is inert — do not treat `ema_rate: 0.9999` as a safety net.** `apply_ema`
  (`callbacks.py:125`) has zero call sites, and `EmaCallback.on_fit_start` seeds its shadow
  weights *before* `BasicTrainer.on_fit_start` applies the finetune, i.e. from the random init.
- venus-19 runs offline wandb; sync afterwards with
  `wandb sync -p LightGen <run-dir>` and rename per `AGENTS.md`.

## Pre-launch gate

CPU-only, no GPU probe needed — the load is now an exact algebraic identity rather than an
approximation, so a 100-step A/B would only re-measure it.

1. `[verify] PASS` from the bake step.
2. In the training log: `grep 'shape mismatch'` returns **nothing**, and no `missing keys` line.
   (The old behaviour printed 4 mismatch warnings and two `missing keys` lines.)

## What remains genuinely open

Neither is answerable by a cheap probe; both are answered by the real run.

- **Does the better init produce a better converged VAE at 35,800 steps?** All the evidence
  above is step-0 forward-only. "Patched v3 finishes better" is a well-supported *inference*,
  not a measurement. The target to beat is the shipped `step0154600-0.0088`.
- **Does the decoder actually learn to use `z_pbr`,** or does it degenerate into
  `albedo→emission` with an extra encoder? Watch the column-norm ratio described above.
