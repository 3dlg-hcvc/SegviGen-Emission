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

**Open question — the early loss is worse than from-scratch.** `train/l1`, i_log 10, so roughly
steps 9…99:

```
0.4840  0.1971  0.1309  0.0995  0.0866  0.1016  0.0528  0.0762  0.0790  0.0880
```

v3 starts at **0.4840** where the fully-random `_v2` logged **0.2957**, despite inheriting
99.992% of its weights. Working hypothesis: `encoder.input_layer` and `decoder.from_latent` are
both on the information path and both random, so the pretrained interior receives a feature
distribution it was never trained on. The `albedo→emission` chain does not have this problem —
its `input_layer` is 6→6 and loads intact, which is why it can log 0.00329 at step 0.

Under audit as of 2026-08-15: whether to partially transfer those tensors (pretrained
`input_layer` columns for the base_color slot; pretrained `from_latent` into columns 0:32 with
32:64 zeroed) instead of dropping them wholesale. Any such change must be opt-in — the DiT
configs deliberately reinitialize a mismatched 64→96 input projection and must not be altered.

## Launch notes

- **Never resume a first launch.** `on_fit_start` returns before `_apply_finetune_checkpoint`
  when `resume_mode` is `legacy`/`lightning` (`basic.py:104-106`), so starting into a non-empty
  output dir silently skips the pretrained init — the same defect `_v2` had, by another route.
  `--ckpt last` is safe only because `resolve_ckpt_path` returns `None` on an empty dir
  (`train.py:640-655`). The launcher's `OUT_SUFFIX` keeps the probe out of the real output dir.
- venus-19 runs offline wandb; sync afterwards with
  `wandb sync -p LightGen <run-dir>` and rename per `AGENTS.md`.
