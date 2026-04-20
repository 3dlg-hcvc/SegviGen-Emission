# v15 PBR finetune (10-sample-as-1k, 2×L40S on cs-venus-15) — OOM status

Script: [script/star2/finetune_pbr_10sample_as1k_2gpu_v15.sh](../script/star2/finetune_pbr_10sample_as1k_2gpu_v15.sh)
Config: [configs/scvae/emission_pbr_finetune_10sample_as1k.yaml](../configs/scvae/emission_pbr_finetune_10sample_as1k.yaml)
Node: `cs-venus-15` (2× L40S, 44 GB each)
Logs: `star2:/project/3dlg-hcvc/lightgen/probe/emission_pbr_ft_10s_as1k_v15_*.out`
Outputs (per-node, only visible from venus-15): `/localscratch/dya78/TRELLIS.2-lightning/outputs/emission_pbr_ft_10s_as1k_v15_*`

## TL;DR

As-submitted (default `BS=64`), the script **does not start training** — it OOMs on the encoder's first forward pass. `BS=32` and `BS=32` with `BATCH_SPLIT=4` also OOM. Only `BS=8` and `BS=4` actually start training; both were manually cancelled after ~1 h, so no run has ever reached completion.

## Per-run status (all Apr 16)

| TAG | Outcome | Notes |
|---|---|---|
| `bs64` (default) | **CUDA OOM** | rank 0, 44 GB L40S fully consumed (~43.3 GB in use, 634 MiB free) during first encoder forward; tried to allocate 730 MiB. Exit 1. No output dir written. |
| `bs32` | **CUDA OOM** | rank 1, ~40.9 GB in use, tried to allocate 3.81 GiB. Exit 1. |
| `bs32_split4` | **CUDA OOM** | rank 0, 43.8 GB in use, tried to allocate 540 MiB — even with `BATCH_SPLIT=4`. Exit 1. |
| `bs8` | Started OK, CANCELLED | Steady ~36 GB/GPU. Cancelled at 1 h 02 min. |
| `bs4` | Started OK, CANCELLED | Steady ~30 GB/GPU. Reached step 2000, saved `last.ckpt` + `stepstep=0002000.ckpt` (~13 GB each) in `…_v15_bs4/ckpts/`. Cancelled at 1 h 01 min. |

Probe variants (`probe_bs4`, `probe_w0`, `probe_w4`) all COMPLETED short runs — those succeed.

## Where the OOM happens

Traceback common to all OOM runs — inside the VAE encoder's residual MLP under gradient checkpointing, during the **first training step**:

```
trellis2/trainers/vae/emission_pbr_finetune.py:93  training_losses
  → self.training_models["encoder"](…)
trellis2/models/sc_vaes/sparse_unet_vae.py:398  forward
  → h = block(h)
trellis2/models/sc_vaes/sparse_unet_vae.py:312  forward
  → torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
trellis2/models/sc_vaes/sparse_unet_vae.py:306  _forward
  → h = h.replace(self.mlp(h.feats))      # SiLU here raises OOM
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set in the script — so fragmentation is not the primary culprit.

## What's confirmed on venus-15 `/localscratch`

Output dirs present (via tiny debug allocation):
```
emission_pbr_ft_10s_as1k_v15_bs4/           # bs=4, has ckpts (step 2000)
emission_pbr_ft_10s_as1k_v15_max20k_test/
emission_pbr_ft_10s_as1k_v15_probe_bs4/
emission_pbr_ft_10s_as1k_v15_probe_w0/
emission_pbr_ft_10s_as1k_v15_probe_w4/
v15_max20k_test/
```
No `…_v15_bs64` / `_v15_bs32` dirs — the OOM kills the process before Lightning writes anything.

## Open questions / debug hypotheses

1. Why does **BS=64 per GPU × 2 GPU** OOM at VAE encoder-only, when this was the "unmodified 1k config" that presumably ran fine elsewhere? Check what the 1k reference training was run on (H100? A100 80 GB? likely the assumption behind BS=64).
2. Activation memory inside `SparseUnetVaeEncoder` scales with per-sample sparse voxel count — with the 10-sample-as-1k dataset, check whether a few samples are unusually dense and blowing the budget.
3. `BATCH_SPLIT=4` should have ~quartered the activation memory vs. `BS=32` plain, but still OOM'd at essentially the same point — investigate whether `batch_split` is actually being honored in the encoder path (grep [trellis2/trainers/vae/emission_pbr_finetune.py](../trellis2/trainers/vae/emission_pbr_finetune.py) for split handling).
4. Gradient checkpointing is on (`torch.utils.checkpoint.checkpoint` at line 312) but OOM still happens *inside* the checkpointed region — so the issue is a single-forward activation, not the retained activations between blocks.

## Safe starting point to reproduce/debug

```bash
# On star2 (from cs-star2, after syncing code to venus-15):
BS=4 bash script/star2/finetune_pbr_10sample_as1k_2gpu_v15.sh
# — this is the only BS that has actually started training and saved checkpoints.
```

For interactive debug on venus-15:
```bash
ssh star2
salloc -A 3dlg-hcvc-lab -p 3dlg-hcvc-lab-debug -w cs-venus-15 \
  --gres=gpu:l40s:2 --cpus-per-task=32 --mem=128G --time=2:00:00
# then ssh cs-venus-15 -p24 and run training manually with BS overrides
```

Last touched: 2026-04-20
