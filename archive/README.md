# Archived LightGen experiments (TRELLIS2)

Superseded configs, scripts, and diagnostics moved here during the 2026-07-09
cleanup. Active training/eval stays under `configs/`, `script/`, `data_toolkit/`,
and `vis/`.

## Layout

| Path | Why archived |
|------|----------------|
| `configs/scvae/emission_pbr_cvae_*.yaml` | CVAE experiment configs (channel-hijack / two-stream won) |
| `script/star2/emission_pbr_cvae_*.sh` | Matching CVAE launchers |
| `data_toolkit/test_emission_pbr_cvae_*.py` | CVAE smoke tests |
| `configs/scvae/emission_vae_{overfit,1sample,500ep,10kep,lowlr}.yaml` | Duplicate VAE overfit variants (kept `emission_vae_10sample_overfit.yaml`) |
| `configs/gen/emission_dit_{overfit,full_overfit,10sample_overfit_pbrvae_as1k}.yaml` | Duplicate DiT overfit variants (kept `emission_dit_10sample_overfit.yaml` + `_pbrvae.yaml`) |
| `datasets/lightgen_uv.py` | Unused UV-space LightGen dataset (voxel `lightgen_slat` is active) |
| `vis/legacy/` | Pre-consolidation per-case 3D viewers |
| `vis/compare_*.py`, `vis_vae_ckpt_comparison.py`, `vis_emission_vae_1sample.py` | One-off diagnostics |

## Still active (not archived)

- Channel-hijack + two-stream VAE/DiT configs and fir scripts
- CVAE *trainer* code (`trellis2/trainers/vae/emission_pbr_conditional.py`) — kept for reference; configs/scripts only moved
- Blender 74k prep scripts (`script/star2/prep_74k_*`) — deferred until code-native latents replace them
- Canonical checkpoints under `outputs/` (gitignored): see parent `CLEANUP_PLAN.md` Phase 4

## Restoring

`git mv archive/<path> <original>` and restore any `__init__` / docs references.
