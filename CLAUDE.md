# CLAUDE.md

SegviGen-Emission: the training code of the LightGen paper's SegviGen-Emission baseline and of its retrain
on the released LightgenBench dataset. A fork of the lab's TRELLIS.2-lightning `lightgen` branch at
`202bfb0` (Kevin's `segvigen-lightning`), being cut down to the code those two runs use.

The refactor is in progress: the plan and each step's record are in `REFACTOR_TODO.md`, the checks in
`script/internal/gate/`. Step 15 writes this repo's doc of record. The `lightgen` branch's TRELLIS.2-Emission
guide that this file held before is in git history (tag `pre-refactor`); those lanes are not part of this repo.

Parent repo context: `../AGENTS.md`, `../docs/baselines/segvigen-live-training.md`.

## Environments that run this code

| env | where | Python, torch | what ran in it |
|---|---|---|---|
| `segvigen-bw` (Kevin's, `/localscratch/xya120/miniforge3` on cs-venus-05) | cs-venus-05, 6 × RTX PRO 6000 Blackwell (sm_120) | torch and xformers from the cu128 index, CUDA 12.9, no flash-attn (`ATTN_BACKEND=sdpa`, `SPARSE_ATTN_BACKEND=xformers`); built by `script/bw/bootstrap_segvigen_bw.sh` at tag `pre-refactor`; facts in `BLACKWELL.md` | the paper run's training |
| venv `~/scratch/TRELLIS.2-lightning/env` (TRELLIS2's `script/internal/alliance/build_env.sh`) | Alliance H100 (fir) | 3.11, torch 2.9.1, transformers 5.3.0, flash_attn 2.8.3, lightning 2.6.1 | the release run's training (`script/internal/alliance/segvigen_256_release.sh`) |
| conda `trellis2`, Kevin's jupiter miniforge (`/3dlg-jupiter-project/lightgen/miniforge3`) | Solar A40, L40S, RTX 6000 Ada | 3.10.20, torch 2.6.0 + CUDA 12.4, transformers 4.57.6, flash_attn 2.7.3; no lightning (no training here) | every published dump (paper and release), route 3 of the release's conditioning (`build_cond_tree.py`), the published-chain and conditioning checks |
| `~/envs/trellis2` (Solar shared home) | cs-venus L40S | 3.10.19, torch 2.6.0 + CUDA 12.4, transformers 5.1.0, flash_attn 2.7.3, lightning 2.6.1; flex_gemm for sm_89 only (an A40 fails) | the release's inputs (`script/internal/star2/newdata_*.sbatch`), the training check |
| conda `trellis2` | workstation (cs-3dlg-23, RTX 4090) and sp4090 (`t2lgb_sp4090/env`) | the same versions as `~/envs/trellis2` | the live-code gate, the emit-stage check |

What every env needs: flex_gemm and o_voxel compiled for the GPU's architecture; in the Hugging Face cache,
`microsoft/TRELLIS.2-4B` (the shape and texture encoders and decoders, `pipeline.json`); for route 3 of the
conditioning build, `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0` with a transformers that
loads BiRefNet (5.1 does not; Kevin's 4.57.6 does); for training, lightning 2.6. `setup.sh` is upstream's (our
one change: the `sudo apt install libjpeg-dev` line is commented out); no requirements file exists.
