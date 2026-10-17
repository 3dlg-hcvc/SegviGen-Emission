# CLAUDE.md

SegviGen-Emission: the LightGen paper's SegviGen-Emission baseline and its retrain on the released LightgenBench
dataset. The method is SegviGen's (Lin Li, MIT): TRELLIS.2's 1.3B texture flow model, finetuned from SegviGen's
`full_seg` checkpoint to predict an emission mask as a texture latent, conditioned on the shape latent (channel
concat), the PBR latent (a second token set on the same voxels) and the DINOv3 tokens of the TexVerse thumbnail.
This repo is the lab's TRELLIS.2-lightning `lightgen` branch at `202bfb0` (Kevin's `segvigen-lightning`), cut down to
the code of those two runs (`REFACTOR_TODO.md`). Upstream is TRELLIS.2-lightning `98c2548`: `git diff 98c2548 -- <file>`
shows ours. Commands for users: `README.md`. Parent repo context: `../AGENTS.md`, `../docs/baselines/segvigen-live-training.md`.

## Ours

| path | what |
|---|---|
| `trellis2/models/segvigen_flow.py` | `SegviGenSLatFlowModel`: upstream's `ElasticSLatFlowModel` with the PBR tokens packed after the noisy target |
| `trellis2/datasets/segvigen_256.py` | `SegviGenEmissive256` and its parquet, split and latent helpers |
| `trellis2/trainers/flow_matching/segvigen_flow_trainer.py` | `SegviGenFlowTrainer`: precomputed DINOv3 tokens, per-shape mean loss, `pos_weight` (off in both runs) |
| `configs/gen/segvigen_emissive_256_release.yaml` | the release run's config (its data locations are set by the launcher or `train.sh`) |
| `data_toolkit/{dual_grid_from_glb,encode_shape_from_dual_grid,encode_pbr_from_voxels}.py` | input producers from the `lightgen` branch |
| `data_toolkit/predict_emission_mask_voxels.py` | the predictor: emission masks (and mask × albedo) from a Lightning checkpoint or a bridged file, the published sampler, seed rule and dump layout |
| `scripts_segvigen/` | the lane's producers (`build_output_tex_latents_256.py`, `rekey_full_seg.py`, `newdata/`), the checkpoint bridge (`unrekey_for_eval.py`, `ckpt_step.py`), and `live_eval_one.py` / `live_eval_thumbs.py`, which the published dump chain imports |
| `script/lightgenbench/` | entry points for LightgenBench users |
| `script/internal/{alliance,star2}/` | the release run's launcher, snapshot sidecar and input jobs |
| `script/internal/gate/` | the refactor's checks (`README.md` there) |

Ours inside upstream files: the registry lines (`trellis2/{models,datasets,trainers}/__init__.py`, `train.py`'s
`TRAINER_MAP`), our training hooks at the end of `trainers/callbacks.py` (TRELLIS2's section plus the PEAK_VRAM line
and `OptionalFinalSnapshotCallback`) and `train.py`'s imports and call sites of them, `basic.py` (the `train/lr` log, Hugging Face and
`.safetensors` loading, the `visualize_sample` guard), `utils/data_utils.py` (`ResumableSampler.set_epoch`),
`trainers/data.py` (empty validation list), `flow_matching/mixins/image_conditioned.py` (the DINOv3 layer path),
`setup.sh` (the `libjpeg-dev` line commented out), and, untouched by decision, `o-voxel/` and
`data_toolkit/{blender_script/dump_pbr.py,voxelize_pbr.py,encode_pbr_latent.py}`.

## The two runs

| run | config | checkpoint of record | trained | result |
|---|---|---|---|---|
| paper `segvigen_256_bw_6gpu` | `configs/gen/segvigen_emissive_256.yaml` at `bf0e9de` (jupiter `outputs/segvigen_256_bw_6gpu/config.yaml`) | jupiter `outputs/segvigen_256_bw_6gpu/ckpts/step0023405_snapshot.ckpt` (full state, sha256 `fc7a9bd1…68b9b`) and `step0023405.pt` (bridged, what the dumps loaded) | Kevin, cs-venus-05, 6 × Blackwell, batch 6 × 8, env `segvigen-bw` | the paper's SegviGen-Emission row |
| release `segvigen_256_release_4h100` | `configs/gen/segvigen_emissive_256_release.yaml` via `script/internal/alliance/segvigen_256_release.sh` | jupiter `outputs/segvigen_256_release_4h100/snapshots/step0044722.pt` (bridged, sha256 `687c6bfd…acad6be`); `ckpts/last.ckpt` (step 49,270, full state) | fir job 60584213, 4 × H100, batch 4 × 12 | test IoU 0.1468 ± 0.0083 (5 seeds, per-part GT); `../docs/evaluation.md` § Released dataset |

Both runs use the same training code and the init `full_seg_rekeyed.pt` (jupiter
`datasets/dataset_release_latents/init/`, sha256 `ae4cb246…231981b`; `script/lightgenbench/get_init.sh` rebuilds it).

Keep stable: the registry names `SegviGenSLatFlowModel`, `SegviGenEmissive256`, `SegviGenFlowTrainer` (in every
stored `config.yaml`) and every config key; the checkpoint layouts (Lightning `models.denoiser.*` with the duplicate
`training_models.denoiser.*`, EMA, `sampler_state`, `elastic_controller`; the bridged
`{"state_dict": {"gen3dseg.flow_model.<key>"}}`, 640 keys, raw weights); `scripts_segvigen/unrekey_for_eval.py` and
`ckpt_step.py` (the release sidecar); `scripts_segvigen/live_eval_one.py` (the parent's published chain imports it
from Kevin's falas checkout); branch `retrain-newdata` (the Alliance clones and the launcher's `BRANCH`).

## The published predictions

Every published dump came from the parent's `evaluation/newdata_eval/dump_segvigen256_voxels.py`, which imports
`scripts_segvigen/live_eval_one.py` from Kevin's falas checkout and, through it, the original SegviGen repo's
`Gen3DSeg`, `Sampler` and `load_eval_models` (falas deploy of `QhelDIV/SegviGen-emissive` at `a67a69c`), on the bridged
file, in Kevin's jupiter miniforge env: paper seeds 0–3 on A40, seed 4 on an RTX 6000 Ada; every release dump on A40.
12 Euler steps, guidance 1.0, interval [0.6, 0.9], rescale_t 3.0; per-shape seed
`seed * 1000003 + int(sid[:8], 16) % 100000 + k`; output `<out>/draw<k>/<sid>.npz` (`coords` int16, `pred` float16).
On an A40 in that env, this tree's copy of that chain and its own predictor reproduce those dumps byte for byte, except
on the three smallest shapes of each test set, where the chain does not repeat itself from run to run
(`script/internal/gate/README.md` § 4).

## Environments that run this code

| env | where | Python, torch | what ran in it |
|---|---|---|---|
| `segvigen-bw` (Kevin's, `/localscratch/xya120/miniforge3` on cs-venus-05) | cs-venus-05, 6 × RTX PRO 6000 Blackwell (sm_120) | torch and xformers from the cu128 wheel index, CUDA 12.9, no flash-attn; built by `script/bw/bootstrap_segvigen_bw.sh` (tag `pre-refactor`) | the paper run's training |
| venv `~/scratch/TRELLIS.2-lightning/env` (TRELLIS2's `script/internal/alliance/build_env.sh`) | Alliance H100 (fir) | 3.11, torch 2.9.1, transformers 5.3.0, flash_attn 2.8.3, lightning 2.6.1 | the release run's training |
| conda `trellis2`, Kevin's jupiter miniforge (`/3dlg-jupiter-project/lightgen/miniforge3`) | Solar A40, L40S, RTX 6000 Ada | 3.10.20, torch 2.6.0 + CUDA 12.4, transformers 4.57.6, flash_attn 2.7.3; no lightning (no training here) | every published dump, the release's route-3 conditioning, the published-chain and conditioning checks |
| `~/envs/trellis2` (Solar shared home) | cs-venus L40S | 3.10.19, torch 2.6.0 + CUDA 12.4, transformers 5.1.0, flash_attn 2.7.3, lightning 2.6.1; flex_gemm for sm_89 only (an A40 fails) | the release's inputs (`script/internal/star2/newdata_*.sbatch`), the training check |
| conda `trellis2` | workstation (cs-3dlg-23, RTX 4090) and sp4090 (`t2lgb_sp4090/env`) | the same versions as `~/envs/trellis2` | the live-code gate, the emit-stage check |

What every env needs: flex_gemm and o_voxel compiled for the GPU's architecture; in the Hugging Face cache,
`microsoft/TRELLIS.2-4B` (the shape and texture encoders and decoders, `pipeline.json`); for route 3 of the
conditioning build, `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0` with a transformers that
loads BiRefNet (5.1 does not; 4.57.6 does); for training, lightning 2.6. `setup.sh` is upstream's (our one change:
the `sudo apt install libjpeg-dev` line is commented out); no requirements file exists.

Blackwell (sm_120), from the paper run (Dongchen's cs-venus-19 launcher and `bootstrap_bw05.sh`, confirmed on
cs-venus-05): flash-attn has no sm_120 build, so `ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=xformers` (xformers
works under the bf16 autocast; a standalone fp32 test of it fails falsely); a separate env with torch, torchvision and
xformers from `https://download.pytorch.org/whl/cu128`, `CUDA_HOME=/usr/local/cuda-12.9`,
`TORCH_CUDA_ARCH_LIST=12.0`; `FLEX_GEMM_USE_AUTOTUNE_CACHE=0` and `FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0` (the file
cache races across ranks on shared storage); `SLURM_JOB_NAME=bash` (Lightning spawns its own ranks under one Slurm
task); a pre-filled `HF_HOME` with `HF_HUB_OFFLINE=1`. The paper's conditioning was stored fp16 (bf16 storage would
be bit-identical under the bf16 autocast; fp16 measured a 0.014 % relative loss shift).

## Traps

- A `step<N>-<val/loss>.ckpt` file holds step N's weights under the previous validation's loss (top-k fires on the
  step trigger with the last logged metric), in both runs; select by the snapshots and the IoU, never by that name.
- This package's decoders (upstream TRELLIS.2-lightning) are not the copy the published chain vendored: they keep the
  latent's fp32 after `from_latent`, which the fp16 subdivision head of the stock TRELLIS.2-4B shape decoder rejects,
  and give a voxel without a positive subdivision logit one child. The predictor's `decode_as_published` restores the
  vendored behavior per decoder instance; `convert_to_fp32()` also runs but changes the predictions.
- `train.py` sets no global seed; two runs differ from the first backward step unless flash-attn's backward runs in its
  deterministic mode (the training check does this through its `sitecustomize.py`).
- Kevin's env has no lightning; `~/envs/trellis2` cannot load BiRefNet (route 3 of the conditioning).
