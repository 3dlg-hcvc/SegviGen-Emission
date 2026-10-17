# CLAUDE.md

SegviGen-Emission: SegviGen (TRELLIS.2's 1.3B texture flow model, finetuned from its `full_seg` checkpoint) adapted
to predict an emission mask on sparse voxels; the paper's SegviGen-Emission baseline and its retrain on LightgenBench.
**`README.md` is the doc of record**: what differs from SegviGen and TRELLIS.2, setup, data, the release chain
(`script/lightgenbench/run_chain.sh`), prediction on the test split. This file holds what an agent working here needs
beyond it. The refactor's plan and its "Keep stable" list: `REFACTOR_TODO.md`. This repo is the lab's
TRELLIS.2-lightning `lightgen` branch at `202bfb0` (Kevin's `segvigen-lightning`); upstream is TRELLIS.2-lightning
`98c2548` (`git diff 98c2548 -- <file>` shows ours). The parent lightgen repo holds the evaluation
(`../docs/evaluation.md`), the checkpoint store rules (`../AGENTS.md` § Filesystem) and this lane's history
(`../docs/baselines/segvigen-live-training.md`).

## Environments

| env | where | Python, torch | what ran in it |
|---|---|---|---|
| `segvigen-bw` (Kevin's, `/localscratch/xya120/miniforge3` on cs-venus-05) | cs-venus-05, 6 × RTX PRO 6000 Blackwell (sm_120) | torch and xformers from the cu128 wheel index, CUDA 12.9, no flash-attn; built by `script/bw/bootstrap_segvigen_bw.sh` (tag `pre-refactor`) | the paper run's training |
| venv `~/scratch/TRELLIS.2-lightning/env` (TRELLIS2's `script/internal/alliance/build_env.sh`) | Alliance H100 (fir) | 3.11, torch 2.9.1, transformers 5.3.0, flash_attn 2.8.3, lightning 2.6.1 | the release run's training |
| conda `trellis2`, Kevin's jupiter miniforge (`/3dlg-jupiter-project/lightgen/miniforge3`) | Solar A40, L40S, RTX 6000 Ada | 3.10.20, torch 2.6.0 + CUDA 12.4, transformers 4.57.6, flash_attn 2.7.3; no lightning (no training here) | every published dump, the release's conditioning, the published-chain, predictor and release-layer checks |
| `~/envs/trellis2` (Solar shared home) | cs-venus L40S | 3.10.19, torch 2.6.0 + CUDA 12.4, transformers 5.1.0, flash_attn 2.7.3, lightning 2.6.1; flex_gemm for sm_89 only (an A40 fails) | the release's inputs (`script/internal/star2/newdata_*.sbatch`), the training check |
| conda `trellis2` | workstation (cs-3dlg-23, RTX 4090) and sp4090 (`t2lgb_sp4090/env`) | the same versions as `~/envs/trellis2` | the live-code gate, the emit-stage check |

What every env needs: flex_gemm and o_voxel compiled for the GPU's architecture; in the Hugging Face cache,
`microsoft/TRELLIS.2-4B` (the shape and texture encoders and decoders, `pipeline.json`); for the conditioning,
`facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0` with a transformers that loads BiRefNet (5.1 does
not; 4.57.6 does); for training, lightning 2.6. `setup.sh` is upstream's (our one change: the
`sudo apt install libjpeg-dev` line is commented out); no requirements file exists.

Blackwell (sm_120), from the paper run (the env `script/bw/bootstrap_segvigen_bw.sh` builds, tag `pre-refactor`,
confirmed on cs-venus-05): flash-attn has no sm_120 build, so `ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=xformers`
(xformers works under the bf16 autocast; a standalone fp32 test of it fails falsely); a separate env with torch,
torchvision and xformers from `https://download.pytorch.org/whl/cu128`, `CUDA_HOME=/usr/local/cuda-12.9`,
`TORCH_CUDA_ARCH_LIST=12.0`; `FLEX_GEMM_USE_AUTOTUNE_CACHE=0` and `FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0` (the file cache
races across ranks on shared storage); `SLURM_JOB_NAME=bash` (Lightning spawns its own ranks under one Slurm task); a
pre-filled `HF_HOME` with `HF_HUB_OFFLINE=1`. The paper's conditioning was stored fp16 (bf16 storage would be
bit-identical under the bf16 autocast; fp16 measured a 0.014 % relative loss shift).

## Runs of record

| run | config | checkpoint of record | trained | result |
|---|---|---|---|---|
| paper `segvigen_256_bw_6gpu` | `configs/gen/segvigen_emissive_256.yaml` at `bf0e9de` (jupiter `outputs/segvigen_256_bw_6gpu/config.yaml`) | jupiter `outputs/segvigen_256_bw_6gpu/ckpts/step0023405_snapshot.ckpt` (full state, sha256 `fc7a9bd1…68b9b`) and `step0023405.pt` (bridged, what the dumps loaded) | Kevin, cs-venus-05, 6 × Blackwell, batch 6 × 8, env `segvigen-bw` | the paper's SegviGen-Emission row |
| release `segvigen_256_release_4h100` | `configs/gen/segvigen_emissive_256_release.yaml` via `script/internal/alliance/segvigen_256_release.sh` | jupiter `outputs/segvigen_256_release_4h100/snapshots/step0044722.pt` (bridged, sha256 `687c6bfd…acad6be`); `ckpts/last.ckpt` (step 49,270, full state) | fir job 60584213, 4 × H100, batch 4 × 12 | test IoU 0.1468 ± 0.0083 (5 seeds, per-part GT); `../docs/evaluation.md` § Released dataset |

Both runs use the same training code and the init `full_seg_rekeyed.pt` (jupiter
`datasets/dataset_release_latents/init/`, sha256 `ae4cb246…231981b`; `run_chain.sh train` rebuilds it in
`data/ckpt_init/`). Jupiter (`/cs/3dlg-jupiter-project/lightgen/outputs/<run>/`) is the source, `outputs/<run>/` here
the cache; name the file, never let anything pick one. The paper config lives only in its run's `config.yaml` and in
tag `pre-refactor` (D17).

### The published predictions

Every published dump came from the parent's `evaluation/newdata_eval/dump_segvigen256_voxels.py`, which imports
`scripts_segvigen/live_eval_one.py` from Kevin's falas checkout and, through it, the original SegviGen repo's
`Gen3DSeg`, `Sampler` and `load_eval_models` (falas deploy of `QhelDIV/SegviGen-emissive` at `a67a69c`), on the bridged
file, in Kevin's jupiter miniforge env: paper seeds 0–3 on A40, seed 4 on an RTX 6000 Ada; every release dump on A40.
12 Euler steps, guidance 1.0, interval [0.6, 0.9], rescale_t 3.0; per-shape seed
`seed * 1000003 + int(sid[:8], 16) % 100000 + k`; output `<out>/draw<k>/<sid>.npz` (`coords` int16, `pred` float16).
On an A40 in that env, this tree's copy of that chain and its own predictor reproduce those dumps byte for byte, except
on the three smallest shapes of each test set, where the chain does not repeat itself from run to run
(`script/internal/gate/README.md` § 4).

## Keep stable (until the cut-over, step 17)

The registry names `SegviGenSLatFlowModel`, `SegviGenEmissive256`, `SegviGenFlowTrainer` (in every stored
`config.yaml`) and every config key; the checkpoint layouts (Lightning `models.denoiser.*` with the duplicate
`training_models.denoiser.*`, EMA, `sampler_state`, `elastic_controller`; the bridged
`{"state_dict": {"gen3dseg.flow_model.<key>"}}`, 640 keys, raw weights); `scripts_segvigen/unrekey_for_eval.py` and
`ckpt_step.py` (the release sidecar); `scripts_segvigen/live_eval_one.py` (the parent's published chain imports it
from Kevin's falas checkout); branch `retrain-newdata` (the Alliance clones and the launcher's `BRANCH`).

## Commands

```bash
python train.py --config <cfg> --output_dir outputs/<run> [--ckpt last|none] [--num_gpus 4] [--override k=v ...] [--tryrun]
bash script/lightgenbench/run_chain.sh latents|masks|cond|train|predict <data_root>   # the release chain (README.md)
bash script/internal/alliance/segvigen_256_release.sh                                 # the release run on Alliance
python data_toolkit/predict_emission.py --help                                        # the predictor itself
python -m pytest data_toolkit/test_checkpoint_cadence.py                              # the one test (5 tests)
```

Tee every training run's stdout to `log/<TS>-<name>.log` (`run_chain.sh` does) and log to wandb project `LightGen`
under a chain-named run (`--override trainer.args.wandb_name=...`); with `wandb_enable` false or missing,
`train.py:build_logger` falls back to a CSV logger silently. Cluster work goes through the compute-access skill; the
internal launchers are under `script/internal/` (`alliance/` for the release run, `star2/` for its inputs, `gate/`).

## Checking a code change (refactor gate)

Tooling and its doc in `script/internal/gate/` (`README.md` there); data and outputs outside the repo, in
`C = /local-scratch/localhome/dya78/lightgen_eval_local/segvigen_refactor_checks/` and star2 `~/segvigen_refactor/`. A
tree under test is a `git archive` of the commit with a `TREE_REV` file; compare a run only with `pre-refactor` runs of
the same GPU type and env. Run a check only where a change can reach it.

```bash
G=script/internal/gate
bash $G/run_live.sh paper|release <tree> <out> <shapes>; python $G/compare_live.py --a <baseline> --b <out> --shapes <file>   # 4090
sbatch ... $G/star2/published_gate.sbatch <tree tgz> <name>              # the published chain, A40, Kevin's env
sbatch ... $G/star2/train_check.sbatch <tree tgz> <name> fresh|resume|resume_paper; python $G/compare_traincheck.py --a <rec> --b <rec>   # L40S
sbatch ... $G/star2/native_predict.sbatch <tree tgz> <name> kevin        # the predictor on the gate's shapes, A40
sbatch ... $G/star2/native_full.sbatch <tree tgz> <name> paper|release <seed>   # a whole published test set, byte for byte
sbatch ... $G/star2/lightgenbench_check.sbatch <tree tgz> <name> $G/lgb_mini4.txt   # the release layer end to end, L40S
sbatch ... $G/star2/cond_check.sbatch <tree tgz> <name> old|new          # the conditioning producer
```

Every check repeats exactly on its GPU type (the training check with flash-attn's backward made deterministic by
`traincheck/sitecustomize.py`), so the bar is identity, except where `README.md` there says otherwise.

## Source map

| where | what |
|---|---|
| `trellis2/models/segvigen_flow.py` | `SegviGenSLatFlowModel`: upstream's `ElasticSLatFlowModel` with the PBR tokens packed after the noisy target |
| `trellis2/datasets/segvigen_256.py` | `SegviGenEmissive256` and its parquet, split and latent helpers |
| `trellis2/trainers/flow_matching/segvigen_flow_trainer.py` | `SegviGenFlowTrainer`: precomputed DINOv3 tokens, per-shape mean loss, `pos_weight` (off in both runs) |
| `trellis2/trainers/callbacks.py` | upstream's callbacks, then ours (TRELLIS2's section plus the PEAK_VRAM line and `OptionalFinalSnapshotCallback`) |
| `configs/gen/segvigen_emissive_256_release.yaml` | the release run's config (its data locations are set by the launcher or `run_chain.sh`) |
| `data_toolkit/dual_grid_from_glb.py` | the dual grid from a GLB, TRELLIS2's file (the release: `--loader lightgenbench --check_cells`) |
| `data_toolkit/{encode_shape_from_dual_grid,encode_pbr_from_voxels}.py` | the shape and PBR latents, TRELLIS2's files |
| `data_toolkit/{build_lightgenbench_index,encode_emission_mask_latents,build_cond_tree,thumbnail_cond,verify_and_pack,init_segvigen_from_full_seg}.py` | the release producers: the index and split rows, the emission-mask latents, the DINOv3 conditioning (SegviGen's thumbnail preprocessing), the check and pack, the init from SegviGen's `full_seg` |
| `data_toolkit/predict_emission.py` | the predictor: masks and mask × albedo from a Lightning checkpoint or a bridged file, the published sampler, seed rule and dump layout |
| `scripts_segvigen/` | the checkpoint bridge (`unrekey_for_eval.py`, `ckpt_step.py`) and `live_eval_one.py` / `live_eval_thumbs.py`, which the published dump chain imports |
| `script/lightgenbench/` | `prepare_lightgenbench.sh`, `run_chain.sh` |
| `script/internal/{alliance,star2,gate}/` | the release run's launcher and snapshot sidecar, its input jobs, the refactor's checks |

Upstream's edits that stay in its files are listed in the README (What differs) and in `REFACTOR_TODO.md` step 12.

## Traps

1. **Never start a first launch in a non-empty run directory.** `on_fit_start` skips the `finetune_ckpt` load when it
   resumes (`trellis2/trainers/basic.py`), so the `full_seg` init is lost silently; a missing `finetune_ckpt` key is a
   silent no-op too. `run_chain.sh train` refuses a run directory with checkpoints unless `RESUME=1`.
2. **A `step<N>-<val/loss>.ckpt` holds step N's weights under the previous validation's loss** (top-k fires on the
   step trigger with the last logged metric), in both runs; select by the snapshots and the IoU, never by that name.
3. **This package's decoders are not the copy the published chain vendored.** Upstream TRELLIS.2-lightning keeps the
   latent's fp32 after `from_latent`, which the fp16 subdivision head of the stock TRELLIS.2-4B shape decoder rejects,
   and gives a voxel without a positive subdivision logit one child. The predictor's `decode_as_published` restores
   the vendored behavior per decoder instance; `convert_to_fp32()` also runs but changes the predictions.
4. **The smallest shapes do not repeat.** FlexGEMM autotunes a kernel by timing for every problem size not in its
   cache, so shapes under about 10,000 decoded voxels can differ from run to run, in the published chain as here.
   Its cache lives per GPU name in `~/.flex_gemm/`; on a shared NFS home ranks race on it.
5. **Training does not repeat by default.** `train.py` sets no global seed, and flash-attn's backward is not
   deterministic; the training check makes both repeat through its `sitecustomize.py`.
6. **Two envs.** Kevin's env has no lightning; `~/envs/trellis2` cannot load BiRefNet (the conditioning). The release
   chain takes `PYTHON=` per stage for this.
7. **`drop_last=True` with tiny data:** a train split smaller than `num_gpus × batch_size_per_gpu` gives no batches.

## Conventions

- `README.md` first: a user-facing change updates it; this file holds only what an agent needs beyond it.
- Our code goes in files of its own beside upstream's, except the training callbacks at the end of `callbacks.py`;
  where TRELLIS2 (`../TRELLIS2/`) carries the same file, both make the same call (`REFACTOR_TODO.md`).
- Evaluation lives in the parent repo (`evaluation/pointsampled/`, doc `docs/evaluation.md`): masks from
  `data_toolkit/predict_emission.py` are scored on mesh-surface points there.
