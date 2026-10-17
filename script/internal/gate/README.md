# Refactor checks

The checks behind `REFACTOR_TODO.md` (step 7). Every step that touches code is checked against runs
of tag `pre-refactor` on the same GPU type and environment; outputs and baselines live outside the repo
(workstation `/local-scratch/localhome/dya78/lightgen_eval_local/segvigen_refactor_checks/`, `C` below;
star2 `~/segvigen_refactor/`). A tree under test is a `git archive` of the commit (with a `TREE_REV`
file), extracted beside the baselines; the tooling always comes from this directory, never from the
tree under test.

## 1. Live-code gate (every code step)

`live_gate.py` builds the denoiser from a run's `config.yaml` through the tree's registries, loads a
checkpoint strictly, reads each shape through the tree's `SegviGenEmissive256`, and saves the denoiser
output at fixed noise for t = 0.25, 0.5, 0.75 and the 12-step sampled latent (published sampler values
and seed rule). `run_live.sh paper|release <tree> <out> <shapes>` fills in the data roots and the
checkpoints (paper `outputs/segvigen_256_bw_6gpu/ckpts/step0023405.pt` on validation shapes, since the
paper's emission latents cover train and val only; release `snapshots/step0044722.pt` on test shapes).
`compare_live.py --a <baseline> --b <out> --shapes <file>` compares per shape and key.

- Shapes: `paper_val20.txt`, `release_test20.txt` (routine); `paper_val183.txt`, `release_test200.txt` (full).
- Workstation 4090 (`trellis2` env): two runs of `pre-refactor` are byte-identical (20/20 and 20/20), so
  the pass bar there is identity. Baselines `C/gate/rtx4090-ws/{paper_val20,release_test20}_run1`,
  `{paper_val183,release_test200}_run1`. Time: about 1 s per shape plus 1–3 min to load.
- sp4090 (`t2lgb_sp4090/env` plus a private `pyarrow` in `C/pydeps`, runner `C/run_sp.sh` on that box):
  byte-identical to the workstation (20/20 on both checkpoints), so a full gate can be split between
  the two boxes and compared with the workstation baselines.

## 2. Published-chain gate (steps that touch `scripts_segvigen/{live_eval_one,live_eval_thumbs,unrekey_for_eval,ckpt_step}.py`)

`star2/published_gate.sbatch <tree tgz> <name>` on one Solar A40 in Kevin's jupiter miniforge env, the
env and card of every published dump: the parent's dump script (`dump_published.py`, a copy whose only
change is `PROBE` = the tree under test) on the 5 paper smoke shapes and 20 release test shapes; the
tree's `unrekey_for_eval.py` on the paper snapshot, compared tensor by tensor with the published bridged
file (`compare_bridged.py`); the tree's `ckpt_step.py` on the paper snapshot and the release
`last.ckpt`. Compare dumps with the parent's `evaluation/newdata_eval/segvigen256/compare_dumps.py`.

- `pre-refactor`, A40 (cs-venus-07 and cs-venus-14): runs byte-identical to each other and to the
  published A40 dumps (paper seed 0: 5/5; release step 44722 seed 0: 20/20); bridge 640/640 tensors
  identical; `ckpt_step.py` reads `23405 31` and `49270 64`.

## 3. Training check (steps that touch training code)

`train_check.sh <tree> <out> fresh|resume|resume_paper` runs the tree's real `train.py` with its release
config on one GPU; `star2/train_check.sbatch` wraps it for Solar (`~/envs/trellis2`). `fresh`: from the
`full_seg` init, 16 release train shapes, batch 1, 32 steps (two epochs, two validations, two saves).
`resume`: from the release `ckpts/last.ckpt` (step 49,270) on the full release data, 17 steps.
`resume_paper`: from the paper snapshot, 17 steps. `traincheck/sitecustomize.py` (through `PYTHONPATH`)
seeds the process and again before `Trainer.fit`, turns on deterministic mode, runs flash-attn's backward
in its deterministic mode, and records per step the batch ids, lr, loss, gradient norm and weight sum,
per validation the metrics and the EMA sum; `summarize_ckpt.py` records every saved file.
`compare_traincheck.py --a <baseline> --b <out>` compares both.

- Without the flash-attn switch, two runs of unchanged code on an L40S share every batch id and the
  step-1 loss but differ from the step-1 gradient on (gradient norm 1.7e-4 apart at step 1).
- With it, `pre-refactor` on Solar L40S (cs-venus-16 and cs-venus-17) is exact: fresh 145/145 numbers and
  resume 74/74 identical between two runs, so the pass bar is identity. About 4 min per run. Baselines
  `~/segvigen_refactor/out/train_l40s_det_pre-refactor_{fresh,resume,resume_paper}_run1` (copied to
  `C/star2_out/`). The 24 GB 4090s are not used for this check.

## 4. Predictor check (step 10)

`star2/native_predict.sbatch <tree tgz> <name> kevin|ours` runs `data_toolkit/predict_emission.py` on the
published-chain gate's shapes (5 paper, 20 release, seed 0). `star2/native_full.sbatch <tree tgz> <name> paper|release
<seed>` remakes one whole published test-set dump (paper 204 shapes, release 200; masks and mask × albedo) on the card
that made it and compares it with `compare_masks.py --identical`. `star2/small_shapes.sbatch` runs the published chain
and the predictor twice each on the three smallest test shapes and a control.

- Tree `native4` (the step-10 predictor), A40, Kevin's env: paper seeds 0–3 and release seeds 0–4 (1,816 dumps, masks
  and mask × albedo) byte-identical except the three smallest shapes of each test set (`3c2165d6`, `93e02c90`, and
  `67055b9d` or `1f2044e5`; under 10,000 decoded voxels), which pass the 2026-09-16 bars.
- On those shapes the published chain does not repeat itself: two runs of it differ from each other and from the
  published dumps as much as the predictor does, and the control (`889cdabc`, 10,628 voxels) is identical in every
  run (`C/star2_out/diag_small/`). Each run autotunes 5–7 flex_gemm kernels live, by timing: the likely cause.
- Paper seed 4 was made on an RTX 6000 Ada; Solar's only one (cs-bd-01) fails torch's CUDA initialization
  (2026-10-12), so that seed is unchecked.

## 5. Conditioning check (steps that touch the conditioning producer)

`star2/cond_check.sbatch <tree tgz> <name> old|new` runs the tree's `build_cond_tree.py` (`data_toolkit/` since step
15d, `scripts_segvigen/newdata/` before) on one Solar A40 in Kevin's env: routes 1–2 (reuse, convert) on
`cond_io6.txt` and route 3 (build: thumbnail → BiRefNet → DINOv3) on `cond_build5.txt`. `old` is a tree whose route 3
imports the original SegviGen tree (the falas deploy), `new` the port in `data_toolkit/thumbnail_cond.py`.

- Step 11c: old and new byte-identical on all 11 shapes; routes 1–2 identical to jupiter's
  `dataset_release_latents/segvigen_cond_fp16`, route 3 within about 1e-3 mean (the release's route 3 ran elsewhere).

## 6. Release-layer check (steps that touch `script/lightgenbench/` or the producers)

`star2/lightgenbench_check.sbatch <tree tgz> <name> lgb_mini4.txt` builds a download folder of the release's own files
for 4 shapes (2 train, 1 val, 1 test: per split a voxel tar and a thumbnail tar, `splits.json`, `checksums.sha256`)
and runs `prepare_lightgenbench.sh` and every `run_chain.sh` stage on one Solar L40S (Kevin's env; `train --tryrun` in
`~/envs/trellis2`), then compares with jupiter's `dataset_release_latents` and the published test prediction.

- Step 15e: dual grids byte-identical (4/4), shape latents within 1.7e-3, PBR latents 3.4e-3, mask latents 2.7e-3,
  conditioning 6.0e-4 mean, init identical (640/640); the test shape's prediction within the bars of the published
  dump (`030adbda…`, a 6-voxel mask, IoU 1.0). The latents differ at float level only (re-encoded on another card),
  as at step 14. `prepare_lightgenbench.sh` alone on jupiter's copy of the Hugging Face release (`SKIP_DOWNLOAD=1`):
  78 tars checked, 36,826 shapes, its index equal to jupiter's (the parquet's contents; its bytes differ).
