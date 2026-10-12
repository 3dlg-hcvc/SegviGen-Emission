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
