# SegviGen-Emission refactor — plan

Turn Kevin's `segvigen-lightning` fork into a clean SegviGen-Emission: only the code behind the paper and
release runs, our changes as edits on upstream TRELLIS.2-lightning's modules or as new files beside
upstream's in its layout, generic entry points for LightgenBench users. Modeled on
`../TEXGen-Emission/.claude/worktrees/refactor/REFACTOR_TODO.md` (TEXGen) and
`../TRELLIS2/.claude/worktrees/trellis2-emission-refactor/REFACTOR_TODO.md` (TRELLIS2, same upstream);
the rules are theirs unless a line says otherwise. This repo forks our TRELLIS2 `lightgen` branch, so
where both repos carry the same file the two plans make the same call. Paths are relative to
`SegviGen-Emission/`; `../` is the parent lightgen repo.

The 2026-09-16 restructure of this lane (spec `../docs/superpowers/specs/2026-09-16-segvigen-paper-lane-design.md`,
plan `../docs/superpowers/plans/2026-09-16-segvigen-paper-lane.md`) stopped on 2026-09-17 after its
baseline capture and a trim that was abandoned (branch `trim-abandoned-2026-09-17`). Its file inventory,
provenance records and the gate's before leg are reused here. Its approach is not: it moved files into
`deprecated/`, renamed classes without aliases and replaced the published inference with a new native
path. It also predates the release retrain, so its lists would move live release code away
(`unrekey_for_eval.py`, `ckpt_step.py`, `build_output_tex_latents_256.py`, three `data_toolkit/` producers).

## Progress

Order: 1, 7, then 2–6 and 8–17 (the deletions in 2–6 need step 7's gates and baselines).

- [x] 1. Housekeeping (tag and worktree done; the paper checkpoint onto jupiter, D21)
- [x] 2. Dead root files, one-off scripts and the live-evaluation loop
- [x] 3. `archive/`, `docs/`, `reports/`, stale ignore rules
- [x] 4. `blender_packages/` (upstream's demo material stays, D12)
- [x] 5. Launchers (per D17)
- [x] 5b. Move the kept cluster scripts into `script/internal/`
- [x] 6. Configs (per D17)
- [x] 7. Check setup: live-code gate, published-chain gate, training check, noise baselines (no code change)
- [x] 8. Our modules (per D13)
- [x] 9. Coverage run (no code change)
- [x] 10. An inference path in this repo (per D18; the remade dumps stand in for the rescore)
- [x] 11. The chain scripts in `scripts_segvigen/`
- [x] 12. Our hooks in upstream's `train.py`, `basic.py`, `callbacks.py`
- [x] 13. Environment files
- [x] 14. Release layer: generic entry points for LightgenBench users
- [x] 15. Docs
- [ ] 16. Dongchen's review, the merge into `release` (D4, D11) and parent registration (needs his go)
- [ ] 17. Cut-over (per D8)

## How we work

- One step = one commit with one purpose. Removals and moves never share a commit with a behavior
  change. Cleanups change no behavior; fixes go under "Later", one step each, only if wanted.
- Since 2026-10-11 the steps run back to back in this worktree, one commit per step (Dongchen
  answered the decisions and said "before you merge it to the repo, you should ask me to review
  before merging"). A step whose checks pass is committed and reported in one line; its **Result**
  is at most three lines (what changed, commits, verdict); decisions taken inside a step go to its
  Result and its commit message. Step 16 (the merge) waits for his review and go. Dongchen can add
  **Instructions** to any step.
- Within a step, stop and ask only for: (1) an open decision the step needs; (2) a check that fails, unless the
  cause is an obvious slip in the step; (3) something the plan does not cover (a needed fix, a
  behavior change, a file outside the step's list); (4) an action outside the worktree and our
  compute: a push, a parent-repo commit, the merge, a write to jupiter's shared tree, a message to
  Kevin. Jobs on our machines for the checks need no go.
- After each step that touches code the live-code gate must pass. Steps that touch training code
  also need the training check; steps that touch `scripts_segvigen/{live_eval_one,live_eval_thumbs,unrekey_for_eval,ckpt_step}.py`
  or anything they import also need the published-chain gate. A check passes when outputs are
  identical, or every difference is within the noise of two runs of tag `pre-refactor` on the same
  GPU type; `missing` must be 0; a NaN or an empty input never passes.
- Run checks in parallel across our machines (Dongchen 2026-10-11: "use my other lab machines and
  cluster resources"; workstation 4090, sp4090, xg4090, Solar L40S and A40, Alliance H100; the
  compute-access skill): split a gate by shape list, run a training check's before, after
  and noise runs at once, prepare the next step while checks run (commit it only after the step
  before passes). Skip a machine that needs a 2FA login. Submit Solar jobs without `-w`: a pinned job
  on our own nodes evicts another lab's running job.
- Compare only within one GPU type and environment. Measured on the published chain: L40S against
  the published A40 dumps reads mask IoU 0.9675–1.0 on five shapes, none byte-identical
  (`../evaluation/newdata_eval/segvigen256/records/smoke_before.md`); A40 against A40 in the same env
  read 3/3 and 61/61 shapes byte-identical, but one shard on another A40 node read 60/61 (cause not
  known; the flex_gemm autotune cache is one candidate). Each GPU type gets its own baseline and noise
  runs from `pre-refactor` (step 7).
- Upstream is TRELLIS.2-lightning `origin/main` (`98c2548`), the lab's Lightning port of
  microsoft/TRELLIS.2. Two layers of ours sit on it: our `lightgen` branch up to `upstream-base`
  (`202bfb0`; 351 files, the TRELLIS.2-Emission lanes) and Kevin's 34 commits (`202bfb0..bff5c7e`,
  81 files). `git diff 98c2548 -- <file>` shows all of ours, `git diff upstream-base -- <file>`
  Kevin's alone. Our changes are edits to upstream's modules or new files beside upstream's in its
  layout (D13): our code lives in files of its own beside upstream's; inside an upstream file only the
  line that registers our class stays (`TRAINER_MAP`, the three registries). A file that only keeps a
  published registry name alive is a one-line alias. Upstream code stays untouched.
- `o-voxel/` is not touched (D14).
- A doc that names a removed file is fixed in the same step.
- Each step updates this repo's own docs it affects (`README.md`, `CLAUDE.md`, `BLACKWELL.md`) in the
  same commit. Parent docs (`../AGENTS.md`, `../docs/baselines/segvigen.md`,
  `../docs/baselines/segvigen-live-training.md`, `../docs/status/{refactor,checkpoints}.md`,
  `../docs/evaluation.md`) change only at step 16: a step that makes one stale adds an item to step
  16's parent-docs list.
- All edits happen in the worktree `.claude/worktrees/segvigen-emission-refactor` (branch
  `segvigen-emission-refactor`, from tag `pre-refactor` = `bff5c7e`); `git diff pre-refactor` shows
  every edit. The main checkout stays on `retrain-newdata`. The reviewed result lands on branch
  `release` (D4).
- In the main checkout: never `git add -A` and never `git clean -x`/`-X`.
- The parent does not know this repo yet: `/SegviGen-Emission/` is in the parent's `.git/info/exclude`
  and has no `.gitmodules` entry or gitlink. No session commits a gitlink before step 16 (an
  unreachable gitlink breaks `git submodule` for every clone). Nothing is pushed without Dongchen's
  go: `origin` is Kevin's `QhelDIV/segvigen-lightning`.

## Runs and the gate

| run | config | checkpoint | trained |
|---|---|---|---|
| paper `segvigen_256_bw_6gpu` | `configs/gen/segvigen_emissive_256.yaml` as at `bf0e9de` (in effect through step 23405; the launcher re-copied the yaml from Kevin's falas worktree at each launch, `script/bw/launch_segvigen_256_6gpu.sbatch:67-70`, and his later edit there changes only data paths, dated after the snapshot: `../evaluation/newdata_eval/segvigen256/records/kevin_worktree_edit_2026-08-27.diff`); batch 6 × 8 from the launcher (the yaml says 4) | falas `/cs/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/segvigen_256_bw_6gpu_ckpts_mirror/step0023405_snapshot.ckpt` (10,480,576,446 B, sha256 `fc7a9bd1…68b9b`, full Lightning state); bridged copy jupiter `trellis2_bw/eval_outputs/segvigen256_ourval/bridged/step0023405.pt` (md5 `79bf775e…`). No run dir on jupiter (D21) | cs-venus-05, 6 × Blackwell, Kevin's env `segvigen-bw` |
| release `segvigen_256_release_4h100` | `configs/gen/segvigen_emissive_256_release.yaml` at `4c80876` via `script/alliance/segvigen_256_release.sh` (4 × 12) | jupiter `outputs/segvigen_256_release_4h100/snapshots/step0044722.pt` (bridged layout, raw weights, sha256 `687c6bfd…acad6be`); `ckpts/last.ckpt` (step 49,270, full state) for resume; `config.yaml`, `command.txt` beside them | fir job 60584213, 4 × H100, venv `~/scratch/TRELLIS.2-lightning/env` |

Both runs train with the same code: `git diff bf0e9de bff5c7e -- trellis2 train.py
configs/gen/segvigen_emissive_256.yaml script/bw` is empty. Both share the init
`full_seg_rekeyed.pt` (jupiter `datasets/dataset_release_latents/init/`, sha256 `ae4cb246…231981b`).

**The published dump chain (both runs) runs nothing of this repo's `trellis2/`:**

1. The checkpoint goes through `scripts_segvigen/unrekey_for_eval.py` into the bridged layout
   `{"state_dict": {"gen3dseg.flow_model.<key>": t}}` (640 keys, raw weights, not EMA). The release
   sidecar does this for every snapshot (`script/alliance/snapshot_sidecar.sh:23-28`).
2. `../evaluation/newdata_eval/dump_segvigen256_voxels.py` (run from frozen jupiter copies:
   `trellis2_bw/eval_inputs/segvigen256_ourval/` for the paper, `trellis2_bw/eval_outputs/release_eval/`
   for the release) imports `scripts_segvigen/live_eval_one.py` from Kevin's falas checkout
   (`PROBE`, `:36-39`). That module puts the original SegviGen repo's falas deploy
   (`QhelDIV/SegviGen-emissive` at `a67a69c`) first on `sys.path` (`live_eval_one.py:42-47`), so
   `trellis2` (upstream TRELLIS.2, vendored there), `inference_full.Gen3DSeg` and
   `eval_emissive.load_eval_models` all come from that deploy.
3. 12 Euler steps, `guidance_strength` 1.0, `guidance_rescale` 0, `guidance_interval` [0.6, 0.9],
   `rescale_t` 3.0; decode `* 0.5 + 0.5`; per-shape seed `seed * 1000003 + int(sid[:8], 16) % 100000 + k`
   with `torch.manual_seed`, plus `np.random.seed(seed)`. No deterministic mode anywhere.
4. Kevin's jupiter miniforge env `trellis2`; paper seeds 0–3 on A40, seed 4 on an RTX 6000 Ada;
   every release dump on A40.
5. Scored by `../evaluation/pointsampled/pipeline.py --voxel_pred_root` in the workstation `lightgen`
   env (`segvigen256/score_many.sh`, `segvigen256/release_score_driver.sh`); MAE and PSNR from
   mask × albedo dumps (`segvigen256/dump_ma_reuse.py`, jupiter `release_ma.sbatch`).

So TEXGen's gate (inference through the repo's own code) does not exist here. Step 7 builds three checks:

- **Live-code gate** (every code step): a tool under `script/internal/gate/` that builds the denoiser
  from a run's `config.yaml` through this repo's registries, loads a checkpoint (`models.denoiser.*`
  from a Lightning file, `gen3dseg.flow_model.*` from a bridged one, strict key accounting), reads the
  gate shapes through `SegviGenEmissive256` and its collate, and saves (a) the denoiser output at fixed
  noise and three fixed `t`, and (b) the latent after 12 Euler steps with the shipped sampler values
  and the published seed rule, before decoding. Per tensor: byte identity, mean and max |d|. Paper
  and release checkpoints. The dataset admits only shapes with an emission latent, and the paper's
  256 emission latents exist for train and val only (0 of the 204 test shapes:
  `../evaluation/newdata_eval/dump_segvigen256_voxels.py:129-132`), so the paper checkpoint runs on
  validation shapes (`../evaluation/newdata_eval/eval_sets/val_agentic_clean.txt`) and its test shapes
  belong to the published-chain gate. Routine steps: 20 paper validation shapes and 20 release
  shapes; full: all of them, for steps that touch the dataset, the model, the trainer or checkpoint
  loading, and before the merge.
- **Published-chain gate** (the four scripts in Keep stable 4–5): the dump script with `PROBE` taken
  from the tree under test (a copy under `script/internal/gate/`; the parent file stays), Kevin's
  env, A40, the five smoke shapes and a release subset; compared to a fresh `pre-refactor` run and,
  reported only, to the published A40 dumps (`segvigen256_ourval/dump_step0023405_seed0_test`,
  `release_eval/dumps_test/step0044722/seed00`).
- **Training check** (steps that touch training code; D5).

## Keep stable (other code depends on these)

1. The directory `SegviGen-Emission/`, unregistered in the parent until step 16.
2. Registry names in every stored `config.yaml` (the release run's on jupiter, `bf0e9de`'s paper
   config, Kevin's run dirs): `SegviGenSLatFlowModel` (`trellis2/models/__init__.py:12`),
   `SegviGenEmissive256` (`trellis2/datasets/__init__.py:23`), `SegviGenFlowTrainer`
   (`trellis2/trainers/__init__.py:22`, `train.py:57`); upstream's `DinoV3FeatureExtractor`,
   `AdaptiveGradClipper`, `LinearMemoryController`. Every key of the two published configs stays
   accepted: unknown trainer kwargs are set silently (`trellis2/trainers/basic.py:85-86`), so a renamed key such as
   `finetune_ckpt`, `pos_weight` or `disable_final_snapshot` becomes a silent no-op.
3. Checkpoint contents: `state_dict` keys `models.denoiser.*` and the duplicate
   `training_models.denoiser.*`, `ema.state["0.999"]`, `sampler_state`, `elastic_controller`;
   filenames `step{N:07d}-{val/loss:.4f}.ckpt`, `last.ckpt`, `snapshots/step<N>.pt`; the run dir's
   `config.yaml` (`train.py:625`), `command.txt`, `wandb_run_id.txt` (`train.py:374-376`). The init
   `full_seg_rekeyed.pt` is `{"denoiser": <bare state dict>}`. The bridged layout of step 1 of the
   chain (`unrekey_for_eval.py:51-52,99-108`). Attribute and submodule names inside the denoiser
   (they make these keys).
4. `scripts_segvigen/unrekey_for_eval.py <in> <out>` and `scripts_segvigen/ckpt_step.py <ckpt>`: the
   release sidecar calls both (`snapshot_sidecar.sh:23,28`); the parent's
   `segvigen256/bridge.sbatch.tmpl:12-19` runs the falas copy of the first.
5. `scripts_segvigen/live_eval_one.py` (its constants, `load_cond`, the seed rule at `:304`) and what
   it imports at load, `live_eval_thumbs` and `unrekey_for_eval` (`:56-57`): the published dump
   script imports it from Kevin's falas checkout. Its hardcoded falas and jupiter paths
   (`:42-47`) stay until step 10 or 17 replaces the chain.
6. The dump layout the evaluator reads: `<out>/draw<k>/<sid>.npz` with `coords` int16 [N, 3] and
   `pred` float16 [N, 3]; release `<out>/step<N7>/seed<SS>/draw0/`; the sampler values and seed rule
   of the chain above.
7. The data contract of `SegviGenEmissive256`: `data_root`, `parquet_file`, `split_json`,
   `<shape_latent_root>/shape_latents/<shape_latent_name>/`, `<pbr_latent_root>/pbr_latents/<pbr_latent_name>/`
   (`segvigen_256.py:72-73`),
   `emission_latent_dir/<sid>.npz`, `cond_search_dirs` holding `cond_pos.pth` or `cond.pth`,
   `token_counts.json`. Producers: `scripts_segvigen/newdata/*`, `build_output_tex_latents_256.py`,
   `script/star2/newdata_*.sbatch`; the built tree is jupiter `datasets/dataset_release_latents/`
   (README inside).
8. Branches and code in use elsewhere. `retrain-newdata` stays at `bff5c7e`: the release launcher
   sets `BRANCH="retrain-newdata"` (`segvigen_256_release.sh:20`) and pulls a git bundle (`:123-124`);
   the Alliance clones `~/scratch/segvigen-lightning`, the compute-access skill's `hedge.sh`
   (`../segvigen-lightning/script/alliance/segvigen_256_release.sh`) and jupiter
   `trellis2_bw/code_stage/segvigen-lightning-retrain` use it. `main` (= `bf0e9de` = `origin/main`),
   tag `pre-trim-2026-09-16`, `upstream-base`, `release` and `trim-abandoned-2026-09-17` stay (D16).
9. Kevin's falas checkout `/cs/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe` (at
   `bf0e9de`) and the falas SegviGen deploy (`a67a69c`) are what the parent's chain runs today
   (`dump_segvigen256_voxels.py:36-39`, `segvigen256/{bridge,dump_shard}.sbatch.tmpl`,
   `segvigen256/overnight_driver.py:27-28`, `segvigen256/release_dump.sbatch:18-35`); the refactor
   never edits them. Frozen copies on jupiter stay: `archive/2026-09-16-segvigen-lightning-kevin/`
   (bundle of Kevin's refs), `trellis2_bw/eval_inputs/segvigen256_ourval/`,
   `trellis2_bw/eval_outputs/release_eval/code_*`.
10. The parent's filesystem keys `segvigen256` and `_segvi256`; the display name `SegviGen-Emission`.

## Decisions

Carried over from TEXGen and TRELLIS2 (say if one does not apply here):

- D1 Removed files are deleted, not moved to `deprecated/`; git history, the tags (`pre-refactor`,
  `pre-trim-2026-09-16`) and the jupiter bundle keep them. (The 2026-09-16 trim moved them into
  `deprecated/2026-09-16/`.)
- D2 Checks cover both the paper and the release checkpoints.
- D3 Cleanup steps change no behavior.
- D5 Steps that touch training code get the training check: a short run of the real `train.py`
  with the release config, per-step loss compared before and after, plus one resume from a published
  checkpoint. `train.py` sets no global seed (only the sampler shuffle is seeded,
  `trellis2/utils/data_utils.py:115`); the check seeds through a `sitecustomize.py` on `PYTHONPATH`
  (at process start, and again right before `trainer.fit`), not by editing `train.py`.
- D6 Every loss branch and config-switched option in kept code stays (`pos_weight`,
  `balanced_pos_weight`, `p_uncond`, `batch_split`, the per-shape mean loss); only their configs go.
- D7 Work happens on branch `segvigen-emission-refactor`; the parent is touched only at step 16.
- D9 Upstream code we do not use stays untouched, files and blocks alike (upstream's configs, among
  them `slat_flow_imgshape2tex_dit_1_3B_512_bf16.yaml`; upstream's `data_toolkit/` scripts; models,
  pipelines, samplers, the demo files of D12).
- D10 (open, before publishing) Internal scripts in the published code: `script/internal/` holds our
  Slurm accounts and paths. Publish as is, publish a snapshot without it, or move it to the parent
  repo? Recommended: a published snapshot without it.

Dongchen's answers of 2026-10-11 (all answered except D10):

- [x] D4 The repo stays this one (the clone of Kevin's `QhelDIV/segvigen-lightning`, directory
  `SegviGen-Emission/`); the reviewed result lands on a branch `release` ("continue use this repo
  but a new branch named release"). The local `release` today is `bf0e9de`, the 2026-09-16 lane's
  pointer with no commits of its own (`main` and tag `pre-trim-2026-09-16` hold the same commit), so
  step 16 fast-forwards it to the reviewed head; no history is lost. A push needs his go.
- [x] D8 After the merge the parent's chain (Keep stable 9), the Alliance clones and `hedge.sh` move to
  this repo (step 17), once step 10 has given it its own inference path. The jupiter copies stay frozen.
- [x] D11 Superseded by D4: the result lands on `release`, not `main`; `main` (= Kevin's `bf0e9de`)
  and `retrain-newdata` stay where they are. The merge happens only after Dongchen has reviewed the
  branch ("before you merge it to the repo, you should ask me to review before merging").
- [x] D12 Upstream's demo material stays (`app.py`, `app_texturing.py`, `example.py`,
  `example_texturing.py`, `assets/`): "no need" to delete it, as TRELLIS2's D12.
- [x] D13 Our code stays in files of its own beside upstream's, as TRELLIS2's D13:
  `trellis2/models/segvigen_flow.py` (subclass of `ElasticSLatFlowModel`, about 37 new lines in
  `forward`), `trellis2/trainers/flow_matching/segvigen_flow_trainer.py` (subclass of
  `ImageConditionedSparseFlowMatchingCFGTrainer`), `trellis2/datasets/segvigen_256.py` (a standalone
  `Dataset`, taking over the `lightgen_slat.py` helpers it needs, step 8b). TRELLIS2's D13 (answered
  2026-10-10) also moves our hooks out of upstream's `train.py`, `basic.py`, `callbacks.py` into files
  of their own, only the registration lines staying: step 12 does the same here, so both forks of the
  branch agree. Folding into upstream's classes (TEXGen's way) is not done.
- [x] D14 `o-voxel/` stays untouched, our two edits included (`o_voxel/convert/volumetic_attr.py`,
  the `emissiveMipmaps` fix in `src/convert/volumetic_attr.cpp`), and so do our edits to upstream's
  `data_toolkit/{blender_script/dump_pbr.py,voxelize_pbr.py,encode_pbr_latent.py}` (as TRELLIS2's D14).
- [x] D16 The local branches `main`, `retrain-newdata`, `trim-abandoned-2026-09-17` and `upstream-base`
  stay untouched; `release` per D4.
- [x] D17 Only the release launchers and config stay. The paper's (`configs/gen/segvigen_emissive_256.yaml`,
  `script/bw/{launch_segvigen_256_6gpu.sbatch,bootstrap_segvigen_bw.sh,stage_data_256_local.sh,stage_data_256_local.sbatch}`,
  root `build256_array.sbatch`, `test256_build.sbatch`, `convert_cond_fp16_array.sbatch`) go after D21
  puts `bf0e9de`'s config on jupiter beside the checkpoint; this paper run has no other run dir of
  record. (Kept, the paper launcher could not have run anyway: it re-copies Kevin's falas
  `segvigen_256.py`, which imports `lightgen_slat`, over the local one, `launch_segvigen_256_6gpu.sbatch:67-70`.)
- [x] D18 A native predictor on the live classes (option b), step 10, after the cleanup: a new numeric
  path (the lightning package's `modules/sparse/norm.py` and `models/sc_vaes/sparse_unet_vae.py`
  differ from the old repo's vendored copy, so it cannot match the published bytes), checked against
  the published-chain gate on the same GPU and env; the five-seed release test rescore needs its own go.
- [x] D19 The old submodule `../SegviGen-emissive/` goes ("we should only keep the codebase that is
  used in our paper"), at step 16, once step 10 has passed and the release conditioning's route 3
  (`scripts_segvigen/newdata/build_cond_tree.py:14,64-71` imports `img_to_cond`, BiRefNet and
  `DinoV3FeatureExtractor` from the original tree) has a replacement here. Also on its list: the
  parent's 512-lane runners (`../evaluation/newdata_eval/sweeps/job_segvigen.sh`,
  `sweeps/star2/segvigen_worker.sbatch`). It is pinned at `f89ba61`, 33 commits behind the falas
  deploy `a67a69c` the dumps used.
- [x] D20 The names stay: `SegviGenSLatFlowModel`, `SegviGenEmissive256`, `SegviGenFlowTrainer` and
  their files (Keep stable 2); no 2026-09-16 renames.
- [x] D21 The paper checkpoint of record goes onto jupiter in step 1:
  `outputs/segvigen_256_bw_6gpu/ckpts/step0023405_snapshot.ckpt` (10.5 GB) and `step0023405.pt` (the
  bridged file, 2.6 GB), `config.yaml` = `bf0e9de`'s config, `SHA256SUMS` on both ends, a MANIFEST
  row; `df` first, stop under 50 GB free.

## Done

1. (part) Tag `pre-refactor` = `bff5c7e` (the `retrain-newdata` head: the paper code `bf0e9de` plus
   the 10 release-retrain commits) and this worktree (branch `segvigen-emission-refactor`),
   2026-10-06. The tag is local. `.claude/` is already ignored (`.gitignore:11`), so no ignore commit
   was needed.
2. Dongchen answered the decisions on 2026-10-11 (D10 stays open until publishing).

## Steps

### 1. Housekeeping

- [x] Tag `pre-refactor` and the worktree (Done 1).
- [x] Commit this plan on the branch (Dongchen answered the decisions on 2026-10-11).
- [x] D21: the paper checkpoint of record onto jupiter `outputs/segvigen_256_bw_6gpu/`.
- [x] The 2026-09-16 records stay in the parent (`../evaluation/newdata_eval/segvigen256/records/`,
  `compare_dumps.py`, `provenance.py`); step 7 reuses them.
- [x] `../docs/status/refactor.md` says SegviGen-Emission was "not restructured": step 16's list.

**Instructions:**

**Result:** Plan `9d0fce5`. D21 done 2026-10-12: jupiter `outputs/segvigen_256_bw_6gpu/` holds the snapshot
(sha256 `fc7a9bd1…` on both ends and as recorded 2026-09-16), the bridged file (md5 as recorded),
`bf0e9de`'s config and `SHA256SUMS`; MANIFEST row added.

### 2. Dead root files, one-off scripts and the live-evaluation loop

Not used by either run's training, data or published inference.

- [x] Root: the 10 PNGs of the `lightgen` branch (`emission_*.png`, `vae_channel_vis.png`;
  `.gitignore:22-23` matches some).
- [x] Kevin's probes and smokes: root `probe_batch`, `probe_bs_venus05`, `probe_bs_venus05_v2`,
  `probe_bs_256_venus05`, `smoke_dit10`, `smoke_resume`, `smoke_segvigen`, `smoke_segvigen_resume`,
  `smoke_256_venus05`, `count_256_split`, `count_256_split_fast`, `run_fp16_check{,2,3}`,
  `run_verify_fp16{,_v2}` (all `.sbatch`); `script/bw/smoke_{gpu,train}_venus05.sbatch`;
  `scripts_segvigen/{check_256_coords_vs_shape,check_cond_fp16_forward,verify_cond_fp16,round_trip_check,_drain_backlog,_submit_cond_ablation,_repair_cond_ablation}.py`.
- [x] Kevin's live-evaluation loop (the dashboard, not the published numbers):
  `scripts_segvigen/{live_eval_driver,live_eval_merge,live_eval_slices,submit_eval_chain,rescore_baseline_dump}.py`,
  root `live_eval_{bridge,driver,merge,one,shard}.sbatch`, `rescore_baseline.sbatch`. Kept:
  `live_eval_one.py`, `live_eval_thumbs.py`, `unrekey_for_eval.py`, `ckpt_step.py` (Keep stable 4–5).
- [x] `data_toolkit/`, the `lightgen` branch's 63 added files except the three release producers
  `dual_grid_from_glb.py`, `encode_shape_from_dual_grid.py`, `encode_pbr_from_voxels.py`
  (`script/star2/newdata_encode.sbatch:28,30`, `scripts_segvigen/newdata/build_dual_grid.py:43`),
  `test_checkpoint_cadence.py` (it tests the kept `build_checkpoint_callback`) and
  `test_emission_vae_resume.py` (goes with its trainer in 8c). Deleted: `ablate_zpbr_v3.py`,
  `aggregate_{emission,latent}_stats.py`, `analyze_dark_regions.py`, `bake_{albedo,pbr}2emission_v3_init.py`,
  `blender_script/dump_pbr_nonormalize.py`, `bridge_voxel_to_uv.py` and its test, `check_74k_progress.py`,
  `compute_normalization_stats.py`, `datasets/lightgen_74k.py`, `diag_{dit_decode,emission_range}.py`,
  `encode_10_samples.py`, `encode_emission_latent{,_10samples}.py`,
  `encode_emission_pbrfinetune_{10samples,dataset}.py`, `encode_emission_twostream_dataset.py`,
  `encode_single_sample_{latents,shape_latent}.py`, `extract_74k_tars.sh`, `patch_dit_config_stats.py`,
  `predict_emission_dit_74k.py`, `predict_emission_voxels{,_twostream}.py`,
  `prep_{10sample_as1k,10sample_dataset,1k_dataset}.py`,
  `prep_74k_{bootstrap,dual_grid,dump,dump_dualgrid_stream}.py`, `regenerate_1k_dual_grid.py`,
  `stage_test_voxels.py`, `test_emission_dit{,_10samples,_10samples_pbrvae,_1k,_74k}.py`,
  `test_emission_vae_e2e.py`, `test_pbr_vae.py`, `test_single_emission_sample.py`,
  `verify_albedo2emission_v3/`, `verify_pbr2emission_v3/`, `vis_emission_vae.py`,
  `viz_uv_{albedo,alignment,gallery}.py`.
- [x] `vis/` (5 files, the `lightgen` branch's viewers).

Gate: live-code subset; the published-chain gate (the loop's removal must not touch what
`live_eval_one.py` imports).

**Instructions:**

**Result:** 109 files deleted (10 PNGs, 18 one-off launchers, 7 one-off scripts, the live-evaluation loop's 11
files, 58 `data_toolkit/` files, `vis/`); `CLAUDE.md` replaced by a short interim note (the old guide named the
deleted files; step 15 writes the doc of record). Live-code gate 20/20 + 20/20 identical; published chain identical.

### 3. `archive/`, `docs/`, `reports/`, stale ignore rules

- [x] `archive/` (30 files, the `lightgen` branch's 2026-07-09 cleanup), `docs/` (9: its April–June
  plans and specs), `reports/` (2 status reports).
- [x] `.gitignore`: the `lightgen` branch's rules for its emission lanes and PNGs; Kevin's block
  (`pysite/`, `data/`, `outputs/`, `log/`, `wandb/`, `__pycache__/`, `*.pt`) stays. Check whether an
  unanchored `data/` also matches tracked files under `o-voxel/third_party/` (TRELLIS2 found 12) and
  that nothing new shows as untracked.

**Instructions:**

**Result:** `archive/` (30), `docs/` (9), `reports/` (2) deleted; `.gitignore` keeps `.claude/` of the `lightgen`
block and anchors Kevin's `data/` as `/data/` (12 tracked eigen files were matched); the registry comment naming
`archive/` removed. Live-code gate 20/20 + 20/20 identical.

### 4. `blender_packages/` (upstream's demo material stays, D12)

- [x] `blender_packages/` (133 files: Pillow vendored for Blender). Its users are the `lightgen`
  branch's Blender prep (deleted in steps 2 and 5) and upstream's `data_toolkit/dump_pbr.py`, whose
  Blender script prepends the directory to `sys.path` (our edit, D14). Neither run uses Blender; check
  that the script still finds a Pillow without the directory, else keep it or ask (TRELLIS2 deleted
  it in its step 4, `d31f794`).

**Instructions:**

**Result:** `blender_packages/` (133 files) deleted; `dump_pbr.py`'s path line already checks that the directory
exists (D14 keeps it) and upstream's `install_pillow.py` route applies, as in TRELLIS2. Live-code gate 20/20 + 20/20.

### 5. Launchers (per D17)

`script/` holds 67 files (alliance 2, bw 8, fir 25, star2 28, vulcan 4); plus the root `.sbatch`
runners not handled in step 2.

- [x] Kept (release): `script/alliance/{segvigen_256_release.sh,snapshot_sidecar.sh}`,
  `script/star2/newdata_{dual_grid,encode,cond,pack}.sbatch`.
- [x] Deleted per D17 (the paper's): `script/bw/{launch_segvigen_256_6gpu.sbatch,bootstrap_segvigen_bw.sh,stage_data_256_local.sh,stage_data_256_local.sbatch}`,
  root `build256_array.sbatch`, `test256_build.sbatch`, `convert_cond_fp16_array.sbatch`.
- [x] Deleted: Kevin's 512 lane, never trained (`script/bw/{launch_segvigen_bw_6gpu.sbatch,extract_staged_shards.sh}`,
  root `stage_shards_{train,val}.sbatch`, `extract_val_only_venus05.sbatch`,
  `scripts_segvigen/stage_shards.py`); the `lightgen` branch's `script/fir/` (25), `script/vulcan/`
  (4) and its 24 files in `script/star2/` (emission VAE and DiT, PBR VAE finetune, 74k prep).

**Instructions:**

**Result:** 66 files deleted: the paper's launchers (D17), Kevin's 512 lane, the `lightgen` branch's `script/fir/`,
`script/vulcan/` and 24 `script/star2/` launchers; kept the release's six. `BLACKWELL.md` notes its cited scripts are
at tag `pre-refactor`. Live-code gate 20/20 + 20/20 identical.

### 5b. Move the kept cluster scripts into `script/internal/`

`script/internal/{alliance,star2}/` (and `bw/` if D17 keeps it). Nothing pulls this branch, so the
move is safe here; at the merge it changes the path `hedge.sh` uses (step 16's list). Check the
paths inside the scripts (`segvigen_256_release.sh:20` names the config; `newdata_dual_grid.sbatch:17,20-21`
reaches into the parent repo).

**Instructions:**

**Result:** `script/{alliance,star2}/` → `script/internal/{alliance,star2}/` (6 files, `git mv`); the launcher's three
sidecar calls, the usage comments and the release config's comment follow; `bash -n` passes, the parsed config is
unchanged. Parent doc `docs/baselines/segvigen-live-training.md:126-127` goes to step 16's list.

### 6. Configs (per D17)

- [x] Kept: upstream's configs (D9) and `configs/gen/segvigen_emissive_256_release.yaml`; the paper's
  `segvigen_emissive_256.yaml` goes (D17; its copy of record is jupiter `outputs/segvigen_256_bw_6gpu/config.yaml`).
- [x] Deleted: `configs/gen/segvigen_emissive_direct.yaml` (512 lane) and the `lightgen` branch's 25
  `emission_*` configs (12 under `configs/gen/`, 13 under `configs/scvae/`). The training check brings
  its own config (step 7).

**Instructions:**

**Result:** 27 configs deleted (the paper's, the 512 lane's, the `lightgen` branch's 25 `emission_*`); three comments
naming them now point at jupiter or tag `pre-refactor` (syntax trees and the parsed release config unchanged).
Live-code gate 20/20 + 20/20; published chain identical (5/5, 20/20, bridge 640/640).

### 7. Check setup: live-code gate, published-chain gate, training check, noise baselines (no code change)

All runs on tag `pre-refactor`, in parallel across machines. Tooling goes to `script/internal/gate/`;
data and run outputs outside the repo.

- [x] The checkpoints, named, never picked: the release `snapshots/step0044722.pt` and `ckpts/last.ckpt`
  with `config.yaml` (rsync, sha256 against `.backup_sha256.txt`; `df` first); the paper
  `step0023405_snapshot.ckpt` read from the falas mirror (sha256 `fc7a9bd1…`) and the bridged
  `step0023405.pt`.
- [x] The live-code gate tool (§ Runs and the gate) and a numeric comparison: per shape and tensor,
  byte identity, mean and max |d|; NaN on one side fails; verdict EMPTY on an empty shapes file.
  TEXGen's `scripts/internal/gate/compare_numeric.py` is the pattern.
- [x] The subsets, fixed in files. Live-code gate: 20 paper validation shapes (the paper's emission
  latents cover train and val only), 20 of the release `trellis2_bw/eval_outputs/release_eval/test200.txt`
  if the release emission latents cover them, else of `val200.txt`. Published-chain gate: the five
  test shapes of `records/smoke_sids.txt` and 20 of the release test shapes. Inputs: paper from
  jupiter `trellis2_bw/latents_v2/{shape,pbr}_latents/*_256`, the emission npz on falas
  (`segvigen_emissive/outputs/latents_256/output_tex_latents/segvigen_emissive_mask_256`) and the
  conditioning dirs in the published search order (falas `segvigen_emissive/outputs/cond_fp16`,
  jupiter `segvigen_emissive/dataset_direct/{train_72k_agentic,val_72k,test_72k}`); release from jupiter
  `datasets/dataset_release_latents/`. Check that `SegviGenEmissive256` admits every gate shape
  (`trellis2/datasets/segvigen_256.py:100-108` skips a shape missing any input).
- [x] Live-code gate noise: two runs per GPU type (the workstation 4090 if the 1.3B denoiser fits
  for inference; Solar L40S with `~/envs/trellis2`, whose flex_gemm has no sm_86 kernels, so no A40).
- [x] The published-chain gate: the dump script copy with `PROBE` from the tree under test; baseline
  and noise run on A40 in Kevin's env (as `segvigen256/release_dump.sbatch`; never on Blackwell nodes,
  `:11`); the numbers against the published A40 dumps recorded once.
- [x] The training check through the real `train.py` and `SegviGenFlowTrainer`:
  - a tiny training root from `datasets/dataset_release_latents/` (a few train and two val shapes:
    shape, PBR and emission-mask latents, conditioning, metadata), fixed split;
  - a check config extending the release config: local paths, batch 1, a few dozen steps, one
    validation, one save, wandb off, `disable_final_snapshot=true` as both launchers pass;
  - the `sitecustomize.py` of D5, recording per step the batch ids, lr, loss, gradient norm and weight
    sum, and for the saved file its keys, EMA, `sampler_state` and `elastic_controller`;
  - fresh (from `full_seg_rekeyed.pt`) and resume from the release `ckpts/last.ckpt` (step 49,270)
    and from the paper `step0023405_snapshot.ckpt`, so the check shows the code under test still loads
    both published files;
  - memory: the 1.3B denoiser with AdamW and EMA likely does not train on a 24 GB 4090 even at batch 1
    (measure; else Solar L40S); which env trains here (`~/envs/trellis2` or Kevin's) is measured too.
  Not covered: `batch_split`, multi-GPU DDP, the Blackwell env of the paper run.
- [x] The existing test `data_toolkit/test_checkpoint_cadence.py` on `pre-refactor`; record the result.

**Instructions:**

**Result:** Tooling and record in `script/internal/gate/` (README). Two `pre-refactor` runs agree exactly in
all three checks: live-code gate on the workstation 4090 and sp4090 (also identical to each other), published
chain on A40 (also byte-identical to the published dumps), training check on L40S. Pass bar: identity.

### 8. Our modules (per D13)

- [x] 8a. The 512 lane's dataset `trellis2/datasets/segvigen_direct.py` (`SegviGenEmissiveDirect`) and
  its registry line (`datasets/__init__.py:22`). First, because it imports `_sparse_collate` from
  `lightgen_slat` (`segvigen_direct.py:45`). Live-code subset.
- [x] 8b. The dataset helpers. `segvigen_256.py` imports `_build_parquet_samples`, `_read_npz_as_sparse`
  and `_sparse_collate` from the `lightgen` branch's `trellis2/datasets/lightgen_slat.py`
  (`segvigen_256.py:46`; defined at `lightgen_slat.py:138,78,95`), and `_build_parquet_samples` calls
  `_apply_split_indices` (`:180`, called at `:175`) and reads `_DEFAULT_SPLIT_JSONS` (`:129`). The four
  functions and the constant move verbatim into `segvigen_256.py`, then `lightgen_slat.py` (its
  `LightGen*` classes at `:210,361`, used by neither run) and its registry lines
  (`datasets/__init__.py:18-21,53`) go. Full live-code gate and the training check.
- [x] 8c. Our trainers used by neither run: `trellis2/trainers/vae/emission_{vae,pbr_finetune,pbr_conditional}.py`,
  `trellis2/utils/model_surgery.py`, their `TRAINER_MAP` entries (`train.py:46-48`; `train.py` imports
  every map entry at load) and registry lines, and `data_toolkit/test_emission_vae_resume.py`.
  Training check.
- Per D13 (a) the model and trainer files stay as they are. The 2026-09-16 plan and the abandoned
  trim are references for 8b only; take neither their renames nor their `deprecated/` moves.

**Instructions:**

**Result:** 8a `e653a13` (512-lane dataset), 8b `c2fcc59` (helpers moved verbatim into `segvigen_256.py`, syntax trees
identical; `lightgen_slat.py` gone), 8c (the three `Emission*` trainers, `model_surgery.py`, their test and map lines). Every
sub-step: live-code gate identical (8b full: 183/183, 200/200), training check fresh/resume/resume_paper identical.

### 9. Coverage run (no code change)

The training check (fresh and resume, validation, a save) under a line-coverage tracer, plus the
live-code gate, the published-chain gate, and the release producers on a few shapes. Output: per
file, the lines that never run. "Never ran" is not "dead" for multi-GPU and `batch_split`, the
`pos_weight` branches (off in both runs), the elastic memory controller's branches, or the Blackwell
switches (`ATTN_BACKEND`, `SPARSE_ATTN_BACKEND`, `FLEX_GEMM_*`).

**Instructions:**

**Result:** At `da9ecd1` (training fresh/resume/resume_paper, published chain, live-code gate on 5+5 shapes): 445/840 of our
executable lines ran; report `C/coverage/step9/report.txt`. Not run: wandb paths, the image-encoder path, `pos_weight` (D6),
`live_eval_one.py`'s own CLI, timing branches. The release producers were not imported; step 11 ran the two it changed (untraced).

### 10. An inference path in this repo (per D18; needs Dongchen's go)

Under D18 (b): `data_toolkit/predict_emission_mask_voxels.py` (name per D20) on the live classes:
loads a Lightning or a bridged file, the shipped sampler values, the seed rule, the dump layout of
Keep stable 6, every data root a required argument; mask × albedo output as
`segvigen256/dump_ma_reuse.py` makes it. Checks: against the published-chain gate on the same GPU and
env, with bars set from step 7's noise (the 2026-09-16 bars: mask IoU ≥ 0.95 for masks of 100 voxels
or more, XOR ≤ 5 below, coordinate-set difference ≤ 0.5%); then the five-seed release test rescore
(its own go) against IoU 0.1468 ± 0.0083 (per seed 0.1542, 0.1420, 0.1559, 0.1361, 0.1457:
`../evaluation/newdata_eval/segvigen256/records/release_retrain_val_selection_perpart.tsv:16-20`).

**Instructions:**

**Result:** `data_toolkit/predict_emission_mask_voxels.py`; its `decode_as_published` runs the stock decoders as the chain's
vendored copy did (fp16 after `from_latent`, no fallback child): the whole gap. A40, Kevin's env: 1,816 published test dumps remade (paper
seeds 0–3, release 0–4), byte-identical but for the 3 smallest shapes per test set, where the published chain does not repeat itself
(gate README § 4); paper seed 4 (RTX 6000 Ada) unchecked. No rescore: the remade dumps carry the published numbers.

### 11. The chain scripts in `scripts_segvigen/`

- [x] Release producers: `newdata/{build_metadata,build_dual_grid,build_cond_tree,verify_and_pack}.py`,
  `build_output_tex_latents_256.py` (the emit stage; hardcoded paths `:61,81,83`), `rekey_full_seg.py`
  (the init, both runs), `build_token_manifest.py` and `convert_cond_fp16.py` (paper producers; the
  release reused 30,102 of the paper's conditioning files: jupiter
  `datasets/dataset_release_latents/README.md:43`): keep or delete
  after checking who calls each.
- [x] The bridge and eval scripts of Keep stable 4–5 keep their paths until step 10 or 17.
- [x] Hardcoded paths and dead branches; `build_cond_tree.py`'s route 3 into the original SegviGen
  tree (D19); the parent-repo imports: `newdata_dual_grid.sbatch:20-21`
  (`data_processing.data_preparation.config`) and `build_dual_grid.py:66-70`
  (`data_processing.data_preparation.{contract,loader}`, `utils.glb_io`).
- [x] Placement: `scripts_segvigen/` → `data_toolkit/` beside upstream's producers, or keep the folder
  (decided here, D13).
- Check: each producer on a few release shapes before and after, outputs compared byte for byte, and
  against jupiter `datasets/dataset_release_latents/`.

**Instructions:**

**Result:** 11a `9d26e84` (paper producers `convert_cond_fp16.py`, `build_token_manifest.py`), 11b `68b2099` (emit stage: release
mode only, byte-identical on 5 shapes), 11c (route 3 on this repo's code via `newdata/thumbnail_cond.py`, MIT notice kept; paper roots
optional; 11 shapes byte-identical old vs new in Kevin's env). The folder stays `scripts_segvigen/`. Step 14: the dual grid imports the parent repo.

### 12. Our hooks in upstream's `train.py`, `basic.py`, `callbacks.py`

Ours in upstream files (`git diff 98c2548`): `train.py` (the `TRAINER_MAP` entries `:42-57`, the
wandb run-id restore `:374-376`, `build_checkpoint_callback` `:416`, `CheckpointTimingCallback` with
Kevin's `batch_log_every` passthrough `:487`, Kevin's `disable_final_snapshot` passthrough `:496`),
`trainers/basic.py` (`_resolve_hf_path` and `.safetensors` loading, the `train/lr` log, the `[val]` and
`[train]` prints, the `visualize_sample` guard in `snapshot_dataset`), `trainers/callbacks.py`
(`CheckpointTimingCallback`, Kevin's peak-VRAM log `:261-272` used only by probes,
`disable_final_snapshot` on `SnapshotCallback`), and the small ones: `utils/data_utils.py`
(`ResumableSampler.set_epoch`, an upstream bug; `BalancedResumableSampler` inherits it),
`trainers/data.py` (empty validation list), `flow_matching/mixins/image_conditioned.py` (DINOv3 layer
path), the three registries. Decide what stays; any change here changes how training runs behave.
Every hook both forks carry gets the same decision as TRELLIS2's step 12. Training check for each
change.

**Instructions:**

**Result:** `trellis2/trainers/lightgen_hooks.py` = TRELLIS2's file plus the peak-VRAM line and a `SnapshotCallback` with
`disable_final_snapshot`; `callbacks.py` is upstream's again; kept in upstream files the same hunks TRELLIS2 kept. Training check
identical in all three modes (also the `[train]`/`[val]`/`[ckpt-timing]` lines); live-code gate 20/20 + 20/20.

### 13. Environment files

- [x] `setup.sh` (upstream's pins, torch 2.6 + CUDA 12.4; our one change: the `libjpeg-dev` line
  commented out); no requirements file exists.
- [x] The envs that ran: `segvigen-bw` (the paper training: cs-venus-05 Blackwell, Kevin's
  `/localscratch/xya120/miniforge3`, built by `script/bw/bootstrap_segvigen_bw.sh:46-101`; facts in
  `BLACKWELL.md`); the Alliance venv `~/scratch/TRELLIS.2-lightning/env` (the release training, built
  by TRELLIS2's `script/alliance/build_env.sh`); Solar `~/envs/trellis2` (the release producers; sm_89
  for flex_gemm); Kevin's jupiter miniforge `trellis2` (every published dump). One description that
  matches what runs.
- [x] If a dependency goes, update the import probes in the launchers.

**Instructions:**

**Result:** `CLAUDE.md` lists the five envs that ran this code, their versions and what ran in each, and what every env needs
(Kevin's env has no lightning: inference and conditioning only). No env file changed; no import probe needed an update.

### 14. Release layer: generic entry points for LightgenBench users

No user or cluster paths in this layer. Needs steps 10 and 11; its README commands are written in
step 15.

- [x] The inputs. The Hugging Face release carries no geometry and no latents, so a user builds
  them: TexVerse GLBs → dual grid → shape latent (`encode_shape_from_dual_grid.py`); released PBR
  voxels → PBR latent (`encode_pbr_from_voxels.py`);
  emission voxels → emission-mask latent (`build_output_tex_latents_256.py`); thumbnails → DINOv3
  conditioning (`build_cond_tree.py`, after D19's route 3 question); metadata and splits
  (`build_metadata.py`), check and pack (`verify_and_pack.py`). The dual grid of record is
  `scripts_segvigen/newdata/build_dual_grid.py`, not `dual_grid_from_glb.py` alone: it runs that
  script's routine with the bake's mesh loader and normalization swapped in from the parent repo
  (`load_and_merge`, `TIGHT_SCALE_NUM`, `resolve_glb`; `build_dual_grid.py:66-82`) and checks the
  occupied cells against the released PBR voxels (`:113-116`); a public recipe ports those parent
  pieces or depends on them. One generic script or README sequence; check on a few shapes that it
  reproduces jupiter `datasets/dataset_release_latents/`.
- [x] The init: `full_seg_rekeyed.pt` from Hugging Face `fenghora/SegviGen` `full_seg.ckpt` through
  `rekey_full_seg.py`.
- [x] Training: the release config plus the launcher's overrides (`segvigen_256_release.sh:93-106`:
  4 × 12, `batch_split`, `ckpt_last_every_n_steps` = steps per epoch, `disable_final_snapshot=true`,
  the data knobs); the config's two leftover jupiter `cond_search_dirs` (`:44-45`, replaced by the
  launcher's override) go; the README states the GPU memory needed.
- [x] Inference: step 10's predictor on the release test split.
- [x] The internal launchers stay as they are.

**Instructions:**

**Result:** Part 1 `681cbfd`: `script/lightgenbench/{prepare_inputs,get_init,train}.sh`; the release config's two jupiter
`cond_search_dirs` go. On a 4-shape mini release (Solar L40S): dual grids byte-identical to jupiter's, latents within 3.4e-3, route-3
conditioning within 6e-4 mean, init identical (640/640), `train.sh` dry run trains a step. Part 2: `predict.sh` (seeds, split = the
usable rows of `release_index.parquet`); on jupiter's release inputs (A40, Kevin's env) it remakes the seed-0 test dumps, byte-identical
but for the step-10 small shapes (masks 198/200, mask × albedo 197/200); on the 4090 mini release it runs.

### 15. Docs

One doc of record, keeping upstream's attribution and citation (`README.md`).

- [x] Inside: `README.md` (upstream's), `CLAUDE.md` (today the `lightgen` branch's TRELLIS.2-Emission
  guide, wrong for this repo), `BLACKWELL.md` (its facts into the doc), `data_toolkit/README.md`; the
  two runs, the checkpoints of record with hashes, the chain, what the published numbers ran on. The
  2026-09-16 spec's doc outline is a source for content, not taken whole.
- [x] The release commands (step 14).

**Instructions:**

**Result:** Part 1 (this commit): `CLAUDE.md` is the doc of record (ours, the two runs and their checkpoints, the published chain,
the environments with `BLACKWELL.md`'s facts, traps); `README.md` gets a SegviGen-Emission section on top (the LightgenBench
commands, credits) with upstream's below; `BLACKWELL.md` goes. Part 2: the README's prediction section (`predict.sh`, the evaluator call).

### 16. Dongchen's review, the merge into `release` (D4, D11) and parent registration (needs his go)

- [ ] The full live-code gate, the published-chain gate and the training check on the branch head,
  against the `pre-refactor` baselines; the tests.
- [ ] Ask Dongchen to review the branch (`git diff pre-refactor`, the plan's Results); stop until he
  gives the go.
- [ ] With the go: fast-forward `release` (today `bf0e9de`, an ancestor of `pre-refactor`) to the
  reviewed head. A push to `origin` (Kevin's `QhelDIV/segvigen-lightning`) needs its own go and
  push access.
- [ ] In the parent, once the commit is reachable from a remote: one commit with the `.gitmodules`
  entry and the gitlink together, and `/SegviGen-Emission/` out of `.git/info/exclude`; per D19, the
  old submodule's removal in its own commit.
- [ ] Parent docs: `../AGENTS.md` (the submodule table lists `SegviGen-emissive/`; layout, env and
  task tables, the docs index), `../docs/baselines/segvigen.md`, `../docs/baselines/segvigen-live-training.md`
  (§ Where the code is), `../docs/status/refactor.md` (the SegviGen-Emission bullet),
  `../docs/status/checkpoints.md`, `../docs/evaluation.md`; a path-scoped rule
  `../.claude/rules/segvigen.md` beside `texgen.md` and `trellis2.md` (propose it).
- [ ] Outside the repo: the compute-access skill's `hedge.sh` if step 5b moved the release launcher
  (step 5b did: the launcher is `script/internal/alliance/segvigen_256_release.sh`; `hedge.sh` takes the path as an
  argument, so only its callers' notes change).
- [ ] Parent doc paths after step 5b: `../docs/baselines/segvigen-live-training.md:126-127` names
  `script/star2/newdata_*.sbatch` and `script/alliance/segvigen_256_release.sh`.

**Instructions:**

**Result:**

### 17. Cut-over (per D8)

- [ ] The parent's chain (Keep stable 9) from Kevin's falas checkout to this repo:
  `dump_segvigen256_voxels.py`, `segvigen256/{bridge,dump_shard}.sbatch.tmpl`, `release_dump.sbatch`,
  `overnight_driver.py`.
- [ ] The Alliance clones and `hedge.sh` to `release`.
- [ ] The jupiter copies stay frozen.

**Instructions:**

**Result:**

## Later, only if wanted (behavior changes, one step each)

- [ ] **Checkpoint labels.** A `step<N>-<valloss>.ckpt` file pairs step N's weights with the previous
  validation's loss (top-k fires on the step trigger with the last logged metric):
  `step0043206-0.3069.ckpt` holds step 43,206 with 42,448's loss; the paper run's files too.
  `build_checkpoint_callback` is shared with TRELLIS2.
- [ ] **Snapshots.** The sidecar can skip a snapshot (step 758 in the release run; logged since
  `bff5c7e`), and the final step is never saved (`last.ckpt` is 49,270 of 50,000 steps).
- [ ] **A global seed in `train.py`** (none today).
- [ ] **Silent loads**, as TRELLIS2's Later list: `strict=False` loads, unknown trainer kwargs set
  silently, a missing `finetune_ckpt` is a no-op.
- [ ] **Our model and trainer folded into upstream's classes** (D13 (b)).
- [ ] **The paper config's batch**: the yaml says 4 per GPU, the launcher ran 8 (only if D17 keeps it).
- [ ] **Tests**: the 2026-09-16 plan's designs (data contract, config resolution, seed rule,
  checkpoint key mapping, training hooks).

## Out of scope

- `o-voxel/` (D14).
- The parent's evaluation chain beyond step 17; the evaluator's `load_mesh` UV drop
  (`../evaluation/pointsampled/geometry.py:6-17`).
- Re-scoring any published number, beyond the rescore D18 needs (its own go).
- Kevin's `main` and his remote; a message to Kevin (Dongchen 2026-09-16), unless D4 needs one.
- `lightgen` commits after `202bfb0`.
- The position frame (voxels in [−0.5, 0.5], atlas in [−1, 1]): tracked in `../docs/status/refactor.md`.
- The paper text.
