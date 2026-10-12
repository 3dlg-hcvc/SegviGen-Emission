# Blackwell (sm_120) environment facts

> The launchers, bootstrap and staging scripts this file cites (`script/bw/`, the `lightgen` branch's `script/star2/` launchers) were deleted in the refactor's step 5 (D17), the paper's `scripts_segvigen/convert_cond_fp16.py` in step 11; they are at tag `pre-refactor`. Step 15 folds these facts into the doc of record.

What actually made training work on the RTX Pro 6000 Blackwell cards
(cs-venus-05/19/21), gathered before running anything for real on venus-05
(owner instruction, board `lightning_port`, 2026-08-26) and applied in
`script/bw/launch_segvigen_256_6gpu.sbatch` and
`script/bw/bootstrap_segvigen_bw.sh`. Each fact below is attributed to
where it came from: Dongchen's prior venus-19 launcher (already
sm_120-proven), his `bootstrap_bw05.sh` recipe, or our own measurement on
venus-05 this run.

## Attention backend

**Fact:** flash-attn has no sm_120 support. Set `ATTN_BACKEND=sdpa` and
`SPARSE_ATTN_BACKEND=xformers`.

**Source:** Dongchen's venus-19 launcher,
`script/star2/emission_dit_pbr2emission_74k_v3_nonzero_nocopy_venus19.sh`
(commit `1dc7824`), lines documenting "flash-attn unsupported ->
ATTN_BACKEND=sdpa + SPARSE_ATTN_BACKEND=xformers (venus-19-proven)".

**Our confirmation (venus-05, job 257555):** `flash_attn` import fails
with `ModuleNotFoundError`, as expected. A standalone `xformers.ops.
memory_efficient_attention` unit test with **float32** inputs also failed
(`capability (12, 0) too new` for the fp32 fallback kernels -- the fa2/fa3
xformers kernels only support bf16/fp16 on this card). That failure is a
property of the isolated test, not of real training: training runs under
`mix_precision_mode: amp, mix_precision_dtype: bfloat16`, and under that
autocast xformers works correctly -- confirmed by 30-step (job 257857) and
270+-step (job 257933, still running as of this writing) real training
runs with healthy, non-NaN losses. **Lesson for anyone smoke-testing
xformers standalone on sm_120: test with bf16/fp16 tensors, not fp32, or
the test will report a false failure.**

## flex_gemm autotune cache

**Fact:** flex_gemm's autotune file-cache races across ranks when it lives
on a shared filesystem. Set `FLEX_GEMM_USE_AUTOTUNE_CACHE=0` and
`FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0` (in-memory-only autotune).

**Source:** Dongchen's venus-19 launcher (same file/commit as above).

## Lightning SLURM launcher vs. native spawn

**Fact:** Lightning's SLURM-coordinated launcher errors (`ntasks=1` vs.
`devices=N` mismatch) under the star2 single-Slurm-task convention (one
task, `--num_gpus N` native-spawns Lightning's own N processes -- NOT
`--ntasks-per-node=<gpus>`). Set `SLURM_JOB_NAME=bash` to make Lightning's
SLURM-environment detection back off and let native spawn happen.

**Source:** Dongchen's venus-19 launcher (same file/commit as above),
where it is documented as "star2 single-task convention, same as
script/star2/emission_dit_1k_probe.sh".

## CUDA toolchain

**Fact:** `CUDA_HOME=/usr/local/cuda-12.9` (on venus-05/19), with
`PATH`/`LD_LIBRARY_PATH` extended from it, and
`TORCH_CUDA_ARCH_LIST="12.0"` so any JIT-built extension targets sm_120
specifically rather than the default arch list.

**Source:** Dongchen's `bootstrap_bw05.sh`
(`/3dlg-jupiter-project/lightgen/trellis2_bw/bootstrap_bw05.sh`), step
"[3/6] CUDA toolchain exports".

## Torch / xformers build

**Fact:** use a SEPARATE conda env from the normal `trellis2` env (which
only supports up to sm_90/Ampere-class cards and will silently misbehave
or fail on sm_120). Install `torch`/`torchvision`/`xformers` from the
cu128 wheel index (`--index-url
https://download.pytorch.org/whl/cu128`), not the default PyPI index.
No flash-attn install step at all (see above).

**Source:** `bootstrap_bw05.sh`, step "[2/6] torch cu128 + xformers";
reproduced verbatim (env renamed `segvigen-bw`, everything else the same
recipe) in `script/bw/bootstrap_segvigen_bw.sh`.

## HF weights: local cache, no live fetch

**Fact:** set `HF_HOME` to a pre-populated cache and `HF_HUB_OFFLINE=1`.
Blackwell nodes' outbound network path to huggingface.co is not something
to depend on mid-run; a live fetch attempt with `HF_HUB_OFFLINE` unset
will stall a job that should otherwise start immediately.

**Source:** `bootstrap_bw05.sh` step "[5.5/6] HF caches", which rsyncs
`$J/hf_stage/<model>` into `~/.cache/huggingface/hub` if not already
present, and Dongchen's venus-19 launcher setting
`export HF_HOME=/localscratch/dya78/hf_home` after noting "venus-19 root
FS was 100% full -> HF cache relocated to /localscratch". Our own
bootstrap does the same rsync-if-missing for
`facebook/dinov3-vitl16-pretrain-lvd1689m` (the one HF model our port
needs live, since everything else is a local checkpoint file).

## Bootstrap idempotency

**Fact:** the bootstrap script must tolerate being re-run after a partial
failure without redoing completed work or getting stuck on a half-created
state: recreate a conda env if it exists but has no working `python`
(half-created from a cancelled prior run), skip `pip install` steps whose
package already imports successfully, skip `git clone` if `.git` already
exists (fetch/pull instead), skip HF rsync if the model dir is already
present.

**Source:** `bootstrap_bw05.sh`'s own idempotency checks throughout (e.g.
"a cancelled earlier run can leave a half-created env; nuke if python
missing"), reproduced in `script/bw/bootstrap_segvigen_bw.sh`.

## Private repo access

**Fact:** unlike Dongchen's public `TRELLIS.2-lightning` repo (which
`bootstrap_bw05.sh` clones over anonymous HTTPS), our fork
(`segvigen-lightning`) is **private**, so it needs SSH key or token auth
from the compute node.

**Our confirmation (venus-05, job 257491):** the node already had a
working SSH deploy key/agent for the QhelDIV GitHub account --
`git clone git@github.com:QhelDIV/segvigen-lightning.git` succeeded with
no extra setup. This had been flagged as an open question before the
bootstrap was actually run; it is resolved, not still open.

## Conditioning tensor dtype (relevant when re-staging)

Not an sm_120 fact, but it bites anyone staging this lane onto a node:
store the precomputed DINOv3 `cond_pos.pth` as **bfloat16**, not fp32 or
fp16. The trainer runs `mix_precision_dtype: bfloat16`, so autocast casts
cond to bf16 inside every forward anyway, which makes bf16 storage
bit-identical to fp32 at half the bytes (measured: cosine 1.00000000 and
loss delta exactly 0.0 across 20 shapes, job 258059).
`scripts_segvigen/convert_cond_fp16.py` now defaults to bf16.

The set currently staged on venus-05 is **fp16**, written before that
measurement existed; the owner ruled to keep it (measured effect 0.014%
relative loss shift, roughly 1000x below minibatch gradient noise). Full
numbers for both dtypes, and the scale context behind that call, are in
that script's docstring. Read it before reconverting.

## What is NOT documented here

Data staging (node-local disk vs. NFS, the jupiter throughput ceiling that
motivated moving the 256 lane's data local) is a separate, already-fixed
problem -- see `script/bw/stage_data_256_local.sh` and its header comment,
not an sm_120-specific fact.
