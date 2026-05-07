# LightGen 74k — Voxelization & Latent-Encoding Pipeline

**Date:** 2026-05-07
**Branch:** `lightgen`
**Author:** Claude (with @dya78)

## 1. Problem & goal

The emission DiT and emission VAE work has been validated on a 1k subset
(1,093 samples under `data/lightgen_1k/`). To scale training to the full
emissive pool, we need shape latents and PBR latents prepared for the
**74k subset** defined by
`data_splits_emissive_74k_pinned.json`:

| split | count |
| --- | ---:|
| train | 73,251 |
| val   | 112 |
| test  | 109 |
| **total** | **73,472** |

This split indexes positions in `df_SomgProc_final.parquet` (854,287 rows;
824,858 with `success==True`, the dataloader's view). The 1k baseline's
val/test sha256s are pinned into the new val/test for back-comparability
with prior runs.

The artifacts the downstream emission DiT consumes per sample are:

* `pbr_voxels_256/{sha}.vxz` — sparse PBR voxels (input for PBR encoder).
* `emission_voxels_256/{sha}.vxz` — sparse emission voxels (input for
  emission encoder; not encoded in this spec, downstream).
* `dual_grid_256/{sha}.vxz` — sparse dual-grid (input for shape encoder).
* `pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{sha}.npz` — PBR latent.
* `shape_latents/shape_enc_next_dc_f16c32_fp16_256/{sha}.npz` — shape
  latent.

This spec covers the full prep pipeline that produces all five for the
74k subset. The **emission latent** is intentionally out of scope — it
depends on a trained emission VAE encoder that is itself an open
research thread.

## 2. Constraints & background

### 2.1 Correctness invariant — Blender-pickle vertex set

`dual_grid_{res}/{sha}.vxz` **must** be derived from
`pbr_dumps/{sha}.pickle` (Blender-imported vertices), not from the GLB
via `trimesh`. Blender's GLTF importer and trimesh produce different
vertex sets for the same `.glb`, so a trimesh-derived dual grid will
have sparse `coords` that disagree with the PBR / emission voxels. The
DiT enforces this with `assert torch.equal(x_0.coords, shape_z.coords)`,
which previously failed for ~90 % of the 1k samples until
`data_toolkit/regenerate_1k_dual_grid.py` was added (untracked,
2026-05-07). This pipeline keeps the Blender-pickle path everywhere.

### 2.2 Blender install (no sudo, per-node)

Compute nodes do not have Blender installed and the user has no sudo
access. Install Blender 4.5.1 (the version `data_toolkit/dump_pbr.py`
expects) under `/localscratch/dya78/blender-4.5.1-linux-x64/` on each
node we land on. The install is idempotent: skip if the binary already
exists. Pillow is installed into Blender's bundled Python via
`data_toolkit/blender_script/install_pillow.py`.

### 2.3 Output storage — jupyter shared FS, tar-bundled

Final outputs land on the jupyter shared NFS mount at
`/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/` (visible as
`/cs/3dlg-jupiter-project/...` from the jupyter side). 74T free, no
quota concerns. **Outputs migrate as tarballs**, not loose files —
shared NFS handles a few large objects much better than ~370k small
ones.

### 2.4 Blender pickle is transient and large

A Blender PBR dump is ~29 MB. For one shard (~9.2k samples), peak
`pbr_dumps/` size is ~270 GB; for the full 74k it would be ~2.1 TB.
Pickles are deleted at the end of each shard's CPU job — only the
voxels and latents are tarred.

### 2.5 Sharding & job topology

Eight parallel shards (rank 0..7 of world_size=8). Per shard, **two
sbatch jobs** chained via `--dependency=afterok`:

* **Job N-AB** (CPU-heavy, GPU-light): Stage A (Blender dump) +
  Stage B (voxelize_pbr + dual_grid). Tars voxels and uploads to
  jupyter.
* **Job N-C** (GPU): Stage C (encode_pbr_latent +
  encode_shape_from_dual_grid). Pulls voxels tar, untars, encodes,
  tars latents, uploads.

Sbatch flags follow the existing star2 convention (no `--partition`):
`--gres`, `--cpus-per-task`, `--mem`, `--time`. Slurm routes the job to
any node matching the resource ask.

### 2.6 Cluster conventions referenced

* Per-node tmux sessions named after the node — relevant only for
  interactive probing, not for the production run.
* Tee every non-trivial command into `log/${TS}-<task>.log`.
* No edits on the cluster — edit locally, push, cluster pulls from
  origin in the sbatch body.
* Check before installing — Blender install is idempotent on
  `/localscratch`.
* Probe before sbatch — a single-shard smoke pilot (50 samples) runs
  end-to-end before the 16-job fan-out.
* Big files migrate via rsync, not git.

## 3. Output layout

On jupyter, when all 16 jobs finish:

```
/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/
├── df_SomgProc_final.parquet              # 357 MB, source-of-truth
├── data_splits_emissive_74k_pinned.json   # split → positions in parquet
├── metadata.csv                           # sha256, glb_1k_path, split
├── voxels/
│   ├── part0.tar  …  part7.tar            # ~5 GB × 8 ≈ 40 GB total
├── latents/
│   ├── part0.tar  …  part7.tar            # ~1 GB × 8 ≈ 8 GB total
└── logs/
    ├── ${TS}-prep_74k_AB_rank<N>.log
    └── ${TS}-prep_74k_C_rank<N>.log
```

**Tar contents** (paths relative to the eventual extraction root, so the
8 tars merge cleanly):

```
voxels/partN.tar:
  pbr_voxels_256/{sha}.vxz
  emission_voxels_256/{sha}.vxz
  dual_grid_256/{sha}.vxz

latents/partN.tar:
  pbr_latents/tex_enc_next_dc_f16c32_fp16_256/{sha}.npz
  shape_latents/shape_enc_next_dc_f16c32_fp16_256/{sha}.npz
```

Downstream consumers extract all 16 tars into a single
`lightgen_74k/` tree on their `/localscratch` via
`data_toolkit/extract_74k_tars.sh`.

## 4. `metadata.csv` schema

Built once at bootstrap, immutable across stages. Columns:

| column | type | source |
| --- | --- | --- |
| `sha256` | str | `df_SomgProc_final.parquet` index (success-filtered) |
| `glb_1k_path` | str | parquet column — relative path under `/3dlg-falas/.../TexVerse-1K/` |
| `split` | str | `train` / `val` / `test`, derived from the 74k split JSON |

Each stage's per-rank `new_records/part_<rank>.csv` is diagnostic only
— downstream training reads parquet + split + per-sample latent
`.npz` files directly, not `metadata.csv`.

## 5. Components

### 5.1 New scripts (under `data_toolkit/`)

#### `prep_74k_bootstrap.py`

One-shot setup. Run **locally** (parquet is local; ssh is fine but
unnecessary). Steps:

1. Read `/local-scratch/localhome/dya78/code/lightgen/data/baked_uv/df_SomgProc_final.parquet`.
2. Read `/local-scratch/localhome/dya78/code/lightgen/data_processing/annotation/data_splits_emissive_74k_pinned.json`.
3. Filter parquet to `success==True` to mirror the dataloader's view.
4. For each (train, val, test) index list in the split JSON, look up
   sha256 + `glb_1k_path` from the success-filtered df.
5. Write `metadata.csv` with the schema above.
6. Copy `df_SomgProc_final.parquet` and the split JSON to
   `/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/` (rsync).
7. Print final counts to stdout for sanity.

CLI: `python data_toolkit/prep_74k_bootstrap.py`. No flags needed
for the canonical run; `--dry-run` prints counts without copying.

#### `prep_74k_dump.py` (Stage A)

Wrapper around the `blender --background --python
blender_script/dump_pbr.py …` subprocess loop, sharded.

CLI: `--root <localscratch dir> --rank N --world_size 8 --workers 8
[--blender /localscratch/dya78/blender-4.5.1-linux-x64/blender]
[--limit N for smoke]`.

Behaviour:
* Read `<root>/metadata.csv`. Take the rank-N slice
  (`len(meta)*N//W` … `len(meta)*(N+1)//W`).
* For each `(sha, glb_rel)`: skip if `pbr_dumps/{sha}.pickle` already
  exists; otherwise resolve the GLB path under
  `/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K/<glb_rel>`,
  spawn Blender, write pickle to a tempfile, atomic-rename into place.
* `ProcessPoolExecutor(max_workers=workers)`, surface failures into
  `pbr_dumps/new_records/part_<rank>_failures.txt` (one sha per line +
  short reason).

#### `prep_74k_dual_grid.py` (Stage B-2)

Mirror of `regenerate_1k_dual_grid.py:_dual_grid_from_pickle`. Reads
`pbr_dumps/{sha}.pickle` and writes `dual_grid_256/{sha}.vxz`.

CLI: `--root --rank --world_size --workers --resolution=256 [--limit]`.

Why a new script and not the in-repo `data_toolkit/dual_grid.py`:
the in-repo script filters on `mesh_dumped==True` and reads from a
`mesh_dumps/` subdir. Adapting it (column rename + `datasets.<name>`
plumbing + dir symlink) is more code than the ~100-line wrapper that
copies `_dual_grid_from_pickle` verbatim.

#### `extract_74k_tars.sh`

Plain bash, one-liner-style. Args: `<jupyter base> <target root>`.
Untars all 16 (8 voxels + 8 latents) into the target root, which
becomes a self-contained `lightgen_74k/` tree mirroring
`data/lightgen_1k/` shape.

#### `check_74k_progress.py`

Reads each tar on jupyter, counts members per artifact dir, compares
against the expected sha list from the split JSON, prints a table
(per-shard %-complete, missing sha list per artifact). Used post-run
for triage.

### 5.2 Reused as-is (no edits)

* `data_toolkit/voxelize_pbr.py` — Stage B-1. Already accepts
  `--rank/--world_size/--max_workers`, reads `pbr_dumps/{sha}.pickle`,
  writes both `pbr_voxels_{res}/{sha}.vxz` and
  `emission_voxels_{res}/{sha}.vxz`. Driven by `metadata.csv`.
* `data_toolkit/encode_pbr_latent.py` — Stage C-1. Already sharded.
  Reads `pbr_voxels_256/`, writes `pbr_latents/<enc_name>/`.
* `data_toolkit/encode_shape_from_dual_grid.py` — Stage C-2. Already
  sharded (commit d5dc0e6). Parquet-driven, reads
  `dual_grid_256/`, writes
  `shape_latents/shape_enc_next_dc_f16c32_fp16_256/`.

### 5.3 New launch scripts (under `script/star2/`)

* `prep_74k_install_blender.sh` — sourced helper, idempotent install
  of Blender 4.5.1 + Pillow under `/localscratch/dya78/`. Called at
  the start of the AB job body.
* `prep_74k_smoke.sh` — single-shard end-to-end smoke pilot on a
  50-sample subset (`--limit 50`). Submits one AB+C pair, exits.
* `prep_74k_stage_ab.sh` — emits sbatch text for one AB job; takes
  `RANK` and `WORLD_SIZE` as positional args. Runs Stage A → Stage B-1
  → Stage B-2 → tar + upload voxels.
* `prep_74k_stage_c.sh` — sbatch text for one C job; takes the AB
  job's jobid as `--dependency=afterok` and `RANK` / `WORLD_SIZE`.
  Pulls voxels tar, runs encoders, tars + uploads latents.
* `prep_74k_chain.sh` — top-level local orchestrator.
  For `N` in `0..7`: `ssh star2 sbatch …` for AB, capture jobid,
  then `ssh star2 sbatch --dependency=afterok:<jobid> …` for C.
  Prints a 16-row table of jobids + shards.

## 6. Operational flow per shard

### 6.1 Job N-AB (`prep_74k_stage_ab.sh`)

Resource ask: `--gres=gpu:1` (idle, smallest available), `--cpus-per-task=32`,
`--mem=64G`, `--time=3-00:00`.

```bash
#!/bin/zsh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00
#SBATCH --output=/localscratch/dya78/log/prep_74k_AB_rank${RANK}.out
#SBATCH -J prep_74k_AB_r${RANK}

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
git pull origin lightgen
conda activate trellis2

# Idempotent Blender install on this node's /localscratch
. script/star2/prep_74k_install_blender.sh

LOCAL=/localscratch/dya78/lightgen_74k_part${RANK}
JUP=/3dlg-jupiter-project/lightgen/dataset/lightgen_74k
mkdir -p ${LOCAL}/log
TS=$(date +%Y%m%d-%H%M%S)
LOG=${LOCAL}/log/${TS}-prep_74k_AB_rank${RANK}.log

# Pull metadata + parquet + split from jupyter (one-time, small).
rsync -av ${JUP}/metadata.csv ${LOCAL}/
rsync -av ${JUP}/df_SomgProc_final.parquet ${LOCAL}/
rsync -av ${JUP}/data_splits_emissive_74k_pinned.json ${LOCAL}/

(
  set -e
  python data_toolkit/prep_74k_dump.py     --root ${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} --workers 8
  python data_toolkit/voxelize_pbr.py      lightgen_74k --root ${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} --max_workers 16 --resolution 256
  python data_toolkit/prep_74k_dual_grid.py --root ${LOCAL} --rank ${RANK} --world_size ${WORLD_SIZE} --workers 16 --resolution 256
) 2>&1 | tee "${LOG}"
AB_EXIT=${pipestatus[1]}

if [[ ${AB_EXIT} -eq 0 ]]; then
  cd ${LOCAL}
  tar -cf part${RANK}_voxels.tar pbr_voxels_256 emission_voxels_256 dual_grid_256
  rsync -av part${RANK}_voxels.tar ${JUP}/voxels/
  rsync -av ${LOG} ${JUP}/logs/
  rm -rf pbr_dumps   # free /localscratch
fi

exit ${AB_EXIT}
```

`voxelize_pbr.py` requires a `datasets.<name>` module via its
`importlib.import_module(f'datasets.{sys.argv[1]}')` shim. We add a
minimal `data_toolkit/datasets/lightgen_74k.py` that mirrors the
existing dataset modules — exposes `add_args(parser)` and
`foreach_instance(metadata, download_root, func, max_workers, no_file,
desc)` (use the same multiprocessing-based runner the other dataset
modules use; copy from the closest existing one).

### 6.2 Job N-C (`prep_74k_stage_c.sh`)

Resource ask: `--gres=gpu:1` (any), `--cpus-per-task=16`,
`--mem=64G`, `--time=0-06:00`. Submitted with `--dependency=afterok:<AB jobid>`.

```bash
#!/bin/zsh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00
#SBATCH --output=/localscratch/dya78/log/prep_74k_C_rank${RANK}.out
#SBATCH -J prep_74k_C_r${RANK}

source ~/.zshrc
cd /localscratch/dya78/TRELLIS.2-lightning
git pull origin lightgen
conda activate trellis2

LOCAL=/localscratch/dya78/lightgen_74k_part${RANK}
JUP=/3dlg-jupiter-project/lightgen/dataset/lightgen_74k
mkdir -p ${LOCAL}/log
TS=$(date +%Y%m%d-%H%M%S)
LOG=${LOCAL}/log/${TS}-prep_74k_C_rank${RANK}.log

(
  set -e
  rsync -av ${JUP}/metadata.csv ${LOCAL}/
  rsync -av ${JUP}/df_SomgProc_final.parquet ${LOCAL}/
  rsync -av ${JUP}/voxels/part${RANK}_voxels.tar ${LOCAL}/
  cd ${LOCAL} && tar -xf part${RANK}_voxels.tar && cd -

  python data_toolkit/encode_pbr_latent.py            --root ${LOCAL} --rank 0 --world_size 1 --resolution 256
  python data_toolkit/encode_shape_from_dual_grid.py  --data_root ${LOCAL} --parquet ${LOCAL}/df_SomgProc_final.parquet --rank 0 --world_size 1 --resolution 256
) 2>&1 | tee "${LOG}"
C_EXIT=${pipestatus[1]}

if [[ ${C_EXIT} -eq 0 ]]; then
  cd ${LOCAL}
  tar -cf part${RANK}_latents.tar pbr_latents shape_latents
  rsync -av part${RANK}_latents.tar ${JUP}/latents/
  rsync -av ${LOG} ${JUP}/logs/
fi

exit ${C_EXIT}
```

Job N-C runs the encoders on the *full* extracted shard (`--rank 0
--world_size 1`); world-size sharding for encoders is unnecessary
because each shard is already small enough (~9.2k samples × ~150 ms
on one GPU ≈ 25 min for PBR + shape combined).

### 6.3 Pilot smoke run

Before fan-out, `prep_74k_smoke.sh` submits **one** AB+C pair with
`--limit 50` (passed through to both `prep_74k_dump.py` and
`prep_74k_dual_grid.py`; `voxelize_pbr.py` and the encoders pick up
only the 50 already-prepped samples). Tar size will be small (~30 MB)
— the smoke pilot validates the full path including upload.

The smoke pilot also catches:
* Blender install failures (binary missing, wget blocked, Pillow
  install hang).
* `/3dlg-falas/.../TexVerse-1K/` GLB visibility from compute nodes
  (Blender's first call fails fast if mount is missing).
* Sbatch resource-ask shape (any zsh / GPU-gres / nodelist surprises).
* Tar+rsync to jupyter (perms, path).

### 6.4 Fan-out

After smoke passes:

```bash
bash script/star2/prep_74k_chain.sh
```

This emits 16 `ssh star2 sbatch …` calls, captures jobids, prints:

```
shard  AB jobid    C jobid     status
0      1234567     1234568     submitted
1      1234569     1234570     submitted
…
```

User can then `squeue -u dya78` to track.

### 6.5 Wallclock budget

Per shard, ~9,184 samples. Stage A dominates. Conservatively:

| stage | per-sample | × samples | per-shard wallclock |
| --- | ---:| ---:| ---:|
| Stage A (Blender, 8 workers) | ~60 s / 8 = 7.5 s | 9,184 | ~19 h |
| Stage B-1 (voxelize, 16 workers) | ~3 s / 16 ≈ 0.2 s | 9,184 | ~30 min |
| Stage B-2 (dual_grid, 16 workers) | ~2 s / 16 ≈ 0.13 s | 9,184 | ~20 min |
| Stage C-1 (PBR encode, 1 GPU) | ~150 ms | 9,184 | ~25 min |
| Stage C-2 (shape encode, 1 GPU) | ~150 ms | 9,184 | ~25 min |

Total per-shard ~21 h end-to-end. Job AB walltime ask of 3 days is
3-4× safety. Job C walltime ask of 6 h is 6-8× safety.

## 7. Failure handling

### 7.1 Per-sample failures

Each Python stage already implements skip-if-exists + per-sample
try/except. Failures append to `<artifact_dir>/new_records/part_<rank>_failures.txt`.
The AB and C jobs do not abort on per-sample failure; they continue and
exit 0 if the stage finishes. Downstream `check_74k_progress.py`
surfaces the dropouts.

### 7.2 Per-shard failures

If Job N-AB fails (Slurm signal, OOM, dead node), the shard's
`/localscratch` may be wiped. Resubmit the AB job; if pickles +
voxels survive, the inner skip-if-exists picks up where it left off.
If `/localscratch` was wiped, the shard restarts cleanly.

If Job N-C fails after AB succeeded, the voxels tar already lives on
jupyter — resubmit only the C job, manually setting
`--dependency=afterok:<dummy>` to none.

### 7.3 OOM mitigation

Blender peak RSS is ~2-4 GB per process. With `--workers 8` on a
64 GB node, peak ~32 GB for Blender + ~10 GB for the orchestrator
Python + headroom for the OS — fits. If `--workers 16` is attempted
and OOMs, drop to 8 in the launch script and resubmit. **Smoke pilot
launches with `--workers 8` as the conservative default**.

### 7.4 Disk pressure on `/localscratch`

Pickles: 9,184 × 29 MB ≈ 270 GB. Voxels + dual_grid: ~5 GB.
`/localscratch` on star2 nodes is typically ~1 TB+, comfortable
margin. If a node turns out to have less, drop pickles incrementally
(modify `prep_74k_dual_grid.py` to delete each pickle after it
processes it) — not required for v1.

### 7.5 1k-baseline sample overlap

~1,093 samples in the 74k pool overlap with the prior 1k pipeline.
We re-prep them from scratch on jupyter; the encoders' skip-if-exists
does not save us here because outputs land in a different directory
than the local 1k tree. ~1.5 % wasted Blender wallclock — accepted in
favour of one-source-of-truth on jupyter.

## 8. Extraction / consumption (downstream)

A trainer that wants to use the 74k dataset:

```bash
# On the trainer's node, with /localscratch mounted:
bash data_toolkit/extract_74k_tars.sh \
  /3dlg-jupiter-project/lightgen/dataset/lightgen_74k \
  /localscratch/dya78/lightgen_74k
```

After extraction, the directory looks identical to `data/lightgen_1k/`
in shape:

```
lightgen_74k/
├── df_SomgProc_final.parquet
├── data_splits_emissive_74k_pinned.json
├── metadata.csv
├── pbr_voxels_256/
├── emission_voxels_256/
├── dual_grid_256/
├── pbr_latents/tex_enc_next_dc_f16c32_fp16_256/
└── shape_latents/shape_enc_next_dc_f16c32_fp16_256/
```

So downstream training configs can swap `data/lightgen_1k/` →
`/localscratch/dya78/lightgen_74k/` with no code changes.

`CLAUDE.md`'s "Data Paths" gets one new bullet describing this.

## 9. Out of scope (this spec)

* Emission VAE encoding for the 74k subset — depends on a trained
  emission VAE encoder and gets its own spec when that work converges.
* Aesthetic-score filtering — the 74k split is the deliverable; any
  further filtering is a downstream slice.
* Re-rendering thumbnails or `pbr_dumps` archival — not used by the
  DiT.
* New evaluation harness — out of scope; downstream concern.

## 10. Acceptance criteria

1. Smoke pilot (50 samples, one AB + one C) completes end-to-end and
   uploads `voxels/part_smoke.tar` + `latents/part_smoke.tar` to
   jupyter.
2. `prep_74k_chain.sh` submits 16 jobs and prints jobids without error.
3. After all 16 jobs finish, `check_74k_progress.py` reports ≥ 95 %
   per-shard completion. Sub-95 % triages dropouts and decides retry.
4. `extract_74k_tars.sh` produces a tree that the existing
   `LightGenSLatEmissionLatent` dataset class can consume with only a
   `data_root` swap.
