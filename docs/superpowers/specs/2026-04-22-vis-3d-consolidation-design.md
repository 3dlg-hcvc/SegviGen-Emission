# 3D Visualization Consolidation — Design

**Date:** 2026-04-22
**Topic:** Consolidate the 3D interactive visualization scripts under `vis/` into two task-family scripts.
**Status:** Draft (approved for plan-writing)

## Goal

Replace the current ~8 `vis_3d_*.py` scripts with two unified scripts that share a common helper module. Each unified script must:

- Render an interactive HTML per task family (VAE reconstruction, DiT generation).
- Support a single experiment or N experiments side-by-side in one HTML.
- Show the original GLB mesh alongside GT emission voxels and each experiment's prediction, so the viewer always knows what object they're looking at.
- Let the user rotate / zoom every panel together (shared camera).

Out of scope: the 2D max-projection PNG scripts (`vis_emission_vae.py`, `vis_emission_vae_1sample.py`, `vis_vae_ckpt_comparison.py`, `compare_emission_vae_epochs.py`, `compare_voxelization_paths.py`). They stay as-is.

## Non-goals

- Unifying DiT and VAE into one script. They have different data paths (DiT needs images + DINOv3 + precomputed latents + CFG), and the user explicitly wants them separate.
- Changing the semantics of the underlying models, trainers, or data pipeline. This is a visualization refactor only.
- Rewriting the 2D PNG scripts.

## Deliverables

1. `vis/_vis_common.py` — shared helpers (sample loader, voxel reader, shared-frame computer, Three.js renderers, HTML template).
2. `vis/vis_3d_vae.py` — unified VAE-reconstruction visualization (covers `vis_3d_emission_vae.py`, `vis_3d_pbr_vae_on_emission.py`, `vis_3d_pbr_vae_val.py`, `vis_3d_pbr_finetune.py`, `vis_3d_overfit.py`, and the common case of `vis_voxel_vs_glb.py` via "zero experiments" mode).
3. `vis/vis_3d_dit.py` — unified DiT→decode visualization (covers `vis_3d_emission_dit.py`, `vis_3d_emission_dit_pbrvae.py`).
4. `vis/legacy/` — existing `vis_3d_*.py` moved here in the same commit. They are not deleted: they remain the ground-truth reference for the HTML output format during migration and for the niche `vis_voxel_vs_glb.py` roundtrip-validation case.

## Panel layout (shared across both scripts)

Every sample renders as a horizontal strip of Three.js panels:

```
[ GLB ]  [ GT emission ]  [ experiment_1 ]  ...  [ experiment_N ]
```

- Zero experiments → data-only mode: just `[GLB] [GT]`. This is the common case of `vis_voxel_vs_glb.py` and needs no separate script.
- One experiment → two prediction columns (GT + one recon) plus the GLB.
- N experiments → N+2 panels per sample.

All panels share one `OrbitControls` instance that broadcasts camera pose to every panel's camera on `onChange`. Sample dropdown swaps point-cloud buffers in place; camera pose is preserved across sample changes.

## Three.js-only rendering, and why

Plotly and Three.js cannot share a camera. The "rotate together" requirement therefore forces one engine for all panels. Three.js is already the engine used by the existing scripts with synced-rotation behavior (`vis/vis_3d_pbr_vae_val.py:200–214`) and is the only engine that can render the GLB. Conclusion: every panel uses Three.js — voxel panels as point clouds, GLB panel as a GLTFLoader mesh.

## Shared-frame scaling

Voxel data lives on an integer grid (`0..res`) and was previously normalized per-panel to a unit cube. GLB meshes live in arbitrary mesh units. Per-panel normalization causes the known scaling issue where objects no longer line up between panels.

Fix: one affine transform, computed from the GT emission voxel coordinate bbox, applied uniformly to every panel.

```
center = (coords_min + coords_max) / 2
scale  = max(coords_max - coords_min) / 2
transform(p) = (p - center) / scale
```

- GT voxels: transformed by this affine.
- Prediction voxels: share the same integer grid as GT, so the same affine applies (no-op relative to GT).
- GLB mesh: vertices transformed by the same affine before upload to the GPU. If the voxelization pipeline introduced a fixed coordinate-system swap (axis reorder, sign flip) between GLB mesh space and `.vxz` grid space, that correction is applied as a one-time constant transform **before** the shared affine, not as a per-sample fit.

The scale is computed from GT emission voxels, not from the GLB bbox, so empty-GLB / missing-GLB cases don't break the frame.

## Shared helpers (`vis/_vis_common.py`)

Single source of truth for everything currently duplicated across the 8 scripts.

- **`load_samples(source, filter)`** — source can be a parquet path, a `data_splits_*.json`, or an explicit SHA list. Returns a list of `SampleRecord(sha, emission_voxel_path, pbr_voxel_path, glb_path, image_path)`, filtered to entries where required files exist. Canonicalizes the three selection shapes currently found across scripts.
- **`read_emission_voxel(path, threshold)`** — wraps `o_voxel.io.read_vxz`, applies emission→sRGB gamma using the form in `vis/vis_3d_emission_vae.py:95–109`, returns `(coords: int32[N,3], colors_srgb: float32[N,3])`.
- **`compute_shared_frame(gt_coords)`** — returns `(center, scale)` as above.
- **`build_point_cloud_panel(coords, colors, center, scale)`** — emits the Three.js JS that constructs one voxel viewer. Derived from `vis/vis_3d_pbr_vae_val.py:169–196`.
- **`build_glb_panel(glb_url, center, scale, pre_transform=None)`** — emits the Three.js JS that loads a GLB via GLTFLoader and applies `pre_transform ∘ (x - center)/scale` to the mesh vertices.
- **`render_html(samples, panels_per_sample, title, out_path)`** — emits the full HTML: the panel grid, the sample `<select>` dropdown, the shared `OrbitControls`, and the camera-broadcast JS. Derived from the combination of `vis/vis_3d_pbr_vae_val.py:118–124` (dropdown) and `vis/vis_3d_pbr_vae_val.py:200–214` (camera sync), generalized from 2 panels to N panels.

Each helper is written so that `vis_3d_vae.py` and `vis_3d_dit.py` never duplicate voxel I/O, color-space, scaling, or HTML-template code.

## `vis/vis_3d_vae.py`

CLI:

```
python vis/vis_3d_vae.py \
  --dataset <parquet_path | split_json_path> \
  --samples <all | first:<N> | sha:<sha1,sha2,...> | split:<name>> \
  [--experiment <name>=<run_dir_or_ckpt_path>] ... \
  --out visualizations/<name>.html
```

Behaviour:

- `--experiment` is repeatable. Each experiment becomes one labelled prediction column. Zero occurrences → data-only mode (`GLB | GT`).
- Each experiment's path is resolved as follows:
  - If the path is a directory (`run_dir`): load `run_dir/config.yaml` and `run_dir/ckpts/last.ckpt` (or the top-`val/loss` checkpoint if `save_top_k` was in effect).
  - If the path is a `.ckpt` file: walk upward until a `config.yaml` is found.
- Trainer type is read from `config.yaml` (`trainer.name`):
  - `EmissionVaeTrainer` → two-stream concat VAE: trainable emission encoder + frozen PBR encoder, concat latents, decode to 3-channel emission.
  - `EmissionPbrFinetuneTrainer` → PBR VAE channel hijack: pack emission as base_color with constant metallic/roughness/alpha, encode+decode, take first 3 output channels.
  - Any other trainer → error out with a clear message.
- Per-sample panels: `[GLB] [GT] [experiment_1_recon] ... [experiment_N_recon]`. Each prediction panel's title shows the experiment's label.
- Samples dropdown only — experiments are always-on columns.

## `vis/vis_3d_dit.py`

CLI:

```
python vis/vis_3d_dit.py \
  --dataset <parquet_path | split_json_path> \
  --samples <all | first:<N> | sha:<sha1,sha2,...> | split:<name>> \
  --experiment <name>=<dit_run_dir_or_ckpt> [--vae_run <vae_run_or_ckpt>] ... \
  --out visualizations/<name>.html
```

Behaviour:

- Same panel layout and scaling as `vis_3d_vae.py`.
- Each experiment loads the DiT (image-conditioned flow-matching) + DINOv3 + a VAE decoder (the VAE checkpoint is either specified per-experiment via `--vae_run` or defaulted to the DiT training config's referenced VAE).
- The prediction panel renders `DiT sample → VAE decode`, identical voxel-point-cloud form to the VAE script's prediction panel.
- Conditioning image path is read from the parquet row per sample, so the input set is the same shape as the VAE script's.

## Migration plan

1. Land `vis/_vis_common.py` and `vis/vis_3d_vae.py`. Reproduce one HTML per replaced script's primary use case and eyeball-diff against the corresponding legacy output.
2. Land `vis/vis_3d_dit.py`. Same smoke test.
3. Move existing `vis/vis_3d_*.py` into `vis/legacy/` in the same commit. The 2D PNG scripts stay at `vis/` root since they are out of scope.
4. Update `CLAUDE.md` "Visualization scripts" note to point at the two new scripts.

## Risks and mitigations

- **GLB/voxel coordinate mismatch.** If the voxelization pipeline uses a different convention than the GLB (axis order, sign flip), panels will look mirrored/rotated even with correct scaling. Mitigation: inspect one known sample end-to-end during step (1) of migration, encode any needed fixed correction as the `pre_transform` argument to `build_glb_panel`. This is a single constant, not a per-sample fit.
- **Point count.** Three.js can render high-voxel-count samples (>500k) but the browser struggles past ~1M points. Mitigation: keep the `--max_points` flag from the legacy scripts; if a sample exceeds the cap, subsample uniformly before upload.
- **Missing GLB files.** Some dataset rows may not have a GLB on disk. Mitigation: when GLB is missing, show a placeholder panel with "GLB unavailable" text; don't skip the sample.
- **Checkpoint API drift.** Reading `trainer.name` from `config.yaml` is brittle if trainer classes are renamed. Mitigation: keep the trainer-type resolution in one helper with a clear error message; the cost of updating one `if/elif` beats the cost of adding a `--trainer` CLI flag that users have to remember.
