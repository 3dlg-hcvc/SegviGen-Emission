# 3D Visualization Consolidation — Design

**Date:** 2026-04-22
**Topic:** Consolidate the 3D interactive visualization scripts under `vis/` into two task-family scripts.
**Status:** Draft v2 (post-Codex review; approved for plan-writing)

## Goal

Replace the current `vis_3d_*.py` scripts with two unified scripts that share a common helper module. Each unified script must:

- Render an interactive HTML per task family (VAE reconstruction, DiT generation).
- Support a single experiment or N experiments side-by-side in one HTML.
- Show the original GLB mesh alongside GT emission voxels and each experiment's prediction, so the viewer always knows what object they're looking at.
- Let the user rotate / zoom every panel together (shared camera).

Out of scope: the 2D max-projection PNG scripts (`vis_emission_vae.py`, `vis_emission_vae_1sample.py`, `vis_vae_ckpt_comparison.py`, `compare_emission_vae_epochs.py`, `compare_voxelization_paths.py`). They stay as-is.

## Non-goals

- Unifying DiT and VAE into one script. They have different data paths (DiT needs images + DINOv3 + precomputed latents + CFG), and the user explicitly wants them separate.
- Supporting the pretrained HuggingFace PBR VAE as an `--experiment` kind. Use case is not current; the two legacy scripts that cover it (`vis_3d_pbr_vae_on_emission.py`, `vis_3d_pbr_vae_val.py`) stay as-is in `vis/legacy/`.
- Changing the semantics of the underlying models, trainers, or data pipeline. This is a visualization refactor only.
- Rewriting the 2D PNG scripts.

## Deliverables

1. `vis/_vis_common.py` — shared helpers (sample loader, voxel reader, frame computation, Three.js panel builders, HTML template, trainer-type resolver).
2. `vis/vis_3d_vae.py` — unified VAE-reconstruction visualization. Covers the common cases of `vis_3d_emission_vae.py`, `vis_3d_pbr_finetune.py`, `vis_3d_overfit.py`, and the "no model, just GLB + GT" case of `vis_voxel_vs_glb.py`.
3. `vis/vis_3d_dit.py` — unified DiT→decode visualization. Covers `vis_3d_emission_dit.py` and `vis_3d_emission_dit_pbrvae.py`.
4. `vis/legacy/` — all existing `vis_3d_*.py` moved here in the same commit. They are not deleted: they are the ground-truth reference for HTML output during migration, and the pretrained-PBR-VAE / voxelization-roundtrip scripts remain live there.

## Panel layout (shared across both scripts)

Every sample renders as a horizontal strip of Three.js panels:

```
[ GLB ]  [ GT emission ]  [ experiment_1 ]  ...  [ experiment_N ]
```

- Zero experiments → data-only mode: just `[GLB] [GT]`. This subsumes the common case of `vis_voxel_vs_glb.py` (no separate script needed for that).
- One experiment → three panels (GLB + GT + recon).
- N experiments → N+2 panels per sample.

Each prediction panel has a **one-line caption** above it showing `emission L1: <value>` for that sample (mean absolute error in sRGB space between prediction voxels and GT voxels). Matches what `vis/vis_3d_overfit.py:270` and `vis/vis_3d_emission_dit.py:196` already show. No other metrics.

Sample `<select>` dropdown swaps panel buffers in place (doesn't rebuild viewers), so the camera pose is preserved across sample changes — important for comparison.

## Three.js-only rendering, and why

Plotly and Three.js cannot share a camera. The "rotate together" requirement therefore forces one engine for all panels. Three.js is already the engine used by the existing scripts with synced-rotation behavior (`vis/vis_3d_pbr_vae_val.py:200–214`) and is the only engine that can render the GLB. Conclusion: every panel uses Three.js — voxel panels as point clouds, GLB panel as a GLTFLoader mesh.

### Camera-sync mechanism

One `OrbitControls` per canvas (they are listeners bound to a DOM element; you cannot bind one instance to many canvases). On the `change` event of any canvas's controls, broadcast `position`, `target`, `zoom` to every other canvas's camera and call `update()` on their controls. This matches the pattern in `vis/vis_3d_pbr_vae_val.py:200–214` generalized from 2 panels to N.

## Shared-frame scaling

The known scaling issue comes from voxel panels being normalized per-panel while GLB lives in arbitrary mesh units. Fix: put every drawable into the same coordinate frame **before** rendering, using the same normalization the voxelization pipeline already applied.

**Canonical frame.** The voxelization pipeline (`data_toolkit/voxelize_pbr.py:77–86`) normalizes each mesh per-sample to `[-0.5, 0.5]^3` via `center = (vmin + vmax) / 2`, `scale = 0.99999 / (vmax - vmin).max()`, `vertex' = (vertex - center) * scale`. Voxel coords are then generated with `aabb=[-0.5, -0.5, -0.5] → [0.5, 0.5, 0.5]` on an integer grid `[0, res)^3`. So the natural unifying frame is **`[-0.5, 0.5]^3`, mesh-normalized space.**

**Transform chain.** Every panel applies this fixed chain in order:

1. **Source-specific → mesh-normalized `[-0.5, 0.5]`.**
   - Voxel panels (GT + predictions): `mesh_pos = voxel_coord / res - 0.5 + 0.5 / res` (center-of-cell offset; `res` is read from the `.vxz` header or the config).
   - GLB panel: re-apply the voxelization normalization to the GLB's own vertices at load time — compute `center`, `scale` from the GLB vertex bbox exactly as in `data_toolkit/voxelize_pbr.py:77–82`, then `vertex' = (vertex - center) * scale`. This puts the GLB into the same `[-0.5, 0.5]^3` box as the voxels.
2. **Blender-Z-up → Three.js-Y-up.** Fixed swap `(x, y, z) → (x, z, y)` applied at buffer-upload time. Matches `vis/vis_3d_overfit.py:293–306`.

No per-sample viewer-side affine is needed beyond step 1. The camera auto-fits to the fixed unit cube.

**Missing GLB fallback.** If the GLB file is missing, show a placeholder panel with "GLB unavailable" text; don't skip the sample. Voxel panels still work because they don't depend on the GLB.

**`pbr_y_up` note.** If the voxelization run used `--pbr_y_up` to convert Blender Z-up dumps to Y-up before voxelization, the GLB on disk may already be in Y-up. For the 10-sample `lightgen_test` set this is a fixed per-dataset choice. The GLB loader applies the same Z-up → Y-up rotation only if the dataset's voxelization used it; it's a single boolean passed to `build_glb_panel`, not a per-sample decision.

## Shared helpers (`vis/_vis_common.py`)

Single source of truth for everything currently duplicated across the existing scripts.

- **`load_samples(data_root, resolution, source, filter, extra_paths)`** — `source` is a parquet path, a `data_splits_*.json`, or an explicit SHA list. Paths are derived from `data_root` + `resolution` + `sha` using the same convention as `trellis2/datasets/lightgen_slat.py:409` (NOT from parquet columns): `emission_voxels_{res}/{sha}.vxz`, `pbr_voxels_{res}/{sha}.vxz`, `latents/{latent_name}/{sha}.npz`, `thumbnails/{sha}/000.webp`, `glb_path` resolved from metadata. Returns a list of `SampleRecord(sha, emission_voxel_path, pbr_voxel_path, latent_path, image_path, glb_path)`, filtered to entries where required files exist.
- **`read_emission_voxel(path, threshold)`** — wraps `o_voxel.io.read_vxz`, applies emission→sRGB gamma per `vis/vis_3d_emission_vae.py:95–109`, returns `(coords: int32[N,3], colors_srgb: float32[N,3], res: int)`.
- **`resolve_experiment(path) → ExperimentHandle`** — path is a `run_dir` or `.ckpt`. Walks upward from a `.ckpt` to find `config.yaml`. Checkpoint selection rule (deterministic, in order):
  1. If `last.ckpt` exists in `ckpts/`, use it.
  2. Else, parse filenames matching `step*-{val/loss:.4f}.ckpt` and pick the one with the smallest `val/loss`.
  3. Else, error out with a clear message.
  Returns `ExperimentHandle(config, ckpt_path, trainer_name)`.
- **`load_vae_from_handle(handle) → (encode_fn, decode_fn)`** — dispatches on `trainer.name`:
  - `EmissionVaeTrainer` → two-stream concat VAE: trainable emission encoder + frozen PBR encoder, decoder input reshaped for 64-channel concat. See `trellis2/trainers/vae/emission_vae.py`.
  - `EmissionPbrFinetuneTrainer` → PBR VAE channel-hijack: `_pack_emission_as_pbr` (emission → base_color, constants for metallic/roughness/alpha), decoder output takes first 3 channels. See `trellis2/trainers/vae/emission_pbr_finetune.py`.
  - Other → error out.
  Both `vis_3d_vae.py` and `vis_3d_dit.py` call this same helper for their VAE step, so DiT decode branches correctly on the VAE family.
- **`compute_emission_l1(pred_coords, pred_colors, gt_coords, gt_colors)`** — mean L1 in sRGB space over the union of voxel positions (zero-fill where one side is absent). Used for the per-panel caption.
- **`build_point_cloud_panel(coords, colors_srgb, res)`** — emits the Three.js JS that constructs one voxel viewer: `voxel_coord / res - 0.5 + 0.5 / res`, then Y/Z swap at upload.
- **`build_glb_panel(glb_url, z_up_to_y_up: bool)`** — emits the Three.js JS that loads a GLB via GLTFLoader, computes mesh bbox, applies `vertex' = (vertex - center) * scale` (same as voxelization), optionally applies the Z-up → Y-up rotation from `data_toolkit/voxelize_pbr.py:62–75`, then Y/Z swap at upload. Renders full PBR materials (including base color — this is the "what is the object?" panel, not an emission-only reference).
- **`render_html(samples, panel_specs, captions, title, out_path)`** — emits the full HTML: panel grid, sample `<select>` dropdown, per-canvas `OrbitControls`, cross-canvas `change`-event broadcast. Derived from `vis/vis_3d_pbr_vae_val.py:118–214`, generalized from 2 panels to N.

Each helper is written so that `vis_3d_vae.py` and `vis_3d_dit.py` never duplicate voxel I/O, color-space, scaling, camera-sync, or HTML-template code.

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
- Each experiment is resolved via `resolve_experiment` and loaded via `load_vae_from_handle`.
- Per-sample panels: `[GLB] [GT] [experiment_1_recon] ... [experiment_N_recon]`. Each prediction panel has an `emission L1` caption and shows the experiment's label.
- Samples dropdown only — experiments are always-on columns.

## `vis/vis_3d_dit.py`

CLI:

```
python vis/vis_3d_dit.py \
  --dataset <parquet_path | split_json_path> \
  --samples <all | first:<N> | sha:<sha1,sha2,...> | split:<name>> \
  --experiment <name>=<dit_run_dir_or_ckpt>:<vae_run_dir_or_ckpt> [--experiment ...] \
  --out visualizations/<name>.html
```

Behaviour:

- `--experiment` is **required at least once** (no data-only mode — DiT without a VAE can't produce voxels to render).
- Each experiment's value has the form `<dit_path>:<vae_path>` (colon-separated). Both paths go through `resolve_experiment`. The VAE is loaded via the same `load_vae_from_handle` used by `vis_3d_vae.py`, so DiT decode branches correctly on `EmissionVaeTrainer` vs `EmissionPbrFinetuneTrainer`.
- The prediction panel renders `DiT sample → VAE decode`, identical voxel-point-cloud form to the VAE script's prediction panel.
- Image conditioning path is resolved by `load_samples` from `data_root + thumbnails/{sha}/000.webp` (matching the dataset class), not from a parquet column.
- Same panel layout, same shared-frame scaling, same camera sync as `vis_3d_vae.py`.

## Migration plan

1. Land `vis/_vis_common.py` and `vis/vis_3d_vae.py`. Smoke test: reproduce one HTML per replaced script's primary use case and eyeball-diff against the corresponding legacy output.
2. Land `vis/vis_3d_dit.py`. Same smoke test.
3. Move existing `vis/vis_3d_*.py` into `vis/legacy/` in the same commit. The 2D PNG scripts stay at `vis/` root since they are out of scope.
4. Update `CLAUDE.md` "Visualization scripts" note to point at the two new scripts.

## Risks and mitigations

- **GLB/voxel alignment verification.** Re-applying the voxelization normalization to GLB vertices is mathematically equivalent to what the pipeline did, but any mismatch (e.g. `pbr_y_up` applied during voxelization but not here, or a subtly different bbox definition) will show as misaligned panels. Mitigation: during step (1) of migration, pick one known-good sample and eyeball GLB + GT-voxel overlap; if misaligned, diff against `data_toolkit/voxelize_pbr.py:60–86` line by line.
- **Point count.** Three.js can render high-voxel-count samples (>500k) but the browser struggles past ~1M points. Mitigation: keep the `--max_points` flag from the legacy scripts; if a sample exceeds the cap, subsample uniformly before upload.
- **Missing GLB files.** Some dataset rows may not have a GLB on disk. Mitigation: when GLB is missing, show a placeholder panel with "GLB unavailable" text; don't skip the sample.
- **Trainer-class rename breaks `resolve_experiment`.** Reading `trainer.name` from `config.yaml` is brittle if trainer classes are renamed. Mitigation: keep trainer-type resolution in one helper with a clear error message; the cost of updating one `if/elif` beats the cost of adding a `--trainer` CLI flag that users have to remember.
- **`vis_common.py` helper count creep.** If the helper module grows past ~400 lines, consider splitting into `_vis_io.py` + `_vis_three.py`. Not worth splitting preemptively.
