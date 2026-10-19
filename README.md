# SegviGen for emission

This repository adapts **SegviGen** from part segmentation to **emission**. Given a 3D shape as sparse voxels with
its PBR materials, and an image of the object, it predicts which surface voxels emit light: SegviGen's model, built
on TRELLIS.2's 1.3B texture flow model, is finetuned to generate an emission mask as a texture latent. It is the
SegviGen-Emission baseline of LightgenBench
([Hugging Face `3dlg-hcvc/LightgenBench`](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)).

SegviGen, TRELLIS.2 and most of this code are the work of their authors:

> **SegviGen: Repurposing 3D Generative Model for Part Segmentation** (arXiv preprint, 2026). Lin Li, Haoran Feng,
> Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, Lu Sheng.
> [Paper](https://arxiv.org/abs/2603.16869) · [Project page](https://fenghora.github.io/SegviGen-Page/) ·
> [Code](https://github.com/Nelipot-Lee/SegviGen) · [Model](https://huggingface.co/fenghora/SegviGen) ·
> [Demo](https://huggingface.co/spaces/fenghora/SegviGen)

> **Native and Compact Structured Latents for 3D Generation** (tech report, 2025). Jianfeng Xiang,
> Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao,
> Nicholas Jing Yuan, Jiaolong Yang. Microsoft.
> [Paper](https://arxiv.org/abs/2512.14692) · [Project page](https://microsoft.github.io/TRELLIS.2) ·
> [Code](https://github.com/microsoft/TRELLIS.2) · [Model](https://huggingface.co/microsoft/TRELLIS.2-4B)

The training code is 3DLG-HCVC's port of TRELLIS.2 to PyTorch Lightning. If you use this code, please cite
SegviGen and TRELLIS.2 ([below](#citation)).

## What differs from SegviGen and TRELLIS.2

- **Model** (`trellis2/models/segvigen_flow.py`, `SegviGenSLatFlowModel`): TRELLIS.2's `ElasticSLatFlowModel` as
  SegviGen uses it. Each token carries the noisy target latent and the shape latent (channel concat); the PBR latent
  of the same voxels follows as a second token set; the tokens attend to DINOv3 features of the object's thumbnail.
  The target is the emission mask (emitting voxels white, the rest black), encoded with TRELLIS.2's texture encoder.
- **Training** (`trellis2/trainers/flow_matching/segvigen_flow_trainer.py`, `SegviGenFlowTrainer`): from SegviGen's
  `full_seg` checkpoint, with precomputed DINOv3 tokens and the loss averaged per shape.
- **Data** (`trellis2/datasets/segvigen_256.py`, `SegviGenEmissive256`): 256³ voxels; the shape and PBR latents of
  TRELLIS.2's own encoders, the mask latents and the conditioning, built by the producers in `data_toolkit/`.
- **Prediction** (`data_toolkit/predict_emission.py`): TRELLIS.2's texture sampler and decoders; it writes the mask on
  the 256 grid, and the mask times the asset's albedo for the color metrics.
- **Training callbacks** (the end of `trellis2/trainers/callbacks.py`): one wandb run across resumes, the best
  checkpoint by `val/loss` plus `last.ckpt`, a training log line every `i_log` steps, and an option to skip the
  snapshot at the end of training.

Our code sits in files of its own, except those training callbacks, which follow upstream's in
`trellis2/trainers/callbacks.py`. The other edits left in upstream's files are small: `train.py` and the
`__init__.py` registries of `trellis2/{models,datasets,trainers}` register ours; `trellis2/trainers/basic.py` logs
the learning rate, loads `.safetensors` and Hugging Face paths and skips previews for datasets without them;
`ResumableSampler.set_epoch` (`trellis2/utils/data_utils.py`, missing upstream, so every epoch shuffled alike); an
empty validation set for Lightning 2 (`trellis2/trainers/data.py`); DINOv3 in transformers 5.5
(`flow_matching/mixins/image_conditioned.py`); emission in O-Voxel and in
`data_toolkit/{voxelize_pbr.py,blender_script/dump_pbr.py}`, including an O-Voxel fix (it sampled the emissive
texture with the roughness texture's mipmaps); `data_toolkit/encode_pbr_latent.py` runs its encoder in fp32;
`setup.sh` no longer calls `sudo apt`. All of them:

```bash
git diff --diff-filter=M 98c2548 -- train.py trellis2/ o-voxel/ data_toolkit/ setup.sh
```

## Setup

```bash
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
pip install lightning==2.6.1 pyarrow wandb
```

`setup.sh` is TRELLIS.2's (Linux, CUDA Toolkit 12.4, conda env `trellis2`; `. ./setup.sh --help` lists the parts).
The conditioning stage needs a transformers that loads BiRefNet (4.57.6 does, 5.1 does not); training needs
lightning 2.6. Our runs: training in torch 2.9.1 + CUDA 12.6, transformers 5.3.0, flash-attn 2.8.3 (H100);
conditioning and prediction in torch 2.6.0 + CUDA 12.4, transformers 4.57.6, flash-attn 2.7.3 (A40). FlexGEMM and
O-Voxel are compiled for the GPU they are built on. `PYTHON=<interpreter>` picks the env for each call below.

On first use the code downloads from Hugging Face `microsoft/TRELLIS.2-4B` (the shape and texture encoders and
decoders, the sampler settings), `facebook/dinov3-vitl16-pretrain-lvd1689m`, `briaai/RMBG-2.0` (the thumbnail's
background removal) and `fenghora/SegviGen` (`full_seg.ckpt`). DINOv3 and RMBG-2.0 are gated: request access on
their pages first, and log in (`huggingface-cli login`). Run every command below from the repository root.

## Data

LightgenBench is gated too: accept its terms on the dataset page.

```bash
bash script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>
```

This downloads `splits.json`, `checksums.sha256` and the voxel and thumbnail tars (15.8 GB), checks `splits.json`
against the released split and every tar against `checksums.sha256`, unpacks them into `<data_root>/release/<uuid>/`
and writes the index: `meta/release_index.parquet` (the shapes) and `meta/release_splits.json` (row numbers of the
36,426 train / 200 val / 200 test shapes).

The model also reads each shape's shape and PBR latents, its emission-mask latent and its conditioning, which the
dataset does not carry:

```bash
TEXVERSE_1K=<dir> bash script/lightgenbench/run_chain.sh latents <data_root>
bash script/lightgenbench/run_chain.sh masks <data_root>
bash script/lightgenbench/run_chain.sh cond <data_root>
```

`<dir>` holds the shapes' source files from [TexVerse-1K](https://huggingface.co/datasets/YiboZhang2001/TexVerse-1K),
as `glbs/glbs_1k/<group>/<uuid>_1024.glb`. `latents` builds each shape's dual grid
(`data_toolkit/dual_grid_from_glb.py --loader lightgenbench`: the mesh loader LightgenBench's data preparation uses,
from the lightgen repository that holds this one, or `LIGHTGEN_REPO=<dir>`), checks that its voxels are those of the
shape's PBR voxels, and encodes the shape and PBR latents with TRELLIS.2's encoders. `masks` encodes each shape's
emission mask; `cond` removes the thumbnail's background and computes its DINOv3 tokens (SegviGen's preprocessing,
`data_toolkit/thumbnail_cond.py`), then checks every input and writes `meta/release_splits_usable.json`, the rows
training reads. Checked against our release on 4 shapes: the dual grids are byte-identical, the latents and the
conditioning differ at float level.

## Training

```bash
bash script/lightgenbench/run_chain.sh train <data_root>
```

This is the recipe of our release run (`configs/gen/segvigen_emissive_256_release.yaml`): 50,000 steps, AdamW at
lr 1e-5, from SegviGen's `full_seg` (the stage fetches and rekeys it into `data/ckpt_init/full_seg_rekeyed.pt`), on 4
× H100 80 GB with 12 shapes per GPU (global batch 48). `run_chain.sh` refuses another global batch unless
`ANY_BATCH=1`; its header lists the other settings (`NUM_GPUS`, `BS`, `MAX_STEPS`, `OUTPUTS`, ...). Each stage tees
its output to `log/<time>-lightgenbench_<stage>.log`; training also logs to wandb when you are logged in (project
`LightGen`).

- **A first launch starts in an empty run directory.** The model starts from `full_seg`, and a run that resumes a
  checkpoint skips that load; `run_chain.sh` refuses a run directory with checkpoints unless `RESUME=1`, which
  continues an interrupted run.
- **Checkpoints pass through `$TMPDIR`** on their way to disk (11 GB), and Python puts Unix sockets there, whose paths
  must stay short: with a small `/tmp`, set `CHAIN_TMPDIR=<a short path on a disk with room>`. The stage checks before
  it trains.
- `last.ckpt` is refreshed once per epoch (758 steps), as in the release run. A top-k file `step<N>-<val/loss>.ckpt`
  holds step N's weights under the previous validation's loss.

## Prediction on the test split

```bash
CKPT=outputs/segvigen_emissive_256_release/ckpts/<file>.ckpt bash script/lightgenbench/run_chain.sh predict <data_root>
```

Nothing picks a checkpoint for you: we report step 44,722, picked on the validation split by IoU. This writes
`outputs/predict_test/seed<N>/mask/draw0/<uuid>.npz` for the 200 test shapes and seeds 0-4 (`SEEDS`), with `coords`
(the voxels, at 256³) and `pred` (the mask, three equal channels, nominally in [0, 1]), and `.../ma/draw0/` (mask × the asset's
albedo), using 12 sampling steps and TRELLIS.2's texture sampler settings. Each shape's noise comes from its uuid and
the seed, not from its place in the list. The LightGen repository's point-sampled evaluator scores them
(`evaluation/pointsampled/pipeline.py --mode eval --gt_emission texture_only --voxel_pred_root <dir>`): voxel IoU from
the masks, MAE and PSNR from mask × albedo. Our release run: test IoU 0.1468 ± 0.0083 over the five seeds.

## Repository

| path | contents |
|---|---|
| `configs/gen/` | the release config, and TRELLIS.2's own |
| `trellis2/` | models, datasets, trainers (TRELLIS.2's package; ours in files of their own, our training callbacks at the end of `trainers/callbacks.py`) |
| `train.py` | training entry point (`--config`, `--output_dir`, `--override key=value`) |
| `script/lightgenbench/` | `prepare_lightgenbench.sh`, `run_chain.sh`: everything above |
| `data_toolkit/` | TRELLIS.2's data scripts, and ours: the dual grid, the encoders, the mask latents, the conditioning, the check, the init, the index, the predictor |
| `o-voxel/` | TRELLIS.2's voxel library (with emission) |
| `example.py`, `example_texturing.py`, `app.py`, `app_texturing.py` | TRELLIS.2's image-to-3D and texturing demos, unchanged ([their usage](https://github.com/microsoft/TRELLIS.2#-usage)) |

## License

SegviGen is released under the MIT License (Copyright (c) 2026 Lin Li); TRELLIS.2 under the [MIT License](LICENSE).
Some dependencies have their own terms: [nvdiffrast](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt),
[nvdiffrec](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt), and `briaai/RMBG-2.0`, which the
conditioning stage runs: its weights are for non-commercial use only
([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en); commercial use needs an agreement with
BRIA). LightgenBench shapes keep the licenses of their source models (see the dataset card).

## Citation

```bibtex
@article{li2026segvigen,
    title={SegviGen: Repurposing 3D Generative Model for Part Segmentation},
    author={Lin Li and Haoran Feng and Zehuan Huang and Haohua Chen and Wenbo Nie and Shaohua Hou and Keqing Fan and Pan Hu and Sheng Wang and Buyu Li and Lu Sheng},
    journal={arXiv preprint arXiv:2603.16869},
    year={2026}
}

@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
