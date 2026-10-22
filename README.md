# SegviGen-Emission

SegviGen-Emission finetunes [SegviGen](https://github.com/Nelipot-Lee/SegviGen), a part-segmentation model built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), to predict which surface voxels of a 3D shape emit light, as a baseline of LightGenBench.

## Environment Setup

Run every command from the root of this repository.
TRELLIS.2's setup script needs Linux and the CUDA Toolkit 12.4.

```bash
# Create the conda env "trellis2" and build TRELLIS.2's extensions
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Packages for training and the data scripts
pip install lightning==2.6.1 pyarrow wandb
```

The `cond` stage below needs a transformers that loads BiRefNet (4.57.6 does, 5.1 does not).
If you install it in a second env, run that stage with `PYTHON=<that env's python>`.

## Data Setup

The dataset download is described at https://github.com/3dlg-hcvc/lightgenbench.
The commands below prepare it for this baseline, in order:

```bash
# Download the voxel, dual-grid and thumbnail files into <download_dir> and unpack them into <data_root>
bash script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>

# Shape and PBR latents, from the dual grids and the voxels
bash script/lightgenbench/run_chain.sh latents <data_root>

# Emission-mask latents (the training target)
bash script/lightgenbench/run_chain.sh masks <data_root>

# Image conditioning from the thumbnails, then a check of every training input
bash script/lightgenbench/run_chain.sh cond <data_root>
```

A shape without a dual grid in `<data_root>/dual_grid_256/` is built from its [TexVerse-1K](https://huggingface.co/datasets/YiboZhang2001/TexVerse-1K) GLB when `TEXVERSE_1K=<dir>` is set.
Each stage runs on `NUM_GPUS` GPUs (default 4).

## Training

```bash
# Train from SegviGen's full_seg checkpoint; checkpoints go to outputs/segvigen_emissive_256_release/ckpts/
bash script/lightgenbench/run_chain.sh train <data_root>

# Continue an interrupted run
RESUME=1 bash script/lightgenbench/run_chain.sh train <data_root>
```

The global batch `NUM_GPUS` × `BS` (default 4 × 12) must be 48; `ANY_BATCH=1` allows another.

## Prediction

```bash
# Choose the checkpoint on the validation split (SPLIT=val), then predict the test split with seeds 0-4;
# writes predictions to outputs/predict_test/seed<N>/
# (mask/draw0/: the emission mask; ma/draw0/: the mask times the albedo)
CKPT=outputs/segvigen_emissive_256_release/ckpts/<checkpoint>.ckpt bash script/lightgenbench/run_chain.sh predict <data_root>
```

Score the predictions with the LightGenBench evaluation: https://github.com/3dlg-hcvc/lightgenbench

## License

SegviGen is released under the MIT License (Copyright (c) 2026 Lin Li); TRELLIS.2 under the [MIT License](LICENSE).
The `cond` stage runs [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), whose weights are for non-commercial use only ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) and [nvdiffrec](https://github.com/NVlabs/nvdiffrec) have their own licenses.

## Citation

If you use this code, please cite SegviGen and TRELLIS.2:

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
