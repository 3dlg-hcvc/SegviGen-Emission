#!/bin/bash
#SBATCH -J count_256_split
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/count_256_split_%j.log
set -x
D=/localscratch/xya120
REPO=$D/segvigen-lightning
source $D/miniforge3/etc/profile.d/conda.sh
conda activate segvigen-bw
cd "$REPO"
python3 - <<'EOF'
import sys
sys.path.insert(0, ".")
from trellis2.datasets.segvigen_256 import SegviGenEmissive256

common = dict(
    data_root="/3dlg-jupiter-project/lightgen/trellis2_bw/lightgen_74k_newbake",
    parquet_file="/3dlg-jupiter-project/lightgen/trellis2_bw/lightgen_74k_newbake/df_SomgProc_final.parquet",
    split_json="/3dlg-jupiter-project/lightgen/trellis2_bw/lightgen_74k_newbake/data_splits_emissive_74k_stratified_newbake_vae_agentic.json",
    shape_latent_root="/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2",
    shape_latent_name="shape_enc_next_dc_f16c32_fp16_256",
    pbr_latent_root="/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2",
    pbr_latent_name="tex_enc_next_dc_f16c32_fp16_256",
    emission_latent_dir="/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/latents_256/output_tex_latents/segvigen_emissive_mask_256",
    cond_search_dirs=[
        "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/train_72k_agentic",
        "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/val_72k",
    ],
    shape_normalization={"mean": [0.0]*32, "std": [1.0]*32},
    tex_normalization={"mean": [0.0]*32, "std": [1.0]*32},
)
for split in ["train", "val"]:
    ds = SegviGenEmissive256(split=split, **common)
    print(f"FULL_COUNT split={split} usable={len(ds)}")
EOF
