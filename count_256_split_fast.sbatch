#!/bin/bash
#SBATCH -J count_256_fast
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/count_256_fast_%j.log
set -x
D=/localscratch/xya120
REPO=$D/segvigen-lightning
source $D/miniforge3/etc/profile.d/conda.sh
conda activate segvigen-bw
cd "$REPO"
python3 - <<'EOF'
import sys, os, json, time
sys.path.insert(0, ".")
sys.path.insert(0, "/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe")
import pandas as pd

t0 = time.time()
DATA_ROOT = "/3dlg-jupiter-project/lightgen/trellis2_bw/lightgen_74k_newbake"
SPLIT_JSON = os.path.join(DATA_ROOT, "data_splits_emissive_74k_stratified_newbake_vae_agentic.json")
PARQUET = os.path.join(DATA_ROOT, "df_SomgProc_final.parquet")
SHAPE_DIR = "/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2/shape_latents/shape_enc_next_dc_f16c32_fp16_256"
PBR_DIR   = "/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2/pbr_latents/tex_enc_next_dc_f16c32_fp16_256"
EMIT_DIR  = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/latents_256/output_tex_latents/segvigen_emissive_mask_256"
COND_DIRS = [
    "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/train_72k_agentic",
    "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/val_72k",
]

print("listing shape/pbr/emit/cond dirs...")
shape_ids = {f[:-4] for f in os.listdir(SHAPE_DIR) if f.endswith(".npz")}
pbr_ids   = {f[:-4] for f in os.listdir(PBR_DIR) if f.endswith(".npz")}
emit_ids  = {f[:-4] for f in os.listdir(EMIT_DIR) if f.endswith(".npz")}
cond_ids  = set()
for d in COND_DIRS:
    cond_ids |= set(os.listdir(d))
print(f"shape={len(shape_ids)} pbr={len(pbr_ids)} emit={len(emit_ids)} cond={len(cond_ids)}  ({time.time()-t0:.1f}s)")

print("loading parquet + split json...")
df = pd.read_parquet(PARQUET)
if "success" in df.columns:
    df = df[df["success"] == True]
all_sids = [str(i) for i in df.index]
split_data = json.load(open(SPLIT_JSON))
print(f"parquet rows (post-success-filter)={len(all_sids)}  ({time.time()-t0:.1f}s)")

for split in ["train", "val", "test"]:
    entry = split_data.get(split, {})
    idx_list = entry.get("indices", entry) if isinstance(entry, dict) else entry
    sids = [all_sids[i] for i in idx_list]
    sids_set = set(sids)
    usable = sids_set & shape_ids & pbr_ids & emit_ids & cond_ids
    print(f"FULL_COUNT split={split}: {len(sids_set)} in split_json, {len(usable)} usable "
          f"(missing: shape={len(sids_set - shape_ids)} pbr={len(sids_set - pbr_ids)} "
          f"emit={len(sids_set - emit_ids)} cond={len(sids_set - cond_ids)})")
print(f"total time {time.time()-t0:.1f}s")
EOF
