#!/bin/bash
# Release-layer check (steps 14, 15e): script/lightgenbench/{prepare_lightgenbench,run_chain}.sh of a tree under test on
# one Solar GPU, on a 4-shape download folder (2 train, 1 val, 1 test) made of the release's own files: per split one
# voxel tar and one thumbnail tar, splits.json and checksums.sha256. prepare (SKIP_DOWNLOAD=1, the mini splits.json's
# hash pinned) and the latents, masks, cond and predict stages run in Kevin's jupiter miniforge env (its transformers
# loads BiRefNet); train's dry run (--tryrun, one step) in ~/envs/trellis2 (lightning 2.6). Each output is compared with
# jupiter's datasets/dataset_release_latents (dual grids and init byte for byte; latents, masks and conditioning by
# their largest difference) and the test shape's prediction (release step 44,722, seed 0) with the published dump.
# Results go to $HOME/segvigen_refactor/out/<name>/.
#
#   sbatch --gres=gpu:l40s:1 -c 8 --mem=96G --time=02:00:00 -J sv_lgb \
#     --output=$HOME/segvigen_refactor/logs/%x-%j.log lightgenbench_check.sbatch <tree tgz> <out name> <uuid list>
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --ntasks=1
set -uo pipefail
TGZ="${1:?tree tgz}"; NAME="${2:?out name}"; LIST="${3:?uuid list: 2 train, 1 val, 1 test}"
H=$HOME/segvigen_refactor
J=/3dlg-jupiter-project/lightgen; REL=$J/datasets/dataset_release; L=$J/datasets/dataset_release_latents
PUB=$J/trellis2_bw/eval_outputs/release_eval/dumps_test/step0044722/seed00/draw0
LOC=/localscratch/dya78/sv_lgb_$SLURM_JOB_ID; mkdir -p $LOC/tree $LOC/dl $LOC/out; trap 'rm -rf $LOC' EXIT
tar -xzf "$TGZ" -C $LOC/tree; TREE=$(find $LOC/tree -mindepth 1 -maxdepth 1 -type d | head -1)
mapfile -t U < "$LIST"
LG=$HOME/code/lightgen; OURS=$HOME/envs/trellis2/bin/python
GLB=$(cd $LG && $OURS -c 'from data_processing.data_preparation.config import SOLAR_GLB_ROOT; print(SOLAR_GLB_ROOT)')
export HF_HOME=$J/hf_cache TRITON_CACHE_DIR=$LOC/triton TORCHINDUCTOR_CACHE_DIR=$LOC/inductor
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$LOC/flex_gemm.json; cp $HOME/.flex_gemm/autotune_cache.json $FLEX_GEMM_AUTOTUNE_CACHE_PATH 2>/dev/null || true
echo "[lgb] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) tree=$(cat $TREE/TREE_REV 2>/dev/null) glb=$GLB"

# the download folder: the release's files for these shapes, laid out as on Hugging Face
split_of() { case $1 in 0|1) echo train;; 2) echo val;; 3) echo test;; esac; }
for i in 0 1 2 3; do
  s=$(split_of $i); mkdir -p $LOC/dl/data/$s/voxels $LOC/dl/data/$s/thumbnail
  tar -C $REL -rf $LOC/dl/data/$s/voxels/voxels-00000.tar ${U[$i]}/emission_voxels.vxz ${U[$i]}/pbr_voxels.vxz
  tar -C $REL -rf $LOC/dl/data/$s/thumbnail/thumbnail-00000.tar ${U[$i]}/thumbnail.png
done
printf '{"train": ["%s", "%s"], "val": ["%s"], "test": ["%s"]}\n' "${U[0]}" "${U[1]}" "${U[2]}" "${U[3]}" > $LOC/dl/splits.json
(cd $LOC/dl && find data -name '*.tar' | sort | xargs sha256sum > checksums.sha256)
MINI_SHA=$(sha256sum $LOC/dl/splits.json | cut -d' ' -f1)

fail=0
kevin() { ( source $J/miniforge3/etc/profile.d/conda.sh; conda activate trellis2; PYTHON=python3 "$@" ); }
step() { local n=$1; shift; echo "[lgb] $(date +%T) $n"; "$@" > $LOC/out/$n.log 2>&1 || { fail=$((fail+1)); echo "[lgb] $n FAILED"; }; tail -n 2 $LOC/out/$n.log; }
cd $TREE
step prepare kevin env SKIP_DOWNLOAD=1 SPLITS_SHA256=$MINI_SHA WORKERS=4 bash script/lightgenbench/prepare_lightgenbench.sh $LOC/dl $LOC/root
step latents kevin env TEXVERSE_1K=$GLB LIGHTGEN_REPO=$LG NUM_GPUS=1 WORKERS=2 bash script/lightgenbench/run_chain.sh latents $LOC/root
step masks kevin env NUM_GPUS=1 bash script/lightgenbench/run_chain.sh masks $LOC/root
step cond kevin env NUM_GPUS=1 WORKERS=4 bash script/lightgenbench/run_chain.sh cond $LOC/root
step train env PYTHON=$OURS SLURM_JOB_NAME=bash NUM_GPUS=1 BS=1 ANY_BATCH=1 OUTPUTS=$LOC/outputs CHAIN_TMPDIR=$LOC/tmp \
  bash script/lightgenbench/run_chain.sh train $LOC/root --tryrun
step predict kevin env SEEDS=0 OUTPUTS=$LOC/outputs CKPT=$J/outputs/segvigen_256_release_4h100/snapshots/step0044722.pt \
  bash script/lightgenbench/run_chain.sh predict $LOC/root

mkdir -p $H/out/$NAME; cp $LOC/out/*.log $H/out/$NAME/
$OURS - $LOC/root $L $TREE/data/ckpt_init/full_seg_rekeyed.pt "${U[@]}" > $H/out/$NAME/compare.txt 2>&1 <<'PYEOF' || fail=$((fail+1))
import filecmp, os, sys
import numpy as np, torch
root, L, init = sys.argv[1:4]; uuids = sys.argv[4:]
bad = 0
def report(what, ok, detail):
    global bad; bad += not ok; print(f"{what}: {'ok' if ok else 'DIFFERENT'} {detail}")
same = [filecmp.cmp(f"{root}/dual_grid_256/{u}.vxz", f"{L}/dual_grid_256/{u}.vxz", shallow=False) for u in uuids]
report("dual grids byte-identical", all(same), f"{sum(same)}/{len(uuids)}")
for sub in ("shape_latents/shape_enc_next_dc_f16c32_fp16_256", "pbr_latents/tex_enc_next_dc_f16c32_fp16_256", "segvigen_emission_mask_npz"):
    d, c = 0.0, True
    for u in uuids:
        a, b = np.load(f"{root}/{sub}/{u}.npz"), np.load(f"{L}/{sub}/{u}.npz")
        c &= all(np.array_equal(a[k], b[k]) for k in a.files if k != "feats")
        if "feats" in a.files:
            d = max(d, float(np.abs(a["feats"].astype(np.float64) - b["feats"].astype(np.float64)).max()))
    report(sub, c and d <= 5e-3, f"other keys equal={c} feats max|d|={d:.2e}")
d = 0.0
for u in uuids:
    a = torch.load(f"{root}/segvigen_cond_fp16/{u}/cond_pos.pth", map_location="cpu", weights_only=False)
    b = torch.load(f"{L}/segvigen_cond_fp16/{u}/cond_pos.pth", map_location="cpu", weights_only=False)
    a, b = (x["cond"] if isinstance(x, dict) else x for x in (a, b))
    d = max(d, float((a.float() - b.float()).abs().mean()))
report("conditioning", d <= 1e-3, f"mean|d| max over shapes {d:.2e}")
a = torch.load(init, map_location="cpu", weights_only=False)["denoiser"]
b = torch.load(f"{L}/init/full_seg_rekeyed.pt", map_location="cpu", weights_only=False)["denoiser"]
report("init identical", set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in b), f"keys {len(a)}/{len(b)}")
print(f"COMPARE {'PASS' if bad == 0 else 'FAIL'}")
sys.exit(1 if bad else 0)
PYEOF
cat $H/out/$NAME/compare.txt
echo "${U[3]}" > $LOC/test.txt
$OURS $TREE/script/internal/gate/compare_masks.py --a $PUB --b $LOC/outputs/predict_test/seed0/mask/draw0 --shapes $LOC/test.txt \
  > $H/out/$NAME/predict_vs_published.txt 2>&1 || fail=$((fail+1))
tail -n 2 $H/out/$NAME/predict_vs_published.txt
echo "[lgb] $(date) done, $fail failed"
exit $(( fail > 0 ))
