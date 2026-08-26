#!/bin/bash
# Blackwell (RTX Pro 6000, sm_120) bootstrap for OUR SegviGen-on-lightning
# port, adapted from Dongchen's bootstrap_bw05.sh
# (/3dlg-jupiter-project/lightgen/trellis2_bw/bootstrap_bw05.sh). Same
# recipe (separate torch cu128 + xformers env, no flash-attn -- it does not
# support sm_120 yet, extensions rebuilt for arch 12.0), pointed at our
# user's local scratch and our own private repo instead of the shared
# upstream one.
#
# NOT FOR SUBMISSION YET: venus-05 and venus-19 are both DRAINING for
# "Mercury deployment" as of 2026-08-26 (owner-set exclusion, see
# reference_solar_blackwell_nodes.md and the standing exclusions list).
# This script is prep only -- ready to run once a node is back and the
# owner clears it. Idempotent throughout (matches upstream's pattern): safe
# to re-run after a partial failure.
#
# UNVERIFIED (flagged, not assumed): whether venus-05/19 have outbound
# GitHub access for a PRIVATE repo. QhelDIV/segvigen-lightning is private;
# this script clones over SSH using the same deploy pattern as the
# workstation's "mine" remote, which requires an SSH key for the QhelDIV
# GitHub account to already be provisioned on the node (or reachable via a
# forwarded agent). Dongchen's script could use a plain HTTPS clone because
# upstream TRELLIS.2-lightning is public; ours cannot. If no such key
# exists on the node, this step will hang/fail at the git clone and needs
# either a deploy key placed at $D/.ssh, forwarded agent auth, or a
# personal access token swapped into an HTTPS remote URL instead.
set -uo pipefail
D=/localscratch/xya120
REPO_SSH=git@github.com:QhelDIV/segvigen-lightning.git
J=/3dlg-jupiter-project/lightgen/trellis2_bw
export TMPDIR=$D/tmp; mkdir -p "$TMPDIR"
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-16}

echo "=== [0/6] miniforge3 (per-node, idempotent) ==="
if [ ! -x $D/miniforge3/bin/conda ]; then
  mkdir -p $D
  curl -fsSL -o $D/miniforge_installer.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash $D/miniforge_installer.sh -b -p $D/miniforge3
  rm -f $D/miniforge_installer.sh
fi
source $D/miniforge3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "=== [1/6] env segvigen-bw (python 3.10) ==="
if [ -d $D/miniforge3/envs/segvigen-bw ] && [ ! -x $D/miniforge3/envs/segvigen-bw/bin/python ]; then
  echo "removing half-created segvigen-bw env"
  conda env remove -n segvigen-bw -y 2>/dev/null || rm -rf $D/miniforge3/envs/segvigen-bw
fi
[ -x $D/miniforge3/envs/segvigen-bw/bin/python ] || conda create -y -n segvigen-bw python=3.10
conda activate segvigen-bw
python -V

echo "=== [2/6] torch cu128 + xformers ==="
python -c 'import torch' 2>/dev/null || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c 'import xformers' 2>/dev/null || pip install xformers --index-url https://download.pytorch.org/whl/cu128
# GPU_SMOKE=0 (default): matches upstream's caution -- verify CUDA context
# init on this node is actually healthy (a maintenance drain does not
# guarantee that) before assuming a GPU smoke test is safe. Rerun with
# GPU_SMOKE=1 once the node is confirmed sane.
GPU_SMOKE=${GPU_SMOKE:-0}
python - <<'EOF'
import torch, xformers
print("torch", torch.__version__, "built-for-cuda", torch.version.cuda)
print("xformers", xformers.__version__)
EOF

echo "=== [3/6] CUDA toolchain exports ==="
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
nvcc --version | tail -1

echo "=== [4/6] repo clone (private SSH; see UNVERIFIED note above) + main branch ==="
if [ ! -d $D/segvigen-lightning/.git ]; then
  mkdir -p $D/segvigen-lightning
  cd $D/segvigen-lightning
  git init
  git remote add origin ${REPO_SSH} 2>/dev/null || git remote set-url origin ${REPO_SSH}
  git fetch origin main
  git checkout -B main origin/main
  git branch --set-upstream-to=origin/main main
else
  cd $D/segvigen-lightning
  git fetch origin main && git checkout main && git pull --ff-only origin main
fi
echo "repo at: $(git log --oneline -1)"
mkdir -p data log outputs

echo "=== [5/6] setup.sh stages (no flash-attn; sm_120 builds) ==="
for x in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do rm -rf /tmp/extensions/$x 2>/dev/null || true; done
source setup.sh --basic || echo "WARN: basic stage returned nonzero (pillow-simd is the usual culprit)"
pip install -q pillow
pip install -q pyarrow wandb easydict
source setup.sh --nvdiffrast
source setup.sh --nvdiffrec
source setup.sh --cumesh
source setup.sh --o-voxel
source setup.sh --flexgemm || echo "WARN: flexgemm build failed"

echo "=== [5.5/6] HF caches (DINOv3 image cond model + any base ckpts) ==="
mkdir -p ~/.cache/huggingface/hub
for m in models--facebook--dinov3-vitl16-pretrain-lvd1689m; do
  if [ ! -d ~/.cache/huggingface/hub/$m ] && [ -d $J/hf_stage/$m ]; then
    rsync -a $J/hf_stage/$m ~/.cache/huggingface/hub/
  fi
done
ls ~/.cache/huggingface/hub/ 2>/dev/null || echo "WARN: no HF cache staged yet; DinoV3FeatureExtractor will try to fetch live (needs HF_HUB_OFFLINE=0 or a pre-populated cache)"

echo "=== [6/6] import smoke (CPU-only unless GPU_SMOKE=1) ==="
python - <<EOF
import importlib, os
ok = True
mods = ["torch", "lightning", "nvdiffrast.torch", "o_voxel", "flex_gemm", "cumesh", "xformers", "transformers"]
if os.environ.get("GPU_SMOKE", "${GPU_SMOKE}") == "1":
    mods.append("xformers.ops")
for m in mods:
    try:
        importlib.import_module(m)
        print("OK  ", m)
    except Exception as e:
        ok = False
        print("FAIL", m, "->", repr(e)[:200])
if os.environ.get("GPU_SMOKE", "${GPU_SMOKE}") == "1":
    import torch
    print("cuda avail:", torch.cuda.is_available())
    ok = ok and torch.cuda.is_available()
else:
    print("cuda check SKIPPED (rerun with GPU_SMOKE=1 once the node is confirmed sane post-drain)")
print("SMOKE", "PASS" if ok else "FAIL")
EOF
grep -q 'machine api.wandb.ai' ~/.netrc 2>/dev/null && echo "wandb netrc: present" || echo "WARN: no wandb entry in netrc (use WANDB_MODE=offline if not set up)"
echo "=== bootstrap_segvigen_bw done $(date) ==="
