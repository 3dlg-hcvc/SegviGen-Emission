#!/bin/bash
# Item 2(b): GPU smoke on venus-05 after bootstrap -- torch.cuda + flash_attn
# import (expected to be ABSENT/unused, see the sm_120 facts below) + one
# tiny forward. Also answers the owner's node-local data question (see the
# ls/du lines at the end): does Dongchen's node-local data/ hold real
# copies or symlinks back to jupiter.
#SBATCH -J smoke_gpu_venus05
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/smoke_gpu_venus05_%j.log
set -x
D=/localscratch/xya120
source $D/miniforge3/etc/profile.d/conda.sh
conda activate segvigen-bw
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
# sm_120 facts (job 10 harvest, venus-19-proven in Dongchen's launchers):
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

nvidia-smi
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("tiny matmul OK, result mean:", y.mean().item())
try:
    import flash_attn
    print("flash_attn import OK (unexpected on sm_120):", flash_attn.__version__)
except Exception as e:
    print("flash_attn import FAILED (expected on sm_120):", repr(e)[:200])
import xformers, xformers.ops
print("xformers", xformers.__version__)
q = torch.randn(1, 8, 128, 64, device="cuda")
out = xformers.ops.memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
print("xformers memory_efficient_attention OK, out shape:", tuple(out.shape))
EOF

echo "=== owner question: does Dongchen's node-local data/ hold real copies or symlinks? ==="
ls -la /localscratch/dya78/TRELLIS.2-lightning/data/ 2>/dev/null | head -15
du -sh /localscratch/dya78/TRELLIS.2-lightning/data/* 2>/dev/null | head -10
