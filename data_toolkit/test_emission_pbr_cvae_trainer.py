"""Smoke: EmissionPbrConditionalVaeTrainer.training_losses runs end-to-end on
a tiny fake batch and returns a finite scalar loss."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from trellis2.trainers.vae.emission_pbr_conditional import EmissionPbrConditionalVaeTrainer
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
from trellis2.modules.sparse.basic import SparseTensor


# Build a tiny encoder / decoder so the smoke runs fast.
encoder = SparseUnetVaeEncoder(
    in_channels=3,
    model_channels=[8, 16, 32, 64],
    latent_channels=32,
    num_blocks=[0, 1, 1, 1],
    block_type=["SparseConvNeXtBlock3d"] * 4,
    down_block_type=["SparseResBlockS2C3d"] * 3,
    block_args=[{"use_checkpoint": False}] * 4,
    use_fp16=False,
)
decoder = SparseUnetVaeDecoder(
    out_channels=3,
    model_channels=[64, 32, 16, 8],
    latent_channels=64,
    num_blocks=[1, 1, 1, 0],
    block_type=["SparseConvNeXtBlock3d"] * 4,
    up_block_type=["SparseResBlockC2S3d"] * 3,
    block_args=[{"use_checkpoint": False}] * 4,
    use_fp16=False,
    pred_subdiv=False,
)

# Minimal trainer instance (skip __init__ — no finetune_ckpt, no Lightning setup).
trainer = EmissionPbrConditionalVaeTrainer.__new__(EmissionPbrConditionalVaeTrainer)
trainer.models = {"encoder": encoder, "decoder": decoder}
trainer.training_models = {"encoder": encoder, "decoder": decoder}
trainer.loss_type = "l1"
trainer.lambda_kl = 1e-7
trainer.emission_channels = 3
trainer.stage_a_steps = 100
trainer.lr_interior_stage_b = 3e-5
trainer._stage_b_applied = False

# Fake sparse batch: 8 active voxels at level 0; 1 latent voxel at level 8 (after
# 3 stride-2 downsamples — coord scale is /8).
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device)
decoder.to(device)

# 8 voxels at coords (0,0,0..7) so they form a single 2x2x2 cube downsampled to (0,0,0).
N = 8
vx_grid = torch.tensor(
    [[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
    dtype=torch.int32,
)
vx_coords = torch.cat([torch.zeros(N, 1, dtype=torch.int32), vx_grid], dim=-1).to(device)
vx_feats = torch.randn(N, 3, device=device)
x = SparseTensor(vx_feats, vx_coords)

# 1 latent voxel at (0,0,0). After 3 stride-2 downsamples, the 2x2x2 cube collapses to 1.
M = 1
lat_coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device)
pbr_latent = SparseTensor(torch.randn(M, 32, device=device), lat_coords)

try:
    terms, _ = trainer.training_losses(x=x, pbr_latent=pbr_latent)
    assert "loss" in terms, "loss key missing"
    loss = terms["loss"]
    assert torch.isfinite(loss).all(), f"loss is not finite: {loss}"
    print(f"PASS: training_losses returns finite loss = {float(loss):.4f}")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(f"\nFAIL: training_losses raised: {exc!r}")
    sys.exit(1)
