"""SegviGen's training semantics on the framework's sparse flow trainer.

Differences from ImageConditionedSparseFlowMatchingCFGTrainer, each deliberate:

1. PRECOMPUTED CONDITIONING. The dataset may hand `cond` as the DINOv3 token
   tensor itself (B, 1029, 1024) instead of an image. In that case the on-GPU
   encoder never runs and neg_cond is zeros_like(cond), which is exactly what
   the standalone lane precomputed (its cond.pth stores the same zero negative).
   Image input still works and takes the parent's encode path.
2. PER-SHAPE MEAN LOSS REDUCTION (always on, independent of weighting).
   Each shape's token-mean is computed first, then averaged over shapes in
   the batch. A token-count-weighted batch mean is NOT the same gradient;
   the assessment page flags this as a decision, and the decision here is:
   keep our lane's semantics so runs remain comparable to the standalone
   arm. The standalone lane's --pos_weight / --balanced_pos_weight
   per-voxel weighting is reproduced too, but DORMANT by default
   (pos_weight 1.0, balanced_pos_weight 0.0 -- owner-set 2026-08-26: the
   production recipe trains unweighted; the July w5 weighted experiments
   lost). emis_mask is only required when a config turns weighting back
   on (dataset's require_mask defaults to false accordingly).
3. p_uncond defaults to 0.0: our lane never trained with CFG dropout.

Batch keys consumed here: x_0, cond, emis_mask (optional, weighting only),
sha256; everything else (concat_cond, appearance_cond) passes through to
the model, which is how SegviGenSLatFlowModel receives its extra tokens.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ...modules import sparse as sp
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .sparse_flow_matching import ImageConditionedSparseFlowMatchingCFGTrainer

__all__ = ["SegviGenFlowTrainer"]


class SegviGenFlowTrainer(ImageConditionedSparseFlowMatchingCFGTrainer):
    def __init__(self, *args, pos_weight: float = 1.0,
                 balanced_pos_weight: float = 0.0, p_uncond: float = 0.0, **kwargs):
        super().__init__(*args, p_uncond=p_uncond, **kwargs)
        self.pos_weight = float(pos_weight)
        self.balanced_pos_weight = float(balanced_pos_weight)

    def get_cond(self, cond, **kwargs):
        if isinstance(cond, torch.Tensor) and cond.ndim == 3:
            # precomputed DINOv3 tokens: skip the encoder, zero negative
            kwargs["neg_cond"] = torch.zeros_like(cond)
            # skip ImageConditionedMixin.get_cond (which would run the encoder
            # on what is not an image); call the CFG mixin's directly
            return ClassifierFreeGuidanceMixin.get_cond(self, cond, **kwargs)
        return super().get_cond(cond, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        # mirror of get_cond above: run_snapshot's sampling path hits this,
        # and ImageConditionedMixin.get_inference_cond would call
        # encode_image() on our precomputed (B, 1029, 1024) DINOv3 tokens
        # and assert (it expects a raw image tensor/PIL list).
        if isinstance(cond, torch.Tensor) and cond.ndim == 3:
            kwargs["neg_cond"] = torch.zeros_like(cond)
            return ClassifierFreeGuidanceMixin.get_inference_cond(self, cond, **kwargs)
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        # ImageConditionedMixin.vis_cond assumes cond is a real image tensor
        # and hands it straight to save_image; our precomputed cond has no
        # image representation, so there is nothing honest to render here.
        # Returning {} means "no cond panel", not "broken panel".
        if isinstance(cond, torch.Tensor) and cond.ndim == 3:
            return {}
        return super().vis_cond(cond, **kwargs)

    def _per_voxel_weights(self, mask_feats: torch.Tensor) -> torch.Tensor:
        if self.balanced_pos_weight > 0:
            p = mask_feats.mean().clamp(min=1e-4)
            W = torch.clamp((1 - p) / p, max=self.balanced_pos_weight)
            w = 1 + (W - 1) * mask_feats
        elif self.pos_weight != 1.0:
            w = 1 + (self.pos_weight - 1) * mask_feats
        else:
            return torch.ones_like(mask_feats)
        return w / w.mean().clamp(min=1e-8)

    def training_losses(self, x_0: sp.SparseTensor, cond=None,
                        emis_mask=None, sha256=None, **kwargs) -> Tuple[Dict, Dict]:
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **{})

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        per_shape = []
        for i in range(x_0.shape[0]):
            sl = x_0.layout[i]
            se = (pred.feats[sl] - target.feats[sl]) ** 2
            if emis_mask is not None:
                w = self._per_voxel_weights(emis_mask.feats[sl][:, 0:1])
                per_shape.append((w * se).mean())
            else:
                per_shape.append(se.mean())
        loss = torch.stack(per_shape).mean()

        terms = edict()
        terms["mse"] = loss.detach()
        terms["loss"] = loss
        return terms, {}
