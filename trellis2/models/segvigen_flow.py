"""SegviGen's voxel-axis 2N conditioning as a subclass of the stock flow model.

SegviGen conditions on appearance by adding the appearance latent as a SECOND
set of tokens at the same voxel coordinates (sequence 2N), rather than as extra
channels. That keeps in_channels at 64, so every pretrained tensor of
slat_flow_imgshape2tex_dit_1_3B_512_bf16 is reused, input projection included.
The trick lives entirely in this wrapper: the stock model underneath is untouched.

forward() packs [noised target | appearance] per shape before the stock
forward and keeps the first half of each shape's output after it. It calls
super().forward() so SparseTransformerElasticMixin sees the DOUBLED sequence,
which is the sequence that actually costs memory.

Port of inference_full.Gen3DSeg (the training/inference wrapper the standalone
SegviGen trainer uses).
"""
from typing import List, Optional, Union

import torch

from ..modules import sparse as sp
from .structured_latent_flow import ElasticSLatFlowModel

__all__ = ["SegviGenSLatFlowModel"]


class SegviGenSLatFlowModel(ElasticSLatFlowModel):
    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        appearance_cond: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        assert appearance_cond is not None, (
            "SegviGenSLatFlowModel needs the dataset to supply appearance_cond "
            "(the input texture latent as a SparseTensor on x's coordinates)")
        assert concat_cond is not None, (
            "SegviGenSLatFlowModel needs concat_cond (the shape latent)")

        feats, coords, ccf, lens = [], [], [], []
        for i in range(x.shape[0]):
            sl = x.layout[i]
            n = sl.stop - sl.start
            feats += [x.feats[sl], appearance_cond.feats[sl]]
            coords += [x.coords[sl], appearance_cond.coords[sl]]
            # shape latent rides along duplicated, one copy per half
            ccf += [concat_cond.feats[sl], concat_cond.feats[sl]]
            lens.append(n)
        x2 = sp.SparseTensor(torch.cat(feats), torch.cat(coords))
        cc2 = x2.replace(torch.cat(ccf))

        # timestep: one t per shape; the doubled sequence keeps the same batch
        # indices, so the per-token adaLN lookup is unchanged.
        h = super().forward(x2, t, cond, concat_cond=cc2, **kwargs)

        out_f, out_c, b = [], [], 0
        for n in lens:
            out_f.append(h.feats[b:b + n])
            out_c.append(h.coords[b:b + n])
            b += 2 * n
        return sp.SparseTensor(torch.cat(out_f), torch.cat(out_c))
