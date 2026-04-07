from typing import *
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, l2_loss


class EmissionPbrFinetuneTrainer(BasicTrainer):
    """
    Finetune the pretrained PBR VAE on emission data.

    Strategy: pack emission RGB into the PBR VAE's 6-channel input format
    (base_color=emission, metallic=0, roughness=1, alpha=1) and compute
    reconstruction loss only on the first 3 output channels (emission).

    This allows 100% weight reuse from the pretrained PBR VAE with no
    shape mismatches.  Use ``finetune_ckpt`` in the config to initialise
    encoder/decoder from the pretrained weights.

    Batch dict keys (from LightGenSLatEmission):
        x   SparseTensor  [N, emission_ch + pbr_ch]  (only emission_ch used)

    Args:
        loss_type          : "l1" or "l2"
        lambda_kl          : KL regularisation weight
        emission_channels  : number of emission channels at the start of x.feats (default 3)
    """

    # Default padding values in [-1, 1] normalised space
    # metallic=0 → -1, roughness=1 → 1, alpha=1 → 1
    PAD_METALLIC: float = -1.0
    PAD_ROUGHNESS: float = 1.0
    PAD_ALPHA: float = 1.0

    def __init__(
        self,
        *args,
        loss_type: str = "l1",
        lambda_kl: float = 1e-7,
        emission_channels: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.emission_channels = emission_channels

    def _pack_emission_as_pbr(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """Pack emission (3ch) into PBR 6-channel format.

        Output channels: [emission_rgb(3), metallic(1), roughness(1), alpha(1)]
        """
        emission = x.feats[:, :self.emission_channels]  # [N, 3]
        n = emission.shape[0]
        pad = torch.tensor(
            [[self.PAD_METALLIC, self.PAD_ROUGHNESS, self.PAD_ALPHA]],
            device=emission.device, dtype=emission.dtype,
        ).expand(n, -1)  # [N, 3]
        return x.replace(torch.cat([emission, pad], dim=-1))  # [N, 6]

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / max(1, torch.cuda.device_count()))),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(self, x: sp.SparseTensor, **kwargs):
        """
        x : SparseTensor  [N, emission_ch + pbr_ch]
            Full features from LightGenSLatEmission — only emission_ch used.
        """
        target = x.feats[:, :self.emission_channels]  # [N, 3]
        x_packed = self._pack_emission_as_pbr(x)      # [N, 6]

        z, mean, logvar = self.training_models["encoder"](
            x_packed, sample_posterior=True, return_raw=True
        )
        y = self.training_models["decoder"](z)

        # Loss only on emission channels (first 3 of 6)
        pred_emission = y.feats[:, :self.emission_channels]

        terms = edict(loss=0.0)
        if self.loss_type == "l1":
            terms["l1"] = l1_loss(target, pred_emission)
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == "l2":
            terms["l2"] = l2_loss(target, pred_emission)
            terms["loss"] = terms["loss"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type!r}")

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        return terms, {}

    @torch.no_grad()
    def snapshot_dataset(self, num_samples: int = 100, batch_size: int = 4):
        """Skip — sparse 3D data not visualisable as images."""
        pass

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool = False) -> Dict:
        """Report mean-absolute emission reconstruction error on the training set."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=min(batch_size, len(self.dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
        )
        self.models["encoder"].eval()
        self.models["decoder"].eval()
        errors: List[float] = []
        for data in dataloader:
            data = recursive_to_device(data, self.device)
            x = data["x"]
            target = x.feats[:, :self.emission_channels]
            x_packed = self._pack_emission_as_pbr(x)

            z = self.models["encoder"](x_packed)
            y = self.models["decoder"](z)
            pred_emission = y.feats[:, :self.emission_channels]
            errors.append((target - pred_emission).abs().mean().item())
            if len(errors) * batch_size >= num_samples:
                break
        self.models["encoder"].train()
        self.models["decoder"].train()
        mean_err = float(np.mean(errors)) if errors else 0.0
        return {"emission_recon_l1": {"value": torch.tensor(mean_err), "type": "scalar"}}
