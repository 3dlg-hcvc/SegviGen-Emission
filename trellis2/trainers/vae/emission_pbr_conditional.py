"""Conditional emission VAE: encoder ingests emission only; decoder is
conditioned on the pre-encoded PBR latent via channel-concat at the decoder
input. Surgery on the pretrained PBR VAE in `_apply_finetune_checkpoint`."""
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
from ...utils.model_surgery import surgically_init_emission_pbr_cvae


class EmissionPbrConditionalVaeTrainer(BasicTrainer):
    """Conditional emission VAE finetuned from the pretrained PBR VAE.

    Surgery (applied in `_apply_finetune_checkpoint`):
        * encoder.input_layer        : 6 -> 64 (3 -> 64 with cols 0:3 of pretrained)
        * decoder.from_latent        : 32 -> 1024  (64 -> 1024 with cols 0:32 of pretrained, 32:64 zero)
        * decoder.output_layer       : 1024 -> 6  (1024 -> 3 with rows 0:3 of pretrained)
        * all `blocks.*` / `to_subdiv` / `to_latent` : copied 1:1.

    Training schedule:
        Stage A (step < stage_a_steps): freeze encoder.blocks / decoder.blocks;
            train surgical layers + to_latent only. Single param group @ lr_peak.
        Stage B (step >= stage_a_steps): unfreeze all interior blocks; two
            param groups — surgical layers @ lr_peak, interior @ lr_interior_stage_b.

    Batch dict keys (from extended LightGenSLatEmission):
        x           SparseTensor [N, 3]   emission voxels
        pbr_latent  SparseTensor [M, 32]  pre-encoded PBR latent
    """

    def __init__(
        self,
        *args,
        loss_type: str = "l1",
        lambda_kl: float = 1e-7,
        emission_channels: int = 3,
        stage_a_steps: int = 1000,
        lr_interior_stage_b: float = 3e-5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.emission_channels = emission_channels
        self.stage_a_steps = stage_a_steps
        self.lr_interior_stage_b = lr_interior_stage_b
        self._stage_b_applied = False

    # ------------------------------------------------------------------
    # Surgery hook — replaces the default partial-load behaviour
    # ------------------------------------------------------------------
    def _apply_finetune_checkpoint(self) -> None:
        if self._finetune_applied:
            return
        finetune_ckpt = getattr(self, "finetune_ckpt", None)
        if not isinstance(finetune_ckpt, dict) or not finetune_ckpt:
            self._finetune_applied = True
            return
        enc_path = finetune_ckpt.get("encoder")
        dec_path = finetune_ckpt.get("decoder")
        if enc_path is None or dec_path is None:
            raise ValueError(
                "EmissionPbrConditionalVaeTrainer requires both 'encoder' and "
                "'decoder' paths in finetune_ckpt."
            )
        self._warn(f"[surgery] enc={enc_path}")
        self._warn(f"[surgery] dec={dec_path}")
        surgically_init_emission_pbr_cvae(
            self.models["encoder"],
            self.models["decoder"],
            enc_path,
            dec_path,
        )
        self._finetune_applied = True

    # ------------------------------------------------------------------
    # Stage A / Stage B param-group machinery
    #
    # The base trainer's `configure_optimizers` does
    # `build_optimizer(cfg, list(self.parameters()))` — i.e. all params land in
    # param-group 0. That conflicts with later `add_param_group(interior)`
    # because interior would be in two groups (double-update, two LRs).
    # We override `configure_optimizers` to build the optimizer over the
    # surgical params ONLY (Stage A) or both groups from scratch (Stage B
    # resume), so `_apply_stage_b` can cleanly add interior as group 1.
    # ------------------------------------------------------------------
    def _interior_parameters(self) -> List[torch.nn.Parameter]:
        return [
            *self.models["encoder"].blocks.parameters(),
            *self.models["decoder"].blocks.parameters(),
        ]

    def _surgical_parameters(self) -> List[torch.nn.Parameter]:
        enc = self.models["encoder"]
        dec = self.models["decoder"]
        return [
            *enc.input_layer.parameters(),
            *enc.to_latent.parameters(),
            *dec.from_latent.parameters(),
            *dec.output_layer.parameters(),
        ]

    def _freeze_interior(self) -> None:
        for p in self._interior_parameters():
            p.requires_grad_(False)

    def _unfreeze_interior(self) -> None:
        for p in self._interior_parameters():
            p.requires_grad_(True)

    def configure_optimizers(self):
        from ..builders import build_optimizer, build_lr_scheduler

        starting_in_stage_b = self.global_step >= self.stage_a_steps

        if starting_in_stage_b:
            # All params trainable; build two-group optimizer up front.
            self._unfreeze_interior()
            param_groups = [
                {"params": self._surgical_parameters()},  # uses optimizer args' lr
                {"params": self._interior_parameters(), "lr": self.lr_interior_stage_b},
            ]
            self._stage_b_applied = True
        else:
            # Stage A: freeze interior, build optimizer over surgical only.
            self._freeze_interior()
            param_groups = [{"params": self._surgical_parameters()}]
            self._stage_b_applied = False

        optimizer = build_optimizer(self.optimizer_cfg, param_groups)
        scheduler = build_lr_scheduler(self.lr_scheduler_cfg, optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _apply_stage_b(self) -> None:
        if self._stage_b_applied:
            return
        self._unfreeze_interior()
        opts = self.optimizers()
        opt = opts[0] if isinstance(opts, (list, tuple)) else opts
        opt.add_param_group(
            {"params": self._interior_parameters(), "lr": self.lr_interior_stage_b}
        )
        self._stage_b_applied = True
        self._warn(f"[stage-b] Unfroze interior blocks at step {self.global_step}; "
                   f"added param group @ lr={self.lr_interior_stage_b}")

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == self.stage_a_steps and not self._stage_b_applied:
            self._apply_stage_b()

    # ------------------------------------------------------------------
    # Dataloader (identical to EmissionVaeTrainer)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def training_losses(
        self,
        x: sp.SparseTensor,
        pbr_latent: sp.SparseTensor,
        **kwargs,
    ):
        """Emission VAE forward + L1 reconstruction + KL.

        x.feats:           [N, 3]  — emission target (also encoder input)
        pbr_latent.feats:  [M, 32] — pre-encoded PBR conditioning
        """
        target = x.feats[:, : self.emission_channels]  # [N, 3]

        z, mean, logvar = self.training_models["encoder"](
            x, sample_posterior=True, return_raw=True
        )  # z: SparseTensor [M, 32]

        # Sanity: latent coords must match conditioning coords.
        assert torch.equal(z.coords, pbr_latent.coords), (
            f"z_emit / pbr_latent coord mismatch: {z.coords.shape} vs {pbr_latent.coords.shape}"
        )

        z_cat = z.replace(
            torch.cat([z.feats, pbr_latent.feats], dim=-1)
        )  # [M, 64]
        y = self.training_models["decoder"](z_cat)  # SparseTensor [N, 3]

        terms = edict(loss=0.0)
        if self.loss_type == "l1":
            terms["l1"] = l1_loss(target, y.feats)
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == "l2":
            terms["l2"] = l2_loss(target, y.feats)
            terms["loss"] = terms["loss"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type!r}")

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        terms["stage"] = torch.tensor(1.0 if self._stage_b_applied else 0.0)
        return terms, {}

    # ------------------------------------------------------------------
    # Snapshot (mean-absolute emission reconstruction error)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def snapshot_dataset(self, num_samples: int = 100, batch_size: int = 4):
        """Sparse 3-D — no image snapshot."""
        pass

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool = False) -> Dict:
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
            pbr_latent = data["pbr_latent"]
            target = x.feats[:, : self.emission_channels]
            z = self.models["encoder"](x)
            z_cat = z.replace(
                torch.cat([z.feats, pbr_latent.feats], dim=-1)
            )
            y = self.models["decoder"](z_cat)
            errors.append((target - y.feats).abs().mean().item())
            if len(errors) * batch_size >= num_samples:
                break
        self.models["encoder"].train()
        self.models["decoder"].train()
        mean_err = float(np.mean(errors)) if errors else 0.0
        return {"emission_recon_l1": {"value": torch.tensor(mean_err), "type": "scalar"}}
