from __future__ import annotations

from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

import torch
from torch import nn


class ElasticControllerCallback(pl.Callback):
    def __init__(self, elastic_cfg: Optional[Dict[str, Any]]) -> None:
        super().__init__()
        self.elastic_cfg = elastic_cfg
        self.controller = None
        self._ctx = None
        self._pending_state = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.elastic_cfg:
            return
        from trellis2.utils import elastic_utils

        name = self.elastic_cfg.get("name")
        args = self.elastic_cfg.get("args", {})
        controller_cls = getattr(elastic_utils, name, None)
        if controller_cls is None:
            raise ValueError(f"Unknown elastic controller: {name}")
        controller = controller_cls(**args)

        elastic_models = []
        models = getattr(pl_module, "models", None)
        if isinstance(models, (dict, nn.ModuleDict)):
            for model in models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    elastic_models.append(model)
        if not elastic_models:
            if trainer.is_global_zero:
                print("[ElasticControllerCallback] No elastic modules found; disabling controller.")
            return

        for model in elastic_models:
            model.register_memory_controller(controller)
        self.controller = controller
        if self._pending_state is not None:
            self.controller.load_state_dict(self._pending_state)
            self._pending_state = None

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        if self.controller is None:
            return
        self._ctx = self.controller.record()
        self._ctx.__enter__()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        if self.controller is None:
            return
        metrics = self.controller.log()
        if metrics:
            for key, value in metrics.items():
                pl_module.log(f"elastic/{key}", value, on_step=True, on_epoch=False, sync_dist=True)

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        if self.controller is not None:
            checkpoint["elastic_controller"] = self.controller.state_dict()

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        if self.controller is None:
            self._pending_state = checkpoint.get("elastic_controller")
            return
        state = checkpoint.get("elastic_controller")
        if state is not None:
            self.controller.load_state_dict(state)


class EmaCallback(pl.Callback):
    def __init__(self, ema_rates: Optional[list[float]]) -> None:
        super().__init__()
        if ema_rates is None:
            ema_rates = []
        self.ema_rates = [float(r) for r in ema_rates]
        self._ema_state: Dict[str, Dict[str, Any]] = {}
        self._param_names: list[str] = []
        self._params: list[Any] = []

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.ema_rates:
            return
        names = []
        params = []
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                names.append(name)
                params.append(param)
        self._param_names = names
        self._params = params
        if not self._ema_state:
            for rate in self.ema_rates:
                self._ema_state[str(rate)] = {name: p.detach().clone() for name, p in zip(names, params)}
        else:
            for key, bucket in self._ema_state.items():
                self._ema_state[key] = {n: t.to(pl_module.device) for n, t in bucket.items()}

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self._ema_state:
            return
        with torch.no_grad():
            for rate in self.ema_rates:
                bucket = self._ema_state.get(str(rate))
                if bucket is None:
                    continue
                for name, param in zip(self._param_names, self._params):
                    ema_param = bucket[name]
                    ema_param.mul_(rate).add_(param.detach(), alpha=1.0 - rate)

    @contextmanager
    def apply_ema(self, pl_module: "pl.LightningModule", rate: Optional[float] = None):
        if not self._ema_state:
            yield
            return
        if rate is None:
            rate = self.ema_rates[0]
        bucket = self._ema_state.get(str(rate))
        if bucket is None:
            yield
            return
        backup = {name: param.detach().clone() for name, param in zip(self._param_names, self._params)}
        try:
            for name, param in zip(self._param_names, self._params):
                param.data.copy_(bucket[name])
            yield
        finally:
            for name, param in zip(self._param_names, self._params):
                param.data.copy_(backup[name])

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        if self._ema_state:
            checkpoint["ema"] = {
                "rates": self.ema_rates,
                "state": {k: {n: t.cpu() for n, t in v.items()} for k, v in self._ema_state.items()},
            }

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        ema = checkpoint.get("ema")
        if not ema:
            return
        rates = [float(r) for r in ema.get("rates", [])]
        state = ema.get("state", {})
        self.ema_rates = rates
        self._ema_state = {}
        for key, bucket in state.items():
            self._ema_state[str(key)] = {n: t for n, t in bucket.items()}


class SamplerStateCallback(pl.Callback):
    def _get_sampler(self, trainer: "pl.Trainer"):
        datamodule = trainer.datamodule
        if datamodule is not None and getattr(datamodule, "train_sampler", None) is not None:
            return datamodule.train_sampler
        loaders = trainer.train_dataloader
        loader = loaders[0] if isinstance(loaders, list) else loaders
        if loader is not None:
            return getattr(loader, "sampler", None)
        return None

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        sampler = self._get_sampler(trainer)
        if sampler is not None and hasattr(sampler, "state_dict"):
            checkpoint["sampler_state"] = sampler.state_dict()

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        sampler = self._get_sampler(trainer)
        state = checkpoint.get("sampler_state")
        if sampler is not None and state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state)


class SnapshotCallback(pl.Callback):
    def __init__(
        self,
        *,
        snapshot_batch_size: int = 4,
        val_num_samples: int = 16,
        sample_every_n_steps: int = 0,
        skip_initial_samples: bool = False,
    ) -> None:
        super().__init__()
        self.snapshot_batch_size = int(snapshot_batch_size)
        self.val_num_samples = int(val_num_samples)
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.skip_initial_samples = bool(skip_initial_samples)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.is_global_zero:
            return
        if hasattr(pl_module, "snapshot_dataset"):
            pl_module.snapshot_dataset(batch_size=self.snapshot_batch_size)
        if not self.skip_initial_samples and hasattr(pl_module, "snapshot"):
            pl_module.snapshot(suffix="init", num_samples=self.val_num_samples, batch_size=self.snapshot_batch_size)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        if not trainer.is_global_zero:
            return
        if self.sample_every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0:
            return
        if step % self.sample_every_n_steps != 0:
            return
        if hasattr(pl_module, "snapshot"):
            pl_module.snapshot(
                suffix=f"step{step:07d}",
                num_samples=self.val_num_samples,
                batch_size=self.snapshot_batch_size,
            )

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.is_global_zero:
            return
        if hasattr(pl_module, "snapshot"):
            pl_module.snapshot(suffix="final", num_samples=self.val_num_samples, batch_size=self.snapshot_batch_size)
