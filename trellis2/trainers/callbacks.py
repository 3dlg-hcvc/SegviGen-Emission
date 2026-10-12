from __future__ import annotations

import time
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


class CheckpointTimingCallback(pl.Callback):
    """Log checkpoint save/load events with timestamps + per-step batch time
    every N steps. Lets us correlate KL/L1 spikes against ckpt I/O.

    Emits to stdout (rank 0 only) — gets picked up by the tee log:

        [ckpt-timing] LOAD step=2420 t=1747312345.12
        [ckpt-timing] SAVE_START step=4488 t=...
        [ckpt-timing] RESUME_AFTER_SAVE step=4489 gap=132.4s t=...
        [ckpt-timing] BATCH_TIME step=4500 dt=1.42s
    """

    def __init__(self, batch_log_every: int = 100) -> None:
        super().__init__()
        self.batch_log_every = int(batch_log_every)
        self._save_t0: Optional[float] = None
        self._batch_t0: Optional[float] = None

    def _emit(self, msg: str) -> None:
        # Rank-0 only via Lightning's is_global_zero (set via trainer at runtime).
        # We can't easily check rank here without trainer; on_*_checkpoint
        # callbacks already only fire on rank 0 in Lightning's default
        # checkpoint flow. on_train_batch_* fires on every rank, so we gate
        # those manually via the trainer arg.
        print(msg, flush=True)

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        step = checkpoint.get("global_step", "?")
        epoch = checkpoint.get("epoch", "?")
        self._emit(f"[ckpt-timing] LOAD step={step} epoch={epoch} t={time.time():.3f}")

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", 0))
        self._save_t0 = time.time()
        self._emit(f"[ckpt-timing] SAVE_START step={step} epoch={epoch} t={self._save_t0:.3f}")

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        # First train batch after a save → roughly when ckpt write + post-save
        # callback chain finished blocking the trainer.
        if self._save_t0 is not None:
            dt = time.time() - self._save_t0
            step = int(getattr(trainer, "global_step", 0))
            self._emit(
                f"[ckpt-timing] RESUME_AFTER_SAVE step={step} gap={dt:.2f}s t={time.time():.3f}"
            )
            self._save_t0 = None
        self._batch_t0 = time.time()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        if self._batch_t0 is None:
            return
        dt = time.time() - self._batch_t0
        self._batch_t0 = None
        step = int(getattr(trainer, "global_step", 0))
        if self.batch_log_every > 0 and step > 0 and step % self.batch_log_every == 0:
            self._emit(f"[ckpt-timing] BATCH_TIME step={step} dt={dt:.2f}s")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Peak VRAM for the whole run (throughput/VRAM probes read this line;
        # torch.cuda.max_memory_allocated is a running high-water mark, so
        # this captures the worst step even if it wasn't the last one).
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            alloc_gb = torch.cuda.max_memory_allocated(i) / 1e9
            reserved_gb = torch.cuda.max_memory_reserved(i) / 1e9
            self._emit(f"[ckpt-timing] PEAK_VRAM device={i} allocated_gb={alloc_gb:.2f} reserved_gb={reserved_gb:.2f}")


class SnapshotCallback(pl.Callback):
    def __init__(
        self,
        *,
        snapshot_batch_size: int = 4,
        val_num_samples: int = 16,
        sample_every_n_steps: int = 0,
        skip_initial_samples: bool = False,
        disable_final_snapshot: bool = False,
    ) -> None:
        super().__init__()
        self.snapshot_batch_size = int(snapshot_batch_size)
        self.val_num_samples = int(val_num_samples)
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.skip_initial_samples = bool(skip_initial_samples)
        # Escape hatch for datasets that don't implement visualize_sample yet
        # (e.g. SegviGenEmissive256): on_train_end's snapshot is otherwise
        # unconditional and will try to save_image() raw SparseTensor /
        # string batch fields. Fixing that needs a real visualize_sample on
        # the dataset (follow-up); until then, set this true to skip only
        # the final snapshot while keeping everything else (mid-training
        # snapshots via sample_every_n_steps, checkpoints) intact.
        self.disable_final_snapshot = bool(disable_final_snapshot)

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
        if self.disable_final_snapshot:
            return
        if hasattr(pl_module, "snapshot"):
            pl_module.snapshot(suffix="final", num_samples=self.val_num_samples, batch_size=self.snapshot_batch_size)
