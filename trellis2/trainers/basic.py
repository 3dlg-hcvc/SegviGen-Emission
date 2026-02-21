from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from contextlib import nullcontext
import os
from abc import abstractmethod

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils

from trellis2.utils.data_utils import recursive_to_device
from trellis2.utils.general_utils import save_image_with_notes
from .utils import str_to_dtype
from .builders import build_lr_scheduler, build_models, build_optimizer


def flatten_dict(data: Dict[str, Any], sep: str = "/", prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, sep=sep, prefix=name))
        else:
            flat[name] = value
    return flat


class BasicTrainer(pl.LightningModule):
    def __init__(
        self,
        models: Dict[str, Any],
        dataset: Any = None,
        *,
        optimizer: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Dict[str, Any]] = None,
        grad_clip: Optional[Any] = None,
        fp16_mode: Optional[str] = None,
        mix_precision_mode: Optional[str] = None,
        mix_precision_dtype: Any = "float16",
        batch_split: int = 1,
        snapshot_batch_size: int = 4,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if optimizer is None:
            optimizer = {}

        self.models_cfg = models
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.grad_clip_cfg = grad_clip

        # Legacy compatibility: fp16_mode overrides mix_precision_mode in old configs.
        if fp16_mode is not None and mix_precision_mode is None:
            mix_precision_mode = "amp" if str(fp16_mode).lower() in ("amp", "inflat_all") else fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype) if isinstance(mix_precision_dtype, str) else mix_precision_dtype

        self.batch_split = int(batch_split)
        self.snapshot_batch_size = int(snapshot_batch_size)
        self._dataset = dataset
        self.output_dir = output_dir

        built = build_models(models)
        self.models = nn.ModuleDict(built)
        self.training_models = self.models

        self._grad_clipper = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def on_fit_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            self._dataset = getattr(dm, "train_dataset", None)

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = build_optimizer(self.optimizer_cfg, params)
        scheduler = build_lr_scheduler(self.lr_scheduler_cfg, optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx=None,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        *args,
        **kwargs,
    ):
        if self.grad_clip_cfg is None:
            return

        if isinstance(self.grad_clip_cfg, (float, int)):
            torch.nn.utils.clip_grad_norm_(self.parameters(), float(self.grad_clip_cfg))
            return

        name = self.grad_clip_cfg.get("name")
        cfg_args = self.grad_clip_cfg.get("args", {})
        if name == "AdaptiveGradClipper":
            if self._grad_clipper is None:
                from trellis2.utils.grad_clip_utils import AdaptiveGradClipper

                self._grad_clipper = AdaptiveGradClipper(**cfg_args)
            self._grad_clipper(self.parameters())
            return

        from trellis2.utils import grad_clip_utils

        if self._grad_clipper is None:
            clipper_cls = getattr(grad_clip_utils, name, None)
            if clipper_cls is None:
                raise ValueError(f"Unknown grad clipper: {name}")
            self._grad_clipper = clipper_cls(**cfg_args)
        self._grad_clipper(self.parameters())

    def _autocast_context(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None or self.device.type != "cuda":
            return nullcontext()

        precision = getattr(trainer, "precision", None)
        if precision in ("16-mixed", "16"):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision in ("bf16-mixed", "bf16"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        if self.mix_precision_mode in ("amp", "mixed"):
            return torch.autocast(device_type="cuda", dtype=self.mix_precision_dtype)

        return nullcontext()

    def _extract_terms(self, output: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(output, tuple) and len(output) == 2:
            return output[0], output[1]
        if isinstance(output, dict):
            return output, {}
        raise TypeError(f"Unexpected training_losses output type: {type(output)}")

    def _log_terms(self, terms: Dict[str, Any], prefix: str, *, batch_size: int, on_step: bool, on_epoch: bool):
        flat = flatten_dict(terms)
        for key, value in flat.items():
            if not torch.is_tensor(value) and not isinstance(value, (float, int)):
                continue
            name = f"{prefix}/{key}"
            is_dynamic_bin = any(seg.startswith("bin_") for seg in key.split("/"))
            self.log(
                name,
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=(key == "loss"),
                batch_size=batch_size,
                sync_dist=not is_dynamic_bin,
                rank_zero_only=is_dynamic_bin,
            )

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        with self._autocast_context():
            terms, _ = self._extract_terms(self.training_losses(**batch))
        x0 = batch.get("x_0")
        if hasattr(x0, "shape"):
            batch_size = int(x0.shape[0])
        elif hasattr(x0, "layout"):
            batch_size = int(len(x0.layout))
        elif isinstance(batch, dict) and batch:
            first = next(iter(batch.values()))
            batch_size = int(first.shape[0]) if hasattr(first, "shape") else 1
        else:
            batch_size = 1
        self._log_terms(terms, "train", batch_size=batch_size, on_step=True, on_epoch=True)
        return terms["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        with self._autocast_context():
            terms, _ = self._extract_terms(self.training_losses(**batch))
        x0 = batch.get("x_0")
        if hasattr(x0, "shape"):
            batch_size = int(x0.shape[0])
        elif hasattr(x0, "layout"):
            batch_size = int(len(x0.layout))
        elif isinstance(batch, dict) and batch:
            first = next(iter(batch.values()))
            batch_size = int(first.shape[0]) if hasattr(first, "shape") else 1
        else:
            batch_size = 1
        self._log_terms(terms, "val", batch_size=batch_size, on_step=False, on_epoch=True)
        return terms

    @torch.no_grad()
    def visualize_sample(self, sample):
        if self.dataset is not None and hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        return sample

    def _get_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "default_root_dir", None):
            return str(trainer.default_root_dir)
        return os.getcwd()

    @torch.no_grad()
    def snapshot_dataset(self, num_samples: int = 100, batch_size: int = 4):
        if self.dataset is None:
            return
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[: min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            vis = self.visualize_sample(data)
            if isinstance(vis, dict):
                for key, val in vis.items():
                    save_cfg.setdefault(f"dataset_{key}", []).append(val)
            else:
                save_cfg.setdefault("dataset", []).append(vis)

        out_dir = self._get_output_dir()
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        value_range = getattr(self.dataset, "value_range", (0, 1))
        for name, image in save_cfg.items():
            tv_utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(out_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix: Optional[str] = None, num_samples: int = 64, batch_size: int = 4, verbose: bool = False):
        if suffix is None:
            step = int(getattr(getattr(self, "trainer", None), "global_step", 0))
            suffix = f"step{step:07d}"

        with self._autocast_context():
            samples = self.run_snapshot(num_samples=num_samples, batch_size=batch_size, verbose=verbose)

        for key in list(samples.keys()):
            if samples[key]["type"] == "sample":
                vis = self.visualize_sample(samples[key]["value"])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f"{key}_{k}"] = {"value": v, "type": "image"}
                    del samples[key]
                else:
                    samples[key] = {"value": vis, "type": "image"}

        out_dir = self._get_output_dir()
        sample_dir = os.path.join(out_dir, "samples", suffix)
        os.makedirs(sample_dir, exist_ok=True)
        value_range = getattr(self.dataset, "value_range", (0, 1)) if self.dataset is not None else (0, 1)

        for key, payload in samples.items():
            if payload["type"] == "image":
                tv_utils.save_image(
                    payload["value"],
                    os.path.join(sample_dir, f"{key}_{suffix}.jpg"),
                    nrow=max(1, int(np.sqrt(num_samples))),
                    normalize=True,
                    value_range=value_range,
                )
            elif payload["type"] == "number":
                min_val = payload["value"].min()
                max_val = payload["value"].max()
                image = (payload["value"] - min_val) / (max_val - min_val + 1e-8)
                image = tv_utils.make_grid(image, nrow=max(1, int(np.sqrt(num_samples))), normalize=False)
                save_image_with_notes(
                    image,
                    os.path.join(sample_dir, f"{key}_{suffix}.jpg"),
                    notes=f"{key} min: {min_val}, max: {max_val}",
                )

    @abstractmethod
    def training_losses(self, **mb_data):
        pass

    @abstractmethod
    def run_snapshot(self, num_samples: int, batch_size: int = 4, verbose: bool = False) -> Dict[str, Any]:
        pass
