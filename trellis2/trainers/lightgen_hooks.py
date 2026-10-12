"""LightGen's training hooks, kept out of upstream's train.py and trainers/ (REFACTOR_TODO.md, D13).

The same file as TRELLIS2's trellis2/trainers/lightgen_hooks.py (both forks of our lightgen branch), plus
two SegviGen-Emission additions: CheckpointTimingCallback's PEAK_VRAM line at the end of training, and
SnapshotCallback's disable_final_snapshot switch (both of this lane's launchers set it).

train.py calls them in five places:
  * make_wandb_logger: the WandbLogger, with the run id kept in <output_dir>/wandb_run_id.txt so a
    requeued job continues the same wandb run;
  * build_checkpoint_callback: top-k by val/loss plus last.ckpt every ckpt_last_every_n_steps, or
    upstream's step-interval mode;
  * CheckpointTimingCallback: the [ckpt-timing] lines; the snapshot watchers parse SAVE_START and
    RESUME_AFTER_SAVE;
  * TrainTraceCallback: the per-step [train] and per-validation [val] lines (formerly
    BasicTrainer.on_train_batch_end / on_validation_epoch_end). It is the last callback, as
    Lightning calls the module's own hooks after every callback's;
  * SnapshotCallback: upstream's, with disable_final_snapshot (skip only the snapshot at the end of
    training: this lane's dataset has no visualize_sample).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import torch

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from .callbacks import SnapshotCallback as _UpstreamSnapshotCallback


def make_wandb_logger(project, name, entity, output_dir):
    from lightning.pytorch.loggers import WandbLogger

    # Persist the wandb run ID across requeues so resumed jobs continue the
    # same wandb run instead of starting a fresh one each cycle. Delete
    # {output_dir}/wandb_run_id.txt to force a new run (e.g. when starting
    # a new experiment from the same output_dir).
    wandb_id_path = os.path.join(output_dir, "wandb_run_id.txt")
    wandb_id = None
    if os.path.exists(wandb_id_path):
        try:
            with open(wandb_id_path) as f:
                wandb_id = (f.read().strip() or None)
        except Exception:
            wandb_id = None

    logger = WandbLogger(
        project=project,
        name=name,
        entity=entity,
        save_dir=output_dir,
        id=wandb_id,
        resume="allow" if wandb_id else None,
    )

    # First-run only (rank 0): persist the auto-generated run ID for later
    # requeues. Accessing `logger.experiment.id` triggers wandb.init() so
    # the ID is available; non-rank-0 instances of WandbLogger return a
    # dummy experiment whose .id is None, which we ignore.
    if wandb_id is None and int(os.environ.get("SLURM_PROCID", "0")) == 0:
        try:
            new_id = logger.experiment.id
            if new_id:
                os.makedirs(output_dir, exist_ok=True)
                with open(wandb_id_path, "w") as f:
                    f.write(new_id)
        except Exception:
            pass
    return logger


def build_checkpoint_callback(trainer_args: Dict[str, Any], ckpt_dir: str) -> ModelCheckpoint:
    """Construct the run's ModelCheckpoint callback.

    Two modes:
      * Top-k val mode (``save_top_k`` set): keep the K best checkpoints by
        ``monitor_metric`` (written at validation-epoch end) plus a ``last.ckpt``
        for resume. ``last.ckpt`` is refreshed every ``ckpt_last_every_n_steps``
        steps (default 200) so a hard crash loses at most that many steps; on a
        graceful (SIGTERM) requeue the trainer's signal handler still saves exact
        state. NOTE: never set this to 1 — rewriting the multi-GB checkpoint every
        step blocks training on a Lustre write each step (~90% of wall-clock; see
        CheckpointTimingCallback). The top-k save itself is independent of this
        interval — it fires at validation-epoch end via the monitor metric.
      * Legacy step-interval mode (``i_save`` set, no ``save_top_k``): keep all
        checkpoints at the ``i_save`` step interval.
    """
    i_save = trainer_args.get("i_save")
    save_top_k_cfg = trainer_args.get("save_top_k")
    if save_top_k_cfg is not None:
        monitor_metric = str(trainer_args.get("monitor_metric", "val/loss"))
        monitor_mode = str(trainer_args.get("monitor_mode", "min")).lower()
        if monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")
        filename = "step{step:07d}-{" + monitor_metric + ":.4f}"
        last_every_n_steps = int(trainer_args.get("ckpt_last_every_n_steps", 200))
        return ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=filename,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=int(save_top_k_cfg),
            save_last=True,
            every_n_train_steps=last_every_n_steps,
            auto_insert_metric_name=False,
        )
    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:07d}",
        every_n_train_steps=(int(i_save) if i_save is not None else None),
        save_last=True,
        save_top_k=-1,
    )


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


class TrainTraceCallback(pl.Callback):
    """The [train] line per logged step and the [val] line per validation, rank 0 only."""

    @staticmethod
    def _global_zero(trainer, pl_module) -> bool:
        is_zero = getattr(pl_module, "_is_global_zero", None)
        return bool(is_zero()) if callable(is_zero) else bool(getattr(trainer, "is_global_zero", True))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._global_zero(trainer, pl_module):
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        val_items = []
        for key in sorted(metrics.keys()):
            if not str(key).startswith("val/"):
                continue
            value = metrics[key]
            try:
                value = float(value.item()) if torch.is_tensor(value) else float(value)
            except Exception:
                continue
            val_items.append((str(key), value))
        if not val_items:
            return
        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", 0))
        formatted = "  ".join(f"{k}={v:.4f}" for k, v in val_items)
        print(f"[val] step={step}  epoch={epoch}  {formatted}", flush=True)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        # Per-step text trace symmetric to [val]. Cadence follows
        # Trainer.log_every_n_steps (= i_log) so it stays aligned with the
        # wandb / CSV scalar flush — and survives `tee` piping that mangles
        # Lightning's TTY-only progress bar.
        if not self._global_zero(trainer, pl_module):
            return
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        log_every = int(getattr(trainer, "log_every_n_steps", 1) or 1)
        if step % log_every != 0:
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        train_items = []
        for key in sorted(metrics.keys()):
            if not str(key).startswith("train/"):
                continue
            # Skip the on_epoch aggregates — they only update at epoch boundary
            # and would be redundant with the per-step *_step values.
            if str(key).endswith("_epoch"):
                continue
            value = metrics[key]
            try:
                value = float(value.item()) if torch.is_tensor(value) else float(value)
            except Exception:
                continue
            train_items.append((str(key), value))
        if not train_items:
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        formatted = "  ".join(f"{k}={v:.4f}" for k, v in train_items)
        print(f"[train] step={step}  epoch={epoch}  {formatted}", flush=True)


class SnapshotCallback(_UpstreamSnapshotCallback):
    """Upstream's SnapshotCallback with disable_final_snapshot."""

    def __init__(self, *args, disable_final_snapshot: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Escape hatch for datasets that don't implement visualize_sample yet
        # (e.g. SegviGenEmissive256): on_train_end's snapshot is otherwise
        # unconditional and will try to save_image() raw SparseTensor /
        # string batch fields. Fixing that needs a real visualize_sample on
        # the dataset (follow-up); until then, set this true to skip only
        # the final snapshot while keeping everything else (mid-training
        # snapshots via sample_every_n_steps, checkpoints) intact.
        self.disable_final_snapshot = bool(disable_final_snapshot)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.disable_final_snapshot:
            return
        super().on_train_end(trainer, pl_module)
