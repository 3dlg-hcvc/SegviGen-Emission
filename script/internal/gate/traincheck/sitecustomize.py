"""Training-check hooks, loaded through PYTHONPATH by `train_check.sh`; the code under test never imports them.

With TRAINCHECK_RECORD set: seeds the process at start (before train.py builds the models) and again
right before Trainer.fit, turns on deterministic mode, and adds a callback that writes one JSON object
per line to $TRAINCHECK_RECORD:
  fit_start  global step and the weight sum the run starts from
  step       per training step: global step, epoch, batch sha256 ids, lr, loss, gradient norm, weight sum
  val        per validation: the val/* metrics and the EMA sum
The saved checkpoint files are summarized afterwards by `summarize_ckpt.py`.
"""
import json
import os
import random

# The coverage run (step 9) traces through this hook too (see ../covsite/sitecustomize.py).
if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()

REC = os.environ.get("TRAINCHECK_RECORD")
if REC:
    import numpy as np
    import torch

    SEED = int(os.environ.get("TRAINCHECK_SEED", "42"))

    def _seed():
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    _seed()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # flash-attn's backward accumulates dq in a nondeterministic order unless asked not to; trellis2
    # calls these through the module attribute at call time, so wrapping them here reaches every call.
    try:
        import functools

        import flash_attn

        def _det(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                kwargs.setdefault("deterministic", True)
                return fn(*args, **kwargs)
            return wrapper

        for _name in ("flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_qkvpacked_func",
                      "flash_attn_varlen_func", "flash_attn_varlen_kvpacked_func",
                      "flash_attn_varlen_qkvpacked_func"):
            if hasattr(flash_attn, _name):
                setattr(flash_attn, _name, _det(getattr(flash_attn, _name)))
    except ImportError:
        pass

    import lightning.pytorch as pl

    def _emit(obj):
        with open(REC, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def _tsum(tensors):
        s = 0.0
        for t in tensors:
            if torch.is_tensor(t) and t.is_floating_point():
                s += float(t.detach().double().sum())
        return s

    def _ids(batch):
        if isinstance(batch, dict):
            v = batch.get("sha256")
            return list(v) if isinstance(v, (list, tuple)) else v
        if isinstance(batch, (list, tuple)):
            out = []
            for b in batch:
                out += _ids(b) or []
            return out
        return None

    def _ema_sum(trainer):
        for cb in trainer.callbacks:
            state = getattr(cb, "_ema_state", None)
            if state:
                return {k: _tsum(v.values()) for k, v in state.items()}
        return None

    class TrainCheckRecorder(pl.Callback):
        def __init__(self):
            self._gn = None

        def on_fit_start(self, trainer, pl_module):
            _emit(dict(kind="fit_start", step=int(trainer.global_step), weight_sum=_tsum(pl_module.parameters())))

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            sq = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().double().pow(2).sum())
            self._gn = sq ** 0.5

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            opt = trainer.optimizers[0]
            _emit(dict(kind="step", step=int(trainer.global_step), epoch=int(trainer.current_epoch),
                       ids=_ids(batch), lr=float(opt.param_groups[0]["lr"]),
                       loss=float(loss.detach()) if torch.is_tensor(loss) else loss,
                       grad_norm=self._gn, weight_sum=_tsum(pl_module.parameters())))
            self._gn = None

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = {str(k): float(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("val/")}
            _emit(dict(kind="val", step=int(trainer.global_step), epoch=int(trainer.current_epoch),
                       metrics=metrics, ema_sum=_ema_sum(trainer)))

    _orig_fit = pl.Trainer.fit

    def _fit(self, *args, **kwargs):
        self.callbacks.append(TrainCheckRecorder())
        _seed()
        pl.seed_everything(SEED, workers=True)
        return _orig_fit(self, *args, **kwargs)

    pl.Trainer.fit = _fit
