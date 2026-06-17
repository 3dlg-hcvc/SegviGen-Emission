"""Regression test for the every-step checkpoint bug.

Bug: `train.py` built the top-k ModelCheckpoint with `every_n_train_steps=1`,
so `last.ckpt` (multi-GB) was rewritten every training step — ~90% of wall-clock
went to Lustre I/O. Fix: refresh `last.ckpt` on a step interval
(`ckpt_last_every_n_steps`, default 200) instead of every step. Top-k validation
checkpointing is unaffected (it fires at validation-epoch end via the monitor).

CPU-only, no GPU/data needed — just constructs the callback.
Run:  python data_toolkit/test_checkpoint_cadence.py
"""
import sys

from train import build_checkpoint_callback


def test_topk_last_ckpt_defaults_to_200_steps():
    cb = build_checkpoint_callback({"save_top_k": 1}, "/tmp/ckpts")
    assert cb._every_n_train_steps == 200, f"expected last.ckpt every 200 steps, got {cb._every_n_train_steps}"


def test_not_every_step():
    """The bug was every_n_train_steps=1 — guard against regressing to it."""
    cb = build_checkpoint_callback({"save_top_k": 1}, "/tmp/ckpts")
    assert cb._every_n_train_steps != 1, "last.ckpt must NOT be saved every step"


def test_topk_settings_preserved():
    """Changing the last.ckpt cadence must not disturb top-k behavior."""
    cb = build_checkpoint_callback(
        {"save_top_k": 3, "monitor_metric": "val/loss", "monitor_mode": "min"}, "/tmp/ckpts"
    )
    assert cb.save_top_k == 3, f"save_top_k changed: {cb.save_top_k}"
    assert cb.monitor == "val/loss", f"monitor changed: {cb.monitor}"
    assert cb.mode == "min", f"mode changed: {cb.mode}"
    assert cb.save_last is True, "save_last must stay True for resume"


def test_last_ckpt_interval_configurable():
    cb = build_checkpoint_callback({"save_top_k": 1, "ckpt_last_every_n_steps": 50}, "/tmp/ckpts")
    assert cb._every_n_train_steps == 50, f"override ignored: {cb._every_n_train_steps}"


def test_legacy_isave_mode_unchanged():
    """The non-top-k branch (i_save) must keep its existing behavior."""
    cb = build_checkpoint_callback({"i_save": 2000}, "/tmp/ckpts")
    assert cb._every_n_train_steps == 2000, f"i_save interval changed: {cb._every_n_train_steps}"
    assert cb.save_top_k == -1, f"legacy save_top_k changed: {cb.save_top_k}"


def main():
    tests = [
        test_topk_last_ckpt_defaults_to_200_steps,
        test_not_every_step,
        test_topk_settings_preserved,
        test_last_ckpt_interval_configurable,
        test_legacy_isave_mode_unchanged,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
