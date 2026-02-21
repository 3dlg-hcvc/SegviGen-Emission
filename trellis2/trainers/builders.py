from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import importlib


def _try_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


def resolve_class(name: str, module_paths: Iterable[str]):
    if "." in name:
        module_path, cls_name = name.rsplit(".", 1)
        module = _try_import(module_path)
        if module is None:
            raise ImportError(f"Failed to import module '{module_path}' for '{name}'.")
        if not hasattr(module, cls_name):
            raise AttributeError(f"Module '{module_path}' has no attribute '{cls_name}'.")
        return getattr(module, cls_name)

    for module_path in module_paths:
        module = _try_import(module_path)
        if module is None:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    paths = ", ".join(module_paths)
    raise ValueError(f"Unable to resolve class '{name}' in modules: {paths}.")


def build_from_config(cfg: Dict[str, Any], module_paths: Iterable[str]):
    if cfg is None:
        raise ValueError("Config is None")
    name = cfg.get("name")
    if not name:
        raise ValueError("Config missing 'name'")
    args = cfg.get("args", {})
    cls = resolve_class(name, module_paths)
    return cls(**args)


def build_models(models_cfg: Dict[str, Dict[str, Any]], module_paths: Optional[Iterable[str]] = None):
    if module_paths is None:
        module_paths = (
            "trellis2.models",
            "trellis2.modules",
        )
    models = {}
    for key, cfg in (models_cfg or {}).items():
        models[key] = build_from_config(cfg, module_paths)
    return models


def build_dataset(dataset_cfg: Dict[str, Any]):
    module_paths = ("trellis2.datasets",)
    return build_from_config(dataset_cfg, module_paths)


def build_optimizer(cfg: Dict[str, Any], params):
    if cfg is None:
        raise ValueError("Optimizer config is None")
    name = cfg.get("name", "AdamW")
    args = cfg.get("args", {})

    torch_optim = importlib.import_module("torch.optim")
    if hasattr(torch_optim, name):
        return getattr(torch_optim, name)(params, **args)

    custom_module = importlib.import_module("trellis2.trainers.utils")
    if hasattr(custom_module, name):
        return getattr(custom_module, name)(params, **args)

    raise ValueError(f"Unknown optimizer: {name}")


def build_lr_scheduler(cfg: Optional[Dict[str, Any]], optimizer):
    if not cfg:
        return None
    name = cfg.get("name")
    if not name:
        raise ValueError("lr_scheduler config missing 'name'")
    args = cfg.get("args", {})

    torch_sched = importlib.import_module("torch.optim.lr_scheduler")
    if hasattr(torch_sched, name):
        return getattr(torch_sched, name)(optimizer, **args)

    custom_module = importlib.import_module("trellis2.trainers.utils")
    if hasattr(custom_module, name):
        return getattr(custom_module, name)(optimizer, **args)

    raise ValueError(f"Unknown lr scheduler: {name}")
