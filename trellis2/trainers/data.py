from __future__ import annotations

from typing import Any, Dict, Optional
import copy
import functools

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .builders import build_dataset
from trellis2.utils.data_utils import BalancedResumableSampler, ResumableSampler


def _copy_dataset_args(dataset):
    if hasattr(dataset, "_init_kwargs"):
        return dict(dataset._init_kwargs)
    if hasattr(dataset, "_init_args"):
        return dict(dataset._init_args)
    return None


class TrellisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        dataset: Dict[str, Any],
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        val_split_file: Optional[str] = None,
        val_split_name: Optional[str] = None,
        val_batch_size: Optional[int] = None,
        use_balanced_sampler: Optional[bool] = None,
        split_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_cfg = copy.deepcopy(dataset)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.val_split_file = val_split_file
        self.val_split_name = val_split_name
        self.val_batch_size = int(val_batch_size) if val_batch_size is not None else None
        self.use_balanced_sampler = use_balanced_sampler
        self.split_size = int(split_size) if split_size is not None else None
        self.shuffle = bool(shuffle)
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = build_dataset(self.dataset_cfg)

        if self.val_dataset is None and (self.val_split_file or self.val_split_name):
            base_kwargs = _copy_dataset_args(self.train_dataset)
            if base_kwargs is None:
                base_kwargs = dict(self.dataset_cfg.get("args", {}))
            if self.val_split_file:
                base_kwargs["split_file"] = self.val_split_file
            if self.val_split_name:
                base_kwargs["split"] = self.val_split_name
            val_cfg = {
                "name": self.dataset_cfg.get("name"),
                "args": base_kwargs,
            }
            self.val_dataset = build_dataset(val_cfg)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        collate = None
        if hasattr(self.train_dataset, "collate_fn"):
            collate = self.train_dataset.collate_fn
            if self.split_size is not None:
                collate = functools.partial(collate, split_size=self.split_size)

        sampler = None
        use_balanced = self.use_balanced_sampler
        if use_balanced is None:
            use_balanced = hasattr(self.train_dataset, "loads")
        if use_balanced:
            try:
                sampler = BalancedResumableSampler(
                    self.train_dataset,
                    shuffle=self.shuffle,
                    batch_size=self.batch_size,
                )
            except AssertionError:
                sampler = ResumableSampler(self.train_dataset, shuffle=self.shuffle, drop_last=True)
        elif self.shuffle:
            sampler = ResumableSampler(self.train_dataset, shuffle=True, drop_last=True)
        self.train_sampler = sampler

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None and self.shuffle),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
            collate_fn=collate,
            sampler=sampler,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        collate = None
        if hasattr(self.val_dataset, "collate_fn"):
            collate = self.val_dataset.collate_fn
        val_batch_size = self.val_batch_size or self.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=max(0, self.num_workers // 2),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
            collate_fn=collate,
        )
