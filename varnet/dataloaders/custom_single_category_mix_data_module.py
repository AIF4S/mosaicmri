"""
Data Module for mixing a single anatomical category from dataset1 with all data from dataset2.

Derived from `custom_mix_data_module.py` and `custom_category_data_module_clean.py`:
- Dataset1 is filtered to ONLY include files whose category matches `category_name` (case-insensitive)
  according to a JSON mapping file (`category_mapping_path`).
- Dataset2 is loaded normally using `SliceDataset`.
- The two datasets are concatenated (optionally with randomized order controlled by `mix_seed`).

Directory expectations (same as other modules):
  data_path1/{challenge}_train, {challenge}_val, {challenge}_test
  data_path2/{challenge}_train, {challenge}_val, {challenge}_test

JSON mapping expectation:
  { "relative_or_absolute_filename.h5": "CATEGORY" , ... }
If a filename in the mapping is relative, it is resolved against the partition root
(e.g. data_path1/multicoil_train/<relative_filename>). Absolute paths are used as-is.

"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union, List, Dict, Any
import json
import random

import pytorch_lightning as pl
import torch

import fastmri
from fastmri.data import SliceDataset

# Reuse ListSliceDataset utility
from .custom_category_data_module_clean import ListSliceDataset


def _check_both_not_none(val1, val2):
    return (val1 is not None) and (val2 is not None)


def worker_init_fn(worker_id):  # mirror logic for consistent mask_func seeding
    worker_info = torch.utils.data.get_worker_info()
    data = worker_info.dataset  # could be ConcatDataset or SliceDataset

    is_ddp = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_ddp = True

    base_seed = worker_info.seed

    # torch.utils.data.ConcatDataset exposes .datasets list
    if hasattr(data, "datasets"):
        for i, dataset in enumerate(data.datasets):
            if hasattr(dataset, "transform") and dataset.transform is not None and hasattr(dataset.transform, "mask_func") and dataset.transform.mask_func is not None:
                if is_ddp:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank() * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif hasattr(data, "transform") and data.transform is not None and hasattr(data.transform, "mask_func") and data.transform.mask_func is not None:
        if is_ddp:
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


class SingleCategoryMixDataModule(pl.LightningDataModule):
    """
    Mix a SINGLE anatomical category (from dataset1) with full dataset2.

    Args:
        data_path1: Root of first dataset (contains category-mapped files).
        data_path2: Root of second dataset.
        category_mapping_path: JSON mapping file (filename -> category string).
        category_name: Single category to include from dataset1 (case-insensitive).
        challenge: fastMRI challenge string ('multicoil' or 'singlecoil').
        train_transform / val_transform / test_transform: Transform callables.
        combine_train_val: If True, train split concatenates train+val (for BOTH datasets, filtering category on both partitions of dataset1).
        test_split: 'val' | 'test' | 'challenge'.
        test_path1/test_path2: Optional override paths for test partition.
        sample_rate / val_sample_rate / test_sample_rate: Slice-level subsampling fractions.
        volume_sample_rate / val_volume_sample_rate / test_volume_sample_rate: Volume-level subsampling fractions.
        train_filter / val_filter / test_filter: Optional callables filtering raw samples (applied after category filtering for dataset1; passed through to SliceDataset for dataset2 via raw_sample_filter when building intermediate sets).
        use_dataset_cache_file: Cache metadata.
        batch_size, num_workers: DataLoader params.
        distributed_sampler: Use DistributedSampler for train and VolumeSampler for eval.
        mix_seed: Seed controlling ordering of the two component datasets in concatenation.
    """

    def __init__(
        self,
        data_path1: Path,
        data_path2: Path,
        category_mapping_path: Path,
        category_name: str,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
    val_source: str = "mixed",
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path1: Optional[Path] = None,
        test_path2: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        mix_seed: int = 42,
    ):
        super().__init__()

        # validation of mutually exclusive sampling flags
        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Set either sample_rate or volume_sample_rate, not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError("Set either val_sample_rate or val_volume_sample_rate, not both.")
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError("Set either test_sample_rate or test_volume_sample_rate, not both.")

        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.category_mapping_path = category_mapping_path
        self.category_name = category_name.upper()
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.val_source = val_source
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path1 = test_path1
        self.test_path2 = test_path2
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.mix_seed = mix_seed

        allowed_val_sources = {"mixed", "data1", "data2"}
        if self.val_source not in allowed_val_sources:
            raise ValueError(
                f"val_source must be one of {sorted(allowed_val_sources)}, got {self.val_source!r}"
            )

    # ---------------- internal helpers -----------------
    def _collect_category_files(self, root: Path) -> List[Path]:
        """Return list of existing file paths at partition root matching category_name."""
        with open(self.category_mapping_path, "r") as f:
            mapping = json.load(f)
        files: List[Path] = []
        target = self.category_name
        for fname, cat in mapping.items():
            if str(cat).upper() != target:
                continue
            raw_path = Path(fname)
            file_path = raw_path if raw_path.is_absolute() else (root / raw_path)
            if file_path.is_file():
                files.append(file_path)
        return files

    def _build_dataset1(self, root: Path, transform: Callable, sample_rate: Optional[float], volume_sample_rate: Optional[float]) -> ListSliceDataset:
        files = self._collect_category_files(root)
        if not files:
            print(f"[SingleCategoryMix] No files found for category {self.category_name} at {root}.")
        ds = ListSliceDataset(
            fnames=files,
            challenge=self.challenge,
            transform=transform,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            use_dataset_cache=self.use_dataset_cache_file,
            dataset_cache_file="dataset_cache.pkl",
        )
        return ds

    def _build_dataset2(self, root: Path, transform: Callable, sample_rate: Optional[float], volume_sample_rate: Optional[float], raw_sample_filter: Optional[Callable]) -> SliceDataset:
        ds = SliceDataset(
            root=root,
            transform=transform,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            challenge=self.challenge,
            use_dataset_cache=self.use_dataset_cache_file,
            raw_sample_filter=raw_sample_filter,
        )
        return ds

    def _create_data_loader(self, data_transform: Callable, data_partition: str, sample_rate: Optional[float] = None, volume_sample_rate: Optional[float] = None) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = self.val_sample_rate if sample_rate is None else sample_rate
                volume_sample_rate = self.val_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = self.test_sample_rate if sample_rate is None else sample_rate
                volume_sample_rate = self.test_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
                raw_sample_filter = self.test_filter
            else:
                raw_sample_filter = None

        # resolve partition roots
        if data_partition in ("test", "challenge"):
            root1 = self.test_path1 if self.test_path1 is not None else (self.data_path1 / f"{self.challenge}_{data_partition}")
            root2 = self.test_path2 if self.test_path2 is not None else (self.data_path2 / f"{self.challenge}_{data_partition}")
        else:
            root1 = self.data_path1 / f"{self.challenge}_{data_partition}"
            root2 = self.data_path2 / f"{self.challenge}_{data_partition}"

        datasets: List[torch.utils.data.Dataset] = []

        if is_train and self.combine_train_val:
            # Build category dataset from train+val partitions of dataset1
            train_root1 = self.data_path1 / f"{self.challenge}_train"
            val_root1 = self.data_path1 / f"{self.challenge}_val"
            ds1_train = self._build_dataset1(train_root1, data_transform, sample_rate, volume_sample_rate)
            ds1_val = self._build_dataset1(val_root1, data_transform, sample_rate, volume_sample_rate)
            ds2_train = self._build_dataset2(self.data_path2 / f"{self.challenge}_train", data_transform, sample_rate, volume_sample_rate, raw_sample_filter) # fixed 50% sampling for FastMRI
            ds2_val = self._build_dataset2(self.data_path2 / f"{self.challenge}_val", data_transform, sample_rate, volume_sample_rate, raw_sample_filter) # fixed 50% sampling for FastMRI
            datasets.extend([ds1_train, ds1_val, ds2_train, ds2_val])
        else:
            ds1 = self._build_dataset1(root1, data_transform, sample_rate, volume_sample_rate)
            ds2 = self._build_dataset2(root2, data_transform, sample_rate, volume_sample_rate, raw_sample_filter)
            datasets.extend([ds1, ds2])

        # Randomize dataset order deterministically
        rng = random.Random(self.mix_seed)
        rng.shuffle(datasets)

        concat_dataset = torch.utils.data.ConcatDataset(datasets)

        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(concat_dataset, shuffle=True)
            else:
                # VolumeSampler requires slice-level dataset with volume boundaries; ConcatDataset loses explicit interface.
                # Fallback: no volume-aware sampler, keep deterministic ordering.
                print("[SingleCategoryMix] Warning: VolumeSampler not applied for eval with ConcatDataset.")

        dataloader = torch.utils.data.DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
        )
        return dataloader

    # --------------- Lightning hooks ---------------
    def prepare_data(self):
        if not self.use_dataset_cache_file:
            return
        # Touch cache by instantiating minimal datasets for each partition
        partitions = ["train", "val", self.test_split]
        for part in partitions:
            if part in ("test", "challenge"):
                root1 = self.test_path1 if self.test_path1 is not None else (self.data_path1 / f"{self.challenge}_{part}")
                root2 = self.test_path2 if self.test_path2 is not None else (self.data_path2 / f"{self.challenge}_{part}")
            else:
                root1 = self.data_path1 / f"{self.challenge}_{part}"
                root2 = self.data_path2 / f"{self.challenge}_{part}"
            _ = self._build_dataset1(root1, self.train_transform, self.sample_rate, self.volume_sample_rate)
            _ = self._build_dataset2(root2, self.train_transform, self.sample_rate, self.volume_sample_rate, self.train_filter)

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        if self.val_source == "mixed":
            return self._create_data_loader(self.val_transform, data_partition="val")

        # Build a single-source validation set (either dataset1 category-only or dataset2 full).
        sample_rate = self.val_sample_rate
        volume_sample_rate = self.val_volume_sample_rate
        raw_sample_filter = self.val_filter

        root1 = self.data_path1 / f"{self.challenge}_val"
        root2 = self.data_path2 / f"{self.challenge}_val"

        if self.val_source == "data1":
            ds = self._build_dataset1(root1, self.val_transform, sample_rate, volume_sample_rate)
        else:  # "data2"
            ds = self._build_dataset2(root2, self.val_transform, sample_rate, volume_sample_rate, raw_sample_filter)

        sampler = None
        if self.distributed_sampler:
            # Keep things simple and deterministic for eval; DistributedSampler is fine.
            sampler = torch.utils.data.DistributedSampler(ds, shuffle=False)

        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition=self.test_split)

    # --------------- Utility ---------------
    def get_category_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"category": self.category_name, "files": 0}
        root = self.data_path1 / f"{self.challenge}_train"
        stats["files"] = len(self._collect_category_files(root))
        return stats

    # --------------- CLI args ---------------
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path1", type=Path, required=True, help="Path to first dataset root")
        parser.add_argument("--data_path2", type=Path, required=True, help="Path to second dataset root")
        parser.add_argument("--category_mapping_path", type=Path, required=True, help="JSON mapping of filename -> category")
        parser.add_argument("--category_name", type=str, required=True, help="Single category name to include from dataset1")
        parser.add_argument("--challenge", choices=("singlecoil", "multicoil"), default="singlecoil")
        parser.add_argument(
            "--val_source",
            choices=("mixed", "data1", "data2"),
            default="mixed",
            help="Which dataset to use for validation: mixed concat, dataset1-only (category-filtered), or dataset2-only.",
        )
        parser.add_argument("--test_split", choices=("val", "test", "challenge"), default="test")
        parser.add_argument("--test_path1", type=Path, default=None)
        parser.add_argument("--test_path2", type=Path, default=None)
        parser.add_argument("--sample_rate", type=float, default=None)
        parser.add_argument("--val_sample_rate", type=float, default=None)
        parser.add_argument("--test_sample_rate", type=float, default=None)
        parser.add_argument("--volume_sample_rate", type=float, default=None)
        parser.add_argument("--val_volume_sample_rate", type=float, default=None)
        parser.add_argument("--test_volume_sample_rate", type=float, default=None)
        parser.add_argument("--use_dataset_cache_file", type=bool, default=True)
        parser.add_argument("--combine_train_val", type=bool, default=False)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--distributed_sampler", type=bool, default=False)
        parser.add_argument("--mix_seed", type=int, default=42)
        return parser
