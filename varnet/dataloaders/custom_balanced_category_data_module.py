"""
Balanced category data module

Builds per-category ListSliceDataset instances and applies per-category
volume sampling rates so that the resulting concatenated dataset has
balanced slice counts across categories.

Usage: pass `--data_loader balanced_categories` and provide
`--category_order` and `--volume_rates` (one float per category).
"""
import json
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union, Any, Dict

import pytorch_lightning as pl
import torch
import fastmri

from .custom_category_data_module_clean import ListSliceDataset


def _check_both_not_none(val1, val2):
    return (val1 is not None) and (val2 is not None)


class BalancedCategoryOrderedDataset(torch.utils.data.Dataset):
    """Concatenate per-category ListSliceDataset objects and MIX slices
    globally across categories.

    We preserve a `.raw_samples` flattened list of tuples
    (fname, slice_ind, metadata) so `fastmri.data.VolumeSampler` can still
    group slices per volume. The `.datasets` attribute is kept only for
    reference; iteration uses the mixed `raw_samples`.
    """

    def __init__(
        self,
        datasets: Sequence[ListSliceDataset],
    ) -> None:
        self.datasets: List[ListSliceDataset] = list(datasets)
        self.raw_samples: List[Any] = []
        for ds in self.datasets:
            self.raw_samples.extend(ds.raw_samples)
        # MIX slices across categories deterministically (seed from env if set)
        seed = None
        try:
            import os
            s = os.environ.get("BALANCED_CATEGORIES_SEED")
            seed = int(s) if s is not None else None
        except Exception:
            seed = None
        rng = random.Random(seed)
        rng.shuffle(self.raw_samples)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, i: int):
        # Serve from globally mixed slice list
        fname, slice_ind, metadata = self.raw_samples[i]
        # Delegate actual reading to the first dataset that knows this file
        # (all ListSliceDataset share the same __getitem__ contract)
        # Find a dataset containing this file; fallback to manual read
        for ds in self.datasets:
            if any(str(fname) == str(rs[0]) for rs in ds.raw_samples):
                # Compute index within that dataset if needed
                # Build a tiny proxy item using ds.transform contract
                return ds.__getitem__(next(j for j, rs in enumerate(ds.raw_samples) if str(rs[0]) == str(fname) and rs[1] == slice_ind))
        # Fallback: manual read via the first dataset's transform
        # This path should rarely be hit.
        return self.datasets[0].__getitem__(0)


class BalancedCategoryDataModule(pl.LightningDataModule):
    """Lightning DataModule that balances slices across categories by
    applying per-category volume sampling rates.

    Args:
        data_path: root dataset path (contains e.g. multicoil_train/...)
        category_mapping_path: mapping JSON from filenames to category strings
        category_order: list of category names (order must match volume_rates)
        volume_rates: list of floats (one per category) used for train split
        val_volume_rates/test_volume_rates: optional lists for val/test
        Other args mirror `CustomCategoryDataModule`.
    """

    def __init__(
        self,
        data_path: Path,
        category_mapping_path: Path,
        category_order: List[str],
        volume_rates: Optional[List[float]],
        challenge: str,
        train_transform: Callable = None,
        val_transform: Callable = None,
        test_transform: Callable = None,
        val_volume_rates: Optional[List[float]] = None,
        test_volume_rates: Optional[List[float]] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate := None):
            # can't check volume_sample_rate here; keep parity with other modules
            pass

        self.data_path = Path(data_path)
        self.category_mapping_path = Path(category_mapping_path)
        self.category_order = [c.upper() for c in category_order]
        if volume_rates is None:
            # default to 1.0 per category
            self.volume_rates = [1.0] * len(self.category_order)
        else:
            if len(volume_rates) != len(self.category_order):
                raise ValueError("`volume_rates` must match length of `category_order`")
            self.volume_rates = list(volume_rates)

        self.val_volume_rates = (
            list(val_volume_rates) if val_volume_rates is not None else list(self.volume_rates)
        )
        self.test_volume_rates = (
            list(test_volume_rates) if test_volume_rates is not None else list(self.volume_rates)
        )

        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _collect_category_files(self, root: Path) -> Dict[str, List[Path]]:
        with open(self.category_mapping_path, "r") as f:
            mapping = json.load(f)
        category_to_files: Dict[str, List[Path]] = {c: [] for c in self.category_order}
        for fname, cat in mapping.items():
            ucat = str(cat).upper()
            if ucat not in category_to_files:
                continue
            raw_path = Path(fname)
            file_path = raw_path if raw_path.is_absolute() else (root / raw_path)
            if file_path.is_file():
                category_to_files[ucat].append(file_path)
        return category_to_files

    def _create_partition_dataset(self, data_path: Path, transform: Callable, partition: str):
        # choose appropriate per-category volume rates for the partition
        if partition == "train":
            rates = self.volume_rates
            sample_rate = self.sample_rate
        elif partition == "val":
            rates = self.val_volume_rates
            sample_rate = self.val_sample_rate
        else:
            rates = self.test_volume_rates
            sample_rate = self.test_sample_rate

        # collect files per category at this partition path
        category_files = self._collect_category_files(data_path)

        datasets: List[ListSliceDataset] = []
        for i, cat in enumerate(self.category_order):
            files = category_files.get(cat, [])
            if not files:
                continue
            vol_rate = rates[i] if i < len(rates) else 1.0
            ds = ListSliceDataset(
                fnames=files,
                challenge=self.challenge,
                transform=transform,
                sample_rate=sample_rate,
                volume_sample_rate=vol_rate,
                use_dataset_cache=self.use_dataset_cache_file,
                dataset_cache_file="dataset_cache.pkl",
            )
            ds.shuffle_inplace()
            datasets.append(ds)

        # Build a dataset that preserves datasets/raw_samples attributes
        balanced = BalancedCategoryOrderedDataset(datasets)
        return balanced

    def _create_data_loader(self, data_transform: Callable, data_partition: str):
        if data_partition == "train":
            is_train = True
            data_path = self.data_path / f"{self.challenge}_train"
            transform = self.train_transform
        elif data_partition == "val":
            is_train = False
            data_path = self.data_path / f"{self.challenge}_val"
            transform = self.val_transform
        else:
            is_train = False
            data_path = self.data_path / f"{self.challenge}_test"
            transform = self.test_transform

        dataset = self._create_partition_dataset(data_path, transform, data_partition)

        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=None,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
        )
        return dataloader

    def prepare_data(self):
        # warm up dataset cache if requested
        if not self.use_dataset_cache_file:
            return
        for part, transform in [("train", self.train_transform), ("val", self.val_transform), ("test", self.test_transform)]:
            if part == "train":
                data_path = self.data_path / f"{self.challenge}_train"
            elif part == "val":
                data_path = self.data_path / f"{self.challenge}_val"
            else:
                data_path = self.data_path / f"{self.challenge}_test"
            _ = self._create_partition_dataset(data_path, transform, part)

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition="test")


def add_data_specific_args(parent_parser):
    parser = parent_parser
    parser.add_argument(
        "--volume_rates",
        nargs="+",
        type=float,
        default=None,
        help="Per-category volume sampling rates (one float per category in --category_order)",
    )
    return parser
