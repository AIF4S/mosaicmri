"""
Custom Data Module for filtering MRI datasets by anatomical categories.

This module extends the FastMRI data module to support filtering and ordering
data based on anatomical categories (e.g., SPINE, KNEE, HIP, etc.) from a 
category mapping JSON file.
"""

import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union, Any, Dict

import pytorch_lightning as pl
import torch

import fastmri
from fastmri.data import SliceDataset
import h5py
import numpy as np
import xml.etree.ElementTree as etree
import pickle


def _et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """Internal copy of ElementTree query (avoid extra import path)."""
    s = "."
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}
    for el in qlist:
        s = s + f"//{prefix}:{el}"
    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")
    return str(value.text)


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, "CategoryOrderedSliceDataset"
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if hasattr(data, 'datasets') and hasattr(data.datasets[0], 'transform'):
        # Handle CategoryOrderedSliceDataset
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
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
    elif data.transform.mask_func is not None:
        # Handle regular SliceDataset
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True
    return False


class ListSliceDataset(torch.utils.data.Dataset):
    """Custom SliceDataset variant that takes an explicit list of file paths.

    Mirrors core logic from fastmri SliceDataset but does not walk a root dir.
    Allows upstream code to pre-filter filenames (e.g., by anatomical category).
    """

    def __init__(
        self,
        fnames: Sequence[Union[str, Path]],
        challenge: str,
        transform: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path] = "dataset_cache.pkl",
    ):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.dataset_cache_file = Path(dataset_cache_file)
        # Normalize paths
        self.fnames = [Path(f) for f in fnames]
        self.raw_samples: List[Any] = []  # will store tuples identical to SliceDataset

        # defaults
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # Cache loading (keyed by tuple of filenames)
        cache_key = tuple(sorted(str(p) for p in self.fnames))
        if self.dataset_cache_file.exists() and use_dataset_cache:
            try:
                with open(self.dataset_cache_file, "rb") as f:
                    dataset_cache: Dict[str, Any] = pickle.load(f)
            except (EOFError, pickle.UnpicklingError, ValueError, AttributeError):
                # Cache file can be corrupted when multiple jobs touch the same
                # path; recover by rebuilding metadata cache.
                dataset_cache = {}
        else:
            dataset_cache = {}

        if dataset_cache.get(cache_key) is None or not use_dataset_cache:
            for fname in sorted(self.fnames):
                metadata, num_slices = self._retrieve_metadata(fname)
                for slice_ind in range(num_slices):
                    self.raw_samples.append((fname, slice_ind, metadata))
            if dataset_cache.get(cache_key) is None and use_dataset_cache:
                dataset_cache[cache_key] = self.raw_samples
                tmp_cache = self.dataset_cache_file.with_name(
                    f"{self.dataset_cache_file.name}.{os.getpid()}.tmp"
                )
                with open(tmp_cache, "wb") as f:
                    pickle.dump(dataset_cache, f)
                os.replace(tmp_cache, self.dataset_cache_file)
        else:
            self.raw_samples = dataset_cache[cache_key]

        # Subsampling
        if sample_rate < 1.0:
            random.shuffle(self.raw_samples)
            keep = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:keep]
        elif volume_sample_rate < 1.0:
            vol_names = sorted(list({f[0].stem for f in self.raw_samples}))
            random.shuffle(vol_names)
            keep_vols = vol_names[: round(len(vol_names) * volume_sample_rate)]
            self.raw_samples = [rs for rs in self.raw_samples if rs[0].stem in keep_vols]

    def _retrieve_metadata(self, fname: Path):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(_et_query(et_root, enc + ["x"])),
                int(_et_query(et_root, enc + ["y"])),
                int(_et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(_et_query(et_root, rec + ["x"])),
                int(_et_query(et_root, rec + ["y"])),
                int(_et_query(et_root, rec + ["z"])),
            )
            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(_et_query(et_root, lims + ["center"]))
            enc_limits_max = int(_et_query(et_root, lims + ["maximum"])) + 1
            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max
            num_slices = hf["kspace"].shape[0]
            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }
        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, slice_ind, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][slice_ind]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][slice_ind] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)
        if self.transform is None:
            return (kspace, mask, target, attrs, fname.name, slice_ind)
        return self.transform(kspace, mask, target, attrs, fname.name, slice_ind)

    def shuffle_inplace(self):
        random.shuffle(self.raw_samples)


class CategoryOrderedSliceDataset(torch.utils.data.Dataset):
    """Build concatenated dataset ordered by anatomical categories.

    For each category:
      1. Collect file paths from mapping JSON.
      2. Instantiate ListSliceDataset with those paths.
      3. Shuffle slices within the category only.
      4. Append to overall ordered list (no cross-category mixing).
    """

    def __init__(
        self,
        root: Path,
        category_mapping_path: Path,
        category_order: List[str],
        challenge: str,
        transform: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path] = "dataset_cache.pkl",
    ):
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        with open(category_mapping_path, "r") as f:
            mapping = json.load(f)

        # Normalize category order to uppercase for matching
        order_upper = [c.upper() for c in category_order]

        # Build list of EXISTING file paths for each category preserving order
        category_to_files: Dict[str, List[Path]] = {c: [] for c in order_upper}
        missing_counts: Dict[str, int] = {c: 0 for c in order_upper}
        for fname, cat in mapping.items():
            ucat = cat.upper()
            if ucat not in category_to_files:
                continue
            raw_path = Path(fname)
            if not raw_path.is_absolute():
                file_path = root / raw_path
            else:
                file_path = raw_path
            if file_path.is_file():
                category_to_files[ucat].append(file_path)
            else:
                missing_counts[ucat] += 1

        self.datasets: List[ListSliceDataset] = []
        self.raw_samples: List[Any] = []

        for cat in order_upper:
            file_list = category_to_files.get(cat, [])
            if not file_list:
                print(f"No existing files found for category {cat}. (Missing listed: {missing_counts.get(cat,0)})")
                continue
            cat_ds = ListSliceDataset(
                fnames=file_list,
                challenge=challenge,
                transform=transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                use_dataset_cache=use_dataset_cache,
                dataset_cache_file=dataset_cache_file,
            )
            # Shuffle slices within category
            cat_ds.shuffle_inplace()
            self.datasets.append(cat_ds)
            self.raw_samples.extend(cat_ds.raw_samples)
            print(f"Category {cat}: {len(cat_ds)} slices (shuffled within category).")

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        # Find which dataset contains index i
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)
        raise IndexError("Index out of range")


class CustomCategoryDataModule(pl.LightningDataModule):
    """
    Custom Data module for filtering MRI datasets by anatomical categories.
    
    This class handles configurations for training on filtered MRI datasets based
    on anatomical categories. It uses a JSON mapping file to determine which files
    belong to which categories and organizes them in a specified order.
    """

    def __init__(
        self,
        data_path: Path,
        category_mapping_path: Path,
        category_order: List[str],
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root directory of dataset.
            category_mapping_path: Path to JSON file with filename to category mapping.
            category_order: List of categories to include in order (e.g., ['SPINE', 'KNEE']).
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: Optional test path.
            sample_rate: Fraction of slices of the training data split to use.
            val_sample_rate: Same as sample_rate, but for val split.
            test_sample_rate: Same as sample_rate, but for test split.
            volume_sample_rate: Fraction of volumes of the training data split to use.
            val_volume_sample_rate: Same as volume_sample_rate, but for val split.
            test_volume_sample_rate: Same as volume_sample_rate, but for test split.
            use_dataset_cache_file: Whether to cache dataset metadata.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler.
        """
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )

        self.data_path = data_path
        self.category_mapping_path = category_mapping_path
        self.category_order = [cat.upper() for cat in category_order]
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
            elif data_partition == "test":
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )

        # Determine data path
        if data_partition in ("test", "challenge") and self.test_path is not None:
            data_path = self.test_path
        else:
            data_path = self.data_path / f"{self.challenge}_{data_partition}"

        # Create CategoryOrderedSliceDataset (train may optionally combine val later if desired)
        dataset = CategoryOrderedSliceDataset(
            root=data_path,
            category_mapping_path=self.category_mapping_path,
            category_order=self.category_order,
            challenge=self.challenge,
            transform=data_transform,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            use_dataset_cache=self.use_dataset_cache_file,
        )

        # Ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=False,  # Keep files in order, don't shuffle
        )

        return dataloader

    def prepare_data(self):
        """Prepare datasets for training."""
        if self.use_dataset_cache_file:
            # Prepare cache for dataset
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / f"{self.challenge}_test"
            
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                test_path,
            ]
            
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            
            # Build cache by instantiating ListSliceDataset per category (if desired)
            for data_path, data_transform in zip(data_paths, data_transforms):
                with open(self.category_mapping_path, "r") as f:
                    mapping = json.load(f)
                for cat in self.category_order:
                    # Collect only existing files
                    existing_files: List[Path] = []
                    missing = 0
                    for k, v in mapping.items():
                        if v.upper() != cat:
                            continue
                        raw_path = Path(k)
                        if not raw_path.is_absolute():
                            file_path = data_path / raw_path
                        else:
                            file_path = raw_path
                        if file_path.is_file():
                            existing_files.append(file_path)
                        else:
                            missing += 1
                    if not existing_files:
                        continue
                    _ = ListSliceDataset(
                        fnames=existing_files,
                        challenge=self.challenge,
                        transform=data_transform,
                        sample_rate=self.sample_rate,
                        volume_sample_rate=self.volume_sample_rate,
                        use_dataset_cache=True,
                        dataset_cache_file="dataset_cache.pkl",
                    )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split
        )

    def get_category_statistics(self):
        """
        Get statistics about the number of files per category in the dataset.
        
        Returns:
            dict: Category counts for filtered dataset
        """
        category_counts = {}
        for category in self.category_order:
            category_counts[category] = 0
        
        # Load category mapping directly
        with open(self.category_mapping_path, 'r') as f:
            category_mapping = json.load(f)
        
        for filename, category in category_mapping.items():
            if category.upper() in self.category_order:
                category_counts[category.upper()] += 1
        
        return category_counts

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to dataset root directory",
        )
        parser.add_argument(
            "--category_mapping_path",
            default=None,
            type=Path,
            help="Path to JSON file containing filename to category mapping",
        )
        parser.add_argument(
            "--category_order",
            default=None,
            nargs='+',
            type=str,
            help="List of categories to include in order (e.g., SPINE KNEE HIP)",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to test data",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--test_split",
            choices=("val", "test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help="Fraction of slices to use (train split only)",
        )
        parser.add_argument(
            "--val_sample_rate",
            default=None,
            type=float,
            help="Fraction of slices to use (val split only)",
        )
        parser.add_argument(
            "--test_sample_rate",
            default=None,
            type=float,
            help="Fraction of slices to use (test split only)",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes to use (train split only)",
        )
        parser.add_argument(
            "--val_volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes to use (val split only)",
        )
        parser.add_argument(
            "--test_volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes to use (test split only)",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser
