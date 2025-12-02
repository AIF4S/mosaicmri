"""
Custom Data Module for mixing two MRI datasets.

This module extends the FastMRI data module to support combining datasets from 
two different sources while maintaining the expected directory structure.
"""

import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch

import fastmri
from fastmri.data import CombinedSliceDataset, SliceDataset


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
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
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True
    return False


class CustomMixDataModule(pl.LightningDataModule):
    """
    Custom Data module for mixing two MRI datasets.
    
    This class handles configurations for training on mixed MRI datasets. 
    It combines two datasets by creating CombinedSliceDataset instances that
    mix volumes from both datasets for each split (train/val/test).
    
    Both datasets should follow the same directory structure:
    dataset1_path/multicoil_train, multicoil_val, multicoil_test
    dataset2_path/multicoil_train, multicoil_val, multicoil_test
    """

    def __init__(
        self,
        data_path1: Path,
        data_path2: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
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
        mix_seed: int = 42,  # Seed for mixing randomization
    ):
        """
        Args:
            data_path1: Path to root directory of first dataset.
            data_path2: Path to root directory of second dataset.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path1: Optional test path for dataset1.
            test_path2: Optional test path for dataset2.
            sample_rate: Fraction of slices of the training data split to use.
            val_sample_rate: Same as sample_rate, but for val split.
            test_sample_rate: Same as sample_rate, but for test split.
            volume_sample_rate: Fraction of volumes of the training data split to use.
            val_volume_sample_rate: Same as volume_sample_rate, but for val split.
            test_volume_sample_rate: Same as volume_sample_rate, but for test split.
            train_filter: A callable for filtering training examples.
            val_filter: Same as train_filter, but for val split.
            test_filter: Same as train_filter, but for test split.
            use_dataset_cache_file: Whether to cache dataset metadata.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler.
            mix_seed: Random seed for mixing datasets.
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

        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
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
            raw_sample_filter = self.train_filter
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
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.test_filter

        # Determine paths for both datasets
        if data_partition in ("test", "challenge"):
            if self.test_path1 is not None and self.test_path2 is not None:
                data_path1 = self.test_path1
                data_path2 = self.test_path2
            else:
                data_path1 = self.data_path1 / f"{self.challenge}_{data_partition}"
                data_path2 = self.data_path2 / f"{self.challenge}_{data_partition}"
        else:
            data_path1 = self.data_path1 / f"{self.challenge}_{data_partition}"
            data_path2 = self.data_path2 / f"{self.challenge}_{data_partition}"

        # Create the mixed dataset
        dataset: Union[SliceDataset, CombinedSliceDataset]
        
        if is_train and self.combine_train_val:
            # Combine train and val for both datasets
            data_paths = [
                self.data_path1 / f"{self.challenge}_train",
                self.data_path1 / f"{self.challenge}_val", 
                self.data_path2 / f"{self.challenge}_train",
                self.data_path2 / f"{self.challenge}_val",
            ]
            data_transforms = [data_transform] * 4
            challenges = [self.challenge] * 4
            sample_rates, volume_sample_rates = None, None
            if sample_rate is not None:
                sample_rates = [sample_rate] * 4
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate] * 4
        else:
            # Mix both datasets for the current partition
            data_paths = [data_path1, data_path2]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]

        # Randomize the order of datasets before creating combined dataset
        randomized_data = list(zip(data_paths, data_transforms, challenges))
        rng = random.Random(self.mix_seed)
        rng.shuffle(randomized_data)
        data_paths, data_transforms, challenges = zip(*randomized_data)
        data_paths, data_transforms, challenges = list(data_paths), list(data_transforms), list(challenges)
        
        # Adjust sample rates if they exist
        if sample_rates is not None:
            sample_rates = [sample_rates[0]] * len(data_paths)
        if volume_sample_rates is not None:
            volume_sample_rates = [volume_sample_rates[0]] * len(data_paths)

        # Create combined dataset
        dataset = CombinedSliceDataset(
            roots=data_paths,
            transforms=data_transforms,
            challenges=challenges,
            sample_rates=sample_rates,
            volume_sample_rates=[None, 0.6], # Balanced mixing !!!!!!!!!!!!!!!!!
            use_dataset_cache=self.use_dataset_cache_file,
            raw_sample_filter=raw_sample_filter,
        )

        # Ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def prepare_data(self):
        """Prepare datasets for training."""
        if self.use_dataset_cache_file:
            # Prepare cache for both datasets
            all_paths = []
            all_transforms = []
            
            # Dataset 1 paths
            if self.test_path1 is not None:
                test_path1 = self.test_path1
            else:
                test_path1 = self.data_path1 / f"{self.challenge}_test"
            
            all_paths.extend([
                self.data_path1 / f"{self.challenge}_train",
                self.data_path1 / f"{self.challenge}_val",
                test_path1,
            ])
            
            # Dataset 2 paths
            if self.test_path2 is not None:
                test_path2 = self.test_path2
            else:
                test_path2 = self.data_path2 / f"{self.challenge}_test"
            
            all_paths.extend([
                self.data_path2 / f"{self.challenge}_train", 
                self.data_path2 / f"{self.challenge}_val",
                test_path2,
            ])
            
            all_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ] * 2  # Same transforms for both datasets

            for i, (data_path, data_transform) in enumerate(zip(all_paths, all_transforms)):
                sample_rate = self.sample_rate
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=None,
                    challenge=self.challenge,
                    use_dataset_cache=True,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path1",
            default=None,
            type=Path,
            help="Path to first dataset root directory",
        )
        parser.add_argument(
            "--data_path2", 
            default=None,
            type=Path,
            help="Path to second dataset root directory",
        )
        parser.add_argument(
            "--test_path1",
            default=None,
            type=Path,
            help="Path to test data for first dataset",
        )
        parser.add_argument(
            "--test_path2",
            default=None,
            type=Path, 
            help="Path to test data for second dataset",
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
        parser.add_argument(
            "--mix_seed",
            default=42,
            type=int,
            help="Random seed for mixing datasets",
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
