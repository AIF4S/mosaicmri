"""\
Data Module for mixing a JSON-selected subset from dataset1 with all data from dataset2.

Motivation
----------
You already have `SingleCategoryMixDataModule` which filters dataset1 using a
`category_mapping_path` (filename -> category) and a `category_name`.

This module replaces that filtering step by selecting **specific slices** from
an external dataset JSON (like `dreamsim_ensemble_emb_p128_msk_mri.json`).

Expected JSON schema
--------------------
The selector JSON should contain a top-level `files` mapping like:

{
  "files": {
    "meas_...h5": {
      "path": "/data/.../multicoil_train/meas_...h5",
      "split": "train" | "val" | "test",
      "slices": {"0": {"freq": 1}, "1": {"freq": 1}, ...}
    },
    ...
  }
}

Selection semantics
-------------------
Dataset1 is built from the set of **(volume, slice_idx)** pairs specified in the
JSON. This module never falls back to using all slices from a volume.

Notes
-----
- Dataset2 is loaded normally via `fastmri.data.SliceDataset`.
- Both datasets are concatenated and optionally shuffled (dataset-order shuffle)
  via `mix_seed`, mirroring `SingleCategoryMixDataModule`.

"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Iterable, Tuple

import json
import random

import pytorch_lightning as pl
import torch

from fastmri.data import SliceDataset

# We reuse seeding logic + ListSliceDataset metadata retrieval conventions.
from .custom_single_category_mix_data_module import worker_init_fn, _check_both_not_none


class JsonSliceDataset(torch.utils.data.Dataset):
    """A lightweight dataset that yields only selected slices from explicit files.

    It wraps `SliceDataset` per-volume access pattern by storing `raw_samples`
    tuples: (fname: Path, slice_ind: int, metadata: dict).

    This mirrors `ListSliceDataset` from `custom_category_data_module_clean.py`, but
    instead of enumerating *all* slice indices from each file, it only uses the
    indices requested in the JSON.

    Caching behavior intentionally stays simple (no dataset_cache.pkl) because the
    slice list is already precomputed.
    """

    def __init__(
        self,
        raw_samples: List[Tuple[Path, int, Dict[str, Any]]],
        challenge: str,
        transform: Optional[Callable] = None,
    ):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = raw_samples

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, slice_ind, metadata = self.raw_samples[i]
        import h5py
        import numpy as np

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][slice_ind]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][slice_ind] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            return (kspace, mask, target, attrs, fname.name, slice_ind)
        return self.transform(kspace, mask, target, attrs, fname.name, slice_ind)


def _load_selector_json(selector_json_path: Path) -> Dict[str, Any]:
    with open(selector_json_path, "r") as f:
        sel = json.load(f)
    if not isinstance(sel, dict) or "files" not in sel or not isinstance(sel["files"], dict):
        raise ValueError(f"Selector JSON at {selector_json_path} must contain a top-level 'files' dict")
    return sel


def _resolve_partition_root(data_path1: Path, challenge: str, partition: str, override: Optional[Path]) -> Path:
    if partition in ("test", "challenge"):
        if override is not None:
            return override
        return data_path1 / f"{challenge}_{partition}"
    return data_path1 / f"{challenge}_{partition}"


def _resolve_file_path(root: Path, fname: str, json_path: Optional[str]) -> Path:
    p = Path(json_path) if json_path else Path(fname)
    if p.is_absolute() and p.is_file():
        return p
    # if JSON path is absolute but doesn't exist here, fall back to root/fname
    return root / Path(fname).name


def _build_raw_samples_from_json(
    *,
    root: Path,
    selector: Dict[str, Any],
    challenge: str,
    desired_split: Optional[str],
    use_json_slices: bool,
) -> List[Tuple[Path, int, Dict[str, Any]]]:
    """Build (fname, slice_idx, metadata) list for JsonSliceDataset."""

    # Import here to avoid heavyweight dependency at import time.
    from .custom_category_data_module_clean import ListSliceDataset

    raw: List[Tuple[Path, int, Dict[str, Any]]] = []

    for fname, meta in selector["files"].items():
        if not isinstance(meta, dict):
            continue

        if desired_split is not None and meta.get("split") is not None:
            if str(meta.get("split")) != str(desired_split):
                continue

        fpath = _resolve_file_path(root, fname, meta.get("path"))
        if not fpath.is_file():
            # Keep silent to avoid huge logs; caller can add diagnostics if desired.
            continue

        # Retrieve metadata + num_slices using ListSliceDataset helper.
        # We instantiate a tiny one-file dataset to reuse its internal metadata logic.
        helper = ListSliceDataset(
            fnames=[fpath],
            challenge=challenge,
            transform=None,
            # Important: ListSliceDataset forbids setting BOTH sample_rate and volume_sample_rate
            # (even if both are 1.0). We don't need any subsampling here; we're just retrieving
            # metadata + bounds checking, so keep both as None.
            sample_rate=None,
            volume_sample_rate=None,
            use_dataset_cache=False,
        )
        # helper.raw_samples currently contains all slices; pull metadata from first tuple.
        if not helper.raw_samples:
            continue
        _fname0, _slice0, metadata = helper.raw_samples[0]

        # Default / intended behavior: select ONLY the slice indices explicitly listed in JSON.
        # This is what makes the resulting dataset length match the JSON's `num_slices`.
        if use_json_slices:
            if not (isinstance(meta.get("slices"), dict) and meta["slices"]):
                # If we require JSON-slice selection but slices are missing, skip the volume.
                continue
            slice_ids: Iterable[int] = []
            try:
                slice_ids = sorted({int(k) for k in meta["slices"].keys()})
            except Exception:
                # fallback: ignore malformed slice keys
                slice_ids = []

            for s in slice_ids:
                # ensure in bounds by checking against helper length per volume
                if 0 <= s < len(helper.raw_samples):
                    raw.append((fpath, int(s), metadata))
    return raw


class JsonSliceMixDataModule(pl.LightningDataModule):
    """Mix a JSON-selected subset from dataset1 with full dataset2."""

    def __init__(
        self,
        data_path1: Path,
        data_path2: Path,
        selector_json_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        *,
    # Kept for backward compatibility, but full-volume selection is not supported.
    # This must remain True.
    use_json_slices: bool = True,
        require_json_split: bool = False,
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

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Set either sample_rate or volume_sample_rate, not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError("Set either val_sample_rate or val_volume_sample_rate, not both.")
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError("Set either test_sample_rate or test_volume_sample_rate, not both.")

        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.selector_json_path = selector_json_path
        self.challenge = challenge

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        if not use_json_slices:
            raise ValueError(
                "JsonSliceMixDataModule does not support selecting whole volumes. "
                "Set use_json_slices=True to select only the slices explicitly listed in the JSON."
            )
        self.use_json_slices = True
        self.require_json_split = require_json_split

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

        # Load selector once
        self._selector = _load_selector_json(self.selector_json_path)

        # If the JSON provides a global num_slices, we can sanity-check that our selection
        # matches it when `use_json_slices=True`.
        self._expected_num_slices = None
        if isinstance(self._selector.get("num_slices"), int):
            self._expected_num_slices = int(self._selector["num_slices"])

    # ---------------- internal helpers -----------------
    def _build_dataset1(self, root: Path, transform: Callable, partition: str) -> JsonSliceDataset:
        desired_split = partition if self.require_json_split else None
        raw_samples = _build_raw_samples_from_json(
            root=root,
            selector=self._selector,
            challenge=self.challenge,
            desired_split=desired_split,
            use_json_slices=self.use_json_slices,
        )
        # Hard sanity check: when using JSON slice lists, the dataset length should match
        # the JSON's declared total number of slices (e.g. 8293) *unless* you enable
        # require_json_split, in which case the JSON may include other partitions.
        if (
            self.use_json_slices
            and (self._expected_num_slices is not None)
            and (not self.require_json_split)
            and len(raw_samples) != self._expected_num_slices
        ):
            raise ValueError(
                f"[JsonSliceMix] Dataset1 slice selection mismatch: built {len(raw_samples)} slices, "
                f"but selector JSON declares num_slices={self._expected_num_slices}. "
                "This usually means you are accidentally including whole volumes or skipping files due to path resolution."
            )
        if not raw_samples:
            print(f"[JsonSliceMix] Warning: no selected slices found for dataset1 at {root} (partition={partition}).")
        return JsonSliceDataset(raw_samples=raw_samples, challenge=self.challenge, transform=transform)

    def _build_dataset2(self, root: Path, transform: Callable, sample_rate: Optional[float], volume_sample_rate: Optional[float], raw_sample_filter: Optional[Callable]) -> SliceDataset:
        return SliceDataset(
            root=root,
            transform=transform,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            challenge=self.challenge,
            use_dataset_cache=self.use_dataset_cache_file,
            raw_sample_filter=raw_sample_filter,
        )

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

        # Resolve roots
        if data_partition in ("test", "challenge"):
            root1 = self.test_path1 if self.test_path1 is not None else (self.data_path1 / f"{self.challenge}_{data_partition}")
            root2 = self.test_path2 if self.test_path2 is not None else (self.data_path2 / f"{self.challenge}_{data_partition}")
        else:
            root1 = self.data_path1 / f"{self.challenge}_{data_partition}"
            root2 = self.data_path2 / f"{self.challenge}_{data_partition}"

        datasets: List[torch.utils.data.Dataset] = []

        if is_train and self.combine_train_val:
            # dataset1 from train + val using same selector
            ds1_train = self._build_dataset1(self.data_path1 / f"{self.challenge}_train", data_transform, partition="train")
            ds1_val = self._build_dataset1(self.data_path1 / f"{self.challenge}_val", data_transform, partition="val")

            ds2_train = self._build_dataset2(self.data_path2 / f"{self.challenge}_train", data_transform, sample_rate, volume_sample_rate, raw_sample_filter)
            ds2_val = self._build_dataset2(self.data_path2 / f"{self.challenge}_val", data_transform, sample_rate, volume_sample_rate, raw_sample_filter)

            datasets.extend([ds1_train, ds1_val, ds2_train, ds2_val])
        else:
            ds1 = self._build_dataset1(root1, data_transform, partition=data_partition)
            ds2 = self._build_dataset2(root2, data_transform, sample_rate, volume_sample_rate, raw_sample_filter)
            datasets.extend([ds1, ds2])

        rng = random.Random(self.mix_seed)
        rng.shuffle(datasets)
        concat_dataset = torch.utils.data.ConcatDataset(datasets)

        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(concat_dataset, shuffle=True)
            else:
                print("[JsonSliceMix] Warning: VolumeSampler not applied for eval with ConcatDataset.")

        return torch.utils.data.DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
        )

    # --------------- Lightning hooks ---------------
    def prepare_data(self):
        # No caching preparation beyond what SliceDataset does.
        return

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        if self.val_source == "mixed":
            return self._create_data_loader(self.val_transform, data_partition="val")

        # single-source validation
        sample_rate = self.val_sample_rate
        volume_sample_rate = self.val_volume_sample_rate
        raw_sample_filter = self.val_filter

        root1 = self.data_path1 / f"{self.challenge}_val"
        root2 = self.data_path2 / f"{self.challenge}_val"

        if self.val_source == "data1":
            ds = self._build_dataset1(root1, self.val_transform, partition="val")
        else:
            ds = self._build_dataset2(root2, self.val_transform, sample_rate, volume_sample_rate, raw_sample_filter)

        sampler = None
        if self.distributed_sampler:
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
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Quick stats for dataset1 selection (counts may differ per partition roots)."""
        files = self._selector.get("files", {})
        n_files = len(files)
        n_slices = 0
        for _fname, meta in files.items():
            if isinstance(meta, dict) and isinstance(meta.get("slices"), dict):
                n_slices += len(meta["slices"])  # as represented in JSON
        return {
            "files_in_json": n_files,
            "slice_entries_in_json": n_slices,
            "use_json_slices": True,
        }

    # --------------- CLI args ---------------
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path1", type=Path, required=True, help="Path to first dataset root")
        parser.add_argument("--data_path2", type=Path, required=True, help="Path to second dataset root")
        parser.add_argument("--selector_json_path", type=Path, required=True, help="Dataset JSON containing selected files/slices")
        parser.add_argument("--challenge", choices=("singlecoil", "multicoil"), default="singlecoil")
        parser.add_argument(
            "--use_json_slices",
            type=bool,
            default=True,
            help="Must be True. This dataloader always selects only slice indices listed in the JSON.",
        )
        parser.add_argument("--require_json_split", type=bool, default=False, help="If True, dataset1 considers only JSON entries whose meta.split matches the current partition.")
        parser.add_argument(
            "--val_source",
            choices=("mixed", "data1", "data2"),
            default="mixed",
            help="Which dataset to use for validation: mixed concat, dataset1-only, or dataset2-only.",
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
