"""\
DataModule: JSON-selected training slices + category-filtered validation (MSK).

What it does
------------
- Train set is built from a selector JSON that enumerates explicit (file, slice)
  pairs. Only those slices are used (no volume fallback).
- Val set is built from the MSK dataset2 `multicoil_val` directory, filtered to
  a target category (default `KNEE`) using a filename->category JSON mapping.

This is a small, purpose-built module to support workflows like:
- train on dreamsim-selected slices
- validate on one MSK anatomy category only

Selector JSON schema
--------------------
Expected schema:

{
  "files": {
    "meas_...h5": {
      "path": "/abs/or/rel/path/to/meas_...h5",  # optional
      "split": "train"|"val"|"test",            # optional
      "slices": {"0": {...}, "1": {...}, ...}
    },
    ...
  }
}

Notes
-----
- The slice list is taken from `meta["slices"].keys()`.
- Path resolution prefers `meta["path"]` if provided; otherwise uses
  `root/<fname>`.

"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch

from fastmri.data import SliceDataset

from .custom_single_category_mix_data_module import worker_init_fn
from .custom_category_data_module_clean import ListSliceDataset


class JsonSliceDataset(torch.utils.data.Dataset):
    """Dataset that yields only explicitly selected slices from explicit files."""

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
        raise ValueError(
            f"Selector JSON at {selector_json_path} must contain a top-level 'files' dict"
        )
    return sel


def _resolve_file_path(root: Path, fname: str, json_path: Optional[str]) -> Path:
    p = Path(json_path) if json_path else Path(fname)
    if p.is_absolute() and p.is_file():
        return p
    # If JSON path is absolute but doesn't exist in this environment, fall back
    # to root/fname (common when moving JSONs between machines).
    return root / Path(fname).name


def _retrieve_metadata(fname: Path) -> Tuple[Dict[str, Any], int]:
    """Retrieve (metadata, num_slices) like fastMRI SliceDataset does."""
    import h5py
    import xml.etree.ElementTree as etree

    def _et_query(
        root: etree.Element,
        qlist: Iterable[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
    ) -> str:
        s = "."
        prefix = "ismrmrd_namespace"
        ns = {prefix: namespace}
        for el in qlist:
            s = s + f"//{prefix}:{el}"
        value = root.find(s, ns)
        if value is None:
            raise RuntimeError("Element not found")
        return str(value.text)

    with h5py.File(fname, "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(_et_query(et_root, enc + ["x"])),
            int(_et_query(et_root, enc + ["y"])),
            int(_et_query(et_root, enc + ["z"])),
        )
        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(_et_query(et_root, lims + ["center"]))
        enc_limits_max = int(_et_query(et_root, lims + ["maximum"])) + 1
        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max
        num_slices = hf["kspace"].shape[0]
        metadata: Dict[str, Any] = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            **hf.attrs,
        }
    return metadata, int(num_slices)


def _build_raw_samples_from_json(
    *,
    root: Path,
    selector: Dict[str, Any],
    desired_split: Optional[str],
) -> List[Tuple[Path, int, Dict[str, Any]]]:
    raw: List[Tuple[Path, int, Dict[str, Any]]] = []

    for fname, meta in selector["files"].items():
        if not isinstance(meta, dict):
            continue

        if desired_split is not None and meta.get("split") is not None:
            if str(meta.get("split")) != str(desired_split):
                continue

        fpath = _resolve_file_path(root, fname, meta.get("path"))
        if not fpath.is_file():
            continue

        if not (isinstance(meta.get("slices"), dict) and meta["slices"]):
            continue

        metadata, num_slices = _retrieve_metadata(fpath)

        try:
            slice_ids = sorted({int(k) for k in meta["slices"].keys()})
        except Exception:
            slice_ids = []

        for s in slice_ids:
            if 0 <= s < num_slices:
                raw.append((fpath, int(s), metadata))

    return raw


def _load_category_mapping(mapping_path: Path) -> Dict[str, str]:
    with open(mapping_path, "r") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError(f"Category mapping at {mapping_path} must be a JSON object")
    return {str(k): str(v) for k, v in m.items()}


def _is_knee_file(fname: str, mapping: Dict[str, str]) -> bool:
    """Return True if fname is mapped to category KNEE.

    The mapping keys might be absolute paths, relative paths, or just basenames.
    We try all three.
    """
    p = Path(fname)
    # Mapping keys in this repo are commonly either:
    # - just the basename (meas_....h5)
    # - a relative path within the dataset
    # - an absolute path from a different machine
    # So: try a few normalizations that are robust.
    # Try both filename-only matches and common relative forms.
    # In particular, your mapping may use keys like:
    #   "meas_....h5" (basename)
    #   "multicoil_val/meas_....h5" (split-relative)
    candidates = [
        p.name,
        p.stem,
        f"multicoil_val/{p.name}",
        f"multicoil_val/{p.stem}",
        str(p),
        str(p.with_suffix("")),
    ]
    for k in candidates:
        cat = mapping.get(k)
        if cat is not None and str(cat).upper() == "KNEE":
            return True
    return False


class JsonTrainKneeValDataModule(pl.LightningDataModule):
    """Train on explicit JSON slices; validate on one MSK category."""

    def __init__(
        self,
        data_path_train_root: Path,
        selector_json_path: Path,
        data_path_val_root: Path,
        category_mapping_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        *,
        require_json_split: bool = False,
        val_category: str = "KNEE",
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        super().__init__()
        self.data_path_train_root = Path(data_path_train_root)
        self.selector_json_path = Path(selector_json_path)
        self.data_path_val_root = Path(data_path_val_root)
        self.category_mapping_path = Path(category_mapping_path)
        self.challenge = challenge

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.require_json_split = require_json_split
        self.val_category = str(val_category).upper()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

        self._selector = _load_selector_json(self.selector_json_path)
        self._category_mapping = _load_category_mapping(self.category_mapping_path)

    def prepare_data(self):
        return

    def _train_dataset(self) -> JsonSliceDataset:
        desired_split = "train" if self.require_json_split else None
        raw_samples = _build_raw_samples_from_json(
            root=self.data_path_train_root,
            selector=self._selector,
            desired_split=desired_split,
        )
        if not raw_samples:
            print(
                f"[JsonTrainKneeVal] Warning: built empty train dataset from {self.selector_json_path} "
                f"with root {self.data_path_train_root}"
            )
        return JsonSliceDataset(raw_samples=raw_samples, challenge=self.challenge, transform=self.train_transform)

    def _val_dataset(self) -> SliceDataset:
        # Mirror the working category loader: resolve file paths explicitly from
        # the mapping JSON and build a ListSliceDataset from that list.
        # This avoids relying on SliceDataset(raw_sample_filter=...) behavior.

        root = Path(self.data_path_val_root)

        category_files: List[Path] = []
        for key, cat in self._category_mapping.items():
            if str(cat).upper() != self.val_category:
                continue
            raw_path = Path(key)
            file_path = raw_path if raw_path.is_absolute() else (root / raw_path)
            if file_path.is_file():
                category_files.append(file_path)

        if not category_files:
            print(
                f"[JsonTrainKneeVal] Warning: no {self.val_category} files found under "
                f"{root} using mapping {self.category_mapping_path}"
            )

        ds = ListSliceDataset(
            fnames=sorted(category_files),
            challenge=self.challenge,
            transform=self.val_transform,
            sample_rate=None,
            volume_sample_rate=None,
            use_dataset_cache=True,
            dataset_cache_file=f"dataset_cache_{self.__class__.__name__}_{os.getpid()}.pkl",
        )

        try:
            kept_vols = len({rs[0].name for rs in ds.raw_samples})
            print(
                f"[JsonTrainKneeVal] Val dataset: {kept_vols} "
                f"{self.val_category} volumes (explicit list)"
            )
        except Exception:
            pass

        return ds

    def train_dataloader(self):
        dataset = self._train_dataset()
        print(f"[JsonTrainKneeVal] Training dataset size: {len(dataset)} samples")
        sampler = None
        if self.distributed_sampler:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=(sampler is None),
        )

    def val_dataloader(self):
        dataset = self._val_dataset()
        print(f"[JsonTrainKneeVal] Validation dataset size: {len(dataset)} samples")
        sampler = None
        if self.distributed_sampler:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):
        # Not requested; keep same as val for now.
        return self.val_dataloader()

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path_train_root",
            type=Path,
            required=True,
            help="Root directory used to resolve JSON-selected training files (contains challenge_train/ or flat files)",
        )
        parser.add_argument(
            "--selector_json_path",
            type=Path,
            required=True,
            help="Selector JSON containing explicit (file, slice) pairs for training.",
        )
        parser.add_argument(
            "--data_path_val_root",
            type=Path,
            required=True,
            help="Validation root directory (e.g. .../varnet_msk_dataset2/multicoil_val)",
        )
        parser.add_argument(
            "--category_mapping_path",
            type=Path,
            required=True,
            help="Filename->category mapping JSON used to filter validation category.",
        )
        parser.add_argument(
            "--val_category",
            type=str,
            default="KNEE",
            help="Validation category to filter (e.g., KNEE, SHOULDER).",
        )
        parser.add_argument(
            "--require_json_split",
            action="store_true",
            help="If set, only JSON entries with meta.split=='train' are used for training.",
        )
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--distributed_sampler", action="store_true")
        return parser
