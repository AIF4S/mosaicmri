"""
Rebuttal data modules for explicit train/val configurations.

Implemented experiment variants:
1) train: knee-only                         -> val: knee-only
2) train: full random slices (limit N)      -> val: knee-only
3) train: knee-only single-contrast         -> val: knee-only
4) train: full random single-contrast (N)   -> val: knee-only
"""

from __future__ import annotations

import json
import os
import re
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import h5py
import pytorch_lightning as pl
import torch

from .custom_category_data_module_clean import ListSliceDataset, worker_init_fn


def _is_ignored_file(name: str) -> bool:
    # Keep compatibility with this repo's existing dataloaders (no lowfield ignore list here).
    return False


def _to_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="ignore")
    return str(value)


def _normalize_text(value) -> str:
    # Normalize common naming variations (e.g., PD-FS vs PD_FS).
    return _to_text(value).strip().upper().replace("-", "_")


def _load_category_mapping(category_mapping_path: Path) -> Dict[str, str]:
    with open(category_mapping_path, "r") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid category mapping JSON at {category_mapping_path}")
    return {str(k): _normalize_text(v) for k, v in mapping.items()}


def _resolve_from_mapping_key(root: Path, key: str) -> Optional[Path]:
    raw_path = Path(key)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                root / raw_path,
                root / raw_path.name,
            ]
        )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _collect_anatomy_files(root: Path, category_mapping: Dict[str, str], anatomy_name: str) -> List[Path]:
    target = _normalize_text(anatomy_name)
    files: List[Path] = []
    seen: Set[str] = set()
    for key, cat in category_mapping.items():
        if cat != target:
            continue
        fpath = _resolve_from_mapping_key(root, key)
        if fpath is None:
            continue
        if _is_ignored_file(fpath.name):
            continue
        s = str(fpath)
        if s not in seen:
            seen.add(s)
            files.append(fpath)
    return sorted(files)


def _collect_knee_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    return _collect_anatomy_files(root, category_mapping, "KNEE")


def _collect_shoulder_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    return _collect_anatomy_files(root, category_mapping, "SHOULDER")


def _collect_full_files(root: Path) -> List[Path]:
    files = [p for p in sorted(root.glob("*.h5")) if not _is_ignored_file(p.name)]
    return files


def _collect_non_knee_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    all_files = _collect_full_files(root)
    knee_files = set(str(p) for p in _collect_knee_files(root, category_mapping))
    return [p for p in all_files if str(p) not in knee_files]


def _collect_non_ankle_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    all_files = _collect_full_files(root)
    ankle_files = set(str(p) for p in _collect_anatomy_files(root, category_mapping, "ANKLE"))
    return [p for p in all_files if str(p) not in ankle_files]

def _collect_non_shoulder_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    all_files = _collect_full_files(root)
    shoulder_files = set(str(p) for p in _collect_anatomy_files(root, category_mapping, "SHOULDER"))
    return [p for p in all_files if str(p) not in shoulder_files]

def _collect_non_hip_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    all_files = _collect_full_files(root)
    hip_files = set(str(p) for p in _collect_anatomy_files(root, category_mapping, "HIP"))
    return [p for p in all_files if str(p) not in hip_files]

def _collect_non_wrist_hand_files(root: Path, category_mapping: Dict[str, str]) -> List[Path]:
    all_files = _collect_full_files(root)
    wrist_hand_files: Set[str] = set()
    for key, cat in category_mapping.items():
        c = _normalize_text(cat)
        if "WRIST" not in c and "HAND" not in c:
            continue
        fpath = _resolve_from_mapping_key(root, key)
        if fpath is not None:
            wrist_hand_files.add(str(fpath))
    return [p for p in all_files if str(p) not in wrist_hand_files]


def _file_contrast(path: Path) -> Optional[str]:
    with h5py.File(path, "r") as hf:
        if "contrast" in hf.attrs:
            return _normalize_text(hf.attrs["contrast"])
        if "acquisition" in hf.attrs:
            return _normalize_text(hf.attrs["acquisition"])
    return None


def _parse_contrast_targets(contrast_name: str) -> List[str]:
    # Accept separators like '+', ',' or whitespace (e.g., "PD+PD_FS", "PD,PD_FS", "PD PD_FS").
    parts = [_normalize_text(p) for p in re.split(r"[,\+\s]+", contrast_name) if p]
    # Keep insertion order but deduplicate.
    return list(dict.fromkeys(parts))


def _filter_files_by_contrast(files: List[Path], contrast_name: str) -> List[Path]:
    targets = set(_parse_contrast_targets(contrast_name))
    out: List[Path] = []
    for fpath in files:
        try:
            ctr = _file_contrast(fpath)
        except Exception:
            continue
        if ctr in targets:
            out.append(fpath)
    return out


class _BaseRebuttalMarchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        category_mapping_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Optional[Callable] = None,
        *,
        train_slice_limit: Optional[int] = None,
        contrast_name: Optional[str] = None,
        match_contrast_ratio: bool = False,
        random_seed: int = 42,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        use_dataset_cache_file: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.category_mapping_path = Path(category_mapping_path)
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_slice_limit = train_slice_limit
        self.contrast_name = contrast_name
        self.match_contrast_ratio = bool(match_contrast_ratio)
        self.random_seed = int(random_seed)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.use_dataset_cache_file = use_dataset_cache_file
        self._category_mapping = _load_category_mapping(self.category_mapping_path)
        self._slice_count_cache: Dict[str, int] = {}

    @property
    def train_root(self) -> Path:
        return self.data_path / f"{self.challenge}_train"

    @property
    def val_root(self) -> Path:
        return self.data_path / f"{self.challenge}_val"

    def _file_slice_count(self, fpath: Path) -> int:
        key = str(fpath)
        if key in self._slice_count_cache:
            return self._slice_count_cache[key]
        with h5py.File(fpath, "r") as hf:
            n = int(hf["kspace"].shape[0])
        self._slice_count_cache[key] = n
        return n

    def _sum_slices(self, files: List[Path]) -> int:
        return sum(self._file_slice_count(f) for f in files)

    def _select_files_to_budget(
        self,
        files: List[Path],
        slice_budget: Optional[int],
        *,
        seed_override: Optional[int] = None,
    ) -> Tuple[List[Path], int]:
        if slice_budget is None:
            selected = sorted(files)
            return selected, self._sum_slices(selected)

        budget = int(slice_budget)
        if budget <= 0:
            raise ValueError("--train_slice_limit must be > 0 when provided")

        shuffled = list(files)
        rng = random.Random(self.random_seed if seed_override is None else seed_override)
        rng.shuffle(shuffled)

        selected: List[Path] = []
        total = 0
        for fpath in shuffled:
            ns = self._file_slice_count(fpath)
            if total + ns <= budget:
                selected.append(fpath)
                total += ns

        # Avoid empty train set if every file has more slices than the budget.
        if not selected and shuffled:
            smallest = min(shuffled, key=self._file_slice_count)
            selected = [smallest]
            total = self._file_slice_count(smallest)

        return sorted(selected), int(total)

    def _select_files_to_budget_with_ratio(
        self,
        files: List[Path],
        slice_budget: int,
        contrast_targets: List[str],
    ) -> Tuple[List[Path], int, Dict[str, int]]:
        target_set = set(contrast_targets)
        files_by_contrast: Dict[str, List[Path]] = {c: [] for c in contrast_targets}
        for fpath in files:
            try:
                ctr = _file_contrast(fpath)
            except Exception:
                continue
            if ctr in target_set:
                files_by_contrast[ctr].append(fpath)

        pool_slices = {c: self._sum_slices(files_by_contrast[c]) for c in contrast_targets}
        total_pool = sum(pool_slices.values())
        if total_pool <= 0:
            sel, total = self._select_files_to_budget(files, slice_budget)
            return sel, total, {}

        desired_budget: Dict[str, int] = {}
        remainders: List[Tuple[float, str]] = []
        allocated = 0
        for c in contrast_targets:
            raw = float(slice_budget) * float(pool_slices[c]) / float(total_pool)
            base = int(raw)
            desired_budget[c] = base
            allocated += base
            remainders.append((raw - base, c))

        for _, c in sorted(remainders, reverse=True):
            if allocated >= slice_budget:
                break
            desired_budget[c] += 1
            allocated += 1

        selected_files: List[Path] = []
        selected_slices_by_contrast: Dict[str, int] = {c: 0 for c in contrast_targets}
        selected_keys: Set[str] = set()

        for i, c in enumerate(contrast_targets):
            part_files, part_slices = self._select_files_to_budget(
                files_by_contrast[c],
                desired_budget[c],
                seed_override=self.random_seed + 1009 * (i + 1),
            )
            selected_files.extend(part_files)
            selected_slices_by_contrast[c] = part_slices
            selected_keys.update(str(f) for f in part_files)

        total_selected = sum(selected_slices_by_contrast.values())
        remaining = max(0, int(slice_budget) - int(total_selected))

        # Greedy fill remaining budget by selecting files from contrasts with biggest
        # current deficit relative to desired per-contrast budgets.
        if remaining > 0:
            progress = True
            while progress and remaining > 0:
                progress = False
                deficits = [
                    (desired_budget[c] - selected_slices_by_contrast[c], c)
                    for c in contrast_targets
                ]
                for _deficit, c in sorted(deficits, reverse=True):
                    candidates = [
                        f
                        for f in files_by_contrast[c]
                        if str(f) not in selected_keys and self._file_slice_count(f) <= remaining
                    ]
                    if not candidates:
                        continue
                    pick = max(candidates, key=self._file_slice_count)
                    ns = self._file_slice_count(pick)
                    selected_files.append(pick)
                    selected_keys.add(str(pick))
                    selected_slices_by_contrast[c] += ns
                    remaining -= ns
                    progress = True
                    break

        selected_files = sorted(selected_files)
        total_selected = self._sum_slices(selected_files)
        return selected_files, int(total_selected), selected_slices_by_contrast

    def _build_list_dataset(self, files: List[Path], transform: Callable) -> ListSliceDataset:
        # Avoid cross-run cache corruption when multiple trainings write cache files
        # from the same working directory concurrently.
        cache_file = f"dataset_cache_{self.__class__.__name__}_{os.getpid()}.pkl"
        return ListSliceDataset(
            fnames=files,
            challenge=self.challenge,
            transform=transform,
            sample_rate=None,
            volume_sample_rate=None,
            use_dataset_cache=self.use_dataset_cache_file,
            dataset_cache_file=cache_file,
        )

    def _build_val_dataset(self) -> ListSliceDataset:
        knee_files = _collect_knee_files(self.val_root, self._category_mapping)
        if not knee_files:
            print(f"[RebuttalMarch] Warning: val KNEE file list is empty under {self.val_root}")
        dataset = self._build_list_dataset(knee_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation knee-only: files={len(knee_files)} slices={len(dataset)}"
        )
        return dataset

    def _build_train_dataset(self) -> ListSliceDataset:
        raise NotImplementedError

    def train_dataloader(self):
        dataset = self._build_train_dataset()
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
        dataset = self._build_val_dataset()
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
        return self.val_dataloader()

    def prepare_data(self):
        return


class KneeOnlyKneeValDataModule(_BaseRebuttalMarchDataModule):
    """train: knee-only, val: knee-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        knee_files = _collect_knee_files(self.train_root, self._category_mapping)
        if not knee_files:
            print(f"[RebuttalMarch] Warning: train KNEE file list is empty under {self.train_root}")
        selected_files, selected_slices = self._select_files_to_budget(
            knee_files,
            self.train_slice_limit,
        )
        dataset = self._build_list_dataset(selected_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train knee-only: files={len(selected_files)}/{len(knee_files)} "
            f"slices={len(dataset)} (~{selected_slices}) limit={self.train_slice_limit}"
        )
        return dataset


class FullRandomKneeValDataModule(_BaseRebuttalMarchDataModule):
    """train: full dataset random slices with limit, val: knee-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        files = _collect_full_files(self.train_root)
        selected_files, selected_slices = self._select_files_to_budget(
            files,
            self.train_slice_limit,
        )
        dataset = self._build_list_dataset(selected_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train mixed-anatomy file-sampled: files={len(selected_files)}/{len(files)} "
            f"slices={len(dataset)} (~{selected_slices}) limit={self.train_slice_limit}"
        )
        return dataset


class FullNoKneeNoKneeValDataModule(_BaseRebuttalMarchDataModule):
    """train: all files except KNEE, val: all files except KNEE."""

    def _build_train_dataset(self) -> ListSliceDataset:
        non_knee_files = _collect_non_knee_files(self.train_root, self._category_mapping)
        if not non_knee_files:
            print(
                f"[RebuttalMarch] Warning: train non-KNEE file list is empty under {self.train_root}"
            )
        dataset = self._build_list_dataset(non_knee_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train non-KNEE only: files={len(non_knee_files)} slices={len(dataset)}"
        )
        return dataset

    def _build_val_dataset(self) -> ListSliceDataset:
        non_knee_files = _collect_non_knee_files(self.val_root, self._category_mapping)
        if not non_knee_files:
            print(
                f"[RebuttalMarch] Warning: val non-KNEE file list is empty under {self.val_root}"
            )
        dataset = self._build_list_dataset(non_knee_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation non-KNEE only: files={len(non_knee_files)} slices={len(dataset)}"
        )
        return dataset


class FullNoAnkleNoAnkleValDataModule(_BaseRebuttalMarchDataModule):
    """train: all files except ANKLE, val: all files except ANKLE."""

    def _build_train_dataset(self) -> ListSliceDataset:
        non_ankle_files = _collect_non_ankle_files(self.train_root, self._category_mapping)
        if not non_ankle_files:
            print(
                f"[RebuttalMarch] Warning: train non-ANKLE file list is empty under {self.train_root}"
            )
        dataset = self._build_list_dataset(non_ankle_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train non-ANKLE only: files={len(non_ankle_files)} slices={len(dataset)}"
        )
        return dataset

    def _build_val_dataset(self) -> ListSliceDataset:
        non_ankle_files = _collect_non_ankle_files(self.val_root, self._category_mapping)
        if not non_ankle_files:
            print(
                f"[RebuttalMarch] Warning: val non-ANKLE file list is empty under {self.val_root}"
            )
        dataset = self._build_list_dataset(non_ankle_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation non-ANKLE only: files={len(non_ankle_files)} slices={len(dataset)}"
        )
        return dataset

class FullNoShoulderNoShoulderValDataModule(_BaseRebuttalMarchDataModule):
    """train: all files except SHOULDER, val: all files except SHOULDER."""

    def _build_train_dataset(self) -> ListSliceDataset:
        non_shoulder_files = _collect_non_shoulder_files(self.train_root, self._category_mapping)
        if not non_shoulder_files:
            print(
                f"[RebuttalMarch] Warning: train non-SHOULDER file list is empty under {self.train_root}"
            )
        dataset = self._build_list_dataset(non_shoulder_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train non-SHOULDER only: files={len(non_shoulder_files)} slices={len(dataset)}"
        )
        return dataset

    def _build_val_dataset(self) -> ListSliceDataset:
        non_shoulder_files = _collect_non_shoulder_files(self.val_root, self._category_mapping)
        if not non_shoulder_files:
            print(
                f"[RebuttalMarch] Warning: val non-SHOULDER file list is empty under {self.val_root}"
            )
        dataset = self._build_list_dataset(non_shoulder_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation non-SHOULDER only: files={len(non_shoulder_files)} slices={len(dataset)}"
        )
        return dataset

class FullNoHipNoHipValDataModule(_BaseRebuttalMarchDataModule):
    """train: all files except HIP, val: all files except HIP."""

    def _build_train_dataset(self) -> ListSliceDataset:
        non_hip_files = _collect_non_hip_files(self.train_root, self._category_mapping)
        if not non_hip_files:
            print(
                f"[RebuttalMarch] Warning: train non-HIP file list is empty under {self.train_root}"
            )
        dataset = self._build_list_dataset(non_hip_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train non-HIP only: files={len(non_hip_files)} slices={len(dataset)}"
        )
        return dataset

    def _build_val_dataset(self) -> ListSliceDataset:
        non_hip_files = _collect_non_hip_files(self.val_root, self._category_mapping)
        if not non_hip_files:
            print(
                f"[RebuttalMarch] Warning: val non-HIP file list is empty under {self.val_root}"
            )
        dataset = self._build_list_dataset(non_hip_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation non-HIP only: files={len(non_hip_files)} slices={len(dataset)}"
        )
        return dataset

class FullNoWristHandNoWristHandValDataModule(_BaseRebuttalMarchDataModule):
    """train: all files except WRIST/HAND, val: all files except WRIST/HAND."""

    def _build_train_dataset(self) -> ListSliceDataset:
        non_wrist_hand_files = _collect_non_wrist_hand_files(self.train_root, self._category_mapping)
        if not non_wrist_hand_files:
            print(
                f"[RebuttalMarch] Warning: train non-WRIST/HAND file list is empty under {self.train_root}"
            )
        dataset = self._build_list_dataset(non_wrist_hand_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train non-WRIST/HAND only: files={len(non_wrist_hand_files)} slices={len(dataset)}"
        )
        return dataset

    def _build_val_dataset(self) -> ListSliceDataset:
        non_wrist_hand_files = _collect_non_wrist_hand_files(self.val_root, self._category_mapping)
        if not non_wrist_hand_files:
            print(
                f"[RebuttalMarch] Warning: val non-WRIST/HAND file list is empty under {self.val_root}"
            )
        dataset = self._build_list_dataset(non_wrist_hand_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation non-WRIST/HAND only: files={len(non_wrist_hand_files)} slices={len(dataset)}"
        )
        return dataset


class KneeOnlySingleContrastKneeValDataModule(_BaseRebuttalMarchDataModule):
    """train: knee-only single-contrast, val: knee-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        if not self.contrast_name:
            raise ValueError("--contrast_name is required for knee-only-single-contrast")
        knee_files = _collect_knee_files(self.train_root, self._category_mapping)
        filtered = _filter_files_by_contrast(knee_files, self.contrast_name)
        if not filtered:
            print(
                f"[RebuttalMarch] Warning: no train KNEE files matched contrast "
                f"{self.contrast_name!r}"
            )
        selected_files, selected_slices = self._select_files_to_budget(
            filtered,
            self.train_slice_limit,
        )
        dataset = self._build_list_dataset(selected_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train knee contrast-filtered: contrast={self.contrast_name} "
            f"files={len(selected_files)}/{len(filtered)} slices={len(dataset)} "
            f"(~{selected_slices}) limit={self.train_slice_limit}"
        )
        return dataset


class FullRandomSingleContrastKneeValDataModule(_BaseRebuttalMarchDataModule):
    """train: full dataset single-contrast random slices with limit, val: knee-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        if not self.contrast_name:
            raise ValueError("--contrast_name is required for full-random-single-contrast")
        files = _collect_full_files(self.train_root)
        filtered = _filter_files_by_contrast(files, self.contrast_name)

        ratio_debug: Dict[str, int] = {}
        if self.train_slice_limit is not None and self.match_contrast_ratio:
            selected_files, selected_slices, ratio_debug = self._select_files_to_budget_with_ratio(
                filtered,
                int(self.train_slice_limit),
                _parse_contrast_targets(self.contrast_name),
            )
        else:
            selected_files, selected_slices = self._select_files_to_budget(
                filtered,
                self.train_slice_limit,
            )

        dataset = self._build_list_dataset(selected_files, self.train_transform)
        ratio_msg = ""
        if ratio_debug:
            ratio_msg = f" ratio_slices={ratio_debug}"
        print(
            f"[RebuttalMarch] Train mixed-anatomy contrast-filtered: contrast={self.contrast_name} "
            f"files={len(selected_files)}/{len(filtered)} slices={len(dataset)} "
            f"(~{selected_slices}) limit={self.train_slice_limit} match_ratio={self.match_contrast_ratio}"
            f"{ratio_msg}"
        )
        return dataset


class _BaseShoulderValDataModule(_BaseRebuttalMarchDataModule):
    """Base class for shoulder-val experiments."""

    def _build_val_dataset(self) -> ListSliceDataset:
        shoulder_files = _collect_shoulder_files(self.val_root, self._category_mapping)
        if not shoulder_files:
            print(f"[RebuttalMarch] Warning: val SHOULDER file list is empty under {self.val_root}")
        dataset = self._build_list_dataset(shoulder_files, self.val_transform)
        print(
            f"[RebuttalMarch] Validation shoulder-only: files={len(shoulder_files)} slices={len(dataset)}"
        )
        return dataset


class ShoulderOnlyShoulderValDataModule(_BaseShoulderValDataModule):
    """train: shoulder-only (file-sampled), val: shoulder-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        shoulder_files = _collect_shoulder_files(self.train_root, self._category_mapping)
        if not shoulder_files:
            print(f"[RebuttalMarch] Warning: train SHOULDER file list is empty under {self.train_root}")
        selected_files, selected_slices = self._select_files_to_budget(
            shoulder_files,
            self.train_slice_limit,
        )
        dataset = self._build_list_dataset(selected_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train shoulder-only: files={len(selected_files)}/{len(shoulder_files)} "
            f"slices={len(dataset)} (~{selected_slices}) limit={self.train_slice_limit}"
        )
        return dataset


class FullRandomShoulderValDataModule(_BaseShoulderValDataModule):
    """train: mixed-anatomy file-sampled, val: shoulder-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        files = _collect_full_files(self.train_root)
        selected_files, selected_slices = self._select_files_to_budget(
            files,
            self.train_slice_limit,
        )
        dataset = self._build_list_dataset(selected_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train mixed-anatomy file-sampled (shoulder val): files={len(selected_files)}/{len(files)} "
            f"slices={len(dataset)} (~{selected_slices}) limit={self.train_slice_limit}"
        )
        return dataset


class ShoulderContrastFamilyShoulderValDataModule(_BaseShoulderValDataModule):
    """train: shoulder-only + contrast-family filter, val: shoulder-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        if not self.contrast_name:
            raise ValueError("--contrast_name is required for shoulder contrast-family run")
        shoulder_files = _collect_shoulder_files(self.train_root, self._category_mapping)
        filtered = _filter_files_by_contrast(shoulder_files, self.contrast_name)
        if not filtered:
            print(
                f"[RebuttalMarch] Warning: no train SHOULDER files matched contrast "
                f"{self.contrast_name!r}"
            )
        selected_files, selected_slices = self._select_files_to_budget(
            filtered,
            self.train_slice_limit,
        )
        dataset = self._build_list_dataset(selected_files, self.train_transform)
        print(
            f"[RebuttalMarch] Train shoulder contrast-filtered: contrast={self.contrast_name} "
            f"files={len(selected_files)}/{len(filtered)} slices={len(dataset)} "
            f"(~{selected_slices}) limit={self.train_slice_limit}"
        )
        return dataset


class FullRandomContrastFamilyShoulderValDataModule(_BaseShoulderValDataModule):
    """train: mixed-anatomy + contrast-family filter, val: shoulder-only."""

    def _build_train_dataset(self) -> ListSliceDataset:
        if not self.contrast_name:
            raise ValueError("--contrast_name is required for mixed-anatomy contrast-family run")
        files = _collect_full_files(self.train_root)
        filtered = _filter_files_by_contrast(files, self.contrast_name)

        ratio_debug: Dict[str, int] = {}
        if self.train_slice_limit is not None and self.match_contrast_ratio:
            selected_files, selected_slices, ratio_debug = self._select_files_to_budget_with_ratio(
                filtered,
                int(self.train_slice_limit),
                _parse_contrast_targets(self.contrast_name),
            )
        else:
            selected_files, selected_slices = self._select_files_to_budget(
                filtered,
                self.train_slice_limit,
            )

        dataset = self._build_list_dataset(selected_files, self.train_transform)
        ratio_msg = ""
        if ratio_debug:
            ratio_msg = f" ratio_slices={ratio_debug}"
        print(
            f"[RebuttalMarch] Train mixed-anatomy contrast-filtered (shoulder val): contrast={self.contrast_name} "
            f"files={len(selected_files)}/{len(filtered)} slices={len(dataset)} "
            f"(~{selected_slices}) limit={self.train_slice_limit} match_ratio={self.match_contrast_ratio}"
            f"{ratio_msg}"
        )
        return dataset
