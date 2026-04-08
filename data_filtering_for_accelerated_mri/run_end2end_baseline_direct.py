#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py

# Reduce CUDA allocator fragmentation by default for long ViT runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from src.train_eval_setups.end_to_end import train  # noqa: E402
from src.train_eval_setups.end_to_end.utils import natural_sort  # noqa: E402


@dataclass
class TrainConfig:
    name: str
    num_epochs: int
    samples_seen: int
    epochs_setup: Dict[int, str]

    def dict(self):
        return {
            "name": self.name,
            "num_epochs": self.num_epochs,
            "samples_seen": self.samples_seen,
            "epochs_setup": self.epochs_setup,
        }


@dataclass
class EvalConfig:
    name: str
    classic: List[str]
    pathology: List[str]


def _to_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="ignore")
    return str(value)


def _normalize_text(value) -> str:
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
        s = str(fpath)
        if s not in seen:
            seen.add(s)
            files.append(fpath)
    return sorted(files)


def _collect_full_files(root: Path) -> List[Path]:
    return sorted(root.glob("*.h5"))


def _sample_files(files: List[Path], file_fraction: float, seed: int) -> List[Path]:
    if file_fraction >= 1.0:
        return sorted(files)
    n_total = len(files)
    n_keep = max(1, int(round(n_total * file_fraction)))
    rng = random.Random(int(seed))
    return sorted(rng.sample(files, n_keep))


def _file_slice_count(fpath: Path) -> int:
    with h5py.File(fpath, "r") as hf:
        if "kspace" not in hf:
            raise KeyError(f"Missing 'kspace' in {fpath}")
        return int(hf["kspace"].shape[0])


def _sum_slices(files: List[Path]) -> int:
    return sum(_file_slice_count(f) for f in files)


def _select_files_to_budget(files: List[Path], slice_budget: int, seed: int) -> Tuple[List[Path], int]:
    budget = int(slice_budget)
    if budget <= 0:
        raise ValueError("--train_slice_limit must be > 0")

    shuffled = list(files)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)

    selected: List[Path] = []
    total = 0
    for fpath in shuffled:
        ns = _file_slice_count(fpath)
        if total + ns <= budget:
            selected.append(fpath)
            total += ns

    # Avoid empty train set if every file has more slices than budget.
    if not selected and shuffled:
        smallest = min(shuffled, key=_file_slice_count)
        selected = [smallest]
        total = _file_slice_count(smallest)

    return sorted(selected), int(total)


def _build_or_load_manifest(
    h5_dir: Path,
    out_json: Path,
    dataset_name: str,
    split: str,
    rebuild: bool,
    selected_files: Optional[List[Path]] = None,
    file_fraction: float = 1.0,
    subset_seed: int = 42,
) -> Dict:
    # When selected_files is explicitly provided (custom filtering), always
    # rebuild so manifests cannot silently mismatch new selection criteria.
    if selected_files is None and out_json.exists() and not rebuild:
        with open(out_json, "r") as f:
            return json.load(f)

    files = sorted(selected_files) if selected_files is not None else sorted(h5_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    if file_fraction < 1.0:
        import random

        n_total = len(files)
        n_keep = max(1, int(round(n_total * file_fraction)))
        rng = random.Random(int(subset_seed))
        files = sorted(rng.sample(files, n_keep))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_name": dataset_name,
        "uuid": "",
        "num_slices": 0,
        "files": {},
    }

    total_slices = 0
    for i, fp in enumerate(files, start=1):
        with h5py.File(fp, "r") as hf:
            if "kspace" not in hf:
                raise KeyError(f"Missing 'kspace' in {fp}")
            n_slices = int(hf["kspace"].shape[0])
        total_slices += n_slices
        data["files"][fp.name] = {
            "path": str(fp),
            "dataset_name": dataset_name,
            "split": split,
            "slices": {str(s): {"freq": 1} for s in range(n_slices)},
        }
        if i % 200 == 0:
            print(f"[manifest] {dataset_name}: {i}/{len(files)} files")

    data["num_slices"] = total_slices
    with open(out_json, "w") as f:
        json.dump(data, f, indent=4)
    return data


def _choose_latest_checkpoint(checkpoint_dir: Path) -> Path:
    ckpts = [str(p) for p in checkpoint_dir.glob("epoch_*.pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return Path(natural_sort(ckpts)[-1])


def build_parser():
    p = argparse.ArgumentParser(description="Direct End2End baseline runner (no yml).")

    p.add_argument("--model", choices=("u-net", "vit"), required=True)
    p.add_argument("--train_h5_dir", type=Path, required=True)
    p.add_argument("--eval_h5_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument("--mask_type", choices=("random", "equispaced_fraction", "equispaced"), default="random")
    p.add_argument("--center_fractions", type=float, nargs="+", default=[0.04])
    p.add_argument("--accelerations", type=int, nargs="+", default=[8])

    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--num_checkpoints", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--finetune", type=str, default=None)
    p.add_argument("--samples_seen", type=int, default=None)
    p.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of train volumes (.h5 files) to include (0,1].")
    p.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of val volumes (.h5 files) to include (0,1].")
    p.add_argument("--subset_seed", type=int, default=42, help="Seed used for train subset sampling.")
    p.add_argument("--selection_seed", type=int, default=24, help="Seed for anatomy/slice-budget file selection.")
    p.add_argument("--train_slice_limit", type=int, default=None, help="Max train slices via file-level random selection.")
    p.add_argument("--category_mapping_path", type=Path, default=None, help="Path to filename->anatomy category JSON.")
    p.add_argument("--train_anatomy", type=str, default=None, help="Optional anatomy filter for train files (e.g., KNEE).")
    p.add_argument("--val_anatomy", type=str, default=None, help="Optional anatomy filter for val files (e.g., KNEE).")

    p.add_argument("--manifest_dir", type=Path, default=None)
    p.add_argument("--rebuild_manifests", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--val_every_n_epochs", type=int, default=1)

    # Weights & Biases logging
    p.add_argument("--enable_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mskmri-baselines")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_log_images", action="store_true")
    p.add_argument("--wandb_num_images", type=int, default=4)

    # UNet baseline hyperparams
    p.add_argument("--unet_chans", type=int, default=128)
    p.add_argument("--unet_num_pools", type=int, default=4)

    # ViT baseline hyperparams
    p.add_argument("--vit_avrg_img_size", type=int, default=320)
    p.add_argument("--vit_patch_size", type=int, nargs=2, default=[10, 10])
    p.add_argument("--vit_in_chans", type=int, default=1)
    p.add_argument("--vit_embed_dim", type=int, default=48)
    p.add_argument("--vit_depth", type=int, default=10)
    p.add_argument("--vit_num_heads", type=int, default=16)

    return p


def main():
    args = build_parser().parse_args()
    if not (0.0 < float(args.train_fraction) <= 1.0):
        raise ValueError(f"--train_fraction must be in (0, 1], got {args.train_fraction}")
    if not (0.0 < float(args.val_fraction) <= 1.0):
        raise ValueError(f"--val_fraction must be in (0, 1], got {args.val_fraction}")
    if args.train_slice_limit is not None and int(args.train_slice_limit) <= 0:
        raise ValueError(f"--train_slice_limit must be > 0, got {args.train_slice_limit}")

    run_dir = args.output_dir / args.run_name
    train_dir = run_dir / "train"
    eval_dir = run_dir / "eval"
    manifest_dir = args.manifest_dir if args.manifest_dir else (run_dir / "manifests")
    train_manifest = manifest_dir / "train.json"
    eval_manifest = manifest_dir / "eval.json"

    use_train_filter = bool(args.train_anatomy) or (args.train_slice_limit is not None)
    use_val_filter = bool(args.val_anatomy)
    if (use_train_filter or use_val_filter) and args.category_mapping_path is None:
        raise ValueError("--category_mapping_path is required when using --train_anatomy/--val_anatomy or slice-budget rebuttal settings.")

    category_mapping = None
    if args.category_mapping_path is not None:
        if not args.category_mapping_path.exists():
            raise FileNotFoundError(f"Category mapping not found: {args.category_mapping_path}")
        category_mapping = _load_category_mapping(args.category_mapping_path)

    selected_train_files = None
    selected_train_slices = None
    if use_train_filter:
        if args.train_anatomy:
            selected_train_files = _collect_anatomy_files(args.train_h5_dir, category_mapping, args.train_anatomy)
        else:
            selected_train_files = _collect_full_files(args.train_h5_dir)

        if not selected_train_files:
            raise RuntimeError(f"Train file selection is empty (train_h5_dir={args.train_h5_dir}, train_anatomy={args.train_anatomy}).")

        selected_train_files = _sample_files(selected_train_files, float(args.train_fraction), int(args.subset_seed))
        if args.train_slice_limit is not None:
            selected_train_files, selected_train_slices = _select_files_to_budget(
                selected_train_files,
                int(args.train_slice_limit),
                seed=int(args.selection_seed),
            )
        else:
            selected_train_slices = _sum_slices(selected_train_files)

    selected_eval_files = None
    if use_val_filter:
        selected_eval_files = _collect_anatomy_files(args.eval_h5_dir, category_mapping, args.val_anatomy)
        if not selected_eval_files:
            raise RuntimeError(f"Val file selection is empty (eval_h5_dir={args.eval_h5_dir}, val_anatomy={args.val_anatomy}).")
        selected_eval_files = _sample_files(selected_eval_files, float(args.val_fraction), int(args.subset_seed))

    # Prevent stale manifests from silently ignoring new subset/fraction choices.
    auto_rebuild_manifests = bool(args.rebuild_manifests)
    if (
        float(args.train_fraction) < 1.0
        or float(args.val_fraction) < 1.0
        or selected_train_files is not None
        or selected_eval_files is not None
    ):
        auto_rebuild_manifests = True

    train_manifest_data = _build_or_load_manifest(
        h5_dir=args.train_h5_dir,
        out_json=train_manifest,
        dataset_name=f"{args.run_name}_train",
        split="train",
        rebuild=auto_rebuild_manifests,
        selected_files=selected_train_files,
        file_fraction=(1.0 if selected_train_files is not None else float(args.train_fraction)),
        subset_seed=int(args.subset_seed),
    )
    eval_manifest_data = _build_or_load_manifest(
        h5_dir=args.eval_h5_dir,
        out_json=eval_manifest,
        dataset_name=f"{args.run_name}_eval",
        split="val",
        rebuild=auto_rebuild_manifests,
        selected_files=selected_eval_files,
        file_fraction=(1.0 if selected_eval_files is not None else float(args.val_fraction)),
        subset_seed=int(args.subset_seed),
    )

    if args.model == "u-net":
        hyperparams = {
            "chans": args.unet_chans,
            "num_pools": args.unet_num_pools,
        }
        train_batch_size = int(args.batch_size)
        grad_accum_steps = 1
        use_amp = False
        amp_dtype = "float16"
    else:
        hyperparams = {
            "avrg_img_size": args.vit_avrg_img_size,
            "patch_size": list(args.vit_patch_size),
            "in_chans": args.vit_in_chans,
            "embed_dim": args.vit_embed_dim,
            "depth": args.vit_depth,
            "num_heads": args.vit_num_heads,
            "use_checkpoint": True,
        }
        requested_batch = int(args.batch_size)
        world_size_for_batch = max(1, int(args.world_size))
        max_vit_global_batch_no_accum = 4 * world_size_for_batch
        # Keep per-GPU ViT batch <=4 by default for memory stability.
        if requested_batch > max_vit_global_batch_no_accum:
            train_batch_size = max_vit_global_batch_no_accum
            grad_accum_steps = int(math.ceil(requested_batch / train_batch_size))
        else:
            train_batch_size = requested_batch
            grad_accum_steps = 1
        use_amp = False
        amp_dtype = "float16"

    model_config = {
        "model": args.model,
        "hyperparams": hyperparams,
        "accel_factors": list(args.accelerations),
        "accelerations": list(args.accelerations),
        "center_fractions": list(args.center_fractions),
        "mask_type": args.mask_type,
        "lr": args.lr,
        "batch_size": int(train_batch_size),
        "val_dataset_path": str(eval_manifest.resolve()),
        "val_every_n_epochs": int(args.val_every_n_epochs),
        # Validation dataloading is more stable with fewer workers when many
        # DDP ranks are active in parallel.
        "val_num_workers": int(max(1, min(args.num_workers, 4))),
        "val_use_output_mask": False,
        "use_wandb": bool(args.enable_wandb),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "wandb_name": args.wandb_name if args.wandb_name else args.run_name,
        "wandb_log_images": bool(args.wandb_log_images),
        "wandb_num_images": int(args.wandb_num_images),
        # Built-in stability settings for ViT runs (no extra CLI flags needed).
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "grad_accum_steps": int(grad_accum_steps),
        "effective_batch_size": int(train_batch_size) * int(grad_accum_steps),
    }

    samples_seen = args.samples_seen
    if samples_seen is None:
        samples_seen = int(train_manifest_data["num_slices"]) * int(args.num_epochs)

    train_cfg = TrainConfig(
        name=args.run_name,
        num_epochs=args.num_epochs,
        samples_seen=samples_seen,
        epochs_setup={e: str(train_manifest.resolve()) for e in range(1, args.num_epochs + 1)},
    )
    eval_cfg = EvalConfig(
        name=f"{args.run_name}_eval",
        classic=[str(eval_manifest.resolve())],
        pathology=[],
    )

    print("=== Direct Run Config ===")
    print(f"run_dir={run_dir}")
    print(f"model={args.model}")
    print(f"train_fraction={args.train_fraction} val_fraction={args.val_fraction} subset_seed={args.subset_seed}")
    print(
        "selection: "
        f"train_anatomy={args.train_anatomy} val_anatomy={args.val_anatomy} "
        f"train_slice_limit={args.train_slice_limit} selection_seed={args.selection_seed}"
    )
    if selected_train_files is not None:
        print(f"selected_train_files={len(selected_train_files)} selected_train_slices~={selected_train_slices}")
    if selected_eval_files is not None:
        print(f"selected_eval_files={len(selected_eval_files)}")
    print(f"train_manifest={train_manifest} slices={train_manifest_data['num_slices']}")
    print(f"eval_manifest={eval_manifest} slices={eval_manifest_data['num_slices']}")
    print(f"mask_type={args.mask_type} center_fractions={args.center_fractions} accelerations={args.accelerations}")
    print(
        f"lr={args.lr} requested_batch_size={args.batch_size} "
        f"train_batch_size={model_config['batch_size']} "
        f"grad_accum_steps={model_config['grad_accum_steps']} "
        f"effective_batch_size={model_config['effective_batch_size']} "
        f"num_epochs={args.num_epochs} samples_seen={samples_seen}"
    )
    print(f"num_workers={args.num_workers} num_checkpoints={args.num_checkpoints}")
    print("=========================")

    if args.dry_run:
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_args = argparse.Namespace(
        world_size=args.world_size,
        output_dir=str(train_dir.resolve()),
        finetune=args.finetune,
        num_workers=args.num_workers,
        num_checkpoints=args.num_checkpoints,
        resume=args.resume,
        model_seed=args.model_seed,
        path_to_model_summary=str((run_dir / "model_summary.json").resolve()),
    )

    train.main(
        args=train_args,
        model_config=model_config,
        train_config=train_cfg,
        train_dataset_base_path="/",
    )

    if args.skip_eval:
        print("Skipping eval.")
        return

    ckpt_path = _choose_latest_checkpoint(train_dir / "checkpoints")
    # Lazy import so training-only runs do not require eval dependencies like lpips.
    from src.train_eval_setups.end_to_end import evaluate  # noqa: E402

    eval_args = argparse.Namespace(
        outfile=str((run_dir / "eval_summary.json").resolve()),
        model_path=str(ckpt_path.resolve()),
        num_workers=args.num_workers,
    )

    prev_cwd = Path.cwd()
    os.chdir(eval_dir)
    try:
        evaluate.main(
            args=eval_args,
            setup_config=model_config,
            setup_config_file=f"direct_{args.model}",
            eval_config=eval_cfg,
            eval_dataset_base_path="/",
        )
    finally:
        os.chdir(prev_cwd)

    print(f"Done. checkpoint={ckpt_path}")
    print(f"Eval summary: {run_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
