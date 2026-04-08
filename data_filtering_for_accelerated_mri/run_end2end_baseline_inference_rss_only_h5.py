#!/usr/bin/env python3
"""
Run end-to-end baseline inference (U-Net / ViT) with the same dataloader
pipeline used in training/validation.

By default this script saves:
  - metrics.json summary
  - PNG sample images (input / reconstruction / target / error)

H5 recon output saving is optional via --save_h5.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import fastmri.data.transforms as T
from fastmri.evaluate import nmse as compute_nmse
from fastmri.evaluate import psnr as eval_psnr
from fastmri.evaluate import ssim as eval_ssim

from src.train_eval_setups.end_to_end.utils import setup, get_dataloader


def _to_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    out = x.detach().cpu()
    while out.ndim > 2:
        out = out[0]
    return out


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


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
        candidates.extend([root / raw_path, root / raw_path.name])
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


def _build_manifest_from_files(files: List[Path], out_json: Path, dataset_name: str, split: str) -> Dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_name": dataset_name,
        "uuid": "",
        "num_slices": 0,
        "files": {},
    }

    total_slices = 0
    for fp in files:
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

    data["num_slices"] = total_slices
    with open(out_json, "w") as f:
        json.dump(data, f, indent=4)
    return data


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, object]:
    if args.model == "u-net":
        hyperparams = {
            "chans": int(args.unet_chans),
            "num_pools": int(args.unet_num_pools),
        }
    else:
        hyperparams = {
            "avrg_img_size": int(args.vit_avrg_img_size),
            "patch_size": [int(args.vit_patch_size[0]), int(args.vit_patch_size[1])],
            "in_chans": int(args.vit_in_chans),
            "embed_dim": int(args.vit_embed_dim),
            "depth": int(args.vit_depth),
            "num_heads": int(args.vit_num_heads),
        }

    model_config = {
        "model": args.model,
        "hyperparams": hyperparams,
        # Explicitly keep the same random-mask forward model as VarNet inference.
        "accel_factors": [8],
        "accelerations": [8],
        "center_fractions": [0.04],
        "mask_type": "random",
    }

    model, data_transform, forward_fn = setup("test", model_config, accl_factors=[8], seed=args.model_seed)
    # Keep the transform masking active to match VarNet-style inference:
    # random mask with the same center_fraction/acceleration specified above.

    checkpoint = torch.load(args.checkpoint_file, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format in {args.checkpoint_file}")

    state_dict = _clean_state_dict_keys(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model = model.to(device)
    return model, (data_transform, forward_fn)


def build_source_index(data_path: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)

    for p in sorted(data_path.rglob("*.h5")):
        name = p.name
        if name in index and index[name] != p:
            duplicates[name].extend([index[name], p])
        else:
            index[name] = p

    if duplicates:
        msg = ["Duplicate filenames found under data_path; cannot map reliably by basename:"]
        for name, paths in duplicates.items():
            uniq = sorted({str(x) for x in paths})
            msg.append(f"- {name}")
            msg.extend([f"    {x}" for x in uniq])
        raise RuntimeError("\n".join(msg))

    return index


def build_source_index_from_files(files: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(files):
        name = p.name
        if name in index and index[name] != p:
            duplicates[name].extend([index[name], p])
        else:
            index[name] = p

    if duplicates:
        msg = ["Duplicate filenames found in selected files; cannot map reliably by basename:"]
        for name, paths in duplicates.items():
            uniq = sorted({str(x) for x in paths})
            msg.append(f"- {name}")
            msg.extend([f"    {x}" for x in uniq])
        raise RuntimeError("\n".join(msg))
    return index


def _get_recon_size(src_h5: Path) -> Tuple[int, int] | None:
    with h5py.File(src_h5, "r") as hf:
        if "recon_size" not in hf.attrs:
            return None
        rs = hf.attrs["recon_size"]
        return int(rs[0]), int(rs[1])


def _compute_metrics(recon_2d: np.ndarray, target_t: torch.Tensor, max_value: float) -> Dict[str, float]:
    # Match training validation pipeline exactly.
    target_2d = _to_2d_tensor(target_t).numpy()
    h = min(int(recon_2d.shape[-2]), int(target_2d.shape[-2]))
    w = min(int(recon_2d.shape[-1]), int(target_2d.shape[-1]))
    recon_2d = T.center_crop(torch.from_numpy(recon_2d), (h, w)).numpy()
    target_2d = T.center_crop(torch.from_numpy(target_2d), (h, w)).numpy()

    if not np.isfinite(max_value) or max_value <= 0:
        max_value = float(np.max(target_2d))
        max_value = max(max_value, 1e-8)

    if not np.isfinite(recon_2d).all():
        recon_2d = np.nan_to_num(recon_2d, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(target_2d).all():
        target_2d = np.nan_to_num(target_2d, nan=0.0, posinf=0.0, neginf=0.0)

    target_3d = target_2d[None, ...]
    recon_3d = recon_2d[None, ...]
    ssim_value = float(eval_ssim(target_3d, recon_3d, max_value).item())
    psnr_value = float(eval_psnr(target_3d, recon_3d, max_value).item())
    if not np.isfinite(ssim_value):
        ssim_value = 1.0 if np.allclose(recon_2d, target_2d) else 0.0
    if not np.isfinite(psnr_value):
        psnr_value = 99.0 if np.allclose(recon_2d, target_2d) else 0.0

    return {
        "NMSE": float(compute_nmse(target_2d, recon_2d)),
        "PSNR": psnr_value,
        "SSIM": ssim_value,
    }


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (255.0 * x).astype(np.uint8)


def _save_png_samples(
    output_dir: Path,
    sample_prefix: str,
    input_2d: np.ndarray,
    recon_2d: np.ndarray,
    target_2d: Optional[np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(input_2d)).save(output_dir / f"{sample_prefix}_input.png")
    Image.fromarray(_to_uint8(recon_2d)).save(output_dir / f"{sample_prefix}_recon.png")
    if target_2d is not None:
        Image.fromarray(_to_uint8(target_2d)).save(output_dir / f"{sample_prefix}_target.png")
        err = np.abs(target_2d - recon_2d)
        Image.fromarray(_to_uint8(err)).save(output_dir / f"{sample_prefix}_error.png")


def save_rss_only_outputs(
    outputs: Dict[str, List[Tuple[int, np.ndarray]]],
    source_index: Dict[str, Path],
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    for fname in sorted(outputs.keys()):
        if fname not in source_index:
            raise FileNotFoundError(f"Could not find source file for {fname} under data_path.")

        src = source_index[fname]
        dst = output_path / fname
        if dst.exists() and not overwrite:
            continue

        recon = np.stack([arr for _, arr in sorted(outputs[fname], key=lambda x: x[0])], axis=0).astype(np.float32)

        with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
            fout.create_dataset("reconstruction_rss", data=recon)
            for k, v in fin.attrs.items():
                fout.attrs[k] = v


def run_inference(args: argparse.Namespace) -> None:
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if not args.checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_file}")
    if args.anatomy is not None and args.category_mapping_path is None:
        raise ValueError("--category_mapping_path is required when --anatomy is set.")
    if args.category_mapping_path is not None and not args.category_mapping_path.exists():
        raise FileNotFoundError(f"Category mapping not found: {args.category_mapping_path}")

    device = torch.device(args.device)
    model, (data_transform, forward_fn) = load_model(args=args, device=device)

    if args.anatomy is not None:
        category_mapping = _load_category_mapping(args.category_mapping_path)
        selected_files = _collect_anatomy_files(args.data_path, category_mapping, args.anatomy)
        if not selected_files:
            raise RuntimeError(f"No files selected for anatomy={args.anatomy} under {args.data_path}")
    else:
        selected_files = sorted(args.data_path.rglob("*.h5"))
        if not selected_files:
            raise RuntimeError(f"No .h5 files found under {args.data_path}")

    source_index = build_source_index_from_files(selected_files)
    manifest_dir = args.output_path / "_manifests"
    if args.anatomy is None:
        manifest_name = "inference_all.json"
        dataset_name = "inference_all"
    else:
        anatomy_norm = _normalize_text(args.anatomy)
        manifest_name = f"inference_{anatomy_norm}.json"
        dataset_name = f"inference_{anatomy_norm}"

    manifest_path = manifest_dir / manifest_name
    manifest = _build_manifest_from_files(
        files=selected_files,
        out_json=manifest_path,
        dataset_name=dataset_name,
        split="test",
    )
    dataloader = get_dataloader(
        str(manifest_path.resolve()),
        data_transform,
        num_workers=args.num_workers,
        augment_data=False,
        batch_size=1,
        shuffle=False,
        load_sensitivity_maps=True,
    )
    print(
        f"Inference selection: anatomy={_normalize_text(args.anatomy) if args.anatomy else 'ALL'} "
        f"files={len(selected_files)} slices={manifest['num_slices']}"
    )

    recon_size_cache: Dict[str, Tuple[int, int] | None] = {}
    outputs: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)

    per_volume_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    global_metrics: Dict[str, List[float]] = defaultdict(list)
    png_dir = args.output_path / "png_samples"
    total_steps = len(dataloader)
    num_png = max(0, int(args.num_png_samples))
    png_indices: Set[int] = set()
    if num_png > 0 and total_steps > 0:
        png_indices = set(np.linspace(0, total_steps - 1, num=min(num_png, total_steps), dtype=int).tolist())

    for step_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        with torch.no_grad():
            output = forward_fn(model, batch, device)
            output_2d = _to_2d_tensor(output)
            input_2d = _to_2d_tensor(batch.image)

            fname = str(batch.fname[0])
            slice_num = int(batch.slice_num[0].item())

            if fname not in source_index:
                raise FileNotFoundError(f"{fname} not found under {args.data_path}")
            if fname not in recon_size_cache:
                recon_size_cache[fname] = _get_recon_size(source_index[fname])

            recon_size = recon_size_cache[fname]
            if recon_size is not None:
                ch = min(int(output_2d.shape[-2]), int(recon_size[0]))
                cw = min(int(output_2d.shape[-1]), int(recon_size[1]))
                output_2d = T.center_crop(output_2d, (ch, cw))
                input_2d = T.center_crop(input_2d, (ch, cw))

            recon_np = output_2d.numpy()
            if args.save_h5:
                outputs[fname].append((slice_num, recon_np))

            target = batch.target
            if isinstance(target, torch.Tensor) and target.numel() > 1 and target.ndim >= 3:
                target_2d = _to_2d_tensor(target).numpy()
                max_value = float(batch.max_value.flatten()[0].item())
                metrics = _compute_metrics(recon_np, target, max_value)
                for k, v in metrics.items():
                    if np.isfinite(v):
                        vf = float(v)
                        per_volume_metrics[fname][k].append(vf)
                        global_metrics[k].append(vf)
            else:
                target_2d = None

            if step_idx in png_indices:
                prefix = f"{Path(fname).stem}_slice{slice_num:03d}"
                _save_png_samples(
                    output_dir=png_dir,
                    sample_prefix=prefix,
                    input_2d=input_2d.numpy(),
                    recon_2d=recon_np,
                    target_2d=target_2d,
                )

    if args.save_h5:
        save_rss_only_outputs(
            outputs=outputs,
            source_index=source_index,
            output_path=args.output_path,
            overwrite=args.overwrite,
        )
        print(f"Saved {len(outputs)} H5 files to: {args.output_path}")

    metrics_payload = {
        "model": args.model,
        "checkpoint_file": str(args.checkpoint_file),
        "data_path": str(args.data_path),
        "anatomy": (_normalize_text(args.anatomy) if args.anatomy else "ALL"),
        "num_files": len(selected_files),
        "num_slices": int(manifest["num_slices"]),
        "num_png_samples": int(num_png),
        "save_h5": bool(args.save_h5),
    }

    if global_metrics:
        per_volume_summary = {
            fname: {metric: float(np.mean(values)) for metric, values in metric_dict.items()}
            for fname, metric_dict in per_volume_metrics.items()
        }
        global_summary = {
            metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for metric, values in global_metrics.items()
        }
        metrics_payload["metrics"] = {
            "per_volume": per_volume_summary,
            "global": global_summary,
        }
        nmse_stats = global_summary["NMSE"]
        psnr_stats = global_summary["PSNR"]
        ssim_stats = global_summary["SSIM"]
        print("Reconstruction metrics (slice averages per volume):")
        print(
            f"  NMSE: mean={nmse_stats['mean']:.6f}, std={nmse_stats['std']:.6f}\n"
            f"  PSNR: mean={psnr_stats['mean']:.3f}, std={psnr_stats['std']:.3f}\n"
            f"  SSIM: mean={ssim_stats['mean']:.4f}, std={ssim_stats['std']:.4f}"
        )
    else:
        metrics_payload["metrics"] = None
        print("No targets available: skipped metric computation.")

    args.output_path.mkdir(parents=True, exist_ok=True)
    metrics_file = args.output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=("u-net", "vit"), required=True)
    parser.add_argument("--checkpoint_file", type=Path, required=True, help="Path to trained .pt/.ckpt file.")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to input H5 dataset.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save metrics/PNG outputs.")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run inference on.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output H5 files if --save_h5 is used.")
    parser.add_argument("--save_h5", action="store_true", help="Also save reconstruction H5 outputs.")
    parser.add_argument("--num_png_samples", type=int, default=12, help="Number of PNG sample slices to save.")
    parser.add_argument("--model_seed", type=int, default=42, help="Seed for model construction.")
    parser.add_argument("--category_mapping_path", type=Path, default=None, help="Path to filename->anatomy category JSON.")
    parser.add_argument("--anatomy", type=str, default=None, help="Optional anatomy filter (e.g., KNEE).")

    # U-Net hyperparams
    parser.add_argument("--unet_chans", type=int, default=128)
    parser.add_argument("--unet_num_pools", type=int, default=4)

    # ViT hyperparams
    parser.add_argument("--vit_avrg_img_size", type=int, default=320)
    parser.add_argument("--vit_patch_size", type=int, nargs=2, default=[10, 10])
    parser.add_argument("--vit_in_chans", type=int, default=1)
    parser.add_argument("--vit_embed_dim", type=int, default=48)
    parser.add_argument("--vit_depth", type=int, default=10)
    parser.add_argument("--vit_num_heads", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
