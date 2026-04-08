"""Anatomy-filtered VarNet inference using category-mapping JSON.

Based on `run_pretrained_varnet_inference.py`, but instead of running on the
whole split, this script evaluates only one anatomy category (e.g., KNEE or
SHOULDER) selected via `--anatomy`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from tqdm import tqdm

import fastmri.data.subsample as subsample
import fastmri.data.transforms as T
from fastmri.evaluate import nmse as compute_nmse
from fastmri.evaluate import psnr as compute_psnr
from fastmri.evaluate import ssim as compute_ssim
from fastmri.models import VarNet

from dataloaders.custom_category_data_module_clean import CategoryOrderedSliceDataset

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url: str, fname: str) -> None:
    response = requests.get(url, timeout=30, stream=True)
    response.raise_for_status()

    chunk_size = 8 * 1024 * 1024
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            if not chunk:
                continue
            progress_bar.update(len(chunk))
            fh.write(chunk)


def compute_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor, max_value: float | None
) -> dict[str, float]:
    reconstruction_np = np.squeeze(reconstruction.detach().cpu().numpy())
    target_np = np.squeeze(target.detach().cpu().numpy())

    if reconstruction_np.ndim != 2 or target_np.ndim != 2:
        raise ValueError(
            f"Expected 2D images for metric computation, got shapes "
            f"{reconstruction_np.shape} and {target_np.shape}"
        )

    if max_value is None or max_value <= 0:
        max_value = float(target_np.max() if target_np.size else 1.0)

    target_for_ssim = target_np[None, ...]
    recon_for_ssim = reconstruction_np[None, ...]

    return {
        "NMSE": float(compute_nmse(target_np, reconstruction_np)),
        "PSNR": float(compute_psnr(target_np, reconstruction_np, maxval=max_value)),
        "SSIM": float(compute_ssim(target_for_ssim, recon_for_ssim, maxval=max_value)),
    }


def run_varnet_model(batch: Any, model: torch.nn.Module, device: torch.device):
    crop_size = batch.crop_size

    masked_kspace = batch.masked_kspace.to(device)
    mask = batch.mask.to(device)

    num_low_frequencies = batch.num_low_frequencies
    if isinstance(num_low_frequencies, torch.Tensor):
        num_low_frequencies = num_low_frequencies.to(device)
    else:
        num_low_frequencies = torch.tensor(num_low_frequencies, device=device)

    output = model(masked_kspace, mask, num_low_frequencies)

    if output.shape[-1] < crop_size[0] or output.shape[-2] < crop_size[1]:
        crop_size = (
            min(output.shape[-2], int(crop_size[0])),
            min(output.shape[-1], int(crop_size[1])),
        )

    output = T.center_crop(output, crop_size)

    target = None
    if isinstance(batch.target, torch.Tensor) and batch.target.ndim >= 2:
        target = batch.target
        if target.ndim == 2:
            target = target.unsqueeze(0)
        target, output = T.center_crop_to_smallest(target, output)
        target = target[0]
        output = output[0]
    else:
        output = output[0]

    max_value = batch.max_value
    if isinstance(max_value, torch.Tensor):
        max_value = float(max_value.flatten()[0].item())
    else:
        max_arr = np.asarray(max_value).reshape(-1)
        max_value = float(max_arr[0]) if max_arr.size else 0.0

    return (
        output,
        int(batch.slice_num[0]),
        str(batch.fname[0]),
        target,
        max_value,
        masked_kspace,
    )


def save_visualization(
    *,
    fname: str,
    slice_num: int,
    masked_kspace: torch.Tensor,
    target_cpu: np.ndarray,
    output_cpu: np.ndarray,
    psnr: float | None,
    ssim: float | None,
    output_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for visualization saving. "
            "Install matplotlib or run with --save_every 0."
        )

    masked_kspace_np = masked_kspace.detach().cpu().numpy()
    if masked_kspace_np.shape[-1] == 2:
        kspace_cplx = masked_kspace_np[..., 0] + 1j * masked_kspace_np[..., 1]
    else:
        kspace_cplx = masked_kspace_np
    kspace_mag = np.log(np.abs(kspace_cplx[0]) + 1e-9)

    masked_kspace_t = masked_kspace
    if masked_kspace_t.shape[-1] == 2:
        masked_kspace_t = torch.view_as_complex(masked_kspace_t)

    img_coils = torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.fftshift(masked_kspace_t, dim=(-2, -1)),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    img_rss = torch.sqrt((img_coils.abs() ** 2).sum(dim=0)).detach().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    axs[0].imshow(kspace_mag, cmap="gray")
    axs[0].set_title("Masked k-space (log-mag)")
    axs[0].axis("off")

    axs[1].imshow(img_rss[0], cmap="gray")
    axs[1].set_title("Zero-filled (from masked)")
    axs[1].axis("off")

    axs[2].imshow(target_cpu, cmap="gray")
    axs[2].set_title("Target")
    axs[2].axis("off")

    axs[3].imshow(output_cpu, cmap="gray")
    axs[3].set_title("Output")
    axs[3].axis("off")

    title_psnr = f"{psnr:.2f} dB" if psnr is not None else "N/A"
    title_ssim = f"{ssim:.4f}" if ssim is not None else "N/A"
    fig.suptitle(f"{fname} | slice {slice_num} | PSNR: {title_psnr} | SSIM: {title_ssim}")

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{Path(fname).stem}_slice{slice_num}_viz.png"
    plt.savefig(out_file)
    plt.close(fig)


def build_anatomy_dataloader(args: argparse.Namespace):
    mask_func = subsample.create_mask_for_mask_type(
        "random",
        [args.center_fractions],
        [args.accelerations],
    )
    data_transform = T.VarNetDataTransform(mask_func=mask_func)
    anatomy = str(args.anatomy).upper()

    dataset = CategoryOrderedSliceDataset(
        root=args.data_path,
        category_mapping_path=args.category_mapping_path,
        category_order=[anatomy],
        challenge=args.challenge,
        transform=data_transform,
        # Disable cache to avoid shared-cache corruption when training/inference
        # processes access cache concurrently on the same machine.
        use_dataset_cache=False,
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )


def normalize_anatomy_label(raw_anatomy: str) -> str:
    key = str(raw_anatomy).strip().upper().replace("-", "_").replace("/", "_")
    alias_to_category = {
        "KNEE": "KNEE",
        "SHOULDER": "SHOULDER",
        "ANKLE": "ANKLE",
        "SPINE": "SPINE",
        "HIP": "HIP",
        "WRIST_HAND_THUMB": "WRIST_HAND_THUMB",
        "WRIST_HAND": "WRIST_HAND_THUMB",
        "WRISTHAND": "WRIST_HAND_THUMB",
        "WRIST_THUMB": "WRIST_HAND_THUMB",
        "HAND_WRIST": "WRIST_HAND_THUMB",
    }
    if key not in alias_to_category:
        valid = ", ".join(sorted(set(alias_to_category.values())))
        raise ValueError(f"Unsupported anatomy '{raw_anatomy}'. Supported categories: {valid}")
    return alias_to_category[key]


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.state_dict_file is None:
        model_name = MODEL_FNAMES["varnet_knee_mc"]
        if not Path(model_name).exists():
            download_model(VARNET_FOLDER + model_name, model_name)

        model = VarNet(
            num_cascades=args.cascades,
            pools=4,
            chans=18,
            sens_pools=4,
            sens_chans=8,
        )
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
        return model

    from fastmri.pl_modules import VarNetModule

    module = VarNetModule.load_from_checkpoint(str(args.state_dict_file))
    return module


def run_inference(args: argparse.Namespace) -> None:
    args.anatomy = normalize_anatomy_label(args.anatomy)
    device = torch.device(args.device)

    model = load_model(args)
    model = model.eval().to(device)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data path {args.data_path} does not exist.")
    if not args.category_mapping_path.exists():
        raise FileNotFoundError(f"Category mapping not found: {args.category_mapping_path}")

    dataloader = build_anatomy_dataloader(args)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    print(f"Anatomy filter: {str(args.anatomy).upper()}")
    print(f"Slices to process: {len(dataloader)}")

    outputs = defaultdict(list)
    per_volume_metrics = defaultdict(lambda: defaultdict(list))
    global_metrics = defaultdict(list)
    files_with_errors: list[str] = []

    processed_files_for_viz: set[str] = set()
    file_count = 0

    start_time = time.perf_counter()

    for batch in tqdm(dataloader, desc=f"Inference ({str(args.anatomy).upper()}-only)"):
        with torch.no_grad():
            try:
                output, slice_num, fname, target, max_value, masked_kspace = run_varnet_model(
                    batch, model, device
                )

                output_cpu = output.detach().cpu()
                outputs[fname].append((slice_num, output_cpu))

                if target is None:
                    continue

                target_cpu_t = target.detach().cpu()
                metrics = compute_metrics(output_cpu, target_cpu_t, max_value)
                for name, value in metrics.items():
                    per_volume_metrics[fname][name].append(value)
                    global_metrics[name].append(value)

                if (
                    args.save_every > 0
                    and fname not in processed_files_for_viz
                    and slice_num == args.viz_slice
                ):
                    processed_files_for_viz.add(fname)
                    file_count += 1
                    if file_count % args.save_every == 0:
                        save_visualization(
                            fname=fname,
                            slice_num=slice_num,
                            masked_kspace=masked_kspace,
                            target_cpu=target_cpu_t.numpy(),
                            output_cpu=output_cpu.numpy(),
                            psnr=metrics.get("PSNR"),
                            ssim=metrics.get("SSIM"),
                            output_path=output_path,
                        )

            except Exception as e:
                fname0 = str(getattr(batch, "fname", ["<unknown>"])[0])
                if fname0 not in files_with_errors:
                    files_with_errors.append(fname0)
                print(f"Error processing {fname0} slice {int(batch.slice_num[0])}: {e}")

    for fname in list(outputs.keys()):
        outputs[fname] = np.stack([out.numpy() for _, out in sorted(outputs[fname])])

    if global_metrics:
        per_volume_summary = {
            fname: {metric: float(np.mean(values)) for metric, values in metric_dict.items()}
            for fname, metric_dict in per_volume_metrics.items()
        }
        global_summary = {
            metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for metric, values in global_metrics.items()
        }

        (output_path / "metrics.json").write_text(
            json.dumps({"per_volume": per_volume_summary, "global": global_summary}, indent=2)
        )

        print("Reconstruction metrics (slice averages per volume):")
        for metric, stats in global_summary.items():
            print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        print("No ground-truth targets found; metrics were not computed.")

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f}s for {len(dataloader)} slices")
    if files_with_errors:
        print(f"Files with errors ({len(files_with_errors)}): {files_with_errors[:20]}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to lightning checkpoint (if omitted, downloads fastMRI knee weights)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to split folder (e.g., .../multicoil_test)",
    )
    parser.add_argument(
        "--category_mapping_path",
        type=Path,
        required=True,
        help="JSON mapping filename/path -> category string",
    )
    parser.add_argument(
        "--anatomy",
        type=str,
        required=True,
        help=(
            "Anatomy to evaluate. Supported canonical categories include "
            "KNEE, SHOULDER, ANKLE, SPINE, HIP, WRIST_HAND_THUMB. "
            "Aliases like WRISTHAND/WRIST_HAND are accepted."
        ),
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default="multicoil",
        choices=("singlecoil", "multicoil"),
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--cascades", type=int, default=8)
    parser.add_argument(
        "--accelerations",
        type=int,
        default=8,
        help="Acceleration factor for random undersampling mask",
    )
    parser.add_argument(
        "--center_fractions",
        type=float,
        default=0.04,
        help="Center fraction for random undersampling mask",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save visualization every N volumes (0 disables)",
    )
    parser.add_argument(
        "--viz_slice",
        type=int,
        default=5,
        help="Which slice index to visualize per volume",
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
