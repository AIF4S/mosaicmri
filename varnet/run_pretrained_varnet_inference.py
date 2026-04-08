"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn import metrics
import torch
from tqdm import tqdm

import fastmri
import fastmri.data.subsample as subsample
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.evaluate import nmse as compute_nmse
from fastmri.evaluate import psnr as compute_psnr
from fastmri.evaluate import ssim as compute_ssim
from fastmri.models import VarNet
from fastmri.data.transforms import apply_mask, VarNetSample

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}

import torch
import numpy as np
from fastmri.data import transforms as T


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def compute_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor, max_value: float | None
) -> dict[str, float]:
    """
    Compute standard MRI reconstruction metrics (NMSE, PSNR, SSIM) for one slice.
    """
    reconstruction_np = reconstruction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    reconstruction_np = np.squeeze(reconstruction_np)
    target_np = np.squeeze(target_np)

    if reconstruction_np.ndim != 2 or target_np.ndim != 2:
        raise ValueError(
            f"Expected 2D images for metric computation, got shapes "
            f"{reconstruction_np.shape} and {target_np.shape}"
        )

    if max_value is None or max_value <= 0:
        max_value = float(target_np.max() if target_np.size else 1.0)

    target_for_ssim = target_np[None, ...]
    recon_for_ssim = reconstruction_np[None, ...]

    metrics = {
        "NMSE": float(compute_nmse(target_np, reconstruction_np)),
        "PSNR": float(compute_psnr(target_np, reconstruction_np, maxval=max_value)),
        "SSIM": float(compute_ssim(target_for_ssim, recon_for_ssim, maxval=max_value)),
    }

    return metrics


class VarNetDataTransformAutoMask(T.VarNetDataTransform):
    """
    Aplica la máscara en el eje correcto automáticamente:
    - Si la máscara estándar funciona sin swap (ky en -2), usa eso.
    - Si no, hace swap H<->W SOLO para enmascarar y vuelve.
    """
    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        k = torch.as_tensor(kspace)             # (C, H, W, 2)
        acq_start, acq_end = 0, k.shape[-2]
        if "padding" in attrs:
            acq_start = int(attrs["padding"][0])
            acq_end   = int(attrs["padding"][1])

        seed = None
        if self.use_seed:
            seed = tuple(map(ord, fname))

        def _try_apply(kspace_like):
            if self.mask_func is None:
                return kspace_like, None, 0
            try:
                return apply_mask(kspace_like, self.mask_func, seed=seed, padding=(acq_start, acq_end))
            except Exception:
                # cuando padding no cuadra con ese eje, cae aquí
                return None, None, None

        # 1) intentar sin swap (asume ky en -2)
        masked_a, mask_a, nlow_a = _try_apply(k)
        ok_a = (masked_a is not None) and (mask_a is not None) and (mask_a.max() > 0)

        # 2) intentar con swap (C, W, H, 2)
        ks_sw = k.permute(0, 2, 1, 3).contiguous()
        masked_b, mask_b, nlow_b = _try_apply(ks_sw)
        ok_b = (masked_b is not None) and (mask_b is not None) and (mask_b.max() > 0)

        if ok_a and (not ok_b):
            masked_k, mask_t = masked_a, mask_a
        elif ok_b and (not ok_a):
            masked_k, mask_t = masked_b.permute(0, 2, 1, 3).contiguous(), mask_b.permute(0, 2, 1, 3).contiguous()
        elif ok_a and ok_b:
            # elegir la que tenga más cobertura (más 1s)
            if mask_b.sum() > mask_a.sum():
                masked_k, mask_t = masked_b.permute(0, 2, 1, 3).contiguous(), mask_b.permute(0, 2, 1, 3).contiguous()
            else:
                masked_k, mask_t = masked_a, mask_a
        else:
            # último recurso: no aplicar máscara
            masked_k, mask_t, nlow_a = k, None, 0

        # resto igual a VarNetDataTransform
        if "recon_size" in attrs:
            crop_size = (int(attrs["recon_size"][0]), int(attrs["recon_size"][1]))
        else:
            crop_size = (int(k.shape[-3]), int(k.shape[-2]))

        if target is None:
            img = T.ifft2c(masked_k)
            targ = T.complex_abs(img)
            targ = T.center_crop(targ, crop_size)
        else:
            targ = torch.as_tensor(target)
        masked_k = kspace
        # all true mask_t
        # convert to boolean mask
        mask_t = torch.ones_like(mask_t, dtype=torch.bool)
        return VarNetSample(
            masked_kspace=masked_k,
            mask=mask_t,
            num_low_frequencies=nlow_a if ok_a or (not ok_b) else nlow_b,
            target=targ,
            fname=fname,
            slice_num=slice_num,
            max_value=torch.tensor(attrs["max"]) if "max" in attrs else None,
            crop_size=torch.tensor(crop_size),
        )

def run_varnet_model(batch, model, device, vis_kspace=False):
    crop_size = batch.crop_size

    masked_kspace = batch.masked_kspace.to(device)
    mask = batch.mask.to(device)
    num_low_frequencies = batch.num_low_frequencies
    if isinstance(num_low_frequencies, torch.Tensor):
        num_low_frequencies = num_low_frequencies.to(device)
    else:
        num_low_frequencies = torch.tensor(num_low_frequencies, device=device)

    if vis_kspace:
        output, kspace_output = model( masked_kspace, mask, num_low_frequencies) 
    else:   
        output = model( masked_kspace, mask, num_low_frequencies) 
        kspace_output = None

    if output.shape[-1] < crop_size[0] or output.shape[-2] < crop_size[1]:
        crop_size = (min(output.shape[-2], crop_size[0]), min(output.shape[-1], crop_size[1]))
    
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
        max_value = float(max_value)

    return output, int(batch.slice_num[0]), batch.fname[0], target, max_value, masked_kspace, kspace_output

def vis(batch, slice_num, masked_kspace, target_cpu, output_cpu, kspace_output, psnr, ssim, output_path):

    masked_kspace_np = masked_kspace.cpu().numpy() if hasattr(masked_kspace, 'numpy') else masked_kspace
    masked_kspace_mag = np.log(np.abs(masked_kspace_np[0, ..., 0] + 1e-9))

    if kspace_output is not None:

        kspace_output_np = kspace_output.cpu().numpy() if hasattr(kspace_output, 'numpy') else kspace_output
        kspace_output_mag = np.log(np.abs(kspace_output_np[0, ..., 0] + 1e-9))
        n_plots = 5
    else:
        n_plots = 4

    if masked_kspace.shape[-1] == 2:
        masked_kspace_cplx = torch.view_as_complex(masked_kspace)
    else:
        masked_kspace_cplx = masked_kspace
    # Zero-filled (IFFT) reconstruction
    img_coils = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(masked_kspace_cplx, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    img_rss = torch.sqrt((img_coils.abs() ** 2).sum(dim=0))

    fig, axs = plt.subplots(1, n_plots, figsize=(22, 5))
    axs[0].imshow(masked_kspace_mag[0], cmap='gray')
    axs[0].set_title('Masked k-space (log-mag)')
    axs[0].axis('off')
    axs[1].imshow(img_rss.cpu().numpy()[0], cmap='gray')
    axs[1].set_title('Zero-filled (from masked)')
    axs[1].axis('off')
    axs[2].imshow(target_cpu, cmap='gray')
    axs[2].set_title('Target')
    axs[2].axis('off')
    axs[3].imshow(output_cpu, cmap='gray')
    axs[3].set_title('Output')
    axs[3].axis('off')

    if kspace_output is not None:
        axs[4].imshow(kspace_output_mag[0], cmap='gray')
        axs[4].set_title('k-space output (mag)')
        axs[4].axis('off')

    # Put PSNR/SSIM in the main title
    title_psnr = f"{psnr:.2f} dB" if psnr is not None else "N/A"
    title_ssim = f"{ssim:.4f}"   if ssim is not None else "N/A"
    fig.suptitle(f"{batch.fname[0]} | slice {slice_num} | PSNR: {title_psnr} | SSIM: {title_ssim}", fontsize=12)

    plt.savefig(output_path / f"{batch.fname[0]}_slice{slice_num}_visualization.png")
    plt.close(fig)


def run_inference(state_dict_file, data_path, output_path, device, cascades=12):
    # download the state_dict if we don't have it
    if state_dict_file is None:
        doname=False
        model_name = MODEL_FNAMES["varnet_knee_mc"]
        if not Path(model_name).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + model_name, model_name)

        state_dict_file = model_name
        print(args.cascades)
        model = VarNet(num_cascades=cascades, pools=4, chans=18, sens_pools=4, sens_chans=8)

        model.load_state_dict(torch.load(state_dict_file))
    else:
        doname=True
        print("\n Loading Model from checkpoint!!!\n")
        from fastmri.pl_modules import VarNetModule
        model = VarNetModule.load_from_checkpoint(state_dict_file)
    
    model = model.eval()
    print("\n ACCELERATIONS", args.accelerations)
    print("\n CENTER FRACTIONS", args.center_fractions)
    print("\n CASCADES", cascades)
    print("\n RANDOM MASK")

    mask_func = subsample.create_mask_for_mask_type(
        "random", [args.center_fractions], [args.accelerations]
    )
    # NEW VARNET DATA TRANSFORM WITH AUTO MASK AXIS
    # data loader setup
    data_transform = T.VarNetDataTransform(mask_func=mask_func)
    dataset = SliceDataset(root=data_path, transform=data_transform, challenge="multicoil")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    per_volume_metrics = defaultdict(lambda: defaultdict(list))
    global_metrics = defaultdict(list)
    model = model.to(device)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")

    # Modify output_path based on data_path content
    if doname:
        folder_name = str(output_path).split("/")[-1]
        epoch = int(str(state_dict_file).split("/")[-1].split("=")[1][:2])

        root_name = str(output_path).split("/")[:-1]
        root_name = Path("/".join(root_name))
        if "msk" in str(data_path).lower():
            print("\nDetected MSK dataset in data_path.")
            output_path = root_name / "test_on_msk" / f"{folder_name}"
        elif "fastmri" in str(data_path).lower():
            print("\nDetected fastMRI dataset in data_path.")
            output_path = root_name / "test_on_fastmri" / f"{folder_name}_{epoch}"

    output_path.mkdir(parents=True, exist_ok=True)
    files_with_errors = []
    count = 0
    file_count = 0
    processed_files = set()
    print(len(dataloader), "slices to process.")

    for batch in tqdm(dataloader, desc="Running inference"):
        count += 1
  
        with torch.no_grad():
            try:
                output, slice_num, fname, target, max_value, masked_kspace, kspace_output = run_varnet_model(
                    batch, model, device
                )
                output_cpu = output.detach().cpu()
                target_cpu = target.detach().cpu()
                
                metrics = compute_metrics(output_cpu, target_cpu, max_value)
                for name, value in metrics.items():
                    per_volume_metrics[fname][name].append(value)
                    global_metrics[name].append(value)
                psnr = metrics["PSNR"]
                ssim = metrics["SSIM"]

                # Track unique files and save visualization every 10 files
                if fname not in processed_files and slice_num == 5:
                    processed_files.add(fname)
                    file_count += 1
                    if file_count % args.save_every == 0:
                        vis(batch, slice_num, masked_kspace, target_cpu.numpy(), output_cpu.numpy(), kspace_output, psnr, ssim, output_path)
                
                outputs[fname].append((slice_num, output_cpu))
            
            except Exception as e:
                if batch.fname[0] not in files_with_errors:
                    files_with_errors.append(batch.fname[0])
                    print(f"Error processing file {batch.fname[0]}, slice {batch.slice_num[0]}: {e}")
                continue

    # save outputs
    for fname in outputs:
        # Ensure all tensors are moved to CPU before converting to numpy
        outputs[fname] = np.stack([out.cpu().numpy() for _, out in sorted(outputs[fname])])

    #fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    if global_metrics:
        per_volume_summary = {
            fname: {
                metric: float(np.mean(values)) for metric, values in metric_dict.items()
            }
            for fname, metric_dict in per_volume_metrics.items()
        }
        global_summary = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric, values in global_metrics.items()
        }

        metrics_report = {
            "per_volume": per_volume_summary,
            "global": global_summary,
        }

        metrics_path = output_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_report, indent=2))

        print("Reconstruction metrics (slice averages per volume):")
        for metric, stats in global_summary.items():
            print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        print("No ground-truth targets found; metrics were not computed.")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")
    print(f"Files with errors: {files_with_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )
    parser.add_argument(
        "--vis_kspace",
        action="store_true"
    )
    parser.add_argument(
        "--cascades",
        type=int,
        default=8,
        help="Number of cascades in VarNet model",
    )
    parser.add_argument(
        "--accelerations",
        type=int,
        default=8,
        help="Acceleration factor for mask",
    )
    parser.add_argument(
        "--center_fractions",
        type=float,
        default=0.04,
        help="Center fraction for mask",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save visualization every N unique files",
    )

    args = parser.parse_args()

    run_inference(
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
        args.cascades
    )
