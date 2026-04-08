#!/usr/bin/env python3
"""
Run VarNet inference and save H5 outputs with:
  - reconstruction_rss
  - all source attrs

No kspace is saved in outputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import requests
import torch
from tqdm import tqdm

import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.transforms import VarNetSample
from fastmri.models import VarNet


VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {"varnet_knee_mc": "knee_leaderboard_state_dict.pt"}


class VarNetDataTransformNoMask:
    """
    Transform for already-accelerated k-space data.
    It does not apply any additional masking.
    """

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        kspace_torch = T.to_tensor(kspace)

        if "recon_size" in attrs:
            crop_size = (int(attrs["recon_size"][0]), int(attrs["recon_size"][1]))
        else:
            crop_size = (int(kspace_torch.shape[-3]), int(kspace_torch.shape[-2]))

        if target is not None:
            target_torch = T.to_tensor(target)
        else:
            # Not used for this inference path; keep a placeholder tensor for collation.
            target_torch = torch.tensor(0)

        # All-ones mask means "no extra masking" for model forward.
        mask_torch = torch.ones((1, 1, kspace_torch.shape[-2], 1), dtype=torch.bool)

        return VarNetSample(
            masked_kspace=kspace_torch,
            mask=mask_torch,
            num_low_frequencies=torch.tensor(0),
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=torch.tensor(float(attrs["max"])) if "max" in attrs else torch.tensor(0.0),
            crop_size=torch.tensor(crop_size),
        )


def download_model(url: str, fname: Path) -> None:
    response = requests.get(url, timeout=30, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(fname, "wb") as fh, tqdm(
        desc=f"Downloading {fname.name}", total=total_size, unit="iB", unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(8 * 1024 * 1024):
            fh.write(chunk)
            pbar.update(len(chunk))


def _load_model_from_pt(state_path: Path, cascades: int) -> VarNet:
    model = VarNet(num_cascades=cascades, pools=4, chans=18, sens_pools=4, sens_chans=8)
    raw = torch.load(state_path, map_location="cpu")

    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]

    if isinstance(raw, dict):
        cleaned = {}
        for k, v in raw.items():
            nk = k
            if nk.startswith("varnet."):
                nk = nk[len("varnet.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v
        try:
            model.load_state_dict(cleaned, strict=False)
            return model
        except Exception:
            model.load_state_dict(raw, strict=False)
            return model

    raise ValueError(f"Unsupported checkpoint format at {state_path}")


def load_model(state_dict_file: Path | None, cascades: int, device: torch.device) -> torch.nn.Module:
    if state_dict_file is None:
        model_name = MODEL_FNAMES["varnet_knee_mc"]
        state_path = Path(model_name)
        if not state_path.exists():
            download_model(VARNET_FOLDER + model_name, state_path)
        model = _load_model_from_pt(state_path, cascades=cascades)
    else:
        state_path = Path(state_dict_file)
        if state_path.suffix == ".ckpt":
            from fastmri.pl_modules import VarNetModule

            loaded = VarNetModule.load_from_checkpoint(state_path)
            model = loaded.varnet if hasattr(loaded, "varnet") else loaded
        else:
            model = _load_model_from_pt(state_path, cascades=cascades)

    model.eval()
    return model.to(device)


def run_varnet_model(batch, model, device: torch.device):
    crop_size_raw = batch.crop_size
    if isinstance(crop_size_raw, torch.Tensor):
        if crop_size_raw.ndim == 2:
            crop_h = int(crop_size_raw[0, 0].item())
            crop_w = int(crop_size_raw[0, 1].item())
        elif crop_size_raw.ndim == 1:
            crop_h = int(crop_size_raw[0].item())
            crop_w = int(crop_size_raw[1].item())
        else:
            raise ValueError(f"Unexpected crop_size tensor shape: {tuple(crop_size_raw.shape)}")
    else:
        crop_h = int(crop_size_raw[0])
        crop_w = int(crop_size_raw[1])
    crop_size = (crop_h, crop_w)

    masked_kspace = batch.masked_kspace.to(device)
    mask = batch.mask.to(device)
    num_low_frequencies = batch.num_low_frequencies
    if isinstance(num_low_frequencies, torch.Tensor):
        num_low_frequencies = num_low_frequencies.to(device)
    else:
        num_low_frequencies = torch.tensor(num_low_frequencies, device=device)

    output = model(masked_kspace, mask, num_low_frequencies)

    if output.shape[-1] < crop_size[1] or output.shape[-2] < crop_size[0]:
        crop_size = (min(output.shape[-2], crop_size[0]), min(output.shape[-1], crop_size[1]))
    output = T.center_crop(output, crop_size)

    if output.ndim >= 3:
        output = output[0]

    return output.detach().cpu().numpy(), int(batch.slice_num[0]), str(batch.fname[0])


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

        recon = np.stack([arr for _, arr in sorted(outputs[fname], key=lambda x: x[0])], axis=0)

        with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
            fout.create_dataset("reconstruction_rss", data=recon)

            for k, v in fin.attrs.items():
                fout.attrs[k] = v


def run_inference(
    state_dict_file: Path | None,
    data_path: Path,
    output_path: Path,
    device: torch.device,
    cascades: int,
    num_workers: int,
    overwrite: bool,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    model = load_model(state_dict_file=state_dict_file, cascades=cascades, device=device)

    # Input files are already accelerated; do not apply additional masking.
    data_transform = VarNetDataTransformNoMask()
    dataset = SliceDataset(root=data_path, transform=data_transform, challenge="multicoil")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)

    source_index = build_source_index(data_path)
    outputs: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            recon, slice_num, fname = run_varnet_model(batch, model, device)
            outputs[fname].append((slice_num, recon))

    save_rss_only_outputs(
        outputs=outputs,
        source_index=source_index,
        output_path=output_path,
        overwrite=overwrite,
    )

    print(f"Saved {len(outputs)} files to: {output_path}")
    print("Each file contains: reconstruction_rss + attrs.")
    print("No kspace was saved.")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda", type=str, help="Device to run inference on.")
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to checkpoint (.ckpt/.pt). If omitted, downloads knee pretrained .pt.",
    )
    parser.add_argument("--data_path", type=Path, required=True, help="Path to input H5 dataset.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save output H5 files.")
    parser.add_argument("--cascades", type=int, default=8, help="Number of VarNet cascades.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    args = parser.parse_args()

    run_inference(
        state_dict_file=args.state_dict_file,
        data_path=args.data_path,
        output_path=args.output_path,
        device=torch.device(args.device),
        cascades=args.cascades,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
