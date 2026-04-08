# MosaicMRI

<p align="center">
  <a href="https://mosaicmri.ai" target="_blank" rel="noopener noreferrer">
    <img src="title.png" alt="MosaicMRI" width="100%" />
  </a>
</p>

<p align="center">
  <strong>Project website:</strong> <a href="https://mosaicmri.ai">mosaicmri.ai</a>
</p>

<p align="center">
  <a href="https://mosaicmri.ai"><img alt="website" src="https://img.shields.io/badge/website-mosaicmri.ai-2563EB?style=flat-square"></a>
  <a href="https://www.mosaicmri.ai/#cite"><img alt="paper" src="https://img.shields.io/badge/paper-citation%20%2F%20pdf-6B7280?style=flat-square"></a>
  <a href="https://www.mosaicmri.ai/benchmark/"><img alt="benchmark" src="https://img.shields.io/badge/benchmark-mosaicmri-FACC15?style=flat-square"></a>
</p>

MosaicMRI is a large-scale raw musculoskeletal MRI dataset and benchmark for accelerated reconstruction under substantial clinical variability in anatomy, contrast, orientation, and coil configuration. This repository provides reference training and inference code, benchmark validation utilities, and reproducibility assets used in MosaicMRI experiments.

## Repository Details

- `varnet/`: primary VarNet training and inference pipeline for MosaicMRI.
- `data_filtering_for_accelerated_mri/`: U-Net/ViT baselines and data-filtering workflows used in extended experiments.
- `BENCHMARKS/`: benchmark validation notebooks and shared validation utilities.
- `notebooks/`: dataset analysis notebooks for summary statistics and exploratory reporting.

Large generated artifacts such as logs, checkpoints, cached tensors, and local outputs are intentionally excluded from version control.

## Expected Dataset Structure

Most scripts assume the following directory layout:

```text
MosaicMRI/
  multicoil_train/
  multicoil_val/
  multicoil_test/
  anatomy_generalization_challenge/
    ankle/
  contrast_generalization_challenge/
    T1_FS/
```

## Environment

The reference environment is provided in `varnet/environment.yml`.

```bash
conda env create -f varnet/environment.yml
conda activate mosaic_mri_varnet
```

## Quick Start

Reference CLI entry points:

```bash
python varnet/train_mosaic_mri_varnet.py --help
python varnet/run_pretrained_varnet_inference.py --help
```

Benchmark validation tools are available in `BENCHMARKS/`, including:

- `validate_multicoil_test_reconstructions.ipynb`
- `validate_ankle_challenge_reconstructions.ipynb`
- `validate_contrast_challenge_reconstructions.ipynb`
- `validate_reconstructions_core.py`

## Citation and Attribution

If you use MosaicMRI in academic work, please cite the MosaicMRI dataset paper (see the citation section at https://mosaicmri.ai/#cite). VarNet code in this repository builds on the original fastMRI/VarNet framework and should be cited accordingly.
