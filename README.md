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
  <a href="https://mosaicmri.ai"><img alt="Website" src="https://img.shields.io/badge/Website-mosaicmri.ai-0A66C2?style=for-the-badge&logo=googlechrome&logoColor=white"></a>
  <a href="https://www.mosaicmri.ai/#cite"><img alt="Dataset Paper" src="https://img.shields.io/badge/Dataset%20Paper-Citation%20%2F%20PDF-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://www.mosaicmri.ai/benchmark/"><img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-MosaicMRI-1E8E3E?style=for-the-badge"></a>
  <a href="https://github.com/paularguello07/msk_mri_dataset"><img alt="GitHub" src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white"></a>
</p>

<p align="center">
  <a href="varnet/"><img alt="VarNet Baseline" src="https://img.shields.io/badge/Baseline-VarNet-4C1D95?style=flat-square"></a>
  <a href="data_filtering_for_accelerated_mri/runs/train/"><img alt="U-Net and ViT Baselines" src="https://img.shields.io/badge/Baselines-U--Net%20%2B%20ViT-0F766E?style=flat-square"></a>
  <a href="BENCHMARKS/"><img alt="Benchmark Tools" src="https://img.shields.io/badge/Benchmark%20Tools-BENCHMARKS-374151?style=flat-square"></a>
</p>

This repository contains code for MosaicMRI dataset analysis, VarNet training/inference, and benchmark validation workflows.

## Released Scope

The public release is focused on:

- `varnet/` core training and inference code
- `BENCHMARKS/` reconstruction validation utilities for benchmark:
  - `validate_multicoil_test_reconstructions.ipynb`
  - `validate_ankle_challenge_reconstructions.ipynb`
  - `validate_contrast_challenge_reconstructions.ipynb`
  - `validate_reconstructions_core.py`
- `notebooks/` analysis notebooks:
  - `final_dataset_analysis.ipynb`
  - `final_dataset_category_stats.ipynb`
  - `final_read_msk_dataset.ipynb`

Generated outputs, local runs, and internal tooling are intentionally excluded from release workflows.

## Expected Dataset Layout

Most scripts and notebooks assume a generic layout like:

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

## Environments

The VarNet folder includes a full environment file:

- `varnet/environment.yml`

Sample run scripts use the environment name `mosaic_mri_varnet`.

## Quick Start

### 1. Train VarNet

```bash
python varnet/train_mosaic_mri_varnet.py --help
```

### 2. Run pretrained inference

```bash
python varnet/run_pretrained_varnet_inference.py --help
```
