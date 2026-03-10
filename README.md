# MosaicMRI

This repository contains code for MosaicMRI dataset analysis, VarNet training/inference, and benchmark validation notebooks for reconstruction submissions.

## Released Scope

The public release is focused on:

- `varnet/` core training and inference code
- `BENCHMARKS/` reconstruction validation utilities:
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

### 3. Validate reconstructions for benchmark submission

Open one of:

- `BENCHMARKS/validate_multicoil_test_reconstructions.ipynb`
- `BENCHMARKS/validate_ankle_challenge_reconstructions.ipynb`
- `BENCHMARKS/validate_contrast_challenge_reconstructions.ipynb`

Validation workflow in each notebook:

1. Check names, shapes, keys, and size limits.
2. Detect if `kspace` is still present.
3. Optionally export reconstruction-only files (`ismrmrd_header` + `reconstruction_rss`).
4. Offer ZIP only when files pass all checks, `kspace` is absent, and size is within limit.
