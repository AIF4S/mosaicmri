# MosaicMRI VarNet

This directory contains the MosaicMRI-focused VarNet training and inference code.

## Main Files

- `train_mosaic_mri_varnet.py`: primary training entry point
- `run_pretrained_varnet_inference.py`: inference script using pretrained checkpoints
- `dataloaders/`: dataset/data-module implementations
- `runs/sample_run.sh`: minimal sample training launcher
- `runs/inferences/sample_inference.sh`: minimal sample inference launcher

## Expected Data Layout

```text
MosaicMRI/
  multicoil_train/
  multicoil_val/
  multicoil_test/
```

## Environment

Use `varnet/environment.yml` and activate an environment named `mosaic_mri_varnet`.

Example:

```bash
conda env create -f varnet/environment.yml
conda activate mosaic_mri_varnet
```

If your local env name differs, update the sample scripts accordingly.

## Training

```bash
python train_mosaic_mri_varnet.py --help
```

Use `runs/sample_run.sh` as a template for your local paths and GPU setup.

## Inference

```bash
python run_pretrained_varnet_inference.py --help
```

Use `runs/inferences/sample_inference.sh` as a template for local inference runs.

## Notes

- This release keeps only minimal sample launchers in `runs/` for reproducibility.
- Large outputs, logs, and experiment-specific launch scripts are excluded from release workflows.
