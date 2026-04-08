# MosaicMRI VarNet

This directory contains the MosaicMRI-focused VarNet training and inference code.

## Attribution

This implementation builds on the original fastMRI VarNet work and codebase:

- fastMRI repository (Meta AI / NYU): https://github.com/facebookresearch/fastMRI
- VarNet paper: Sriram et al., *End-to-End Variational Networks for Accelerated MRI Reconstruction* (MICCAI 2020)

## Main Files

- `train_mosaic_mri_varnet.py`: primary training entry point
- `run_pretrained_varnet_inference.py`: inference script using pretrained checkpoints
- `dataloaders/`: custom data modules used by training
- `runs/sample_run.sh`: minimal sample training launcher
- `runs/inferences/sample_inference.sh`: minimal sample inference launcher

## Expected Data Layout

```text
MosaicMRI/
  multicoil_train/
  multicoil_val/
  multicoil_test/
```

Most loaders expect this naming convention under each dataset root.

## Environment

Use `varnet/environment.yml` and activate `mosaic_mri_varnet`.

```bash
conda env create -f varnet/environment.yml
conda activate mosaic_mri_varnet
```

## Training and Inference

```bash
python train_mosaic_mri_varnet.py --help
python run_pretrained_varnet_inference.py --help
```

Use `runs/sample_run.sh` and `runs/inferences/sample_inference.sh` as templates.

## Dataloaders

`train_mosaic_mri_varnet.py` selects a loader with `--data_loader`.

### `default` (FastMRI built-in)

- Module: `fastmri.pl_modules.FastMriDataModule`
- Use when training/evaluating on one dataset root (`--data_path`)
- Supports standard fastMRI slice/volume sampling flags and optional test path override

### `custom_mix`

- Module: `dataloaders/custom_mix_data_module.py` (`CustomMixDataModule`)
- Mixes two dataset roots (`--data_path1`, `--data_path2`) split-wise
- Optional `--combine_train_val` concatenates train+val for both datasets during training
- Useful for domain mixing (e.g., MosaicMRI + fastMRI)
- Important implementation note: current code sets `volume_sample_rates=[None, 0.6]` in `CombinedSliceDataset`, so dataset2 gets fixed volume sampling behavior regardless of CLI sampling flags

### `categories`

- Module: `dataloaders/custom_category_data_module_clean.py` (`CustomCategoryDataModule`)
- Uses `--category_mapping_path` (filename -> anatomy) and `--category_order`
- Builds category-filtered datasets and concatenates them in the specified order
- Slices are shuffled within each category, but the category block order is preserved
- Useful for anatomy-curriculum or anatomy-order experiments

### `balanced_categories`

- Module: `dataloaders/custom_balanced_category_data_module.py` (`BalancedCategoryDataModule`)
- Requires `--category_mapping_path`, `--category_order`, `--volume_rates`
- Creates one per-category dataset with per-category `volume_sample_rate`, then concatenates and globally mixes slices
- Best option when you want more balanced category contribution during training

### `knees_only`

- Module: `dataloaders/custom_single_category_mix_data_module.py` (`SingleCategoryMixDataModule`)
- Alias in training script for single-category mixing with `category_name="knee"`
- Dataset1 is filtered to KNEE via `--category_mapping_path`; dataset2 is full
- Supports `--val_source`:
  - `mixed`: validation uses mixed dataset1+dataset2
  - `data1`: validation on category-filtered dataset1 only
  - `data2`: validation on dataset2 only
- Good for anatomy-transfer style setups

### `json_train_knee_val`

- Module: `dataloaders/custom_json_train_knee_val_data_module.py` (`JsonTrainKneeValDataModule`)
- Train: explicit JSON-selected slices from `--selector_json_path`, resolved with `--data_path_train_root`
- Val: KNEE-only volumes from `--data_path_val_root` filtered via `--category_mapping_path`
- Designed for workflows like: train on preselected slices, validate on KNEE benchmark subset

## Dataloader Helpers (internal)

- `ListSliceDataset`:
  - File-list-driven slice dataset with fastMRI-compatible sample tuples
  - Used by category/balanced/single-category modules
- `CategoryOrderedSliceDataset`:
  - Builds per-category `ListSliceDataset` blocks in fixed category order
- `JsonSliceDataset`:
  - Lightweight dataset used by JSON-based loaders for explicit selected slices

## Common CLI Patterns

- Single dataset:
  - `--data_loader default --data_path MosaicMRI`
- Two-dataset mixing:
  - `--data_loader custom_mix --data_path1 MosaicMRI --data_path2 <other_root>`
- Category-based:
  - `--data_loader categories --data_path MosaicMRI --category_mapping_path <json> --category_order SPINE KNEE ...`
- JSON-selected train slices + KNEE val:
  - `--data_loader json_train_knee_val --data_path_train_root <root> --data_path_val_root <val_root> --selector_json_path <json> --category_mapping_path <json>`

## Notes

- This release keeps minimal sample launchers in `runs/` for reproducibility.
- Large outputs, logs, and experiment-specific run scripts are intentionally excluded.
