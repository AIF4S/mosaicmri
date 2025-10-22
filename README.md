# MSK MRI Dataset

## Environments

There are two separate environments:

1) Notebook environment (name: `mri`) – for running the notebooks and twixtools
- File: `./environment.yml`

2) Converter environment (name: `ismrmrd_env`) – for the CLI converter `siemens_to_ismrmrd`
- File: `./dat_2_h5/environment.yml`

### Create envs

- Install mamba into base once (faster, recommended):

```bash
conda install -n base -c conda-forge mamba
```

- Create the converter env (needed for DAT→H5):

```bash
cd dat_2_h5
mamba env create -f environment.yml
# then activate when converting
conda activate ismrmrd_env
```

## Convert .dat → .h5 (ISMRMRD)

Script: `dat_2_h5/dat2mrdv2.sh`

- Uses the parameter map `parameter_maps/IsmrmrdParameterMap_NewData.xml` (+ XSL) by default.
- Other parameter maps under `parameter_maps/examples/` are only examples.
- Input folder = `data/datasets/msk_mri`
- Output layout (created automatically):
  - `data/datasets/msk_mri_h5/h5` – converted scans
  - `data/datasets/msk_mri_h5/noise_v3/` – first noise measurement when present