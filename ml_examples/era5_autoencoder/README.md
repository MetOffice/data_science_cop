# ERA5 Autoencoder Tutorial

## Overview
This tutorial walks through preparing low-resolution ERA5/WeatherBench pressure-level data, training a convolutional autoencoder, and evaluating reconstruction quality. The core learning goal is to understand an end-to-end geospatial ML workflow: dataset preprocessing, model training, experiment logging, and diagnostics. It is set up for both interactive notebook-based exploration and batch execution on JASMIN via SLURM. By the end, you should be able to reproduce a full training run, inspect outputs/checkpoints, and compare model quality using quantitative and visual evaluation.

## File Guide
- `config.json`: Shared configuration for platform-dependent data roots and dataset naming/templates used by the tutorial scripts.
- `era5_autoencoder_data_prep.py`: CLI data-prep script that reads WeatherBench NetCDF files, computes normalization statistics, and writes original/normalized ERA5 data to Zarr stores.
- `ERA5_data_prep.ipynb`: Notebook version of the data preparation workflow for interactive, step-by-step exploration.
- `train_era5_autoencoder.py`: Main training/evaluation CLI module containing the PyTorch dataset class, autoencoder model, training loop, and optional MLflow integration.
- `era5_autoencoder_train.ipynb`: Notebook version of model training with explanatory cells and iterative experimentation workflow.
- `era5_autoencoder_evaluation.ipynb`: Notebook focused on post-training inspection, inference checks, and evaluation/diagnostic plotting.
- `run_era5_data_prep.sh`: SLURM batch job script that activates a CPU conda env and runs the data-prep pipeline.
- `run_era5_ae_train.sh`: SLURM batch job script that activates a GPU conda env, stages normalized data to local scratch, and runs autoencoder training.
- `submit_era5_data_prep.sh`: Convenience wrapper that submits the data-prep batch job via `sbatch` with timestamped log paths.
- `submit_era5_ae.sh`: Convenience wrapper that submits the training batch job via `sbatch` with timestamped log paths.

## Environments
- **Data prep (CPU):** `/gws/nopw/j04/mohc_shared/dscop/conda_envs/ai4c_hack_cli_cpu`
- **Training (GPU):** `/gws/ssde/j25a/mmh_storage/ai4c_conda/ai4c_cli_gpu`

If you prefer named envs instead of absolute env paths, adapt the activation line to your local setup (for example, `conda activate <env_name>`).

## How To Run

### 1) Run data preparation directly
From this directory:

```bash
conda activate /gws/nopw/j04/mohc_shared/dscop/conda_envs/ai4c_hack_cli_cpu
python era5_autoencoder_data_prep.py \
  --start-year 1980 \
  --end-year 2016 \
  --data-out-dir /gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/mlready \
  --config config.json
```

### 2) Run training directly
From this directory:

```bash
conda activate /gws/ssde/j25a/mmh_storage/ai4c_conda/ai4c_cli_gpu
python train_era5_autoencoder.py \
  --config-path config.json \
  --model-out-dir /gws/ssde/j25a/mmh_storage/user/shaddad/experiments/era5_autoencoder \
  --batch-size 16 \
  --num-epochs 10 \
  --learning-rate 0.001 \
  --data-dir /gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/mlready/norm
```

Optional MLflow arguments (if you have a running server):

```bash
python train_era5_autoencoder.py \
  --config-path config.json \
  --model-out-dir /path/to/experiments \
  --batch-size 16 \
  --num-epochs 10 \
  --learning-rate 0.001 \
  --data-dir /path/to/norm \
  --mlflow-url http://localhost \
  --mlflow-port 4455
```

### 3) Submit via SLURM (recommended on JASMIN)

#### Data prep
```bash
bash submit_era5_data_prep.sh
```

#### Training
```bash
bash submit_era5_ae.sh
```

You can also submit the run scripts directly:

```bash
sbatch run_era5_data_prep.sh
sbatch run_era5_ae_train.sh
```

## Notes
- The existing SLURM scripts include site-specific paths (`cd`, log directories, and data locations); update these before first use in another account/project area.
- In `run_era5_ae_train.sh`, the Python entrypoint is currently `ERA5_autoencoder.py`; in this folder the training CLI is `train_era5_autoencoder.py`, so adjust that line if needed.
- `submit_era5_data_prep.sh` currently submits `util/run_era5_data_prep.sh`; if running from this directory, you likely want `run_era5_data_prep.sh`.

