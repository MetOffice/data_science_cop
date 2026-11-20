# Climate Zones ML tutorial

![climate_zones](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/K%C3%B6ppen-Geiger_Climate_Classification_Map_%281980%E2%80%932016%29_no_borders.png/1920px-K%C3%B6ppen-Geiger_Climate_Classification_Map_%281980%E2%80%932016%29_no_borders.png) 
The climate of different areas of the world is classified into different types, based on the Koppen-Geiger scheme. As the climate changes, some areas may change the classification of their climate 
In this series of tutorial notebooks, we willlearn how to build a machine learning pipeline by building a model to predict the climate zone of a location from key climate means and indicators. We will look at how climate models predict shifts in the classification of climate zones as the climate changes in the future.

# Tutorial Pathway

- [Data Exploration](ClimateZones_DataExploration.ipynb)
- [Building a Training Pipeline](ClimateZones_TrainingPipeline.ipynb)
- [Predictions and Visualisation](ClimateZones_InferenceVisualisation.ipynb)
- [Model Evaluation](ClimateZones_Evaluation.ipynb)
- [Training in PyTorch](ClimateZones_Training_Torch.ipynb)

Additional notebooks - demonstrating things not part of the main learning pathway:
- [Data Preparation](ClimateZones_DataPrep.ipynb)

# Running this tutorial

This tutorial has been tested on the scitools ml community environment, which can be accessed as follows:
- Met Office:
  - From the command line - Load the module from the commend line `module load scitools/community/ml`.
  - In Jupyterhub - select the `ml-0.1.0 Python` kernel for the each of the notebooks.
- Other platforms: Create a conda environment based on the [scitools ml yaml file](https://github.com/MetOffice/ssstack/blob/main/environments/ml-0.1.0.yml)

# Climate Zones CLI

In addition to the notebooks that make up the tutorial, there are also python script versions, to demonstrate how the training can be run as a script. There is also a batch wrapper script to demonstrate how it can be submitted to a  compute cluster through the slurm batch scheduler, or run as part of a cylc workflow. 
In this example it is configured for the [ORCHID GPU cluster on the JASMIN compute infrastructure](https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/) or on SPICE.

### Files

- Python Scripts
  - `climatezones_data_prep.py`: Python script for preparing the training data from the downloaded climate zones netcdf files.
  - `climatezones_training_torch.py`: refactored Python script with `main()` and CLI arguments.
  - `climatezones_evaluation.py`: Python script for evaluating the output of the training script.
- Batch Scripts
- `run_climatezones_data_prep.sh`: batch script for `sbatch` that activates takes the downloaded climate zones netcdf files and creates the tabular ML-ready data for training.
- `run_climatezones_training.sh`: batch script for `sbatch` that starts MLflow, and runs training.
- `run_climatezones_evaluation.sh`: batch script for `sbatch` that evaluates the output of the training script.
- Cylc Workflow
  - `workflow/flow.cylc`: workflow for running the training script as a cylc workflow.
  - `workflow/flow.opt`: workflow options for running the cylc workflow.

### Python CLI arguments

```bash
python climatezones_training_torch.py \
  --config-path /path/to/config.json \
  --learning-rate 0.001 \
  --batch-size 16 \
  --num-epochs 10
```

### Submit to SLURM
Edit environment variables at the top of `run_climatezones_training.sh`, then submit:

```bash
sbatch run_climatezones_training.sh /path/to/repo /path/tyo/log/dir /path/to/input/dir/ /pat5h/to/data/output/dir $CONDA_ENV_NAME
```
With suitable arguments given to the sbatch command, it can be run on different SLURM platforms, for example SPICE or JASMIN.

### Cylc workflow
A [cylc workflow](https://cylc.github.io/cylc-doc/latest/html/) orchestrates the running of batch jobs in the correct order and with the correct configuration.

The configuration is done the `flow.opt` file, which is supplied as an argument to the cylc commands. The cylc workflow can be run as follows:

```bash
cd workflow
cylc vip --workflow-name my_training_demo --set-file flow.opt   
``` 



