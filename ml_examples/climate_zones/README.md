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

## Climate Zones PyTorch Training CLI

In addition to the notebooks that make up the tutorial, there is also python script version, to demonstrate how the training can be run as a script. There is also a batch wraper script to demomstrate how it can be submit to a  compute cluster through the slurm batch scheduler. In this example it is configured for the [ORCHID GPU cluster on the JASMIN compute infrastructure](https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/).

### Files

- `ClimateZones_Training_Torch.py`: refactored Python script with `main()` and CLI arguments.
- `run_climate_zones_torch.slurm`: batch script for `sbatch` that activates `dscop-pytorch`, starts MLflow, and runs training.

### Python CLI arguments

```bash
python ClimateZones_Training_Torch.py \
  --config-path /path/to/config.json \
  --learning-rate 0.001 \
  --batch-size 16 \
  --num-epochs 10
```

### Submit to SLURM

Edit environment variables at the top of `run_climate_zones_torch.slurm`, then submit:

```bash
sbatch run_climate_zones_torch.slurm
```


