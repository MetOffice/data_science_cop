#!/usr/bin/env python
# coding: utf-8

# # Training a neural network in PyTorch
# 
# This notebook demonstrates training a classifier in PyTorch. In order to train a model, there are three main steps:
# 1. Initialize the data and model
# 1. Run the training loop
# 1. Ensure training success with some small evaluation
# 
# Training a model can be as simple as calling a model.fit() function on some modified data, however there are a number of desired qualities when it comes to model training that will be presented in this notebook: Reproducibility of the model, tracking of model training, and helping model generalization through data randomization

# In[1]:


#imports

# file handling
import os
import pathlib
import sys

from pytorch_lightning.loggers import MLFlowLogger

import dask
import dask.array

# math operators
import numpy as np
import pytorch_lightning as pl

# ml
import torch
import zarr

import datetime
from tempfile import TemporaryDirectory

# training helpers
import mlflow.pytorch
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler, visualize
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    RichProgressBar,
)  # this progress bar works through jupyterHub on spice

# defined in directory (model related definitions)
import cbh_data_definitions
# cbh_data_definitions.register_cache()
import cbh_torch_lstm
import cbh_torch_MLP

print("pl ver:", pl.__version__)
print("mlflow ver:", mlflow.__version__)
print("torch ver:", torch.__version__)
print("Python ver:", sys.version_info)


# In[2]:


# Reload packages to allow for ease of notebook updates
RELOAD_PACKAGES = True
if RELOAD_PACKAGES:
    import importlib
    importlib.reload(cbh_torch_lstm)
    importlib.reload(cbh_torch_MLP)
    importlib.reload(cbh_data_definitions)


# In[3]:


# intialize some settings: mlflow, data directory, resources
root_data_directory = pathlib.Path(os.environ["SCRATCH"]) / "cbh_data"

dev_data_path = root_data_directory / "analysis_ready" / "dev_randomized.zarr"
training_data_path = root_data_directory / "analysis_ready" / "train_randomized.zarr"

mlflow_command_line_run = """
    mlflow server --port 5001 --backend-store-uri sqlite:///mlflowSQLserver.db  --default-artifact-root ./mlflow_artifacts/
"""
alt_mlflow_command_line_run = """
mlflow server --port 5001 --backend-store-uri sqlite:////data/users/hsouth/mlflow/mlflowServer.db  --default-artifact-root /data/users/hsouth/mlflow/mlflow_artifacts --host 0.0.0.0
""" # This command is used, as /data/ will not run out of storage quota like /home/ will with many runs
mlflow_server_address = 'vld425'
mlflow_server_port = 5001
mlflow_server_uri = f'http://{mlflow_server_address}:{mlflow_server_port:d}'
mlflow_artifact_root = pathlib.Path('/data/users/hsouth/mlflow/mlflow_artifacts')

hparams_for_mlflow = {}

CPU_COUNT = 8
RAM_GB = 64
hparams_for_mlflow['CPU Count'] = CPU_COUNT
hparams_for_mlflow['Compute Memory'] = RAM_GB


# In[4]:


#initialize data
(
    train_input,
    train_labels,
    _,
) = cbh_data_definitions.load_data_from_zarr(training_data_path)

(
    dev_input, 
    dev_labels, 
    _
) = cbh_data_definitions.load_data_from_zarr(dev_data_path)

# the cloud volume is not needed for the task, so isn't saved on the load
# show a chunk
train_input


# In[5]:


# settings for data limiting (used for development e.g. faster training runs)
LIMIT_DATA = False
LIMIT_DATA_INT = -1
if LIMIT_DATA:
    LIMIT_DATA_INT = 10024
    train_input = train_input[:LIMIT_DATA_INT]
    train_labels = train_labels[:LIMIT_DATA_INT]
    # train_cloud_volume = train_cloud_volume[:LIMIT_DATA_INT]
    dev_input = dev_input[:LIMIT_DATA_INT]
    dev_labels = dev_labels[:LIMIT_DATA_INT]
    # dev_cloud_volume = dev_cloud_volume[:LIMIT_DATA_INT]
hparams_for_mlflow['Limited sample number'] =  LIMIT_DATA_INT  


# In order to make machine learning reproducible, hyperparameters will be logged using MLflow so that models can be redefined exactly as they were, and a seed for our random functions e.g. random shuffling of data: The random seed ensures that the same "random" shuffle of the data is performed each time.

# In[6]:


# reproducibility with seed everything
seed_everything_int = 42
seed_everything(seed_everything_int)
hparams_for_mlflow['Random seed'] = seed_everything_int


# ## Perform the network initialization and training
# 
# for each different model architecture, various different hyperparameters must be defined. For example in the case of an LSTM, bi-directionality is a simple to implement extension to the model type as this is a parameter of the layer definition in pytorch so can be included in the LSTM hyperparameter definition, there is no concept of input sequence direction in a simple MLP, so this hyperparameter doesn't apply 
# 
# training related hyperparameters are also defined. It is important to track exactly how much training is performed (epoch and step), the batch size, the learning rate and the optimizer.

# In[7]:


p_l_p_f_m = np.load("./per_level_per_feat_mean.npz")
p_l_p_f_s = np.load("./per_level_per_feat_std.npz")
p_f_m = np.load("./per_feat_mean.npz")
p_f_s = np.load("./per_feat_std.npz")
p_l_p_f_s += 5.0e-12 # prevent divide by 0
p_f_m


# In[8]:


# define model and hyperparameters
mlp_layernum = 3             
model_hyperparameter_dictionary = {
    "LSTM": {
        "input_size": train_input.shape[2],  # input size is the cell input (feat dim)
        "lstm_layers": 2,
        "lstm_hidden_size": 100,
        "output_size": 100,  # for each height layer, predict one value for cloud base prob
        "height_dimension": train_input.shape[1],
        "embed_size": 20,
        "BILSTM": True,
        "batch_first": True,
        "skip_connection":True,
        "backward_lstm_differing_transitions":False,
        "lr": 0.001460,
        "norm_mat_mean":torch.from_numpy(p_l_p_f_m.astype(np.float32)), 
        "norm_mat_std":torch.from_numpy(p_l_p_f_s.astype(np.float32)),
        "norm_method":"p_l_p_f",
        "linear_instead_of_conv_cap":True,
    },
    "MLP": {
        "input_size": train_input.shape[2] * train_input.shape[1],
        "ff_nodes": mlp_layernum * [256],
        "output_size": train_input.shape[1],
        "lr": 1.0e-3,
        "activation": "relu",
        "layer_num": mlp_layernum,
        "norm_mat_mean":torch.from_numpy(p_l_p_f_m.astype(np.float32)), 
        "norm_mat_std":torch.from_numpy(p_l_p_f_s.astype(np.float32)),
        "norm_method":"p_l_p_f",
    },
}

model_definition_dictionary = {
    "LSTM": cbh_torch_lstm.CloudBaseLSTM(**model_hyperparameter_dictionary["LSTM"]),
    "MLP": cbh_torch_MLP.CloudBaseMLP(**model_hyperparameter_dictionary["MLP"]),
}
model_picked = "LSTM"
model = model_definition_dictionary[model_picked]  # pick a model
hparams_for_mlflow["Model defined hparams"] = model_hyperparameter_dictionary[model_picked]


# define training related hyperparameters

epochs = 1
hparams_for_mlflow["Max epochs"] = epochs
collate_fn = cbh_data_definitions.dataloader_collate_with_dask
print("Data chunk size:", train_input.chunksize[0])
print("Factors of chunk: ", [n for n in range(1, train_input.chunksize[0] + 1) if train_input.chunksize[0] % n == 0])
batch_size = 100
hparams_for_mlflow["Batch size"] = batch_size


# The pytorch dataloader is also initialized, usually the dataloader does not change the outcome of ML training, however in our case there are multiple strategies for batch selection implemented, so tracking these settings are important.

# In[9]:


train_loader, val_loader = None, None

single_proc_workers = True # False causes crashes in some cases or in the case of 1 chunk at a time, makes data access slower
if single_proc_workers:
    WORKERS_CPU_COUNT=0
else:
    WORKERS_CPU_COUNT = CPU_COUNT

data_loader_hparam_dict = {
    'batch_size':batch_size,
    'num_workers':WORKERS_CPU_COUNT,
    # 'pin_memory':False,
    'collate_fn':None, # using 1chunk method
    'thread_count_for_dask':CPU_COUNT,
    'method':'1chunk',
}
shuffle_training_data=False,

datamodule = cbh_data_definitions.CBH_DataModule(
        train_input, train_labels,
        dev_input, dev_labels,
        **data_loader_hparam_dict,
        randomize_chunkwise = True,
        
    )
data_loader_hparam_dict['shuffle_training_data']=shuffle_training_data
hparams_for_mlflow['data loader hparams'] = data_loader_hparam_dict


# Next we initialize an MLflow experiement, MLflow serves to share model training information in collaboration, allow for monitoring of current training processes, and tracking of hyperparameters + models + metrics for reproducing saving and loading training experiements. The MLflow server is launched from the command line, and connected to through the below cell

# In[10]:


experiment_name = 'cbh-label-model-runs'
# experiment_name = 'test-setup-for-model-runs'

# torch.set_num_threads(CPU_COUNT)

mlflow.set_tracking_uri(mlflow_server_uri)
# make vars global
mlf_exp = None
mlf_exp_id = None
try: 
    print('Creating experiment')
    mlf_exp_id = mlflow.create_experiment(experiment_name)
    mlf_exp = mlflow.get_experiment(mlf_exp_id)
except mlflow.exceptions.RestException:
    mlf_exp = mlflow.get_experiment_by_name(experiment_name)


# Autologging of the experiement did not work in this implementation, the best guess as to why is due to dasks lock on the data which prevents saving model weights, or displaying model attributes in some cases which autologging uses. To circumvent this, a custom checkpointing function is implemented into the PyTorch-Lighning's implementation for logging to MLflow

# In[11]:


class MLFlowLogger(pl.loggers.MLFlowLogger): #overwrite mlflogger
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, model_checkpoint: pl.callbacks.ModelCheckpoint) -> None:
        """
        Called after model checkpoint callback saves a new checkpoint.
        """
        best_chkpt = torch.load(model_checkpoint.best_model_path)
        # print(best_chkpt)
        # print(best_chkpt['callbacks'])
        checkpoint_for_mlflow = {
            "val loss": float(best_chkpt['callbacks'][list(key for key in list(best_chkpt['callbacks'].keys()) if "ModelCheckpoint" in key)[0]]['current_score']),
            # "train loss at step-1": list(train_loss_metric.value for train_loss_metric in self._mlflow_client.get_metric_history(run.info.run_id, "Train loss") if (int(train_loss_metric.step) == int(best_chkpt['global_step']-1)))[0],
            "global_step": best_chkpt['global_step'],
            "model_state_dict": best_chkpt['state_dict'],
            "checkpoint": best_chkpt,

        }
        with TemporaryDirectory() as tmpdirname:
            f_name = os.path.join(tmpdirname, f"{datetime.datetime.now()}best_model_checkpoint-step_{best_chkpt['global_step']}.pt")
            torch.save(checkpoint_for_mlflow, f_name)
            mlflow.log_artifact(f_name)


# Here the training is run, but not before some more model settings:timeout for training, naming of the MLflow experiement, defining when to validate, how to report progress in the notebook, and defining when to save a model. After training is run, another copy of the model is saved

# In[12]:


import warnings
warnings.filterwarnings("ignore")


# In[13]:


max_time = "02:12:00:00"  # dd:hh:mm:ss

hparams_for_mlflow["Training timeout"] = max_time

timestamp_template = '{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
run_name_template = 'cbh_challenge_{network_name}_' + timestamp_template
current_run_name = run_name_template.format(network_name=model.__class__.__name__,
                                                dt=datetime.datetime.now()
                                               )

# with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
with mlflow.start_run(experiment_id=mlf_exp.experiment_id, run_name=current_run_name) as run:

    mlflow.pytorch.autolog()
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_server_uri, run_id=run.info.run_id)


    # define trainer
    time_for_checkpoint = datetime.timedelta(minutes=15)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        train_time_interval=time_for_checkpoint,
        dirpath=run.info.artifact_uri,
        monitor="Val loss",
        save_on_train_epoch_end=False,
        mode="min"
    )
    callbacks = [checkpoint_callback, RichProgressBar()]
    trainer_hparams = {
        'max_epochs':epochs,
        'deterministic':True,
        'val_check_interval':0.01, # val every percentage of the data
        'devices':"auto",
        'accelerator':"auto",
        'max_time':max_time,
        'replace_sampler_ddp':False,
        'enable_checkpointing':True,
        'strategy':None,
        'callbacks':callbacks,
        'logger':mlf_logger,
    }
    hparams_for_mlflow["Trainer hparams"] = trainer_hparams
    mlf_logger.log_hyperparams(hparams_for_mlflow)
    trainer = pl.Trainer(
        **trainer_hparams
    )
    
    
    trainer.fit(model=model, datamodule=datamodule)
    path_to_save = '{dt.year:04d}{dt.month:02d}{dt.day:02d}-{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'.format(dt=datetime.datetime.now())
    trainer.save_checkpoint(filepath=run.info.artifact_uri + f'/post_epoch_modelchkpt_{path_to_save}')
    with TemporaryDirectory() as tmpdirname:
            f_name = os.path.join(tmpdirname, f"{run.info.run_id}-post_epoch_checkpoint_logged.pt")
            trainer.save_checkpoint(filepath=f_name)
            mlflow.log_artifact(f_name)
print("Ended run", run.info.run_id)
    # print(visualize([prof, rprof, cprof], filename='profile_loop.html', save=True))


# ## Display and evaluate results
# 
# After training, explore some of the results from our model. tracking of experiment progress can be done during training, but it is a good idea to get an idea of final performance, or quickly investigate aspects of the model which cannot be monitored during training such as class imbalance of predictions. More ensurement of properties of the training can also be performed in this section, such as recalling MLflow variables in order to double check correct logging functionality.

# In[14]:


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
print_auto_logged_info(run)


# In[15]:


# test model functionality
example_batch = next(iter(datamodule.train_dataloader()))
inputs = example_batch[0]
try:
    preds, _ = model(inputs, heights)
except:
    if model_picked == "LSTM":
        inputs = example_batch[0]
    else:     
        print(example_batch[0].shape, "inp pre-flat")
        inputs = torch.flatten(example_batch[0], start_dim=1)
    preds = model(inputs)
print(preds.shape, "prediction output")
pred_label = np.argmax(preds.detach().numpy(), axis=1)
print(pred_label.shape, "prediction label shape")
targs = example_batch[1]
targs = np.array(targs)
print(targs.shape, "targ shape")
correct = targs == pred_label
print("Correct samples:", np.count_nonzero(correct))
print("Total samples tested:", len(correct))
print("Accuracy:", (np.count_nonzero(correct) / len(correct) * 100), "%")
print(
    "Model predictions binned: (Class labels), (Counts):",
    np.unique(pred_label, return_counts=True),
)
print(
    "Model targets binned: (Class labels), (Counts):",
    np.unique(targs, return_counts=True),
)
eg_batch_metrics = {
    "Correct samples" : np.count_nonzero(correct),
    "Total samples tested" : len(correct),
    "Accuracy" : (np.count_nonzero(correct) / len(correct) * 100),

    "Model predictions binned: (Class labels), (Counts)"
     : str(np.unique(pred_label, return_counts=True)),

}


# In[16]:


eg_batch_metrics = {
    "Single batch example validation metrics/Correct samples" : np.count_nonzero(correct),
    "Single batch example validation metrics/Total samples tested" : len(correct),
    "Single batch example validation metrics/Accuracy" : np.count_nonzero(correct) / len(correct) * 100,
}


# In[17]:


mlf_logger.log_metrics(eg_batch_metrics)


# In[18]:


# display mlflow output
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
mlflow.end_run()


# In[ ]:




