import os
import pathlib
import sys

from pytorch_lightning.loggers import MLFlowLogger

import dask
import dask.array
import numpy as np
import pytorch_lightning as pl

import torch
import zarr

import datetime
from tempfile import TemporaryDirectory

import mlflow.pytorch
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler, visualize
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    RichProgressBar,
)

import cbh_data_definitions

import cbh_torch_lstm
import cbh_torch_MLP

print("pl ver:", pl.__version__)
print("mlflow ver:", mlflow.__version__)
print("torch ver:", torch.__version__)
print("Python ver:", sys.version_info)

RELOAD_PACKAGES = True
if RELOAD_PACKAGES:
    import importlib
    importlib.reload(cbh_torch_lstm)
    importlib.reload(cbh_torch_MLP)
    importlib.reload(cbh_data_definitions)

root_data_directory = pathlib.Path(os.environ["SCRATCH"]) / "cbh_data"

dev_data_path = root_data_directory / "analysis_ready" / "dev_randomized.zarr"
training_data_path = root_data_directory / "analysis_ready" / "train_randomized.zarr"

mlflow_command_line_run = """
    mlflow server --port 5001 --backend-store-uri sqlite:///mlflowSQLserver.db  --default-artifact-root ./mlflow_artifacts/
"""
alt_mlflow_command_line_run = """
mlflow server --port 5001 --backend-store-uri sqlite:////data/users/hsouth/mlflow/mlflowServer.db  --default-artifact-root /data/users/hsouth/mlflow/mlflow_artifacts --host 0.0.0.0
""" 
mlflow_server_address = 'vld425'
mlflow_server_port = 5001
mlflow_server_uri = f'http://{mlflow_server_address}:{mlflow_server_port:d}'
mlflow_artifact_root = pathlib.Path('/data/users/hsouth/mlflow/mlflow_artifacts')

hparams_for_mlflow = {}

CPU_COUNT = 8
RAM_GB = 64
hparams_for_mlflow['CPU Count'] = CPU_COUNT
hparams_for_mlflow['Compute Memory'] = RAM_GB

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

train_input

LIMIT_DATA = False
LIMIT_DATA_INT = -1
if LIMIT_DATA:
    LIMIT_DATA_INT = 10024
    train_input = train_input[:LIMIT_DATA_INT]
    train_labels = train_labels[:LIMIT_DATA_INT]
    
    dev_input = dev_input[:LIMIT_DATA_INT]
    dev_labels = dev_labels[:LIMIT_DATA_INT]
    
hparams_for_mlflow['Limited sample number'] =  LIMIT_DATA_INT  

seed_everything_int = 42
seed_everything(seed_everything_int)
hparams_for_mlflow['Random seed'] = seed_everything_int

p_l_p_f_m = np.load("./per_level_per_feat_mean.npz")
p_l_p_f_s = np.load("./per_level_per_feat_std.npz")
p_f_m = np.load("./per_feat_mean.npz")
p_f_s = np.load("./per_feat_std.npz")
p_l_p_f_s += 5.0e-12 
p_f_m

mlp_layernum = 3             
model_hyperparameter_dictionary = {
    "LSTM": {
        "input_size": train_input.shape[2],  
        "lstm_layers": 2,
        "lstm_hidden_size": 100,
        "output_size": 100,  
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
model = model_definition_dictionary[model_picked]  
hparams_for_mlflow["Model defined hparams"] = model_hyperparameter_dictionary[model_picked]

epochs = 1
hparams_for_mlflow["Max epochs"] = epochs
collate_fn = cbh_data_definitions.dataloader_collate_with_dask
print("Data chunk size:", train_input.chunksize[0])
print("Factors of chunk: ", [n for n in range(1, train_input.chunksize[0] + 1) if train_input.chunksize[0] % n == 0])
batch_size = 100
hparams_for_mlflow["Batch size"] = batch_size

train_loader, val_loader = None, None

single_proc_workers = True 
if single_proc_workers:
    WORKERS_CPU_COUNT=0
else:
    WORKERS_CPU_COUNT = CPU_COUNT

data_loader_hparam_dict = {
    'batch_size':batch_size,
    'num_workers':WORKERS_CPU_COUNT,
    
    'collate_fn':None, 
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

experiment_name = 'cbh-label-model-runs'

mlflow.set_tracking_uri(mlflow_server_uri)

mlf_exp = None
mlf_exp_id = None
try: 
    print('Creating experiment')
    mlf_exp_id = mlflow.create_experiment(experiment_name)
    mlf_exp = mlflow.get_experiment(mlf_exp_id)
except mlflow.exceptions.RestException:
    mlf_exp = mlflow.get_experiment_by_name(experiment_name)

class MLFlowLogger(pl.loggers.MLFlowLogger): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, model_checkpoint: pl.callbacks.ModelCheckpoint) -> None:
        """
        Called after model checkpoint callback saves a new checkpoint.
        """
        best_chkpt = torch.load(model_checkpoint.best_model_path)
        
        
        checkpoint_for_mlflow = {
            "val loss": float(best_chkpt['callbacks'][list(key for key in list(best_chkpt['callbacks'].keys()) if "ModelCheckpoint" in key)[0]]['current_score']),
            
            "global_step": best_chkpt['global_step'],
            "model_state_dict": best_chkpt['state_dict'],
            "checkpoint": best_chkpt,

        }
        with TemporaryDirectory() as tmpdirname:
            f_name = os.path.join(tmpdirname, f"{datetime.datetime.now()}best_model_checkpoint-step_{best_chkpt['global_step']}.pt")
            torch.save(checkpoint_for_mlflow, f_name)
            mlflow.log_artifact(f_name)

import warnings
warnings.filterwarnings("ignore")

max_time = "02:12:00:00"  

hparams_for_mlflow["Training timeout"] = max_time

timestamp_template = '{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
run_name_template = 'cbh_challenge_{network_name}_' + timestamp_template
current_run_name = run_name_template.format(network_name=model.__class__.__name__,
                                                dt=datetime.datetime.now()
                                               )

with mlflow.start_run(experiment_id=mlf_exp.experiment_id, run_name=current_run_name) as run:

    mlflow.pytorch.autolog()
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_server_uri, run_id=run.info.run_id)

    
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
        'val_check_interval':0.01, 
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
    

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
print_auto_logged_info(run)

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

eg_batch_metrics = {
    "Single batch example validation metrics/Correct samples" : np.count_nonzero(correct),
    "Single batch example validation metrics/Total samples tested" : len(correct),
    "Single batch example validation metrics/Accuracy" : np.count_nonzero(correct) / len(correct) * 100,
}

mlf_logger.log_metrics(eg_batch_metrics)

print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
mlflow.end_run()