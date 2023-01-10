import cbh_data_definitions
import pathlib
import os

import optuna
import pytorch_lightning as pl
import torch
import mlflow
from ray import tune
import ray
import ray.tune
from tempfile import TemporaryDirectory
import ray.tune.search
import ray.tune.search.optuna
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.search import ConcurrencyLimiter
from pytorch_lightning.callbacks import TQDMProgressBar
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import datetime
import cbh_torch_MLP
import cbh_torch_lstm

root_data_directory = pathlib.Path(os.environ["SCRATCH"]) / "cbh_data"

dev_data_path = root_data_directory / "analysis_ready" / "dev_randomized.zarr"
training_data_path = root_data_directory / "analysis_ready" / "train_randomized.zarr"

mlflow_command_line_run = """
    mlflow server --port 5001 --backend-store-uri sqlite:///mlflowSQLserver.db  --default-artifact-root ./mlflow_artifacts/
"""
mlflow_server_address = 'vld425'
mlflow_server_port = 5001
mlflow_server_uri = f'http://{mlflow_server_address}:{mlflow_server_port:d}'
mlflow_artifact_root = pathlib.Path('./mlflow_artifacts/')

hparams_for_mlflow = {}

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

factors_of_chunk = [n for n in range(1, train_input.chunksize[0] + 1) if train_input.chunksize[0] % n == 0]
print("Factors of chunk: ", factors_of_chunk)
hparams_for_mlflow['Limited sample number'] =  -1

experiment_name = 'cbh-hparam-tuning'
CPU_COUNT = 20
RAM_GB = 150
hparams_for_mlflow['CPU Count'] = CPU_COUNT
hparams_for_mlflow['Compute Memory'] = RAM_GB
thread_count_for_dask = CPU_COUNT
dataset_method = '1chunk'
randomize_chunkwise_1chunk = False
shuffle_train_data = False
collate_fn = None 
num_workers_dataloader = 0 
global_trail_number = 0
max_time_for_trial = "00:11:50:00"  
hparams_for_mlflow["Training timeout"] = max_time_for_trial

max_node_num_exclusive = 513
max_layers = 12
factors_for_hparam_choice = [factor for factor in factors_of_chunk if (factor<3300 and factor>3)]
mlp_search_space = {
    "epoch": 1,
    "lr": tune.quniform(0.001, 0.01, 0.0005),
    "data_limit": 4,
    "batch_size": tune.choice(factors_for_hparam_choice),
    "arch_name":"MLP",
    "hidden_layers":tune.randint(1,max_layers),
    "activation":tune.choice(["relu", "tanh"]),
    "input_size":(train_input.shape[2] * train_input.shape[1]),
    "output_size": train_input.shape[1],
    "deterministic":False,
    "chkpt_time":datetime.timedelta(minutes=15),
    "max_time":max_time_for_trial
}
lstm_search_space = {
    "epoch": 1,
    "lr": tune.quniform(0.0001, 0.005, 0.000005),
    "data_limit": 4,
    "batch_size": tune.choice(factors_for_hparam_choice),
    "arch_name":"LSTM",
    "lstm_layers":tune.randint(1,max_layers),
    "input_size": train_input.shape[2],
    "lstm_output_size": tune.randint(1,int(max_node_num_exclusive/(4))),
    "deterministic":False,
    "chkpt_time":datetime.timedelta(minutes=15),
    "max_time":max_time_for_trial,
    "lstm_nodesize":tune.randint(1,int(max_node_num_exclusive/8)),
    "height_dimension": train_input.shape[1],
    "embed_size": tune.randint(0,26),
    "BILSTM": tune.choice([False, True]),
    "backward_lstm_differing_transitions": tune.choice([False, True]),
    "batch_first":True, 
    "skip_connection": tune.choice([False, True]),
    "norm_method": tune.choice(["p_l_p_f", 'p_f', 'layer_relative']),
    "linear_instead_of_conv_cap": tune.choice([False, True]),
    "conv_cap_window":tune.randint(1,int(train_input.shape[1]/2)),
}
p_l_p_f_m = np.load("./per_level_per_feat_mean.npz")
p_l_p_f_s = np.load("./per_level_per_feat_std.npz")
p_f_m = np.load("./per_feat_mean.npz")
p_f_s = np.load("./per_feat_std.npz")
p_l_p_f_s += 5.0e-12 
layer_pattern = 'layer_node_number_{layer_num}_div_8'
for layer_num in range(max_layers):
    mlp_search_space[layer_pattern.format(layer_num=layer_num)] = tune.randint(1,int(max_node_num_exclusive/8))
print(mlp_search_space)

class MLFlowLogger(pl.loggers.MLFlowLogger): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

verbose_objective = False

def objective(ray_config):
    
    warnings.filterwarnings("ignore")
    
    mlflow.set_tracking_uri(mlflow_server_uri)
    
    mlf_exp = None
    mlf_exp_id = None
    try: 
        if verbose_objective: print('Creating experiment')
        mlf_exp_id = mlflow.create_experiment(experiment_name)
        mlf_exp = mlflow.get_experiment(mlf_exp_id)
    except mlflow.exceptions.RestException as e:
        if verbose_objective: print("Caught")
        if False:
            print(e)
        mlf_exp = mlflow.get_experiment_by_name(experiment_name)
    if verbose_objective: print("Success")
    
    datamodule = cbh_data_definitions.CBH_DataModule(
        train_input, train_labels,
        dev_input, dev_labels,
        thread_count_for_dask,
        ray_config['batch_size'],
        num_workers = num_workers_dataloader,
        collate_fn = collate_fn,
        shuffle = shuffle_train_data,
        randomize_chunkwise = randomize_chunkwise_1chunk,
        method=dataset_method,
        val_batch_size=6400,
        data_limit=ray_config['data_limit']
    )
    
    if ray_config['arch_name'] == "MLP":
        ff_nodes_strings = []
        for key in ray_config:
            if key.startswith("layer_node_number_"):
                ff_nodes_strings.append(key)
        ff_nodes_strings = sorted(ff_nodes_strings)
        ff_nodes = [(8*ray_config[ff_node_num]) for ff_node_num in ff_nodes_strings]
        if verbose_objective: print(ray_config['hidden_layers'])
        if verbose_objective: print(ff_nodes)
        model = cbh_torch_MLP.CloudBaseMLP(
            ray_config['input_size'],
            ff_nodes,
            ray_config['output_size'],
            ray_config['hidden_layers'],
            ray_config['activation'],
            ray_config['lr'],
        )
        
    elif ray_config['arch_name'] == "LSTM":
        if ray_config['norm_method'] == "p_l_p_f":
            norm_mat_m = torch.from_numpy(p_l_p_f_m.astype(np.float32))
            norm_mat_s = torch.from_numpy(p_l_p_f_s.astype(np.float32))
        elif ray_config['norm_method'] == "p_f":
            norm_mat_m = torch.from_numpy(p_f_m.astype(np.float32))
            norm_mat_s = torch.from_numpy(p_f_s.astype(np.float32))
        else:
            norm_mat_m = None
            norm_mat_s = None
        lstm_node_dim = ray_config['lstm_nodesize'] * 8
        model = cbh_torch_lstm.CloudBaseLSTM(
            ray_config['input_size'], 
            ray_config['lstm_layers'], 
            lstm_node_dim, 
            ray_config['lstm_output_size'], 
            ray_config['height_dimension'],
            ray_config['embed_size'], 
            ray_config['BILSTM'], 
            ray_config['backward_lstm_differing_transitions'], 
            ray_config['batch_first'], 
            ray_config['lr'], 
            ray_config['skip_connection'],
            ray_config['norm_method'],
            norm_mat_std=norm_mat_s,
            norm_mat_mean=norm_mat_m,
            linear_instead_of_conv_cap=ray_config['linear_instead_of_conv_cap']
        )
    if verbose_objective: print("Finished model init")
    timestamp_template = '{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
    run_name_template = 'cbh_challenge_{network_name}_' + timestamp_template
    current_run_name = run_name_template.format(network_name=model.__class__.__name__,
                                                    dt=datetime.datetime.now()
                                                   )
    
    with mlflow.start_run(experiment_id=mlf_exp.experiment_id, run_name=current_run_name) as run:
        mlflow.pytorch.autolog()
        mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_server_uri, run_id=run.info.run_id)
        if verbose_objective: print("Finished init logger")
        
        time_for_checkpoint = ray_config['chkpt_time']
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            train_time_interval=time_for_checkpoint,
            dirpath=run.info.artifact_uri,
            monitor="val_loss_mean",
            save_on_train_epoch_end=False,
            mode="min"
        )
        callbacks = [
            checkpoint_callback, 
            TQDMProgressBar(refresh_rate=0), 
            TuneReportCallback(
                {"val_loss_mean": "val_loss_mean",},
                on="validation_end"
            ),
            EarlyStopping(
                "val_loss_mean", min_delta=0.0, patience=40,
                divergence_threshold=30.
            ),
        ] 
        
        if verbose_objective: print("Finished define callbacks")
        trainer_hparams = {
            'max_epochs':ray_config['epoch'],
            'deterministic':ray_config['deterministic'],
            'val_check_interval':0.01, 
            'devices':"auto",
            'accelerator':"auto",
            'max_time':ray_config['max_time'], 
            'replace_sampler_ddp':False,
            'enable_checkpointing':True,
            'strategy':None,
            'callbacks':callbacks,
            'logger':mlf_logger,
        }
        if verbose_objective: print("Finished init hparams kwargs")
        hparams_for_mlflow['ray_config'] = ray_config
        mlf_logger.log_hyperparams(hparams_for_mlflow)
        if verbose_objective: print("Finished log hparams mlflow")
        if verbose_objective: print(trainer_hparams)
        trainer = pl.Trainer(
            **trainer_hparams
        )
        if verbose_objective: print("REACH all init before fit")
        trainer.fit(model=model, datamodule=datamodule)
        path_to_save = '{dt.year:04d}{dt.month:02d}{dt.day:02d}-{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'.format(dt=datetime.datetime.now())
        trainer.save_checkpoint(filepath=run.info.artifact_uri + f'/post_epoch_modelchkpt_{path_to_save}')

import warnings
warnings.filterwarnings("ignore")

searcher = OptunaSearch(metric=["val_loss_mean"], mode=["min"])
max_concurrent_trails = CPU_COUNT-2
algo = ConcurrencyLimiter(searcher, max_concurrent=max_concurrent_trails)
num_hparam_trials = 500

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=num_hparam_trials,
        time_budget_s=datetime.timedelta(hours=71.5)
    ),
    run_config=ray.air.config.RunConfig(
        local_dir=str(pathlib.Path(os.environ["SCRATCH"]) / "cbh_data")
        
    ),
    param_space=lstm_search_space,
)
results = tuner.fit()