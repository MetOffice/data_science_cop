#!/usr/bin/env python
# coding: utf-8



import pathlib
import os
import datetime
import json
import re
import functools
import argparse

import numpy 
import xarray

import matplotlib
import matplotlib.pyplot
import cartopy.crs

import sklearn
import sklearn.preprocessing
import sklearn.tree

import mlflow

import torch

import scores

def get_platform_dir(select_platform, config):
    try:
        root_path = pathlib.Path(config['default_dirs'][select_platform]) / 'weatherbench'
    except KeyError:
        root_path = pathlib.Path(os.environ['HOME']) / 'weatherbench'
    return root_path

def get_config(config_path):
    with open (config_path,'r') as tutorial_config:
        tutorial_config = json.load(tutorial_config)
    return tutorial_config

def setup_mlflow(mlflow_url, mlflow_port, exp_name):
    mlflow_server_uri = f'{mlflow_url}:{mlflow_port}'

    print(f'connecting to mlflow server {mlflow_server_uri}')

    mlflow.set_tracking_uri(mlflow_server_uri)

    mlflow.pytorch.autolog()

    if mlflow.get_experiment_by_name(exp_name) is None:
        exp_id = mlflow.create_experiment(exp_name)
    current_experiment = mlflow.get_experiment_by_name(exp_name)
    return mlflow_server_uri, current_experiment
    

class WeatherbenchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, time_period, variables, levels, is_train=True):
        self._is_train=is_train
        self._variables = variables
        self._levels = levels

        self._ds_norm = xarray.open_zarr(data_dir)
        self._ds_norm = self._ds_norm.loc[{'time':slice(*time_period)}]
        self._ds_norm = self._ds_norm[self._variables]
        self._ds_norm = self._ds_norm.loc[dict(level=self._levels)]

        self._time_list = (self._ds_norm[ list(self._ds_norm.keys())[0] ]['time'].values)
        self.num_channels = len(self._ds_norm.data_vars)*len(self._ds_norm['level'])


    def __str__(self):
        return str(self._wb_ds)

    def __repr_html__(self):
        return self._wb_ds.__rept_html__()

    def __len__(self):
        return len(self._time_list)

    def __getitem__(self, idx):
        selected_time = self._ds_norm.time[idx].values
        select_ds = self._ds_norm.loc[{'time': selected_time}]
        if type(idx) == int:
            reshape_args = (self.num_channels, len(select_ds['lat']),len(select_ds['lon']) )
        else:
            reshape_args = (-1, self.num_channels, len(select_ds['lat']),len(select_ds['lon']) )
        select_array = numpy.stack(
            [select_ds[v1].to_numpy() for v1 in select_ds.data_vars],
            axis=1).reshape(reshape_args)


        select_tensor = torch.tensor(
            select_array,
            dtype=torch.float32,
        )
        return select_tensor

def create_data_loaders(wb_arco_path, var_subset, pl_subset, batch_size, train_interval, val_interval):
    """
    """
    wb_train_ds = WeatherbenchDataset(wb_arco_path, 
                                      (train_interval[0], train_interval[1]),                        
                                      is_train=True,
                                      variables = var_subset,
                                      levels=pl_subset,
                                     )
    wb_val_ds = WeatherbenchDataset(wb_arco_path, 
                                    (val_interval[0], val_interval[1]),
                                    is_train=False,
                                    variables = var_subset,
                                    levels=pl_subset,
                                   )

    print(f'num channels: {wb_train_ds.num_channels}')
    print(f'length train: {len(wb_train_ds)}; length validation: {len(wb_val_ds)}')

    wb_train_loader = torch.utils.data.DataLoader(wb_train_ds,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                              )
    
    wb_val_loader = torch.utils.data.DataLoader(wb_val_ds,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                            )
    return (wb_train_ds, wb_train_loader, wb_val_ds, wb_val_loader)

class Era5AutoEncoder(torch.nn.Module):
    def __init__(self, num_channels, max_pool=False):
        super(Era5AutoEncoder, self).__init__()

        # we have "hard coded" a lot of the architecture hyperparameters in our model class. 
        # Usually you want want to make these arguments for the class so you can vary hyperparameters more easily.
        # Hard coding here makes it easier to follow the architecture definition in the tutorial
        self.num_channels = num_channels
        
        self._latent_array_dims = (-1,32,8,16)
        self._prelatent_size = functools.reduce(lambda a,b:a*b, self._latent_array_dims[1:])
        self._latent_size = 500
        
        self._encoder = self._get_encoder(max_pool)
        self._decoder = self._get_decoder()

    def _get_encoder(self, max_pool):
        if max_pool:
            encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.num_channels, 
                                out_channels=16, 
                                kernel_size=3, 
                                padding=1,
                               ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2),
                torch.nn.Conv2d(in_channels=16, 
                                out_channels=32, 
                                kernel_size=3, 
                                padding=1,
                               ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2),
                torch.nn.Flatten(1,-1)
            )
        else:
            encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.num_channels, 
                                out_channels=16, 
                                kernel_size=3, 
                                padding=1,
                                stride=2,
                               ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=16, 
                                out_channels=32, 
                                kernel_size=3, 
                                padding=1,
                                stride=2,
                               ),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                # torch.nn.Linear(self._prelatent_size, self._latent_size),
                # torch.nn.ReLU(),
            )
        return encoder

    def _get_decoder(self):
        """
        """
        decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2,stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=self.num_channels, kernel_size=2,stride=2),
            # torch.nn.ReLU(),
            # torch.nn.Sigmoid(),   
        )
        return decoder
    def forward(self, x):

        # Get latent representation
        latent = self._encoder(x)

        # Reconstruct input
        reconstructed = self._decoder(latent.view(self._latent_array_dims))
        # reconstructed = self._decoder(latent)

        return reconstructed

def run_training(device, train_loader, val_loader, num_epochs, learning_rate, num_channels, batch_size, current_exp=None, checkpoint_dir=None):
    ae_model = Era5AutoEncoder(num_channels, False).to(device)

    print('num parameters',sum(p.numel() for p in ae_model.parameters() if p.requires_grad))
    loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), 
                                 lr=learning_rate)
    if current_exp is not None:
        with mlflow.start_run(experiment_id=current_exp.experiment_id) as current_run:
            ae_model =  run_train_loop(device,
                                       ae_model,
                                       loss_function,
                                       optimizer,
                                       train_loader,
                                       val_loader,
                                       num_epochs,
                                       learning_rate,
                                       num_channels,
                                       batch_size,
                                       checkpoint_dir,
                                       use_mlflow=True,
                                       )
    else:
        ae_model =  run_train_loop(device,
                                   ae_model,
                                   loss_function,
                                   optimizer,
                                   train_loader,
                                   val_loader,
                                   num_epochs,
                                   learning_rate,
                                   num_channels,
                                   batch_size,
                                   checkpoint_dir,
                                   use_mlflow=False,
                                   )

    return ae_model
        
    

def run_train_loop(device, ae_model, loss_function, optimizer, train_loader, val_loader, num_epochs, learning_rate, num_channels, batch_size, checkpoint_dir,use_mlflow=False):
    train_start_dt = datetime.datetime.now()

    if use_mlflow:
        mlflow.log_param('num_epochs',num_epochs)
        mlflow.log_param('learning_rate',learning_rate)
        mlflow.log_param('batch_size',batch_size)
        
    for epoch_num in range(num_epochs):
        epoch_start_dt = datetime.datetime.now()
        print(epoch_num)
        epoch_train_loss = 0.0
        for batch_ix, X_batch in enumerate(train_loader):
            if (batch_ix % 1000) == 0:
                print(batch_ix)
            optimizer.zero_grad()
            predictions = ae_model.forward(X_batch.to(device))
            loss_batch = loss_function(predictions, X_batch.to(device))
            loss_batch.backward()
            optimizer.step()
            epoch_train_loss += loss_batch.to('cpu').item()
        epoch_train_loss /= len(train_loader)
        print(epoch_train_loss)
        epoch_val_loss = 0.0
        for batch_ix_val, X_batch_val in enumerate(val_loader):
            predictions_val = ae_model.forward(X_batch_val.to(device))
            loss_batch_val = loss_function(predictions_val, X_batch_val.to(device))
            epoch_val_loss += loss_batch_val.to('cpu').item()
        epoch_val_loss /= len(val_loader)
        if use_mlflow:
            mlflow.log_metrics(
                {'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss},
                step=epoch_num,
            )

        print(epoch_train_loss)
        print(epoch_val_loss)
        epoch_duration_minutes = (datetime.datetime.now() - epoch_start_dt) // 60
        print(f'epoch train loop time {epoch_duration_minutes} minutes')
        if checkpoint_dir is not None:
            cp_fname = f'era5_autoencoder_checkoint_{epoch_num:03d}.pth'
            cp_path = checkpoint_dir / cp_fname
            torch.save(ae_model, cp_path)
            print(f'checkpoint for epoch {epoch_num} saved to {cp_path}')
            if use_mlflow:
                mlflow.log_artifact(cp_path)
                            

    train_duration_minutes = (datetime.datetime.now() - train_start_dt) // 60
    print(f'total train loop time {train_duration_minutes} minutes')
    if use_mlflow:
        mlflow.log_param('train_time_minutes', train_duration_minutes)

    model_fname = 'era5_ae_model.pth'
    model_save_path = checkpoint_dir / model_fname
    torch.save(ae_model, model_save_path)
    if use_mlflow:
        mlflow.log_artifact(model_save_path)
        
    return ae_model

def plot_sample_prediction(select_ds, ae_model, device, out_dir):
    """
    create a new data array to contain the model predictions, which can then subsequnetly use the xarray plotting interface
    """
    pred_da = xarray.DataArray(select_ds._ds_norm['temperature'][2].sel(level=850))
    pred_arr = ae_model.forward(select_ds[2].to(device)).to('cpu').detach().numpy() 
    pred_da.values = pred_arr[0,0,:]
    
    # plot results compared to truth
    fig1 = matplotlib.pyplot.figure('sample prediction - temprature', figsize=(10,16))
    ax1 = fig1.add_subplot(2,1,1, title='sample truth: temp 850')
    select_ds._ds_norm['temperature'][2].sel(level=850).plot.contourf(ax=ax1)
    ax1 = fig1.add_subplot(2,1,2, title='sample prediction: temp 850')
    pred_da.plot.contourf(ax=ax1)
    fig1.savefig(out_dir / 'sample_temp_850.png')


def do_evaluation(ae_model, ds1, device):

    metrics = scores.continuous.rmse(
        ae_model.forward(ds1.to(device)).to('cpu').detach().numpy(), 
        ds1.numpy()
    )
    return metrics

def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='train_era5_autoenconder',
        description='Train an autoencoder on lowres era5 data',
    )
    
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--config-path', dest='config_path', type=pathlib.Path, default=pathlib.Path('config.json') )
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--model-out-dir', dest='model_out_dir' , type=pathlib.Path, default=pathlib.Path('.'))
    parser.add_argument('--data-dir',dest='data_dir',type=pathlib.Path, default=None)
    parser.add_argument('--mlflow-url',dest='mlflow_url',type=str,default='')
    parser.add_argument('--mlflow-port',dest='mlflow_port', type=int,default=4455)

    cmd_args = parser.parse_args()
    return cmd_args


def main():
    cmd_args = get_cmd_args()
    tutorial_config = get_config(cmd_args.config_path)

    exp_name = 'era5_autoencoder'
    if cmd_args.mlflow_url == '':
       print('mlflow not being used')
       mlflow_server_uri = ''
       current_exp = None
    else:

        mlflow_server_uri, current_exp = setup_mlflow(cmd_args.mlflow_url, cmd_args.mlflow_port, exp_name)
    
    resolution_dict = {5.625: '5.625deg'}
    var_subset = ['temperature', 'geopotential']
    pl_subset = [500, 850]   

    print(f'training an autoencoder on the follow variables {var_subset}\n and the following pressure levels {pl_subset}')
    current_platform = tutorial_config['platform']
    root_data_dir = get_platform_dir(current_platform, tutorial_config)
    weatherbench_dir = root_data_dir / resolution_dict[5.625]
    
    if cmd_args.data_dir is not None:
        wb_arco_path = cmd_args.data_dir
    else:
        wb_arco_path = root_data_dir / 'wb_arco'

    train_interval = (
        datetime.datetime(1990,1,1,0,0),
        datetime.datetime(2000,1,1,0,0),
    )
    val_interval = (
        datetime.datetime(2010,1,1,0,0),
        datetime.datetime(2012,1,1,0,0),
    )
    
    (wb_train_ds, wb_train_loader, wb_val_ds, wb_val_loader) = create_data_loaders(wb_arco_path, 
                                                                                   var_subset,
                                                                                   pl_subset,
                                                                                   cmd_args.batch_size,
                                                                                   train_interval,
                                                                                   val_interval,
                                                                                  )
    print(f'using data at {wb_arco_path}')

    
    # Autodetect GPU and use if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    exp_dir = cmd_args.model_out_dir
    cdt = datetime.datetime.now()
    run_dir = exp_dir / f'run_{cdt.year:04d}{cdt.month:02d}{cdt.day:02d}_{cdt.hour:02d}{cdt.minute:02d}'
    try:
        run_dir.mkdir(parents=True)
        print(f'created run dir {run_dir}')
    except FileExistsError:
        print(f'run dir {run_dir} already exists')
    ae_model = run_training(device,
                            wb_train_loader,
                            wb_val_loader,
                            cmd_args.num_epochs,
                            cmd_args.learning_rate,
                            wb_train_ds.num_channels,
                            cmd_args.batch_size,
                            current_exp,
                            run_dir,
                            )

    metrics_train = do_evaluation(ae_model, wb_train_ds[:10], device)
    metrics_val = do_evaluation(ae_model, wb_val_ds[:10], device)

    plot_sample_prediction(wb_val_ds, ae_model, device, run_dir)

    print('rmse train')
    print(metrics_train)
    
    print('rmse validation')
    print(metrics_val)


if __name__ == '__main__':
    main()


    
