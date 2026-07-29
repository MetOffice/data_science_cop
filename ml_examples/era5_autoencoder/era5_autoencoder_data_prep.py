#!/usr/bin/env python
# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""
This module takes the low resolution version of ERA5 data that makes up the Weatherbench v1 dataset, and prepares it for use in this exercise. This includes calculating means and std deviation of fields for normalising data ahead of training, and saving in zarr format for improved loading performance.

Link to dataset: https://github.com/pangeo-data/WeatherBench
"""

import pathlib
import argparse
import os
import datetime
import json
import re

import numpy 
import xarray

import matplotlib
import matplotlib.pyplot
import cartopy.crs


def get_platform_dir(select_platform, config):
    try:
        root_path = pathlib.Path(config['default_dirs'][select_platform]) / 'weatherbench'
    except KeyError:
        root_path = pathlib.Path(os.environ['HOME']) / 'weatherbench'
    return root_path


def get_rename_dict(xr_ds, var_lut):
    vars_present = list(xr_ds.data_vars)
    rename_lut = {k1: v1 for k1,v1 in var_lut.items() if k1 in vars_present}
    return rename_lut

def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='ERA5_data_prep',
        description='prepare era5 data for training',
    )
    
    parser.add_argument('--start-year', dest='start_year', type=int, default=1980)
    parser.add_argument('--end-year', dest='end_year', type=int, default=2017 )
    parser.add_argument('--data-out-dir', dest='data_out_dir', type=pathlib.Path)
    parser.add_argument('--root-data-dir', dest='root_data_dir', type=pathlib.Path)

    cmd_args = parser.parse_args()
    return cmd_args



def main():
    cmd_args = get_cmd_args()
    
    root_data_dir = cmd_args.root_data_dir

    wb_zarr_root_dir = cmd_args.data_out_dir 
    if not wb_zarr_root_dir.is_dir():
        wb_zarr_root_dir.mkdir(parents=True)
        
    var_list = ['temperature', 
            'specific_humidity',
            'u_component_of_wind',
            'v_component_of_wind',
            'geopotential'
           ]

    pl_list = [1000,850,700,500,200]
    
    era5_rename_lut = {
        'z': 'geopotential',
        't': 'temperature',
        'q': 'specific_humidity',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
    }
    resolution_dict = {5.625: '5.625deg'}

    weatherbench_dir = root_data_dir / resolution_dict[5.625]
    if not weatherbench_dir.is_dir():
        raise FileNotFoundError('root data directory not found, please correct config file.')
    
    start_period = datetime.datetime(cmd_args.start_year,1,1,0,0)
    end_period = datetime.datetime(cmd_args.end_year,12,31,0,0)

    agg_dims = ['time','lat','lon']

    var_ds_list = []
    var_norm_ds_list = []
    var_stats_dict = {}
    for current_var in var_list:
        print(current_var)
        pattern = re.compile(current_var + r"_(\d{4})_5\.625deg\.nc")
        files_to_load = sorted([ f1 for f1 in (weatherbench_dir / current_var).iterdir() if pattern.match(f1.name)])
        current_ds = xarray.open_mfdataset(files_to_load)
        current_ds = current_ds.loc[{'level': pl_list, 
                                     'time': slice(start_period,end_period)}]
        current_ds = current_ds.rename(get_rename_dict(current_ds, era5_rename_lut))
        current_ds.chunk({'time': 240})
        var_ds_list += [current_ds]

        #calculate norms for variable
        current_std = current_ds.std(dim=agg_dims)
        current_mean = current_ds.mean(dim=agg_dims)
        var_stats_dict[current_var] = {
            'mean': current_mean,
            'std': current_std,
        }
        
        norm_ds = (current_ds - current_mean) / current_std
        var_norm_ds_list += [norm_ds]

    print('data loading complete')

    var_stats_json = { var1: {
        'mean': var_dict['mean'].to_dict()['data_vars'][var1]['data'],
        'std': var_dict['std'].to_dict()['data_vars'][var1]['data'],
    } for var1, var_dict in var_stats_dict.items()}
    
    stats_json_path = wb_zarr_root_dir / 'stats.json'
    with open(stats_json_path,'w') as stats_json_file:
        json.dump(var_stats_json, stats_json_file)
        print(f'stats written to {stats_json_path}')

    wb_zarr_out_dir = wb_zarr_root_dir / 'orig'
    if not wb_zarr_out_dir.is_dir():
        wb_zarr_out_dir.mkdir(parents=True)
    era5_ds =  xarray.merge(var_ds_list).chunk({'time':240})
    era5_ds.to_zarr(wb_zarr_out_dir)
    print(f'zarr of original data written to {wb_zarr_out_dir}')
    
    wb_zarr_norm_dir = wb_zarr_root_dir / 'norm'
    if not wb_zarr_norm_dir.is_dir():
        wb_zarr_norm_dir.mkdir(parents=True)
    era5_norm_ds = xarray.merge(var_norm_ds_list).chunk({'time':240})
    era5_norm_ds.to_zarr(wb_zarr_norm_dir)
    print(f'zarr of normalised data written to {wb_zarr_norm_dir}')
    
if __name__ =='__main__':
    main()
    
        

        
    







