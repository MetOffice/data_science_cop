"""ClimateZones Data Exploration - Environment Validation Test

This script's workflow content is produced by mechanically exporting the reduced reference
notebook (tests/notebooks/test_climatezones_data_exploration.ipynb) via
`jupyter nbconvert --to script`, then wrapping the exported content in main() below, in its
original order, essentially verbatim, with V1 environment-validation reporting logic added
around it.

* Author: Stephen Haddad and Kate Brown; Environment-validation test wrapper and notebook reduction: Jacob Way;
* Affiliation: UK Met Office
* History: 1.0
* Last update: 2026-03-16
* (c) British Crown Copyright 2017-2026, Met Office. Please see LICENSE.md for license details.

Running this script
--------------------
This script is an environment-validation test: its purpose is to check whether the *current*
environment can still execute this workflow. Run it directly, with the environment under test already
active:
    python test_climatezones_data_exploration.py
See the RESULT / Failure Category output below for how to interpret the outcome.

Data statement
--------------
This script is an environment-validation wrapper around a reduced version of the original
ClimateZones Data Exploration tutorial notebook. The underlying data it references, is the Koppen-Geiger Climate Classification dataset created by
GloH2O.

References
----------
- Climate Zones Dataset:
  https://www.gloh2o.org/koppen/#:~:text=The%20K%C3%B6ppen%2DGeiger%20climate%20classification%20maps%20are%20high%2Dresolution,Climate%20Sensitivity%20(ECS)%2C%20and%20historical%20warming%20trend
"""

import subprocess
import sys
import traceback

TEST_NAME = "ClimateZones Test"


def get_git_version():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def classify_exception(exc):
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "ENVIRONMENT FAILURE"
    if isinstance(exc, (FileNotFoundError, OSError)):
        return "DATA UNAVAILABLE"
    return "WORKFLOW FAILURE"


def main():
    print(TEST_NAME)
    print()
    print(f"Git Version: {get_git_version()}")
    print()

    try:

        import json
        import os
        import pathlib

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot
        import cartopy.crs
        import xarray
        import pandas

        CONFIG_PATH = pathlib.Path("../../config.json").resolve()

        with open(CONFIG_PATH, "r") as tutorial_config_file:
            tutorial_config = json.load(tutorial_config_file)


        def get_platform_dir(select_platform, config):
            try:
                root_path = pathlib.Path(config['default_dirs'][select_platform]) / 'climate_zones'
            except KeyError:
                root_path = pathlib.Path(os.environ['HOME']) / 'climate_zones'
            return root_path

        current_platform = tutorial_config['platform']
        root_data_dir = get_platform_dir(current_platform, tutorial_config)

        ml_ready_dir = root_data_dir / 'ml_ready'


        resolutions_dict = {float(k1): v1 for k1, v1 in tutorial_config['resolutions_names'].items()}

        dataset_prefix_dict = tutorial_config['dataset_prefix']

        format_str = 'nc'
        historic_scenario_str = 'historic'


        fname_template = tutorial_config['fname_template']
        time_dir_template = tutorial_config['time_dir_template']
        ml_ready_fname_template = tutorial_config['csv_out_template']


        current_res = 1.0


        select_historic = (1901, 1930)
        select_future = (2071, 2099)
        select_future_scenario = "ssp370"


        # Note: accepts root_dir/suffix as arguments but actually uses the enclosing root_data_dir
        # and format_str variables instead - a known peculiarity of the original notebook, preserved
        # here unchanged (works as a closure over main()'s locals, same as it previously relied on
        # module-level globals).
        def get_data_path(root_dir, time_period, scenario_id, prefix, resolution_str, suffix):
            start_year = time_period[0]
            end_year = time_period[1]
            if scenario_id == historic_scenario_str:
                data_dir = root_data_dir / time_dir_template.format(start_year=start_year, end_year=end_year)
            else:
                data_dir = root_data_dir / time_dir_template.format(start_year=start_year, end_year=end_year) / scenario_id
            data_fname = fname_template.format(prefix=prefix,
                                               res=resolution_str,
                                               suffix=format_str)
            return data_dir / data_fname


        historic_climate_zone_path = get_data_path(root_dir=root_data_dir, time_period=select_historic, scenario_id=historic_scenario_str, prefix=dataset_prefix_dict["climate_zone"],
                                                   resolution_str=resolutions_dict[current_res], suffix=format_str,)

        future_climate_zone_path = get_data_path(root_dir=root_data_dir, time_period=select_future, scenario_id=select_future_scenario, prefix=dataset_prefix_dict["climate_zone"],
                                                 resolution_str=resolutions_dict[current_res], suffix=format_str,)

        historic_climate_mean_path = get_data_path(root_dir=root_data_dir, time_period=select_historic, scenario_id=historic_scenario_str, prefix=dataset_prefix_dict["climate_mean"],
                                                   resolution_str=resolutions_dict[current_res], suffix=format_str,)


        historic_climate_zone_ds = xarray.open_dataset(historic_climate_zone_path)
        future_climate_zone_ds = xarray.open_dataset(future_climate_zone_path)
        historic_climate_mean_ds = xarray.open_dataset(historic_climate_mean_path)


        fig1 = matplotlib.pyplot.figure(figsize=(16, 8))

        ax1 = fig1.add_subplot(1, 1, 1, projection=cartopy.crs.PlateCarree(),)
        diff_arr = (future_climate_zone_ds["kg_class"]!=historic_climate_zone_ds["kg_class"])
        diff_arr.plot.contourf(ax=ax1, transform=cartopy.crs.PlateCarree(), cbar_kwargs={"location": "bottom"},)

        ax1.coastlines()
        ax1.set_title(f"KG Climate Zones diff {select_future} compared to {select_historic}")

        fig1.canvas.draw()
        matplotlib.pyplot.close(fig1)


        january_air_temperature = (historic_climate_mean_ds.loc[{"time": 1}]["air_temperature"])
        fig1 = matplotlib.pyplot.figure(figsize=(10, 5))

        ax1 = fig1.add_subplot(1, 1, 1,projection=cartopy.crs.PlateCarree(),)
        january_air_temperature.plot.contourf(ax=ax1,transform=cartopy.crs.PlateCarree(),)

        ax1.coastlines()
        ax1.set_title("January Air Temperature")

        fig1.canvas.draw()
        matplotlib.pyplot.close(fig1)


        mlready_data_path = ml_ready_dir / ml_ready_fname_template.format(resolution=resolutions_dict[current_res])


        zones_df = pandas.read_csv(mlready_data_path, nrows=25000)


        zones_df['climate_subgroup'].value_counts().plot.bar(figsize=(8,5))
        bar_ax = matplotlib.pyplot.gca()
        bar_ax.get_figure().canvas.draw()
        matplotlib.pyplot.close(bar_ax.get_figure())

        fig1 = matplotlib.pyplot.figure(figsize=(8, 5))
        ax1 = fig1.add_subplot(1, 1, 1, title="distribution of January Air Temperature - Zone A")
        zones_df[zones_df["climate_group"] == "A"]["air_temperature_1.0_mean"].hist()
        ax1.set_xlim(-30, 40)

        fig1.canvas.draw()
        matplotlib.pyplot.close(fig1)

    except Exception as exc:
        category = classify_exception(exc)
        print("RESULT:")
        print("NOT SUCCESSFULLY VALIDATED")
        print()
        print("Failure Category:")
        print(category)
        print()
        print("Exception:")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        return 1

    print("RESULT:")
    print("VALIDATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
