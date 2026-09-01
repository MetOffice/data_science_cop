"""ClimateZones Data Exploration - Environment Validation Test

This script's workflow content originated from the ClimateZones Data Exploration
tutorial notebook and was reduced to provide a representative environment-
validation workflow.

The workflow content below remains substantially derived from that reduced
workflow, with environment-validation reporting and retention functionality
added around it.

* Author: Stephen Haddad and Kate Brown; Environment-validation test wrapper and workflow reduction: Jacob Way;
* Affiliation: UK Met Office
* History: 1.0
* Last update: 2026-08-28
* (c) British Crown Copyright 2017-2026, Met Office. Please see LICENSE.md for license details.

Running this test
-----------------
This script forms part of an environment-validation framework.

It is intended to be executed via:

    run_test_climatezones_data_exploration.sh

which loads the environment under test and invokes this script.

See the validation result and failure-classification output below for guidance
on interpreting outcomes.

Data statement
--------------
This script is an environment-validation implementation based on the original
ClimateZones Data Exploration tutorial workflow.

The underlying data referenced by the workflow is the Koppen-Geiger Climate
Classification dataset created by GloH2O.

References
----------
- Climate Zones Dataset:
  https://www.gloh2o.org/koppen/#:~:text=The%20K%C3%B6ppen%2DGeiger%20climate%20classification%20maps%20are%20high%2Dresolution,Climate%20Sensitivity%20(ECS)%2C%20and%20historical%20warming%20trend
"""


import subprocess
import sys
import traceback
import datetime
import logging

TEST_NAME = "ClimateZones Test"
RESULT_LEVEL = 60


def get_git_version():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"

def get_file_git_status(file_path):
    try:
        unstaged_changes = subprocess.run(
            ["git", "diff", "--quiet", "--", str(file_path),]).returncode

        staged_changes = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", str(file_path),]).returncode

        if unstaged_changes != 0 or staged_changes != 0:
            return "DIRTY"
        return "CLEAN"
    except Exception:
        return "UNKNOWN"

def get_git_file_statuses():
    try:
        from pathlib import Path
        script_path = Path(__file__).resolve()
        wrapper_path = (script_path.parent /"run_test_climatezones_data_exploration.sh").resolve()

        return {
            "python_script": get_file_git_status(script_path),
            "bash_wrapper": get_file_git_status(wrapper_path),
            }
    except Exception:
        return {
            "python_script": "UNKNOWN",
            "bash_wrapper": "UNKNOWN",
        }

def classify_exception(exc):
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "LIKELY ENVIRONMENT FAILURE"
    if isinstance(exc, (PermissionError, MemoryError)):
        return "UNCLEAR WHETHER ENVIRONMENT FAILURE"
    return "LIKELY NON-ENVIRONMENT FAILURE"

def initialise_retention_mode():
    retention = "--retention" in sys.argv
    return retention, []

def configure_logging():
    logging.addLevelName(RESULT_LEVEL, "RESULT")
    try:
        log_level_index = sys.argv.index("--log-level")
        supplied_level = sys.argv[log_level_index + 1].upper()

        if supplied_level == "RESULT":
            logging_level = RESULT_LEVEL
        else:
            logging_level = getattr(logging, supplied_level)

        logging.basicConfig(
            level=logging_level,
            format="%(message)s",
        )

    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )

        logging.warning("Invalid log level supplied. Falling back to INFO.")

    return logging.getLogger(__name__)

def log_result(message):
    LOGGER.log(RESULT_LEVEL, message)

def create_artefact_directory():
    try:
        import pathlib
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artefact_dir = pathlib.Path("../test_run_logs/climatezones_data_exploration") / timestamp
        artefact_dir.mkdir(parents=True, exist_ok=False)
        pathlib.Path("latest_artefact_dir.txt").write_text(str(artefact_dir.resolve()))

        return artefact_dir
    except Exception:
        LOGGER.warning("Run artefacts could not be retained.")
        return None


def save_retained_figures(retained_figures, artefact_dir, retention):
    if not retention:
        return
    try:
        for filename, fig in retained_figures:
            fig.savefig(artefact_dir / filename)
    except Exception:
        LOGGER.warning("Retained figures could not be retained.")

    try:
        import matplotlib.pyplot

        for _, fig in retained_figures:
            matplotlib.pyplot.close(fig)
    except Exception:
        pass

def finalise_run(retention, retained_figures):
    git_statuses = get_git_file_statuses()
    git_version = get_git_version()
    LOGGER.info("")
    LOGGER.info(f"Git Version: {git_version}")
    LOGGER.info(f"Python Script Status: {git_statuses['python_script']}")
    LOGGER.info(f"Bash Wrapper Status: {git_statuses['bash_wrapper']}")
    LOGGER.info("")
    artefact_dir = None
    if retention:
        artefact_dir = create_artefact_directory()
        save_metadata(artefact_dir, git_statuses, git_version)
        save_retained_figures(retained_figures, artefact_dir, retention)

def save_metadata(artefact_dir, git_statuses, git_version):
    try:
        import json

        metadata = {
            "git_version": git_version,
            "test_name": TEST_NAME,
            "arguments": sys.argv,
            "git_statuses": git_statuses,
        }

        with open(artefact_dir / "metadata.json", "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
    except Exception:
        LOGGER.warning("Metadata could not be retained.")

LOGGER = configure_logging()

def main():
    LOGGER.info(TEST_NAME)
    LOGGER.info("")

    retention, retained_figures = initialise_retention_mode()

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

        CONFIG_PATH = pathlib.Path("../../ml_examples/climate_zones/config.json").resolve()

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
        if retention:
            retained_figures.append(("01_climate_zone_diff_map.png", fig1))
        else:
            matplotlib.pyplot.close(fig1)


        january_air_temperature = (historic_climate_mean_ds.loc[{"time": 1}]["air_temperature"])
        fig1 = matplotlib.pyplot.figure(figsize=(10, 5))

        ax1 = fig1.add_subplot(1, 1, 1,projection=cartopy.crs.PlateCarree(),)
        january_air_temperature.plot.contourf(ax=ax1,transform=cartopy.crs.PlateCarree(),)

        ax1.coastlines()
        ax1.set_title("January Air Temperature")

        fig1.canvas.draw()
        if retention:
            retained_figures.append(("02_january_air_temperature_map.png", fig1))
        else:
            matplotlib.pyplot.close(fig1)
        mlready_data_path = ml_ready_dir / ml_ready_fname_template.format(resolution=resolutions_dict[current_res])

        zones_df = pandas.read_csv(mlready_data_path, nrows=25000)
        bar_fig = matplotlib.pyplot.figure(figsize=(8, 5))
        zones_df['climate_subgroup'].value_counts().plot.bar()

        bar_fig.canvas.draw()
        if retention:
            retained_figures.append(("03_climate_subgroup_bar.png", bar_fig))
        else:
            matplotlib.pyplot.close(bar_fig)

        fig1 = matplotlib.pyplot.figure(figsize=(8, 5))
        ax1 = fig1.add_subplot(1, 1, 1, title="distribution of January Air Temperature - Zone A")
        zones_df[zones_df["climate_group"] == "A"]["air_temperature_1.0_mean"].hist()
        ax1.set_xlim(-30, 40)

        fig1.canvas.draw()
        if retention:
            retained_figures.append(("04_zone_a_january_temp_hist.png", fig1))
        else:
            matplotlib.pyplot.close(fig1)

    except Exception as exc:
        category = classify_exception(exc)
        log_result("RESULT:")
        log_result("NOT SUCCESSFULLY VALIDATED")
        LOGGER.error("")
        LOGGER.error("Failure Category:")
        LOGGER.error(category)
        LOGGER.error("")
        LOGGER.error("Exception:")
        LOGGER.error(f"{type(exc).__name__}: {exc}")
        LOGGER.error(traceback.format_exc())
        finalise_run(retention, retained_figures)
        return 1

    log_result("RESULT:")
    log_result("VALIDATED")
    finalise_run(retention, retained_figures)

    return 0


if __name__ == "__main__":
    sys.exit(main())
