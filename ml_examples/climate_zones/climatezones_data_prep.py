#!/usr/bin/env python
# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""Prepare the ML-ready tabular climate zones dataset from gridded netCDF files.

Script version of the ``climatezones_data_prep.ipynb`` notebook. This is not a
core part of the climate zones tutorial, but documents how the tutorial dataset
was prepared from the original Koppen-Geiger dataset.

The source data is gridded netCDF, with three files per time period, scenario
and resolution: the ensemble mean and ensemble standard deviation of monthly
climate variables, and the Koppen-Geiger climate zone class. Historical periods
have a single simulation, whereas future periods have one set of files per
`Shared Socioeconomic Pathway <https://en.wikipedia.org/wiki/Shared_Socioeconomic_Pathways>`_
(SSP) climate change scenario.

Each period/scenario combination is converted to a table with one row per land
point, written to an intermediate CSV, and processed in parallel. The
intermediate files are then merged into one CSV per resolution.

Example::

    python climatezones_data_prep.py \\
        --config-path config.json \\
        --output-dir /path/to/ml_ready \\
        --resolutions 1.0 0.5 \\
        --num-processes 4
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas
import xarray

HISTORIC_SCENARIO = "historic"
DEFAULT_FORMAT = "nc"
PERIOD_SCENARIO_TEMPLATE = "climate_zones_{start}_{end}_{scenario}_{res}.csv"
HISTORIC_PERIODS = [(1901, 1930), (1931, 1960), (1961, 1990), (1991, 2020)]
FUTURE_PERIODS = [(2041, 2070), (2071, 2099)]

Period = Tuple[int, int]
TaskArgs = Tuple[Dict[str, Path], float, Period, str, Path, List[str], List[str]]


@dataclass
class DataPrepConfig:
    """Runtime data preparation configuration parsed from command-line arguments."""

    config_path: Path
    root_data_dir: Optional[Path]
    output_dir: Optional[Path]
    resolutions: Optional[List[float]]
    num_processes: int
    file_format: str


def parse_args(argv: Optional[List[str]] = None) -> DataPrepConfig:
    """Parse command-line arguments for data preparation.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed data preparation configuration.
    """
    parser = argparse.ArgumentParser(
        description="Convert gridded climate zones netCDF data into ML-ready CSV files"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("config.json"),
        help="Path to the tutorial JSON config file",
    )
    parser.add_argument(
        "--root-data-dir",
        type=Path,
        default=None,
        help="Root directory of the source netCDF data. "
        "Defaults to the directory for the platform set in the config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the ML-ready CSV files to. "
        "Defaults to '<root-data-dir>/ml_ready'",
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=None,
        help="Resolutions in degrees to process. Defaults to all in the config",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes. Use 1 to run serially",
    )
    parser.add_argument(
        "--file-format",
        default=DEFAULT_FORMAT,
        help="File extension of the source data files",
    )
    args = parser.parse_args(argv)
    if args.num_processes < 1:
        parser.error("--num-processes must be at least 1")

    return DataPrepConfig(
        config_path=args.config_path,
        root_data_dir=args.root_data_dir,
        output_dir=args.output_dir,
        resolutions=args.resolutions,
        num_processes=args.num_processes,
        file_format=args.file_format,
    )


def load_config(config_path: Path) -> Dict:
    """Load the tutorial JSON config file into a dictionary."""
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def get_platform_dir(select_platform: str, config: Dict) -> Path:
    """Resolve the root climate zones data directory for a platform.

    Falls back to ``$HOME/climate_zones`` if the platform is not in the config.

    Args:
        select_platform: Name of the platform, e.g. ``'mo_linux'``.
        config: Tutorial config dictionary containing ``default_dirs``.

    Returns:
        Path to the root climate zones data directory.
    """
    try:
        root_path = Path(config["default_dirs"][select_platform]) / "climate_zones"
    except KeyError:
        root_path = Path(os.environ["HOME"]) / "climate_zones"
    return root_path


def get_time_periods(config: Dict) -> Dict[Period, List[str]]:
    """Map each time period to the list of scenarios available for it.

    Historical periods have a single historical simulation, whereas future
    periods have one simulation per SSP scenario.

    Args:
        config: Tutorial config with ``historic_scenarios`` and ``future_scenarios``.

    Returns:
        Mapping of ``(start_year, end_year)`` to scenario names.
    """
    time_periods = {p: list(config["historic_scenarios"]) for p in HISTORIC_PERIODS}
    time_periods.update({p: list(config["future_scenarios"]) for p in FUTURE_PERIODS})
    return time_periods


def get_class_lookups(climate_subgroups: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Build lookups from Koppen-Geiger class number to climate group and subgroup.

    The labels are taken from the ``legend.txt`` file in the
    `dataset <https://figshare.com/articles/dataset/High-resolution_1_km_K_ppen-Geiger_maps_for_1901_2099_based_on_constrained_CMIP6_projections/21789074>`_.
    As well as all 30 subgroups, the five broad groups (the first letter of the
    subgroup code) are provided as a simpler classification target.

    Args:
        climate_subgroups: Ordered mapping of subgroup code to description, where
            the position is the class number and the first entry is non-land.

    Returns:
        Tuple of (group lookup, subgroup lookup), indexed by class number.
    """
    subgroup_lookup = list(climate_subgroups.keys())
    group_lookup = [code[0] for code in subgroup_lookup]
    group_lookup[0] = "none"
    return group_lookup, subgroup_lookup


def get_data_path(
    root_dir: Path,
    time_period: Period,
    scenario_id: str,
    prefix: str,
    resolution_str: str,
    suffix: str,
    time_dir_template: str,
    fname_template: str,
) -> Path:
    """Build the path to a source data file.

    Historical data is in the time period directory, whereas future scenario
    data is in a scenario subdirectory of the time period directory.

    Args:
        root_dir: Root directory of the source data.
        time_period: ``(start_year, end_year)`` of the period.
        scenario_id: Scenario name, e.g. ``'historic'`` or ``'ssp245'``.
        prefix: Dataset file prefix, e.g. ``'ensemble_mean'``.
        resolution_str: Resolution string used in file names, e.g. ``'1p0'``.
        suffix: File extension, e.g. ``'nc'``.
        time_dir_template: Template for the time period directory name.
        fname_template: Template for the data file name.

    Returns:
        Path to the data file.
    """
    start_year, end_year = time_period
    data_dir = root_dir / time_dir_template.format(start_year=start_year, end_year=end_year)
    if scenario_id != HISTORIC_SCENARIO:
        data_dir = data_dir / scenario_id
    return data_dir / fname_template.format(prefix=prefix, res=resolution_str, suffix=suffix)


def build_data_paths(
    root_dir: Path,
    time_periods: Dict[Period, List[str]],
    dataset_prefixes: Dict[str, str],
    resolutions: Dict[float, str],
    suffix: str,
    time_dir_template: str,
    fname_template: str,
) -> Dict[Period, Dict[str, Dict[str, Dict[float, Path]]]]:
    """Build the nested dictionary of all source data file paths.

    Args:
        root_dir: Root directory of the source data.
        time_periods: Mapping of time period to scenario names.
        dataset_prefixes: Mapping of dataset ID to file prefix.
        resolutions: Mapping of resolution to resolution string.
        suffix: File extension.
        time_dir_template: Template for the time period directory name.
        fname_template: Template for the data file name.

    Returns:
        Paths indexed as ``[period][scenario][dataset_id][resolution]``.
    """
    return {
        period: {
            scenario_id: {
                ds_id: {
                    res: get_data_path(
                        root_dir, period, scenario_id, prefix, res_str, suffix,
                        time_dir_template, fname_template,
                    )
                    for res, res_str in resolutions.items()
                }
                for ds_id, prefix in dataset_prefixes.items()
            }
            for scenario_id in scenarios
        }
        for period, scenarios in time_periods.items()
    }


def create_dataframe_period(
    data_period_scenario: Dict[str, xarray.Dataset],
    period: Period,
    scenario: str,
    group_lookup: List[str],
    subgroup_lookup: List[str],
) -> pandas.DataFrame:
    """Convert gridded data for one period and scenario into a table of land points.

    Monthly mean and standard deviation fields are unstacked into one column per
    variable and month, e.g. ``precipitation_1.0_mean``. Non-land points (where
    the climate data is missing) are removed.

    Args:
        data_period_scenario: Datasets keyed by ``climate_mean``, ``climate_std``
            and ``climate_zone``.
        period: ``(start_year, end_year)`` of the period.
        scenario: Scenario name.
        group_lookup: Climate group label indexed by class number.
        subgroup_lookup: Climate subgroup label indexed by class number.

    Returns:
        Dataframe with one row per land point.
    """
    print(f"processing period {period} scenario {scenario}")
    df_mean = data_period_scenario["climate_mean"].to_dataframe().unstack(["time"])
    df_mean.columns = ["_".join(map(str, col)) + "_mean" for col in df_mean.columns]
    df_mean = df_mean.reset_index()

    df_std = data_period_scenario["climate_std"].to_dataframe().unstack(["time"])
    df_std.columns = ["_".join(map(str, col)) + "_std" for col in df_std.columns]
    df_std = df_std.reset_index()

    df_zone = data_period_scenario["climate_zone"].to_dataframe().reset_index()

    df_climate_zones = df_mean.merge(df_std, on=["lat", "lon"]).merge(df_zone, on=["lat", "lon"])
    df_climate_zones = df_climate_zones[~df_climate_zones["precipitation_1.0_mean"].isna()].copy()
    df_climate_zones["period_start"] = period[0]
    df_climate_zones["period_end"] = period[1]
    df_climate_zones["scenario"] = scenario

    class_ids = df_climate_zones["kg_class"].astype(int)
    df_climate_zones["climate_group"] = class_ids.map(lambda v: group_lookup[v])
    df_climate_zones["climate_subgroup"] = class_ids.map(lambda v: subgroup_lookup[v])
    return df_climate_zones


def process_period_scenario(
    path_dict: Dict[str, Path],
    current_res: float,
    current_period: Period,
    current_scenario: str,
    out_path: Path,
    group_lookup: List[str],
    subgroup_lookup: List[str],
) -> pandas.DataFrame:
    """Load the data for one resolution, period and scenario and write it to CSV.

    Args:
        path_dict: Source file paths keyed by dataset ID.
        current_res: Resolution in degrees.
        current_period: ``(start_year, end_year)`` of the period.
        current_scenario: Scenario name.
        out_path: Path of the intermediate CSV to write.
        group_lookup: Climate group label indexed by class number.
        subgroup_lookup: Climate subgroup label indexed by class number.

    Returns:
        The processed dataframe.
    """
    current_data = {ds_id: xarray.open_dataset(path) for ds_id, path in path_dict.items()}
    try:
        current_df = create_dataframe_period(
            current_data, current_period, current_scenario, group_lookup, subgroup_lookup
        )
    finally:
        for ds in current_data.values():
            ds.close()
    current_df.to_csv(out_path, index=False)
    return current_df


def build_tasks(
    data_paths: Dict[Period, Dict[str, Dict[str, Dict[float, Path]]]],
    resolutions: Dict[float, str],
    output_dir: Path,
    group_lookup: List[str],
    subgroup_lookup: List[str],
) -> Tuple[Dict[float, List[TaskArgs]], Dict[float, List[Path]]]:
    """Set up the arguments for each parallel task and the intermediate output paths.

    Args:
        data_paths: Output of :func:`build_data_paths`.
        resolutions: Mapping of resolution to resolution string.
        output_dir: Directory for the intermediate CSV files.
        group_lookup: Climate group label indexed by class number.
        subgroup_lookup: Climate subgroup label indexed by class number.

    Returns:
        Tuple of (task argument tuples, intermediate output paths), each keyed
        by resolution.
    """
    task_args: Dict[float, List[TaskArgs]] = {}
    inter_paths: Dict[float, List[Path]] = {}
    for res, res_str in resolutions.items():
        task_args[res] = []
        inter_paths[res] = []
        for period, scenario_paths in data_paths.items():
            for scenario, scenario_data in scenario_paths.items():
                out_path = output_dir / PERIOD_SCENARIO_TEMPLATE.format(
                    res=res_str, start=period[0], end=period[1], scenario=scenario
                )
                inter_paths[res].append(out_path)
                input_paths = {ds_id: res_paths[res] for ds_id, res_paths in scenario_data.items()}
                task_args[res].append(
                    (input_paths, res, period, scenario, out_path, group_lookup, subgroup_lookup)
                )
    return task_args, inter_paths


def merge_intermediate_files(paths: Sequence[Path], out_path: Path) -> pandas.DataFrame:
    """Merge intermediate period/scenario CSV files into a single CSV.

    Args:
        paths: Intermediate CSV files to merge.
        out_path: Path of the merged CSV to write.

    Returns:
        The merged dataframe.
    """
    merged_df = pandas.concat([pandas.read_csv(p) for p in paths], ignore_index=True)
    merged_df.to_csv(out_path, index=False)
    return merged_df


def run_data_prep(cfg: DataPrepConfig) -> Dict[float, Path]:
    """Run the full data preparation for all requested resolutions.

    Args:
        cfg: Data preparation configuration.

    Returns:
        Mapping of resolution to the merged ML-ready CSV path.
    """
    tutorial_config = load_config(cfg.config_path)
    root_data_dir = cfg.root_data_dir or get_platform_dir(tutorial_config["platform"], tutorial_config)
    output_dir = cfg.output_dir or root_data_dir / "ml_ready"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading data from {root_data_dir}, writing to {output_dir}")

    all_resolutions = {float(k): v for k, v in tutorial_config["resolutions_names"].items()}
    if cfg.resolutions is None:
        resolutions = all_resolutions
    else:
        unknown = set(cfg.resolutions) - set(all_resolutions)
        if unknown:
            raise ValueError(f"Resolutions {sorted(unknown)} not in config {sorted(all_resolutions)}")
        resolutions = {r: all_resolutions[r] for r in cfg.resolutions}

    group_lookup, subgroup_lookup = get_class_lookups(tutorial_config["climate_subgroups"])
    data_paths = build_data_paths(
        root_data_dir,
        get_time_periods(tutorial_config),
        tutorial_config["dataset_prefix"],
        resolutions,
        cfg.file_format,
        tutorial_config["time_dir_template"],
        tutorial_config["fname_template"],
    )
    task_args, inter_paths = build_tasks(
        data_paths, resolutions, output_dir, group_lookup, subgroup_lookup
    )

    merged_paths = {}
    with multiprocessing.Pool(cfg.num_processes) as pool:
        # Process coarsest resolution first, as it is quickest.
        for res in sorted(resolutions, reverse=True):
            print(f"processing resolution {res}")
            # Results are discarded and re-read from disk later to reduce memory usage.
            pool.starmap(process_period_scenario, task_args[res])
            out_path = output_dir / tutorial_config["csv_out_template"].format(
                resolution=resolutions[res]
            )
            merge_intermediate_files(inter_paths[res], out_path)
            print(f"written {out_path}")
            merged_paths[res] = out_path
    return merged_paths


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the data preparation."""
    cfg = parse_args(argv)
    start_dt = datetime.datetime.now()
    run_data_prep(cfg)
    print(f"Data preparation took {datetime.datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
