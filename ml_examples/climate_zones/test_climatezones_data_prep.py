# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""Tests for the key functions in ``climatezones_data_prep.py``.

Run with::

    python -m pytest test_climatezones_data_prep.py
"""

import json
from pathlib import Path

import numpy
import pandas
import pytest
import xarray

import climatezones_data_prep as czp

SUBGROUPS = {"non-land": "None", "Af": "Tropical", "BWh": "Arid", "Cfb": "Temperate", "ET": "Polar"}
LAT = [10.5, 0.5, -0.5]
LON = [-1.5, -0.5, 0.5, 1.5]


@pytest.fixture
def tutorial_config() -> dict:
    """Minimal tutorial config matching the structure of ``config.json``."""
    return {
        "platform": "test",
        "default_dirs": {},
        "climate_subgroups": SUBGROUPS,
        "resolutions_names": {"0.5": "0p5", "1.0": "1p0"},
        "dataset_prefix": {
            "climate_mean": "ensemble_mean",
            "climate_std": "ensemble_std",
            "climate_zone": "koppen_geiger",
        },
        "fname_template": "{prefix}_{res}.{suffix}",
        "time_dir_template": "{start_year}_{end_year}",
        "csv_out_template": "climate_zones_{resolution}.csv",
        "future_scenarios": ["ssp126", "ssp585"],
        "historic_scenarios": ["historic"],
    }


def make_datasets(seed: int = 0) -> dict:
    """Create synthetic gridded datasets like the source netCDF files.

    The first latitude row is set to non-land (NaN climate data, class 0).
    """
    rng = numpy.random.default_rng(seed)
    shape = (12, len(LAT), len(LON))
    coords = {"time": numpy.arange(1.0, 13.0), "lat": LAT, "lon": LON}

    def climate_fields():
        precip = rng.random(shape).astype("float32")
        temp = rng.random(shape).astype("float32")
        precip[:, 0, :] = numpy.nan
        temp[:, 0, :] = numpy.nan
        return xarray.Dataset(
            {
                "precipitation": (("time", "lat", "lon"), precip),
                "air_temperature": (("time", "lat", "lon"), temp),
            },
            coords=coords,
        )

    kg_class = rng.integers(1, len(SUBGROUPS), size=shape[1:]).astype("float32")
    kg_class[0, :] = 0
    zone = xarray.Dataset(
        {"kg_class": (("lat", "lon"), kg_class)},
        coords={"lat": LAT, "lon": LON},
    )
    return {"climate_mean": climate_fields(), "climate_std": climate_fields(), "climate_zone": zone}


def write_source_data(root_dir: Path, config: dict, resolutions=("1p0",)) -> None:
    """Write synthetic source netCDF files for every period and scenario."""
    for period, scenarios in czp.get_time_periods(config).items():
        for scenario in scenarios:
            datasets = make_datasets(seed=period[0])
            for res_str in resolutions:
                for ds_id, prefix in config["dataset_prefix"].items():
                    path = czp.get_data_path(
                        root_dir, period, scenario, prefix, res_str, "nc",
                        config["time_dir_template"], config["fname_template"],
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    datasets[ds_id].to_netcdf(path)


def test_get_platform_dir_known_platform():
    config = {"default_dirs": {"jasmin": "/gws/data"}}
    assert czp.get_platform_dir("jasmin", config) == Path("/gws/data/climate_zones")


def test_get_platform_dir_unknown_platform_falls_back_to_home(monkeypatch):
    monkeypatch.setenv("HOME", "/fake/home")
    assert czp.get_platform_dir("other", {"default_dirs": {}}) == Path("/fake/home/climate_zones")


def test_get_time_periods(tutorial_config):
    periods = czp.get_time_periods(tutorial_config)
    assert len(periods) == 6
    assert periods[(1901, 1930)] == ["historic"]
    assert periods[(2071, 2099)] == ["ssp126", "ssp585"]


def test_get_class_lookups():
    group_lookup, subgroup_lookup = czp.get_class_lookups(SUBGROUPS)
    assert subgroup_lookup == ["non-land", "Af", "BWh", "Cfb", "ET"]
    assert group_lookup == ["none", "A", "B", "C", "E"]


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("historic", "/root/1991_2020/ensemble_mean_1p0.nc"),
        ("ssp245", "/root/1991_2020/ssp245/ensemble_mean_1p0.nc"),
    ],
)
def test_get_data_path(scenario, expected):
    path = czp.get_data_path(
        Path("/root"), (1991, 2020), scenario, "ensemble_mean", "1p0", "nc",
        "{start_year}_{end_year}", "{prefix}_{res}.{suffix}",
    )
    assert path == Path(expected)


def test_build_data_paths_structure(tutorial_config):
    data_paths = czp.build_data_paths(
        Path("/root"),
        czp.get_time_periods(tutorial_config),
        tutorial_config["dataset_prefix"],
        {1.0: "1p0", 0.5: "0p5"},
        "nc",
        tutorial_config["time_dir_template"],
        tutorial_config["fname_template"],
    )
    assert set(data_paths[(2041, 2070)]) == {"ssp126", "ssp585"}
    assert set(data_paths[(1901, 1930)]["historic"]) == set(tutorial_config["dataset_prefix"])
    assert data_paths[(2041, 2070)]["ssp126"]["climate_zone"][0.5] == Path(
        "/root/2041_2070/ssp126/koppen_geiger_0p5.nc"
    )


def test_create_dataframe_period():
    group_lookup, subgroup_lookup = czp.get_class_lookups(SUBGROUPS)
    datasets = make_datasets()
    df = czp.create_dataframe_period(datasets, (1961, 1990), "historic", group_lookup, subgroup_lookup)

    # Non-land first latitude row is dropped.
    assert len(df) == (len(LAT) - 1) * len(LON)
    assert 10.5 not in df["lat"].values
    for var in ("precipitation", "air_temperature"):
        for suffix in ("mean", "std"):
            assert f"{var}_1.0_{suffix}" in df.columns
            assert f"{var}_12.0_{suffix}" in df.columns
    assert (df["period_start"] == 1961).all()
    assert (df["period_end"] == 1990).all()
    assert (df["scenario"] == "historic").all()

    row = df.iloc[0]
    expected_mean = datasets["climate_mean"]["precipitation"].sel(
        time=1.0, lat=row["lat"], lon=row["lon"]
    )
    assert row["precipitation_1.0_mean"] == pytest.approx(float(expected_mean))
    class_ids = df["kg_class"].astype(int)
    assert list(df["climate_subgroup"]) == [subgroup_lookup[c] for c in class_ids]
    assert list(df["climate_group"]) == [group_lookup[c] for c in class_ids]


def test_build_tasks(tutorial_config, tmp_path):
    group_lookup, subgroup_lookup = czp.get_class_lookups(SUBGROUPS)
    resolutions = {1.0: "1p0", 0.5: "0p5"}
    data_paths = czp.build_data_paths(
        Path("/root"), czp.get_time_periods(tutorial_config), tutorial_config["dataset_prefix"],
        resolutions, "nc", tutorial_config["time_dir_template"], tutorial_config["fname_template"],
    )
    task_args, inter_paths = czp.build_tasks(
        data_paths, resolutions, tmp_path, group_lookup, subgroup_lookup
    )
    # 4 historic periods x 1 scenario + 2 future periods x 2 scenarios
    assert len(task_args[1.0]) == len(inter_paths[1.0]) == 8
    input_paths, res, period, scenario, out_path, *_ = task_args[0.5][-1]
    assert res == 0.5
    assert (period, scenario) == ((2071, 2099), "ssp585")
    assert out_path == tmp_path / "climate_zones_2071_2099_ssp585_0p5.csv"
    assert input_paths["climate_mean"] == Path("/root/2071_2099/ssp585/ensemble_mean_0p5.nc")


def test_process_period_scenario_writes_csv(tmp_path):
    group_lookup, subgroup_lookup = czp.get_class_lookups(SUBGROUPS)
    path_dict = {}
    for ds_id, ds in make_datasets().items():
        path_dict[ds_id] = tmp_path / f"{ds_id}.nc"
        ds.to_netcdf(path_dict[ds_id])
    out_path = tmp_path / "out.csv"

    df = czp.process_period_scenario(
        path_dict, 1.0, (2041, 2070), "ssp126", out_path, group_lookup, subgroup_lookup
    )
    df_disk = pandas.read_csv(out_path)
    assert len(df_disk) == len(df)
    assert list(df_disk.columns) == list(df.columns)


def test_merge_intermediate_files(tmp_path):
    paths = []
    for i in range(3):
        paths.append(tmp_path / f"part{i}.csv")
        pandas.DataFrame({"a": [i, i], "b": ["x", "y"]}).to_csv(paths[-1], index=False)
    out_path = tmp_path / "merged.csv"
    merged = czp.merge_intermediate_files(paths, out_path)
    assert list(merged.index) == list(range(6))
    pandas.testing.assert_frame_equal(pandas.read_csv(out_path), merged)


def test_parse_args_defaults():
    cfg = czp.parse_args([])
    assert cfg.config_path == Path("config.json")
    assert cfg.root_data_dir is None
    assert cfg.output_dir is None
    assert cfg.resolutions is None
    assert cfg.num_processes == 4
    assert cfg.file_format == "nc"


def test_parse_args_rejects_invalid_num_processes():
    with pytest.raises(SystemExit):
        czp.parse_args(["--num-processes", "0"])


def test_main_end_to_end(tmp_path, tutorial_config):
    root_dir = tmp_path / "source"
    out_dir = tmp_path / "ml_ready"
    write_source_data(root_dir, tutorial_config)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tutorial_config))

    czp.main([
        "--config-path", str(config_path),
        "--root-data-dir", str(root_dir),
        "--output-dir", str(out_dir),
        "--resolutions", "1.0",
        "--num-processes", "2",
    ])

    merged = pandas.read_csv(out_dir / "climate_zones_1p0.csv")
    land_points = (len(LAT) - 1) * len(LON)
    assert len(merged) == 8 * land_points
    assert set(merged["scenario"]) == {"historic", "ssp126", "ssp585"}
    assert len(list(out_dir.glob("climate_zones_*_*_*_1p0.csv"))) == 8
    assert not (out_dir / "climate_zones_0p5.csv").exists()


def test_run_data_prep_rejects_unknown_resolution(tmp_path, tutorial_config):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tutorial_config))
    with pytest.raises(ValueError):
        czp.main([
            "--config-path", str(config_path),
            "--root-data-dir", str(tmp_path),
            "--resolutions", "2.0",
        ])
