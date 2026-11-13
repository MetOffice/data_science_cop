# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""Tests for the key functions in ``climatezones_evaluation.py``.

Run with::

    python -m pytest test_climatezones_evaluation.py
"""

import json
import pickle
from pathlib import Path

import numpy
import pandas
import pytest
import sklearn.tree

import climatezones_evaluation as cze

CLASSES = ["A", "B", "C", "D", "E"]
MONTHS = range(1, 13)


@pytest.fixture
def zones_df() -> pandas.DataFrame:
    """Synthetic ML-ready dataframe resembling the climate zones CSV."""
    rng = numpy.random.default_rng(0)
    n_rows = 400
    data = {}
    for month in MONTHS:
        data[f"precipitation_{month}.0_mean"] = rng.random(n_rows) * 100
        data[f"precipitation_{month}.0_std"] = rng.random(n_rows)
        data[f"air_temperature_{month}.0_mean"] = rng.random(n_rows) * 40 - 10
        data[f"air_temperature_{month}.0_std"] = rng.random(n_rows)
    data["period_start"] = numpy.repeat([1961, 1991], n_rows // 2)
    data["scenario"] = numpy.tile(["historic", "ssp245"], n_rows // 2)
    data["climate_group"] = numpy.array(CLASSES)[numpy.arange(n_rows) % len(CLASSES)]
    return pandas.DataFrame(data)


@pytest.fixture
def tutorial_config(tmp_path: Path) -> dict:
    """Minimal tutorial config pointing at a temporary data directory."""
    return {
        "platform": "test",
        "default_dirs": {"test": str(tmp_path)},
        "random_seed": 12345,
        "resolutions_names": {"0.5": "0p5", "1.0": "1p0"},
        "csv_out_template": "climate_zones_{resolution}.csv",
    }


def test_get_platform_dir_known_platform(tutorial_config):
    root = cze.get_platform_dir("test", tutorial_config)
    assert root == Path(tutorial_config["default_dirs"]["test"]) / "climate_zones"


def test_get_platform_dir_unknown_platform_falls_back_to_home(tutorial_config, monkeypatch):
    monkeypatch.setenv("HOME", "/fake/home")
    assert cze.get_platform_dir("unknown", tutorial_config) == Path("/fake/home/climate_zones")


def test_get_ml_ready_path(tutorial_config):
    path = cze.get_ml_ready_path(tutorial_config, 1.0)
    expected = (
        Path(tutorial_config["default_dirs"]["test"])
        / "climate_zones"
        / "ml_ready"
        / "climate_zones_1p0.csv"
    )
    assert path == expected


def test_get_predictors_selects_precip_then_temp_means(zones_df):
    predictors = cze.get_predictors(list(zones_df.columns))
    assert len(predictors) == 24
    assert all("mean" in p for p in predictors)
    assert all("precipitation" in p for p in predictors[:12])
    assert all("air_temperature" in p for p in predictors[12:])


def test_split_data_is_disjoint_and_complete(zones_df):
    train_df, val_df, test_df = cze.split_data(zones_df, 0.1, 0.1, random_seed=1)
    indices = [set(df.index) for df in (train_df, val_df, test_df)]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert set().union(*indices) == set(zones_df.index)
    assert len(test_df) == pytest.approx(0.1 * len(zones_df), abs=4)
    assert len(val_df) == pytest.approx(0.1 * len(zones_df), abs=4)


def test_split_data_is_reproducible(zones_df):
    first = cze.split_data(zones_df, 0.1, 0.1, random_seed=42)
    second = cze.split_data(zones_df, 0.1, 0.1, random_seed=42)
    for df1, df2 in zip(first, second):
        assert df1.index.equals(df2.index)


def test_build_input_scaler_matches_fitted_scaler(zones_df):
    predictors = cze.get_predictors(list(zones_df.columns))
    x = zones_df[predictors].to_numpy()
    stats = {"input_mean": x.mean(axis=0).tolist(), "input_scale": x.std(axis=0).tolist()}
    scaled = cze.build_input_scaler(stats).transform(x)
    numpy.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
    numpy.testing.assert_allclose(scaled.std(axis=0), 1.0)


def test_build_target_encoder_round_trip():
    encoder = cze.build_target_encoder({"target_classes": CLASSES})
    labels = ["C", "A", "E"]
    encoded = encoder.transform(labels)
    numpy.testing.assert_array_equal(encoded, [2, 0, 4])
    assert list(encoder.inverse_transform(encoded)) == labels


def test_calculate_metrics_perfect_and_imperfect_predictions():
    y_true = numpy.array([0, 0, 1, 1, 2, 2])
    y_pred_bad = numpy.array([0, 1, 1, 1, 2, 2])
    metrics_df = cze.calculate_metrics(
        {"train": (y_true, y_true), "val": (y_true, y_pred_bad)},
        ["A", "B", "C"],
    )
    assert list(metrics_df["climate_group"]) == ["A", "B", "C"]
    expected_cols = {f"{m}_{s}" for m in ("precision", "recall", "f1") for s in ("train", "val")}
    assert expected_cols <= set(metrics_df.columns)
    numpy.testing.assert_allclose(metrics_df["precision_train"], 1.0)
    numpy.testing.assert_allclose(metrics_df["precision_val"], [1.0, 2 / 3, 1.0])
    numpy.testing.assert_allclose(metrics_df["recall_val"], [0.5, 1.0, 1.0])


def test_calculate_metrics_handles_absent_class():
    y_true = numpy.array([0, 1, 0, 1])
    metrics_df = cze.calculate_metrics({"test": (y_true, y_true)}, ["A", "B", "C"])
    assert len(metrics_df) == 3
    assert metrics_df.loc[2, "recall_test"] == 0.0


def test_calculate_confusion_matrix():
    y_true = numpy.array([0, 0, 1, 2])
    y_pred = numpy.array([0, 1, 1, 2])
    conf = cze.calculate_confusion_matrix(y_true, y_pred, num_classes=4)
    assert conf.shape == (4, 4)
    assert conf[0, 0] == 1 and conf[0, 1] == 1
    assert numpy.trace(conf) == 3
    assert conf[3].sum() == 0


def test_parse_args_defaults():
    cfg = cze.parse_args([])
    assert cfg.model_path == Path("decision_tree.pkl")
    assert cfg.target_var == "climate_group"
    assert cfg.resolution == 1.0
    assert not cfg.train_new_model


def test_parse_args_rejects_invalid_fractions():
    with pytest.raises(SystemExit):
        cze.parse_args(["--test-frac", "0.6", "--val-frac", "0.5"])


def _write_inputs(tmp_path, zones_df, tutorial_config):
    """Write config, stats and ML-ready CSV files for an end-to-end run."""
    csv_path = cze.get_ml_ready_path(tutorial_config, 1.0)
    csv_path.parent.mkdir(parents=True)
    zones_df.to_csv(csv_path, index=False)

    predictors = cze.get_predictors(list(zones_df.columns))
    x = zones_df[predictors].to_numpy()
    stats = {
        "input_mean": x.mean(axis=0).tolist(),
        "input_scale": x.std(axis=0).tolist(),
        "target_classes": CLASSES,
    }
    config_path = tmp_path / "config.json"
    stats_path = tmp_path / "stats.json"
    config_path.write_text(json.dumps(tutorial_config))
    stats_path.write_text(json.dumps(stats))
    return config_path, stats_path


def test_main_with_pretrained_model(tmp_path, zones_df, tutorial_config):
    config_path, stats_path = _write_inputs(tmp_path, zones_df, tutorial_config)
    model_path = tmp_path / "decision_tree.pkl"
    model = sklearn.tree.DecisionTreeClassifier(max_depth=3)
    model.fit(numpy.random.default_rng(1).random((50, 24)), numpy.arange(50) % len(CLASSES))
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)
    out_dir = tmp_path / "out"

    cze.main([
        "--config-path", str(config_path),
        "--stats-path", str(stats_path),
        "--model-path", str(model_path),
        "--output-dir", str(out_dir),
    ])

    metrics_df = pandas.read_csv(out_dir / "decision_tree_climate_group_metrics.csv")
    assert list(metrics_df["climate_group"]) == CLASSES
    assert (out_dir / "decision_tree_climate_group_metrics.png").is_file()
    for split_name in cze.SPLIT_NAMES:
        assert (out_dir / f"decision_tree_climate_group_confusion_{split_name}.png").is_file()


def test_main_train_new_model(tmp_path, zones_df, tutorial_config):
    config_path, stats_path = _write_inputs(tmp_path, zones_df, tutorial_config)
    out_dir = tmp_path / "out"

    cze.main([
        "--config-path", str(config_path),
        "--stats-path", str(stats_path),
        "--train-new-model",
        "--output-dir", str(out_dir),
    ])

    metrics_df = pandas.read_csv(out_dir / "decision_tree_climate_group_metrics.csv")
    scores = metrics_df.drop(columns="climate_group")
    assert ((scores >= 0) & (scores <= 1)).all().all()


def test_main_missing_data_raises(tmp_path, tutorial_config):
    config_path = tmp_path / "config.json"
    stats_path = tmp_path / "stats.json"
    config_path.write_text(json.dumps(tutorial_config))
    stats_path.write_text(json.dumps({"input_mean": [], "input_scale": [], "target_classes": []}))
    with pytest.raises(FileNotFoundError):
        cze.main(["--config-path", str(config_path), "--stats-path", str(stats_path)])
