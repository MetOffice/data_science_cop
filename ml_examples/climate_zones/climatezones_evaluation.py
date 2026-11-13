#!/usr/bin/env python
# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""Evaluate a trained climate-zone classifier.

Script version of the ``climatezones_evaluation.ipynb`` tutorial notebook.

A previously trained model (by default a scikit-learn decision tree) is used to
predict the climate zone of each land point from its climate means. The model
is evaluated on the train, validation and test splits using per-class
precision, recall and F1 scores, plus confusion matrices.

* **Precision** is the proportion of data points predicted as being in a class
  that are actually in that class.
* **Recall** is the proportion of data points that are in a class in reality
  that are predicted as being in that class.

The train/validation/test split uses the same random seed as the training
pipeline, so the splits match those used to train the model.

Example::

    python climatezones_evaluation.py \\
        --config-path config.json \\
        --stats-path stats.json \\
        --model-path decision_tree.pkl \\
        --output-dir ./evaluation_output
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402
import sklearn.metrics  # noqa: E402
import sklearn.preprocessing  # noqa: E402
import sklearn.tree  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")
TARGET_OPTIONS = ("climate_group", "climate_subgroup")
METRIC_FUNCTIONS = {
    "precision": sklearn.metrics.precision_score,
    "recall": sklearn.metrics.recall_score,
    "f1": sklearn.metrics.f1_score,
}


@dataclass
class EvaluationConfig:
    """Runtime evaluation configuration parsed from command-line arguments."""

    config_path: Path
    stats_path: Path
    model_path: Optional[Path]
    algorithm_name: str
    resolution: float
    target_var: str
    test_frac: float
    val_frac: float
    train_new_model: bool
    max_depth: int
    min_samples_leaf: int
    min_samples_split: int
    output_dir: Path


def parse_args(argv: Optional[List[str]] = None) -> EvaluationConfig:
    """Parse command-line arguments for model evaluation.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed evaluation configuration.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained climate-zone classification model"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("config.json"),
        help="Path to the tutorial JSON config file",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("stats.json"),
        help="Path to JSON file of normalisation stats and target classes "
        "calculated on the training set",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the pickled pretrained model. "
        "Defaults to '<algorithm-name>.pkl' in the current directory",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution of the ML-ready dataset in degrees (must be in config)",
    )
    parser.add_argument(
        "--target-var",
        choices=TARGET_OPTIONS,
        default="climate_group",
        help="Target variable: 'climate_group' (5 classes) or "
        "'climate_subgroup' (30 classes)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write metrics and plots to",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.test_frac + args.val_frac < 1.0:
        parser.error("--test-frac + --val-frac must be between 0 and 1")

    return EvaluationConfig(
        config_path=args.config_path,
        stats_path=args.stats_path,
        model_path=args.model_path,
        resolution=args.resolution,
        target_var=args.target_var,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        train_new_model=args.train_new_model,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        output_dir=args.output_dir,
    )


def load_json(path: Path) -> Dict:
    """Load a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


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


def get_ml_ready_path(config: Dict, resolution: float) -> Path:
    """Build the path to the ML-ready CSV file for a given resolution.

    Args:
        config: Tutorial config dictionary.
        resolution: Dataset resolution in degrees.

    Returns:
        Path to the ML-ready CSV file.
    """
    resolutions_dict = {float(k): v for k, v in config["resolutions_names"].items()}
    root_data_dir = get_platform_dir(config["platform"], config)
    fname = config["csv_out_template"].format(resolution=resolutions_dict[resolution])
    return root_data_dir / "ml_ready" / fname


def get_predictors(columns: List[str]) -> List[str]:
    """Select predictor columns: monthly precipitation then temperature means.

    Args:
        columns: Column names of the ML-ready dataframe.

    Returns:
        List of predictor column names.
    """
    precip_mean = [c for c in columns if "precipitation" in c and "mean" in c]
    temp_mean = [c for c in columns if "air_temperature" in c and "mean" in c]
    return precip_mean + temp_mean


def split_data(
    zones_df: pandas.DataFrame,
    test_frac: float,
    val_frac: float,
    random_seed: int,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """Split data into train/validation/test sets, stratified by period and scenario.

    Using the same random seed as the training pipeline ensures the splits
    are identical to those used when training the model.

    Args:
        zones_df: ML-ready climate zones dataframe.
        test_frac: Fraction of the full dataset for the test set.
        val_frac: Fraction of the full dataset for the validation set.
        random_seed: Random seed for sampling.

    Returns:
        Tuple of (train, validation, test) dataframes.
    """
    group_cols = ["period_start", "scenario"]
    test_df = zones_df.groupby(group_cols).sample(frac=test_frac, random_state=random_seed)
    remain_df = zones_df.drop(test_df.index)
    # Rescale so val_frac is a fraction of the full dataset, not the remainder.
    val_frac_sub = val_frac / (1.0 - test_frac)
    val_df = remain_df.groupby(group_cols).sample(frac=val_frac_sub, random_state=random_seed)
    train_df = remain_df.drop(val_df.index)
    return train_df, val_df, test_df


def build_input_scaler(stats_dict: Dict) -> sklearn.preprocessing.StandardScaler:
    """Create a StandardScaler from normalisation stats calculated on the train set.

    Args:
        stats_dict: Dictionary with ``input_mean`` and ``input_scale`` lists.

    Returns:
        A StandardScaler ready to transform inputs.
    """
    input_scaler = sklearn.preprocessing.StandardScaler()
    input_scaler.mean_ = numpy.array(stats_dict["input_mean"])
    input_scaler.scale_ = numpy.array(stats_dict["input_scale"])
    input_scaler.var_ = input_scaler.scale_ ** 2
    input_scaler.n_features_in_ = len(input_scaler.mean_)
    return input_scaler


def build_target_encoder(stats_dict: Dict) -> sklearn.preprocessing.LabelEncoder:
    """Create a LabelEncoder from the target classes stored in the stats file.

    Args:
        stats_dict: Dictionary with a ``target_classes`` list.

    Returns:
        A LabelEncoder ready to encode/decode targets.
    """
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_encoder.classes_ = numpy.array(stats_dict["target_classes"], dtype="object")
    return target_encoder


def prepare_splits(
    splits: Dict[str, pandas.DataFrame],
    predictors: List[str],
    target_var: str,
    input_scaler: sklearn.preprocessing.StandardScaler,
    target_encoder: sklearn.preprocessing.LabelEncoder,
) -> Dict[str, Tuple[numpy.ndarray, numpy.ndarray]]:
    """Scale inputs and encode targets for each data split.

    Args:
        splits: Mapping of split name to dataframe.
        predictors: Predictor column names.
        target_var: Target column name.
        input_scaler: Scaler fitted on the train set.
        target_encoder: Encoder for the target classes.

    Returns:
        Mapping of split name to ``(X, y)`` arrays.
    """
    return {
        name: (
            input_scaler.transform(df[predictors].to_numpy()),
            target_encoder.transform(df[target_var]),
        )
        for name, df in splits.items()
    }


def load_or_train_model(
    cfg: EvaluationConfig,
    x_train: numpy.ndarray,
    y_train: numpy.ndarray,
):
    """Load a pretrained model from a pickle file, or train a new decision tree.

    Args:
        cfg: Evaluation configuration.
        x_train: Scaled training inputs (only used when training a new model).
        y_train: Encoded training targets (only used when training a new model).

    Returns:
        A fitted classifier with a ``predict`` method.
    """
    if not cfg.train_new_model:
        with cfg.model_path.open("rb") as model_file:
            return pickle.load(model_file)

    dt_clf = sklearn.tree.DecisionTreeClassifier(
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
    )
    dt_clf.fit(x_train, y_train)
    return dt_clf


def calculate_metrics(
    y_true_pred: Dict[str, Tuple[numpy.ndarray, numpy.ndarray]],
    class_labels: List[str],
    class_column: str = "climate_group",
) -> pandas.DataFrame:
    """Calculate per-class precision, recall and F1 scores for each data split.

    Args:
        y_true_pred: Mapping of split name to ``(y_true, y_pred)`` encoded arrays.
        class_labels: Class names, in encoded order.
        class_column: Name of the column holding the class labels.

    Returns:
        Dataframe with one row per class and a ``<metric>_<split>`` column for
        each metric and split.
    """
    labels = list(range(len(class_labels)))
    metrics_df = pandas.DataFrame({class_column: class_labels})
    for metric_name, metric_func in METRIC_FUNCTIONS.items():
        for split_name, (y_true, y_pred) in y_true_pred.items():
            metrics_df[f"{metric_name}_{split_name}"] = metric_func(
                y_true, y_pred, average=None, labels=labels, zero_division=0
            )
    return metrics_df


def plot_metrics(
    metrics_df: pandas.DataFrame,
    split_names: List[str],
    class_column: str = "climate_group",
) -> matplotlib.figure.Figure:
    """Plot bar charts of per-class precision, recall and F1 scores.

    Args:
        metrics_df: Output of :func:`calculate_metrics`.
        split_names: Names of the splits to plot.
        class_column: Name of the column holding the class labels.

    Returns:
        The matplotlib figure.
    """
    fig = matplotlib.pyplot.figure(figsize=(15, 5))
    for i_metric, metric_name in enumerate(METRIC_FUNCTIONS, start=1):
        ax = fig.add_subplot(1, len(METRIC_FUNCTIONS), i_metric, title=f"{metric_name} scores")
        metrics_df.plot.bar(
            x=class_column,
            y=[f"{metric_name}_{s}" for s in split_names],
            ax=ax,
        )
    fig.tight_layout()
    return fig


def calculate_confusion_matrix(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    num_classes: int,
) -> numpy.ndarray:
    """Calculate the confusion matrix for encoded targets and predictions.

    Values on the diagonal are correct predictions. Off the diagonal, the value
    in row *i* and column *j* is the number of points with true class *i* that
    the model predicted as class *j*.

    Args:
        y_true: Encoded true labels.
        y_pred: Encoded predicted labels.
        num_classes: Total number of classes, so absent classes still appear.

    Returns:
        A ``(num_classes, num_classes)`` confusion matrix.
    """
    return sklearn.metrics.confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def plot_confusion_matrix(
    conf_matrix: numpy.ndarray,
    class_labels: List[str],
    title: str,
) -> matplotlib.figure.Figure:
    """Plot a confusion matrix.

    Args:
        conf_matrix: Confusion matrix from :func:`calculate_confusion_matrix`.
        class_labels: Class names, in encoded order.
        title: Plot title.

    Returns:
        The matplotlib figure.
    """
    display = sklearn.metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=class_labels)
    display.plot()
    display.ax_.set_title(title)
    return display.figure_


def evaluate(cfg: EvaluationConfig) -> pandas.DataFrame:
    """Run the end-to-end evaluation and write metrics and plots to the output dir.

    Args:
        cfg: Evaluation configuration.

    Returns:
        Dataframe of per-class metrics for each split.
    """
    tutorial_config = load_json(cfg.config_path)
    stats_dict = load_json(cfg.stats_path)

    mlready_data_path = get_ml_ready_path(tutorial_config, cfg.resolution)
    if not mlready_data_path.is_file():
        raise FileNotFoundError(f"ML-ready CSV not found: {mlready_data_path}")
    print(f"Loading data from {mlready_data_path}")
    zones_df = pandas.read_csv(mlready_data_path)

    predictors = get_predictors(list(zones_df.columns))
    train_df, val_df, test_df = split_data(
        zones_df, cfg.test_frac, cfg.val_frac, tutorial_config["random_seed"]
    )
    print(f"Train class counts:\n{train_df[cfg.target_var].value_counts()}")

    # Apply the same transforms as were used on the training data.
    input_scaler = build_input_scaler(stats_dict)
    target_encoder = build_target_encoder(stats_dict)
    class_labels = [str(c) for c in target_encoder.classes_]
    xy_splits = prepare_splits(
        dict(zip(SPLIT_NAMES, (train_df, val_df, test_df))),
        predictors,
        cfg.target_var,
        input_scaler,
        target_encoder,
    )

    model = load_or_train_model(cfg, *xy_splits["train"])
    print(f"Using model: {model}")

    y_true_pred = {name: (y, model.predict(x)) for name, (x, y) in xy_splits.items()}

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = f"{cfg.algorithm_name}_{cfg.target_var}"

    metrics_df = calculate_metrics(y_true_pred, class_labels, cfg.target_var)
    metrics_path = cfg.output_dir / f"{out_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics written to {metrics_path}\n{metrics_df}")

    fig = plot_metrics(metrics_df, list(SPLIT_NAMES), cfg.target_var)
    fig.savefig(cfg.output_dir / f"{out_prefix}_metrics.png")
    matplotlib.pyplot.close(fig)

    for split_name, (y_true, y_pred) in y_true_pred.items():
        conf_matrix = calculate_confusion_matrix(y_true, y_pred, len(class_labels))
        fig = plot_confusion_matrix(
            conf_matrix, class_labels, f"{cfg.algorithm_name} confusion matrix ({split_name})"
        )
        fig.savefig(cfg.output_dir / f"{out_prefix}_confusion_{split_name}.png")
        matplotlib.pyplot.close(fig)

    return metrics_df


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the evaluation."""
    cfg = parse_args(argv)
    start_dt = datetime.datetime.now()
    evaluate(cfg)
    print(f"Evaluation took {datetime.datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
