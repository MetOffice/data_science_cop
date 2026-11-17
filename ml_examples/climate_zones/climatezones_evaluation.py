#!/usr/bin/env python
# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
"""Evaluate a trained climate-zone classifier.

Script version of the ``climatezones_evaluation.ipynb`` tutorial notebook.

The PyTorch neural network trained by ``climatezones_training_torch.py`` is used
to predict the climate zone of each land point from its climate means. The model
is evaluated on the train, validation and test splits using per-class
precision, recall and F1 scores, plus confusion matrices.

* **Precision** is the proportion of data points predicted as being in a class
  that are actually in that class.
* **Recall** is the proportion of data points that are in a class in reality
  that are predicted as being in that class.

The train/validation/test split uses the same random seed as the training
script, so the splits match those used to train the model. The normalisation
stats and target classes are read from the ``stats.json`` file logged to MLflow
by the training script.

Example::

    python climatezones_evaluation.py \\
        --config-path config.json \\
        --stats-path stats.json \\
        --model-path cz_model.pth \\
        --output-dir ./evaluation_output
"""

from __future__ import annotations

import __main__
import argparse
import datetime
import json
import os
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
import torch  # noqa: E402

from climatezones_training_torch import ClimateZoneClassifier  # noqa: E402

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
    model_path: Path
    resolution: float
    training_data_dir: Path
    evaluation_dir: Path


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
        "calculated on the training set (logged to MLflow by the training script)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("cz_model.pth"),
        help="Path to the PyTorch model saved by climatezones_training_torch.py",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution of the ML-ready dataset in degrees (must be in config)",
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write evaluation metrics and plots to",
    )
    args = parser.parse_args(argv)


    return EvaluationConfig(
        config_path=args.config_path,
        stats_path=args.stats_path,
        model_path=args.model_path,
        resolution=args.resolution,
        training_data_dir=args.training_data_dir,
        evaluation_dir=args.evaluation_dir,
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


def load_model(model_path: Path) -> Tuple[torch.nn.Module, torch.device]:
    """Load the PyTorch model saved by ``climatezones_training_torch.py``.

    The training script saves the whole model object with ``torch.save``. When
    it is run as a script, the model class is pickled as
    ``__main__.ClimateZoneClassifier``, so the class is registered in
    ``__main__`` here to allow it to be unpickled.

    Args:
        model_path: Path to the saved model.
        device: Torch device to load the model onto.

    Returns:
        The model, in evaluation mode.
    """
    if not hasattr(__main__, "ClimateZoneClassifier"):
        __main__.ClimateZoneClassifier = ClimateZoneClassifier

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.load(model_path,
                       weights_only=False,
                       map_location=torch.device('cpu'),
                       )
    model.eval()
    return model, device


def predict(
    model: torch.nn.Module,
    x: numpy.ndarray,
    device: str,
    batch_size: int = 32,
) -> numpy.ndarray:
    """Predict encoded classes, taking the class with the highest model output.

    Args:
        model: Trained classifier returning one logit per class.
        x: Scaled inputs of shape ``(n_points, n_features)``.
        device: Torch device to run inference on.

    Returns:
        Encoded class predictions of shape ``(n_points,)``.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for x_batch in torch.split(x_tensor, batch_size):
            logits = model(x_batch.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu())
    if not predictions:
        return numpy.empty(0, dtype=numpy.int64)
    return torch.cat(predictions).numpy()


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

def _get_data_splits(args: EvaluationConfig, tutorial_config: Dict) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, Dict, List[str]]:
    """Load the ML-ready data and split into train/val/test sets."""

    stats_dict = load_json(args.stats_path)
    mlready_data_path = get_ml_ready_path(tutorial_config, resolution=args.resolution)

    if not mlready_data_path.is_file():
        raise FileNotFoundError(f"ML-ready CSV not found: {mlready_data_path}")
    print(f"Loading data from {mlready_data_path}")
    zones_df = pandas.read_csv(mlready_data_path)

    train_df, val_df, test_df = split_data(
        zones_df,
        tutorial_config["test_fraction"],
        tutorial_config["val_fraction"],
        tutorial_config["random_seed"],
    )
    predictors = get_predictors(list(train_df.columns))

    return train_df, val_df, test_df, stats_dict, predictors


def evaluate(cfg: EvaluationConfig) -> pandas.DataFrame:
    """Run the end-to-end evaluation and write metrics and plots to the output dir.

    Args:
        cfg: Evaluation configuration.

    Returns:
        Dataframe of per-class metrics for each split.
    """
    tutorial_config = load_json(cfg.config_path)

    train_df, val_df, test_df, stats_dict, predictors = _get_data_splits(cfg, tutorial_config)

    print(f"Train class counts:\n{train_df[tutorial_config['target_var']].value_counts()}")

    # Apply the same transforms as were used on the training data.
    input_scaler = build_input_scaler(stats_dict)
    target_encoder = build_target_encoder(stats_dict)
    class_labels = [str(c) for c in target_encoder.classes_]
    xy_splits = prepare_splits(
        dict(zip(SPLIT_NAMES, (train_df, val_df, test_df))),
        predictors,
        tutorial_config['target_var'],
        input_scaler,
        target_encoder,
    )

    model, device = load_model(cfg.model_path)

    y_true_pred = {
        name: (y, predict(model, x, device))
        for name, (x, y) in xy_splits.items()
    }

    cfg.evaluation_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = calculate_metrics(y_true_pred, class_labels, tutorial_config["target_var"])
    metrics_path = cfg.evaluation_dir / f"climate_zones_{tutorial_config['target_var']}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics written to {metrics_path}\n{metrics_df}")

    fig = plot_metrics(metrics_df, list(SPLIT_NAMES), tutorial_config["target_var"])
    fig.savefig(cfg.evaluation_dir / f"climate_zones_{tutorial_config['target_var']}_metrics.png")
    matplotlib.pyplot.close(fig)

    for split_name, (y_true, y_pred) in y_true_pred.items():
        conf_matrix = calculate_confusion_matrix(y_true, y_pred, len(class_labels))
        fig = plot_confusion_matrix(
            conf_matrix, class_labels, f"climatezones_confusion matrix ({split_name})"
        )
        fig.savefig(cfg.evaluation_dir / f"climate_zones_{tutorial_config['target_var']}_{split_name}.png")
        matplotlib.pyplot.close(fig)

    return metrics_df


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the evaluation."""
    cfg = parse_args(argv)

    start_dt = datetime.datetime.now()
    metrics_df = evaluate(cfg)

    metrics_path = cfg.evaluation_dir / 'cs_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics written to {metrics_path}")

    print(f"Evaluation took {datetime.datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
