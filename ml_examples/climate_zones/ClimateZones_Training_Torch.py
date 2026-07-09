#!/usr/bin/env python
"""Train a climate-zone classifier with PyTorch and track runs in MLflow.

This script refactors the original notebook export into a standard Python CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing
import torch


@dataclass
class TrainConfig:
    """Runtime training configuration parsed from command-line arguments."""

    config_path: Path
    learning_rate: float
    batch_size: int
    num_epochs: int


class ClimateZonesDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper for climate-zone tabular data."""

    def __init__(
        self,
        df_ml: pd.DataFrame,
        predictor_features: List[str],
        target_feature: str,
        scaler: sklearn.preprocessing.StandardScaler,
        label_encoder: sklearn.preprocessing.LabelEncoder,
    ) -> None:
        self._df_ml = df_ml.reset_index(drop=True)

        x_scaled = scaler.transform(self._df_ml[predictor_features])
        y_encoded = label_encoder.transform(self._df_ml[target_feature])

        self.x = torch.tensor(x_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_encoded, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ClimateZoneClassifier(torch.nn.Module):
    """Simple feed-forward neural network for climate-zone classification."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, 60)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(60, 60)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(60, 60)
        self.act3 = torch.nn.ReLU()
        self.output = torch.nn.Linear(60, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        return self.output(x)


def parse_args() -> TrainConfig:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train climate-zone PyTorch model")
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the JSON config file",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )
    args = parser.parse_args()

    return TrainConfig(
        config_path=args.config_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )


def get_platform_dir(select_platform: str, config: Dict) -> Path:
    """Resolve root data directory from config platform mapping."""
    try:
        root_path = Path(config["default_dirs"][select_platform]) / "climate_zones"
    except KeyError:
        root_path = Path(os.environ["HOME"]) / "climate_zones"
    return root_path


def get_classification_metrics(
    model: torch.nn.Module,
    dataset: ClimateZonesDataset,
    set_label: str,
    class_labels: List[str],
    device: torch.device,
) -> pd.DataFrame:
    """Calculate per-class precision and recall for a dataset split."""
    model.eval()
    with torch.no_grad():
        logits = model(dataset.x.to(device)).cpu().numpy()
    y_pred = np.argmax(logits, axis=1)
    y_true = dataset.y.numpy()

    precision = sklearn.metrics.precision_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(class_labels))),
        zero_division=0,
    )
    recall = sklearn.metrics.recall_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(class_labels))),
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "climate_group": class_labels,
            f"precision_{set_label}": precision,
            f"recall_{set_label}": recall,
        }
    )


def prepare_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], str]:
    """Load prepared CSV data and return train/val/test splits plus feature metadata."""
    current_platform = config["platform"]
    root_data_dir = get_platform_dir(current_platform, config)
    ml_ready_dir = root_data_dir / "ml_ready"

    resolutions_dict = {float(k): v for k, v in config["resolutions_names"].items()}
    current_res = 1.0
    ml_ready_fname_template = config["csv_out_template"]
    mlready_data_path = ml_ready_dir / ml_ready_fname_template.format(
        resolution=resolutions_dict[current_res]
    )

    if not mlready_data_path.is_file():
        raise FileNotFoundError(f"ML-ready CSV not found: {mlready_data_path}")

    zones_df = pd.read_csv(mlready_data_path)

    predictors_dict = {
        "precip_mean": [
            c for c in zones_df.columns if "precipitation" in c and "mean" in c
        ],
        "temp_mean": [
            c for c in zones_df.columns if "air_temperature" in c and "mean" in c
        ],
    }
    predictors = predictors_dict["precip_mean"] + predictors_dict["temp_mean"]
    target_var = "climate_group"

    random_seed = config["random_seed"]
    test_frac = 0.1
    val_frac = 0.1
    val_frac_sub = val_frac / (1.0 - test_frac)

    test_df = zones_df.groupby(["period_start", "scenario"]).sample(
        frac=test_frac,
        random_state=random_seed,
    )
    remain_df = zones_df.drop(test_df.index)
    val_df = remain_df.groupby(["period_start", "scenario"]).sample(
        frac=val_frac_sub,
        random_state=random_seed,
    )
    train_df = remain_df.drop(val_df.index)

    return train_df, val_df, test_df, predictors, target_var


def train_model(cfg: TrainConfig) -> None:
    """Execute end-to-end training and logging with MLflow."""
    with cfg.config_path.open("r", encoding="utf-8") as fp:
        tutorial_config = json.load(fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df, predictors, target_var = prepare_data(tutorial_config)

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_df[predictors])

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(train_df[target_var])

    cz_train_ds = ClimateZonesDataset(
        train_df, predictors, target_var, scaler, label_encoder
    )
    cz_val_ds = ClimateZonesDataset(val_df, predictors, target_var, scaler, label_encoder)
    cz_test_ds = ClimateZonesDataset(test_df, predictors, target_var, scaler, label_encoder)

    train_loader = torch.utils.data.DataLoader(
        cz_train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=1,
    )
    val_loader = torch.utils.data.DataLoader(
        cz_val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=1,
    )

    model = ClimateZoneClassifier(
        input_dim=len(predictors),
        num_classes=len(label_encoder.classes_),
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    mlflow_port = os.environ.get("MLFLOW_PORT", "4455")
    mlflow_server_uri = f"http://localhost:{mlflow_port}"
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.pytorch.autolog()

    exp_name = "climate_zones_torch_nn"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    exp = mlflow.get_experiment_by_name(exp_name)

    training_params = {
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
    }

    scaler_stats = {
        "input_mean": [float(v) for v in scaler.mean_],
        "input_scale": [float(v) for v in scaler.scale_],
        "target_classes": [str(c) for c in label_encoder.classes_],
    }

    with mlflow.start_run(experiment_id=exp.experiment_id) as current_run:
        print(f"MLflow run id: {current_run.info.run_id}")
        mlflow.log_params(training_params)
        mlflow.log_dict(scaler_stats, "stats.json")

        for epoch in range(cfg.num_epochs):
            model.train()
            epoch_loss_train = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x.to(device))
                loss = loss_fn(predictions, batch_y.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss_train += float(loss.detach().cpu().item())
            epoch_loss_train /= len(train_loader)

            model.eval()
            epoch_loss_val = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    epoch_loss_val += float(
                        loss_fn(model(x_val.to(device)), y_val.to(device)).item()
                    )
            epoch_loss_val /= len(val_loader)

            mlflow.log_metrics(
                {
                    "cross_entropy_train": epoch_loss_train,
                    "cross_entropy_val": epoch_loss_val,
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch + 1}/{cfg.num_epochs} - "
                f"train_loss={epoch_loss_train:.4f}, val_loss={epoch_loss_val:.4f}"
            )

        class_labels = [str(c) for c in label_encoder.classes_]
        train_metrics = get_classification_metrics(
            model, cz_train_ds, "train", class_labels, device
        )
        val_metrics = get_classification_metrics(
            model, cz_val_ds, "val", class_labels, device
        )
        test_metrics = get_classification_metrics(
            model, cz_test_ds, "test", class_labels, device
        )

        metrics_df = train_metrics.merge(val_metrics, on="climate_group").merge(
            test_metrics, on="climate_group"
        )
        mlflow.log_table(metrics_df, "metrics.json")

        model_path = Path("cz_model.pth")
        torch.save(model, model_path)
        mlflow.log_artifact(str(model_path))


def main() -> None:
    """Parse args and run training."""
    cfg = parse_args()
    train_model(cfg)


if __name__ == "__main__":
    main()
