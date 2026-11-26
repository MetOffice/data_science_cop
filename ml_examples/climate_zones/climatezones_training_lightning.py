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

import numpy as np
import pandas as pd

import sklearn.metrics
import sklearn.preprocessing
import torch
import lightning
import mlflow

@dataclass
class TrainConfig:
    """Runtime training configuration parsed from command-line arguments."""

    config_path: Path
    learning_rate: float
    batch_size: int
    num_epochs: int
    model_output_dir: Path


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


class ClimateZoneClassifier(lightning.LightningModule):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 loss_function,
                 learning_rate: float,
                 ):
        super().__init__()
        self._create_model(input_dim, num_classes)
        self._loss_fn = loss_function
        self._learning_rate = learning_rate

    def _create_model(self, input_dim: int, num_classes: int):
        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, num_classes),
            # torch.nn.Sigmoid(),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self._model(x)
        return x

    # def forward(self, inputs, target):
    #     return self._model(inputs, target)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        batch_train_loss = self._loss_fn(y_pred, y)
        self.log('train_loss', batch_train_loss)
        return batch_train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        batch_val_loss = self._loss_fn(y_pred, y)
        self.log('val_loss', batch_val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        batch_test_loss = self._loss_fn(y_pred, y)
        self.log('test_loss', batch_test_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     self._learning_rate
                                     )
        return optimizer


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

    parser.add_argument(
        "--model-output-dir",
        type=Path,
        required=True,
        help="Directory to save the trained model",
    )

    args = parser.parse_args()

    return TrainConfig(
        config_path=args.config_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        model_output_dir=args.model_output_dir,
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

def get_data_loaders(cfg: TrainConfig, tutorial_config: Dict, batch_size: int) -> Tuple[ClimateZonesDataset, ClimateZonesDataset, ClimateZonesDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str], str]:
    """Prepare data loaders for training, validation, and testing."""
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

    cz_train_loader = torch.utils.data.DataLoader(
        cz_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    cz_val_loader = torch.utils.data.DataLoader(
        cz_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    cz_test_loader = torch.utils.data.DataLoader(
        cz_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )


    scaler_stats = {
        "input_mean": [float(v) for v in scaler.mean_],
        "input_scale": [float(v) for v in scaler.scale_],
        "target_classes": [str(c) for c in label_encoder.classes_],
    }

    return cz_train_ds, cz_val_ds, cz_test_ds, cz_train_loader, cz_val_loader, cz_test_loader, predictors, target_var, scaler_stats

def train_model(cfg: TrainConfig, tutorial_config: Dict, cz_train_loader, cz_val_loader, cz_test_loader, predictors, target_var, scaler_stats) -> ClimateZoneClassifier:
    """Execute end-to-end training and logging with MLflow."""


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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


    cz_classifier = ClimateZoneClassifier(
        input_dim=len(predictors),
        num_classes=len(scaler_stats["target_classes"]),
        loss_function=torch.nn.CrossEntropyLoss(),
        learning_rate=cfg.learning_rate,
    ).to(device)


    num_models_to_save = 3  # save the best 2 models
    monitor_target = 'val_loss'

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath='/gws/ssde/j25b/mohc_shared/users/shaddad/experiments/climate_zones/',
        save_top_k=num_models_to_save,
        monitor=monitor_target,
        )

    trainer = lightning.Trainer(max_epochs=training_params["num_epochs"],
                                callbacks=[checkpoint_callback],
                                )

    with mlflow.start_run(experiment_id=exp.experiment_id) as current_run:
        print(f"MLflow run id: {current_run.info.run_id}")
        mlflow.log_params(training_params)
        mlflow.log_dict(scaler_stats, "stats.json")

        trainer.fit(
            model=cz_classifier,
            train_dataloaders=cz_train_loader,
            val_dataloaders=cz_val_loader,

        )

        class_labels = scaler_stats["target_classes"]
        train_metrics = get_classification_metrics(
            cz_classifier, cz_train_ds, "train", class_labels, device
        )
        val_metrics = get_classification_metrics(
            cz_classifier, cz_val_ds, "val", class_labels, device
        )
        test_metrics = get_classification_metrics(
            cz_classifier, cz_test_ds, "test", class_labels, device
        )

        metrics_df = train_metrics.merge(val_metrics, on="climate_group").merge(
            test_metrics, on="climate_group"
        )
        mlflow.log_table(metrics_df, "metrics.json")
    return cz_classifier

def main() -> None:
    """Parse args and run training."""
    cfg = parse_args()
    with cfg.config_path.open("r", encoding="utf-8") as fp:
        tutorial_config = json.load(fp)
    (cz_train_ds,
     cz_val_ds,
     cz_test_ds,
     cz_train_loader,
     cz_val_loader,
     cz_test_loader,
     predictors,
     target_var,
     scaler_stats) = get_data_loaders(cfg, tutorial_config, cfg.batch_size)
    cz_classifier = train_model(cfg, tutorial_config, cz_train_loader, cz_val_loader, cz_test_loader, predictors, target_var, scaler_stats)
    print(cz_classifier.summary())

if __name__ == "__main__":
    main()
