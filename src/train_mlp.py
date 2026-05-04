import argparse
import logging
import tempfile
import joblib
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data import load_model_ready_dataset, split_features_target
from src.evaluation.metrics import evaluate_torch_binary_classifier
from src.features import build_preprocessor
from src.models.mlp import TelcoMLP
from src.training.mlp import EarlyStopping, fit

logger = logging.getLogger(__name__)


def prepare_data(
    batch_size: int,
    val_size: float,
) -> tuple[DataLoader, DataLoader, torch.Tensor, pd.Series, int]:
    """Carrega o dataset processado e converte para tensores."""
    torch.manual_seed(RANDOM_SEED)
    df = load_model_ready_dataset()
    x, y = split_features_target(df)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor_pipeline.pkl")

    x_train_tensor = torch.tensor(x_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val_processed, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test_processed, dtype=torch.float32)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = x_train_tensor.shape[1]

    return train_loader, val_loader, x_test_tensor, y_test, input_dim


def run_training(
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    dropout_rate: float,
    val_size: float,
    patience: int,
    min_delta: float,
    monitor: str,
) -> dict[str, float]:
    """Função orquestradora: gerencia MLflow, pipeline de dados, treino e avaliação."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_loader, val_loader, x_test_tensor, y_test, input_dim = prepare_data(
        batch_size=batch_size,
        val_size=val_size,
    )

    model = TelcoMLP(input_dim, hidden_dim, dropout_rate)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stopping_mode = "min" if monitor == "val_loss" else "max"
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode=stopping_mode,
    )

    with mlflow.start_run(run_name="pytorch-mlp"):
        mlflow.log_params(
            {
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "dropout_rate": dropout_rate,
                "val_size": val_size,
                "early_stopping_patience": patience,
                "early_stopping_min_delta": min_delta,
                "early_stopping_monitor": monitor,
                "random_seed": RANDOM_SEED,
                "device": str(device),
            }
        )

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            early_stopping=early_stopping,
            monitor=monitor,
            device=device,
            mlflow_logger=mlflow,
        )

        metrics = evaluate_torch_binary_classifier(model, x_test_tensor, y_test, device)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(
            {
                "epochs_trained": len(history.to_dict()["epoch"]),
                "best_epoch": early_stopping.best_epoch or 0,
            }
        )
        mlflow.log_param("early_stopped", early_stopping.early_stopped)

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = history.save_json(Path(temp_dir) / "history.json")
            mlflow.log_artifact(str(history_path), artifact_path="training")

        input_example = x_test_tensor[:5].numpy()

        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pt2",
            input_example=input_example,
        )

    torch.save(model.state_dict(), "models/mlp_model.pth")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a MLP do projeto de Churn com PyTorch."
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Dimensão da camada oculta.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (taxa de aprendizado).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Taxa de dropout para evitar overfitting.",
    )
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=EARLY_STOPPING_MIN_DELTA)
    parser.add_argument(
        "--monitor",
        choices=["val_loss", "val_roc_auc"],
        default=EARLY_STOPPING_MONITOR,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    logger.info("training_started model=pytorch-mlp")

    metrics = run_training(
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        val_size=args.val_size,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
    )

    for m, v in metrics.items():
        logger.info("test_metric=%s value=%.4f", m, v)


if __name__ == "__main__":
    main()
