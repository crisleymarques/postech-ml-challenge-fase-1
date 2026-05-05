import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when the monitored metric stops improving."""

    def __init__(
        self,
        patience: int,
        min_delta: float,
        mode: Literal["min", "max"] = "min",
        restore_best_weights: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.wait_count = 0
        self.early_stopped = False
        self._best_state_dict: dict[str, torch.Tensor] | None = None

    def step(self, metric_value: float, model: nn.Module, epoch: int) -> bool:
        if not math.isfinite(metric_value):
            self.wait_count += 1
            self.early_stopped = self.wait_count >= self.patience
            return self.early_stopped

        if self._is_improvement(metric_value):
            self.best_score = metric_value
            self.best_epoch = epoch
            self.wait_count = 0
            if self.restore_best_weights:
                self._best_state_dict = copy.deepcopy(model.state_dict())
            return False

        self.wait_count += 1
        self.early_stopped = self.wait_count >= self.patience
        return self.early_stopped

    def restore(self, model: nn.Module) -> None:
        if self.restore_best_weights and self._best_state_dict is not None:
            model.load_state_dict(self._best_state_dict)

    def _is_improvement(self, metric_value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return metric_value < self.best_score - self.min_delta
        return metric_value > self.best_score + self.min_delta


class TrainingHistory:
    """Stores per-epoch metrics for later learning-curve analysis."""

    def __init__(self) -> None:
        self.records: dict[str, list[float | int]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_roc_auc": [],
        }

    def append(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, float],
    ) -> None:
        self.records["epoch"].append(epoch)
        self.records["train_loss"].append(train_loss)
        for metric_name in ("val_loss", "val_accuracy", "val_f1", "val_roc_auc"):
            self.records[metric_name].append(val_metrics[metric_name])

    def to_dict(self) -> dict[str, list[float | int]]:
        return self.records

    def save_json(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_json_safe(self.records), indent=2),
            encoding="utf-8",
        )
        return output_path


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        batch_size = batch_features.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / sample_count


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)

            batch_size = batch_features.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            logits_batches.append(logits.detach().cpu())
            target_batches.append(batch_targets.detach().cpu())

    logits = torch.cat(logits_batches).reshape(-1)
    targets = torch.cat(target_batches).reshape(-1).numpy()
    probabilities = torch.sigmoid(logits).numpy()
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "val_loss": running_loss / sample_count,
        "val_accuracy": accuracy_score(targets, predictions),
        "val_f1": f1_score(targets, predictions, zero_division=0),
        "val_roc_auc": _safe_roc_auc(targets, probabilities),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    *,
    epochs: int,
    early_stopping: EarlyStopping,
    monitor: str,
    device: torch.device,
    mlflow_logger: Any | None = None,
) -> TrainingHistory:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    history = TrainingHistory()
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        history.append(epoch, train_loss, val_metrics)

        epoch_metrics = {"train_loss": train_loss, **val_metrics}
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_accuracy=%.4f "
            "val_f1=%.4f val_roc_auc=%.4f",
            epoch,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["val_accuracy"],
            val_metrics["val_f1"],
            val_metrics["val_roc_auc"],
        )

        if mlflow_logger is not None:
            mlflow_logger.log_metrics(_finite_metrics(epoch_metrics), step=epoch)

        if early_stopping.step(val_metrics[monitor], model, epoch):
            logger.info(
                "early_stopping=true epoch=%d monitor=%s best_epoch=%s best_score=%s",
                epoch,
                monitor,
                early_stopping.best_epoch,
                early_stopping.best_score,
            )
            break

    early_stopping.restore(model)
    return history


def _safe_roc_auc(targets: Any, probabilities: Any) -> float:
    try:
        return roc_auc_score(targets, probabilities)
    except ValueError:
        return float("nan")


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
