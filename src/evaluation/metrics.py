from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_sklearn_classifier(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = pipeline.predict(x_test)
    metrics = classification_metrics(y_test, predictions)

    if hasattr(pipeline, "predict_proba"):
        positive_scores = pipeline.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, positive_scores)

    return metrics


def evaluate_torch_binary_classifier(
    model: nn.Module,
    x_test_tensor: torch.Tensor,
    y_test: pd.Series,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor.to(device))
        probabilities = torch.sigmoid(test_outputs).cpu().numpy().reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)

    return {
        **classification_metrics(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def classification_metrics(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
