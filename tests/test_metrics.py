import math

import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import evaluate_torch_binary_classifier


class ConstantModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)


def test_evaluate_torch_binary_classifier_handles_single_class_target() -> None:
    model = ConstantModel()
    x_test_tensor = torch.randn(5, 3)
    y_test = pd.Series([0, 0, 0, 0, 0])

    metrics = evaluate_torch_binary_classifier(
        model=model,
        x_test_tensor=x_test_tensor,
        y_test=y_test,
        device=torch.device("cpu"),
    )

    assert math.isnan(metrics["roc_auc"])
