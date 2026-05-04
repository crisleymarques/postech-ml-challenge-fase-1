import pandas as pd

from src.train_baselines import build_pipeline, evaluate_model


def test_pipeline_trains_and_returns_core_metrics() -> None:
    features = pd.DataFrame(
        {
            "Age": [25, 32, 47, 58, 61, 29],
            "MonthlyCharge": [70.0, 65.0, 40.0, 90.0, 95.0, 50.0],
            "Contract": [
                "Month-to-Month",
                "One Year",
                "Two Year",
                "Month-to-Month",
                "Month-to-Month",
                "Two Year",
            ],
        }
    )
    target = pd.Series([1, 0, 0, 1, 1, 0])

    pipeline = build_pipeline("logistic_regression", features, random_seed=42)
    pipeline.fit(features, target)
    metrics = evaluate_model(pipeline, features, target)

    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= metrics.keys()
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_dummy_classifier_pipeline_trains() -> None:
    features = pd.DataFrame(
        {
            "Age": [25, 32, 47, 58],
            "MonthlyCharge": [70.0, 65.0, 40.0, 90.0],
            "Contract": ["Month-to-Month", "One Year", "Two Year", "Month-to-Month"],
        }
    )
    target = pd.Series([1, 0, 0, 1])

    pipeline = build_pipeline("dummy_classifier", features, random_seed=42)
    pipeline.fit(features, target)
    metrics = evaluate_model(pipeline, features, target)

    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= metrics.keys()
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
