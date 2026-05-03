from contextlib import contextmanager

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.train_baselines as train_baselines
from src.config import TEST_SIZE
from src.train_baselines import evaluate_model_cv


def test_evaluate_model_cv_respects_caller_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CV must receive exactly the matrix the caller passes (train-only contract)."""

    recorded: dict[str, int] = {}

    def recording_cross_validate(
        estimator,
        x,
        y,
        cv=None,
        scoring=None,
        return_train_score=False,
    ):
        recorded["len_x"] = len(x)
        recorded["len_y"] = len(y)
        return {
            "test_accuracy": np.array([1.0]),
            "test_precision": np.array([1.0]),
            "test_recall": np.array([1.0]),
            "test_f1": np.array([1.0]),
            "test_roc_auc": np.array([1.0]),
            "test_pr_auc": np.array([1.0]),
        }

    monkeypatch.setattr(
        "src.train_baselines.cross_validate",
        recording_cross_validate,
    )

    n_train = 23
    x_train = pd.DataFrame({"feat": np.arange(n_train, dtype=float)})
    y_train = pd.Series(([0, 1] * (n_train // 2 + 1))[:n_train])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )

    evaluate_model_cv(pipeline, x_train, y_train, n_splits=3)

    assert recorded["len_x"] == n_train
    assert recorded["len_y"] == n_train


def test_run_training_cv_uses_train_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if evaluate_model_cv is called on the full dataset after hold-out."""
    n = 100
    df = pd.DataFrame(
        {
            "CustomerID": np.arange(n),
            "feat": np.random.default_rng(0).standard_normal(n),
            "target": np.tile([0, 1], n // 2),
        }
    )
    expected_train = int(n * (1.0 - TEST_SIZE))

    monkeypatch.setattr(
        train_baselines,
        "load_telco_dataset",
        lambda data_dir: df,
    )
    monkeypatch.setattr(
        train_baselines,
        "build_dataset_manifest",
        lambda data_dir: {
            "dataset_name": "test",
            "dataset_version": "0" * 64,
            "created_at_utc": "1970-01-01T00:00:00+00:00",
            "files": [{"name": "f.xlsx", "sha256": "a" * 64}],
        },
    )

    captured: dict[str, int] = {}

    def spy_evaluate_model_cv(
        pipeline: Pipeline,
        x: pd.DataFrame,
        y: pd.Series,
        random_seed: int = 42,
        n_splits: int = 5,
    ) -> dict[str, float]:
        captured["n_rows"] = len(x)
        return {
            "cv_accuracy": 0.5,
            "cv_precision": 0.5,
            "cv_recall": 0.5,
            "cv_f1": 0.5,
            "cv_roc_auc": 0.5,
            "cv_pr_auc": 0.5,
        }

    monkeypatch.setattr(
        train_baselines,
        "evaluate_model_cv",
        spy_evaluate_model_cv,
    )
    monkeypatch.setattr(
        train_baselines,
        "save_confusion_matrix",
        lambda *args, **kwargs: args[3],
    )
    monkeypatch.setattr(
        train_baselines,
        "save_feature_names",
        lambda *args, **kwargs: args[1],
    )
    monkeypatch.setattr(train_baselines, "log_dataset_version", lambda *a, **k: None)

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_tags", lambda *a, **k: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None)

    @contextmanager
    def _fake_start_run(*args, **kwargs):
        yield None

    monkeypatch.setattr(mlflow, "start_run", _fake_start_run)

    train_baselines.run_training(
        model_name="dummy_classifier",
        test_size=TEST_SIZE,
        random_seed=42,
    )

    assert captured["n_rows"] == expected_train
    assert captured["n_rows"] < n
