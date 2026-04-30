import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_DIR,
    LEAKAGE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import load_telco_dataset, split_features_target
from src.dataset_version import build_dataset_manifest, write_dataset_manifest
from src.features import build_preprocessor


def build_model(model_name: str, random_seed: int) -> object:
    if model_name == "dummy_classifier":
        return DummyClassifier(strategy="most_frequent", random_state=random_seed)
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str, x: pd.DataFrame, random_seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(x)),
            ("model", build_model(model_name, random_seed)),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    if hasattr(pipeline, "predict_proba"):
        positive_scores = pipeline.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, positive_scores)

    return metrics


def save_confusion_matrix(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / "outputs" / "cache"
    matplotlib_cache_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "matplotlib"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    predictions = pipeline.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Stayed", "Churned"],
    )
    display.plot(values_format="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_feature_names(pipeline: Pipeline, output_path: Path) -> Path:
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    output_path.write_text(
        json.dumps(feature_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def log_dataset_version(manifest: dict, manifest_path: Path) -> None:
    source_files = ",".join(item["name"] for item in manifest["files"])
    mlflow.set_tags(
        {
            "dataset.name": manifest["dataset_name"],
            "dataset.version": manifest["dataset_version"],
            "dataset.hash": manifest["dataset_version"],
            "dataset.source_files": source_files,
        }
    )
    mlflow.log_artifact(str(manifest_path), artifact_path="dataset")


def run_training(
    model_name: str,
    data_dir: Path = DATA_DIR,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> dict[str, float]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_telco_dataset(data_dir)
    x, y = split_features_target(df)
    manifest = build_dataset_manifest(data_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    pipeline = build_pipeline(model_name, x_train, random_seed)

    with mlflow.start_run(run_name=f"baseline-{model_name}"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "random_seed": random_seed,
                "test_size": test_size,
                "target_column": TARGET_COLUMN,
                "n_rows": len(df),
                "n_features": x.shape[1],
                "leakage_columns_removed": ",".join(LEAKAGE_COLUMNS),
                "preprocessor": "median_numeric_most_frequent_categorical_onehot",
            }
        )
        mlflow.log_params(
            {
                f"model__{key}": value
                for key, value in pipeline.named_steps["model"].get_params().items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
        )

        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            manifest_path = write_dataset_manifest(
                manifest,
                artifact_dir / "dataset_manifest.json",
            )
            confusion_matrix_path = save_confusion_matrix(
                pipeline,
                x_test,
                y_test,
                artifact_dir / "confusion_matrix.png",
            )
            feature_names_path = save_feature_names(
                pipeline,
                artifact_dir / "feature_names.json",
            )

            log_dataset_version(manifest, manifest_path)
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="metrics")
            mlflow.log_artifact(str(feature_names_path), artifact_path="features")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=x_train.head(5),
            )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Telco churn baselines with MLflow."
    )
    parser.add_argument(
        "--model",
        choices=["dummy_classifier", "logistic_regression", "all"],
        default="all",
        help="Baseline model to train.",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = (
        ["dummy_classifier", "logistic_regression"]
        if args.model == "all"
        else [args.model]
    )
    for model_name in models:
        metrics = run_training(
            model_name=model_name,
            test_size=args.test_size,
            random_seed=args.seed,
        )
        formatted_metrics = ", ".join(
            f"{metric}={value:.4f}" for metric, value in sorted(metrics.items())
        )
        print(f"{model_name}: {formatted_metrics}")


if __name__ == "__main__":
    main()
