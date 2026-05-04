import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
