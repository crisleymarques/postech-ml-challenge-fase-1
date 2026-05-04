"""Headless plot artifacts (matplotlib imported lazily after env/backend setup)."""

import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline


def save_confusion_matrix(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "outputs" / "cache"
    matplotlib_cache_dir = repo_root / "outputs" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
