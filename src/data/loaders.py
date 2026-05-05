import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import (
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    PROCESSED_DATASET_PATH,
    PROCESSED_MANIFEST_PATH,
    RAW_DATA_DIR,
    RAW_DATA_FILES,
    TARGET_COLUMN,
)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Match the notebook convention: no spaces or underscores in column names."""
    renamed = {
        column: str(column).strip().replace(" ", "").replace("_", "")
        for column in df.columns
    }
    return df.rename(columns=renamed)


def load_raw_tables(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    tables = {}
    for table_name, file_name in RAW_DATA_FILES.items():
        path = data_dir / file_name
        df = pd.read_excel(path, sheet_name=0)
        df = clean_column_names(df).drop(columns=["Count"], errors="ignore")
        tables[table_name] = df
    return tables


def load_telco_dataset(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    tables = load_raw_tables(data_dir)

    df = (
        tables["demographics"]
        .merge(tables["location"], on="CustomerID")
        .merge(tables["services"], on="CustomerID")
        .merge(tables["population"], on="ZipCode")
        .merge(tables["status"], on="CustomerID")
        .drop(columns=["ID"], errors="ignore")
    )

    if "ChurnValue" not in df.columns:
        raise ValueError("Expected column ChurnValue to build target.")

    df[TARGET_COLUMN] = (df["ChurnValue"] > 0).astype(int)
    return df


def load_model_ready_dataset(
    dataset_path: Path = PROCESSED_DATASET_PATH,
    manifest_path: Path | None = None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Load the Git-versioned dataset produced by notebook 01."""
    if manifest_path is None and dataset_path == PROCESSED_DATASET_PATH:
        manifest_path = PROCESSED_MANIFEST_PATH

    if manifest_path is not None and manifest_path.exists():
        validate_processed_manifest(dataset_path, manifest_path)

    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Expected column {target_column} in processed dataset.")
    df[target_column] = df[target_column].astype(int)
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    columns_to_drop = list(ID_COLUMNS) + list(LEAKAGE_COLUMNS) + [target_column]
    x = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target_column]
    return x, y


def get_source_file_paths(data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    return [data_dir / file_name for file_name in RAW_DATA_FILES.values()]


def validate_processed_manifest(dataset_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get("output_sha256")
    if not expected_sha:
        raise ValueError(f"Missing output_sha256 in manifest: {manifest_path}")

    actual_sha = file_sha256(dataset_path)
    if actual_sha != expected_sha:
        raise ValueError(
            "Processed dataset hash mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
