from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
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


def load_raw_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    tables = {}
    for table_name, file_name in RAW_DATA_FILES.items():
        path = data_dir / file_name
        df = pd.read_excel(path, sheet_name=0)
        df = clean_column_names(df).drop(columns=["Count"], errors="ignore")
        tables[table_name] = df
    return tables


def load_telco_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
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


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    columns_to_drop = list(ID_COLUMNS) + list(LEAKAGE_COLUMNS) + [target_column]
    x = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target_column]
    return x, y


def get_source_file_paths(data_dir: Path = DATA_DIR) -> list[Path]:
    return [data_dir / file_name for file_name in RAW_DATA_FILES.values()]
