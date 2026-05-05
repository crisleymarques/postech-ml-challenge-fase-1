<<<<<<< HEAD
=======
import json
>>>>>>> f8290b60dbc7640f2f4cc0fb4d3618e51dad89f7
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_FILES, TARGET_COLUMN
<<<<<<< HEAD
from src.data import load_model_ready_dataset, load_telco_dataset, split_features_target
=======
from src.data import (
    file_sha256,
    load_model_ready_dataset,
    load_telco_dataset,
    split_features_target,
)
>>>>>>> f8290b60dbc7640f2f4cc0fb4d3618e51dad89f7


def test_loader_merges_tables_and_removes_leakage_from_features(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "Customer ID": ["C1", "C2"],
            "Gender": ["Female", "Male"],
            "Age": [30, 45],
        }
    ).to_excel(tmp_path / RAW_DATA_FILES["demographics"], index=False)
    pd.DataFrame(
        {
            "Customer ID": ["C1", "C2"],
            "Zip Code": [90001, 90002],
            "City": ["A", "B"],
        }
    ).to_excel(tmp_path / RAW_DATA_FILES["location"], index=False)
    pd.DataFrame(
        {
            "Customer ID": ["C1", "C2"],
            "Tenure in Months": [2, 24],
            "Monthly Charge": [80.0, 55.0],
        }
    ).to_excel(tmp_path / RAW_DATA_FILES["services"], index=False)
    pd.DataFrame(
        {
            "ID": [1, 2],
            "Zip Code": [90001, 90002],
            "Population": [1000, 2000],
        }
    ).to_excel(tmp_path / RAW_DATA_FILES["population"], index=False)
    pd.DataFrame(
        {
            "Customer ID": ["C1", "C2"],
            "Customer Status": ["Churned", "Stayed"],
            "Churn Label": ["Yes", "No"],
            "Churn Value": [1, 0],
            "Churn Score": [90, 20],
            "Churn Category": ["Competitor", None],
            "Churn Reason": ["Better offer", None],
        }
    ).to_excel(tmp_path / RAW_DATA_FILES["status"], index=False)

    df = load_telco_dataset(tmp_path)
    features, target = split_features_target(df)

    assert df.shape[0] == 2
    assert TARGET_COLUMN in df.columns
    assert target.tolist() == [1, 0]
    assert "ChurnScore" not in features.columns
    assert "CustomerStatus" not in features.columns
    assert "CustomerID" not in features.columns


def test_load_model_ready_dataset_reads_versioned_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "telco_churn_model_ready.csv"
    pd.DataFrame(
        {
            "MonthlyCharge": [80.0, 55.0],
            "Contract": ["Month-to-Month", "Two Year"],
            TARGET_COLUMN: [1, 0],
        }
    ).to_csv(dataset_path, index=False)

    df = load_model_ready_dataset(dataset_path)
    features, target = split_features_target(df)

    assert df[TARGET_COLUMN].tolist() == [1, 0]
    assert target.tolist() == [1, 0]
    assert TARGET_COLUMN not in features.columns
<<<<<<< HEAD
=======


def test_load_model_ready_dataset_validates_manifest_hash(tmp_path: Path) -> None:
    dataset_path = tmp_path / "telco_churn_model_ready.csv"
    manifest_path = tmp_path / "telco_churn_model_ready_manifest.json"

    pd.DataFrame(
        {
            "MonthlyCharge": [80.0, 55.0],
            "Contract": ["Month-to-Month", "Two Year"],
            TARGET_COLUMN: [1, 0],
        }
    ).to_csv(dataset_path, index=False)

    manifest_path.write_text(
        json.dumps({"output_sha256": file_sha256(dataset_path)}),
        encoding="utf-8",
    )
    load_model_ready_dataset(dataset_path=dataset_path, manifest_path=manifest_path)

    manifest_path.write_text(
        json.dumps({"output_sha256": "invalid"}),
        encoding="utf-8",
    )
    try:
        load_model_ready_dataset(dataset_path=dataset_path, manifest_path=manifest_path)
        raise AssertionError("Expected hash mismatch to raise ValueError")
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
>>>>>>> f8290b60dbc7640f2f4cc0fb4d3618e51dad89f7
