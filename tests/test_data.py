from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_FILES, TARGET_COLUMN
from src.data import load_telco_dataset, split_features_target


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
