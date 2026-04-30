from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RANDOM_SEED = 42
TEST_SIZE = 0.2

MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "telco-churn-baselines"

DATASET_NAME = "telco-customer-churn"
TARGET_COLUMN = "target"

RAW_DATA_FILES = {
    "demographics": "Telco_customer_churn_demographics.xlsx",
    "location": "Telco_customer_churn_location.xlsx",
    "services": "Telco_customer_churn_services.xlsx",
    "population": "Telco_customer_churn_population.xlsx",
    "status": "Telco_customer_churn_status.xlsx",
}

ID_COLUMNS = (
    "CustomerID",
    "ID",
)

LEAKAGE_COLUMNS = (
    "ChurnLabel",
    "ChurnValue",
    "CustomerStatus",
    "ChurnScore",
    "ChurnScoreCategory",
    "ChurnCategory",
    "ChurnReason",
)
