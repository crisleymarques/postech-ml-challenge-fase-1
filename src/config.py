from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "val_loss"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"
MLFLOW_EXPERIMENT_NAME = "telco-churn-baselines"

DATASET_NAME = "telco-customer-churn"
TARGET_COLUMN = "target"
PROCESSED_DATASET_FILE = "telco_churn_model_ready.csv"
PROCESSED_MANIFEST_FILE = "telco_churn_model_ready_manifest.json"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / PROCESSED_DATASET_FILE
PROCESSED_MANIFEST_PATH = PROCESSED_DATA_DIR / PROCESSED_MANIFEST_FILE

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
