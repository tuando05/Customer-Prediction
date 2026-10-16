"""Cấu hình dùng chung cho dự án churn prediction.

Gom đường dẫn, tên cột và các hằng số quan trọng vào một chỗ để tránh
hard-code rải rác trong code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "churn_clean.csv"
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_VAL_PATH = PROCESSED_DATA_DIR / "X_val.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_VAL_PATH = PROCESSED_DATA_DIR / "y_val.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

# Kết quả huấn luyện
MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
TRAIN_METRICS_PATH = REPORTS_DIR / "train_metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
IMBALANCE_COMPARISON_PATH = REPORTS_DIR / "imbalance_comparison.csv"
THRESHOLD_CURVE_PATH = REPORTS_DIR / "threshold_curve.csv"
TUNING_RESULTS_PATH = REPORTS_DIR / "tuning_results.json"
SHAP_IMPORTANCE_PATH = REPORTS_DIR / "shap_importance.csv"
SHAP_BAR_PATH = REPORTS_DIR / "shap_bar.png"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary.png"

# Giai đoạn 4
FEATURE_SELECTION_PATH = REPORTS_DIR / "feature_selection.csv"
RESAMPLE_COMPARISON_PATH = REPORTS_DIR / "resample_comparison.csv"
OPTUNA_RESULTS_PATH = REPORTS_DIR / "optuna_results.json"
CALIBRATION_PATH = REPORTS_DIR / "calibration.csv"

# ---------------------------------------------------------------------------
# Theo dõi thí nghiệm (MLflow)
# ---------------------------------------------------------------------------

MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
# Backend SQLite cục bộ (MLflow 3.x khuyến nghị thay cho file store; vẫn local,
# không cần server). Có thể override bằng biến môi trường MLFLOW_TRACKING_URI
# để trỏ tới backend/UI khác.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)
MLFLOW_EXPERIMENT = "churn-telco"

# ---------------------------------------------------------------------------
# Cột & nhãn
# ---------------------------------------------------------------------------

TARGET_COL = "Churn"
ID_COL = "customerID"
TENURE_COL = "tenure"

# Cột target dạng chữ -> nhãn số (1 = churn).
TARGET_MAP = {"No": 0, "Yes": 1}
POSITIVE_LABEL = 1

# ---------------------------------------------------------------------------
# Tham số huấn luyện
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
