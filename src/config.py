"""Cấu hình dùng chung cho dự án churn prediction.

Gom đường dẫn, tên cột và các hằng số quan trọng vào một chỗ để tránh
hard-code rải rác trong code.
"""

from __future__ import annotations

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
