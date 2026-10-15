"""Feature engineering cho bài toán churn (Giai đoạn 2).

Dựa trên insight từ ``notebooks/01_eda.ipynb``, tạo thêm các đặc trưng:

- ``tenure_bucket``: nhóm thời gian gắn bó (0–12 / 13–24 / 25–48 / 49–72).
- ``num_services``: số dịch vụ khách đang dùng.
- ``is_automatic_payment``: thanh toán tự động (1) hay thủ công (0).
- ``is_month_to_month``: hợp đồng theo tháng — tín hiệu rủi ro mạnh nhất.

Các phép biến đổi này **không cần học tham số** (stateless), nên không gây rò rỉ
dữ liệu; vẫn đóng gói dưới dạng transformer để nằm trong pipeline và dùng chung
cho cả huấn luyện lẫn inference.

Cách dùng::

    python -m src.features.build_features --input data/processed/churn_clean.csv --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import CLEAN_DATA_PATH, TENURE_COL

# ---------------------------------------------------------------------------
# Định nghĩa
# ---------------------------------------------------------------------------

# Các cột dịch vụ dạng Yes/No ("Yes" = đang dùng dịch vụ).
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

# Tên các đặc trưng mới (thứ tự ổn định).
NEW_FEATURE_NAMES = [
    "tenure_bucket",
    "num_services",
    "is_automatic_payment",
    "is_month_to_month",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trả về bản sao của ``df`` với các đặc trưng mới được thêm vào."""
    df = df.copy()

    # Số dịch vụ đang dùng.
    service_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
    if service_cols:
        df["num_services"] = sum(
            (df[c].astype("string").str.strip() == "Yes").astype(int)
            for c in service_cols
        )

    # Nhóm thời gian gắn bó.
    if TENURE_COL in df.columns:
        df["tenure_bucket"] = pd.cut(
            df[TENURE_COL], bins=TENURE_BINS, labels=TENURE_LABELS
        ).astype("string")

    # Thanh toán tự động.
    if "PaymentMethod" in df.columns:
        df["is_automatic_payment"] = (
            df["PaymentMethod"]
            .astype("string")
            .str.contains("automatic", case=False, na=False)
            .astype(int)
        )

    # Hợp đồng theo tháng.
    if "Contract" in df.columns:
        df["is_month_to_month"] = (
            df["Contract"].astype("string").str.strip() == "Month-to-month"
        ).astype(int)

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer bọc ``add_engineered_features`` để dùng trong sklearn Pipeline."""

    def fit(self, X, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X) -> pd.DataFrame:
        return add_engineered_features(pd.DataFrame(X))


# ---------------------------------------------------------------------------
# CLI (tiện kiểm tra nhanh)
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo đặc trưng mới cho churn.")
    parser.add_argument("--input", type=Path, default=CLEAN_DATA_PATH, help="CSV dữ liệu sạch.")
    parser.add_argument("--limit", type=int, default=5, help="Số dòng in ra.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    df = pd.read_csv(args.input)
    out = add_engineered_features(df)

    added = [c for c in NEW_FEATURE_NAMES if c in out.columns]
    print(f"Đã thêm {len(added)} đặc trưng: {', '.join(added)}")
    print(f"Tổng số cột: {df.shape[1]} -> {out.shape[1]}")
    print(f"\n{args.limit} dòng đầu (chỉ cột đặc trưng gốc + mới):")
    cols = ["tenure", "Contract", "PaymentMethod", "MonthlyCharges"] + added
    print(out[cols].head(args.limit).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
