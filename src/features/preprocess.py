"""Tiền xử lý dữ liệu churn: làm sạch và dựng pipeline biến đổi đặc trưng.

Bao gồm:

- ``clean_data``: làm sạch dữ liệu thô (chuỗi khoảng trắng, ``TotalCharges``
  sang số, chuẩn hóa chuỗi, mã hóa target, bỏ cột ID).
- ``build_preprocessor``: dựng ``ColumnTransformer`` của scikit-learn để
  impute + scale cột số và one-hot cột phân loại.

Lưu ý: pipeline trong ``build_preprocessor`` cần được ``fit`` **chỉ trên tập
train** để tránh rò rỉ dữ liệu (data leakage), rồi mới ``transform`` tập test.

Cách dùng::

    python -m src.features.preprocess           # làm sạch và lưu data/processed/churn_clean.csv
    python -m src.features.preprocess --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CLEAN_DATA_PATH,
    ID_COL,
    RAW_DATA_PATH,
    TARGET_COL,
    TARGET_MAP,
)


# ---------------------------------------------------------------------------
# Đọc & làm sạch
# ---------------------------------------------------------------------------


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc CSV thô (giữ nguyên kiểu gốc, không tự ép kiểu)."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    return pd.read_csv(path)


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa khoảng trắng thừa ở mọi cột chuỗi và chuyển chuỗi rỗng thành NaN."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        stripped = df[col].astype("string").str.strip()
        df[col] = stripped.mask(stripped.eq(""), pd.NA)
    return df


def clean_data(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """Làm sạch dữ liệu thô.

    - Xóa khoảng trắng thừa, chuỗi rỗng -> NaN.
    - Ép ``TotalCharges`` sang số (11 giá trị trống của khách mới thành NaN).
    - Mã hóa target ``Churn`` từ ``{"No":0, "Yes":1}``.
    - Bỏ cột ID (nếu ``drop_id=True``).
    """
    df = _strip_string_columns(df)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if TARGET_COL in df.columns:
        target_raw = df[TARGET_COL].astype("string").str.strip().str.title()
        unknown = set(target_raw.dropna().unique()) - set(TARGET_MAP)
        if unknown:
            raise ValueError(f"Giá trị target lạ: {sorted(unknown)}")
        df[TARGET_COL] = target_raw.map(TARGET_MAP).astype("Int64")

    if drop_id and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline tiền xử lý
# ---------------------------------------------------------------------------


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Tách danh sách cột số và cột phân loại."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Dựng ColumnTransformer biến đổi đặc trưng.

    - Cột số: impute bằng median + chuẩn hóa (StandardScaler).
    - Cột phân loại: impute bằng giá trị phổ biến nhất + one-hot
      (``handle_unknown="ignore"`` để an toàn khi gặp giá trị mới lúc inference).

    Cần ``fit`` trên tập train trước khi dùng.
    """
    numeric_cols, categorical_cols = split_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu churn.")
    parser.add_argument("--path", type=Path, default=RAW_DATA_PATH, help="File CSV thô.")
    parser.add_argument("--output", type=Path, default=CLEAN_DATA_PATH, help="File CSV đã làm sạch.")
    parser.add_argument("--keep-id", action="store_true", help="Giữ lại cột customerID.")
    parser.add_argument("--no-save", action="store_true", help="Không ghi file, chỉ in tóm tắt.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    raw = load_raw_data(args.path)
    clean = clean_data(raw, drop_id=not args.keep_id)

    numeric_cols, categorical_cols = split_feature_types(clean.drop(columns=[TARGET_COL]))
    print(f"Đã làm sạch: {clean.shape[0]} dòng x {clean.shape[1]} cột")
    print(f"Cột số ({len(numeric_cols)}): {', '.join(numeric_cols)}")
    print(f"Cột phân loại ({len(categorical_cols)}): {', '.join(categorical_cols)}")
    print("NaN còn lại (nếu có):")
    na = clean.isna().sum()
    na = na[na > 0]
    print("  " + (na.to_string().replace("\n", "\n  ") if not na.empty else "(không có)"))

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(args.output, index=False)
        print(f"\nĐã lưu dữ liệu sạch: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
