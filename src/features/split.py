"""Phân chia dữ liệu train/test theo phương pháp stratified.

Chia tập dữ liệu đã làm sạch thành train/test, giữ nguyên tỷ lệ lớp churn ở
cả hai tập (``stratify``), đảm bảo mô hình được đánh giá công bằng.

Cách dùng::

    python -m src.features.split                # chia từ churn_clean.csv (tạo nếu chưa có)
    python -m src.features.split --test-size 0.25 --random-state 42
    python -m src.features.split --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)
from src.features.preprocess import clean_data, load_raw_data


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu đã làm sạch; nếu chưa có thì làm sạch từ dữ liệu thô."""
    if path.exists():
        return pd.read_csv(path)
    print(f"Chưa có {path}, sẽ làm sạch từ {RAW_DATA_PATH}...")
    return clean_data(load_raw_data(RAW_DATA_PATH))


def split_dataset(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia ``df`` thành X_train, X_test, y_train, y_test (stratified theo target)."""
    if target not in df.columns:
        raise KeyError(f"Không tìm thấy cột target '{target}' trong dữ liệu.")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Ghi 4 file train/test vào ``data/processed``."""
    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame().to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame().to_csv(Y_TEST_PATH, index=False)


def _describe_split(name: str, y: pd.Series, total: int) -> str:
    counts = y.value_counts().sort_index()
    parts = [f"churn={int(v)}: {c} ({c / total:.1%})" for v, c in counts.items()]
    return f"{name}: {total} mẫu | " + " | ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chia train/test cho dữ liệu churn.")
    parser.add_argument("--input", type=Path, default=CLEAN_DATA_PATH, help="File dữ liệu đã làm sạch.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Tỷ lệ tập test.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Seed.")
    parser.add_argument("--no-save", action="store_true", help="Không ghi file, chỉ in tóm tắt.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    df = load_clean_data(args.input)
    X_train, X_test, y_train, y_test = split_dataset(
        df, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Tổng: {len(df)} mẫu | train={len(X_train)} | test={len(X_test)}")
    print(_describe_split("Train", y_train, len(X_train)))
    print(_describe_split("Test ", y_test, len(X_test)))
    print(f"Số đặc trưng: {X_train.shape[1]}")

    if not args.no_save:
        save_splits(X_train, X_test, y_train, y_test)
        print("\nĐã lưu:")
        for p in (X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH):
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
