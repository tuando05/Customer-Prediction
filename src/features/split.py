"""Phân chia dữ liệu train/val/test theo phương pháp stratified.

Chia tập dữ liệu đã làm sạch thành train/validation/test, giữ nguyên tỷ lệ lớp
churn ở cả ba tập (``stratify``). Tập validation dùng để chọn threshold ở
Giai đoạn 2; tập test chỉ dùng để đánh giá cuối cùng.

Cách dùng::

    python -m src.features.split                # chia từ churn_clean.csv (tạo nếu chưa có)
    python -m src.features.split --test-size 0.2 --val-size 0.2 --random-state 42
    python -m src.features.split --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
    X_TEST_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.features.preprocess import clean_data, load_raw_data


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu đã làm sạch; nếu chưa có thì làm sạch từ dữ liệu thô."""
    if path.exists():
        return pd.read_csv(path)
    print(f"Chưa có {path}, sẽ làm sạch từ {RAW_DATA_PATH}...")
    return clean_data(load_raw_data(RAW_DATA_PATH))


class DataSplits(NamedTuple):
    """Bộ ba tập train/val/test (giữ nguyên tỷ lệ lớp ở mỗi tập)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_dataset(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Chia ``df`` thành train/val/test (stratified theo target).

    Thực hiện hai bước ``train_test_split``:

    1. Tách train khỏi phần còn lại (``val_size + test_size``).
    2. Chia phần còn lại thành val/test theo tỷ lệ tương ứng.
    """
    if target not in df.columns:
        raise KeyError(f"Không tìm thấy cột target '{target}' trong dữ liệu.")
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size và test_size phải > 0 và tổng < 1.")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Bước 1: tách train.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state, stratify=y
    )

    # Bước 2: chia phần còn lại thành val/test.
    test_ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=test_ratio, random_state=random_state, stratify=y_rest
    )

    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)


def save_splits(splits: DataSplits) -> None:
    """Ghi 6 file train/val/test vào ``data/processed``."""
    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.X_train.to_csv(X_TRAIN_PATH, index=False)
    splits.X_val.to_csv(X_VAL_PATH, index=False)
    splits.X_test.to_csv(X_TEST_PATH, index=False)
    splits.y_train.to_frame().to_csv(Y_TRAIN_PATH, index=False)
    splits.y_val.to_frame().to_csv(Y_VAL_PATH, index=False)
    splits.y_test.to_frame().to_csv(Y_TEST_PATH, index=False)


def _describe_split(name: str, y: pd.Series, total: int) -> str:
    counts = y.value_counts().sort_index()
    parts = [f"churn={int(v)}: {c} ({c / total:.1%})" for v, c in counts.items()]
    return f"{name}: {total} mẫu | " + " | ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chia train/test cho dữ liệu churn.")
    parser.add_argument("--input", type=Path, default=CLEAN_DATA_PATH, help="File dữ liệu đã làm sạch.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Tỷ lệ tập test.")
    parser.add_argument("--val-size", type=float, default=VAL_SIZE, help="Tỷ lệ tập validation.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Seed.")
    parser.add_argument("--no-save", action="store_true", help="Không ghi file, chỉ in tóm tắt.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    df = load_clean_data(args.input)
    splits = split_dataset(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    print(f"Tổng: {len(df)} mẫu")
    for name, y in (
        ("Train", splits.y_train),
        ("Val  ", splits.y_val),
        ("Test ", splits.y_test),
    ):
        print(_describe_split(name, y, len(y)))
    print(f"Số đặc trưng: {splits.X_train.shape[1]}")

    if not args.no_save:
        save_splits(splits)
        print("\nĐã lưu:")
        for p in (
            X_TRAIN_PATH,
            X_VAL_PATH,
            X_TEST_PATH,
            Y_TRAIN_PATH,
            Y_VAL_PATH,
            Y_TEST_PATH,
        ):
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
