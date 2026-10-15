"""Huấn luyện mô hình baseline (Logistic Regression) cho bài toán churn.

Giai đoạn MVP: một pipeline đơn giản nhưng hoàn chỉnh —

    dữ liệu thô -> làm sạch -> chia tập -> tiền xử lý -> LogisticRegression

Pipeline (gồm cả bước tiền xử lý) được lưu vào ``artifacts/churn_model.joblib``
để tái sử dụng khi serving. Chỉ **fit** trên tập train, tránh rò rỉ dữ liệu.

Cách dùng::

    python -m src.models.train
    python -m src.models.train --class-weight balanced
    python -m src.models.train --include-test      # xem thêm metric trên test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    TRAIN_METRICS_PATH,
    X_TEST_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.features.build_features import FeatureEngineer, add_engineered_features
from src.features.preprocess import build_preprocessor
from src.models.evaluate import evaluate_model, format_metrics


def load_xy(x_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Đọc cặp đặc trưng/nhãn đã chia sẵn (X_*.csv, y_*.csv)."""
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    return X, y


def build_pipeline(
    X_train: pd.DataFrame,
    class_weight: str | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Dựng pipeline: feature engineering + tiền xử lý + LogisticRegression.

    ``ColumnTransformer`` được dựng dựa trên các cột **sau** khi tạo đặc trưng,
    nên ``FeatureEngineer`` phải chạy trước nó.
    """
    engineered = add_engineered_features(X_train)
    preprocessor = build_preprocessor(engineered)
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit pipeline và trả về chính nó."""
    model.fit(X_train, y_train)
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện baseline churn.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' giúp xử lý mất cân bằng lớp.",
    )
    parser.add_argument("--C", type=float, default=1.0, help="Nghịch đảo độ mạnh regularization.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng quyết định.")
    parser.add_argument("--include-test", action="store_true", help="Đánh giá thêm trên tập test.")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="Nơi lưu model.")
    parser.add_argument("--metrics-out", type=Path, default=TRAIN_METRICS_PATH, help="Nơi lưu metrics.")
    parser.add_argument("--no-save", action="store_true", help="Không ghi model/metrics.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    class_weight = None if args.class_weight == "none" else "balanced"

    X_train, y_train = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

    model = build_pipeline(X_train, class_weight=class_weight, C=args.C)
    train(model, X_train, y_train)

    print(f"Đã huấn luyện LogisticRegression (class_weight={class_weight}, C={args.C})")
    print(f"  Train: {len(X_train)} mẫu | Val: {len(X_val)} mẫu")

    metrics: dict[str, dict[str, float]] = {
        "train": evaluate_model(model, X_train, y_train, threshold=args.threshold),
        "val": evaluate_model(model, X_val, y_val, threshold=args.threshold),
    }
    if args.include_test:
        X_test, y_test = load_xy(X_TEST_PATH, Y_TEST_PATH)
        metrics["test"] = evaluate_model(model, X_test, y_test, threshold=args.threshold)

    for name, m in metrics.items():
        print(f"  [{name:5}] {format_metrics(m)}")

    if not args.no_save:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "LogisticRegression",
            "params": {"class_weight": class_weight, "C": args.C},
            "n_features": int(X_train.shape[1]),
            "metrics": metrics,
        }
        args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nĐã lưu model: {args.model_out}")
        print(f"Đã lưu metrics: {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
