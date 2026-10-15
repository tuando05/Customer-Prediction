"""Thí nghiệm xử lý mất cân bằng lớp (Giai đoạn 2 / M2).

So sánh cùng các mô hình dưới hai chiến lược:

- ``none``: không xử lý (baseline).
- ``balanced``: dùng ``class_weight='balanced'`` (và ``scale_pos_weight`` cho XGBoost).

Mọi cấu hình dùng chung feature engineering, tiền xử lý và folds CV để so sánh
công bằng. Metric chính: PR-AUC (xếp hạng) và Recall (mục tiêu kinh doanh).

Cách dùng::

    python -m src.models.imbalance
    python -m src.models.imbalance --models logreg xgb --folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    IMBALANCE_COMPARISON_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.models.compare import compare_models
from src.models.registry import MODEL_NAMES
from src.models.train import load_xy

STRATEGIES: dict[str, str | None] = {"none": None, "balanced": "balanced"}


def run_experiment(
    models: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    folds: int = 5,
) -> pd.DataFrame:
    """Chạy so sánh cho từng chiến lược, gộp thành một bảng."""
    frames = []
    for label, class_weight in STRATEGIES.items():
        df = compare_models(
            models, X, y, X_val, y_val, folds=folds, class_weight=class_weight
        )
        df.index.name = "model"
        df.insert(0, "strategy", label)
        frames.append(df)
    return pd.concat(frames)


def _format_table(df: pd.DataFrame) -> str:
    d = df.reset_index()
    header = (
        f"{'strategy':9} {'model':8} {'pr_auc':>16} {'recall':>8} "
        f"{'f1':>8} {'val_pr_auc':>11} {'val_recall':>11}"
    )
    lines = [header, "-" * len(header)]
    for _, r in d.iterrows():
        pr = f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.3f}"
        lines.append(
            f"{r['strategy']:9} {r['model']:8} {pr:>16} {r['recall_mean']:>8.4f} "
            f"{r['f1_mean']:>8.4f} {r['val_pr_auc']:>11.4f} {r['val_recall']:>11.4f}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thí nghiệm mất cân bằng lớp.")
    parser.add_argument(
        "--models", nargs="*", default=MODEL_NAMES, choices=MODEL_NAMES
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=IMBALANCE_COMPARISON_PATH, help="File CSV kết quả."
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    X, y = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

    ratio = (y == 0).sum() / (y == 1).sum()
    print(f"Thí nghiệm mất cân bằng | {args.folds}-fold CV | neg/pos = {ratio:.2f}")
    print(f"  Train: {len(X)} mẫu | Val: {len(X_val)} mẫu\n")

    df = run_experiment(args.models, X, y, X_val, y_val, folds=args.folds)
    print(_format_table(df))

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.round(4).to_csv(args.output)
        print(f"\nĐã lưu: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
