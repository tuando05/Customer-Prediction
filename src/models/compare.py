"""So sánh nhiều mô hình bằng cross-validation (Giai đoạn 2 / M3).

Mọi mô hình được đánh giá trên **cùng một cách xử lý dữ liệu** (feature
engineering + tiền xử lý nằm trong pipeline) và **cùng folds** StratifiedKFold,
với cùng bộ metric — để so sánh công bằng. Metric chính để xếp hạng là PR-AUC.

Cách dùng::

    python -m src.models.compare
    python -m src.models.compare --models logreg rf xgb --folds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    MODEL_COMPARISON_PATH,
    RANDOM_STATE,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.models.evaluate import evaluate_model
from src.models.registry import MODEL_NAMES, build_estimator, build_model_pipeline
from src.models.train import load_xy

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}


def cross_validate_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, float], float]:
    """Chạy cross-validation cho một mô hình, trả về (metrics trung bình, thời gian)."""
    pipeline = build_model_pipeline(X, build_estimator(name))
    start = time.perf_counter()
    result = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise"
    )
    elapsed = time.perf_counter() - start

    row: dict[str, float] = {}
    for metric in SCORING:
        values = result[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    row["fit_seconds"] = round(elapsed, 2)
    return row, elapsed


def holdout_val_scores(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Fit trên train, đánh giá trên validation (threshold 0.5)."""
    pipeline = build_model_pipeline(X, build_estimator(name))
    pipeline.fit(X, y)
    metrics = evaluate_model(pipeline, X_val, y_val)
    return {
        "val_pr_auc": round(metrics["pr_auc"], 4),
        "val_recall": round(metrics["recall"], 4),
        "val_f1": round(metrics["f1"], 4),
    }


def compare_models(
    models: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    folds: int = 5,
) -> pd.DataFrame:
    """So sánh các mô hình, trả về DataFrame xếp hạng theo PR-AUC."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows: dict[str, dict[str, float]] = {}
    for name in models:
        cv_row, _ = cross_validate_model(name, X, y, cv)
        cv_row.update(holdout_val_scores(name, X, y, X_val, y_val))
        rows[name] = cv_row
    df = pd.DataFrame(rows).T
    return df.sort_values("pr_auc_mean", ascending=False)


def _format_table(df: pd.DataFrame) -> str:
    lines = []
    header = (
        f"{'model':8} {'pr_auc':>16} {'roc_auc':>8} {'recall':>8} "
        f"{'f1':>8} {'val_pr_auc':>11} {'val_recall':>11} {'sec':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, r in df.iterrows():
        pr = f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.3f}"
        lines.append(
            f"{name:8} {pr:>16} {r['roc_auc_mean']:>8.4f} {r['recall_mean']:>8.4f} "
            f"{r['f1_mean']:>8.4f} {r['val_pr_auc']:>11.4f} {r['val_recall']:>11.4f} "
            f"{r['fit_seconds']:>7.1f}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh mô hình churn bằng CV.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Danh sách mô hình cần so sánh.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Số folds cho CV.")
    parser.add_argument(
        "--output", type=Path, default=MODEL_COMPARISON_PATH, help="File CSV kết quả."
    )
    parser.add_argument("--no-save", action="store_true", help="Không ghi file CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    X, y = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

    print(f"So sánh {len(args.models)} mô hình | {args.folds}-fold CV | metric chính: PR-AUC")
    print(f"  Train: {len(X)} mẫu | Val: {len(X_val)} mẫu\n")

    df = compare_models(args.models, X, y, X_val, y_val, folds=args.folds)
    print(_format_table(df))
    print(f"\n=> Mô hình tốt nhất theo PR-AUC: {df.index[0]}")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.round(4).to_csv(args.output)
        print(f"Đã lưu: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
