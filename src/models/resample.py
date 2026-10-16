"""So sánh xử lý mất cân bằng: class weights vs resampling (Giai đoạn 4 / P4-2).

Bốn chiến lược:

- ``none``: không xử lý gì (mốc so sánh).
- ``balanced``: ``class_weight='balanced'`` (LogReg) / ``scale_pos_weight`` (XGBoost) — đang dùng.
- ``smote``: SMOTE oversampling lớp thiểu số, **chỉ trong fold train**.
- ``smote_tomek``: SMOTE kết hợp làm sạch Tomek links.

Resampling dùng ``imblearn`` pipeline nên chỉ áp dụng khi ``fit`` — không rò rỉ sang
fold validation/test.

Cách dùng::

    python -m src.models.resample --model xgb
    python -m src.models.resample --model logreg --strategies none balanced smote
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import tracking
from src.config import (
    RANDOM_STATE,
    RESAMPLE_COMPARISON_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.models.evaluate import evaluate_model
from src.models.registry import (
    MODEL_NAMES,
    build_estimator,
    build_model_pipeline,
    resolve_estimator_params,
)
from src.models.train import load_xy

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall"}

STRATEGIES = ["none", "balanced", "smote", "smote_tomek"]


def _make_sampler(strategy: str):
    if strategy == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if strategy == "smote_tomek":
        return SMOTETomek(random_state=RANDOM_STATE)
    return None


def build_resample_pipeline(
    X: pd.DataFrame,
    model: str,
    strategy: str,
    params: dict,
    y: pd.Series | None = None,
    tuned: bool = True,
):
    """Dựng pipeline theo chiến lược mất cân bằng.

    - ``none`` / ``balanced``: pipeline sklearn thường (không sampler).
    - ``smote`` / ``smote_tomek``: pipeline imblearn có bước ``sampler``.
    """
    if strategy in ("none", "balanced"):
        class_weight = "balanced" if strategy == "balanced" else "none"
        est_params = resolve_estimator_params(
            model, class_weight=class_weight, y=y, tuned=tuned
        )
        est_params.update(params)
        estimator = build_estimator(model, **est_params)
        return build_model_pipeline(X, estimator)

    est_params = resolve_estimator_params(model, class_weight="none", y=y, tuned=tuned)
    est_params.update(params)
    estimator = build_estimator(model, **est_params)

    base = build_model_pipeline(X, estimator)
    sampler = _make_sampler(strategy)
    steps = list(base.steps[:2]) + [("sampler", sampler)] + list(base.steps[2:])
    return ImbPipeline(steps)


def evaluate_strategy(
    strategy: str,
    model: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    folds: int = 5,
    tuned: bool = True,
) -> dict[str, float]:
    """CV + holdout validation cho một chiến lược."""
    params = params or {}
    pipeline = build_resample_pipeline(X, model, strategy, params, y=y, tuned=tuned)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    start = time.perf_counter()
    result = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise"
    )
    elapsed = time.perf_counter() - start

    row: dict[str, float] = {
        "pr_auc_mean": float(np.mean(result["test_pr_auc"])),
        "pr_auc_std": float(np.std(result["test_pr_auc"])),
        "roc_auc_mean": float(np.mean(result["test_roc_auc"])),
        "recall_mean": float(np.mean(result["test_recall"])),
        "seconds": round(elapsed, 1),
    }

    pipeline.fit(X, y)
    val = evaluate_model(pipeline, X_val, y_val)
    row["val_pr_auc"] = round(val["pr_auc"], 4)
    row["val_recall"] = round(val["recall"], 4)
    row["val_f1"] = round(val["f1"], 4)
    return row


def compare_resampling(
    model: str,
    strategies: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    folds: int = 5,
    tuned: bool = True,
) -> pd.DataFrame:
    """So sánh các chiến lược, trả về DataFrame xếp hạng theo CV PR-AUC."""
    rows = {
        s: evaluate_strategy(s, model, X, y, X_val, y_val, params, folds, tuned)
        for s in strategies
    }
    df = pd.DataFrame(rows).T
    return df.sort_values("pr_auc_mean", ascending=False)


def _format_table(df: pd.DataFrame) -> str:
    header = (
        f"{'strategy':14} {'pr_auc(cv)':>14} {'roc_auc':>8} {'recall':>8} "
        f"{'val_pr_auc':>11} {'val_recall':>11}"
    )
    lines = [header, "-" * len(header)]
    for name, r in df.iterrows():
        pr = f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.3f}"
        lines.append(
            f"{name:14} {pr:>14} {r['roc_auc_mean']:>8.4f} {r['recall_mean']:>8.4f} "
            f"{r['val_pr_auc']:>11.4f} {r['val_recall']:>11.4f}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh xử lý mất cân bằng.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="xgb")
    parser.add_argument(
        "--strategies", nargs="*", default=STRATEGIES, choices=STRATEGIES
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng tham số đã tune (reports/tuning_results.json) nếu có.",
    )
    parser.add_argument("--output", type=Path, default=RESAMPLE_COMPARISON_PATH)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    X, y = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

    print(
        f"So sánh {len(args.strategies)} chiến lược | model={args.model} | {args.folds}-fold CV "
        f"| tuned={args.tuned}\n"
    )
    df = compare_resampling(
        args.model, args.strategies, X, y, X_val, y_val, folds=args.folds, tuned=args.tuned
    )
    print(_format_table(df))
    print(f"\n=> Tốt nhất theo CV PR-AUC: {df.index[0]}")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.round(4).to_csv(args.output)
        print(f"Đã lưu: {args.output}")

    if not args.no_mlflow:
        for name, r in df.iterrows():
            tracking.track_run(
                f"p4-resample-{name}",
                params={"model": args.model, "strategy": name, "folds": args.folds},
                metrics={
                    "cv.pr_auc_mean": r["pr_auc_mean"],
                    "cv.pr_auc_std": r["pr_auc_std"],
                    "val.pr_auc": r["val_pr_auc"],
                    "val.recall": r["val_recall"],
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
