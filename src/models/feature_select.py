"""So sánh các bộ đặc trưng bằng cross-validation (Giai đoạn 4 / P4-1).

Từ SHAP (M5) và EDA, một số đặc trưng **dư thừa hoặc gần như bằng 0**:

- ``is_month_to_month``: trùng thông tin với ``Contract_Month-to-month``.
- các dummy ``*_No internet service``: giá trị gần 0.

Module này so sánh nhiều **bộ đặc trưng** trên cùng folds CV và cùng model để
chọn bộ tốt nhất một cách công bằng (metric chính: PR-AUC).

Cách dùng::

    python -m src.models.feature_select --model xgb
    python -m src.models.feature_select --model logreg --no-tuned
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import tracking
from src.config import (
    FEATURE_SELECTION_PATH,
    RANDOM_STATE,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.features.select import DUPLICATE_FEATURES, NO_INTERNET_DUMMY_SUFFIX
from src.models.evaluate import evaluate_model
from src.models.registry import (
    MODEL_NAMES,
    build_estimator,
    build_model_pipeline,
    resolve_estimator_params,
)
from src.models.train import load_xy

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall"}

# Các bộ đặc trưng đem so sánh: tên -> (drop, drop_suffix).
FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "full": {},
    "no_dup": {"drop": DUPLICATE_FEATURES},
    "no_internet_dummies": {"drop_suffix": NO_INTERNET_DUMMY_SUFFIX},
    "lean": {"drop": DUPLICATE_FEATURES, "drop_suffix": NO_INTERNET_DUMMY_SUFFIX},
}


def evaluate_feature_set(
    name: str,
    model: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    folds: int = 5,
) -> dict[str, float]:
    """CV + đánh giá holdout trên validation cho một bộ đặc trưng."""
    spec = FEATURE_SETS[name]
    estimator = build_estimator(model, **params)
    pipeline = build_model_pipeline(
        X, estimator, drop=spec.get("drop"), drop_suffix=spec.get("drop_suffix")
    )

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
    row["n_features"] = int(
        len(pipeline.named_steps["preprocessor"].get_feature_names_out())
        if "select" not in pipeline.named_steps
        else len(pipeline.named_steps["select"].feature_names_out_)
    )
    return row


def compare_feature_sets(
    model: str,
    sets: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    folds: int = 5,
) -> pd.DataFrame:
    """So sánh nhiều bộ đặc trưng, trả về DataFrame xếp hạng theo CV PR-AUC."""
    params = params or {}
    rows = {
        name: evaluate_feature_set(name, model, params, X, y, X_val, y_val, folds)
        for name in sets
    }
    df = pd.DataFrame(rows).T
    return df.sort_values("pr_auc_mean", ascending=False)


def _format_table(df: pd.DataFrame) -> str:
    header = (
        f"{'feature_set':22} {'pr_auc(cv)':>14} {'roc_auc':>8} {'recall':>8} "
        f"{'val_pr_auc':>11} {'n_feat':>7}"
    )
    lines = [header, "-" * len(header)]
    for name, r in df.iterrows():
        pr = f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.3f}"
        lines.append(
            f"{name:22} {pr:>14} {r['roc_auc_mean']:>8.4f} {r['recall_mean']:>8.4f} "
            f"{r['val_pr_auc']:>11.4f} {int(r['n_features']):>7}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh các bộ đặc trưng.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="xgb")
    parser.add_argument(
        "--sets", nargs="*", default=list(FEATURE_SETS), choices=list(FEATURE_SETS)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng tham số đã tune (reports/tuning_results.json) nếu có.",
    )
    parser.add_argument(
        "--class-weight", choices=["none", "balanced"], default="balanced"
    )
    parser.add_argument("--output", type=Path, default=FEATURE_SELECTION_PATH)
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

    class_weight = None if args.class_weight == "none" else "balanced"
    params = resolve_estimator_params(
        args.model, class_weight=class_weight, y=y, tuned=args.tuned
    )

    print(
        f"So sánh {len(args.sets)} bộ đặc trưng | model={args.model} | {args.folds}-fold CV "
        f"| class_weight={class_weight} | tuned={args.tuned}\n"
    )
    df = compare_feature_sets(
        args.model, args.sets, X, y, X_val, y_val, params=params, folds=args.folds
    )
    print(_format_table(df))
    print(f"\n=> Bộ tốt nhất theo CV PR-AUC: {df.index[0]}")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.round(4).to_csv(args.output)
        print(f"Đã lưu: {args.output}")

    if not args.no_mlflow:
        for name, r in df.iterrows():
            tracking.track_run(
                f"p4-features-{name}",
                params={
                    "model": args.model,
                    "feature_set": name,
                    "folds": args.folds,
                    "class_weight": class_weight,
                    "n_features": int(r["n_features"]),
                },
                metrics={
                    "cv.pr_auc_mean": r["pr_auc_mean"],
                    "cv.pr_auc_std": r["pr_auc_std"],
                    "cv.roc_auc_mean": r["roc_auc_mean"],
                    "val.pr_auc": r["val_pr_auc"],
                    "val.recall": r["val_recall"],
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
