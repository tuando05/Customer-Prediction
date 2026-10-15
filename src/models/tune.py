"""Tinh chỉnh siêu tham số (Giai đoạn 2 / M4).

Dùng ``RandomizedSearchCV`` (scikit-learn) với ``StratifiedKFold`` và
``scoring='average_precision'`` (PR-AUC) — đúng metric chính đã chốt.

Mọi không gian tham số nằm trên pipeline hoàn chỉnh (feature engineering +
tiền xử lý + classifier), nên không rò rỉ dữ liệu giữa các fold.

Cách dùng::

    python -m src.models.tune                          # tune xgb + rf
    python -m src.models.tune --models xgb --n-iter 30 --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    RANDOM_STATE,
    TUNING_RESULTS_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.models.evaluate import evaluate_model
from src.models.registry import MODEL_NAMES, build_estimator, build_model_pipeline
from src.models.train import load_xy

SCORING = "average_precision"  # PR-AUC

# Không gian tìm kiếm (tên tham số có tiền tố ``classifier__``).
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "xgb": {
        "classifier__n_estimators": [200, 300, 400, 600],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__gamma": [0.0, 0.1, 0.3],
        "classifier__scale_pos_weight": [1, 2.77],
    },
    "rf": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced"],
    },
    "logreg": {
        "classifier__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        "classifier__class_weight": [None, "balanced"],
    },
}

TUNED_MODELS = ["xgb", "rf"]


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int,
    folds: int,
) -> dict[str, Any]:
    """Tinh chỉnh một mô hình, trả về dict kết quả (gồm điểm CV và val)."""
    pipeline = build_model_pipeline(X, build_estimator(name))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start

    best = search.best_estimator_
    val = evaluate_model(best, X_val, y_val)
    return {
        "model": name,
        "cv_pr_auc": round(float(search.best_score_), 4),
        "val_pr_auc": round(val["pr_auc"], 4),
        "val_recall": round(val["recall"], 4),
        "val_f1": round(val["f1"], 4),
        "best_params": {
            k: (v if not isinstance(v, np.generic) else v.item())
            for k, v in search.best_params_.items()
        },
        "seconds": round(elapsed, 1),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tinh chỉnh siêu tham số.")
    parser.add_argument("--models", nargs="*", default=TUNED_MODELS, choices=MODEL_NAMES)
    parser.add_argument("--n-iter", type=int, default=15, help="Số tổ hợp thử cho mỗi mô hình.")
    parser.add_argument("--folds", type=int, default=5, help="Số folds CV.")
    parser.add_argument("--output", type=Path, default=TUNING_RESULTS_PATH)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    X, y = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

    print(
        f"Tuning {args.models} | {args.folds}-fold CV | scoring={SCORING} | "
        f"n_iter={args.n_iter} | Train={len(X)} Val={len(X_val)}\n"
    )

    results = []
    for name in args.models:
        res = tune_model(name, X, y, X_val, y_val, args.n_iter, args.folds)
        results.append(res)
        print(
            f"[{name:8}] cv_pr_auc={res['cv_pr_auc']:.4f} | "
            f"val_pr_auc={res['val_pr_auc']:.4f} val_recall={res['val_recall']:.4f} "
            f"val_f1={res['val_f1']:.4f} | {res['seconds']}s"
        )
        print(f"           best_params={res['best_params']}")

    best = max(results, key=lambda r: r["cv_pr_auc"])
    print(f"\n=> Tốt nhất theo CV PR-AUC: {best['model']} ({best['cv_pr_auc']:.4f})")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Đã lưu: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
