"""Tinh chỉnh siêu tham số bằng Optuna (Giai đoạn 4 / P4-3).

Dùng **TPE sampler** với nhiều trial hơn ``RandomizedSearchCV`` (M4), cùng
``StratifiedKFold`` và ``scoring='average_precision'`` (PR-AUC) để so sánh công bằng.

Kết quả:

- Lưu toàn bộ best params vào ``reports/optuna_results.json``.
- Nếu **tốt hơn** kết quả tuning cũ của cùng model, cập nhật
  ``reports/tuning_results.json`` (nguồn mà ``--tuned`` dùng) để model cuối hưởng lợi.

Cách dùng::

    python -m src.models.tune_optuna --models xgb --n-trials 40
    python -m src.models.tune_optuna --models xgb rf --timeout 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import tracking
from src.config import (
    OPTUNA_RESULTS_PATH,
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

SCORING = "average_precision"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Không gian tìm kiếm (khóa có tiền tố ``classifier__``)
# ---------------------------------------------------------------------------


def _space_xgb(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "classifier__n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "classifier__max_depth": trial.suggest_int("max_depth", 2, 8),
        "classifier__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classifier__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "classifier__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "classifier__gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "classifier__reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "classifier__reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "classifier__scale_pos_weight": trial.suggest_categorical(
            "scale_pos_weight", [1.0, 2.77]
        ),
    }


def _space_rf(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "classifier__n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
        "classifier__max_depth": trial.suggest_categorical(
            "max_depth", [None, 5, 10, 20, 30]
        ),
        "classifier__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "classifier__min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "classifier__max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "classifier__class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
    }


def _space_logreg(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "classifier__C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "classifier__class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
    }


PARAM_SPACES: dict[str, Any] = {
    "xgb": _space_xgb,
    "rf": _space_rf,
    "logreg": _space_logreg,
}


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def tune_model_optuna(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 40,
    folds: int = 5,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Chạy Optuna cho một model, trả về dict kết quả (schema như ``tune.py``)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = PARAM_SPACES[name](trial)
        pipeline = build_model_pipeline(X, build_estimator(name))
        pipeline.set_params(**params)
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise"
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.perf_counter() - start

    best_params = {
        f"classifier__{k}": v for k, v in study.best_params.items()
    }
    best_pipeline = build_model_pipeline(X, build_estimator(name))
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X, y)
    val = evaluate_model(best_pipeline, X_val, y_val)

    return {
        "model": name,
        "cv_pr_auc": round(float(study.best_value), 4),
        "val_pr_auc": round(val["pr_auc"], 4),
        "val_recall": round(val["recall"], 4),
        "val_f1": round(val["f1"], 4),
        "best_params": {
            k: (v if not isinstance(v, np.generic) else v.item())
            for k, v in best_params.items()
        },
        "n_trials": len(study.trials),
        "seconds": round(elapsed, 1),
    }


def update_tuning_results(
    result: dict[str, Any], path: Path = TUNING_RESULTS_PATH
) -> bool:
    """Cập nhật ``tuning_results.json`` nếu kết quả mới tốt hơn. Trả về ``True`` nếu ghi."""
    entries: list[dict[str, Any]] = []
    if Path(path).exists():
        entries = json.loads(Path(path).read_text(encoding="utf-8"))

    current = next((e for e in entries if e.get("model") == result["model"]), None)
    if current is not None and current.get("cv_pr_auc", 0) >= result["cv_pr_auc"]:
        return False

    entries = [e for e in entries if e.get("model") != result["model"]]
    entries.append(result)
    entries.sort(key=lambda e: e["model"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tinh chỉnh siêu tham số bằng Optuna.")
    parser.add_argument("--models", nargs="*", default=["xgb"], choices=MODEL_NAMES)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=None, help="Giới hạn thời gian (giây).")
    parser.add_argument("--output", type=Path, default=OPTUNA_RESULTS_PATH)
    parser.add_argument("--no-update-tuned", action="store_true", help="Không ghi đè tuning_results.json.")
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
        f"Optuna tuning {args.models} | {args.folds}-fold CV | scoring={SCORING} | "
        f"n_trials={args.n_trials} | timeout={args.timeout}\n"
    )

    results = []
    for name in args.models:
        res = tune_model_optuna(
            name, X, y, X_val, y_val, args.n_trials, args.folds, args.timeout
        )
        results.append(res)
        print(
            f"[{name:8}] cv_pr_auc={res['cv_pr_auc']:.4f} | val_pr_auc={res['val_pr_auc']:.4f} "
            f"val_recall={res['val_recall']:.4f} | {res['n_trials']} trials | {res['seconds']}s"
        )
        print(f"           best_params={res['best_params']}")

    best = max(results, key=lambda r: r["cv_pr_auc"])
    print(f"\n=> Tốt nhất theo CV PR-AUC: {best['model']} ({best['cv_pr_auc']:.4f})")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Đã lưu: {args.output}")

    if not args.no_update_tuned:
        for res in results:
            if update_tuning_results(res):
                print(
                    f"Đã cập nhật {TUNING_RESULTS_PATH.name}: {res['model']} "
                    f"-> cv_pr_auc={res['cv_pr_auc']:.4f}"
                )

    if not args.no_mlflow:
        for res in results:
            tracking.track_run(
                f"p4-optuna-{res['model']}",
                params={
                    "model": res["model"],
                    "n_trials": res["n_trials"],
                    "folds": args.folds,
                    "best_params": res["best_params"],
                },
                metrics={
                    "cv.pr_auc": res["cv_pr_auc"],
                    "val.pr_auc": res["val_pr_auc"],
                    "val.recall": res["val_recall"],
                    "val.f1": res["val_f1"],
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
