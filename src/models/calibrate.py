"""Calibration xác suất (Giai đoạn 4 / P4-4).

XGBoost (và cây nói chung) thường cho xác suất **chưa hiệu chỉnh** — dùng để xếp
hạng thì tốt nhưng độ tin cậy của con số xác suất thấp. Module này so sánh:

- ``none``: model gốc.
- ``sigmoid``: Platt scaling (LogisticRegression trên điểm số).
- ``isotonic``: hồi quy đẳng điệu (linh hoạt hơn, cần nhiều dữ liệu).

Đo bằng **Brier score**, **log loss** và đường cong calibration; PR-AUC/ROC-AUC
hầu như không đổi (calibration không đổi thứ hạng).

Cách dùng::

    python -m src.models.calibrate --model xgb
    python -m src.models.calibrate --model xgb --methods none isotonic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from src import tracking  # noqa: E402
from src.config import (  # noqa: E402
    CALIBRATION_CURVE_PATH,
    CALIBRATION_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.models.evaluate import evaluate_model  # noqa: E402
from src.models.registry import (  # noqa: E402
    MODEL_NAMES,
    build_estimator,
    build_model_pipeline,
    resolve_estimator_params,
)
from src.models.train import load_xy  # noqa: E402

METHODS = ["none", "sigmoid", "isotonic"]


def fit_calibrated(model: str, method: str, params: dict, X, y, folds: int = 5):
    """Fit pipeline (kèm calibration nếu ``method`` != ``none``)."""
    pipeline = build_model_pipeline(X, build_estimator(model, **params))
    if method == "none":
        pipeline.fit(X, y)
        return pipeline
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=folds)
    calibrated.fit(X, y)
    return calibrated


def score_calibration(estimator, X_val, y_val) -> dict[str, float]:
    """Metric xếp hạng + metric calibration trên tập validation."""
    base = evaluate_model(estimator, X_val, y_val)
    proba = estimator.predict_proba(X_val)[:, 1]
    return {
        "pr_auc": round(base["pr_auc"], 4),
        "roc_auc": round(base["roc_auc"], 4),
        "recall": round(base["recall"], 4),
        "f1": round(base["f1"], 4),
        "brier": round(float(brier_score_loss(y_val, proba)), 4),
        "log_loss": round(float(log_loss(y_val, proba)), 4),
    }


def compare_calibration(
    model: str,
    methods: list[str],
    X,
    y,
    X_val,
    y_val,
    params: dict | None = None,
    folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, tuple]]:
    """So sánh các phương pháp calibration; trả về (bảng điểm, dữ liệu đường cong)."""
    params = params or {}
    rows: dict[str, dict[str, float]] = {}
    curves: dict[str, tuple] = {}
    for method in methods:
        estimator = fit_calibrated(model, method, params, X, y, folds)
        rows[method] = score_calibration(estimator, X_val, y_val)
        proba = estimator.predict_proba(X_val)[:, 1]
        curves[method] = calibration_curve(y_val, proba, n_bins=10, strategy="quantile")
    df = pd.DataFrame(rows).T.sort_values("brier")
    return df, curves


def plot_calibration(curves: dict[str, tuple], path: Path) -> None:
    """Vẽ đường cong calibration (reliability diagram)."""
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="hoàn hảo")
    for method, (prob_true, prob_pred) in curves.items():
        plt.plot(prob_pred, prob_true, marker="o", label=method)
    plt.xlabel("Xác suất dự đoán")
    plt.ylabel("Tỷ lệ dương thực tế")
    plt.title("Đường cong calibration (validation)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh calibration xác suất.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="xgb")
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--tuned", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--class-weight", choices=["none", "balanced"], default="balanced"
    )
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--curve", type=Path, default=CALIBRATION_CURVE_PATH)
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

    print(f"Calibration | model={args.model} | methods={args.methods} | tuned={args.tuned}\n")
    df, curves = compare_calibration(
        args.model, args.methods, X, y, X_val, y_val, params=params, folds=args.folds
    )
    print(df.to_string())
    print(f"\n=> Brier tốt nhất: {df.index[0]}")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output)
        plot_calibration(curves, args.curve)
        print(f"Đã lưu: {args.output}")
        print(f"Đã lưu: {args.curve}")

    if not args.no_mlflow:
        for method, r in df.iterrows():
            tracking.track_run(
                f"p4-calibration-{method}",
                params={"model": args.model, "method": method, "folds": args.folds},
                metrics={
                    "val.pr_auc": r["pr_auc"],
                    "val.roc_auc": r["roc_auc"],
                    "val.brier": r["brier"],
                    "val.log_loss": r["log_loss"],
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
