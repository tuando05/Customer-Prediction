"""Phân tích SHAP để giải thích mô hình (Giai đoạn 2 / M5).

Tính giá trị SHAP trên không gian đặc trưng **sau tiền xử lý** (one-hot/scaled),
lưu:

- ``reports/shap_importance.csv``: mức đóng góp trung bình |SHAP| của từng đặc trưng.
- ``reports/shap_bar.png`` và ``reports/shap_summary.png``: biểu đồ.

Mặc định dùng đặc trưng đã tune (nếu có ``reports/tuning_results.json``).

Cách dùng::

    python -m src.models.explain --model xgb
    python -m src.models.explain --model logreg --class-weight balanced --sample 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.config import (  # noqa: E402
    RANDOM_STATE,
    SHAP_BAR_PATH,
    SHAP_IMPORTANCE_PATH,
    SHAP_SUMMARY_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
)
from src.models.registry import (  # noqa: E402
    MODEL_NAMES,
    build_estimator,
    build_model_pipeline,
    load_tuned_params,
    resolve_estimator_params,
)
from src.models.train import load_xy  # noqa: E402

TREE_MODELS = {"xgb", "rf", "dtree"}


def _positive_class_values(shap_values) -> np.ndarray:
    """Chuẩn hóa output SHAP về ma trận (n_samples, n_features) cho lớp churn."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[0])
    values = np.asarray(shap_values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        return values[:, :, 1]
    return values


def _make_explainer(model_name: str, classifier, X_transformed: np.ndarray):
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(classifier)
    return shap.LinearExplainer(classifier, X_transformed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích SHAP.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="xgb")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--sample", type=int, default=500, help="Số mẫu để tính SHAP.")
    parser.add_argument("--top", type=int, default=20, help="Số đặc trưng hiển thị.")
    parser.add_argument("--tuned", action="store_true", default=True, help="Dùng tham số đã tune.")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    X, y = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)

    params = resolve_estimator_params(
        args.model, class_weight=args.class_weight, y=y, tuned=args.tuned
    )
    estimator = build_estimator(args.model, **params)
    pipeline = build_model_pipeline(X, estimator)
    pipeline.fit(X, y)

    sample = X.sample(n=min(args.sample, len(X)), random_state=RANDOM_STATE)
    features_step = pipeline.named_steps["features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(features_step.transform(sample))
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = _make_explainer(args.model, classifier, X_transformed)
    values = _positive_class_values(explainer.shap_values(X_transformed))

    importance = (
        pd.Series(np.abs(values).mean(axis=0), index=feature_names)
        .sort_values(ascending=False)
    )

    print(f"Model={args.model} | giải thích {len(sample)} mẫu | {len(feature_names)} đặc trưng sau biến đổi")
    print(f"\nTop {args.top} đặc trưng quan trọng nhất (mean |SHAP|):")
    print(importance.head(args.top).round(4).to_string())

    if not args.no_save:
        SHAP_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        importance.round(6).rename("mean_abs_shap").to_csv(SHAP_IMPORTANCE_PATH)

        shap.summary_plot(values, X_transformed, feature_names=feature_names,
                          plot_type="bar", max_display=args.top, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_BAR_PATH, dpi=120, bbox_inches="tight")
        plt.close()

        shap.summary_plot(values, X_transformed, feature_names=feature_names,
                          max_display=args.top, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PATH, dpi=120, bbox_inches="tight")
        plt.close()

        print(f"\nĐã lưu: {SHAP_IMPORTANCE_PATH}")
        print(f"Đã lưu: {SHAP_BAR_PATH}")
        print(f"Đã lưu: {SHAP_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
