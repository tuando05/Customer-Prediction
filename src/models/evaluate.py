"""Đánh giá mô hình churn.

Cung cấp bộ metric thống nhất để so sánh mọi mô hình một cách công bằng:
Accuracy, Precision, Recall, F1, ROC-AUC và PR-AUC.

PR-AUC (average precision) đặc biệt quan trọng với dữ liệu mất cân bằng như
bài toán churn, vì ROC-AUC có thể lạc quan quá mức khi lớp thiểu số nhỏ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def get_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Lấy điểm xác suất (hoặc decision score) cho lớp churn dương."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính bộ metric từ nhãn thật và điểm số dự đoán."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    return metrics


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Đánh giá một mô hình đã huấn luyện trên tập (X, y)."""
    return compute_metrics(y, get_scores(model, X), threshold=threshold)


def format_metrics(metrics: dict[str, float]) -> str:
    """Định dạng dict metric thành một dòng dễ đọc."""
    parts = [f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS]
    cm = (
        f"TP={metrics['true_positive']} FP={metrics['false_positive']} "
        f"FN={metrics['false_negative']} TN={metrics['true_negative']}"
    )
    return f"threshold={metrics['threshold']:.2f} | " + " ".join(parts) + f" | {cm}"
