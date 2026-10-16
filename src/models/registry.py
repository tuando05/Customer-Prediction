"""Registry các mô hình và hàm dựng pipeline cho bài toán churn.

Gom việc khởi tạo estimator và lắp pipeline (feature engineering + tiền xử lý +
classifier) vào một chỗ, để ``train.py`` và ``compare.py`` dùng chung, đảm bảo
mọi mô hình được đánh giá trên cùng một cách xử lý dữ liệu.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, TUNING_RESULTS_PATH
from src.features.build_features import FeatureEngineer, add_engineered_features
from src.features.preprocess import build_preprocessor
from src.features.select import DropFeatures

MODEL_NAMES = ["logreg", "dtree", "rf", "xgb"]


def _make_logreg(**kw) -> LogisticRegression:
    return LogisticRegression(
        C=kw.get("C", 1.0),
        max_iter=kw.get("max_iter", 1000),
        class_weight=kw.get("class_weight"),
        random_state=RANDOM_STATE,
    )


def _make_dtree(**kw) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=kw.get("max_depth"),
        min_samples_leaf=kw.get("min_samples_leaf", 1),
        class_weight=kw.get("class_weight"),
        random_state=RANDOM_STATE,
    )


def _make_rf(**kw) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=kw.get("n_estimators", 300),
        max_depth=kw.get("max_depth"),
        min_samples_leaf=kw.get("min_samples_leaf", 1),
        min_samples_split=kw.get("min_samples_split", 2),
        max_features=kw.get("max_features", "sqrt"),
        class_weight=kw.get("class_weight"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _make_xgb(**kw) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=kw.get("n_estimators", 300),
        learning_rate=kw.get("learning_rate", 0.1),
        max_depth=kw.get("max_depth", 4),
        subsample=kw.get("subsample", 0.9),
        colsample_bytree=kw.get("colsample_bytree", 0.9),
        min_child_weight=kw.get("min_child_weight", 1),
        gamma=kw.get("gamma", 0.0),
        reg_alpha=kw.get("reg_alpha", 0.0),
        reg_lambda=kw.get("reg_lambda", 1.0),
        scale_pos_weight=kw.get("scale_pos_weight") or 1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )


MODEL_FACTORIES: dict[str, Callable[..., BaseEstimator]] = {
    "logreg": _make_logreg,
    "dtree": _make_dtree,
    "rf": _make_rf,
    "xgb": _make_xgb,
}


def build_estimator(name: str, **kwargs) -> BaseEstimator:
    """Khởi tạo estimator theo tên (``logreg``, ``dtree``, ``rf``, ``xgb``)."""
    key = name.lower()
    if key not in MODEL_FACTORIES:
        raise ValueError(f"Model không hợp lệ: {name}. Chọn một trong {MODEL_NAMES}.")
    return MODEL_FACTORIES[key](**kwargs)


def build_model_pipeline(
    X: pd.DataFrame,
    estimator: BaseEstimator,
    drop: list[str] | None = None,
    drop_suffix: list[str] | None = None,
) -> Pipeline:
    """Lắp pipeline hoàn chỉnh: feature engineering -> tiền xử lý -> (chọn lọc) -> estimator.

    ``ColumnTransformer`` được dựng từ các cột **sau** feature engineering nên
    ``FeatureEngineer`` phải đứng trước. Việc này chỉ dùng *tên cột* (không dùng
    giá trị) để cấu hình, nên có thể dựng trên toàn bộ tập train mà không rò rỉ.

    Nếu truyền ``drop``/``drop_suffix``, thêm bước :class:`DropFeatures` sau tiền
    xử lý (khi đó ``ColumnTransformer`` xuất ra pandas để loại cột theo tên).
    """
    engineered = add_engineered_features(X)
    preprocessor = build_preprocessor(engineered)

    steps: list[tuple[str, object]] = [
        ("features", FeatureEngineer()),
        ("preprocessor", preprocessor),
    ]
    if drop or drop_suffix:
        preprocessor.set_output(transform="pandas")
        steps.append(("select", DropFeatures(drop=drop, drop_suffix=drop_suffix)))
    steps.append(("classifier", estimator))

    return Pipeline(steps=steps)


# ---------------------------------------------------------------------------
# Tham số đã tune & xử lý mất cân bằng
# ---------------------------------------------------------------------------


def load_tuned_params(model: str, path: Path = TUNING_RESULTS_PATH) -> dict:
    """Đọc ``best_params`` đã tune cho một model (bỏ tiền tố ``classifier__``)."""
    if not Path(path).exists():
        return {}
    for entry in json.loads(Path(path).read_text(encoding="utf-8")):
        if entry.get("model") == model:
            return {
                k.replace("classifier__", ""): v
                for k, v in entry.get("best_params", {}).items()
            }
    return {}


def resolve_estimator_params(
    model: str,
    class_weight: str | None = "none",
    y: pd.Series | None = None,
    tuned: bool = True,
) -> dict:
    """Gộp tham số đã tune với cấu hình xử lý mất cân bằng.

    - ``tuned=True``: lấy ``best_params`` từ ``reports/tuning_results.json`` (nếu có).
    - ``class_weight='balanced'``: bật class_weight; với XGBoost đặt
      ``scale_pos_weight = neg/pos`` (chỉ khi chưa có từ tune).
    """
    params = load_tuned_params(model) if tuned else {}

    if class_weight == "balanced":
        params.setdefault("class_weight", "balanced")
        if y is not None and "scale_pos_weight" not in params:
            positives = int((y == 1).sum())
            negatives = int((y == 0).sum())
            params["scale_pos_weight"] = negatives / positives if positives else 1.0
    else:
        params.pop("class_weight", None)
        params.pop("scale_pos_weight", None)
    return params
