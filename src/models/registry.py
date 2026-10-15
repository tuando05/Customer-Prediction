"""Registry các mô hình và hàm dựng pipeline cho bài toán churn.

Gom việc khởi tạo estimator và lắp pipeline (feature engineering + tiền xử lý +
classifier) vào một chỗ, để ``train.py`` và ``compare.py`` dùng chung, đảm bảo
mọi mô hình được đánh giá trên cùng một cách xử lý dữ liệu.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.features.build_features import FeatureEngineer, add_engineered_features
from src.features.preprocess import build_preprocessor

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


def build_model_pipeline(X: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    """Lắp pipeline hoàn chỉnh: feature engineering -> tiền xử lý -> estimator.

    ``ColumnTransformer`` được dựng từ các cột **sau** feature engineering nên
    ``FeatureEngineer`` phải đứng trước. Việc này chỉ dùng *tên cột* (không dùng
    giá trị) để cấu hình, nên có thể dựng trên toàn bộ tập train mà không rò rỉ.
    """
    engineered = add_engineered_features(X)
    preprocessor = build_preprocessor(engineered)
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
