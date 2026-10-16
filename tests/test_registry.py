"""Tests cho registry mô hình."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import X_TEST_PATH
from src.models.registry import MODEL_NAMES, build_estimator, build_model_pipeline


def test_build_estimator_tat_ca_model():
    for name in MODEL_NAMES:
        estimator = build_estimator(name)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


def test_model_khong_hop_le():
    with pytest.raises(ValueError):
        build_estimator("khong_ton_tai")


def test_logreg_class_weight():
    estimator = build_estimator("logreg", class_weight="balanced")
    assert estimator.class_weight == "balanced"


def test_xgb_scale_pos_weight():
    estimator = build_estimator("xgb", scale_pos_weight=2.77)
    assert estimator.scale_pos_weight == 2.77


def test_xgb_truyen_du_tham_so_tune():
    """Chống tái phát bug: các tham số tune phải thực sự được truyền vào estimator."""
    estimator = build_estimator(
        "xgb",
        min_child_weight=6,
        gamma=0.78,
        reg_alpha=1.5,
        reg_lambda=2.5,
    )
    assert estimator.min_child_weight == 6
    assert estimator.gamma == 0.78
    assert estimator.reg_alpha == 1.5
    assert estimator.reg_lambda == 2.5


@pytest.mark.skipif(not X_TEST_PATH.exists(), reason="Chưa có dữ liệu đã chia.")
def test_pipeline_co_du_ba_buoc():
    X = pd.read_csv(X_TEST_PATH).head(20)
    pipeline = build_model_pipeline(X, build_estimator("logreg"))
    assert list(pipeline.named_steps) == ["features", "preprocessor", "classifier"]
