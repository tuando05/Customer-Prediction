"""Tests cho model inference."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import MODEL_PATH, X_TEST_PATH
from src.models.predict import load_predictor

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Chưa có model. Chạy `python -m src.models.train` trước.",
)


def test_predict_proba_trong_khoang_0_1():
    predictor = load_predictor()
    X = pd.read_csv(X_TEST_PATH).head(20)
    proba = predictor.predict_proba(X)
    assert len(proba) == 20
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_tra_nhan_0_1():
    predictor = load_predictor()
    X = pd.read_csv(X_TEST_PATH).head(20)
    labels = predictor.predict(X)
    assert set(labels).issubset({0, 1})


def test_thieu_dac_trung_bao_loi():
    predictor = load_predictor()
    X = pd.read_csv(X_TEST_PATH).head(5).drop(columns=["tenure"])
    with pytest.raises(ValueError, match="Thiếu đặc trưng"):
        predictor.predict_proba(X)
