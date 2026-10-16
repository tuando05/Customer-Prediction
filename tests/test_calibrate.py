"""Tests cho calibration xác suất (P4-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.calibrate import METHODS, fit_calibrated, score_calibration


def _data():
    X = pd.DataFrame(
        {
            "tenure": list(range(1, 25)),
            "Contract": ["Month-to-month", "Two year", "One year", "Two year"] * 6,
            "MonthlyCharges": [10.0 + i for i in range(24)],
        }
    )
    y = pd.Series([1, 0, 0, 1] * 6)
    return X, y


def test_methods_day_du():
    assert METHODS == ["none", "sigmoid", "isotonic"]


def test_fit_calibrated_none_khong_boc():
    X, y = _data()
    est = fit_calibrated("dtree", "none", {}, X, y)
    assert not isinstance(est, CalibratedClassifierCV)
    proba = est.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_calibrated_sigmoid_tra_ve_calibrated():
    X, y = _data()
    est = fit_calibrated("dtree", "sigmoid", {}, X, y, folds=2)
    assert isinstance(est, CalibratedClassifierCV)
    proba = est.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_score_calibration_co_brier_va_log_loss():
    X, y = _data()
    est = fit_calibrated("dtree", "isotonic", {}, X, y, folds=2)
    scores = score_calibration(est, X, y)
    assert 0.0 <= scores["brier"] <= 1.0
    assert scores["log_loss"] >= 0.0
    assert set(scores) >= {"pr_auc", "roc_auc", "recall", "f1", "brier", "log_loss"}
    assert not np.isnan(scores["pr_auc"])
