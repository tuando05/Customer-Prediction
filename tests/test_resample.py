"""Tests cho so sánh chiến lược mất cân bằng (P4-2)."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from src.models.resample import STRATEGIES, _make_sampler, build_resample_pipeline


def _data():
    X = pd.DataFrame(
        {
            "tenure": list(range(1, 13)),
            "Contract": ["Month-to-month", "Two year", "One year", "Two year"]
            * 3,
            "MonthlyCharges": [10.0 * i for i in range(1, 13)],
        }
    )
    y = pd.Series([1, 0] * 6)
    return X, y


def test_strategies_day_du():
    assert STRATEGIES == ["none", "balanced", "smote", "smote_tomek"]


def test_make_sampler():
    assert _make_sampler("smote").__class__ is SMOTE
    assert _make_sampler("smote_tomek").__class__.__name__ == "SMOTETomek"
    assert _make_sampler("none") is None
    assert _make_sampler("balanced") is None


def test_pipeline_khong_sampler_voi_none_va_balanced():
    X, y = _data()
    for strategy in ("none", "balanced"):
        pipe = build_resample_pipeline(X, "dtree", strategy, {}, y=y)
        assert isinstance(pipe, Pipeline)
        assert "sampler" not in pipe.named_steps


def test_pipeline_co_sampler_voi_smote():
    X, y = _data()
    pipe = build_resample_pipeline(X, "dtree", "smote", {}, y=y)
    assert isinstance(pipe, ImbPipeline)
    assert "sampler" in pipe.named_steps


def test_smote_fit_duoc_va_du_doan():
    X, y = _data()
    pipe = build_resample_pipeline(X, "dtree", "smote", {}, y=y)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
