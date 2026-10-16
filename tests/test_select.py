"""Tests cho chọn lọc đặc trưng (DropFeatures) và pipeline có bước 'select'."""

from __future__ import annotations

import pandas as pd

from src.features.select import (
    DUPLICATE_FEATURES,
    NO_INTERNET_DUMMY_SUFFIX,
    DropFeatures,
)
from src.models.feature_select import FEATURE_SETS
from src.models.registry import build_estimator, build_model_pipeline


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [1.0, 2.0],
            "is_month_to_month": [1.0, 0.0],
            "Contract_Month-to-month": [1.0, 0.0],
            "StreamingTV_No internet service": [0.0, 1.0],
        }
    )


def test_drop_theo_ten():
    sel = DropFeatures(drop=["is_month_to_month"]).fit(_frame())
    assert sel.columns_to_drop_ == ["is_month_to_month"]
    out = sel.transform(_frame())
    assert "is_month_to_month" not in out.columns
    assert "Contract_Month-to-month" in out.columns


def test_drop_theo_hau_to():
    sel = DropFeatures(drop_suffix=["_No internet service"]).fit(_frame())
    out = sel.transform(_frame())
    assert "StreamingTV_No internet service" not in out.columns


def test_feature_names_out_sau_fit():
    sel = DropFeatures(
        drop=DUPLICATE_FEATURES, drop_suffix=NO_INTERNET_DUMMY_SUFFIX
    ).fit(_frame())
    assert list(sel.feature_names_out_) == ["tenure", "Contract_Month-to-month"]
    assert sel.get_feature_names_out().tolist() == ["tenure", "Contract_Month-to-month"]


def test_feature_names_out_truoc_fit():
    sel = DropFeatures(drop=DUPLICATE_FEATURES)
    names = sel.get_feature_names_out(list(_frame().columns))
    assert "is_month_to_month" not in names
    assert "tenure" in names


def test_feature_sets_hop_le():
    assert set(FEATURE_SETS) == {"full", "no_dup", "no_internet_dummies", "lean"}
    assert FEATURE_SETS["full"] == {}
    for spec in FEATURE_SETS.values():
        assert set(spec).issubset({"drop", "drop_suffix"})


def test_pipeline_co_buoc_select_khi_drop():
    X = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
        }
    )
    y = pd.Series([1, 0, 0, 1])
    pipe = build_model_pipeline(
        X, build_estimator("dtree"), drop=DUPLICATE_FEATURES
    )
    assert "select" in pipe.named_steps
    pipe.fit(X, y)
    n_in = len(pipe.named_steps["select"].feature_names_in_)
    n_out = len(pipe.named_steps["select"].feature_names_out_)
    assert n_out == n_in - 1
    assert pipe.predict(X).shape == (4,)


def test_pipeline_khong_co_select_mac_dinh():
    X = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [10.0, 20.0]})
    pipe = build_model_pipeline(X, build_estimator("dtree"))
    assert "select" not in pipe.named_steps
    assert list(pipe.named_steps) == ["features", "preprocessor", "classifier"]
