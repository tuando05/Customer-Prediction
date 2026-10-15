"""Tests cho feature engineering."""

from __future__ import annotations

import pandas as pd

from src.features.build_features import (
    NEW_FEATURE_NAMES,
    SERVICE_COLUMNS,
    FeatureEngineer,
    add_engineered_features,
)


def _sample() -> pd.DataFrame:
    row0 = {c: "No" for c in SERVICE_COLUMNS}
    row0.update(
        {
            "tenure": 5,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "PhoneService": "Yes",
            "OnlineSecurity": "Yes",
        }
    )
    row1 = {c: "No" for c in SERVICE_COLUMNS}
    row1.update(
        {
            "tenure": 60,
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
        }
    )
    return pd.DataFrame([row0, row1])


def test_them_du_cot_moi():
    out = add_engineered_features(_sample())
    for col in NEW_FEATURE_NAMES:
        assert col in out.columns


def test_num_services():
    out = add_engineered_features(_sample())
    assert out.loc[0, "num_services"] == 2
    assert out.loc[1, "num_services"] == 0


def test_tenure_bucket():
    out = add_engineered_features(_sample())
    assert out.loc[0, "tenure_bucket"] == "0-12"
    assert out.loc[1, "tenure_bucket"] == "49-72"


def test_payment_va_contract():
    out = add_engineered_features(_sample())
    assert out.loc[0, "is_automatic_payment"] == 0
    assert out.loc[1, "is_automatic_payment"] == 1
    assert out.loc[0, "is_month_to_month"] == 1
    assert out.loc[1, "is_month_to_month"] == 0


def test_feature_engineer_transformer():
    out = FeatureEngineer().fit_transform(_sample())
    assert isinstance(out, pd.DataFrame)
    assert set(NEW_FEATURE_NAMES).issubset(out.columns)
    assert len(out) == 2


def test_khong_sua_du_lieu_goc():
    df = _sample()
    add_engineered_features(df)
    assert "num_services" not in df.columns  # hàm không mutate input
