"""Tests cho tuning (kiểm tra cấu hình, không chạy CV nặng)."""

from __future__ import annotations

from src.models.registry import MODEL_NAMES
from src.models.tune import PARAM_DISTRIBUTIONS, TUNED_MODELS


def test_param_keys_co_tien_to_classifier():
    for params in PARAM_DISTRIBUTIONS.values():
        for key in params:
            assert key.startswith("classifier__")


def test_mo_hinh_tune_hop_le():
    assert set(PARAM_DISTRIBUTIONS).issubset(set(MODEL_NAMES))
    assert set(TUNED_MODELS).issubset(set(MODEL_NAMES))
