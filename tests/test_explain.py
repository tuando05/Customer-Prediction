"""Tests cho tiện ích SHAP (không chạy SHAP thật)."""

from __future__ import annotations

import numpy as np

from src.models.explain import _positive_class_values, load_tuned_params


def test_positive_class_values_2d():
    arr = np.arange(6, dtype=float).reshape(2, 3)
    assert _positive_class_values(arr).shape == (2, 3)


def test_positive_class_values_list():
    class0 = np.zeros((2, 3))
    class1 = np.ones((2, 3))
    out = _positive_class_values([class0, class1])
    assert np.allclose(out, class1)


def test_positive_class_values_3d():
    arr = np.zeros((2, 3, 2))
    arr[:, :, 1] = 5.0
    out = _positive_class_values(arr)
    assert np.allclose(out, 5.0)


def test_load_tuned_params_tra_ve_dict():
    assert isinstance(load_tuned_params("xgb"), dict)
