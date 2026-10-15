"""Tests cho chọn threshold."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import pick_threshold, threshold_curve

Y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
SCORES = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.6, 0.3, 0.7, 0.45])


def test_threshold_curve_hop_le():
    curve = threshold_curve(Y, SCORES)
    assert {"threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"} <= set(curve.columns)
    assert curve["f1"].between(0, 1).all()
    assert curve["recall"].between(0, 1).all()


def test_pick_max_f1():
    curve = threshold_curve(Y, SCORES)
    chosen = pick_threshold(Y, SCORES, strategy="max_f1")
    assert np.isclose(chosen["f1"], curve["f1"].max())


def test_pick_target_recall():
    chosen = pick_threshold(Y, SCORES, strategy="target_recall", min_recall=0.8)
    assert chosen["recall"] >= 0.8


def test_pick_min_cost():
    chosen = pick_threshold(Y, SCORES, strategy="min_cost", cost_fn=5, cost_fp=1)
    curve = threshold_curve(Y, SCORES)
    costs = 5 * curve["fn"] + 1 * curve["fp"]
    assert np.isclose(chosen["cost"], costs.min())


def test_strategy_khong_hop_le():
    with pytest.raises(ValueError):
        pick_threshold(Y, SCORES, strategy="sai")
