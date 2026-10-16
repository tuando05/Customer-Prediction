"""Tests cho tuning bằng Optuna (cấu hình + cập nhật kết quả, không chạy study nặng)."""

from __future__ import annotations

import json

import optuna

from src.models.registry import MODEL_NAMES
from src.models.tune_optuna import PARAM_SPACES, update_tuning_results

FIXED = {
    "xgb": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.1,
        "scale_pos_weight": 2.77,
    },
    "rf": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    "logreg": {"C": 1.0, "class_weight": "balanced"},
}


def test_param_spaces_hop_le():
    assert set(PARAM_SPACES).issubset(set(MODEL_NAMES))


def test_moi_khong_gian_tra_ve_tham_so_co_tien_to():
    for name, space in PARAM_SPACES.items():
        trial = optuna.trial.FixedTrial(FIXED[name])
        params = space(trial)
        assert params, name
        assert all(k.startswith("classifier__") for k in params)


def test_update_tuning_results_ghi_khi_tot_hon(tmp_path):
    path = tmp_path / "tuning_results.json"
    path.write_text("[]", encoding="utf-8")

    better = {"model": "xgb", "cv_pr_auc": 0.70, "best_params": {"classifier__max_depth": 5}}
    assert update_tuning_results(better, path) is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["cv_pr_auc"] == 0.70


def test_update_tuning_results_bo_qua_khi_te_hon(tmp_path):
    path = tmp_path / "tuning_results.json"
    path.write_text(
        json.dumps([{"model": "xgb", "cv_pr_auc": 0.70}]), encoding="utf-8"
    )
    worse = {"model": "xgb", "cv_pr_auc": 0.65, "best_params": {}}
    assert update_tuning_results(worse, path) is False
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["cv_pr_auc"] == 0.70


def test_update_tuning_results_thay_the_cung_model(tmp_path):
    path = tmp_path / "tuning_results.json"
    path.write_text(
        json.dumps([{"model": "xgb", "cv_pr_auc": 0.60}, {"model": "rf", "cv_pr_auc": 0.58}]),
        encoding="utf-8",
    )
    better = {"model": "xgb", "cv_pr_auc": 0.68, "best_params": {}}
    assert update_tuning_results(better, path) is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    models = sorted(e["model"] for e in saved)
    assert models == ["rf", "xgb"]
    assert next(e for e in saved if e["model"] == "xgb")["cv_pr_auc"] == 0.68
