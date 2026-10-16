"""Tests cho lớp theo dõi MLflow (không cần MLflow thật)."""

from __future__ import annotations

from src import tracking
from src.tracking import _clean_param, _flatten


def test_flatten_lam_phang_dict_long_nhau():
    flat = _flatten({"val": {"pr_auc": 0.6, "recall": 0.8}, "model": "xgb"})
    assert flat == {"val.pr_auc": 0.6, "val.recall": 0.8, "model": "xgb"}


def test_flatten_giu_nguyen_gia_tri_khong_long():
    assert _flatten({"threshold": 0.53}) == {"threshold": 0.53}


def test_clean_param_bo_xuong_dong_va_cat_do_dai():
    text = _clean_param("a\nb\r\n" + "x" * 300)
    assert "\n" not in text and "\r" not in text
    assert len(text) == 250


def test_clean_param_ep_kieu():
    assert _clean_param(True) == "True"
    assert _clean_param(None) == "None"
    assert _clean_param(2.77) == "2.77"


def test_track_run_tat_thi_tra_ve_false():
    assert tracking.track_run("test", enabled=False) is False


def test_is_available_tra_ve_bool():
    assert isinstance(tracking.is_available(), bool)
