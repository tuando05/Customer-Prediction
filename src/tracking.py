"""Theo dõi thí nghiệm với MLflow (Giai đoạn 3 / P3-1).

Module này là **lớp mỏng, tùy chọn (optional)** quanh MLflow:

- Lưu cục bộ vào ``mlruns/`` (file store, không cần server) theo mặc định.
- Nếu MLflow chưa cài → chỉ in cảnh báo và bỏ qua, **không** làm hỏng pipeline.
- Có thể tắt hoàn toàn bằng biến môi trường ``CHURN_MLFLOW=0`` hoặc cờ CLI
  ``--no-mlflow``.

Cách dùng::

    from src import tracking

    tracking.track_run(
        "final-xgb",
        params={"model": "xgb", "threshold": 0.53},
        metrics={"val": {"pr_auc": 0.67}, "test": {"pr_auc": 0.62}},
        artifacts=["artifacts/churn_model.joblib"],
    )
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

DEFAULT_ENABLED = os.environ.get("CHURN_MLFLOW", "1") != "0"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _import_mlflow():
    """Import MLflow theo kiểu lazy; trả về ``None`` nếu chưa cài."""
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        return None
    return mlflow


def is_available() -> bool:
    """MLflow đã được cài hay chưa."""
    return _import_mlflow() is not None


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Làm phẳng dict lồng nhau một cấp: ``{"val": {"pr_auc": 0.6}}`` → ``val.pr_auc``."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def _clean_param(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text[:250]


def _clean_metric(value: Any) -> float:
    return float(value)


def track_run(
    run_name: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Iterable[str | Path] | None = None,
    enabled: bool | None = None,
) -> bool:
    """Ghi một lần chạy vào MLflow.

    Trả về ``True`` nếu đã ghi, ``False`` nếu bị bỏ qua (tắt hoặc thiếu MLflow).
    Không bao giờ raise vì lỗi log — theo dõi thí nghiệm là phụ trợ.
    """
    if enabled is None:
        enabled = DEFAULT_ENABLED
    if not enabled:
        return False

    mlflow = _import_mlflow()
    if mlflow is None:
        print("[mlflow] Chưa cài đặt — bỏ qua theo dõi. `pip install mlflow` để bật.")
        return False

    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params({k: _clean_param(v) for k, v in _flatten(params).items()})
            if metrics:
                for key, value in _flatten(metrics).items():
                    try:
                        mlflow.log_metric(key, _clean_metric(value))
                    except (TypeError, ValueError):
                        continue
            for artifact in artifacts or []:
                path = Path(artifact)
                if not path.exists():
                    continue
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    mlflow.log_artifact(str(path), artifact_path="plots")
                else:
                    mlflow.log_artifact(str(path))
    except Exception as exc:  # pragma: no cover - phòng hờ, không nên làm hỏng pipeline
        print(f"[mlflow] Bỏ qua do lỗi khi log: {exc}")
        return False
    return True
