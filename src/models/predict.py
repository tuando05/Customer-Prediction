"""Inference: tải model đã huấn luyện và dự đoán churn.

Lớp :class:`ChurnPredictor` gói model (đã gồm sẵn bước tiền xử lý) và xử lý
việc làm sạch đặc trưng đầu vào giống hệt lúc huấn luyện.

Cách dùng nhanh::

    python -m src.models.predict --input data/processed/X_test.csv --limit 5
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.features.preprocess import clean_features


class ChurnPredictor:
    """Wrapper quanh pipeline sklearn đã huấn luyện."""

    def __init__(self, model) -> None:
        self.model = model

    @property
    def feature_names(self) -> list[str]:
        """Tên các đặc trưng mà model mong đợi (theo thứ tự).

        Với Pipeline, tên có thể nằm ở bước đầu tiên (ví dụ ``FeatureEngineer``)
        chứ không phải trên chính Pipeline.
        """
        names = getattr(self.model, "feature_names_in_", None)
        if names is None and hasattr(self.model, "steps"):
            for _, step in self.model.steps:
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    break
        return list(names) if names is not None else []

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = clean_features(X)
        expected = self.feature_names
        if expected:
            missing = [c for c in expected if c not in X.columns]
            if missing:
                raise ValueError(f"Thiếu đặc trưng bắt buộc: {missing}")
            extra = [c for c in X.columns if c not in expected]
            X = X.drop(columns=extra)
            X = X[expected]
        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về xác suất churn (lớp 1) cho từng dòng."""
        return self.model.predict_proba(self._prepare(X))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Trả về nhãn 0/1 theo ngưỡng quyết định."""
        return (self.predict_proba(X) >= threshold).astype(int)


def load_predictor(model_path: Path = MODEL_PATH) -> ChurnPredictor:
    """Tải model từ đĩa; báo lỗi rõ ràng nếu chưa có."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {model_path}. "
            "Hãy chạy `python -m src.models.train` trước."
        )
    return ChurnPredictor(joblib.load(model_path))


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """Trả về predictor dùng chung (cache 1 lần cho cả tiến trình)."""
    return load_predictor()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán churn trên file CSV đặc trưng.")
    parser.add_argument("--input", type=Path, required=True, help="CSV đặc trưng (không có target).")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Đường dẫn model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng quyết định.")
    parser.add_argument("--limit", type=int, default=10, help="Số dòng in ra.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    predictor = load_predictor(args.model)
    X = pd.read_csv(args.input)
    proba = predictor.predict_proba(X)
    labels = (proba >= args.threshold).astype(int)

    preview = pd.DataFrame(
        {"churn_probability": proba, "churn": labels}
    ).head(args.limit)
    print(f"Đã dự đoán {len(X)} dòng (threshold={args.threshold}). {args.limit} dòng đầu:")
    print(preview.to_string(index=False))
    print(f"\nTỷ lệ dự đoán churn: {labels.mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
