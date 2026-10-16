"""Chọn lọc đặc trưng sau tiền xử lý (Giai đoạn 4 / P4-1).

Cung cấp :class:`DropFeatures` — transformer loại bỏ cột **theo tên** sau khi
``ColumnTransformer`` đã sinh ra ma trận đặc trưng (one-hot/scaled). Nhờ đó có
thể bỏ các cột dư thừa mà không phải viết lại pipeline tiền xử lý.

Ví dụ: bỏ đặc trưng trùng lặp ``is_month_to_month`` (đã có ``Contract``) và các
dummy ``*_No internet service`` (gần như bằng 0 theo SHAP).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Các cột dư thừa/≈0 phát hiện từ SHAP (M5) và EDA.
DUPLICATE_FEATURES = ["is_month_to_month"]
NO_INTERNET_DUMMY_SUFFIX = ["_No internet service"]


class DropFeatures(BaseEstimator, TransformerMixin):
    """Loại bỏ cột theo tên chính xác hoặc theo hậu tố.

    Kỳ vọng đầu vào là ``DataFrame`` (đặt ``preprocessor.set_output(transform="pandas")``
    trước bước này).
    """

    def __init__(self, drop=None, drop_suffix=None):
        self.drop = drop
        self.drop_suffix = drop_suffix

    def fit(self, X, y=None) -> DropFeatures:
        columns = list(pd.DataFrame(X).columns)
        drop = set(self.drop or [])
        suffixes = list(self.drop_suffix or [])

        to_drop = [c for c in columns if c in drop]
        to_drop += [c for c in columns if any(c.endswith(s) for s in suffixes)]

        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.columns_to_drop_ = sorted(set(to_drop))
        dropped = set(self.columns_to_drop_)
        self.feature_names_out_ = np.asarray(
            [c for c in columns if c not in dropped], dtype=object
        )
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        present = [c for c in self.columns_to_drop_ if c in df.columns]
        return df.drop(columns=present)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Trả về tên đặc trưng còn lại (cho sklearn ``get_feature_names_out``)."""
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        if input_features is None:
            raise ValueError("Cần truyền input_features khi chưa fit.")
        columns = [str(c) for c in np.asarray(input_features).ravel()]
        drop = set(self.drop or [])
        suffixes = list(self.drop_suffix or [])
        kept = [
            c
            for c in columns
            if c not in drop and not any(c.endswith(s) for s in suffixes)
        ]
        return np.asarray(kept, dtype=object)
