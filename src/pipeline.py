"""Pipeline huấn luyện end-to-end (Giai đoạn 2 / M-H).

Chạy trọn luồng từ dữ liệu thô đến model + threshold sẵn sàng phục vụ:

1. Làm sạch dữ liệu thô -> ``data/processed/churn_clean.csv``.
2. Chia train/val/test (stratified) -> ``data/processed/*.csv``.
3. Huấn luyện model cuối (mặc định XGBoost đã tune + balanced), chọn threshold
   trên validation, đánh giá trên test, lưu ``artifacts/churn_model.joblib`` và
   ``artifacts/threshold.json``.

Cách dùng::

    python -m src.pipeline
    python -m src.pipeline --model xgb
"""

from __future__ import annotations

import argparse
import sys

from src.features import preprocess, split
from src.models import select_threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline end-to-end cho churn.")
    parser.add_argument("--model", default="xgb", help="Model cuối (mặc định xgb).")
    parser.add_argument("--no-test", action="store_true", help="Không đánh giá trên test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)

    print("=" * 72)
    print("[1/3] Làm sạch dữ liệu")
    print("=" * 72)
    preprocess.main([])

    print("\n" + "=" * 72)
    print("[2/3] Chia train/val/test")
    print("=" * 72)
    split.main([])

    print("\n" + "=" * 72)
    print("[3/3] Huấn luyện model cuối + chọn threshold")
    print("=" * 72)
    threshold_args = ["--model", args.model]
    if not args.no_test:
        threshold_args.append("--include-test")
    select_threshold.main(threshold_args)

    print("\nHoàn tất pipeline. Model & threshold đã sẵn sàng trong artifacts/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
