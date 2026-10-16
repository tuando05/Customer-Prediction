"""Tạo mẫu dữ liệu nhỏ (được commit) để test/CI chạy được.

Lấy mẫu **phân tầng theo nhãn churn** từ dữ liệu thô, lưu vào
``tests/data/sample_raw.csv``. Nhờ đó CI (nơi không có ``data/raw`` bị gitignore)
vẫn huấn luyện được một model nhỏ và chạy bộ test thật.

Cách dùng::

    python -m scripts.make_sample_data            # 200 dòng
    python -m scripts.make_sample_data --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "sample_raw.csv"


def make_sample(
    n: int = 200,
    seed: int = 42,
    raw_path: Path = RAW_DATA_PATH,
    output: Path = SAMPLE_PATH,
) -> pd.DataFrame:
    """Lấy mẫu phân tầng ``n`` dòng từ dữ liệu thô và lưu ra CSV."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu thô: {raw_path}. "
            "Tải dataset Telco và đặt vào data/raw/ trước."
        )
    df = pd.read_csv(raw_path)
    n_classes = df[TARGET_COL].nunique()
    per_class = max(1, n // n_classes)
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in df.groupby(TARGET_COL, sort=True)
    ]
    sample = pd.concat(parts, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    return sample


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Tạo mẫu dữ liệu nhỏ cho test/CI.")
    parser.add_argument("--n", type=int, default=200, help="Tổng số dòng mong muốn.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=SAMPLE_PATH)
    args = parser.parse_args(argv)

    sample = make_sample(args.n, args.seed, output=args.output)
    print(f"Đã lưu {len(sample)} dòng -> {args.output}")
    print(f"Tỷ lệ churn: {sample[TARGET_COL].value_counts(normalize=True).round(3).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
