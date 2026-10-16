"""Kiểm tra chất lượng dữ liệu (data validation) cho bộ Telco Customer Churn.

Script này kiểm tra "sức khỏe" của tập dữ liệu thô trước khi đưa vào modeling:

1. Kích thước và tên cột.
2. Missing values (bao gồm cả chuỗi khoảng trắng như ``" "`` — isna() không bắt được).
3. Trùng lặp (dòng và ID khách hàng).
4. Cột số bị đọc nhầm thành chuỗi (ví dụ ``TotalCharges``) và số giá trị không ép được sang số.
5. Cột target: phân phối lớp và mức mất cân bằng.
6. Cột phân loại: số lượng giá trị, danh sách giá trị, cột hằng số.
7. Cột số: min/max/mean/std và cờ bất thường (giá trị âm, vô cực).

Cách dùng::

    python -m src.features.validate_data
    python -m src.features.validate_data --path data/raw/otel.csv --target Churn
    python -m src.features.validate_data --no-report        # chỉ in ra màn hình

Kết quả được in ra màn hình và (mặc định) lưu vào
``reports/data_check_report.txt``. Mã thoát: 0 nếu không có lỗi FAIL, 1 nếu có.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cấu hình mặc định
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_REPORT_PATH = Path("reports/data_check_report.txt")
DEFAULT_TARGET = "Churn"
DEFAULT_ID_COL = "customerID"

# Cột phân loại có <= ngưỡng này giá trị được coi là "hợp lệ để one-hot".
CATEGORICAL_MAX_UNIQUE = 15

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Cấu trúc report
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Kết quả của một hạng mục kiểm tra."""

    name: str
    status: str = STATUS_PASS
    lines: list[str] = field(default_factory=list)


class DataCheckReport:
    """Thu thập các CheckResult và render thành báo cáo dạng text."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @property
    def counts(self) -> dict[str, int]:
        counts = {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 0}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def render(self, source: str) -> str:
        counts = self.counts
        bar = "=" * 72
        lines = [
            bar,
            "DATA CHECK REPORT — Telco Customer Churn",
            bar,
            f"Nguồn: {source}",
            (
                f"Kiểm tra: {len(self.results)} | "
                f"PASS: {counts[STATUS_PASS]} | "
                f"WARN: {counts[STATUS_WARN]} | "
                f"FAIL: {counts[STATUS_FAIL]}"
            ),
        ]
        for i, r in enumerate(self.results, 1):
            lines.append("")
            lines.append(f"[{i}] {r.name} — [{r.status}]")
            lines.extend(f"    {line}" for line in r.lines)
        lines.append("")
        lines.append(bar)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Các hàm kiểm tra
# ---------------------------------------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    """Đọc CSV, giữ nguyên giá trị gốc (không tự ép kiểu) để kiểm tra trung thực."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    return pd.read_csv(path, keep_default_na=True)


def check_shape(df: pd.DataFrame) -> CheckResult:
    r = CheckResult("Kích thước dữ liệu")
    r.lines.append(f"{df.shape[0]} dòng x {df.shape[1]} cột")
    r.lines.append(f"Cột: {', '.join(map(str, df.columns))}")
    if df.empty:
        r.status = STATUS_FAIL
        r.lines.append("=> DataFrame rỗng!")
    return r


def _blank_string_mask(series: pd.Series) -> pd.Series:
    """Đánh dấu các giá trị là chuỗi nhưng chỉ chứa khoảng trắng (vd: ' ', '\\t')."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)
    return series.astype("string").str.strip().eq("").fillna(False)


def find_numeric_like_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Trả về các cột phi số mà (gần như) toàn bộ giá trị không-trống ép được sang số.

    Đây là dấu hiệu cột bị đọc nhầm thành chuỗi, ví dụ ``TotalCharges``.
    """
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    result: list[str] = []
    for col in df.columns:
        if col in numeric_cols or col == target:
            continue
        blank = _blank_string_mask(df[col])
        non_blank = df[col].notna() & ~blank
        if not non_blank.any():
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced[non_blank].notna().all():
            result.append(col)
    return result


def check_missing(df: pd.DataFrame) -> CheckResult:
    r = CheckResult("Missing values")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]

    if na_counts.empty:
        r.lines.append("Không có NaN theo pandas.isna().")
    else:
        for col, cnt in na_counts.items():
            r.lines.append(f"{col}: {cnt} ({cnt / len(df):.2%})")
        r.status = STATUS_WARN

    # Phát hiện chuỗi khoảng trắng đóng vai trò như missing.
    blanks: dict[str, int] = {}
    for col in df.columns:
        mask = _blank_string_mask(df[col])
        n = int(mask.sum())
        if n > 0:
            blanks[col] = n
    if blanks:
        r.lines.append("Chuỗi khoảng trắng (thực chất là missing nhưng isna() bỏ sót):")
        for col, cnt in blanks.items():
            r.lines.append(f"  {col}: {cnt} ({cnt / len(df):.2%})")
        r.status = STATUS_WARN
    else:
        r.lines.append("Không có chuỗi khoảng trắng.")

    if r.status == STATUS_PASS:
        r.lines.append("=> Không phát hiện missing.")
    return r


def check_duplicates(df: pd.DataFrame, id_col: str | None) -> CheckResult:
    r = CheckResult("Trùng lặp")
    dup_rows = int(df.duplicated().sum())
    r.lines.append(f"Dòng trùng lặp hoàn toàn: {dup_rows}")
    if dup_rows:
        r.status = STATUS_WARN

    if id_col and id_col in df.columns:
        dup_ids = int(df[id_col].duplicated().sum())
        r.lines.append(f"Giá trị trùng ở cột ID '{id_col}': {dup_ids}")
        if dup_ids:
            r.status = STATUS_FAIL
    elif id_col:
        r.lines.append(f"(Bỏ qua cột ID '{id_col}': không có trong dữ liệu)")
    return r


def check_dtypes(df: pd.DataFrame, target: str | None) -> CheckResult:
    r = CheckResult("Kiểu dữ liệu & cột số bị đọc nhầm")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    r.lines.append(f"Cột số ({len(numeric_cols)}): {', '.join(numeric_cols) or '(không có)'}")
    r.lines.append(f"Cột phi số ({len(non_numeric_cols)}): {', '.join(map(str, non_numeric_cols)) or '(không có)'}")

    # Cột phi số nhưng gần như toàn giá trị số -> nghi đọc nhầm kiểu (như TotalCharges).
    numeric_like = find_numeric_like_columns(df, target)
    if numeric_like:
        r.status = STATUS_WARN
        r.lines.append("Cột có tiềm năng là số nhưng đang ở dạng chuỗi (nên ép sang số):")
        for col in numeric_like:
            blank = _blank_string_mask(df[col])
            n_blank = int(blank.sum())
            extra = f" ({n_blank} giá trị trống sẽ thành NaN sau khi ép)" if n_blank else ""
            r.lines.append(f"  {col}{extra}")

    # Cột phi số có vài giá trị không ép được sang số (không tính giá trị trống).
    for col in non_numeric_cols:
        if col in numeric_like or col == target:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        blank = _blank_string_mask(df[col])
        failures = int((coerced.isna() & ~blank & df[col].notna()).sum())
        if 0 < failures < len(df):
            r.status = STATUS_WARN
            bad = df.loc[
                coerced.isna() & ~blank & df[col].notna(), col
            ].unique()[:5]
            r.lines.append(
                f"{col}: {failures} giá trị không ép được sang số — ví dụ {list(map(repr, bad))}"
            )
    return r


def check_target(df: pd.DataFrame, target: str | None) -> CheckResult:
    r = CheckResult(f"Cột target: {target}")
    if not target or target not in df.columns:
        r.status = STATUS_FAIL
        r.lines.append("Không tìm thấy cột target trong dữ liệu.")
        return r

    vc = df[target].value_counts(dropna=False)
    total = len(df)
    for value, cnt in vc.items():
        r.lines.append(f"{value!r}: {cnt} ({cnt / total:.2%})")

    n_classes = int((vc > 0).sum())
    r.lines.append(f"Số lớp: {n_classes}")
    if n_classes == 2:
        minority = int(vc.min())
        ratio = vc.max() / minority if minority else float("inf")
        r.lines.append(f"Tỷ lệ mất cân bằng (lớn/nhỏ): {ratio:.2f}:1")
        if ratio >= 3:
            r.status = STATUS_WARN
            r.lines.append("=> Mất cân bằng lớp đáng kể, cần cân nhắc class weights / resampling.")
        elif ratio >= 1.5:
            r.status = STATUS_WARN
            r.lines.append("=> Mất cân bằng nhẹ.")
        else:
            r.lines.append("=> Các lớp khá cân bằng.")
    else:
        r.status = STATUS_WARN
        r.lines.append("=> Target không phải bài toán 2 lớp — kiểm tra lại.")
    return r


def check_categorical(df: pd.DataFrame, target: str | None, id_col: str | None) -> CheckResult:
    r = CheckResult("Cột phân loại")
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    numeric_like = set(find_numeric_like_columns(df, target))
    cat_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in numeric_like and c != target and c != id_col
    ]

    if not cat_cols:
        r.lines.append("(không có cột phân loại)")
        return r

    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        values = df[col].dropna().unique()
        preview = ", ".join(map(repr, values[: CATEGORICAL_MAX_UNIQUE])) if len(values) <= CATEGORICAL_MAX_UNIQUE else "quá nhiều để liệt kê"
        r.lines.append(f"{col}: {nunique} giá trị | {preview}")
        if nunique <= 1:
            r.status = STATUS_WARN
            r.lines.append("  => cột hằng số, nên loại bỏ.")
        elif nunique > CATEGORICAL_MAX_UNIQUE:
            r.status = STATUS_WARN
            r.lines.append(f"  => cardinality cao ({nunique}), cân nhắc xử lý riêng.")
    return r


def check_numeric(df: pd.DataFrame, target: str | None) -> CheckResult:
    r = CheckResult("Cột số")
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target
    ]
    if not numeric_cols:
        r.lines.append("(không có cột số)")
        return r

    desc = df[numeric_cols].describe().T
    for col, row in desc.iterrows():
        r.lines.append(
            f"{col}: min={row['min']:.2f} max={row['max']:.2f} "
            f"mean={row['mean']:.2f} std={row['std']:.2f}"
        )

    negatives = {
        col: int((df[col] < 0).sum()) for col in numeric_cols if (df[col] < 0).any()
    }
    if negatives:
        r.status = STATUS_WARN
        for col, cnt in negatives.items():
            r.lines.append(f"{col}: có {cnt} giá trị âm — kiểm tra lại.")

    inf_mask = np.isinf(df[numeric_cols].to_numpy(dtype="float64", na_value=np.nan))
    n_inf = int(np.nansum(inf_mask))
    if n_inf:
        r.status = STATUS_WARN
        r.lines.append(f"Phát hiện {n_inf} giá trị vô cực (inf).")
    return r


# ---------------------------------------------------------------------------
# Điều phối
# ---------------------------------------------------------------------------


def run_checks(
    df: pd.DataFrame,
    target: str | None = DEFAULT_TARGET,
    id_col: str | None = DEFAULT_ID_COL,
) -> DataCheckReport:
    report = DataCheckReport()
    report.add(check_shape(df))
    report.add(check_missing(df))
    report.add(check_duplicates(df, id_col))
    report.add(check_dtypes(df, target))
    report.add(check_target(df, target))
    report.add(check_categorical(df, target, id_col))
    report.add(check_numeric(df, target))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng dữ liệu churn.")
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA_PATH, help="Đường dẫn file CSV.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Tên cột target.")
    parser.add_argument("--id-col", type=str, default=DEFAULT_ID_COL, help="Tên cột ID khách hàng.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Đường dẫn file report đầu ra.")
    parser.add_argument("--no-report", action="store_true", help="Không ghi report ra file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Đảm bảo in được tiếng Việt trên console Windows (mặc định cp1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    df = load_data(args.path)
    report = run_checks(df, target=args.target, id_col=args.id_col)
    text = report.render(str(args.path))
    print(text)

    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"\nĐã lưu report: {args.report}")

    return 1 if report.counts[STATUS_FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
