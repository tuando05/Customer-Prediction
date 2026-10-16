"""Chọn threshold quyết định tối ưu (Giai đoạn 2 / M-G).

Huấn luyện model tốt nhất hiện tại (mặc định ``logreg`` + ``class_weight='balanced'``),
dò threshold trên tập **validation**, chọn điểm vận hành theo mục tiêu, rồi lưu
model + threshold để API dùng.

Cách dùng::

    python -m src.models.select_threshold                          # tối đa F1
    python -m src.models.select_threshold --strategy target_recall --min-recall 0.8
    python -m src.models.select_threshold --strategy min_cost --cost-fn 5 --cost-fp 1
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import tracking
from src.config import (
    MODEL_PATH,
    THRESHOLD_CURVE_PATH,
    THRESHOLD_PATH,
    X_TEST_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
)
from src.models.evaluate import (
    evaluate_model,
    format_metrics,
    get_scores,
    pick_threshold,
    threshold_curve,
)
from src.models.registry import (
    MODEL_NAMES,
    build_estimator,
    build_model_pipeline,
    resolve_estimator_params,
)
from src.models.train import load_xy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chọn threshold tối ưu cho model churn.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="xgb")
    parser.add_argument(
        "--class-weight", choices=["none", "balanced"], default="balanced"
    )
    parser.add_argument(
        "--tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng tham số đã tune (reports/tuning_results.json) nếu có.",
    )
    parser.add_argument(
        "--strategy",
        choices=["max_f1", "target_recall", "min_cost"],
        default="max_f1",
    )
    parser.add_argument("--min-recall", type=float, default=None, help="Cho strategy target_recall.")
    parser.add_argument("--cost-fn", type=float, default=1.0, help="Cho strategy min_cost.")
    parser.add_argument("--cost-fp", type=float, default=1.0, help="Cho strategy min_cost.")
    parser.add_argument("--include-test", action="store_true", help="Đánh giá thêm trên test.")
    parser.add_argument("--no-save", action="store_true", help="Không ghi model/threshold.")
    parser.add_argument("--no-mlflow", action="store_true", help="Không log vào MLflow.")
    return parser.parse_args(argv)


def select_threshold(
    model: str,
    class_weight: str | None,
    strategy: str,
    min_recall: float | None,
    cost_fn: float,
    cost_fp: float,
    X_train,
    y_train,
    X_val,
    y_val,
    tuned: bool = True,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Fit model, dò threshold trên val, trả về (pipeline, threshold_info, curve)."""
    params = resolve_estimator_params(model, class_weight=class_weight, y=y_train, tuned=tuned)
    estimator = build_estimator(model, **params)
    pipeline = build_model_pipeline(X_train, estimator)
    pipeline.fit(X_train, y_train)

    scores = get_scores(pipeline, X_val)
    curve = threshold_curve(y_val, scores)
    chosen = pick_threshold(
        y_val,
        scores,
        strategy=strategy,
        min_recall=min_recall,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    return pipeline, chosen, curve


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    class_weight = None if args.class_weight == "none" else "balanced"

    X_train, y_train = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

    pipeline, chosen, curve = select_threshold(
        args.model, class_weight, args.strategy, args.min_recall, args.cost_fn, args.cost_fp,
        X_train, y_train, X_val, y_val, tuned=args.tuned,
    )

    base = evaluate_model(pipeline, X_val, y_val, threshold=0.5)
    best = evaluate_model(pipeline, X_val, y_val, threshold=chosen["threshold"])

    print(f"Model={args.model} class_weight={class_weight} tuned={args.tuned} | strategy={args.strategy}")
    print(f"  [val @0.50] {format_metrics(base)}")
    print(f"  [val @chosen={chosen['threshold']:.2f}] {format_metrics(best)}")

    metrics = {"val_at_0.50": base, "val_at_chosen": best}
    if args.include_test:
        X_test, y_test = load_xy(X_TEST_PATH, Y_TEST_PATH)
        metrics["test_at_chosen"] = evaluate_model(
            pipeline, X_test, y_test, threshold=chosen["threshold"]
        )
        print(f"  [test @chosen] {format_metrics(metrics['test_at_chosen'])}")

    if not args.no_save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        THRESHOLD_PATH.write_text(
            json.dumps(
                {
                    "threshold": chosen["threshold"],
                    "strategy": args.strategy,
                    "model": args.model,
                    "class_weight": class_weight,
                    "tuned": args.tuned,
                    "min_recall": args.min_recall,
                    "cost_fn": args.cost_fn,
                    "cost_fp": args.cost_fp,
                    "val_metrics": best,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        THRESHOLD_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        curve.round(4).to_csv(THRESHOLD_CURVE_PATH, index=False)
        print(f"\nĐã lưu model: {MODEL_PATH}")
        print(f"Đã lưu threshold: {THRESHOLD_PATH} (={chosen['threshold']:.2f})")
        print(f"Đã lưu threshold curve: {THRESHOLD_CURVE_PATH}")

    if not args.no_mlflow and tracking.track_run(
        f"final-{args.model}",
        params={
            "model": args.model,
            "class_weight": class_weight,
            "tuned": args.tuned,
            "strategy": args.strategy,
            "threshold": chosen["threshold"],
            "min_recall": args.min_recall,
            "cost_fn": args.cost_fn,
            "cost_fp": args.cost_fp,
        },
        metrics=metrics,
        artifacts=[MODEL_PATH, THRESHOLD_PATH, THRESHOLD_CURVE_PATH],
    ):
        print("Đã log lần chạy vào MLflow (mlruns/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
