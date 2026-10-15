"""FastAPI app cung cấp model churn qua REST API.

Endpoints:
- ``GET /health`` — kiểm tra dịch vụ và trạng thái model.
- ``POST /predict`` — dự đoán churn cho một khách hàng.

Chạy::

    uvicorn app.main:app --reload
    # Swagger UI: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.models.predict import get_default_threshold, get_predictor

DISCLAIMER = (
    "Xác suất churn là ước lượng của mô hình, không phải quyết định kinh doanh. "
    "Cần cân nhắc chi phí bỏ sót và báo động sai trước khi hành động."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp model khi khởi động; nếu lỗi thì để /health báo trạng thái."""
    try:
        get_predictor()
        app.state.model_ready = True
        app.state.model_error = None
    except Exception as exc:  # noqa: BLE001 - muốn app vẫn khởi động để /health phản hồi
        app.state.model_ready = False
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Dự đoán khả năng khách hàng rời bỏ dịch vụ (Telco Churn).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Kiểm tra dịch vụ còn sống và model đã sẵn sàng chưa."""
    return HealthResponse(status="ok", model_loaded=bool(getattr(app.state, "model_ready", False)))


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    features: CustomerFeatures,
    threshold: float | None = Query(
        None, ge=0.0, le=1.0, description="Ngưỡng quyết định (mặc định: threshold đã chọn)"
    ),
) -> PredictionResponse:
    """Dự đoán xác suất và nhãn churn cho một khách hàng."""
    if not getattr(app.state, "model_ready", False):
        detail = getattr(app.state, "model_error", None) or "Model chưa sẵn sàng."
        raise HTTPException(status_code=503, detail=detail)

    if threshold is None:
        threshold = get_default_threshold()

    predictor = get_predictor()
    X = pd.DataFrame([features.model_dump()])
    probability = float(predictor.predict_proba(X)[0])

    return PredictionResponse(
        churn=probability >= threshold,
        churn_probability=round(probability, 6),
        threshold=threshold,
    )
