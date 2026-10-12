"""Pydantic schemas cho request/response của API dự đoán churn.

Tên trường trùng khớp với tên cột trong bộ dữ liệu Telco để dễ ánh xạ.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class CustomerFeatures(BaseModel):
    """Đặc trưng của một khách hàng để dự đoán churn."""

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE})

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1, description="0 = không phải người cao tuổi, 1 = có")
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, description="Số tháng gắn bó")
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0, description="Tổng cước tới nay (có thể null)")


class PredictionResponse(BaseModel):
    """Kết quả dự đoán cho một khách hàng."""

    churn: bool
    churn_probability: float = Field(ge=0, le=1)
    threshold: float


class HealthResponse(BaseModel):
    """Trạng thái dịch vụ."""

    status: str
    model_loaded: bool
