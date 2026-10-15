"""Tests cho FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

EXAMPLE = {
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


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict(client):
    if not client.get("/health").json()["model_loaded"]:
        pytest.skip("Model chưa được huấn luyện.")
    resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["churn"], bool)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert 0.0 <= body["threshold"] <= 1.0
    assert body["churn"] == (body["churn_probability"] >= body["threshold"])


def test_predict_voi_threshold(client):
    if not client.get("/health").json()["model_loaded"]:
        pytest.skip("Model chưa được huấn luyện.")
    resp = client.post("/predict?threshold=0.1", json=EXAMPLE)
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 0.1


def test_predict_thieu_truong(client):
    resp = client.post("/predict", json={"gender": "Female"})
    assert resp.status_code == 422
