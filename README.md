# Customer Churn Prediction

Dự đoán khách hàng rời bỏ dịch vụ (churn) trên bộ dữ liệu **Telco Customer Churn**,
tập trung vào **Machine Learning Engineering (MLE)** theo hướng **MVP-first**.

## Mục tiêu

- **MLE (trọng tâm):** EDA, feature engineering, huấn luyện – đánh giá – so sánh mô hình,
  tinh chỉnh siêu tham số, xử lý mất cân bằng lớp, chọn threshold tối ưu.
- **Model serving:** đưa mô hình tốt nhất lên FastAPI.
- **MLOps (hỗ trợ, sau MVP):** tracking experiment và đóng gói ở mức tối thiểu.

## Cấu trúc dự án

```text
.
├── data/
│   ├── raw/                 # dữ liệu gốc (không commit)
│   └── processed/           # dữ liệu đã xử lý (không commit)
├── notebooks/
│   ├── 01_eda.ipynb         # khám phá dữ liệu
│   └── 02_modeling.ipynb    # thử nghiệm & so sánh mô hình
├── src/
│   ├── features/
│   │   └── build_features.py   # preprocessing + feature engineering
│   ├── models/
│   │   ├── train.py            # huấn luyện
│   │   ├── evaluate.py         # đánh giá
│   │   ├── tune.py             # tinh chỉnh siêu tham số
│   │   └── predict.py          # inference
│   └── pipeline.py             # pipeline huấn luyện end-to-end
├── app/
│   ├── main.py               # FastAPI
│   └── schemas.py            # Pydantic schemas
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── reports/                  # kết quả đánh giá (không commit)
├── artifacts/                # model đã huấn luyện (không commit)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Lộ trình

1. **Giai đoạn 1 — MVP:** dữ liệu thô → Logistic Regression baseline → FastAPI (`/health`, `/predict`).
2. **Giai đoạn 2 — Cải thiện mô hình:** EDA sâu, xử lý mất cân bằng, feature engineering,
   so sánh Decision Tree / Random Forest / XGBoost, tuning, SHAP, chọn threshold.
3. **Giai đoạn 3 — Kiểm thử & đóng gói:** tests, MLflow, Docker, GitHub Actions.

> Trạng thái hiện tại: mới dựng **cấu trúc**, chưa triển khai code.
