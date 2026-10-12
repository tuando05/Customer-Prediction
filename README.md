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
│   ├── config.py               # đường dẫn & hằng số dùng chung
│   ├── features/
│   │   ├── preprocess.py       # làm sạch + pipeline tiền xử lý
│   │   ├── split.py            # chia train/val/test (stratified)
│   │   └── build_features.py   # feature engineering (Giai đoạn 2)
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

## Chạy nhanh (MVP)

```bash
# 1. Kiểm tra chất lượng dữ liệu thô
python -m src.features.validate_data

# 2. Làm sạch dữ liệu -> data/processed/churn_clean.csv
python -m src.features.preprocess

# 3. Chia train/val/test (stratified) -> data/processed/*.csv
python -m src.features.split

# 4. Huấn luyện baseline Logistic Regression -> artifacts/churn_model.joblib
python -m src.models.train
```

## Lộ trình

1. **Giai đoạn 1 — MVP:** dữ liệu thô → Logistic Regression baseline → FastAPI (`/health`, `/predict`).
2. **Giai đoạn 2 — Cải thiện mô hình:** EDA sâu, xử lý mất cân bằng, feature engineering,
   so sánh Decision Tree / Random Forest / XGBoost, tuning, SHAP, chọn threshold.
3. **Giai đoạn 3 — Kiểm thử & đóng gói:** tests, MLflow, Docker, GitHub Actions.

> Trạng thái hiện tại: đã có kiểm tra dữ liệu, tiền xử lý, chia tập và baseline Logistic Regression.

