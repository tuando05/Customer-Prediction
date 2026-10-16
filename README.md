# Customer Churn Prediction

![CI](https://github.com/tuando05/Customer-Prediction/actions/workflows/ci.yml/badge.svg)

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
│   ├── tracking.py             # theo dõi thí nghiệm với MLflow
│   ├── features/
│   │   ├── preprocess.py       # làm sạch + pipeline tiền xử lý
│   │   ├── split.py            # chia train/val/test (stratified)
│   │   └── build_features.py   # feature engineering (Giai đoạn 2)
│   ├── models/
│   │   ├── train.py            # huấn luyện
│   │   ├── evaluate.py         # đánh giá
│   │   ├── registry.py         # đăng ký các mô hình
│   │   ├── compare.py          # so sánh mô hình bằng CV
│   │   ├── imbalance.py        # thí nghiệm mất cân bằng lớp
│   │   ├── select_threshold.py # chọn threshold quyết định
│   │   ├── explain.py          # phân tích SHAP
│   │   ├── tune.py             # tinh chỉnh siêu tham số
│   │   └── predict.py          # inference
│   └── pipeline.py             # pipeline huấn luyện end-to-end
├── app/
│   ├── main.py               # FastAPI
│   └── schemas.py            # Pydantic schemas
├── tests/
│   ├── data/sample_raw.csv   # mẫu 200 dòng cho CI (đã commit)
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_registry.py
│   ├── test_evaluate.py
│   ├── test_tune.py
│   ├── test_explain.py
│   └── test_tracking.py
├── scripts/
│   └── make_sample_data.py   # sinh mẫu dữ liệu cho test/CI
├── .github/workflows/ci.yml  # CI: lint + build model từ mẫu + tests
├── reports/                  # kết quả đánh giá (không commit)
├── artifacts/                # model đã huấn luyện (không commit)
├── mlruns/                   # artifact MLflow (không commit)
├── mlflow.db                 # metadata MLflow/SQLite (không commit)
├── docs/
│   ├── phase2_plan.md        # kế hoạch chi tiết Giai đoạn 2
│   ├── phase2_results.md     # kết quả Giai đoạn 2
│   ├── phase3_plan.md        # kế hoạch chi tiết Giai đoạn 3
│   └── deployment.md         # hướng dẫn triển khai & vận hành
├── Dockerfile                # image API (model huấn luyện sẵn khi build)
├── .dockerignore
├── requirements.txt          # phụ thuộc đầy đủ (dev/EDA)
├── requirements-serve.txt    # phụ thuộc tối giản cho image
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

# 4. Huấn luyện model (baseline) -> artifacts/churn_model.joblib
python -m src.models.train

# 5. Huấn luyện model cuối + chọn threshold (mặc định XGBoost tuned + balanced)
python -m src.models.select_threshold

# 6. (tùy chọn) Dự đoán nhanh trên file CSV
python -m src.models.predict --input data/processed/X_test.csv --limit 5

# 7. Chạy API  (Swagger UI: http://127.0.0.1:8000/docs)
uvicorn app.main:app --reload

# 8. Chạy tests
pytest -q
```

> Hoặc chạy trọn gói từ dữ liệu thô đến model cuối: `python -m src.pipeline`.

Ví dụ gọi API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{
  \"gender\": \"Female\", \"SeniorCitizen\": 0, \"Partner\": \"Yes\", \"Dependents\": \"No\",
  \"tenure\": 1, \"PhoneService\": \"No\", \"MultipleLines\": \"No phone service\",
  \"InternetService\": \"DSL\", \"OnlineSecurity\": \"No\", \"OnlineBackup\": \"Yes\",
  \"DeviceProtection\": \"No\", \"TechSupport\": \"No\", \"StreamingTV\": \"No\",
  \"StreamingMovies\": \"No\", \"Contract\": \"Month-to-month\", \"PaperlessBilling\": \"Yes\",
  \"PaymentMethod\": \"Electronic check\", \"MonthlyCharges\": 29.85, \"TotalCharges\": 29.85
}"
```

## Theo dõi thí nghiệm (MLflow)

Các lệnh `train` / `compare` / `tune` / `select_threshold` tự động log
params, metrics và artifacts vào MLflow. Backend là **SQLite cục bộ**
(`mlflow.db`), artifacts nằm trong `mlruns/` — không cần server.

```bash
# Xem giao diện MLflow  ->  http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- Tên run: `train-<model>`, `compare-<model>`, `tune-<model>` / `tune-summary`, `final-<model>`.
- Tắt theo dõi: thêm cờ `--no-mlflow` hoặc đặt biến môi trường `CHURN_MLFLOW=0`.
- Đổi backend/UI khác: đặt biến môi trường `MLFLOW_TRACKING_URI`.

## Docker

Image **tự huấn luyện model khi build** (dùng `requirements-serve.txt` để nhẹ), nên
cần `data/raw/*.csv` trong thư mục dự án trước khi build.

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
# -> http://127.0.0.1:8000/health  |  http://127.0.0.1:8000/docs
```

## Kiểm thử & CI

```bash
pytest -q         # 35 test (model, API, features, registry, evaluate, tune, explain, tracking)
ruff check .      # lint
```

CI (`.github/workflows/ci.yml`) chạy lint + tests trên mỗi lần push/PR. Vì `data/`
bị gitignore, CI **dựng một model nhỏ từ mẫu đã commit** `tests/data/sample_raw.csv`
để bộ test chạy thật, không bị skip.

## Lộ trình

1. **Giai đoạn 1 — MVP:** dữ liệu thô → Logistic Regression baseline → FastAPI (`/health`, `/predict`).
2. **Giai đoạn 2 — Cải thiện mô hình:** EDA sâu, xử lý mất cân bằng, feature engineering,
   so sánh Decision Tree / Random Forest / XGBoost, tuning, SHAP, chọn threshold.
3. **Giai đoạn 3 — Kiểm thử & đóng gói:** tests, MLflow, Docker, GitHub Actions.

> Trạng thái hiện tại: **Giai đoạn 3 đang thực hiện** — đã xong MLflow tracking (P3-1).
> Best model (Giai đoạn 2) = **XGBoost tuned + `scale_pos_weight` balanced + threshold 0.53**
> — val PR-AUC 0.673 / Recall 0.78, test PR-AUC 0.623 / Recall 0.75.

