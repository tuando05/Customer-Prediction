# Hướng dẫn triển khai & vận hành

Tài liệu này đủ để người khác dựng lại dự án và chạy API từ đầu.

## 1. Yêu cầu

- **Python 3.12** (dự án phát triển trên 3.12.10; tối thiểu 3.10).
- Dữ liệu thô **Telco Customer Churn** đặt tại
  `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (file này không được commit).
- (Tùy chọn) **Docker** nếu muốn chạy bằng container.

## 2. Cài đặt & chạy local

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 2. Cài phụ thuộc
pip install -r requirements.txt

# 3. Chạy trọn pipeline: làm sạch -> chia tập -> huấn luyện model cuối + threshold
python -m src.pipeline

# 4. Chạy API
uvicorn app.main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

Kết quả pipeline:

- `artifacts/churn_model.joblib` — model cuối (feature engineering + tiền xử lý + XGBoost).
- `artifacts/threshold.json` — threshold quyết định (mặc định API dùng giá trị này).

## 3. Gọi API

```bash
# Kiểm tra trạng thái (model_loaded = true khi đã có model)
curl http://127.0.0.1:8000/health

# Dự đoán (dùng threshold mặc định trong artifacts/threshold.json)
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{
  \"gender\": \"Female\", \"SeniorCitizen\": 0, \"Partner\": \"Yes\", \"Dependents\": \"No\",
  \"tenure\": 1, \"PhoneService\": \"No\", \"MultipleLines\": \"No phone service\",
  \"InternetService\": \"DSL\", \"OnlineSecurity\": \"No\", \"OnlineBackup\": \"Yes\",
  \"DeviceProtection\": \"No\", \"TechSupport\": \"No\", \"StreamingTV\": \"No\",
  \"StreamingMovies\": \"No\", \"Contract\": \"Month-to-month\", \"PaperlessBilling\": \"Yes\",
  \"PaymentMethod\": \"Electronic check\", \"MonthlyCharges\": 29.85, \"TotalCharges\": 29.85
}"

# Ghi đè threshold cho riêng request
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.7" -H "Content-Type: application/json" -d "{ ... }"
```

Dự đoán hàng loạt từ CSV:

```bash
python -m src.models.predict --input data/processed/X_test.csv --limit 10
```

## 4. Theo dõi thí nghiệm (MLflow)

Các lệnh `train` / `compare` / `tune` / `select_threshold` tự log vào MLflow
(backend SQLite cục bộ `mlflow.db`, artifact trong `mlruns/`).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# -> http://127.0.0.1:5000
```

## 5. Docker

Image **huấn luyện model sẵn khi build**, dùng `requirements-serve.txt` (nhẹ hơn
`requirements.txt`). Vì cần dữ liệu để train, **phải có** file CSV trong `data/raw/`
trước khi build.

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api

curl http://127.0.0.1:8000/health   # -> {"status":"ok","model_loaded":true}
```

Ghi chú:

- `CHURN_MLFLOW=0` được đặt sẵn trong image (không cần MLflow lúc serving).
- Image có `HEALTHCHECK` gọi `/health` mỗi 30s.
- Trong container, model nằm ở `/app/artifacts/`. Muốn dùng model khác mà không
  build lại: mount volume `-v "$PWD/artifacts:/app/artifacts"`.

## 6. CI (GitHub Actions)

`.github/workflows/ci.yml` chạy khi push `main` hoặc mở PR:

1. Cài `requirements.txt` + `ruff`.
2. `ruff check .`
3. Dựng model nhỏ từ mẫu `tests/data/sample_raw.csv` (vì `data/` bị gitignore).
4. `pytest -q`.

## 7. Biến môi trường

| Biến | Ý nghĩa | Mặc định |
|------|---------|----------|
| `CHURN_MLFLOW` | `0` để tắt log MLflow | `1` (bật) |
| `MLFLOW_TRACKING_URI` | Ghi đè backend MLflow | `sqlite:///mlflow.db` |

## 8. Xử lý sự cố

- **`/health` trả `model_loaded: false`** → chưa có model. Chạy `python -m src.pipeline`
  (hoặc `python -m src.models.select_threshold`) trước.
- **`FileNotFoundError: ... Telco ... csv`** → thiếu dữ liệu thô; tải dataset và đặt
  đúng vào `data/raw/`.
- **Lỗi `UnicodeEncodeError` khi in tiếng Việt trên Windows** → các module đã tự cấu
  hình `stdout` sang UTF-8; nếu chạy tùy biến, đặt `PYTHONUTF8=1`.
- **`docker build` lỗi ở bước huấn luyện** → kiểm tra `data/raw/*.csv` có trong
  thư mục dự án và **không** bị `.dockerignore` loại bỏ.
- **Kết quả khác nhau giữa các lần chạy** → mọi bước đã seed `RANDOM_STATE=42`; kiểm
  tra lại phiên bản thư viện (khác phiên bản scikit-learn/xgboost có thể lệch nhỏ).
