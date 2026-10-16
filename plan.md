# Customer Churn Prediction — MLE Project Plan (MVP-first)

## 1. Mục tiêu dự án

Xây dựng mô hình dự đoán khách hàng rời bỏ dịch vụ (churn), tập trung vào **Machine Learning Engineering (MLE)** và ưu tiên ra **MVP** sớm nhất có thể.

- **MLE (TRỌNG TÂM):** EDA, feature engineering, huấn luyện – đánh giá – so sánh mô hình, tinh chỉnh siêu tham số, xử lý mất cân bằng lớp, chọn ngưỡng tối ưu.
- **Model Serving:** Đưa mô hình tốt nhất lên FastAPI để dùng ngay.
- **MLOps (hỗ trợ, làm sau khi có MVP):** Theo dõi experiment và đóng gói ở mức tối thiểu.

**Dataset:** Telco Customer Churn.

**Tech stack:**
- ML (chính): Python, pandas, numpy, scikit-learn, XGBoost, matplotlib/seaborn (EDA), SHAP.
- Serving: FastAPI.
- MLOps (phụ, sau MVP): MLflow, Docker, GitHub Actions.

## 2. Cấu trúc dự án

```text
churn-mle/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── config.py
│   ├── tracking.py
│   ├── features/
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── registry.py
│   │   ├── compare.py
│   │   ├── imbalance.py
│   │   ├── select_threshold.py
│   │   ├── explain.py
│   │   ├── tune.py
│   │   └── predict.py
│   └── pipeline.py
├── app/
│   ├── main.py
│   └── schemas.py
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── reports/
├── artifacts/
├── mlruns/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

### Vai trò các thành phần

- `notebooks/`: EDA và thử nghiệm mô hình (nơi bắt đầu mọi công việc ML).
- `src/features/`: Preprocessing và feature engineering pipeline.
- `src/models/`: Huấn luyện, tinh chỉnh, đánh giá và dự đoán.
- `src/pipeline.py`: Pipeline huấn luyện end-to-end.
- `src/tracking.py`: Theo dõi thí nghiệm với MLflow (local, SQLite).
- `app/`: Cung cấp model qua REST API.
- `tests/`: Kiểm tra model và API.
- `artifacts/`: Lưu model đã huấn luyện.
- `reports/`: Lưu kết quả đánh giá.
- `mlruns/`: Artifact của MLflow (không commit).

## 3. Quy trình thực hiện

### Giai đoạn 1 — MVP: Từ dữ liệu thô đến API (làm ngay, tối thiểu)

**Mục tiêu:** Có một mô hình chạy được và phục vụ qua API trong thời gian ngắn nhất.

1. Tải dataset, hiểu cấu trúc và phân phối cơ bản.
2. Xử lý nhanh: missing values, mã hóa biến phân loại, chuẩn hóa số liệu.
3. Chia train/val/test (stratified để giữ tỷ lệ churn).
4. Huấn luyện **Logistic Regression** làm baseline đơn giản.
5. Đánh giá nhanh bằng Precision, Recall, F1, ROC-AUC, PR-AUC.
6. Lưu model + preprocessing pipeline.
7. Dựng FastAPI với `GET /health` và `POST /predict`.

**Kết quả (MVP):** Pipeline hoàn chỉnh từ dữ liệu thô → model → API, có thể dự đoán churn.

### Giai đoạn 2 — Cải thiện mô hình (sau MVP)

> Kế hoạch chi tiết: [`docs/phase2_plan.md`](docs/phase2_plan.md).

**Mục tiêu:** Nâng chất lượng dự đoán bằng EDA sâu hơn và so sánh nhiều mô hình.

1. EDA chi tiết: mối quan hệ đặc trưng ↔ churn, ma trận tương quan.
2. Xử lý mất cân bằng lớp: class weights, SMOTE hoặc undersampling.
3. Feature engineering: tạo đặc trưng mới, loại đặc trưng dư thừa.
4. Huấn luyện và so sánh thêm: **Decision Tree, Random Forest, XGBoost**.
5. Tinh chỉnh siêu tham số bằng GridSearchCV/RandomizedSearchCV (hoặc Optuna).
6. Đánh giá bằng cross-validation, cùng bộ metric để so sánh công bằng.
7. Phân tích feature importance (SHAP) để chọn đặc trưng quan trọng.
8. Chọn threshold tối ưu trên validation theo mục tiêu kinh doanh (Recall vs Precision).
9. Đánh giá cuối cùng trên test set, thay model tốt nhất vào API.

**Kết quả:** Mô hình tốt hơn baseline, được đánh giá kỹ và sẵn sàng thay thế trong API.

### Giai đoạn 3 — Kiểm thử & đóng gói (hỗ trợ, sau MVP)

> Kế hoạch chi tiết: [`docs/phase3_plan.md`](docs/phase3_plan.md).

**Mục tiêu:** Làm dự án dễ chạy lại, kiểm thử và trình diễn.

1. Viết test cho model inference và API.
2. Dùng MLflow để log parameters, metrics và artifacts.
3. Đóng gói API bằng Docker.
4. Tự động chạy tests bằng GitHub Actions.
5. Ghi lại hướng dẫn chạy trong README.

**Kết quả:** Dự án dễ tái sử dụng, kiểm thử và trình diễn.

### Giai đoạn 4 — Cải thiện chất lượng mô hình (MLE, sau cùng)

> Kế hoạch chi tiết: [`docs/phase4_plan.md`](docs/phase4_plan.md).

**Mục tiêu:** Nâng PR-AUC bằng feature selection, resampling, tuning nâng cao và calibration.

1. Chọn lọc đặc trưng theo SHAP (bỏ dư thừa / ≈0).
2. So sánh resampling (SMOTE) với class weights.
3. Tinh chỉnh bằng Optuna (nhiều trials hơn).
4. Calibration xác suất.
5. Chốt model + đánh giá test một lần + báo cáo.

**Kết quả:** Mô hình tốt hơn (đo bằng CV) kèm đánh giá trung thực về giới hạn của dữ liệu.

## 4. Nguyên tắc triển khai

- **MVP trước:** Đưa mô hình + API chạy được nhanh nhất, rồi mới tối ưu. Không đợi hoàn hảo mới ra sản phẩm.
- **Ưu tiên MLE:** Dành phần lớn nỗ lực cho EDA, feature engineering và modeling. MLOps chỉ làm khi cần, sau khi có MVP.
- Mỗi mô hình đánh giá bằng cùng một bộ metric để so sánh công bằng.
- Trong MVP, giữ mọi thứ tối giản; chỉ thêm phức tạp khi nó mang lại giá trị đo được.
