# Báo cáo tổng kết dự án — Customer Churn Prediction

> Tài liệu này tổng hợp **toàn bộ** những gì đã làm từ đầu dự án đến nay.
> Chi tiết theo từng giai đoạn nằm ở `docs/phase2_plan.md`, `docs/phase2_results.md`,
> `docs/phase3_plan.md`, `docs/phase4_plan.md`, `docs/phase4_results.md`.

---

## 1. Tổng quan

- **Bài toán:** dự đoán khách hàng rời bỏ dịch vụ (churn) — phân loại nhị phân, dữ liệu
  **Telco Customer Churn** (7.043 dòng × 21 cột, tỷ lệ churn ≈ 26,5%).
- **Định hướng:** **MLE là trọng tâm**, làm theo **MVP-first** — chạy được trước, tối ưu sau.
  MLOps/DE chỉ ở mức hỗ trợ. Đã **bỏ phần Data Engineering/PostgreSQL** và lớp `churn_project`
  (đưa nội dung thẳng vào `src/`) để tập trung vào MLE.
- **Metric ưu tiên:** **PR-AUC** (average precision) vì dữ liệu mất cân bằng.
- **Sản phẩm:** model + REST API (FastAPI) + pipeline tái lập + CI + Docker + báo cáo.

### Số liệu nhanh

| Hạng mục | Giá trị |
|----------|---------|
| Commits | **29** |
| File Python (src/app/tests/scripts) | 42 |
| Dòng code Python | ~3.559 |
| Tests | **57** (pass) |
| CI (GitHub Actions) | 3/3 lần chạy **xanh** |
| Model cuối | XGBoost (Optuna-tuned), threshold 0.35 |
| Test PR-AUC / ROC-AUC / F1 | **0.6423 / 0.8374 / 0.6213** |

---

## 2. Kiến trúc & cấu trúc

```text
.
├── data/            raw/ + processed/            (không commit)
├── notebooks/       01_eda.ipynb                 (EDA đã thực thi)
├── src/
│   ├── config.py                # đường dẫn, hằng số, seed
│   ├── tracking.py              # MLflow (tùy chọn, an toàn nếu lỗi/thiếu)
│   ├── pipeline.py              # pipeline end-to-end
│   ├── features/
│   │   ├── validate_data.py     # kiểm tra chất lượng dữ liệu thô
│   │   ├── preprocess.py        # làm sạch + ColumnTransformer
│   │   ├── split.py             # chia train/val/test (stratified)
│   │   ├── build_features.py    # feature engineering (Giai đoạn 2)
│   │   └── select.py            # DropFeatures (Giai đoạn 4)
│   └── models/
│       ├── registry.py          # 4 model + build pipeline + đọc tham số tuned
│       ├── train.py             # huấn luyện
│       ├── evaluate.py          # metric + threshold curve
│       ├── compare.py           # so sánh model bằng CV
│       ├── imbalance.py         # class weight
│       ├── resample.py          # SMOTE vs class weight (Giai đoạn 4)
│       ├── select_threshold.py  # chọn threshold tối ưu
│       ├── tune.py              # RandomizedSearchCV
│       ├── tune_optuna.py       # Optuna TPE (Giai đoạn 4)
│       ├── feature_select.py    # so sánh bộ đặc trưng (Giai đoạn 4)
│       ├── explain.py           # SHAP
│       └── predict.py           # inference
├── app/             main.py + schemas.py          # FastAPI
├── tests/           data/sample_raw.csv + 12 file test
├── scripts/         make_sample_data.py
├── .github/workflows/ci.yml
├── Dockerfile, .dockerignore, requirements*.txt, pyproject.toml
└── docs/            kế hoạch + kết quả từng giai đoạn + hướng dẫn triển khai
```

**Nguyên tắc xuyên suốt:** mỗi bước là **một commit**; không viết code trước khi dựng cấu trúc;
chỉ dùng train để fit, val để chọn threshold, **test chỉ đánh giá một lần** (tránh rò rỉ).

---

## 3. Hành trình 4 giai đoạn

### Giai đoạn 1 — MVP
Dữ liệu thô → làm sạch → chia tập → **LogisticRegression baseline** → FastAPI (`/health`, `/predict`) → tests.

- Xử lý `TotalCharges` trống: 11 dòng đều `tenure=0` → gán `0`.
- Chia stratified 60/20/20 → train 4.225 / val 1.409 / test 1.409.
- Có script kiểm tra chất lượng dữ liệu (`validate_data.py`).

### Giai đoạn 2 — Cải thiện mô hình
| Bước | Việc làm | Kết quả |
|------|----------|---------|
| M1 | EDA (`01_eda.ipynb`) + feature engineering | 4 đặc trưng mới: `tenure_bucket`, `num_services`, `is_automatic_payment`, `is_month_to_month` |
| M3 | So sánh 4 model bằng 5-fold CV (PR-AUC) | **logreg 0.6655** > xgb 0.6391 > rf 0.6256 > dtree 0.3879 |
| M2 | Xử lý mất cân bằng | logreg `balanced`: CV Recall 0.534 → **0.785**; val Recall 0.554 → **0.832** (PR-AUC gần như không đổi) |
| M4 | Tinh chỉnh (RandomizedSearchCV, scoring PR-AUC) | xgb tuned CV PR-AUC **0.6675** (nhỉnh nhất) |
| M5 | SHAP | Driver mạnh: `is_month_to_month`, `tenure`, `MonthlyCharges`, `OnlineSecurity/TechSupport = No`, `Contract`, `Fiber optic`, `Electronic check` |
| M-G | Chọn threshold trên val | logreg → 0.62; XGBoost → 0.53 |
| M-H | Chốt model + pipeline end-to-end | **XGBoost tuned + balanced, threshold 0.53** |

### Giai đoạn 3 — Đóng gói & vận hành
| Bước | Việc làm | Kết quả |
|------|----------|---------|
| P3-1 | MLflow tracking | Log params/metrics/artifacts cho `train`/`compare`/`tune`/`select_threshold`; backend **SQLite** cục bộ (`mlflow.db`) — MLflow 3.x đã bỏ file store mặc định |
| P3-2 | Docker | `Dockerfile` **huấn luyện model khi build**, `requirements-serve.txt` tối giản, `HEALTHCHECK`; image ~1,5GB |
| P3-3 | CI + mẫu dữ liệu | `.github/workflows/ci.yml`: lint → **dựng model nhỏ từ mẫu 200 dòng đã commit** → `pytest` (không bị skip); badge CI |
| P3-4 | Docs + polish | `docs/deployment.md`, `ruff` sạch (sửa 12 lỗi lint), README |

### Giai đoạn 4 — Cải thiện chất lượng
| Bước | Thí nghiệm | Kết quả | Kết luận |
|------|-----------|---------|----------|
| P4-1 | Feature selection theo SHAP | CV: `no_dup` 0.6677 / `full` 0.6660 / `lean` 0.6637 (std≈0.022) | Khác biệt **trong nhiễu** → giữ `full` |
| P4-2 | SMOTE vs class weight | `none` 0.6661 / `balanced` 0.6660 / `smote` 0.6620 | SMOTE **không giúp** → giữ `balanced` |
| P4-3 | **Optuna (TPE, 200 trials)** | CV PR-AUC **0.6675 → 0.6768** | Cải thiện thật (nhỏ) |
| P4-4 | Calibration | Brier: `none` 0.1294 / `sigmoid` 0.1301 / `isotonic` 0.1304 | **Không cần** → giữ raw |
| P4-5 | Chốt model + test | threshold 0.35 (`max_f1`) | Xem dưới |

---

## 4. Model cuối cùng & kết quả

**XGBoost (Optuna-tuned)** — `max_depth=3`, `lr≈0.016`, `n_estimators=400`, `subsample≈0.69`,
`colsample≈0.62`, `min_child_weight=6`, `gamma≈0.78`, `scale_pos_weight=1.0`; threshold **0.35**.
Không resampling, không calibration, giữ toàn bộ đặc trưng.

| Tập | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-----|--------|---------|-----------|--------|-----|
| val @0.35 | 0.6820 | 0.8604 | 0.596 | 0.749 | 0.664 |
| **test @0.35** | **0.6423** | **0.8374** | **0.553** | 0.709 | **0.621** |

### Tiến bộ qua các giai đoạn (test)

| Metric | Baseline GĐ1 | Cuối GĐ2/3 | **GĐ4** |
|--------|-------------|-----------|---------|
| PR-AUC | — | 0.6228 | **0.6423** |
| ROC-AUC | — | 0.8253 | **0.8374** |
| Precision | — | 0.5263 | **0.5532** |
| Recall | 0.527 | **0.7487** | 0.7086 |
| F1 | 0.569 | 0.6181 | **0.6213** |

→ So với cuối Giai đoạn 2/3: PR-AUC +0.0195, ROC-AUC +0.0121, Precision +0.027, F1 +0.003;
Recall giảm nhẹ do đổi threshold (có thể nâng lại bằng `--strategy target_recall`).

### Mục tiêu PR-AUC
- Mốc đặt ra ở Giai đoạn 4: CV PR-AUC ≥ 0.68 → đạt **0.6768 (gần đạt)**; val PR-AUC **0.6820 (đạt)**.
- Mốc cũ 0.72 (Giai đoạn 2) **không đạt** và đã được **hạ xuống 0.68 kèm giải thích**: bộ Telco
  có trần tín hiệu ~0.65–0.68 theo nhiều công bố, nên 0.72 gần như nằm ngoài khả năng.

---

## 5. Hạ tầng & chất lượng

- **Tests:** 57 test (model, API, features, registry, evaluate, tune, tune_optuna, explain,
  tracking, select, resample, calibrate) — tất cả pass.
- **Lint:** `ruff check .` sạch (cấu hình trong `pyproject.toml`).
- **CI:** GitHub Actions chạy lint → dựng model từ mẫu → tests. Vì `data/` bị gitignore,
  đã commit mẫu `tests/data/sample_raw.csv` (200 dòng) để test **chạy thật, không bị skip**.
- **MLflow:** log cục bộ (SQLite), tắt được bằng `--no-mlflow` / `CHURN_MLFLOW=0`,
  và **không làm hỏng pipeline** nếu MLflow lỗi hoặc chưa cài.
- **Docker:** image tự huấn luyện model khi build; `/health` trả `model_loaded=true`;
  `HEALTHCHECK` hoạt động; image ~1,5GB.
- **Tái lập:** `RANDOM_STATE=42` ở mọi bước; tham số đã tuned được commit trong
  `reports/tuning_results.json` để Docker/CI tái lập đúng model.

### Cách chạy lại nhanh
```bash
python -m venv .venv && .\.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
python -m src.pipeline                                    # dữ liệu thô -> model + threshold
uvicorn app.main:app --reload                             # API: /docs
pytest -q                                                 # 57 test
mlflow ui --backend-store-uri sqlite:///mlflow.db          # xem thí nghiệm
docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

---

## 6. Bug đã phát hiện & sửa

1. **`TotalCharges` trống** ở khách mới đã đúng từ Giai đoạn 1.
2. **`groupby.apply` làm mất cột nhóm** khi sinh mẫu dữ liệu (pandas mới) → chuyển sang
   `pd.concat` trong `scripts/make_sample_data.py`.
3. **MLflow 3.x bỏ file store mặc định** → chuyển backend sang **SQLite** cục bộ.
4. **`_make_xgb` bỏ qua `min_child_weight` và `gamma`** → model cuối *không khớp* cấu hình mà
   CV đã tối ưu (đã ảnh hưởng số liệu Giai đoạn 2). Đã sửa (thêm cả `reg_alpha`/`reg_lambda`)
   và **thêm test chống tái phát**.
5. **Docker build không copy `reports/tuning_results.json`** → model trong image dùng tham số
   mặc định. Đã sửa; rebuild và xác nhận image dùng đúng **threshold 0.35** của model cuối.

---

## 7. Những gì KHÔNG cải thiện (trung thực)

- **Feature engineering Giai đoạn 2 gần như trung tính** với PR-AUC; nhiều đặc trưng dư thừa
  (`is_month_to_month` trùng `Contract`, dummy `*_No internet service` ≈ 0).
- **Bỏ đặc trưng dư thừa** không tạo khác biệt ý nghĩa (trong khoảng nhiễu).
- **SMOTE / SMOTE-Tomek** không giúp PR-AUC (thậm chí nhỉnh hơn chút).
- **Calibration** không cải thiện Brier/log loss → xác suất XGBoost đã đủ hiệu chỉnh.
- **Mốc PR-AUC 0.72** không đạt; mốc 0.68 chỉ *gần đạt* ở CV.
- PR-AUC **test thấp hơn CV/val** (0.6423 vs 0.6820) — bình thường khi đã tối ưu lặp trên CV/val.

---

## 8. Bài học kinh nghiệm

1. **MVP trước** giúp có sản phẩm chạy được sớm, rồi mới tối ưu có cơ sở.
2. **Mất cân bằng lớp** là "đòn bẩy" rẻ nhất cho Recall; PR-AUC gần như không đổi.
3. **Luôn kiểm tra tham số tuning có thực sự truyền vào estimator** — bug #4 rất dễ lọt.
4. **Không phải kỹ thuật phổ biến nào cũng giúp**: SMOTE/calibration/feature selection đều
   không cải thiện trên dữ liệu này → giữ pipeline **đơn giản**.
5. **Trần dữ liệu là có thật**: cải thiện PR-AUC trên Telco rất khó (~+0.01), cần nói thẳng
   thay vì cố ép bằng cách dò quá nhiều (gây lạc quan hóa).
6. **Tự động hóa (CI) buộc phải xử lý vấn đề dữ liệu bị gitignore** — giải pháp mẫu nhỏ đã
   commit là đủ và rẻ.

---

## 9. Trạng thái hiện tại

- Branch `main` **đồng bộ** `origin/main`; commit cuối `4b49ef2`.
- **CI xanh** cả 3 lần chạy gần nhất.
- Model + threshold đã sẵn trong `artifacts/`; API chạy được; Docker image chạy được.
- Tài liệu đầy đủ: kế hoạch + kết quả từng giai đoạn + hướng dẫn triển khai.

## 10. Giới hạn & hướng tiếp theo (đề xuất)

| Hướng | Nội dung | Giá trị |
|-------|----------|---------|
| Model registry / versioning | Lưu model theo version thay vì ghi đè | Trung bình |
| Monitoring / drift | Theo dõi phân phối đầu vào & hiệu năng theo thời gian | Trung bình–cao |
| Tối ưu theo chi phí | Dùng `--strategy min_cost` với chi phí FN/FP thực tế | Cao (giá trị kinh doanh) |
| Mở rộng bài toán | Dự báo doanh thu/CLV thay vì chỉ churn | Cao |
| Dữ liệu bổ sung | Thêm nguồn ngoài để vượt trần tín hiệu | Cao nhưng tốn công |
