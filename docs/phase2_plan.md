# Giai đoạn 2 — Kế hoạch cải thiện mô hình (Chi tiết)

> Trọng tâm của giai đoạn này là **Machine Learning Engineering**: nâng chất lượng
> dự đoán churn. Vẫn giữ nguyên tắc: **chỉ fit trên train, chọn model/threshold trên
> validation, test chỉ dùng một lần ở cuối.**

## 1. Điểm xuất phát (baseline Giai đoạn 1)

| Tập | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-----|---------|--------|-----------|--------|-----|
| train | 0.8487 | 0.6686 | 0.6592 | 0.5522 | 0.6010 |
| val   | 0.8600 | 0.6780 | 0.6972 | 0.5909 | 0.6397 |
| test  | 0.8326 | 0.6224 | 0.6195 | 0.5267 | 0.5694 |

**Vấn đề chính:** Recall còn thấp (~0.59) → bỏ sót nhiều khách churn. Đây là mục tiêu số 1.

## 2. Mục tiêu & tiêu chí đo lường

Đề xuất mục tiêu trên tập **validation** (sẽ chốt lại sau khi rõ mục tiêu kinh doanh):

| Metric | Baseline (val) | Mục tiêu |
|--------|----------------|----------|
| PR-AUC | 0.678 | ≥ 0.72 |
| Recall @ threshold tối ưu | 0.591 | ≥ 0.75 |
| F1 @ threshold tối ưu | 0.640 | ≥ 0.66 |
| ROC-AUC | 0.860 | ≥ 0.86 (không giảm) |

Quy ước: **PR-AUC là metric chính** để xếp hạng model (mất cân bằng lớp); Recall/F1
dùng để chọn threshold.

## 3. Các nhóm công việc

### Nhóm A — EDA (phân tích dữ liệu)
- `notebooks/01_eda.ipynb`:
  - Phân phối biến, tỷ lệ churn theo từng nhóm (`Contract`, `tenure`, `InternetService`...).
  - Ma trận tương quan; kiểm tra đa cộng tuyến (`TotalCharges` vs `tenure` — tương quan 0.83).
  - Phát hiện outlier, giá trị hiếm.
- **Kết quả:** danh sách insight + ứng viên đặc trưng mới, lưu tóm tắt vào `reports/`.

### Nhóm B — Feature engineering
- Điền vào `src/features/build_features.py` (đang là placeholder):
  - `num_services`: số dịch vụ đang dùng (đếm các cột `Yes`).
  - `avg_monthly_charge = TotalCharges / max(tenure, 1)`.
  - Nhóm `tenure` (0–12, 13–24, 25–48, 49–72) và nhóm `MonthlyCharges`.
  - Gộp `PaymentMethod` thành `automatic` vs `manual`.
  - (Cân nhắc) tương tác `Contract` × `InternetService`.
- Tích hợp dưới dạng một bước trong pipeline (tránh rò rỉ: chỉ tính từ dữ liệu train).
- **Kết quả:** `build_features.py` + test đơn vị.

### Nhóm C — Xử lý mất cân bằng lớp
- Thử và so sánh: `class_weight="balanced"`, **SMOTE**, undersampling.
- SMOTE **bắt buộc nằm trong pipeline** (imblearn `Pipeline`) để chỉ oversample trên train fold.
- **Kết quả:** bảng so sánh ảnh hưởng lên Recall/PR-AUC.
- **Phụ thuộc:** `imbalanced-learn`.

### Nhóm D — So sánh mô hình (CV)
- Model: LogisticRegression (baseline), DecisionTree, RandomForest, XGBoost.
- Đánh giá bằng `StratifiedKFold(n_splits=5)` trên **train**, cùng bộ metric.
- Mở rộng `src/models/train.py` để nhận `--model {logreg,dtree,rf,xgb}`.
- **Kết quả:** `reports/model_comparison.csv` + bảng xếp hạng.

### Nhóm E — Tinh chỉnh siêu tham số
- `src/models/tune.py` (placeholder): `RandomizedSearchCV` (hoặc Optuna) + `StratifiedKFold`,
  scoring = `average_precision`.
- Không gian tham số cho RandomForest và XGBoost (giới hạn số tổ hợp để chạy nhanh).
- **Kết quả:** best params + model tốt nhất lưu `artifacts/`.
- **Phụ thuộc:** `optuna` (tuỳ chọn).

### Nhóm F — Giải thích mô hình
- SHAP: `summary_plot`, `bar_plot` cho model tốt nhất.
- Rút ra top đặc trưng; cân nhắc loại đặc trưng dư thừa.
- **Kết quả:** hình/bảng trong `reports/` + kết luận.

### Nhóm G — Chọn threshold theo mục tiêu kinh doanh
- Thêm vào `src/models/evaluate.py`:
  - `threshold_curve()` và `pick_threshold()` theo tiêu chí: tối đa F1, hoặc Recall mục tiêu,
    hoặc tối thiểu hóa chi phí `C_FN * FN + C_FP * FP`.
- Lưu threshold tối ưu vào JSON/artifacts để API dùng làm mặc định.
- **Kết quả:** threshold được chọn + đường Precision/Recall theo threshold.

### Nhóm H — Đánh giá cuối & tích hợp
- Đánh giá best model trên **test đúng 1 lần**.
- `src/pipeline.py`: chạy end-to-end (clean → split → features → train → evaluate → save).
- Cập nhật `src/models/predict.py` + `app/main.py` dùng model & threshold mới.
- **Kết quả:** MVP được nâng cấp, API trả kết quả tốt hơn.

## 4. Thứ tự thực hiện (milestones)

| # | Milestone | Deliverable | Commit |
|---|-----------|-------------|--------|
| M1 | EDA + khung feature engineering | `01_eda.ipynb`, `build_features.py` + test | có |
| M2 | Thí nghiệm mất cân bằng | bảng so sánh imbalance | có |
| M3 | So sánh model bằng CV | `model_comparison.csv` | có |
| M4 | Tuning siêu tham số | best params + model | có |
| M5 | SHAP + chọn đặc trưng | báo cáo SHAP | có |
| M6 | Chọn threshold + đánh giá test + tích hợp API | model cuối + threshold | có |

Mỗi milestone là một commit riêng, có số liệu trước/sau.

## 5. Tiêu chí hoàn thành (Definition of Done)

- [ ] `build_features.py` có test và được dùng trong pipeline.
- [ ] Bảng so sánh ≥ 4 model bằng CV, cùng bộ metric và cùng folds.
- [ ] Best model đạt ngưỡng mục tiêu ở mục 2 (hoặc có giải thích nếu không đạt).
- [ ] Threshold tối ưu được chọn trên val, lưu lại và API dùng mặc định.
- [ ] Đánh giá cuối trên test + báo cáo trong `reports/`.
- [ ] README cập nhật kết quả và cách chạy.

## 6. Rủi ro & lưu ý

- **Rò rỉ dữ liệu:** feature engineering và SMOTE phải nằm trong pipeline, chỉ fit trên train.
- **Overfit khi tuning:** dùng CV + giới hạn không gian tham số; kiểm tra chênh lệch train/val.
- **Metric:** ưu tiên PR-AUC/Recall vì mất cân bằng; không chỉ nhìn accuracy.
- **Đa cộng tuyến:** `TotalCharges` ≈ `tenure × MonthlyCharges` — cân nhắc khi chọn đặc trưng.
- **Chi phí tính toán:** XGBoost + tuning có thể chậm; giới hạn `n_iter`/`n_jobs`.
- **Tái lập:** cố định `random_state`, cùng seed cho mọi so sánh.

## 7. Quyết định cần chốt trước khi bắt đầu

1. **Metric ưu tiên:** PR-AUC tổng hợp, hay Recall ở mức cố định (phụ thuộc chi phí FN vs FP)?
2. **Thư viện bổ sung:** thêm `imbalanced-learn` (SMOTE) và `optuna` (tuning) không?
3. **MLflow:** dùng ngay ở Giai đoạn 2 hay để Giai đoạn 3?
