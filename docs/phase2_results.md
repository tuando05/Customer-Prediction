# Giai đoạn 2 — Kết quả (M1 → M-H)

Tài liệu tóm tắt kết quả Giai đoạn 2. Số liệu đầy đủ lưu trong `reports/`
(không commit); đây là bản tổng hợp đã commit.

## Tóm tắt thí nghiệm

| Bước | Nội dung | Kết quả chính |
|------|----------|----------------|
| M1 | EDA + feature engineering | `tenure_bucket`, `num_services`, `is_automatic_payment`, `is_month_to_month` |
| M3 | So sánh 4 mô hình bằng 5-fold CV (PR-AUC) | logreg 0.6655 > xgb 0.6391 > rf 0.6256 > dtree 0.3879 |
| M2 | Mất cân bằng (`class_weight`/`scale_pos_weight`) | logreg balanced: Recall 0.554 → **0.832**, PR-AUC gần như không đổi |
| M4 | Tuning (RandomizedSearchCV, scoring=PR-AUC) | xgb tuned CV PR-AUC **0.6675** (nhỉnh hơn logreg) |
| M5 | SHAP | Driver mạnh: `is_month_to_month`, `tenure`, `MonthlyCharges`, `OnlineSecurity/TechSupport = No`, `Contract`, `Fiber optic`, `Electronic check` |
| M-H | Chốt model + threshold + pipeline | XGBoost tuned + balanced, threshold **0.53** |

## Model cuối cùng

- **Kiến trúc:** `FeatureEngineer` → (impute + scale / one-hot) → **XGBoost**
  với tham số đã tune + `scale_pos_weight = neg/pos` (balanced).
- **Threshold:** 0.53 (chọn theo `max_f1` trên validation).
- **Artifacts:** `artifacts/churn_model.joblib`, `artifacts/threshold.json`.

### Metric

| Tập | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-----|--------|---------|-----------|--------|-----|
| val @0.53 | 0.6728 | 0.8544 | 0.564 | 0.781 | 0.655 |
| test @0.53 | 0.6228 | 0.8253 | 0.526 | 0.749 | 0.618 |

So với baseline Giai đoạn 1 (logreg, threshold 0.5): test Recall 0.527 → **0.749**, F1 0.569 → **0.618**.

## Ghi chú & bài học

- **Mất cân bằng là đòn bẩy lớn nhất** cho Recall, gần như miễn phí về PR-AUC.
- **Feature engineering trung tính** với mô hình tuyến tính (PR-AUC gần như không đổi);
  `is_month_to_month` trùng thông tin với `Contract_Month-to-month`, các feature mới
  khác đóng góp nhỏ (theo SHAP).
- **LogReg và XGBoost ngang bằng về PR-AUC**; chọn XGBoost vì CV PR-AUC nhỉnh hơn và
  giải thích được bằng SHAP.
- **Chưa đạt mục tiêu đề ra PR-AUC ≥ 0.72** (đạt ~0.67) — bộ Telco có trần tín hiệu khá rõ;
  cần nêu rõ khi báo cáo.

## Definition of Done

- [x] `build_features.py` có test và dùng trong pipeline.
- [x] Bảng so sánh ≥ 4 model bằng CV, cùng metric/folds.
- [~] Best model đạt ngưỡng mục tiêu — **chưa** với PR-AUC 0.72 (đạt 0.67; đã giải thích).
- [x] Threshold tối ưu chọn trên val, lưu lại, API dùng mặc định.
- [x] Đánh giá cuối trên test + báo cáo.
- [x] README cập nhật.

## Bước tiếp theo (Giai đoạn 3)

Tests mở rộng, MLflow, Docker, GitHub Actions.
