# Giai đoạn 4 — Cải thiện chất lượng mô hình (Kế hoạch chi tiết)

> Trọng tâm: **MLE** (đúng như định hướng dự án). Không thêm hạ tầng mới trừ khi cần.

## 1. Điểm xuất phát

| Hạng mục | Hiện tại |
|----------|----------|
| Model cuối | XGBoost tuned + `scale_pos_weight` (balanced) |
| CV PR-AUC (xgb tuned) | **0.6675** |
| Val PR-AUC / Recall | 0.6728 / 0.781 (threshold 0.53) |
| Test PR-AUC / Recall | 0.6228 / 0.749 (threshold 0.53) |
| So sánh | LogReg+balanced CV PR-AUC 0.6632 (gần bằng) |

Từ SHAP (M5) đã biết:
- `is_month_to_month` **gần như trùng** `Contract_Month-to-month` (tách đôi tầm quan trọng).
- `num_services`, `tenure_bucket_*`, `is_automatic_payment` đóng góp **nhỏ**.
- Nhiều dummy `*_No internet service` ≈ **0**.

## 2. Mục tiêu (nói thẳng & thực tế)

- **Chính:** cải thiện **PR-AUC qua CV** so với mốc 0.6675, đồng thời tăng/giữ Recall ở
  precision ~0.55.
- **Mục tiêu số cụ thể:** CV PR-AUC **≥ 0.68**.
- ⚠️ **Về mốc 0.72 cũ:** gần như **vượt trần tín hiệu** của bộ Telco (nhiều công bố chỉ
  đạt average precision ~0.63–0.68). Giai đoạn này **thay mốc 0.72 bằng 0.68** và ghi rõ
  lý do, thay vì cố ép bằng cách dò quá nhiều trên validation (gây overfit).
- **Trung thực:** test chỉ đánh giá **một lần** sau khi đã chốt model + threshold.

## 3. Nhóm công việc

> **Kết quả P4-1 (feature selection):** khác biệt giữa các bộ đặc trưng **không đáng kể**
> (CV PR-AUC: `no_dup` 0.6677, `no_internet_dummies` 0.6670, `full` 0.6660, `lean` 0.6637;
> std ≈ 0.022 → nằm trong nhiễu). Kết luận: các đặc trưng dư thừa **không gây hại** cho
> PR-AUC; `no_dup` (bỏ `is_month_to_month`) nhỉnh hơn **rất nhẹ** và gọn hơn.
>
> **Kết quả P4-2 (resampling):** SMOTE **không giúp** PR-AUC (CV: `none` 0.6661,
> `balanced` 0.6660, `smote_tomek` 0.6639, `smote` 0.6620). `class_weight`/`scale_pos_weight`
> vẫn tốt nhất khi xét cả PR-AUC lẫn Recall. → **giữ `balanced`**, không dùng SMOTE.

### A. Feature selection (SHAP + RFE)
- Bỏ đặc trưng dư thừa/≈0: `is_month_to_month` (trùng `Contract`), dummy `*_No internet service`.
- So sánh **3 bộ đặc trưng**: `full` (hiện tại) / `lean` (bỏ dư thừa) / `shap_top_k`.
- Đo bằng CV PR-AUC cùng folds, cùng model → chọn bộ tốt nhất.

### B. Mất cân bằng nâng cao (resampling)
- So sánh `class_weight/scale_pos_weight` (đang dùng) vs **SMOTE** / **SMOTE-Tomek**.
- Vì PR-AUC đo trên phân phối thật, resampling thường **không tăng** PR-AUC — kiểm chứng
  và ghi lại kết quả trung thực (kể cả nếu thua).

### C. Tuning nâng cao (Optuna)
- **Optuna TPE**, nhiều trials hơn RandomizedSearchCV (vd 60–100), `scoring='average_precision'`.
- Cùng CV folds để so sánh công bằng với M4.

### D. Calibration xác suất
- `CalibratedClassifierCV` (sigmoid/isotonic) trên model tốt nhất.
- Mục tiêu: xác suất **đáng tin** (Brier/đường cong calibration) → chọn threshold hợp lý hơn,
  không nhất thiết tăng PR-AUC (đo cả hai).

### E. Chốt model + đánh giá
- Chọn model cuối theo **CV PR-AUC** (không dò trên test).
- Chọn threshold trên val, đánh giá test **một lần**, cập nhật `src/pipeline.py` + README + báo cáo.

## 4. Milestones

| # | Milestone | Deliverable | Commit |
|---|-----------|-------------|--------|
| P4-1 ✅ | Feature selection | `src/features/select.py` (`DropFeatures`) + `src/models/feature_select.py` + bảng so sánh | có |
| P4-2 ✅ | Resampling | `src/models/resample.py` + so sánh SMOTE vs class weight | có |
| P4-3 | Optuna tuning | `src/models/tune_optuna.py` + `reports/optuna_results.json` | có |
| P4-4 | Calibration | `src/models/calibrate.py` + đường cong calibration | có |
| P4-5 | Chốt & báo cáo | cập nhật pipeline/model cuối + `docs/phase4_results.md` | có |

## 5. Definition of Done

- [x] Có bảng so sánh công bằng (CV, cùng folds) cho các biến thể.
- [ ] Model cuối có **CV PR-AUC tốt hơn hoặc bằng** 0.6675 (kèm khoảng nhiễu).
- [ ] Threshold chọn trên val; test đánh giá **một lần**.
- [ ] Pipeline/API/README cập nhật; tests xanh; CI xanh.
- [ ] Ghi rõ **những gì không cải thiện** (nếu có) — không tô hồng kết quả.

## 6. Rủi ro & lưu ý

- **Trần tín hiệu:** cải thiện có thể rất nhỏ (vài phần nghìn) hoặc không có → cần báo cáo thật.
- **Overfit validation:** dò nhiều bộ đặc trưng/params trên val rồi chọn theo val là không sạch;
  vì vậy **chọn theo CV**, val chỉ để chốt threshold.
- **Thêm phụ thuộc:** `imbalanced-learn`, `optuna` chỉ vào `requirements.txt` (train),
  **không** vào `requirements-serve.txt` (image serving).
- **SMOTE + calibration** có thể xung đột (calibrate trên dữ liệu đã oversample là sai) →
  thứ tự đúng: resample chỉ trong bước train của fold.
- **Thời gian:** Optuna nhiều trials chạy lâu; giới hạn bằng timeout/n_trials hợp lý.

## 7. Quyết định cần chốt

1. **Thư viện:** cho phép thêm `imbalanced-learn` (SMOTE) và `optuna` không, hay chỉ dùng
   scikit-learn (feature selection + calibration)?
2. **Mốc mục tiêu:** chấp nhận hạ xuống **0.68** (thực tế) hay giữ **0.72** (rủi ro cao)?
3. **Calibration:** có làm không (thêm bước P4-4)?
