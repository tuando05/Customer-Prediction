# Giai đoạn 4 — Kết quả (P4-1 → P4-5)

Báo cáo tổng hợp. Số liệu đầy đủ ở `reports/` (không commit); đây là bản tổng hợp.

## Tóm tắt thí nghiệm

| Bước | Nội dung | Kết quả chính | Kết luận |
|------|----------|----------------|----------|
| P4-1 | Feature selection (SHAP) | CV PR-AUC: `no_dup` 0.6677, `no_internet_dummies` 0.6670, `full` 0.6660, `lean` 0.6637 (std ≈ 0.022) | Khác biệt **trong nhiễu** → giữ `full` |
| P4-2 | Resampling (SMOTE) | `none` 0.6661, `balanced` 0.6660, `smote_tomek` 0.6639, `smote` 0.6620 | SMOTE **không giúp** → giữ `balanced` |
| P4-3 | Optuna (TPE) | CV PR-AUC **0.6675 → 0.6768** (+0.009), val PR-AUC 0.6820 | Cải thiện **thật nhưng nhỏ** |
| P4-4 | Calibration | Brier: `none` 0.1294, `sigmoid` 0.1301, `isotonic` 0.1304 | **Không cải thiện** → giữ raw |
| P4-5 | Chốt model + test | threshold 0.35 (max_f1 trên val) | Xem dưới |

## Model cuối cùng

- **XGBoost (Optuna-tuned)** — `max_depth=3`, `lr≈0.016`, `n_estimators=400`,
  `subsample≈0.69`, `colsample≈0.62`, `min_child_weight=6`, `gamma≈0.78`,
  `scale_pos_weight=1.0` *(KHÔNG cân bằng lớp — Optuna chọn vậy để tối ưu PR-AUC;
  Recall được chỉnh bằng threshold)*.
- **Threshold:** 0.35 (`max_f1` trên validation).
- Không calibration, không resampling, giữ toàn bộ đặc trưng.

### Metric

| Tập | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-----|--------|---------|-----------|--------|-----|
| val @0.35 | 0.6820 | 0.8604 | 0.596 | 0.749 | 0.664 |
| **test @0.35** | **0.6423** | **0.8374** | **0.553** | 0.709 | **0.621** |

### So với mốc cuối Giai đoạn 2/3 (XGBoost tuned cũ, threshold 0.53)

| Metric (test) | Cũ | Mới | Δ |
|---------------|-----|-----|---|
| PR-AUC | 0.6228 | **0.6423** | **+0.0195** |
| ROC-AUC | 0.8253 | **0.8374** | **+0.0121** |
| Precision | 0.5263 | **0.5532** | +0.0269 |
| Recall | 0.7487 | 0.7086 | −0.0401 |
| F1 | 0.6181 | **0.6213** | +0.0032 |

→ **PR-AUC / ROC-AUC / Precision / F1 đều tăng**; Recall giảm nhẹ (đánh đổi qua threshold,
có thể nâng lại bằng `--strategy target_recall` nếu ưu tiên bắt khách churn).

## Mục tiêu

- Mốc đặt ra ở Giai đoạn 4: **CV PR-AUC ≥ 0.68**.
- Đạt được: **CV 0.6768** (gần đạt), **val 0.6820** (đạt). → Nói thẳng là **gần đạt**, không phóng đại.

## Bài học quan trọng

- **Bug đã phát hiện & sửa:** `_make_xgb` trước đây **bỏ qua `min_child_weight` và `gamma`**
  → mọi báo cáo từ Giai đoạn 2 coi như giả định những tham số này bằng mặc định; model cuối
  không khớp cấu hình mà CV tối ưu. Nay đã truyền đủ (+`reg_alpha`/`reg_lambda`) và có test.
- **Feature engineering của Giai đoạn 2 gần như trung tính** với PR-AUC (nhiều đặc trưng dư
  thừa), nhất quán với SHAP.
- **Resampling và calibration không phải "công thức thắng"**: kiểm chứng trên dữ liệu này
  đều không cải thiện → giữ pipeline đơn giản.
- Cải thiện PR-AUC trên bộ Telco **rất khó**: ~+0.01 CV cho thấy đang tiệm cận trần tín hiệu.

## Definition of Done

- [x] Bảng so sánh công bằng (CV, cùng folds) cho các biến thể.
- [x] Model cuối có CV PR-AUC tốt hơn mốc 0.6675 (đạt 0.6768).
- [x] Threshold chọn trên val; test đánh giá **một lần**.
- [x] Pipeline/API/README cập nhật; tests xanh; CI xanh.
- [x] Ghi rõ những gì **không** cải thiện (SMOTE, calibration, feature selection).

## Giới hạn & bước tiếp theo

- **Lạc quan hóa do tối ưu lặp:** CV được dùng nhiều lần để chọn (300 trials) → con số CV có
  thể lạc quan hơn thực tế; test chỉ đo một lần và thấp hơn CV (0.6423) — bình thường.
- Nếu ưu tiên Recall: `python -m src.models.select_threshold --strategy target_recall --min-recall 0.8`.
- Hướng khác ít triển vọng: thêm dữ liệu ngoài, hoặc bài toán khác (ví dụ dự báo doanh thu).
