# Giai đoạn 3 — Đóng gói & vận hành (Kế hoạch chi tiết)

> Giai đoạn này **không đổi trọng tâm MLE**. Mục tiêu là biến dự án từ "chạy được
> trên máy" thành "dễ tái lập, kiểm thử tự động và triển khai".

## 1. Điểm xuất phát

Đã có:
- Pipeline end-to-end: `python -m src.pipeline` → model + threshold.
- Best model: **XGBoost tuned + balanced + threshold 0.53** (test PR-AUC 0.623, Recall 0.749).
- FastAPI (`/health`, `/predict`), **29 test** (model, API, features, registry, evaluate, tune, explain).

Chưa có:
- Theo dõi thí nghiệm (experiment tracking).
- Đóng gói container.
- CI tự động.
- Tài liệu triển khai.

## 2. Mục tiêu

- Mọi thí nghiệm được **log lại** (params/metrics/artifacts), so sánh được.
- Dựng lại môi trường và chạy dự án **bằng 1–2 lệnh**.
- Tests chạy **tự động trên CI** mỗi lần push.
- Có hướng dẫn triển khai (local + Docker).

## 3. Các nhóm công việc

### Nhóm A — Experiment tracking (MLflow)
- Thêm `mlflow` vào `requirements.txt`.
- Tích hợp log vào `train.py`, `tune.py`, `compare.py`:
  - params (model + hyperparams), metrics (PR-AUC, ROC-AUC, Recall, F1...), artifacts (model, threshold, plots).
- Lưu local vào `mlruns/` (thêm vào `.gitignore`).
- (Tùy chọn) so sánh run tốt nhất và ghi lại vào `reports/`.

### Nhóm B — Packaging & tái lập
- `pyproject.toml`: bổ sung extras `dev` (mlflow, ruff), cấu hình công cụ.
- Ghim phiên bản phụ thuộc (hoặc `requirements.lock`) để build lặp lại được.
- Đảm bảo seed cố định (`RANDOM_STATE`) ở mọi bước.
- Thêm `Scripts`/lệnh chạy chuẩn trong README.

### Nhóm C — Container hóa (Docker)
- `Dockerfile` (base `python:3.12-slim`), cài deps, copy `src/` + `app/`, chạy `uvicorn`.
- `.dockerignore` (loại `data/`, `artifacts/`, `.venv/`, `notebooks/`).
- `HEALTHCHECK` gọi `/health`.
- (Tùy chọn) `docker-compose.yml` cho tiện chạy.

### Nhóm D — CI (GitHub Actions)
- Workflow `.github/workflows/ci.yml`:
  1. Setup Python 3.12 + cache pip.
  2. Cài deps (+ dev).
  3. Lint (ruff).
  4. `pytest -q`.
- **Vấn đề dữ liệu (quan trọng):** `data/` bị gitignore nên CI không có CSV.
  Cần chọn hướng:
  - **(a)** Commit một **mẫu nhỏ** (ví dụ 200 dòng) làm fixture `tests/data/sample.csv` cho test nhanh.
  - **(b)** Sinh **dữ liệu tổng hợp** trong test (không cần file thật).
  - **(c)** Tải dataset trong CI (cần mạng + URL ổn định).
- Tests phụ thuộc `data/processed` và `artifacts` hiện **skip** khi thiếu → CI coverage thấp;
  cần chuyển sang fixture nhỏ để test thật sự chạy.

### Nhóm E — Tài liệu
- README: cập nhật cách chạy, Docker, CI badge.
- `docs/deployment.md`: hướng dẫn local + Docker + gọi API.
- (Tùy chọn) sơ đồ kiến trúc đơn giản.

### Nhóm F — Chất lượng code (tùy chọn)
- `ruff` (lint + format), `pre-commit` hook.
- Rà soát type hints ở các hàm public.

## 4. Milestones

| # | Milestone | Deliverable | Commit |
|---|-----------|-------------|--------|
| P3-1 ✅ | MLflow tracking | log trong train/compare/tune/select_threshold + `mlflow.db` (SQLite) ignored | có |
| P3-2 | Docker | `Dockerfile`, `.dockerignore`, chạy được API | có |
| P3-3 | CI + fixtures | `.github/workflows/ci.yml`, fixture test, badge | có |
| P3-4 | Docs + polish | `docs/deployment.md`, README, ruff | có |

Mỗi milestone 1 commit; P3-3 phụ thuộc quyết định về dữ liệu ở Nhóm D.

> **Ghi chú P3-1:** MLflow 3.x đã bỏ mặc định filesystem file store; dự án dùng
> backend **SQLite cục bộ** (`mlflow.db`) + artifact trong `mlruns/` — vẫn local,
> không cần server. `src/tracking.py` là lớp mỏng, có thể tắt bằng `--no-mlflow`
> hoặc `CHURN_MLFLOW=0`, và không làm hỏng pipeline nếu MLflow lỗi/thiếu.

## 5. Definition of Done

- [x] MLflow log được params/metrics/artifacts cho ít nhất `train` và `tune`.
- [ ] `docker build` + `docker run` chạy API, `/health` trả `model_loaded=true`.
- [ ] CI tự chạy lint + tests trên push, **không cần** file dữ liệu bị ignore.
- [ ] Tests quan trọng (features/registry/evaluate/threshold) chạy được trên CI.
- [ ] README + `docs/deployment.md` đủ để người khác dựng lại dự án.

## 6. Rủi ro & lưu ý

- **Model trong image:** container cần `artifacts/churn_model.joblib`. Chọn: build sẵn model
  trong image, hoặc mount volume, hoặc huấn luyện lúc khởi động (chậm). Cần chốt.
- **Kích thước image:** XGBoost + scipy + shap khá nặng; dùng `slim` và chỉ cài deps cần cho serving.
- **Dữ liệu trên CI:** nếu không xử lý (Nhóm D), CI sẽ "xanh giả" do test bị skip.
- **MLflow UI** cần chạy thêm tiến trình; chỉ dùng local cho MVP.
- **Windows vs Linux:** line endings (`.gitattributes` đã có), đường dẫn trong Docker là Linux.

## 7. Quyết định cần chốt

1. **Dữ liệu cho CI:** (a) commit mẫu nhỏ / (b) sinh dữ liệu tổng hợp / (c) tải trong CI?
2. **Model trong Docker:** build sẵn vào image / mount volume / train lúc start?
3. **MLflow:** chỉ local (`mlruns/`) hay có backend/UI riêng? (đề xuất: local)
4. **Lint/format:** có thêm `ruff` + `pre-commit` không?
