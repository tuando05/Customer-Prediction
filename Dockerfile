# syntax=docker/dockerfile:1

# Image phục vụ API churn. Model cuối được huấn luyện sẵn ngay trong image
# (cần data/raw/*.csv trong build context).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURN_MLFLOW=0

WORKDIR /app

# Cài phụ thuộc trước để tận dụng cache layer.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Mã nguồn + dữ liệu thô (để huấn luyện model khi build).
COPY src ./src
COPY app ./app
COPY pyproject.toml ./
COPY data/raw ./data/raw

# Huấn luyện model cuối -> artifacts/churn_model.joblib + artifacts/threshold.json
RUN python -m src.pipeline --no-test

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
