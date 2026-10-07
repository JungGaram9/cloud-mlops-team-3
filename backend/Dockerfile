FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app

COPY backend/requirements*.txt /app/backend/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r backend/requirements.txt

COPY data/training_dataset.csv data/test_dataset.csv /app/data/
COPY backend/ /app/backend/
COPY models/ /app/models/
# 새로 내려받아도 로컬 모델 파일 없이 빌드한다. 조도 포함 정확도 85% 미만은 실패한다.
RUN python -m models.train
RUN python -m unittest backend.test_app -v

RUN useradd --create-home --uid 10001 classroom
USER classroom
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
