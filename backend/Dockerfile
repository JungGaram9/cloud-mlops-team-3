FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app

COPY backend/requirements*.txt /app/backend/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r backend/requirements.txt

COPY data/training_dataset.csv data/test_dataset.csv /app/data/
COPY data/dataset_sources.json /app/data/
COPY scripts/merge_data.py /app/scripts/merge_data.py
COPY scripts/runtime_logging.py /app/scripts/runtime_logging.py
COPY backend/ /app/backend/
COPY models/ /app/models/
COPY reports/ /app/reports/
# 빌드에서는 학습·평가·학습 테스트를 실행하지 않는다. 초기 준비는 런타임 trainer가 담당한다.

RUN useradd --create-home --uid 10001 classroom \
    && mkdir -p /app/logs /app/state && chown classroom:classroom /app/logs /app/state
USER classroom
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
