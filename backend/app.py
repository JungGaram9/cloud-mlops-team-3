"""FastAPI·Swagger·학습 작업과 모델 버전 조회를 한 파일에서 관리한다."""

import json
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field
from swagger_ui_bundle import swagger_ui_path

from models.model import (FEATURES, REPORTS_DIR, ROOT, WEIGHTS_DIR, SensorInput, current_version,
                          load_model, model_versions, predict_one, valid_version, weight_path)
from models.train import JOBS_DIR, TrainingConfig, load_dataset, write_json


class Prediction(BaseModel):
    model_version: str
    occupancy: Literal[0, 1] = Field(description="0: 비어 있음, 1: 사용 중")
    label: str
    probability: float = Field(ge=0, le=1, description="사용 중일 확률")


class InputErrorDetail(BaseModel):
    field: str
    reason: str


class InputError(BaseModel):
    error: str
    details: list[InputErrorDetail]


def job_path(job_id: str) -> Path:
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(404, "학습 작업을 찾을 수 없습니다.")
    return JOBS_DIR / f"{job_id}.json"


def read_job(job_id: str, processes: dict | None = None) -> dict:
    path = job_path(job_id)
    if not path.is_file():
        raise HTTPException(404, "학습 작업을 찾을 수 없습니다.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] in ("queued", "running"):
        process = (processes or {}).get(job_id)
        interrupted = process is not None and process.poll() is not None
        if process is None and record.get("pid"):
            command = Path(f"/proc/{record['pid']}/cmdline")
            try:
                interrupted = not command.is_file() or job_id.encode() not in command.read_bytes().split(b"\x00")
            except FileNotFoundError:
                interrupted = True
        if process is None and record["status"] == "queued" and not record.get("pid"):
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["created_at_utc"])).total_seconds()
            interrupted = age > 30
        if interrupted:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] in ("queued", "running"):
                record.update(status="failed", phase="학습 중단", error="학습 프로세스가 중단되었습니다. 새 학습을 시작할 수 있습니다.",
                              finished_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, record)
    record["cancel_requested"] = (JOBS_DIR / f"{job_id}.cancel").is_file()
    return record


def list_jobs(processes=None):
    records = [read_job(path.stem, processes) for path in JOBS_DIR.glob("*.json")]
    return sorted(records, key=lambda item: item["created_at_utc"], reverse=True)


def create_app(model_path: Path | None = None) -> FastAPI:
    def get_pair(application, version=None):
        version = version or current_version()
        if version not in application.state.pairs:
            path = REPORTS_DIR / version / "light_comparison.json"
            if not path.is_file():
                raise HTTPException(404, detail="해당 버전의 평가 보고서가 없습니다.")
            report = json.loads(path.read_text(encoding="utf-8"))
            if not all(weight_path(version, mode).is_file() for mode in report["modes"]):
                raise HTTPException(404, detail="해당 버전의 가중치가 없습니다.")
            pair = {mode: load_model(weight_path(version, mode)) for mode in report["modes"]}
            if any(model.metadata["version"] != version for model in pair.values()):
                raise ValueError("모델의 가중치 버전이 일치하지 않습니다.")
            application.state.pairs[version] = {**pair, "report": report}
            if len(application.state.pairs) > 8:
                oldest = next(iter(application.state.pairs))
                if oldest != version:
                    del application.state.pairs[oldest]
        return application.state.pairs[version]

    @asynccontextmanager
    async def lifespan(application):
        application.state.pairs = {}
        application.state.processes = {}
        application.state.catalog = None
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        if model_path is not None:
            load_model(model_path)
        get_pair(application)
        list_jobs()
        yield

    application = FastAPI(title="강의실 모델 학습 실험실", version="2.0",
                          description="피처와 하이퍼파라미터를 조정해 PyTorch 모델을 학습·평가하고 버전별로 보존합니다.",
                          docs_url=None, redoc_url=None, lifespan=lifespan)
    application.mount("/static/swagger", StaticFiles(directory=swagger_ui_path), name="swagger")

    @application.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/docs")

    @application.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="강의실 모델 Swagger UI",
                                  swagger_js_url="/static/swagger/swagger-ui-bundle.js",
                                  swagger_css_url="/static/swagger/swagger-ui.css",
                                  swagger_favicon_url="/static/swagger/favicon-32x32.png",
                                  swagger_ui_parameters={"tryItOutEnabled": True, "defaultModelsExpandDepth": -1})

    @application.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        return JSONResponse(status_code=400, content={
            "error": "입력의 형식 또는 허용 범위를 확인하세요.",
            "details": [{"field": ".".join(str(part) for part in item["loc"]),
                         "reason": item["type"]} for item in error.errors()],
        })

    @application.exception_handler(HTTPException)
    async def unavailable(request, error):
        return JSONResponse(status_code=error.status_code, content={"error": str(error.detail)})

    version_query = Query(default=None, pattern=r"^v[1-9][0-9]*$", description="생략하면 기본 모델 버전")

    @application.get("/health", summary="기본 모델 준비 상태", tags=["상태"])
    def health():
        version = current_version()
        get_pair(application, version)
        return {"status": "ok", "model_version": version, "framework": "pytorch"}

    @application.get("/metrics", summary="선택 모델 평가", tags=["평가"])
    def metrics(version: str | None = version_query):
        return get_pair(application, version)["report"]["modes"]["with_light"]

    @application.get("/comparison", summary="선택 모델과 가능한 조도 제외 비교", tags=["평가"])
    def comparison(version: str | None = version_query):
        return get_pair(application, version)["report"]

    @application.get("/versions", summary="역대 모델의 피처·설정·평가", tags=["버전 관리"])
    def versions():
        return model_versions(current_version())

    @application.get("/versions/{version}/training", summary="버전의 학습 설정·곡선·데이터·평가", tags=["버전 관리"])
    def training_report(version: str):
        try:
            valid_version(version)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        pair = get_pair(application, version)
        path = REPORTS_DIR / version / "training.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"model_version": version, "legacy": True, "config": None, "models": pair["report"]["modes"],
                "note": "기존 버전은 저장된 학습 정보만 제공합니다. 없는 설정과 손실 곡선을 추정하지 않습니다."}

    @application.post("/versions/{version}/activate", summary="기본 예측 모델 선택", tags=["버전 관리"])
    def activate(version: str):
        try:
            valid_version(version)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        pair = get_pair(application, version)
        try:
            with FileLock(WEIGHTS_DIR / ".registration.lock", timeout=2):
                write_json(WEIGHTS_DIR / "current.json", {"version": version,
                           "pair_fingerprint": pair["report"].get("pair_fingerprint")})
        except Timeout as error:
            raise HTTPException(409, "모델을 저장 중입니다. 잠시 후 다시 시도하세요.") from error
        return {"model_version": version}

    errors = {400: {"model": InputError, "description": "입력 또는 버전 형식 오류"}}

    @application.post("/predict", response_model=Prediction, responses=errors, summary="선택 모델로 예측", tags=["예측"])
    def predict(sensors: SensorInput, version: str | None = version_query,
                mode: Literal["primary", "without_light"] = "primary"):
        key = "with_light" if mode == "primary" else mode
        pair = get_pair(application, version)
        if key not in pair:
            raise HTTPException(404, "해당 버전에는 조도 제외 비교 모델이 없습니다.")
        return predict_one(pair[key], sensors.model_dump())

    @application.post("/compare", response_model=dict[str, Prediction], responses=errors,
                      summary="동일한 입력을 저장된 모델들로 예측", tags=["예측"])
    def compare(sensors: SensorInput, version: str | None = version_query):
        pair = get_pair(application, version)
        return {mode: predict_one(pair[mode], sensors.model_dump()) for mode in pair["report"]["modes"]}

    @application.get("/training/options", summary="기본 설정과 데이터 피처 분포", tags=["학습"])
    def training_options():
        if application.state.catalog is None:
            source = load_dataset(ROOT / "data/training_dataset.csv")
            test = load_dataset(ROOT / "data/test_dataset.csv")
            import numpy as np
            profiles = []
            for index, name in enumerate(FEATURES):
                values = source.x[:, index] if index < 4 else np.array([
                    date.hour if name == "Hour" else date.weekday() for date in source.dates])
                profiles.append({"key": name, "minimum": float(values.min()), "maximum": float(values.max()),
                                 "median": float(np.median(values))})
            application.state.catalog = {"defaults": TrainingConfig().model_dump(), "features": profiles,
                                         "training_data": source.summary, "test_data": test.summary,
                                         "split_strategy": "chronological",
                                         "config_schema": TrainingConfig.model_json_schema()}
        return application.state.catalog

    @application.get("/training/jobs", summary="학습 작업 상태와 이력", tags=["학습"])
    def jobs():
        records = list_jobs(application.state.processes)
        application.state.processes = {key: value for key, value in application.state.processes.items() if value.poll() is None}
        return {"jobs": [{key: value for key, value in record.items() if key != "curves"} for record in records]}

    @application.post("/training/jobs", status_code=202, responses=errors, summary="새 모델 학습 시작", tags=["학습"])
    def start_training(config: TrainingConfig):
        try:
            with FileLock(JOBS_DIR / ".submission.lock", timeout=0):
                active = next((item for item in list_jobs(application.state.processes) if item["status"] in ("queued", "running")), None)
                if active:
                    raise HTTPException(409, f"진행 중인 학습이 있습니다: {active['id']}")
                job_id = uuid.uuid4().hex
                record = {"id": job_id, "status": "queued", "phase": "학습 준비", "progress": 0,
                          "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config.model_dump(),
                          "version": None, "error": None, "curves": {}}
                write_json(job_path(job_id), record)
                try:
                    with (JOBS_DIR / f"{job_id}.log").open("wb") as output:
                        process = subprocess.Popen([sys.executable, "-m", "models.train", "--job-id", job_id],
                                                   cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    application.state.processes[job_id] = process
                    # 무거운 PyTorch import 중에도 재시작한 API가 실제 작업을 찾을 수 있게 PID를 남긴다.
                    latest = json.loads(job_path(job_id).read_text(encoding="utf-8"))
                    latest["pid"] = process.pid
                    write_json(job_path(job_id), latest)
                except OSError as error:
                    record.update(status="failed", error="학습 프로세스를 시작할 수 없습니다.")
                    write_json(job_path(job_id), record)
                    raise HTTPException(503, record["error"]) from error
                return record
        except Timeout as error:
            raise HTTPException(409, "다른 학습 요청을 처리 중입니다. 잠시 후 다시 시도하세요.") from error

    @application.get("/training/jobs/{job_id}", summary="진행률·손실·학습 결과", tags=["학습"])
    def job(job_id: str):
        return read_job(job_id, application.state.processes)

    @application.post("/training/jobs/{job_id}/cancel", summary="학습 중지 요청", tags=["학습"])
    def cancel(job_id: str):
        record = read_job(job_id, application.state.processes)
        if record["status"] not in ("queued", "running"):
            raise HTTPException(409, "이미 종료된 학습입니다.")
        (JOBS_DIR / f"{job_id}.cancel").touch()
        return {"id": job_id, "cancel_requested": True}

    original_openapi = application.openapi

    def openapi():
        schema = original_openapi()
        for path in schema["paths"].values():
            for operation in path.values():
                if isinstance(operation, dict) and "responses" in operation:
                    if "422" in operation["responses"]:
                        operation["responses"].pop("422")
                        operation["responses"].setdefault("400", {"description": "입력 형식 또는 범위 오류"})
        return schema

    application.openapi = openapi
    return application


app = create_app()
