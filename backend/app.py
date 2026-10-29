"""FastAPI·Swagger·학습 작업과 모델 버전 조회를 한 파일에서 관리한다."""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field
from swagger_ui_bundle import swagger_ui_path

from models.model import (FEATURES, REPORTS_DIR, ROOT, WEIGHTS_DIR, SensorInput, current_version,
                          load_model, model_versions, predict_one, valid_version, weight_path)
from models.model import pair_fingerprint
from models.train import (JOBS_DIR, SPLIT_STRATEGY, TrainingConfig, load_dataset, training_input_limits,
                          training_plan, write_json)
from scripts.runtime_logging import RuntimeLogger
from backend.classrooms import ClassroomService, MeasurementInput, SimulationControl, SourceControl

API_STARTED_AT = datetime.now(timezone.utc)


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


def load_job_record(path: Path, job_id: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("id") != job_id:
        raise ValueError("작업 id가 파일명과 일치하지 않습니다.")
    if record.get("status") not in ("queued", "running", "completed", "failed", "cancelled"):
        raise ValueError("작업 status가 없거나 올바르지 않습니다.")
    created = record.get("created_at_utc")
    if not isinstance(created, str) or datetime.fromisoformat(created).utcoffset() is None:
        raise ValueError("created_at_utc에는 시간대가 포함된 시각이 필요합니다.")
    config = record.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("name"), str):
        raise ValueError("작업 config.name이 없거나 올바르지 않습니다.")
    if record.get("pid") is not None and (type(record["pid"]) is not int or record["pid"] <= 0):
        raise ValueError("작업 pid는 양의 정수여야 합니다.")
    return record


def unreadable_job(path: Path, error: Exception, logger=None) -> dict:
    if logger is not None:
        logger.write("api", "training_history", event="unreadable", job_file=path.name,
                     error_type=type(error).__name__, reason=str(error))
    # 손상된 원본을 덮어쓰지 않고 조회에만 사용할 오류 항목을 반환한다.
    return {"id": path.stem, "status": "failed", "phase": "학습 이력 읽기 오류", "history_error": True,
            "created_at_utc": None, "config": {"name": f"읽을 수 없는 이력: {path.name}"},
            "version": None, "curves": {}, "cancel_requested": False,
            "error": f"이 학습 이력만 읽지 못했습니다. 원본 파일은 보존됩니다: {error}"}


def read_job(job_id: str, processes: dict | None = None, logger=None) -> dict:
    path = job_path(job_id)
    if not path.is_file():
        raise HTTPException(404, "학습 작업을 찾을 수 없습니다.")
    try:
        record = load_job_record(path, job_id)
    except (OSError, ValueError) as error:
        return unreadable_job(path, error, logger)
    if record["status"] in ("queued", "running"):
        process = (processes or {}).get(job_id)
        interrupted = False
        if process is not None:
            interrupted = process.poll() is not None
        elif record.get("pid"):
            # API 재시작 후에는 PID가 같은 학습 작업을 가리키는지 확인한다.
            command = Path(f"/proc/{record['pid']}/cmdline")
            try:
                interrupted = not command.is_file() or job_id.encode() not in command.read_bytes().split(b"\x00")
            except FileNotFoundError:
                interrupted = True
        elif record["status"] == "queued":
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["created_at_utc"])).total_seconds()
            interrupted = age > 30
        if interrupted:
            # 프로세스 종료 직전에 완료 상태를 저장했을 수 있으므로 다시 읽는다.
            try:
                record = load_job_record(path, job_id)
            except (OSError, ValueError) as error:
                return unreadable_job(path, error, logger)
            if record["status"] in ("queued", "running"):
                record.update(status="failed", phase="학습 중단", error="학습 프로세스가 중단되었습니다. 새 학습을 시작할 수 있습니다.",
                              finished_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, record)
    record["cancel_requested"] = (JOBS_DIR / f"{job_id}.cancel").is_file()
    return record


def list_jobs(processes=None, logger=None):
    records = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            records.append(read_job(path.stem, processes, logger))
        except HTTPException:
            records.append(unreadable_job(path, ValueError("작업 파일명은 32자리 소문자 16진수여야 합니다."), logger))
    return sorted(records, key=lambda item: item["created_at_utc"] or "", reverse=True)


def create_app(model_path: Path | None = None, *, log_root: Path | None = None, state_dir: Path | None = None,
               enable_classrooms: bool = True, start_simulation: bool = True) -> FastAPI:
    def get_pair(application, version=None):
        version = version or current_version()
        if version in application.state.pairs:
            return application.state.pairs[version]

        path = REPORTS_DIR / version / "light_comparison.json"
        if not path.is_file():
            raise HTTPException(404, detail="해당 버전의 평가 보고서가 없습니다.")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not all(weight_path(version, mode).is_file() for mode in report["modes"]):
            raise HTTPException(404, detail="해당 버전의 가중치가 없습니다.")
        pair = {mode: load_model(weight_path(version, mode)) for mode in report["modes"]}
        if any(model.metadata["version"] != version for model in pair.values()):
            raise ValueError("모델의 가중치 버전이 일치하지 않습니다.")

        signature = pair_fingerprint({mode: {**model.metadata, "state_dict": model.network.state_dict(),
                                             "mean": model.mean, "scale": model.scale} for mode, model in pair.items()})
        application.state.pairs[version] = {**pair, "report": report, "fingerprint": signature}
        if len(application.state.pairs) > 8:
            oldest = next(iter(application.state.pairs))
            if oldest != version:
                del application.state.pairs[oldest]
        return application.state.pairs[version]

    @asynccontextmanager
    async def lifespan(application):
        logger = RuntimeLogger(root=log_root, started_at=API_STARTED_AT)
        application.state.runtime_logger = logger
        logger.write("api", "lifecycle", event="started", process_started_at_utc=API_STARTED_AT.isoformat())
        simulation_task = None
        application.state.classrooms = None
        application.state.classroom_error = None
        try:
            application.state.pairs = {}
            application.state.processes = {}
            application.state.catalog = None
            if model_path is not None:
                load_model(model_path)
            try:
                get_pair(application)
            except Exception as error:
                logger.write("api", "lifecycle", event="model_unavailable", error_type=type(error).__name__)
            try:
                JOBS_DIR.mkdir(parents=True, exist_ok=True)
                list_jobs(logger=logger)
            except (OSError, ValueError) as error:
                logger.write("api", "lifecycle", event="model_management_unavailable", error_type=type(error).__name__)
            if enable_classrooms:
                def classroom_predict(values):
                    pair = get_pair(application)
                    model = pair["with_light"]
                    return {"prediction": predict_one(model, values), "features": model.metadata["features"],
                            "threshold": model.metadata["threshold"], "fingerprint": pair["fingerprint"]}
                try:
                    service = ClassroomService(state_dir or Path(os.environ.get("CLASSROOM_STATE_DIR", ROOT / "state")),
                        classroom_predict, logger, stale_after=float(os.environ.get("SENSOR_STALE_SECONDS", "60")))
                    application.state.classrooms = service
                    if start_simulation:
                        await asyncio.to_thread(service.emit)
                        simulation_task = asyncio.create_task(service.run())
                except Exception as error:
                    if application.state.classrooms is not None:
                        with suppress(Exception):
                            application.state.classrooms.close()
                    application.state.classrooms = None
                    application.state.classroom_error = "강의실 저장소를 준비할 수 없습니다. state 쓰기 볼륨과 API 단일 실행을 확인하세요."
                    logger.write("api", "lifecycle", event="classroom_unavailable", error_type=type(error).__name__)
            yield
        except Exception as error:
            logger.write("api", "lifecycle", event="failed", error_type=type(error).__name__)
            raise
        finally:
            if simulation_task:
                simulation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await simulation_task
            if application.state.classrooms is not None:
                application.state.classrooms.close()
            logger.write("api", "lifecycle", event="stopped")

    application = FastAPI(title="강의실 사용 현황과 모델 관리", version="3.0",
                          description="시연·실센서 측정으로 강의실 현황을 확인하고 기존 모델 실험을 관리합니다. 시연 측정 시각과 모델 시간 피처는 UTC 기준입니다.",
                          docs_url=None, redoc_url=None, lifespan=lifespan)
    application.mount("/static/swagger", StaticFiles(directory=swagger_ui_path), name="swagger")

    @application.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status, error_type = 500, None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            # 헤더·쿼리 문자열·본문 대신 요청 처리 메타데이터만 기록한다.
            application.state.runtime_logger.write("api", "request", event="completed",
                request_id=request.state.request_id, method=request.method, path=request.url.path,
                route=getattr(request.scope.get("route"), "path", None), status_code=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                error_type=error_type or getattr(request.state, "error_type", None) or ("HTTPStatusError" if status >= 400 else None),
                error_fields=getattr(request.state, "error_fields", None))

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
        request.state.error_type = "RequestValidationError"
        request.state.error_fields = [".".join(str(part) for part in item["loc"]) for item in error.errors()]
        return JSONResponse(status_code=400, content={
            "error": "입력의 형식 또는 허용 범위를 확인하세요.",
            "details": [{"field": ".".join(str(part) for part in item["loc"]),
                         "reason": item["type"]} for item in error.errors()],
        })

    @application.exception_handler(HTTPException)
    async def unavailable(request, error):
        request.state.error_type = "HTTPException"
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
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            report = {"model_version": version, "legacy": True, "config": None, "models": pair["report"]["modes"],
                      "note": "기존 버전은 저장된 학습 정보만 제공합니다. 없는 설정과 손실 곡선을 추정하지 않습니다."}
        report["input_limits"] = training_input_limits(report["models"])
        return report

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

    def classrooms():
        if application.state.classrooms is None:
            raise HTTPException(503, application.state.classroom_error or "강의실 서비스가 비활성 상태입니다.")
        return application.state.classrooms

    @application.get("/classrooms", tags=["강의실 현황"], summary="저장된 최신 강의실 상태, 새 추론 없음")
    def classroom_list():
        return classrooms().snapshot()

    @application.get("/classrooms/{classroom_id}", tags=["강의실 현황"], summary="강의실 측정·판별 이력")
    def classroom_detail(classroom_id: str):
        try:
            return classrooms().room_view(classroom_id, detail=True)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @application.post("/classrooms/{classroom_id}/measurements", tags=["강의실 현황"], summary="실센서 측정 수신, UTC에서 시간·요일 파생")
    def receive_measurement(classroom_id: str, measurement: MeasurementInput, request: Request):
        try:
            return classrooms().ingest(classroom_id, measurement_id=measurement.measurement_id,
                observed_at=measurement.observed_at.isoformat(), sensors={name: getattr(measurement, name) for name in
                    ("Temperature", "Humidity", "Light", "CO2")}, source="real", request_id=request.state.request_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @application.put("/classrooms/{classroom_id}/source", tags=["강의실 현황"], summary="수신 소스 전환, 전환 직후 수신 대기")
    def classroom_source(classroom_id: str, change: SourceControl, request: Request):
        try:
            return classrooms().set_source(classroom_id, change.source, request.state.request_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @application.post("/simulation/control", tags=["강의실 현황"], summary="모든 브라우저가 공유하는 UTC 하루 시연 제어")
    def simulation_control(change: SimulationControl, request: Request):
        try:
            return classrooms().control(change.action, change.speed, request.state.request_id)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @application.post("/predict", response_model=Prediction, responses=errors, summary="선택 모델로 예측", tags=["예측"])
    def predict(sensors: SensorInput, request: Request, version: str | None = version_query,
                mode: Literal["primary", "without_light"] = "primary"):
        model_key = "with_light" if mode == "primary" else mode
        pair = get_pair(application, version)
        if model_key not in pair:
            raise HTTPException(404, "해당 버전에는 조도 제외 비교 모델이 없습니다.")
        result = predict_one(pair[model_key], sensors.model_dump())
        model = pair[model_key]
        application.state.runtime_logger.inference(request_id=request.state.request_id, sensors=sensors.model_dump(),
            version=model.metadata["version"], mode=model_key, features=model.metadata["features"],
            threshold=model.metadata["threshold"], fingerprint=pair["fingerprint"], prediction=result)
        return result

    @application.post("/compare", response_model=dict[str, Prediction], responses=errors,
                      summary="동일한 입력을 저장된 모델들로 예측", tags=["예측"])
    def compare(sensors: SensorInput, request: Request, version: str | None = version_query):
        pair = get_pair(application, version)
        results = {mode: predict_one(pair[mode], sensors.model_dump()) for mode in pair["report"]["modes"]}
        for mode, result in results.items():
            model = pair[mode]
            application.state.runtime_logger.inference(request_id=request.state.request_id, sensors=sensors.model_dump(),
                version=model.metadata["version"], mode=mode, features=model.metadata["features"],
                threshold=model.metadata["threshold"], fingerprint=pair["fingerprint"], prediction=result)
        return results

    @application.get("/training/options", summary="기본 설정과 데이터 피처 분포", tags=["학습"])
    def training_options(validation_ratio: float = Query(default=0.2, ge=0.1, le=0.4)):
        paths = [ROOT / "data" / name for name in ("training_dataset.csv", "test_dataset.csv")]
        metadata = ROOT / "data/dataset_sources.json"
        if metadata.is_file():
            paths.append(metadata)
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        if application.state.catalog is None or getattr(application.state, "catalog_signature", None) != signature:
            try:
                source = load_dataset(ROOT / "data/training_dataset.csv")
                test = load_dataset(ROOT / "data/test_dataset.csv")
            except (OSError, ValueError, KeyError) as error:
                raise HTTPException(400, f"학습 자료를 확인하세요: {error}") from error
            application.state.catalog_data = (source, test)
            import numpy as np
            profiles = []
            for index, name in enumerate(FEATURES):
                values = source.x[:, index] if index < 4 else np.array([
                    date.hour if name == "Hour" else date.weekday() for date in source.dates])
                profiles.append({"key": name, "minimum": float(values.min()), "maximum": float(values.max()),
                                 "median": float(np.median(values))})
            application.state.catalog = {"defaults": TrainingConfig().model_dump(), "features": profiles,
                                         "training_data": source.summary, "test_data": test.summary,
                                         "feature_scope": "시간순 학습·검증 분할 전의 전체 CSV",
                                         "split_strategy": SPLIT_STRATEGY,
                                         "config_schema": TrainingConfig.model_json_schema()}
            application.state.catalog_signature = signature
        return {**application.state.catalog,
                "data_plan": training_plan(*application.state.catalog_data, validation_ratio)}

    @application.get("/training/jobs", summary="학습 작업 상태와 이력", tags=["학습"])
    def jobs():
        records = list_jobs(application.state.processes, application.state.runtime_logger)
        application.state.processes = {
            job_id: process for job_id, process in application.state.processes.items()
            if process.poll() is None
        }
        summaries = [{key: value for key, value in record.items() if key != "curves"} for record in records]
        return {"jobs": summaries}

    @application.post("/training/jobs", status_code=202, responses=errors, summary="새 모델 학습 시작", tags=["학습"])
    def start_training(config: TrainingConfig, request: Request):
        try:
            with FileLock(JOBS_DIR / ".submission.lock", timeout=0):
                records = list_jobs(application.state.processes, application.state.runtime_logger)
                active_id = next((item["id"] for item in records if item["status"] in ("queued", "running")), None)
                # 이력이 손상돼도 API가 추적 중인 실제 프로세스는 중복 제출을 막는다.
                active_id = active_id or next((job_id for job_id, process in application.state.processes.items()
                                               if process.poll() is None), None)
                if active_id:
                    raise HTTPException(409, f"진행 중인 학습이 있습니다: {active_id}")
                try:
                    plan = training_options(config.validation_ratio)["data_plan"]
                except (OSError, ValueError, KeyError) as error:
                    raise HTTPException(400, f"학습 자료를 확인하세요: {error}") from error
                if not plan["ready"]:
                    raise HTTPException(400, " / ".join(plan["errors"]))
                job_id = uuid.uuid4().hex
                logger = application.state.runtime_logger
                path = job_path(job_id)
                record = {
                    "id": job_id,
                    "status": "queued",
                    "phase": "학습 준비",
                    "progress": 0,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config.model_dump(),
                    "version": None,
                    "error": None,
                    "curves": {},
                    "data_plan": plan,
                    "log_run_id": logger.run_id, "log_root": str(logger.root), "request_id": request.state.request_id,
                }
                write_json(path, record)
                logger.write("model", "training", event="queued", job_id=job_id, request_id=request.state.request_id)
                try:
                    worker_command = [sys.executable, "-m", "models.train", "--job-id", job_id]
                    with (JOBS_DIR / f"{job_id}.log").open("wb") as output:
                        process = subprocess.Popen(worker_command,
                                                   cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                                   env={**os.environ, "CLASSROOM_LOG_RUN_ID": logger.run_id,
                                                        "CLASSROOM_LOG_ROOT": str(logger.root),
                                                        "CLASSROOM_LOG_REQUEST_ID": request.state.request_id})
                    application.state.processes[job_id] = process
                    # 무거운 PyTorch import 중에도 재시작한 API가 실제 작업을 찾을 수 있게 PID를 남긴다.
                    latest = json.loads(path.read_text(encoding="utf-8"))
                    latest["pid"] = process.pid
                    write_json(path, latest)
                except OSError as error:
                    record.update(status="failed", error="학습 프로세스를 시작할 수 없습니다.")
                    write_json(path, record)
                    logger.write("model", "training", event="finished", job_id=job_id, status="failed",
                                 request_id=request.state.request_id, error_type=type(error).__name__)
                    raise HTTPException(503, record["error"]) from error
                return record
        except Timeout as error:
            raise HTTPException(409, "다른 학습 요청을 처리 중입니다. 잠시 후 다시 시도하세요.") from error

    @application.get("/training/jobs/{job_id}", summary="진행률·손실·학습 결과", tags=["학습"])
    def job(job_id: str):
        return read_job(job_id, application.state.processes, application.state.runtime_logger)

    @application.post("/training/jobs/{job_id}/cancel", summary="학습 중지 요청", tags=["학습"])
    def cancel(job_id: str, request: Request):
        record = read_job(job_id, application.state.processes, application.state.runtime_logger)
        if record["status"] not in ("queued", "running"):
            raise HTTPException(409, "이미 종료된 학습입니다.")
        (JOBS_DIR / f"{job_id}.cancel").touch()
        RuntimeLogger.for_training(record).write("model", "training", event="cancel_requested", job_id=job_id,
            request_id=request.state.request_id, request_run_id=application.state.runtime_logger.run_id)
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
