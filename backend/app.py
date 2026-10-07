"""FastAPI·Swagger와 버전별 모델 쌍 예측을 한 파일에서 관리한다."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from swagger_ui_bundle import swagger_ui_path

from models.model import (REPORTS_DIR, SensorInput, current_version, load_model,
                          model_versions, predict_one, weight_path)


class Prediction(BaseModel):
    model_version: str
    occupancy: Literal[0, 1] = Field(description="0: 비어 있음, 1: 사용 중")
    label: str
    probability: float = Field(ge=0, le=1, description="사용 중일 확률")


class ComparisonPrediction(BaseModel):
    with_light: Prediction
    without_light: Prediction


class InputErrorDetail(BaseModel):
    field: str
    reason: str


class InputError(BaseModel):
    error: str
    details: list[InputErrorDetail]


def create_app(model_path: Path | None = None) -> FastAPI:
    def get_pair(application, version=None):
        version = version or application.state.version
        if version not in application.state.pairs:
            path = REPORTS_DIR / version / "light_comparison.json"
            if not path.is_file() or not all(weight_path(version, mode).is_file() for mode in ("with_light", "without_light")):
                raise HTTPException(404, detail="해당 버전의 두 가중치와 평가 보고서가 모두 필요합니다.")
            pair = {
                "with_light": load_model(weight_path(version)),
                "without_light": load_model(weight_path(version, "without_light")),
                "report": json.loads(path.read_text(encoding="utf-8")),
            }
            if any(pair[mode].metadata["version"] != version for mode in ("with_light", "without_light")):
                raise ValueError("모델 쌍의 가중치 버전이 일치하지 않습니다.")
            application.state.pairs[version] = pair
        return application.state.pairs[version]

    @asynccontextmanager
    async def lifespan(application):
        application.state.version = current_version()
        application.state.pairs = {}
        if model_path is not None:
            load_model(model_path)
        get_pair(application)
        yield

    application = FastAPI(title="강의실 센서 실험 · PyTorch", version="1.0",
                          description="시간·요일을 포함한 입력으로 버전별 조도 포함·제외 모델 쌍을 비교합니다.",
                          docs_url=None, redoc_url=None, lifespan=lifespan)
    application.mount("/static/swagger", StaticFiles(directory=swagger_ui_path), name="swagger")

    @application.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/docs")

    @application.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="강의실 예측 Swagger UI",
                                  swagger_js_url="/static/swagger/swagger-ui-bundle.js",
                                  swagger_css_url="/static/swagger/swagger-ui.css",
                                  swagger_favicon_url="/static/swagger/favicon-32x32.png",
                                  swagger_ui_parameters={"tryItOutEnabled": True, "defaultModelsExpandDepth": -1})

    @application.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        return JSONResponse(status_code=400, content={
            "error": "센서 입력의 형식 또는 허용 범위를 확인하세요.",
            "details": [{"field": ".".join(str(part) for part in item["loc"]),
                         "reason": item["type"]} for item in error.errors()],
        })

    @application.exception_handler(HTTPException)
    async def unavailable(request, error):
        return JSONResponse(status_code=error.status_code, content={"error": str(error.detail)})

    version_query = Query(default=None, pattern=r"^v[1-9][0-9]*$", description="선택할 모델 쌍 버전, 생략하면 현재 모델")

    @application.get("/health", summary="현재 모델 준비 상태", tags=["상태"])
    def health():
        return {"status": "ok", "model_version": application.state.version, "framework": "pytorch"}

    @application.get("/metrics", summary="선택한 버전의 조도 포함 평가", tags=["평가"])
    def metrics(version: str | None = version_query):
        return get_pair(application, version)["report"]["modes"]["with_light"]

    @application.get("/comparison", summary="선택한 버전의 조도 포함·제외 평가", tags=["평가"])
    def comparison(version: str | None = version_query):
        return get_pair(application, version)["report"]

    @application.get("/versions", summary="모델 변경별 버전과 두 가중치 상태", tags=["버전 관리"])
    def versions():
        return model_versions(application.state.version)

    errors = {400: {"model": InputError, "description": "센서 입력 또는 버전 형식 오류"}}

    @application.post("/predict", response_model=Prediction, responses=errors,
                      summary="선택한 버전의 조도 포함 모델로 예측", tags=["예측"])
    def predict(sensors: SensorInput, version: str | None = version_query):
        return predict_one(get_pair(application, version)["with_light"], sensors.model_dump())

    @application.post("/compare", response_model=ComparisonPrediction, responses=errors,
                      summary="센서·시간·요일 한 건을 같은 버전의 두 모델로 예측", tags=["예측"])
    def compare(sensors: SensorInput, version: str | None = version_query):
        pair = get_pair(application, version)
        return {mode: predict_one(pair[mode], sensors.model_dump()) for mode in ("with_light", "without_light")}

    original_openapi = application.openapi

    def openapi():
        schema = original_openapi()
        for path in ("/predict", "/compare", "/metrics", "/comparison"):
            schema["paths"][path]["post" if path in ("/predict", "/compare") else "get"]["responses"].pop("422", None)
        return schema

    application.openapi = openapi
    return application


app = create_app()
