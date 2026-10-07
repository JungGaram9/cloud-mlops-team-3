"""학습·추론에서 공유하는 입력 변환, PyTorch 구조, 모델 쌍과 버전 조회."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "models/weights"
REPORTS_DIR = ROOT / "reports"
CURRENT_PATH = WEIGHTS_DIR / "current.json"
SENSOR_FEATURES = ("Temperature", "Humidity", "Light", "CO2")
FEATURES = (*SENSOR_FEATURES, "Hour", "DayOfWeek")
FEATURE_SETS = {"with_light": FEATURES, "without_light": tuple(name for name in FEATURES if name != "Light")}
ENCODING = "cyclic-time-v1"
THRESHOLD, MIN_ACCURACY, SEED = 0.5, 0.85, 42
MOCK_SENSORS = {"Temperature": 22.5, "Humidity": 45.0, "Light": 450.0, "CO2": 900.0,
                "Hour": 14, "DayOfWeek": 2}


class SensorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False,
                              json_schema_extra={"examples": [MOCK_SENSORS]})
    Temperature: float = Field(ge=-50, le=60, description="기온: 섭씨 -50~60도")
    Humidity: float = Field(ge=0, le=100, description="상대습도: 0~100%")
    Light: float = Field(ge=0, le=1e6, description="조도: 0~1,000,000 lux")
    CO2: float = Field(gt=0, le=1e6, description="이산화탄소 농도: 0 초과~1,000,000 ppm")
    Hour: int = Field(ge=0, le=23, description="측정 시간: 0~23시, CSV date의 시와 동일 기준")
    DayOfWeek: int = Field(ge=0, le=6, description="측정 요일: 월요일 0, 화 1, 수 2, 목 3, 금 4, 토 5, 일 6")


def encoded_features(features: tuple | list, encoding: str = ENCODING) -> list[str]:
    names = []
    for name in features:
        if encoding == ENCODING and name == "Hour":
            names.extend(("HourSin", "HourCos"))
        elif encoding == ENCODING and name == "DayOfWeek":
            names.extend(("DaySin", "DayCos", "IsWeekend"))
        else:
            names.append(name)
    return names


def input_vector(sensors: SensorInput, features: tuple | list = FEATURES,
                 encoding: str = ENCODING) -> list[float]:
    # CSV 학습과 HTTP 추론이 같은 변환을 사용한다. 23시와 0시의 인접성을 보존한다.
    values = sensors.model_dump()
    values.update(HourSin=math.sin(2 * math.pi * sensors.Hour / 24),
                  HourCos=math.cos(2 * math.pi * sensors.Hour / 24),
                  DaySin=math.sin(2 * math.pi * sensors.DayOfWeek / 7),
                  DayCos=math.cos(2 * math.pi * sensors.DayOfWeek / 7),
                  IsWeekend=float(sensors.DayOfWeek >= 5))
    return [float(values[name]) for name in encoded_features(features, encoding)]


class OccupancyNetwork(nn.Module):
    def __init__(self, input_size: int = len(encoded_features(FEATURES)),
                 hidden_layers: list | tuple = (16, 8), activation: str = "relu", dropout: float = 0) -> None:
        super().__init__()
        layers = []
        for width in hidden_layers:
            layers.extend((nn.Linear(input_size, width), nn.ReLU() if activation == "relu" else nn.Tanh()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = width
        layers.append(nn.Linear(input_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def valid_version(version: str) -> str:
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("모델 버전은 v1, v2와 같은 형식이어야 합니다.")
    return version


def current_version() -> str:
    if not CURRENT_PATH.is_file():
        raise FileNotFoundError("현재 모델 정보가 없습니다. 먼저 Docker Compose의 trainer를 실행하세요.")
    return valid_version(json.loads(CURRENT_PATH.read_text(encoding="utf-8"))["version"])


def weight_path(version: str, mode: str = "with_light") -> Path:
    return WEIGHTS_DIR / valid_version(version) / ("model.pt" if mode == "with_light" else "model_without_light.pt")


@dataclass
class LoadedModel:
    network: OccupancyNetwork
    mean: torch.Tensor
    scale: torch.Tensor
    metadata: dict


def load_model(path: Path | None = None) -> LoadedModel:
    path = path or weight_path(current_version())
    if not path.is_file():
        raise FileNotFoundError("모델 파일이 없습니다. 먼저 Docker Compose의 trainer를 실행하세요.")
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    features = tuple(checkpoint.get("features", ()))
    encoding = checkpoint.get("encoding", "sensor-only-v1")
    if (not isinstance(checkpoint.get("version"), str) or checkpoint.get("framework") != "pytorch"
            or checkpoint.get("format_version") != 1 or encoding not in (ENCODING, "sensor-only-v1")
            or not features or len(features) != len(set(features)) or not set(features) <= set(FEATURES)):
        raise ValueError("체크포인트의 버전·프레임워크·입력 규격이 지원하는 모델과 다릅니다.")
    valid_version(checkpoint["version"])
    size = len(encoded_features(features, encoding))
    mean, scale = checkpoint["mean"], checkpoint["scale"]
    if (mean.shape != (size,) or scale.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError("체크포인트의 표준화 값이 잘못되었습니다.")
    threshold = checkpoint["threshold"]
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("체크포인트의 분류 임계값이 잘못되었습니다.")
    network_config = checkpoint.get("network_config", {"hidden_layers": [16, 8], "activation": "relu", "dropout": 0})
    if (not 1 <= len(network_config["hidden_layers"]) <= 3
            or any(type(width) is not int or not 4 <= width <= 128 for width in network_config["hidden_layers"])
            or network_config["activation"] not in ("relu", "tanh") or not 0 <= network_config["dropout"] <= 0.5):
        raise ValueError("체크포인트의 신경망 설정이 잘못되었습니다.")
    network = OccupancyNetwork(size, **network_config)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in network.parameters()):
        raise ValueError("체크포인트의 가중치가 유한한 값이 아닙니다.")
    network.eval()
    metadata = {key: value for key, value in checkpoint.items() if key not in ("state_dict", "mean", "scale")}
    return LoadedModel(network, mean, scale, metadata)


def predict_one(model: LoadedModel, sensors: dict) -> dict:
    validated = SensorInput(**sensors)
    values = torch.tensor([input_vector(validated, model.metadata["features"],
                                        model.metadata.get("encoding", "sensor-only-v1"))], dtype=torch.float32)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model.network((values - model.mean) / model.scale)).item())
    occupancy = int(probability >= model.metadata["threshold"])
    return {"model_version": model.metadata["version"], "occupancy": occupancy,
            "label": "사용 중" if occupancy else "비어 있음", "probability": probability}


def pair_fingerprint(checkpoints: dict) -> str:
    # 버전명·작성 시각·화면 코드는 제외하고 실제 예측을 결정하는 내용만 비교한다.
    digest = hashlib.sha256()
    for mode in sorted(checkpoints):
        checkpoint = checkpoints[mode]
        network_config = checkpoint.get("network_config", {"hidden_layers": [16, 8], "activation": "relu", "dropout": 0})
        contract = {"mode": mode, "features": checkpoint["features"],
                    "encoding": checkpoint.get("encoding", "sensor-only-v1"), "threshold": float(checkpoint["threshold"]),
                    "network_config": {**network_config, "dropout": float(network_config["dropout"])}}
        digest.update(json.dumps(contract, sort_keys=True).encode())
        tensors = {**checkpoint["state_dict"], "mean": checkpoint["mean"], "scale": checkpoint["scale"]}
        for name, tensor in sorted(tensors.items()):
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_versions(active_version: str | None = None) -> dict:
    active_version = active_version or current_version()
    pairs = []
    paths = list(REPORTS_DIR.glob("*/light_comparison.json"))
    paths.sort(key=lambda path: int(valid_version(path.parent.name)[1:]), reverse=True)
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        version = valid_version(report["model_version"])
        models = []
        for mode, variant in report["modes"].items():
            models.append({"mode": mode, "features": variant["features"],
                           "architecture": variant["architecture"], "metrics": variant["model_metrics"],
                           "quality_gate": variant["quality_gate"],
                           "training": {key: value for key, value in variant.get("training", {}).items() if key != "history"},
                           "validation_metrics": variant.get("validation_metrics"),
                           "future_subset_metrics": variant.get("future_subset_metrics"),
                           "checkpoint_exists": weight_path(version, mode).is_file()})
        ready = all(item["checkpoint_exists"] for item in models)
        signature = report.get("pair_fingerprint")
        if ready:
            signature = pair_fingerprint({mode: torch.load(weight_path(version, mode), map_location="cpu", weights_only=True)
                                          for mode in report["modes"]})
        training_path = REPORTS_DIR / version / "training.json"
        training_report = json.loads(training_path.read_text(encoding="utf-8")) if training_path.is_file() else None
        pairs.append({"version": version, "current": version == active_version,
                      "created_at_utc": report["modes"]["with_light"]["created_at_utc"],
                      "test_rows": report["test_rows"], "models": models,
                      "ready": ready, "fingerprint": signature,
                      "name": training_report["config"]["name"] if training_report else f"기존 {version} 모델",
                      "config": training_report["config"] if training_report else None,
                      "duration_seconds": training_report.get("duration_seconds") if training_report else None,
                      "job_id": training_report.get("job_id") if training_report else None,
                      "accuracy_difference_percentage_points": report.get("accuracy_difference_percentage_points")})
    identities = {}
    for pair in reversed(pairs):
        signature = pair["fingerprint"]
        pair["same_model_as"] = identities.get(signature) if signature else None
        if signature:
            identities.setdefault(signature, pair["version"])
    return {"current_version": active_version, "versions": pairs}
