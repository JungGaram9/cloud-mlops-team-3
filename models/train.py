"""현재 버전의 신경망 학습·평가. 검증 손실로 가중치를 선택한다."""

import copy
import csv
import argparse
import hashlib
import io
import json
import os
import platform
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch import nn
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                            confusion_matrix, f1_score, mean_absolute_error, precision_score,
                            recall_score, roc_auc_score)
from scripts.runtime_logging import RuntimeLogger

from models.model import (ENCODING, FEATURES, FEATURE_SETS, MIN_ACCURACY, REPORTS_DIR,
                          ROOT, SEED, SENSOR_FEATURES, THRESHOLD, WEIGHTS_DIR,
                          OccupancyNetwork, SensorInput, encoded_features, input_vector,
                          pair_fingerprint)

MAX_EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 0.01
JOBS_DIR = REPORTS_DIR / "jobs"
SPLIT_STRATEGY = "source_chronological"


class TrainingConfig(BaseModel):
    """웹과 학습 프로세스가 공유하는 학습 설정 및 CPU 실행 한도."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(default="새 학습 실험", min_length=1, max_length=80)
    features: list[Literal["Temperature", "Humidity", "Light", "CO2", "Hour", "DayOfWeek"]] = Field(
        default_factory=lambda: list(FEATURES), min_length=1, max_length=6)
    epochs: int = Field(default=200, ge=1, le=500)
    learning_rate: float = Field(default=0.01, ge=0.00001, le=0.1)
    batch_size: int = Field(default=0, ge=0, le=8192, description="0은 전체 학습 행을 한 배치로 사용")
    hidden_layers: list[int] = Field(default_factory=lambda: [16, 8], min_length=1, max_length=3)
    activation: Literal["relu", "tanh"] = "relu"
    optimizer: Literal["adam", "sgd"] = "adam"
    dropout: float = Field(default=0, ge=0, le=0.5)
    weight_decay: float = Field(default=0, ge=0, le=0.1)
    validation_ratio: float = Field(default=0.2, ge=0.1, le=0.4)
    patience: int = Field(default=30, ge=0, le=100, description="0은 조기 종료 끄기")
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)
    seed: int = Field(default=42, ge=0, le=2147483647)
    normalize: bool = True
    balance_classes: bool = False
    compare_light: bool = True
    make_default: bool = False

    @field_validator("name")
    @classmethod
    def trim_name(cls, value):
        if not value.strip():
            raise ValueError("실험 이름을 입력하세요.")
        return value.strip()

    @field_validator("features")
    @classmethod
    def canonical_features(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("같은 피처를 두 번 선택할 수 없습니다.")
        return [name for name in FEATURES if name in value]

    @field_validator("hidden_layers")
    @classmethod
    def check_layers(cls, value):
        if any(not 4 <= width <= 128 for width in value):
            raise ValueError("각 은닉층의 너비는 4~128이어야 합니다.")
        return value

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, value):
        if value and value < 32:
            raise ValueError("배치 크기는 전체 배치(0) 또는 32~8192이어야 합니다.")
        return value


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    dates: list[datetime]
    summary: dict
    sources: list[str] | None = None
    raw: list[dict] | None = None

    def select(self, indices, columns=None):
        dates = [self.dates[index] for index in indices]
        sources = [self.sources[index] for index in indices] if self.sources is not None else ["legacy"] * len(indices)
        x = self.x[indices]
        return Dataset(x if columns is None else x[:, columns], self.y[indices], dates,
                       summarize(self.y[indices], dates, sources), sources,
                       [self.raw[index] for index in indices] if self.raw is not None else None)


def summarize(y: np.ndarray, dates: list[datetime], sources=None) -> dict:
    result = {"rows": len(y), "class_counts": {"0": int(sum(y == 0)), "1": int(sum(y == 1))},
              "first_timestamp": min(dates).isoformat(sep=" ") if dates else None,
              "last_timestamp": max(dates).isoformat(sep=" ") if dates else None,
              "duplicate_timestamps": len(dates) - len(set(dates))}
    if sources is not None:
        result["sources"] = {name: summarize(y[[i for i, value in enumerate(sources) if value == name]],
                                            [date for date, value in zip(dates, sources) if value == name])
                             for name in sorted(set(sources))}
    return result


def load_dataset(path: Path) -> Dataset:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"date", "Occupancy", *SENSOR_FEATURES}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: 필수 컬럼이 없습니다.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: 데이터가 비어 있습니다.")
    values, labels, dates, raw = [], [], [], []
    for line, row in enumerate(rows, start=2):
        try:
            date = datetime.fromisoformat(row["date"])
            sensors = SensorInput(**{name: float(row[name]) for name in SENSOR_FEATURES},
                                  Hour=date.hour, DayOfWeek=date.weekday())
            label = float(row["Occupancy"])
            if label not in (0.0, 1.0):
                raise ValueError("정답은 0 또는 1이어야 합니다.")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name} {line}행: 센서값·정답·날짜를 확인하세요.") from error
        values.append(input_vector(sensors))
        labels.append(int(label))
        dates.append(date)
        raw.append(sensors.model_dump())
    x, y = np.asarray(values, dtype=np.float32), np.asarray(labels, dtype=np.int64)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = ["legacy"] * len(rows)
    metadata_path = path.parent / "dataset_sources.json"
    source_digest = None
    if metadata_path.is_file():
        content = metadata_path.read_bytes()
        recorded = json.loads(content)[path.name]
        if recorded["sha256"] != digest or len(recorded["sources"]) != len(rows):
            raise ValueError(f"{path.name}: 출처 메타데이터와 CSV 해시·행수가 일치하지 않습니다.")
        sources = recorded["sources"]
        if any(name not in ("legacy", "Room_A", "Room_B") for name in sources):
            raise ValueError("알 수 없는 데이터 출처입니다.")
        source_digest = hashlib.sha256(content).hexdigest()
    summary = {**summarize(y, dates, sources), "file": path.relative_to(ROOT).as_posix(),
               "sha256": digest, "source_metadata_sha256": source_digest}
    return Dataset(x, y, dates, summary, sources, raw)


def fit_input_limits(training: Dataset, features) -> dict:
    """float32 변환 전의 실제 fit 관측값만 영속화한다."""
    if training.raw is None:
        raise ValueError("학습 원본 정밀도 입력값이 없습니다.")
    profiles = {}
    for name in features:
        values = [row[name] for row in training.raw]
        profiles[name] = {"minimum": min(values), "maximum": max(values)}
        if name in ("Hour", "DayOfWeek"):
            profiles[name]["values"] = sorted(set(values))
    return {"available": True, "rows": len(training.y), "features": profiles,
            "first_timestamp": training.summary["first_timestamp"],
            "last_timestamp": training.summary["last_timestamp"],
            "sources": training.summary.get("sources", {}), "split_strategy": SPLIT_STRATEGY}


def training_input_limits(models: dict) -> dict:
    """저장된 데이터 해시·학습 구간을 확인하고 원본 정밀도의 입력 범위를 복원한다."""
    limits = {}
    sources = {}
    for mode, model in models.items():
        try:
            if model.get("input_limits") is not None:
                limits[mode] = copy.deepcopy(model["input_limits"])
                continue
            if model.get("split_strategy") == SPLIT_STRATEGY:
                raise ValueError("출처별 분할 모델의 저장된 fit 입력범위가 없습니다.")
            data = model["training_data"]
            split = model["training_split"]
            if not data.get("file"):
                raise ValueError("이전 다중 데이터 모델의 학습 범위는 현재 통합 CSV로 검증할 수 없습니다.")
            path = (ROOT / data["file"]).resolve()
            if path.parent != (ROOT / "data").resolve() or path.suffix != ".csv":
                raise ValueError("기록된 원본 학습 CSV를 확인할 수 없습니다.")
            if path not in sources:
                content = path.read_bytes()
                rows = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
                sources[path] = (hashlib.sha256(content).hexdigest(),
                                 [(datetime.fromisoformat(row["date"]), row) for row in rows])
            source_hash, records = sources[path]
            if source_hash != data["sha256"]:
                raise ValueError("원본 CSV가 모델에 기록된 학습 데이터 해시와 다릅니다.")
            first = datetime.fromisoformat(split["first_timestamp"])
            last = datetime.fromisoformat(split["last_timestamp"])
            training_rows = [(date, row) for date, row in records if first <= date <= last]
            if (len(training_rows) != split["rows"] or not training_rows
                    or min(date for date, _ in training_rows) != first
                    or max(date for date, _ in training_rows) != last):
                raise ValueError("기록된 학습 구간의 시각·행수를 재구성할 수 없습니다.")
            observed = [SensorInput(**{name: float(row[name]) for name in SENSOR_FEATURES},
                                    Hour=date.hour, DayOfWeek=date.weekday()).model_dump()
                        for date, row in training_rows]
            features = {}
            for name in model["features"]:
                values = [row[name] for row in observed]
                features[name] = {"minimum": min(values), "maximum": max(values)}
                if name in ("Hour", "DayOfWeek"):
                    features[name]["values"] = sorted(set(values))
            limits[mode] = {"available": True, "rows": len(training_rows),
                            "first_timestamp": split["first_timestamp"], "last_timestamp": split["last_timestamp"],
                            "features": features}
        except (OSError, KeyError, ValueError, TypeError) as error:
            limits[mode] = {"available": False, "features": {},
                            "error": f"입력 제한을 준비할 수 없습니다: {error}"}
    return limits


def split_training(source: Dataset, validation_ratio: float = 0.2) -> tuple[Dataset, Dataset]:
    groups = source.sources if source.sources is not None else ["legacy"] * len(source.y)
    fit_indices, validation_indices = [], []
    for name in sorted(set(groups)):
        order = sorted((i for i, value in enumerate(groups) if value == name), key=lambda i: source.dates[i])
        split = int(len(order) * (1 - validation_ratio))
        while 0 < split < len(order) and source.dates[order[split - 1]] == source.dates[order[split]]:
            split -= 1
        if not 0 < split < len(order):
            raise ValueError(f"{name}: 같은 시각 그룹을 유지하면서 fit/validation을 나눌 수 없습니다.")
        fit_indices.extend(order[:split])
        validation_indices.extend(order[split:])
    training, validation = source.select(fit_indices), source.select(validation_indices)
    if any(set(part.y) != {0, 1} for part in (training, validation)):
        raise ValueError("전체 fit·validation 구간에 두 정답 클래스가 모두 필요합니다.")
    return training, validation


def training_plan(source: Dataset, test: Dataset, validation_ratio: float) -> dict:
    plan = {"ready": False, "errors": [], "warnings": [], "split_strategy": SPLIT_STRATEGY,
            "validation_ratio": validation_ratio, "test": test.summary}
    try:
        audit_split(source, test)
        training, validation = split_training(source, validation_ratio)
        plan.update(fit=training.summary, validation=validation.summary)
        names = set(source.sources or ["legacy"]) | set(test.sources or ["legacy"])
        for name in sorted(names):
            for label, part in (("fit", training), ("validation", validation), ("test", test)):
                stats = part.summary["sources"].get(name)
                if stats is None:
                    if label == "test":
                        plan["warnings"].append(f"{name}: 평가 행이 없습니다.")
                    else:
                        plan["errors"].append(f"{name}: {label} 행이 없습니다.")
                elif not all(stats["class_counts"].get(value, 0) for value in ("0", "1")):
                    plan["warnings"].append(f"{name} {label}: 단일 클래스이며 일부 평가 지표가 제한됩니다.")
        plan["ready"] = not plan["errors"]
    except ValueError as error:
        plan["errors"].append(str(error))
    return plan


def audit_split(training: Dataset, test: Dataset) -> dict:
    training_sources = training.sources or ["legacy"] * len(training.y)
    test_sources = test.sources or ["legacy"] * len(test.y)
    shared_timestamps = len(set(zip(training_sources, training.dates)) & set(zip(test_sources, test.dates)))
    training_records = {(name, date, *row, label) for name, date, row, label in
                        zip(training_sources, training.dates, training.x, training.y)}
    test_records = {(name, date, *row, label) for name, date, row, label in zip(test_sources, test.dates, test.x, test.y)}
    shared_records = len(training_records & test_records)
    if shared_timestamps or shared_records:
        raise ValueError("학습·평가 파일에 같은 시점 또는 동일한 기록이 있습니다.")
    boundaries = {name: (min(date for group, date in zip(training_sources, training.dates) if group == name),
                         max(date for group, date in zip(training_sources, training.dates) if group == name))
                  for name in set(training_sources)}
    return {"shared_timestamps": shared_timestamps, "shared_records": shared_records,
            "timestamp_scope": "source_and_timestamp",
            "test_rows_before_training": sum(name in boundaries and date < boundaries[name][0]
                                             for name, date in zip(test_sources, test.dates)),
            "test_rows_after_training": sum(name in boundaries and date > boundaries[name][1]
                                            for name, date in zip(test_sources, test.dates))}


def evaluate(labels: np.ndarray, values: np.ndarray, threshold: float = THRESHOLD) -> dict:
    predicted = (values >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, values)) if len(set(labels)) == 2 else None,
            "average_precision": float(average_precision_score(labels, values)) if len(set(labels)) == 2 else None,
            "probability_mae": float(mean_absolute_error(labels, values)),
            "brier_score": float(brier_score_loss(labels, values)),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def require_accuracy(metrics: dict) -> None:
    if not np.isfinite(metrics["accuracy"]) or metrics["accuracy"] < MIN_ACCURACY:
        raise ValueError(f"조도 포함 모델의 평가 정확도가 최소 기준 {MIN_ACCURACY:.0%}에 미달했습니다.")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 상태 조회와 겹쳐도 잘린 JSON을 읽지 않도록 같은 디렉터리에서 원자적으로 교체한다.
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def train_network(training: Dataset, validation: Dataset, config: TrainingConfig | None = None,
                  progress=None) -> tuple[OccupancyNetwork, dict, dict]:
    config = config or TrainingConfig()
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x_train = torch.from_numpy(training.x)
    # 검증·추론에 적용할 통계는 학습 구간에서만 계산한다.
    if config.normalize:
        mean = x_train.mean(dim=0)
        scale = x_train.std(dim=0, correction=0).clamp_min(1e-6)
    else:
        mean = torch.zeros(x_train.shape[1])
        scale = torch.ones(x_train.shape[1])
    x_train = (x_train - mean) / scale
    x_valid = (torch.from_numpy(validation.x) - mean) / scale
    y_train = torch.from_numpy(training.y).float()
    y_valid = torch.from_numpy(validation.y).float()

    network = OccupancyNetwork(training.x.shape[1], config.hidden_layers, config.activation, config.dropout)
    optimizer_type = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_type(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    positive_weight = ((y_train == 0).sum() / (y_train == 1).sum()) if config.balance_classes else None
    training_criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    # 검증 손실은 정답 분포 그대로 측정하여 학습 가중치에 따른 척도 차이를 피한다.
    validation_criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state = None
    history = []
    batch_size = config.batch_size or len(x_train)
    for epoch in range(1, config.epochs + 1):
        network.train()
        batch_order = torch.randperm(len(x_train)) if batch_size < len(x_train) else torch.arange(len(x_train))
        training_loss_sum = 0.0
        for start in range(0, len(batch_order), batch_size):
            indices = batch_order[start:start + batch_size]
            optimizer.zero_grad()
            loss = training_criterion(network(x_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                raise ValueError("학습 손실이 유한하지 않습니다. 학습률이나 표준화를 조정하세요.")
            loss.backward()
            optimizer.step()
            training_loss_sum += float(loss.detach()) * len(indices)
            if progress:
                progress(None)

        network.eval()
        with torch.inference_mode():
            logits = network(x_valid)
            validation_loss = float(validation_criterion(logits, y_valid))
            accuracy = float(((torch.sigmoid(logits) >= config.threshold).int() == y_valid).float().mean())
        if not np.isfinite(validation_loss):
            raise ValueError("학습 중 검증 손실이 유한한 값이 아닙니다.")

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale_epochs += 1
        point = {"epoch": epoch, "training_loss": training_loss_sum / len(x_train),
                 "validation_loss": validation_loss, "validation_accuracy": accuracy}
        history.append(point)
        if progress:
            progress(point)
        if config.patience and stale_epochs >= config.patience:
            break
    # 조기 종료를 꺼도 마지막 에포크가 아닌 검증 손실 최소 가중치를 선택한다.
    network.load_state_dict(best_state)
    network.eval()
    training_info = {
        **config.model_dump(), "max_epochs": config.epochs, "epochs_run": epoch, "best_epoch": best_epoch,
        "effective_batch_size": min(batch_size, len(x_train)), "history": history,
        "best_validation_loss": best_loss, "selection": "출처별 시간순 검증 구간의 BCE 손실 최소화",
    }
    return network, {"mean": mean, "scale": scale}, training_info


def probabilities(network: OccupancyNetwork, x: np.ndarray, scaler: dict) -> np.ndarray:
    with torch.inference_mode():
        values = (torch.from_numpy(x) - scaler["mean"]) / scaler["scale"]
        return torch.sigmoid(network(values)).numpy()


def train_variant(mode: str, source: Dataset, test: Dataset, audit: dict,
                  config: TrainingConfig | None = None, features=None, progress=None) -> tuple[dict, dict]:
    config = config or TrainingConfig()
    features = tuple(features or FEATURE_SETS[mode])
    all_encoded_features = encoded_features(FEATURES)
    selected_encoded_features = encoded_features(features)
    columns = [all_encoded_features.index(name) for name in selected_encoded_features]
    source = Dataset(source.x[:, columns], source.y, source.dates, source.summary, source.sources, source.raw)
    test = Dataset(test.x[:, columns], test.y, test.dates, test.summary, test.sources, test.raw)
    training, validation = split_training(source, config.validation_ratio)
    network, scaler, training_info = train_network(training, validation, config, progress)
    test_probabilities = probabilities(network, test.x, scaler)
    metrics = evaluate(test.y, test_probabilities, config.threshold)
    rate = float(training.y.mean())
    metadata = {
        "format_version": 1, "framework": "pytorch", "feature_mode": mode,
        "encoding": ENCODING, "encoded_features": selected_encoded_features,
        "architecture": "MLP " + " → ".join(str(width) for width in [len(columns), *config.hidden_layers, 1]) + f", {config.activation}",
        "network_config": {"hidden_layers": config.hidden_layers, "activation": config.activation, "dropout": config.dropout},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(features), "threshold": config.threshold, "random_state": config.seed,
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "numpy", "scikit-learn")},
        "training_data": source.summary,
        "test_data": test.summary, "split_strategy": SPLIT_STRATEGY,
        "input_limits": fit_input_limits(training, features),
        "training_split": training.summary, "validation_split": validation.summary, "training": training_info,
    }
    report = {
        **metadata, "test_data": test.summary, "data_audit": audit,
        "quality_gate": {"minimum_accuracy": MIN_ACCURACY, "passed": metrics["accuracy"] >= MIN_ACCURACY},
        "validation_metrics": evaluate(validation.y, probabilities(network, validation.x, scaler), config.threshold),
        "model_metrics": metrics,
        "baseline": {"description": "학습 구간의 사용 비율을 모든 평가 행에 동일하게 반환",
                     "training_occupancy_rate": rate, "metrics": evaluate(test.y, np.full(len(test.y), rate), config.threshold)},
        "evaluation_notes": ["출처별 시간순 fit/validation 분할이며, 평가 파일은 두 구간에 포함하지 않습니다.",
                             "평가 정답으로 가중치·임계값을 조정하지 않습니다.",
                             "평가 파일에는 학습 기간 이전·이후 시점이 함께 있습니다.",
                             "Hour와 DayOfWeek는 CSV date에서 추출하며 학습·예측이 같은 주기 변환을 사용합니다."],
    }
    test_groups = test.sources or ["legacy"] * len(test.y)
    report["source_metrics"] = {}
    for name in sorted(set(test_groups) | set(source.sources or ["legacy"])):
        indices = [i for i, value in enumerate(test_groups) if value == name]
        report["source_metrics"][name] = {
            "rows": len(indices), "class_counts": {"0": int(sum(test.y[indices] == 0)), "1": int(sum(test.y[indices] == 1))},
            "metrics": evaluate(test.y[indices], test_probabilities[indices], config.threshold) if indices else None}
    last_times = {name: stats["last_timestamp"] for name, stats in source.summary["sources"].items()}
    future_mask = np.array([name in last_times and date > datetime.fromisoformat(last_times[name])
                            for name, date in zip(test_groups, test.dates)])
    if future_mask.any():
        report["future_subset_metrics"] = evaluate(test.y[future_mask], test_probabilities[future_mask], config.threshold)
    print(f"{mode}: 정확도 {metrics['accuracy']:.2%}, F1 {metrics['f1']:.4f}")
    return report, {**metadata, "state_dict": network.state_dict(), **scaler}


def write_training_report(directory: Path, training_report: dict) -> None:
    write_json(directory / "training.json", training_report)
    model_version = training_report["model_version"]
    lines = [f"# {model_version} 학습 보고서", "", f"실험: {training_report['config']['name']}",
             f"학습 작업: {training_report.get('job_id') or '직접 실행'}", "",
             "## 학습 설정", "", "```json",
             json.dumps(training_report["config"], ensure_ascii=False, indent=2), "```", "",
             "## 평가", "", "| 모델 | 피처 | 정확도 | 정밀도 | 재현율 | F1 | 최적 에포크 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for mode, report in training_report["models"].items():
        metrics = report["model_metrics"]
        label = "선택 모델" if mode == "with_light" else "조도 제외 비교 모델"
        lines.append(f"| {label} | {', '.join(report['features'])} | {metrics['accuracy']:.2%} | "
                     f"{metrics['precision']:.2%} | {metrics['recall']:.2%} | {metrics['f1']:.4f} | "
                     f"{report['training']['best_epoch']} |")
    primary = next(iter(training_report["models"].values()))
    lines.extend(["", "## 출처별 분할", "", "| 출처 | fit | validation | test |", "|---|---:|---:|---:|"])
    for name in primary["training_split"].get("sources", {}):
        counts = [part.get("sources", {}).get(name, {}).get("rows", 0)
                  for part in (primary["training_split"], primary["validation_split"], primary["test_data"])]
        lines.append(f"| {name} | {' | '.join(str(count) for count in counts)} |")
    for mode, report in training_report["models"].items():
        lines.extend(["", f"## {mode} 출처별 평가", "", "| 출처 | 행수 | 비어 있음 / 사용 중 | 정밀도 | 재현율 | F1 |",
                      "|---|---:|---|---:|---:|---:|"])
        for name, result in report.get("source_metrics", {}).items():
            metrics = result["metrics"]
            values = " | ".join(f"{metrics[key]:.4f}" if metrics else "기록 없음" for key in ("precision", "recall", "f1"))
            classes = result["class_counts"]
            lines.append(f"| {name} | {result['rows']} | {classes['0']} / {classes['1']} | {values} |")
    lines.extend(["", "검증 손실이 가장 낮은 가중치를 선택했습니다. 손실 곡선과 데이터 해시는 training.json에 보존합니다.",
                  "출처별 시간순 fit/validation 분할을 사용합니다. 출처는 신경망 입력 피처가 아닙니다.",
                  f"평가 CSV SHA-256: {next(iter(training_report['models'].values()))['test_data']['sha256']}",
                  "전체 평가 파일은 학습 기간 이전·이후 기록을 포함하므로 순수한 미래 예측 성능이 아닙니다.",
                  "피처 제거·설정 변경으로 85% 미만이 되어도 실험 모델을 보존합니다."])
    (directory / "training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_model_pair(checkpoints: dict, reports: dict, weights_dir: Path = WEIGHTS_DIR,
                    reports_dir: Path = REPORTS_DIR, *, force_new: bool = False,
                    make_default: bool = True, training_report: dict | None = None) -> str:
    weights_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    signature = pair_fingerprint(checkpoints)
    directories = [path for path in weights_dir.glob("v*") if path.is_dir() and path.name[1:].isdigit()]
    directories.sort(key=lambda path: int(path.name[1:]), reverse=True)
    selected_version = None
    reusable_directories = [] if force_new else directories
    for directory in reusable_directories:
        paths = {"with_light": directory / "model.pt", "without_light": directory / "model_without_light.pt"}
        if all(path.is_file() for path in paths.values()) and (reports_dir / directory.name / "light_comparison.json").is_file():
            existing = {mode: torch.load(path, map_location="cpu", weights_only=True) for mode, path in paths.items()}
            if pair_fingerprint(existing) == signature:
                selected_version = directory.name
                break
    if selected_version is None:
        numbers = [int(path.name[1:]) for path in (*directories, *reports_dir.glob("v*"))
                   if path.name[1:].isdigit()]
        selected_version = f"v{max(numbers, default=0) + 1}"
        for mode in checkpoints:
            checkpoints[mode]["version"] = selected_version
            reports[mode]["version"] = selected_version
        primary_model = reports["with_light"]
        comparison_model = reports.get("without_light")
        accuracy_difference = None
        if comparison_model:
            accuracy_difference = (primary_model["model_metrics"]["accuracy"] - comparison_model["model_metrics"]["accuracy"]) * 100
        comparison = {
            "model_version": selected_version, "pair_fingerprint": signature,
            "test_rows": primary_model["test_data"]["rows"], "threshold": primary_model["threshold"], "modes": reports,
            "accuracy_difference_percentage_points": accuracy_difference,
            "notes": ["웹에서 학습을 시작할 때마다 완료된 새 모델 버전을 기록하며 기존 가중치를 덮어쓰지 않습니다.",
                      "동일한 모델 식별값은 재현된 모델임을 이력에서 표시합니다.",
                      "조도 비교를 켜고 조도와 다른 피처를 선택하면 조도 제외 모델도 같은 설정으로 학습합니다.",
                      "평가 정확도는 고정된 정답 데이터의 지표이며 입력 한 건의 확률과 다릅니다."],
        }
        # 가중치와 모든 보고서를 임시 경로에 완성한 뒤 버전을 공개한다.
        with (tempfile.TemporaryDirectory(prefix=".pending-", dir=weights_dir) as temporary,
              tempfile.TemporaryDirectory(prefix=".pending-", dir=reports_dir) as report_temporary):
            weight_staging = Path(temporary) / selected_version
            weight_staging.mkdir()
            for mode, filename in (("with_light", "model.pt"), ("without_light", "model_without_light.pt")):
                if mode in checkpoints:
                    torch.save(checkpoints[mode], weight_staging / filename)
            report_staging = Path(report_temporary) / selected_version
            report_staging.mkdir()
            write_json(report_staging / "metrics.json", primary_model)
            write_json(report_staging / "light_comparison.json", comparison)
            if training_report:
                training_report = {**training_report, "model_version": selected_version,
                                   "pair_fingerprint": signature, "models": reports}
                write_training_report(report_staging, training_report)
            # 보고서 생성까지 성공한 결과만 외부에서 조회할 수 있는 버전 경로로 옮긴다.
            weight_staging.rename(weights_dir / selected_version)
            report_staging.rename(reports_dir / selected_version)
        print(f"새 모델 쌍 {selected_version} 저장: {signature[:12]}")
    else:
        print(f"모델 쌍이 동일해 {selected_version} 재사용: 기존 가중치·보고서 보존")
    current = {"version": selected_version, "pair_fingerprint": signature}
    path = weights_dir / "current.json"
    if make_default and (not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != current):
        write_json(path, current)
    return selected_version


class TrainingCancelled(Exception):
    """사용자가 중지한 실험은 가중치를 등록하지 않는다."""


def train_experiment(config: TrainingConfig, *, weights_dir=WEIGHTS_DIR, reports_dir=REPORTS_DIR,
                     job_id=None, progress=None, force_new=True, logger=None) -> str:
    weights_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = load_dataset(ROOT / "data/training_dataset.csv")
    test = load_dataset(ROOT / "data/test_dataset.csv")
    plan = training_plan(source, test, config.validation_ratio)
    logger = logger or RuntimeLogger.for_training()
    logger.write("model", "training", event="data_prepared", job_id=job_id, config=config.model_dump(),
                 seed=config.seed, training_data=source.summary, test_data=test.summary, data_plan=plan)
    if not plan["ready"]:
        raise ValueError(" / ".join(plan["errors"]))
    audit = audit_split(source, test)
    # with_light는 호환용 계약 키이며 실제 선택 피처에는 조도가 없을 수도 있다.
    primary_features = config.features
    variants = {"with_light": primary_features}
    if config.compare_light and "Light" in primary_features and len(primary_features) > 1:
        variants["without_light"] = [name for name in primary_features if name != "Light"]
    reports, checkpoints = {}, {}
    for index, (mode, features) in enumerate(variants.items()):
        callback = None
        if progress:
            # 각 모델의 에포크를 전체 실험의 진행률로 연결한다.
            def callback(point, mode=mode, index=index):
                progress(mode, index, len(variants), point)

        reports[mode], checkpoints[mode] = train_variant(mode, source, test, audit, config, features, callback)
    if progress:
        progress("saving", len(variants), len(variants), None)
    # 버전 번호 배정과 기본 모델 변경도 동시에 실행되지 않도록 보호한다.
    with FileLock(weights_dir / ".registration.lock", timeout=5):
        training_report = {
            "job_id": job_id,
            "config": config.model_dump(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split_strategy": SPLIT_STRATEGY, "data_plan": plan,
            "data_audit": audit,
        }
        return save_model_pair(checkpoints, reports, weights_dir, reports_dir, force_new=force_new,
                               make_default=config.make_default, training_report=training_report)


def run_training_job(job_id: str, jobs_dir=JOBS_DIR, weights_dir=WEIGHTS_DIR, reports_dir=REPORTS_DIR):
    path = jobs_dir / f"{job_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    logger = RuntimeLogger.for_training(record)
    record.update(log_run_id=logger.run_id, log_root=str(logger.root))
    config = TrainingConfig(**record["config"])
    logger.write("model", "training", event="started", job_id=job_id, config=config.model_dump(), seed=config.seed)
    record.update(status="running", pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat(),
                  progress=0, phase="데이터 준비", curves={})
    write_json(path, record)
    cancel_path = jobs_dir / f"{job_id}.cancel"
    last_written = 0.0

    def progress(mode, index, total, point):
        nonlocal last_written
        if cancel_path.is_file():
            raise TrainingCancelled()
        if point:
            logger.epoch(job_id, mode, point)
            record["curves"].setdefault(mode, []).append(point)
            completed_fraction = (index + point["epoch"] / config.epochs) / total
            record.update(phase="선택 모델 학습" if mode == "with_light" else "조도 제외 모델 학습",
                          epoch=point["epoch"], mode=mode,
                          progress=min(95, round(completed_fraction * 95, 1)))
        elif mode == "saving":
            logger.write("model", "training", event="saving", job_id=job_id)
            record.update(phase="평가 및 버전 저장", progress=98)
        if time.monotonic() - last_written > 0.4 or mode == "saving":
            write_json(path, record)
            last_written = time.monotonic()

    try:
        weights_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(weights_dir / ".training.lock", timeout=0):
            progress("loading", 0, 1, None)
            model_version = train_experiment(config, weights_dir=weights_dir, reports_dir=reports_dir,
                                             job_id=job_id, progress=progress, logger=logger)
        record.update(status="completed", version=model_version, progress=100, phase="학습 완료")
    except TrainingCancelled:
        record.update(status="cancelled", phase="학습 중지", error=None)
    except Exception as error:
        record.update(status="failed", phase="학습 실패", error=str(error), error_type=type(error).__name__)
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, record)
        logger.training_finished(record, reports_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="기본 모델 준비 또는 웹에서 요청한 학습 실행")
    parser.add_argument("--job-id")
    arguments = parser.parse_args()
    if arguments.job_id:
        if len(arguments.job_id) != 32 or any(char not in "0123456789abcdef" for char in arguments.job_id):
            raise ValueError("학습 작업 식별값이 잘못되었습니다.")
        run_training_job(arguments.job_id)
        return
    # Compose 재실행·이미지 빌드는 사용자 학습 이력을 추가하지 않는다.
    current_path = WEIGHTS_DIR / "current.json"
    if current_path.is_file():
        from models.model import current_version, load_model, weight_path
        model_version = current_version()
        report_path = REPORTS_DIR / model_version / "light_comparison.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for mode in report["modes"]:
            load_model(weight_path(model_version, mode))
        print(f"저장된 기본 모델 {model_version} 사용: 추가 학습 버전을 만들지 않습니다.")
        return
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = RuntimeLogger.for_training()
    direct_job_id = f"bootstrap-{uuid.uuid4().hex}"
    config = TrainingConfig()
    logger.write("model", "training", event="started", job_id=direct_job_id, config=config.model_dump(), seed=config.seed)
    try:
        with FileLock(WEIGHTS_DIR / ".training.lock", timeout=0):
            source = load_dataset(ROOT / "data/training_dataset.csv")
            test = load_dataset(ROOT / "data/test_dataset.csv")
            audit = audit_split(source, test)
            plan = training_plan(source, test, config.validation_ratio)
            logger.write("model", "training", event="data_prepared", job_id=direct_job_id,
                         training_data=source.summary, test_data=test.summary, data_plan=plan)
            if not plan["ready"]:
                raise ValueError(" / ".join(plan["errors"]))
            reports, checkpoints = {}, {}
            for mode in FEATURE_SETS:
                reports[mode], checkpoints[mode] = train_variant(mode, source, test, audit,
                    progress=lambda point, mode=mode: logger.epoch(direct_job_id, mode, point))
            require_accuracy(reports["with_light"]["model_metrics"])
            selected_version = save_model_pair(checkpoints, reports)
        logger.training_finished({"id": direct_job_id, "status": "completed", "version": selected_version}, REPORTS_DIR)
    except Exception as error:
        logger.training_finished({"id": direct_job_id, "status": "failed", "error": str(error),
                                  "error_type": type(error).__name__}, REPORTS_DIR)
        if isinstance(error, Timeout):
            raise RuntimeError("다른 학습이 실행 중입니다. 완료 후 다시 실행하세요.") from error
        raise


if __name__ == "__main__":
    main()
