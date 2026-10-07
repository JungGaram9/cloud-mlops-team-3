"""현재 버전의 신경망 학습·평가. 검증 손실로 가중치를 선택한다."""

import copy
import csv
import hashlib
import json
import platform
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch import nn
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                            confusion_matrix, f1_score, mean_absolute_error, precision_score,
                            recall_score, roc_auc_score)

from models.model import (ENCODING, FEATURES, FEATURE_SETS, MIN_ACCURACY, REPORTS_DIR,
                          ROOT, SEED, SENSOR_FEATURES, THRESHOLD, WEIGHTS_DIR,
                          OccupancyNetwork, SensorInput, encoded_features, input_vector,
                          pair_fingerprint)

MAX_EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 0.01


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    dates: list[datetime]
    summary: dict


def summarize(y: np.ndarray, dates: list[datetime]) -> dict:
    return {"rows": len(y), "class_counts": dict(Counter(str(int(value)) for value in y)),
            "first_timestamp": min(dates).isoformat(sep=" "),
            "last_timestamp": max(dates).isoformat(sep=" "),
            "duplicate_timestamps": len(dates) - len(set(dates))}


def load_dataset(path: Path) -> Dataset:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"date", "Occupancy", *SENSOR_FEATURES}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: 필수 컬럼이 없습니다.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: 데이터가 비어 있습니다.")
    values, labels, dates = [], [], []
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
    x, y = np.asarray(values, dtype=np.float32), np.asarray(labels, dtype=np.int64)
    summary = {**summarize(y, dates), "file": path.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return Dataset(x, y, dates, summary)


def split_training(source: Dataset) -> tuple[Dataset, Dataset]:
    order = sorted(range(len(source.y)), key=lambda index: source.dates[index])
    split = int(len(order) * 0.8)
    datasets = []
    for indices in (order[:split], order[split:]):
        if not indices or set(source.y[indices]) != {0, 1}:
            raise ValueError("시간순 학습·검증 구간에 두 정답 클래스가 모두 필요합니다.")
        dates = [source.dates[index] for index in indices]
        y = source.y[indices]
        datasets.append(Dataset(source.x[indices], y, dates, summarize(y, dates)))
    if max(datasets[0].dates) >= min(datasets[1].dates):
        raise ValueError("학습·검증 구간의 시점이 겹칩니다.")
    return datasets[0], datasets[1]


def audit_split(training: Dataset, test: Dataset) -> dict:
    shared_timestamps = len(set(training.dates) & set(test.dates))
    training_records = {(date, *row, label) for date, row, label in zip(training.dates, training.x, training.y)}
    test_records = {(date, *row, label) for date, row, label in zip(test.dates, test.x, test.y)}
    shared_records = len(training_records & test_records)
    if shared_timestamps or shared_records:
        raise ValueError("학습·평가 파일에 같은 시점 또는 동일한 기록이 있습니다.")
    return {"shared_timestamps": shared_timestamps, "shared_records": shared_records,
            "test_rows_before_training": sum(date < min(training.dates) for date in test.dates),
            "test_rows_after_training": sum(date > max(training.dates) for date in test.dates)}


def evaluate(labels: np.ndarray, values: np.ndarray) -> dict:
    predicted = (values >= THRESHOLD).astype(int)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
            "probability_mae": float(mean_absolute_error(labels, values)),
            "brier_score": float(brier_score_loss(labels, values)),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def require_accuracy(metrics: dict) -> None:
    if not np.isfinite(metrics["accuracy"]) or metrics["accuracy"] < MIN_ACCURACY:
        raise ValueError(f"조도 포함 모델의 평가 정확도가 최소 기준 {MIN_ACCURACY:.0%}에 미달했습니다.")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_network(training: Dataset, validation: Dataset) -> tuple[OccupancyNetwork, dict, dict]:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x_train = torch.from_numpy(training.x)
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, correction=0).clamp_min(1e-6)
    x_train = (x_train - mean) / scale
    x_valid = (torch.from_numpy(validation.x) - mean) / scale
    y_train = torch.from_numpy(training.y).float()
    y_valid = torch.from_numpy(validation.y).float()

    network = OccupancyNetwork(training.x.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
        network.train()
        optimizer.zero_grad()
        loss = criterion(network(x_train), y_train)
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(network(x_valid), y_valid))
        if not np.isfinite(validation_loss):
            raise ValueError("학습 중 검증 손실이 유한한 값이 아닙니다.")
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    network.load_state_dict(best_state)
    network.eval()
    training_info = {
        "max_epochs": MAX_EPOCHS, "epochs_run": epoch, "best_epoch": best_epoch,
        "patience": PATIENCE, "learning_rate": LEARNING_RATE,
        "best_validation_loss": best_loss, "selection": "시간순 검증 구간의 BCE 손실 최소화",
    }
    return network, {"mean": mean, "scale": scale}, training_info


def probabilities(network: OccupancyNetwork, x: np.ndarray, scaler: dict) -> np.ndarray:
    with torch.inference_mode():
        values = (torch.from_numpy(x) - scaler["mean"]) / scaler["scale"]
        return torch.sigmoid(network(values)).numpy()


def train_variant(mode: str, source: Dataset, test: Dataset, audit: dict) -> tuple[dict, dict]:
    features = FEATURE_SETS[mode]
    columns = [encoded_features(FEATURES).index(name) for name in encoded_features(features)]
    source = Dataset(source.x[:, columns], source.y, source.dates, source.summary)
    test = Dataset(test.x[:, columns], test.y, test.dates, test.summary)
    training, validation = split_training(source)
    network, scaler, training_info = train_network(training, validation)
    test_probabilities = probabilities(network, test.x, scaler)
    metrics = evaluate(test.y, test_probabilities)
    rate = float(training.y.mean())
    metadata = {
        "format_version": 1, "framework": "pytorch", "feature_mode": mode,
        "encoding": ENCODING, "encoded_features": encoded_features(features),
        "architecture": f"MLP {len(columns)} → 16 → 8 → 1, ReLU",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(features), "threshold": THRESHOLD, "random_state": SEED,
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "numpy", "scikit-learn")},
        "training_data": source.summary,
        "training_split": training.summary, "validation_split": validation.summary, "training": training_info,
    }
    report = {
        **metadata, "test_data": test.summary, "data_audit": audit,
        "quality_gate": {"minimum_accuracy": MIN_ACCURACY, "passed": metrics["accuracy"] >= MIN_ACCURACY},
        "validation_metrics": evaluate(validation.y, probabilities(network, validation.x, scaler)),
        "model_metrics": metrics,
        "baseline": {"description": "학습 구간의 사용 비율을 모든 평가 행에 동일하게 반환",
                     "training_occupancy_rate": rate, "metrics": evaluate(test.y, np.full(len(test.y), rate))},
        "evaluation_notes": ["같은 파일을 시간순 80/20으로 나누고 각 모델의 검증 손실로 가중치를 선택합니다.",
                             "평가 정답으로 가중치·임계값을 조정하지 않습니다.",
                             "평가 파일에는 학습 기간 이전·이후 시점이 함께 있습니다.",
                             "Hour와 DayOfWeek는 CSV date에서 추출하며 학습·예측이 같은 주기 변환을 사용합니다."],
    }
    future_mask = np.array([date > max(source.dates) for date in test.dates])
    if len(set(test.y[future_mask])) == 2:
        report["future_subset_metrics"] = evaluate(test.y[future_mask], test_probabilities[future_mask])
    if mode == "with_light":
        require_accuracy(metrics)
    print(f"{mode}: 정확도 {metrics['accuracy']:.2%}, F1 {metrics['f1']:.4f}")
    return report, {**metadata, "state_dict": network.state_dict(), **scaler}


def save_model_pair(checkpoints: dict, reports: dict, weights_dir: Path = WEIGHTS_DIR,
                    reports_dir: Path = REPORTS_DIR) -> str:
    weights_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    signature = pair_fingerprint(checkpoints)
    directories = [path for path in weights_dir.glob("v*") if path.is_dir() and path.name[1:].isdigit()]
    directories.sort(key=lambda path: int(path.name[1:]), reverse=True)
    selected_version = None
    for directory in directories:
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
        primary, secondary = reports["with_light"], reports["without_light"]
        comparison = {
            "model_version": selected_version, "pair_fingerprint": signature,
            "test_rows": primary["test_data"]["rows"], "threshold": THRESHOLD, "modes": reports,
            "accuracy_difference_percentage_points":
                (primary["model_metrics"]["accuracy"] - secondary["model_metrics"]["accuracy"]) * 100,
            "notes": ["같은 모델 쌍은 재학습해도 기존 버전을 재사용하며 역대 가중치를 덮어쓰지 않습니다.",
                      "가중치·표준화·피처·입력 변환·임계값이 달라지면 새로운 모델 버전을 저장합니다.",
                      "조도 제외 모델은 조도 입력 없이 별도로 학습합니다.",
                      "평가 정확도는 고정된 정답 데이터의 지표이며 입력 한 건의 확률과 다릅니다."],
        }
        # 완성된 두 가중치와 보고서를 저장한 뒤 현재 모델 정보를 마지막에 교체한다.
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=weights_dir) as temporary:
            staging = Path(temporary) / selected_version
            staging.mkdir()
            for mode, filename in (("with_light", "model.pt"), ("without_light", "model_without_light.pt")):
                torch.save(checkpoints[mode], staging / filename)
            staging.rename(weights_dir / selected_version)
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=reports_dir) as temporary:
            staging = Path(temporary) / selected_version
            staging.mkdir()
            write_json(staging / "metrics.json", primary)
            write_json(staging / "light_comparison.json", comparison)
            staging.rename(reports_dir / selected_version)
        print(f"새 모델 쌍 {selected_version} 저장: {signature[:12]}")
    else:
        print(f"모델 쌍이 동일해 {selected_version} 재사용: 기존 가중치·보고서 보존")
    current = {"version": selected_version, "pair_fingerprint": signature}
    path = weights_dir / "current.json"
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != current:
        temporary = weights_dir / ".current.pending.json"
        write_json(temporary, current)
        temporary.replace(path)
    return selected_version


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = WEIGHTS_DIR / ".training.lock"
    try:
        # 프로세스가 중단돼도 OS 잠금은 해제되어 다음 실행이 막히지 않는다.
        with FileLock(lock_path, timeout=0):
            source = load_dataset(ROOT / "data" / "training_dataset.csv")
            test = load_dataset(ROOT / "data" / "test_dataset.csv")
            audit = audit_split(source, test)
            reports, checkpoints = {}, {}
            for mode in FEATURE_SETS:
                reports[mode], checkpoints[mode] = train_variant(mode, source, test, audit)
            save_model_pair(checkpoints, reports)
    except Timeout as error:
        raise RuntimeError("다른 학습이 실행 중입니다. 완료 후 다시 실행하세요.") from error


if __name__ == "__main__":
    main()
