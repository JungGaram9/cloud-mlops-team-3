"""실행별 JSONL 로그. 쓰기는 프로세스 간 잠금으로 보호하고 실패는 stderr로 알린다."""

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"


class RuntimeLogger:
    def __init__(self, root=None, run_id=None, request_id=None, started_at=None):
        self.root = Path(root) if root is not None else LOGS_DIR
        started_at = started_at or datetime.now(timezone.utc)
        self.run_id = run_id or f"{started_at.astimezone(timezone.utc):%Y%m%dT%H%M%S_%fZ}-{uuid.uuid4().hex[:12]}"
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}_[0-9]{6}Z-[0-9a-f]{12}", self.run_id):
            self.report_error(ValueError("실행 로그 ID 형식이 잘못되었습니다."))
            self.run_id = f"{started_at:%Y%m%dT%H%M%S_%fZ}-{uuid.uuid4().hex[:12]}"
        self.request_id = request_id
        self.directory = self.root / self.run_id
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            for name in ("api.jsonl", "model.jsonl"):
                with (self.directory / name).open("a", encoding="utf-8"):
                    pass
        except Exception as error:
            self.report_error(error)

    @classmethod
    def for_training(cls, record=None):
        record = record or {}
        return cls(root=record.get("log_root") or os.environ.get("CLASSROOM_LOG_ROOT"),
                   run_id=record.get("log_run_id") or os.environ.get("CLASSROOM_LOG_RUN_ID"),
                   request_id=record.get("request_id") or os.environ.get("CLASSROOM_LOG_REQUEST_ID"))

    @staticmethod
    def report_error(error):
        print(f"[runtime_logging] 로그 저장 실패: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    def write(self, channel, event_type, **fields):
        try:
            if channel not in ("api", "model"):
                raise ValueError("알 수 없는 로그 종류입니다.")
            record = {**fields, "schema_version": 1, "event_type": event_type,
                      "run_id": self.run_id, "request_id": fields.get("request_id", self.request_id),
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
            line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{channel}.jsonl"
            with FileLock(str(path) + ".lock", timeout=5):
                with path.open("a", encoding="utf-8", newline="") as output:
                    output.write(line)
                    output.flush()
            return True
        except Exception as error:
            self.report_error(error)
            return False

    def inference(self, *, request_id, sensors, version, mode, features, threshold, fingerprint, prediction,
                  source="experiment", inference_id=None, context=None):
        inference_id = inference_id or uuid.uuid4().hex
        self.write("model", "inference", event="prediction", inference_id=inference_id, source=source,
                   request_id=request_id, sensors=sensors, model_version=version, mode=mode,
                   features=features, threshold=threshold, pair_fingerprint=fingerprint,
                   prediction=prediction, label_source="model_prediction", **(context or {}))
        return inference_id

    def epoch(self, job_id, mode, point):
        # 배치 콜백의 None은 기록하지 않고 에포크별 소규모 통계만 남긴다.
        if point is not None:
            self.write("model", "training", event="epoch", job_id=job_id, mode=mode, **point)

    def training_finished(self, record, reports_dir):
        result = None
        if record.get("status") == "completed" and record.get("version"):
            try:
                report = json.loads((Path(reports_dir) / record["version"] / "light_comparison.json").read_text(encoding="utf-8"))
                result = {"pair_fingerprint": report.get("pair_fingerprint"), "models": {
                    mode: {key: variant.get(key) for key in ("features", "threshold", "test_data", "model_metrics", "source_metrics")}
                    for mode, variant in report["modes"].items()}}
            except Exception as error:
                self.report_error(error)
        self.write("model", "training", event="finished", job_id=record.get("id"), status=record.get("status"),
                   model_version=record.get("version"), error=record.get("error"), error_type=record.get("error_type"),
                   result=result)
