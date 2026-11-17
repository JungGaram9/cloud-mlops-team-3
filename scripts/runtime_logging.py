"""실행별 JSONL 로그. 쓰기는 프로세스 간 잠금으로 보호하고 실패는 stderr로 알린다."""

import json
import hashlib
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

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _recover_tail(self, stream, path):
        """호출자는 같은 파일 잠금을 유지한다. 정상 경로는 끝의 1바이트만 확인한다."""
        stream.seek(0, 2)
        size = stream.tell()
        if not size:
            return
        stream.seek(-1, 2)
        if stream.read(1) == b"\n":
            return
        cursor, start = size, 0
        while cursor:
            length = min(4096, cursor)
            cursor -= length
            stream.seek(cursor)
            block = stream.read(length)
            boundary = block.rfind(b"\n")
            if boundary >= 0:
                start = cursor + boundary + 1
                break
        stream.seek(start)
        partial = stream.read()
        try:
            json.loads(partial)
        except (ValueError, UnicodeError):
            quarantine = path.with_name(path.name + ".partial-" + hashlib.sha256(partial).hexdigest() + ".bin")
            if quarantine.exists():
                preserved = quarantine.read_bytes()
                if not partial.startswith(preserved):
                    raise OSError("부분 로그 격리 파일이 원본과 다릅니다.")
                with quarantine.open("ab") as damaged:
                    remainder = partial[len(preserved):]
                    if remainder and damaged.write(remainder) != len(remainder):
                        raise OSError("부분 로그 원본을 모두 보존하지 못했습니다.")
                    damaged.flush()
                    os.fsync(damaged.fileno())
            else:
                with quarantine.open("xb") as damaged:
                    if damaged.write(partial) != len(partial):
                        raise OSError("부분 로그 원본을 모두 보존하지 못했습니다.")
                    damaged.flush()
                    os.fsync(damaged.fileno())
            stream.truncate(start)
            self.report_error(ValueError(f"부분 끝행 {start}바이트 이후를 {quarantine.name}에 보존했습니다."))
        else:
            stream.seek(0, 2)
            if stream.write(b"\n") != 1:
                raise OSError("로그 끝의 줄바꿈을 기록하지 못했습니다.")
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0, 2)

    def write(self, channel, event_type, **fields):
        try:
            if channel not in ("api", "model"):
                raise ValueError("알 수 없는 로그 종류입니다.")
            record = {**fields, "schema_version": 1, "event_type": event_type,
                      "run_id": self.run_id, "request_id": fields.get("request_id", self.request_id),
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
            line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{channel}.jsonl"
            with FileLock(str(path) + ".lock", timeout=5):
                with path.open("a+b") as output:
                    self._recover_tail(output, path)
                    before = self._signature(path)
                    cache = getattr(self, "_dedup_cache", {})
                    previous, keys = cache.get(channel, (None, set()))
                    if output.write(line) != len(line):
                        raise OSError("로그 행을 일부만 기록했습니다.")
                    output.flush()
                    if previous == before:
                        if key := fields.get("deduplication_key"):
                            keys.add(key)
                        cache[channel] = (self._signature(path), keys)
                    else:
                        cache.pop(channel, None)
                    self._dedup_cache = cache
            return True
        except Exception as error:
            self.report_error(error)
            return False

    def inference(self, *, request_id, sensors, version, mode, features, threshold, fingerprint, prediction,
                  source="experiment", inference_id=None, context=None, time_feature_policy=None):
        inference_id = inference_id or uuid.uuid4().hex
        self.write("model", "inference", event="prediction", inference_id=inference_id, source=source,
                   request_id=request_id, sensors=sensors, model_version=version, mode=mode,
                   features=features, threshold=threshold, pair_fingerprint=fingerprint,
                   prediction=prediction, label_source="model_prediction",
                   time_feature_policy=time_feature_policy or {"id": "unrecorded-legacy", "inference": "caller supplied Hour/DayOfWeek unchanged"},
                   **(context or {}))
        return inference_id

    def write_once(self, channel, event_type, key, **fields):
        """파이프라인 재시도는 정본 inference_id로 같은 JSONL 행의 중복 기록을 막는다."""
        if channel not in ("api", "model"):
            raise ValueError("알 수 없는 로그 종류입니다.")
        path = self.directory / f"{channel}.jsonl"
        with FileLock(str(path) + ".lock", timeout=5):
            with path.open("a+b") as stream:
                self._recover_tail(stream, path)
                cache = getattr(self, "_dedup_cache", {})
                signature = self._signature(path)
                previous, keys = cache.get(channel, (None, set()))
                if previous != signature:
                    keys = set()
                    stream.seek(0)
                    for line in stream:
                        try:
                            existing = json.loads(line).get("deduplication_key")
                            if existing:
                                keys.add(existing)
                        except (ValueError, UnicodeError, AttributeError):
                            continue
                if key in keys:
                    cache[channel] = (signature, keys)
                    self._dedup_cache = cache
                    return False
                record = {**fields, "event_type": event_type, "deduplication_key": key, "schema_version": 1,
                          "run_id": self.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                line = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                if stream.write(line) != len(line):
                    raise OSError("로그 행을 일부만 기록했습니다.")
                stream.flush()
                os.fsync(stream.fileno())
                keys.add(key)
                cache[channel] = (self._signature(path), keys)
                self._dedup_cache = cache
        return True

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
                    mode: {key: variant.get(key) for key in ("features", "threshold", "time_feature_policy", "test_data", "model_metrics", "source_metrics")}
                    for mode, variant in report["modes"].items()}}
            except Exception as error:
                self.report_error(error)
        self.write("model", "training", event="finished", job_id=record.get("id"), status=record.get("status"),
                   model_version=record.get("version"), error=record.get("error"), error_type=record.get("error_type"),
                   result=result, data_plan=record.get("data_plan"))
