"""고정된 가상 현장 관측의 예측·정답 연결·피드백 데이터셋. 실제 센서 정답과 구분한다."""

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from threading import RLock
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from models.model import ROOT, load_model, predict_one, weight_path
from scripts.runtime_logging import RuntimeLogger


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def identity(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    generation_id: str = Field(pattern=r"^dynamic-sensors-v1-[0-9a-f]{12}$")
    model_version: str = Field(pattern=r"^v[1-9][0-9]*$")
    mode: Literal["with_light", "without_light"] = "with_light"
    max_rows: int = Field(default=300, ge=1, le=3000)
    seed: int = Field(default=42, ge=0, le=2147483647)


class FeedbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_rows: int = Field(default=300, ge=1, le=3000)
    seed: int = Field(default=42, ge=0, le=2147483647)


def assessment(records):
    labelled = [row for row in records if row.get("truth") is not None]
    matrix = [[0, 0], [0, 0]]
    for row in labelled:
        matrix[row["truth"]][row["prediction"]["occupancy"]] += 1
    tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    divide = lambda a, b: a / b if b else None
    return {"rows": len(records), "labelled_rows": len(labelled), "unlabelled_rows": len(records) - len(labelled),
            "available": bool(labelled), "single_class": len({row["truth"] for row in labelled}) == 1,
            "confusion_matrix": matrix, "errors": fp + fn,
            "accuracy": divide(tn + tp, len(labelled)), "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "f1": divide(2 * tp, 2 * tp + fp + fn),
            "source": "simulation_verified_virtual_people", "final_real_test": False,
            "natural_prevalence": False}


def sampling_coverage(available, selected):
    from scripts.simulate import EVENT_KINDS, FAULT_KINDS
    result = {}
    for partition in sorted({row["partition"] for row in available}):
        eligible = [row for row in available if row["partition"] == partition]
        included = [row for row in selected if row["partition"] == partition]
        groups = {}
        for prefix, field, kinds in (("event", "events", EVENT_KINDS), ("fault", "sensor_faults", FAULT_KINDS)):
            for kind in kinds:
                total = sum(kind in row.get(field, []) for row in eligible)
                count = sum(kind in row.get(field, []) for row in included)
                groups[f"{prefix}:{kind}"] = {"eligible_rows": total, "selected_rows": count, "omitted_rows": total - count}
        result[partition] = {"eligible_rows": len(eligible), "selected_rows": len(included), "types": groups}
    return result


def analyze(records):
    from scripts.simulate import EVENT_KINDS, FAULT_KINDS
    grouped = {name: [] for name in (*EVENT_KINDS, *("fault:" + kind for kind in FAULT_KINDS),
                                    "weekend", "night", "partition:train", "partition:validation", "partition:test")}
    for row in records:
        stamp = datetime.fromisoformat(row["observed_at"])
        tags = {"partition:" + row["partition"], "room:" + row["room"], *row["events"]}
        tags.update("fault:" + kind for kind in row.get("sensor_faults", []))
        if stamp.weekday() >= 5:
            tags.add("weekend")
        if stamp.hour >= 21 or stamp.hour < 6:
            tags.add("night")
        for tag in tags:
            grouped.setdefault(tag, []).append(row)
    return {"overall": assessment(records), "groups": {key: assessment(value) for key, value in sorted(grouped.items())},
            "mistakes": [row for row in records if row.get("truth") is not None
                         and row["truth"] != row["prediction"]["occupancy"]][:50]}


class PipelineStore:
    _verified_sources = {}
    _cache_lock = RLock()
    def __init__(self, root=ROOT, state_dir=None, logger=None, weights_dir=None):
        self.root = Path(root)
        self.directory = Path(state_dir or os.environ.get("CLASSROOM_STATE_DIR", self.root / "state"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "pipeline.sqlite3"
        self.logger = logger
        self.weights_dir = Path(weights_dir) if weights_dir is not None else None
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL, error TEXT, log_context TEXT);
                CREATE TABLE IF NOT EXISTS observations(run_id TEXT, key TEXT, ordinal INTEGER, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id,key), UNIQUE(run_id,ordinal));
                CREATE TABLE IF NOT EXISTS predictions(run_id TEXT, key TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id,key));
                CREATE TABLE IF NOT EXISTS log_delivery(run_id TEXT, key TEXT, PRIMARY KEY(run_id,key));
                CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
            if "log_context" not in {row[1] for row in db.execute("PRAGMA table_info(runs)")}:
                db.execute("ALTER TABLE runs ADD COLUMN log_context TEXT")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def valid_id(value):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("파이프라인 ID를 확인하세요.")

    def source(self, generation_id):
        from models.train import generation_roots, load_dataset, synthetic_generation
        if not re.fullmatch(r"dynamic-sensors-v1-[0-9a-f]{12}", generation_id):
            raise ValueError("생성 ID를 확인하세요.")
        def signature():
            paths = [self.root / "data/training_dataset.csv", self.root / "data/dataset_sources.json"]
            for base in generation_roots(self.root):
                if base.is_symlink():
                    raise ValueError("합성 데이터 루트의 심볼릭 링크는 허용하지 않습니다.")
                directory = base / generation_id
                if directory.is_symlink():
                    raise ValueError("생성 폴더의 심볼릭 링크는 허용하지 않습니다.")
                if directory.is_dir():
                    paths.append(directory)
                    paths.extend(sorted(directory.iterdir()))
            return tuple((str(path), path.is_symlink(), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                         for path in paths for stat in [path.lstat()])
        before = signature()
        cache_key = (str(self.root.resolve()), str(self.directory.resolve()), generation_id)
        with self._cache_lock:
            cached = self._verified_sources.get(cache_key)
        if cached and cached[0] == before:
            return cached[1]
        source = load_dataset(self.root / "data/training_dataset.csv")
        result = synthetic_generation(generation_id, source, self.root,
                                      partitions=("train", "train_only", "validation", "test"), observations=True)
        if signature() != before:
            raise ValueError("생성물 검증 도중 파일이 바뀌었습니다.")
        # 원본 전체 해시 검증은 최초/변경 시만 한다. 큰 관측 묶음의 메모리 캐시는 2개로 제한한다.
        with self._cache_lock:
            if cache_key not in self._verified_sources and len(self._verified_sources) >= 2:
                self._verified_sources.pop(next(iter(self._verified_sources)))
            self._verified_sources[cache_key] = (before, result)
        return result

    def checkpoint(self, version, mode):
        # 기본 모델 포인터가 아닌 명시한 버전의 파일을 고정한다.
        if self.weights_dir is None:
            path = weight_path(version, mode)
        else:
            from models.model import valid_version
            path = self.weights_dir / valid_version(version) / ("model.pt" if mode == "with_light" else "model_without_light.pt")
        content = path.read_bytes()
        return path, hashlib.sha256(content).hexdigest()

    def persist_run(self, config, info, observations, baseline=None, sampling=None):
        path, fingerprint = self.checkpoint(config.model_version, config.mode)
        spec = {"config": config.model_dump(), "generation": {key: value for key, value in info.items() if key != "observations"},
                "model_fingerprint": fingerprint, "observation_hash": identity(observations), "baseline": baseline,
                "label_source": "verified_virtual_people_gt_zero", "purpose": "repeatable_development_simulation"}
        if sampling is not None:
            spec["sampling"] = sampling
        run_id = identity(spec)
        with self.connect() as db:
            context = {"run_id": self.logger.run_id, "root": str(self.logger.root)} if self.logger else None
            db.execute("INSERT OR IGNORE INTO runs(id,spec,status,error,log_context) VALUES(?,?,?,NULL,?)",
                       (run_id, encode(spec), "ready", encode(context)))
            db.executemany("INSERT OR IGNORE INTO observations VALUES(?,?,?,?)",
                           [(run_id, row["key"], i, encode(row)) for i, row in enumerate(observations)])
        return self.run(run_id)

    def create_run(self, config):
        _, info = self.source(config.generation_id)
        rng = np.random.default_rng(config.seed)
        groups = {}
        for row in info["observations"]:
            groups.setdefault(row["partition"], []).append(row)
        shuffled = [list(rng.permutation(rows)) for _, rows in sorted(groups.items())]
        rows, chosen_keys = [], set()
        from scripts.simulate import EVENT_KINDS, FAULT_KINDS
        # 희귀 사건을 먼저 확보하되 방/episode의 원래 파티션은 바꾸지 않는다.
        for field, kinds in (("events", EVENT_KINDS), ("sensor_faults", FAULT_KINDS)):
            for kind in kinds:
                for group in shuffled:
                    if len(rows) >= config.max_rows:
                        break
                    if any(kind in row.get(field, []) and row["partition"] == group[0]["partition"] for row in rows):
                        continue
                    candidate = next((row for row in group if kind in row.get(field, []) and row["key"] not in chosen_keys), None)
                    if candidate:
                        rows.append(candidate)
                        chosen_keys.add(candidate["key"])
        for i in range(max(map(len, shuffled))):
            if len(rows) >= config.max_rows:
                break
            for group in shuffled:
                if len(rows) >= config.max_rows:
                    break
                if i < len(group) and group[i]["key"] not in chosen_keys:
                    rows.append(group[i])
                    chosen_keys.add(group[i]["key"])
        return self.persist_run(config, info, rows, sampling={
            "policy": "rare_event_and_fault_coverage_then_partition_balanced_without_replacement",
            "natural_prevalence": False, "coverage": sampling_coverage(info["observations"], rows),
            "excluded_before_replay": sum(room.get("excluded_rows", 0) for room in info["rooms"]),
            "exclusions": [{"room": room["room"], "partition": room["partition"],
                            "excluded_rows": room.get("excluded_rows", 0), "reasons": room.get("exclusion_reasons", {})}
                           for room in info["rooms"]]})

    def run(self, run_id):
        self.valid_id(run_id)
        with self.connect() as db:
            record = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if record is None:
                raise ValueError("파이프라인 실행이 없습니다.")
            total = db.execute("SELECT COUNT(*) FROM observations WHERE run_id=?", (run_id,)).fetchone()[0]
            processed = db.execute("SELECT COUNT(*) FROM predictions WHERE run_id=?", (run_id,)).fetchone()[0]
        return {"id": run_id, **json.loads(record["spec"]), "status": record["status"], "error": record["error"],
                "rows": total, "processed": processed}

    def runs(self):
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM runs ORDER BY rowid DESC LIMIT 50")]
        return [self.run(value) for value in ids]

    def records(self, run_id, pending_logs=False):
        self.run(run_id)
        with self.connect() as db:
            query = (
                "SELECT o.payload observation,p.payload prediction FROM observations o JOIN predictions p "
                "ON o.run_id=p.run_id AND o.key=p.key WHERE o.run_id=? ")
            if pending_logs:
                query += "AND NOT EXISTS(SELECT 1 FROM log_delivery l WHERE l.run_id=p.run_id AND l.key=p.key) "
            return [{**json.loads(row["observation"]), **json.loads(row["prediction"])}
                    for row in db.execute(query + "ORDER BY o.ordinal", (run_id,))]

    def process(self, run_id, limit=64):
        run = self.run(run_id)
        if run["status"] == "completed":
            self.flush_logs(run_id)
            return run
        if run["status"] == "cancelled":
            return run
        with FileLock(self.directory / f"pipeline-{run_id}.lock", timeout=0):
            try:
                _, info = self.source(run["config"]["generation_id"])
                if info["manifest_sha256"] != run["generation"]["manifest_sha256"]:
                    raise ValueError("고정한 생성물 manifest가 바뀌었습니다.")
                with self.connect() as db:
                    frozen = [json.loads(row[0]) for row in db.execute("SELECT payload FROM observations WHERE run_id=? ORDER BY ordinal", (run_id,))]
                if identity(frozen) != run["observation_hash"]:
                    raise ValueError("고정 관측 스냅샷이 손상됐습니다.")
                path, fingerprint = self.checkpoint(run["config"]["model_version"], run["config"]["mode"])
                if fingerprint != run["model_fingerprint"]:
                    raise ValueError("고정한 모델 파일이 바뀌었습니다.")
                model = load_model(path)
                if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
                    raise ValueError("모델 로드 중 파일이 바뀌었습니다.")
                with self.connect() as db:
                    db.execute("UPDATE runs SET status='running',error=NULL WHERE id=? AND status!='cancelled'", (run_id,))
                    db.commit()
                    if db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0] == "cancelled":
                        return self.run(run_id)
                    rows = db.execute("SELECT o.* FROM observations o LEFT JOIN predictions p ON o.run_id=p.run_id AND o.key=p.key "
                                      "WHERE o.run_id=? AND p.key IS NULL ORDER BY o.ordinal LIMIT ?", (run_id, limit)).fetchall()
                    for item in rows:
                        if db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0] == "cancelled":
                            break
                        observation = json.loads(item["payload"])
                        prediction = predict_one(model, observation["sensors"])
                        result = {"prediction": prediction, "model_version": run["config"]["model_version"],
                                  "model_fingerprint": fingerprint, "features": model.metadata["features"],
                                  "time_feature_policy": model.metadata.get("time_feature_policy") or {
                                      "id": "unrecorded-legacy", "inference": "caller supplied Hour/DayOfWeek unchanged"},
                                  "time_input_basis": "observation CSV explicit offset local time",
                                  "inference_id": identity([run_id, observation["key"], fingerprint]),
                                  "label_source": "verified_virtual_people_gt_zero"}
                        db.execute("INSERT OR IGNORE INTO predictions VALUES(?,?,?)", (run_id, item["key"], encode(result)))
                        db.commit()
                    remaining = db.execute("SELECT COUNT(*) FROM observations o LEFT JOIN predictions p ON o.run_id=p.run_id AND o.key=p.key "
                                           "WHERE o.run_id=? AND p.key IS NULL", (run_id,)).fetchone()[0]
                    if not remaining:
                        db.execute("UPDATE runs SET status='completed' WHERE id=? AND status!='cancelled'", (run_id,))
                    else:
                        db.execute("UPDATE runs SET status='ready' WHERE id=? AND status='running'", (run_id,))
                # SQLite가 정본이며 JSONL 재전송도 inference_id로 중복을 막는다.
                self.flush_logs(run_id)
            except Exception as error:
                with self.connect() as db:
                    db.execute("UPDATE runs SET status='failed',error=? WHERE id=? AND status!='cancelled'", (str(error), run_id))
        return self.run(run_id)

    def flush_logs(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT log_context FROM runs WHERE id=?", (run_id,)).fetchone()
        context = json.loads(row[0]) if row and row[0] else None
        if context:
            logger = self.logger if self.logger and self.logger.run_id == context["run_id"] and str(self.logger.root) == context["root"] else RuntimeLogger(root=context["root"], run_id=context["run_id"])
            run = self.run(run_id)
            for row in self.records(run_id, pending_logs=True):
                logger.write_once("model", "pipeline_observation", row["inference_id"],
                    pipeline_id=run_id, generation_id=run["config"]["generation_id"], observation=row,
                    inference_id=row["inference_id"], row_id=row["row_id"], room_id=row["room"],
                    model_version=row["model_version"], model_fingerprint=row["model_fingerprint"], source="simulation_pipeline",
                    prediction_label_source="model_prediction", label_source="verified_virtual_people_gt_zero")
                # fsync된 JSONL 기록 뒤에만 전송 확인. 중간 종료 시 같은 키로 안전하게 재전송한다.
                with self.connect() as db:
                    db.execute("INSERT OR IGNORE INTO log_delivery VALUES(?,?)", (run_id, row["key"]))

    def control(self, run_id, action):
        self.run(run_id)
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,error=NULL WHERE id=? AND status!='completed'",
                       ("cancelled" if action == "cancel" else "ready", run_id))
        return self.run(run_id)

    def analysis(self, run_id):
        return {"run": self.run(run_id), **analyze(self.records(run_id))}

    def create_dataset(self, run_id, config):
        run = self.run(run_id)
        if run["status"] != "completed":
            raise ValueError("예측을 완료한 뒤 재학습 데이터를 확정하세요.")
        if run["generation"]["calibration_scope"] != "real_fit_only":
            raise ValueError("기존 생성물은 보정에 실제 validation을 포함합니다. fit-only 생성물로 새 실행을 만드세요.")
        eligible = [row for row in self.records(run_id) if row["partition"] in ("train", "train_only") and row.get("truth") is not None]
        if not eligible:
            raise ValueError("정답이 연결된 학습용 관측이 없습니다.")
        rng = np.random.default_rng(config.seed)
        order = rng.permutation(len(eligible)).tolist()
        count = min(len(order), config.max_rows)
        base = order[:math.ceil(count / 2)]
        remaining = [i for i in order if i not in set(base)]
        errors = [i for i in remaining if eligible[i]["truth"] != eligible[i]["prediction"]["occupancy"]]
        chosen = base + (errors + [i for i in remaining if i not in set(errors)])[:count - len(base)]
        payload = {"run_id": run_id, "generation_id": run["config"]["generation_id"],
                   "manifest_sha256": run["generation"]["manifest_sha256"], "seed": config.seed,
                   "row_keys": [eligible[i]["key"] for i in chosen], "rows": len(chosen),
                   "uniform_rows": len(base), "error_boost_rows": sum(i in errors for i in chosen[len(base):]),
                   "policy": "at_least_half_uniform_then_error_boost_without_replacement",
                   "reference_model": run["config"]["model_version"], "reference_fingerprint": run["model_fingerprint"],
                   "label_source": "verified_virtual_people_gt_zero", "partitions": ["train", "train_only"],
                   "validation_ratio": run["generation"]["provenance"]["real_partition"]["validation_ratio"]}
        payload["coverage"] = sampling_coverage(eligible, [eligible[i] for i in chosen])
        payload["row_context"] = [{"partition": eligible[i]["partition"], "events": eligible[i]["events"],
                                   "sensor_faults": eligible[i].get("sensor_faults", [])} for i in chosen]
        dataset_id = identity(payload)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?)", (dataset_id, encode(payload)))
        return {"id": dataset_id, **payload}

    def datasets(self):
        with self.connect() as db:
            return [{"id": row[0], **json.loads(row[1])} for row in db.execute("SELECT id,payload FROM datasets ORDER BY rowid DESC LIMIT 50")]

    @staticmethod
    def generation_process_alive(record):
        pid = record.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        if os.name == "nt":
            # Windows의 os.kill(pid, 0)은 프로세스를 종료할 수 있어 읽기 전용 핸들로 확인한다.
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            command = Path(f"/proc/{pid}/cmdline")
            return not command.exists() or record["id"] in command.read_text(errors="replace")
        except (OSError, ValueError):
            return False

    def generation_job(self, job_id):
        from scripts.simulate import write_generation_job
        self.valid_id(job_id)
        path = self.directory / "generation_jobs" / f"{job_id}.json"
        if not path.is_file():
            raise ValueError("생성 작업이 없습니다.")
        with FileLock(str(path) + ".lock", timeout=10):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["id"] != job_id:
                raise ValueError("생성 작업 ID가 파일명과 다릅니다.")
            if (record["status"] in ("queued", "running") and not self.generation_process_alive(record)
                    and (record["status"] == "running" or time.time() - record["created_at_epoch"] > 30)):
                record.update(status="failed", error="생성 프로세스가 중단됐습니다. 같은 설정으로 재시도할 수 있습니다.")
                write_generation_job(path, record)
        fields = ("id", "status", "config", "error", "generation_id", "quality_passed", "reused_output", "storage")
        return {key: record[key] for key in fields if key in record}

    def generation_jobs(self):
        directory = self.directory / "generation_jobs"
        if not directory.is_dir():
            return []
        jobs = []
        for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime_ns, reverse=True)[:50]:
            try:
                jobs.append(self.generation_job(path.stem))
            except (OSError, ValueError, KeyError, TypeError):
                jobs.append({"id": path.stem, "status": "failed", "error": "생성 이력을 읽지 못했습니다. 원본 파일은 보존됩니다."})
        return jobs

    def queue_generation(self, config):
        from scripts.simulate import write_generation_job
        directory = self.directory / "generation_jobs"
        directory.mkdir(parents=True, exist_ok=True)
        code_root = Path(__file__).resolve().parents[1]
        job = {"config": config, "root": str(self.root), "output_root": str(self.directory / "simulated"),
               "script_sha256": hashlib.sha256((code_root / "scripts/simulate.py").read_bytes()).hexdigest(),
               "training_sha256": hashlib.sha256((self.root / "data/training_dataset.csv").read_bytes()).hexdigest(),
               "metadata_sha256": hashlib.sha256((self.root / "data/dataset_sources.json").read_bytes()).hexdigest()}
        job_id = identity(job)
        path = directory / f"{job_id}.json"
        with FileLock(directory / ".submission.lock", timeout=0):
            previous = self.generation_job(job_id) if path.exists() else None
            if previous and previous["status"] in ("queued", "running"):
                return previous
            if any(item["status"] in ("queued", "running") for item in self.generation_jobs()):
                raise ValueError("다른 생성 작업이 진행 중입니다. 완료 후 접수하세요.")
            record = {**job, "id": job_id, "status": "queued", "error": None, "pid": None,
                      "created_at_epoch": time.time(), "recover_incomplete": bool(previous and previous["status"] == "failed")}
            with FileLock(str(path) + ".lock", timeout=10):
                write_generation_job(path, record)
                try:
                    with path.with_suffix(".log").open("ab") as output:
                        process = subprocess.Popen([sys.executable, "-m", "scripts.simulate", "--generation-job", str(path)],
                            cwd=code_root, stdout=output, stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
                    record["pid"] = process.pid
                except OSError as error:
                    record.update(status="failed", error=f"생성 프로세스를 시작하지 못했습니다: {error}")
                write_generation_job(path, record)
        return self.generation_job(job_id)

    def feedback(self, dataset_id, source):
        from models.train import synthetic_generation
        self.valid_id(dataset_id)
        with self.connect() as db:
            row = db.execute("SELECT payload FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if row is None or identity(json.loads(row[0])) != dataset_id:
            raise ValueError("확정 데이터셋이 없거나 해시가 다릅니다.")
        info = json.loads(row[0])
        data, generation = synthetic_generation(info["generation_id"], source, self.root)
        if generation["calibration_scope"] != "real_fit_only" or generation["manifest_sha256"] != info["manifest_sha256"]:
            raise ValueError("피드백 데이터셋의 보정 범위·원본 해시가 다릅니다.")
        by_key = {key: i for i, key in enumerate(data.row_keys)}
        if len(set(info["row_keys"])) != info["rows"] or not set(info["row_keys"]).issubset(by_key):
            raise ValueError("학습 파티션 밖의 관측 또는 중복 행이 있습니다.")
        return data.select([by_key[key] for key in info["row_keys"]]), {"id": dataset_id, **info}

    def create_comparison(self, run_id, version, mode="with_light"):
        run = self.run(run_id)
        if run["status"] != "completed":
            raise ValueError("기준 모델 예측을 먼저 완료하세요.")
        records = self.records(run_id)
        observations = [{key: value for key, value in row.items() if key not in
            ("prediction", "model_version", "model_fingerprint", "features", "inference_id", "label_source", "time_feature_policy", "time_input_basis")}
            for row in records if row["partition"] in ("validation", "test")]
        if not observations:
            raise ValueError("비교에 사용할 고정 validation/test 관측이 없습니다.")
        config = ReplayConfig(**{**run["config"], "model_version": version, "mode": mode})
        return self.persist_run(config, run["generation"], observations, baseline=run_id)

    def comparison(self, candidate_id):
        candidate = self.run(candidate_id)
        if not candidate["baseline"]:
            raise ValueError("모델 비교 실행이 아닙니다.")
        baseline = self.records(candidate["baseline"])
        current = self.records(candidate_id)
        keys = {row["key"] for row in current}
        baseline = [row for row in baseline if row["key"] in keys]
        before, after = analyze(baseline), analyze(current)
        return {"baseline": before, "candidate": after, "run": candidate,
                "complete": candidate["status"] == "completed", "same_observations": True,
                "independence": candidate["generation"]["validation_independence"],
                "error_changes": {key: after["groups"][key]["errors"] - before["groups"][key]["errors"]
                                  for key in after["groups"]},
                "note": "양수는 오류 증가입니다. 반복 개발용 시나리오 비교이며 최종 실제 test가 아닙니다. 기본 모델은 자동 변경하지 않습니다."}
