"""강의실 측정 저장·상태 판별과 서버 공유 하루 시연을 관리한다."""

import asyncio
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from models.model import SensorInput

ROOMS = {name: {"id": name, "name": f"시연 {name}호", "floor": int(name[0])} for name in ("101", "102", "103", "201", "202", "203")}
SCHEDULES = {
    "101": [(510, 600), (630, 720), (840, 930), (1080, 1140)],
    "102": [(540, 630), (780, 870), (960, 1050)],
    "103": [(660, 720), (900, 1020)],
    "201": [(480, 570), (600, 690), (780, 900), (1140, 1230)],
    "202": [(570, 660), (870, 960), (1020, 1110)],
    "203": [(1080, 1260)],
}
SENSOR_NAMES = ("Temperature", "Humidity", "Light", "CO2")
HISTORY_LIMIT = 96


class MeasurementInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    measurement_id: str
    observed_at: datetime
    source: Literal["real"] = "real"
    Temperature: float
    Humidity: float
    Light: float
    CO2: float

    @field_validator("measurement_id")
    @classmethod
    def check_id(cls, value):
        if not value.strip() or len(value) > 100:
            raise ValueError("측정 ID는 비어 있지 않은 1~100자로 지정하세요.")
        return value

    @field_validator("observed_at", mode="before")
    @classmethod
    def check_time(cls, value):
        date = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(date, datetime) or date.tzinfo is None or date.utcoffset() is None:
            raise ValueError("측정 시각에 UTC offset 또는 Z가 필요합니다.")
        return date

    @model_validator(mode="after")
    def check_sensors(self):
        date = self.observed_at.astimezone(timezone.utc)
        SensorInput(**{name: getattr(self, name) for name in SENSOR_NAMES}, Hour=date.hour, DayOfWeek=date.weekday())
        return self


class SimulationControl(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["play", "pause", "restart", "speed"]
    speed: Literal[1, 5, 10, 20] | None = None

    @model_validator(mode="after")
    def check_speed(self):
        if self.action == "speed" and self.speed is None:
            raise ValueError("변경할 시연 속도를 지정하세요.")
        return self


class SourceControl(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: Literal["simulation", "real"]


def scenario_sensors(room_id, minute):
    """시간표는 센서 변화에만 사용한다. 재실 상태는 이 함수에서 결정하지 않는다."""
    offset = list(ROOMS).index(room_id)
    activity, co2 = 0.0, 0.0
    for start, end in SCHEDULES[room_id]:
        if minute >= start:
            duration = min(minute, end) - start
            activity += (1 - math.exp(-duration / 3)) * math.exp(-max(0, minute - end) / 5)
            co2 += (360 + offset * 12) * (1 - math.exp(-duration / 18)) * math.exp(-max(0, minute - end) / 25)
    daylight = max(0, math.sin(math.pi * (minute - 480) / 840))
    return {"Temperature": round(20.1 + offset * 0.12 + 0.65 * daylight + 0.6 * activity, 6),
            "Humidity": round(24.3 + offset * 0.2 + 1.4 * math.sin((minute - 480) / 210) - 0.6 * activity, 6),
            "Light": round(8 + 14 * daylight + (470 + offset * 12) * activity, 6),
            "CO2": round(450 + offset * 13 + co2, 6)}


class ClassroomService:
    def __init__(self, directory, predict, logger, *, stale_after=60, now=None):
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError("센서 수신 지연 기준은 0보다 큰 초 단위 값이어야 합니다.")
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.predict, self.logger, self.stale_after = predict, logger, stale_after
        self.lock = threading.RLock()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.owner = FileLock(str(directory / "classrooms.service.lock"), timeout=5)
        self.owner.acquire()
        try:
            self.db = sqlite3.connect(directory / "classrooms.sqlite3", check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS rooms(id TEXT PRIMARY KEY, source TEXT NOT NULL, latest TEXT);
                CREATE TABLE IF NOT EXISTS history(seq INTEGER PRIMARY KEY, room_id TEXT, record TEXT);
                CREATE INDEX IF NOT EXISTS history_room ON history(room_id,seq);
                CREATE TABLE IF NOT EXISTS seen(room_id TEXT, source TEXT, measurement_id TEXT, digest TEXT, receipt TEXT,
                    PRIMARY KEY(room_id,source,measurement_id));
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            """)
            for name in ROOMS:
                self.db.execute("INSERT OR IGNORE INTO rooms VALUES(?, 'simulation', NULL)", (name,))
            stored = self.db.execute("SELECT value FROM settings WHERE key='simulation'").fetchone()
            self.simulation = json.loads(stored[0]) if stored else self.new_session()
            self.save_clock()
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            self.owner.release()
            raise

    def new_session(self):
        return {"session_id": uuid.uuid4().hex, "scenario_id": "classroom-day-v1", "day": self.now().date().isoformat(),
                "minute": 480.0, "playing": True, "speed": 5, "last_emitted": None}

    def save_clock(self):
        self.db.execute("INSERT OR REPLACE INTO settings VALUES('simulation',?)", (json.dumps(self.simulation),))
        self.db.commit()

    def require_room(self, room_id):
        if room_id not in ROOMS:
            raise KeyError("강의실을 찾을 수 없습니다.")

    def ingest(self, room_id, *, measurement_id, observed_at, sensors, source, session_id=None, scenario_id=None, request_id=None):
        self.require_room(room_id)
        date = datetime.fromisoformat(observed_at)
        utc_date = date.astimezone(timezone.utc)
        inputs = SensorInput(**sensors, Hour=utc_date.hour, DayOfWeek=utc_date.weekday()).model_dump()
        identity = {"observed_at": utc_date.isoformat(), "sensors": sensors, "session_id": session_id}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self.lock:
            seen = self.db.execute("SELECT digest, receipt FROM seen WHERE room_id=? AND source=? AND measurement_id=?",
                                   (room_id, source, measurement_id)).fetchone()
            if seen:
                if seen[0] != digest:
                    raise ValueError("같은 측정 ID에 다른 데이터가 전달되었습니다.")
                return {**json.loads(seen[1]), "duplicate": True, "accepted_as_latest": False}
            mode, latest = self.db.execute("SELECT source,latest FROM rooms WHERE id=?", (room_id,)).fetchone()
            latest = json.loads(latest) if latest else None
            if source == "simulation" and mode == "real":
                return {"measurement_id": measurement_id, "accepted_as_latest": False, "ignored": "real_sensor_mode"}
            is_late = (latest is not None and mode == source and latest.get("session_id") == session_id
                       and utc_date <= datetime.fromisoformat(latest["observed_at"]).astimezone(timezone.utc))
            received = self.now().isoformat()
            context = {"classroom_id": room_id, "measurement_id": measurement_id, "observed_at": observed_at,
                       "received_at": received, "session_id": session_id, "scenario_id": scenario_id}
            record = {**context, "source": source, "sensors": sensors, "model_inputs": inputs, "prediction": None,
                      "inference_id": None, "error": None, "accepted_as_latest": not is_late}
            result = None
            if not is_late:
                try:
                    result = self.predict(inputs)
                    record.update(prediction=result["prediction"], inference_id=uuid.uuid4().hex,
                                  features=result["features"], threshold=result["threshold"], pair_fingerprint=result["fingerprint"])
                except Exception as error:
                    record["error"] = {"type": type(error).__name__, "message": "저장 모델로 판별하지 못했습니다."}
            receipt = {"classroom_id": room_id, "source": source, "measurement_id": measurement_id,
                       "inference_id": record["inference_id"], "accepted_as_latest": not is_late,
                       "duplicate": False, "received_at": received, "error": record["error"]}
            with self.db:
                self.db.execute("INSERT INTO seen VALUES(?,?,?,?,?)", (room_id, source, measurement_id, digest, json.dumps(receipt)))
                self.db.execute("INSERT INTO history(room_id,record) VALUES(?,?)", (room_id, json.dumps(record)))
                self.db.execute("DELETE FROM history WHERE room_id=? AND seq NOT IN (SELECT seq FROM history WHERE room_id=? ORDER BY seq DESC LIMIT ?)",
                                (room_id, room_id, HISTORY_LIMIT))
                if not is_late:
                    self.db.execute("UPDATE rooms SET source=?,latest=? WHERE id=?", (source, json.dumps(record), room_id))
            self.logger.write("model", "measurement", event="received", request_id=request_id, source=source,
                              sensors=sensors, accepted_as_latest=not is_late, error=record["error"], **context)
            if result is not None:
                self.logger.inference(request_id=request_id, sensors=inputs, version=result["prediction"]["model_version"],
                    mode="with_light", features=result["features"], threshold=result["threshold"], fingerprint=result["fingerprint"],
                    prediction=result["prediction"], source=source, inference_id=record["inference_id"], context=context)
            return receipt

    def emit(self, request_id=None):
        minute = int(self.simulation["minute"])
        observed = datetime.fromisoformat(self.simulation["day"]).replace(tzinfo=timezone.utc) + timedelta(minutes=minute)
        for name, mode in self.db.execute("SELECT id,source FROM rooms").fetchall():
            if mode == "simulation":
                self.ingest(name, measurement_id=f"{self.simulation['session_id']}:{name}:{minute}",
                    observed_at=observed.isoformat(), sensors=scenario_sensors(name, minute), source="simulation",
                    session_id=self.simulation["session_id"], scenario_id=self.simulation["scenario_id"], request_id=request_id)
        self.simulation["last_emitted"] = minute
        self.save_clock()

    def advance(self, elapsed):
        with self.lock:
            if not self.simulation["playing"]:
                return
            self.simulation["minute"] = min(1320.0, self.simulation["minute"] + max(0, elapsed) * self.simulation["speed"])
            if int(self.simulation["minute"]) != self.simulation["last_emitted"]:
                self.emit()
            if self.simulation["minute"] >= 1320:
                self.simulation["playing"] = False
            self.save_clock()

    def control(self, action, speed=None, request_id=None):
        with self.lock:
            if action == "restart":
                self.simulation = self.new_session()
                self.emit(request_id)
            elif action == "speed":
                self.simulation["speed"] = speed
            elif action == "pause":
                self.simulation["playing"] = False
            elif action == "play":
                if self.simulation["minute"] >= 1320:
                    raise ValueError("하루 시연이 끝났습니다. 처음부터를 선택하세요.")
                self.simulation["playing"] = True
            self.save_clock()
            self.logger.write("model", "simulation", event="control", action=action, request_id=request_id, **self.simulation)
            return self.clock_view()

    def set_source(self, room_id, source, request_id=None):
        self.require_room(room_id)
        with self.lock, self.db:
            if self.db.execute("SELECT source FROM rooms WHERE id=?", (room_id,)).fetchone()[0] == source:
                return self.room_view(room_id)
            self.db.execute("UPDATE rooms SET source=?,latest=NULL WHERE id=?", (source, room_id))
        self.logger.write("model", "measurement", event="source_changed", classroom_id=room_id, source=source, request_id=request_id)
        # 모드 변경 직후는 수신 대기다. 다음 새 시연 측정부터 반영한다.
        return self.room_view(room_id)

    def clock_view(self):
        minute = self.simulation["minute"]
        date = datetime.fromisoformat(self.simulation["day"]).replace(tzinfo=timezone.utc) + timedelta(minutes=minute)
        return {**self.simulation, "observed_at": date.isoformat(), "timezone": "UTC", "start": "08:00", "end": "22:00",
                "finished": minute >= 1320}

    def room_view(self, room_id, detail=False):
        self.require_room(room_id)
        with self.lock:
            mode, raw = self.db.execute("SELECT source,latest FROM rooms WHERE id=?", (room_id,)).fetchone()
            latest = json.loads(raw) if raw else None
            age = max(0, (self.now() - datetime.fromisoformat(latest["received_at"])).total_seconds()) if latest else None
            paused = mode == "simulation" and not self.simulation["playing"]
            status, message = "waiting", "센서 수신 대기"
            if latest:
                if age > self.stale_after:
                    status, message = "stale", "수신 지연 · 연결 확인 필요"
                elif latest["error"]:
                    status, message = "unavailable", latest["error"]["message"]
                else:
                    status = "occupied" if latest["prediction"]["occupancy"] else "vacant"
                    message = "모델이 추정한 사용 상태"
            result = {**ROOMS[room_id], "source": mode, "latest": latest, "status": status, "message": message,
                      "age_seconds": age, "simulation_paused": paused}
            if detail:
                result["history"] = [json.loads(row[0]) for row in self.db.execute(
                    "SELECT record FROM history WHERE room_id=? ORDER BY seq DESC LIMIT ?", (room_id, HISTORY_LIMIT))][::-1]
                result["scenario_schedule"] = [{"start": f"{start // 60:02}:{start % 60:02}", "end": f"{end // 60:02}:{end % 60:02}"}
                                               for start, end in SCHEDULES[room_id]]
            return result

    def snapshot(self):
        with self.lock:
            rooms = [self.room_view(name) for name in ROOMS]
            return {"classrooms": rooms, "simulation": self.clock_view(), "server_time": self.now().isoformat(),
                    "stale_after_seconds": self.stale_after, "history_limit": HISTORY_LIMIT,
                    "counts": {"total": len(rooms), "occupied": sum(r["status"] == "occupied" for r in rooms),
                               "vacant": sum(r["status"] == "vacant" for r in rooms),
                               "attention": sum(r["status"] not in ("occupied", "vacant") for r in rooms)}}

    async def run(self):
        previous = time.monotonic()
        while True:
            await asyncio.sleep(1)
            current = time.monotonic()
            try:
                await asyncio.to_thread(self.advance, current - previous)
            except Exception as error:
                self.logger.write("model", "simulation", event="failed", error_type=type(error).__name__)
                with self.lock:
                    self.simulation.update(playing=False, error="시연 처리를 중단했습니다. 서버 저장소를 확인하세요.")
            previous = current

    def close(self):
        with self.lock:
            try:
                self.save_clock()
            finally:
                self.db.close()
                self.owner.release()
