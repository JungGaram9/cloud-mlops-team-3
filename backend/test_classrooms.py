"""강의실 상태·시계·측정 계약과 저장 모델 단건 추론만 검증한다. 학습·정확도 평가는 없다."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import backend.app as api
from backend.classrooms import ClassroomService, HISTORY_LIMIT, MeasurementInput, ROOMS, scenario_sensors
from models.model import load_model, predict_one, weight_path
from scripts.runtime_logging import RuntimeLogger


class ClassroomTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.time = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
        self.predict = Mock(side_effect=lambda values: {"prediction": {"model_version": "fixture", "occupancy": 1,
            "label": "사용 중", "probability": 0.7}, "features": ["CO2"], "threshold": 0.5, "fingerprint": "f" * 64})
        self.logger = RuntimeLogger(root=self.root / "logs")
        self.service = ClassroomService(self.root / "state", self.predict, self.logger, now=lambda: self.time)
        self.addCleanup(lambda: self.service.close() if self.service is not None else None)

    def measurement(self, room="101", identity="device-1", date=None, **changes):
        return self.service.ingest(room, measurement_id=identity, observed_at=date or self.time.isoformat(),
            sensors={"Temperature": 22.0, "Humidity": 28.0, "Light": 450.0, "CO2": 800.0, **changes}, source="real")

    def test_scenario_is_smooth_reproducible_and_physically_valid(self):
        for room in ROOMS:
            previous = None
            for minute in range(480, 1321):
                values = scenario_sensors(room, minute)
                self.assertEqual(values, scenario_sensors(room, minute))
                MeasurementInput(measurement_id="fixture", observed_at=self.time, **values)
                if previous:
                    self.assertLess(abs(values["CO2"] - previous["CO2"]), 25)
                    self.assertLess(abs(values["Light"] - previous["Light"]), 180)
                    self.assertLess(abs(values["Temperature"] - previous["Temperature"]), 0.4)
                previous = values
        self.assertGreater(scenario_sensors("101", 570)["CO2"], scenario_sensors("101", 480)["CO2"])
        self.assertLess(scenario_sensors("101", 1290)["CO2"], scenario_sensors("101", 1140)["CO2"])

    def test_shared_clock_pause_speed_restart_and_reads_do_not_predict(self):
        self.service.emit()
        initial = self.predict.call_count
        self.service.snapshot()
        self.service.room_view("101", detail=True)
        self.assertEqual(self.predict.call_count, initial)
        self.service.control("pause")
        self.service.advance(100)
        self.assertEqual(self.service.clock_view()["minute"], 480)
        self.service.control("speed", 20)
        self.service.control("play")
        self.service.advance(2)
        self.assertEqual(self.service.clock_view()["minute"], 520)
        old_session = self.service.simulation["session_id"]
        self.service.control("restart")
        self.assertNotEqual(self.service.simulation["session_id"], old_session)
        self.assertEqual(self.service.room_view("101")["latest"]["session_id"], self.service.simulation["session_id"])

    def test_duplicates_late_measurements_and_real_source_isolation(self):
        first = self.measurement()
        self.assertEqual(self.service.room_view("101")["source"], "real")
        self.assertTrue(self.measurement()["duplicate"])
        self.assertEqual(self.predict.call_count, 1)
        with self.assertRaises(ValueError):
            self.measurement(CO2=900.0)
        late = self.measurement(identity="older", date=(self.time - timedelta(minutes=1)).isoformat())
        self.assertFalse(late["accepted_as_latest"])
        self.assertEqual(self.predict.call_count, 1)
        self.service.emit()
        self.service.advance(60)
        self.service.control("restart")
        self.assertEqual(self.service.room_view("101")["latest"]["inference_id"], first["inference_id"])
        self.assertEqual(self.service.room_view("102")["source"], "simulation")
        self.service.set_source("101", "real")
        self.assertIsNotNone(self.service.room_view("101")["latest"])

    def test_waiting_stale_failure_and_virtual_future_clock(self):
        self.assertEqual(self.service.room_view("101")["status"], "waiting")
        self.service.simulation["minute"] = 1300
        self.service.emit()
        self.assertEqual(self.service.room_view("101")["status"], "occupied")
        self.time += timedelta(seconds=61)
        self.assertEqual(self.service.room_view("101")["status"], "stale")
        self.predict.side_effect = RuntimeError("fixture unavailable")
        self.measurement(identity="failed", date=self.time.isoformat())
        self.assertEqual(self.service.room_view("101")["status"], "unavailable")
        self.assertIsNone(self.service.room_view("101")["latest"]["prediction"])

    def test_measurement_timezone_is_preserved_and_features_use_utc(self):
        observed = "2026-10-28T23:00:00+09:00"
        self.measurement(date=observed)
        row = self.service.room_view("101")["latest"]
        self.assertEqual(row["observed_at"], observed)
        self.assertEqual(row["model_inputs"]["Hour"], 14)
        self.assertEqual(row["model_inputs"]["DayOfWeek"], 2)
        with self.assertRaises(ValueError):
            MeasurementInput(measurement_id="bad", observed_at="2026-10-28T23:00:00", **row["sensors"])

    def test_recent_history_is_bounded_and_restart_preserves_state(self):
        for index in range(HISTORY_LIMIT + 4):
            self.measurement(identity=f"device-{index}", date=(self.time + timedelta(minutes=index)).isoformat())
        before = self.service.room_view("101", detail=True)
        self.assertEqual(len(before["history"]), HISTORY_LIMIT)
        self.service.control("pause")
        self.service.close()
        self.service = ClassroomService(self.root / "state", self.predict, self.logger, now=lambda: self.time)
        self.assertEqual(self.service.room_view("101", detail=True)["latest"], before["latest"])
        self.assertFalse(self.service.simulation["playing"])
        self.assertTrue(self.measurement(identity="device-0")["duplicate"])

    def test_stored_model_is_used_for_one_real_prediction(self):
        model = load_model(weight_path("v3"))
        self.service.predict = lambda values: {"prediction": predict_one(model, values), "features": model.metadata["features"],
                                               "threshold": model.metadata["threshold"], "fingerprint": "fixture"}
        receipt = self.measurement()
        latest = self.service.room_view("101")["latest"]
        self.assertEqual(latest["prediction"]["model_version"], "v3")
        self.assertEqual(receipt["inference_id"], latest["inference_id"])
        self.assertIn(latest["prediction"]["occupancy"], (0, 1))


class ClassroomApiTests(unittest.TestCase):
    def test_operational_api_is_independent_of_training_options(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(api, "JOBS_DIR", root / "jobs"), TestClient(api.create_app(
                    log_root=root / "logs", state_dir=root / "state", start_simulation=False)) as client:
                self.assertEqual(client.get("/classrooms").json()["counts"]["total"], 6)
                with patch.object(api, "load_dataset", side_effect=ValueError("fixture data unavailable")):
                    self.assertEqual(client.get("/training/options").status_code, 400)
                    self.assertEqual(client.get("/classrooms").status_code, 200)
                body = {"measurement_id": "api-fixture", "observed_at": "2026-10-28T10:00:00Z",
                        "Temperature": 22.0, "Humidity": 28.0, "Light": 450.0, "CO2": 800.0}
                receipt = client.post("/classrooms/101/measurements", json=body).json()
                self.assertTrue(receipt["accepted_as_latest"])
                self.assertTrue(receipt["inference_id"])
                self.assertTrue(client.post("/classrooms/101/measurements", json=body).json()["duplicate"])
                self.assertEqual(client.get("/classrooms/101").json()["latest"]["inference_id"], receipt["inference_id"])
                self.assertEqual(client.post("/classrooms/101/measurements", json={key: val for key, val in body.items() if key != "CO2"}).status_code, 400)
                self.assertEqual(client.get("/classrooms/999").status_code, 404)
                self.assertEqual(client.post("/simulation/control", json={"action": "pause"}).status_code, 200)
                self.assertEqual(client.get("/versions").status_code, 200)
