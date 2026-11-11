"""강의실 상태·시계·측정 계약과 저장 모델 단건 추론만 검증한다. 학습·정확도 평가는 없다."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import backend.app as api
from backend.classrooms import DAY_END, DAY_START, ClassroomService, HISTORY_LIMIT, MeasurementInput, ROOMS, scenario_sensors
from models.model import current_version, load_model, predict_one, weight_path
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
            for minute in range(DAY_START, DAY_END + 1):
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
        self.assertEqual(self.service.clock_view()["minute"], DAY_START)
        self.service.control("speed", 20)
        self.service.control("play")
        self.service.advance(2)
        self.assertEqual(self.service.clock_view()["minute"], DAY_START + 40)
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

    def test_measurement_timezone_is_preserved_and_features_use_sensor_local_time(self):
        observed = "2026-10-28T23:00:00+09:00"
        self.measurement(date=observed)
        row = self.service.room_view("101")["latest"]
        self.assertEqual(row["observed_at"], observed)
        self.assertEqual(row["model_inputs"]["Hour"], 23)
        self.assertEqual(row["model_inputs"]["DayOfWeek"], 2)
        with self.assertRaises(ValueError):
            MeasurementInput(measurement_id="bad", observed_at="2026-10-28T23:00:00", **row["sensors"])

    def test_midnight_offsets_keep_local_features_and_utc_measurement_identity(self):
        cases = (("101", "2026-10-29T00:30:00+09:00", 0, 3),
                 ("102", "2026-10-28T23:30:00-05:00", 23, 2))
        for room, observed, hour, weekday in cases:
            with self.subTest(observed=observed):
                self.time = datetime.fromisoformat(observed).astimezone(timezone.utc)
                self.measurement(room=room, identity="offset", date=observed, CO2=500.0)
                row = self.service.room_view(room)["latest"]
                self.assertEqual((row["model_inputs"]["Hour"], row["model_inputs"]["DayOfWeek"]), (hour, weekday))
                calls = self.predict.call_count
                self.assertTrue(self.measurement(room=room, identity="offset", date=self.time.isoformat(), CO2=500.0)["duplicate"])
                self.assertFalse(self.measurement(room=room, identity="same-instant", date=self.time.isoformat(), CO2=500.0)["accepted_as_latest"])
                self.assertEqual(self.predict.call_count, calls)
                self.time += timedelta(seconds=30)
                self.measurement(room=room, identity="next", date=self.time.isoformat(), CO2=500.0)
                self.assertEqual(self.service.room_view(room)["light_check"]["duration_seconds"], 30)

    def test_day_boundaries_keep_05_to_24_range_and_next_day_model_features(self):
        start = self.service.clock_view()
        self.assertEqual((start["start"], start["end"], start["minute"]), ("05:00", "24:00", DAY_START))
        self.service.simulation.update(minute=1319.0, last_emitted=None, speed=20)
        self.service.advance(1)
        self.assertEqual(self.service.clock_view()["minute"], 1339)
        self.assertFalse(self.service.clock_view()["finished"])
        self.assertTrue(self.service.simulation["playing"])
        self.service.simulation.update(minute=float(DAY_END - 1), last_emitted=None)
        self.service.advance(1)
        end = self.service.clock_view()
        self.assertEqual(end["minute"], DAY_END)
        self.assertEqual(datetime.fromisoformat(end["observed_at"]), self.time.replace(hour=0) + timedelta(days=1))
        latest = self.service.room_view("101")["latest"]
        self.assertEqual((latest["model_inputs"]["Hour"], latest["model_inputs"]["DayOfWeek"]), (0, 3))
        self.assertTrue(end["finished"])
        self.assertFalse(end["playing"])
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "paused")
        calls = self.predict.call_count
        self.service.advance(10)
        self.assertEqual(self.predict.call_count, calls)
        with self.assertRaises(ValueError):
            self.service.control("play")
        restarted = self.service.control("restart")
        self.assertEqual(restarted["minute"], DAY_START)
        self.assertEqual(self.service.room_view("101")["latest"]["model_inputs"]["Hour"], 5)
        self.assertTrue(restarted["playing"])

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
        version = current_version()
        model = load_model(weight_path(version))
        self.service.predict = lambda values: {"prediction": predict_one(model, values), "features": model.metadata["features"],
                                               "threshold": model.metadata["threshold"], "fingerprint": "fixture"}
        receipt = self.measurement()
        latest = self.service.room_view("101")["latest"]
        self.assertEqual(latest["prediction"]["model_version"], version)
        self.assertEqual(receipt["inference_id"], latest["inference_id"])
        self.assertIn(latest["prediction"]["occupancy"], (0, 1))

    def test_light_check_requires_continuous_measurements_and_preserves_model_prediction(self):
        origin = self.time
        self.measurement(identity="brief", Light=300.0, CO2=650.0, Humidity=99.0)
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "monitoring")
        self.time += timedelta(seconds=59)
        self.measurement(identity="brief-end", Light=299.0, CO2=650.0)
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "clear")
        for minute in range(2, 13):
            self.time = origin + timedelta(minutes=minute)
            self.measurement(identity=f"bright-{minute}", Light=300.0, CO2=650.0, Humidity=90.0)
        room = self.service.room_view("101")
        self.assertEqual(room["status"], "occupied")
        self.assertTrue(room["light_check"]["requires_check"])
        self.assertTrue(room["light_check"]["model_disagrees"])
        self.assertEqual(room["light_check"]["duration_seconds"], 600)
        self.assertEqual(self.service.snapshot()["counts"]["light_check"], 1)
        log = self.logger.directory / "model.jsonl"
        original = log.read_bytes()
        calls = self.predict.call_count
        self.time += timedelta(seconds=10)
        for _ in range(5):
            self.assertEqual(self.service.room_view("101")["light_check"]["duration_seconds"], 600)
            self.service.snapshot()
        self.assertEqual(self.predict.call_count, calls)
        self.assertEqual(log.read_bytes(), original)
        events = [json.loads(line) for line in original.decode().splitlines()]
        changes = [item for item in events if item["event_type"] == "patrol"]
        self.assertEqual([item["state"] for item in changes], ["monitoring", "clear", "monitoring", "attention"])
        self.assertEqual(changes[-1]["rules"]["basis"], "demo_assumption")
        self.assertEqual(changes[-1]["evidence"]["Humidity"], 90.0)

    def test_light_check_resets_after_receive_gap_and_rejects_old_or_future_real_times(self):
        self.measurement(identity="start", CO2=500.0)
        self.time += timedelta(seconds=61)
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "stale")
        self.measurement(identity="resumed", CO2=500.0)
        self.assertEqual(self.service.room_view("101")["light_check"]["duration_seconds"], 0)
        self.time += timedelta(minutes=5)
        self.measurement(identity="old", date=(self.time - timedelta(minutes=2)).isoformat(), CO2=500.0)
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "stale")
        self.measurement(identity="future", date=(self.time + timedelta(seconds=20)).isoformat(), CO2=500.0)
        self.assertFalse(self.service.room_view("101")["light_check"]["requires_check"])
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "stale")

    def test_late_and_duplicate_measurements_do_not_extend_light_check(self):
        original = self.time
        self.measurement(identity="start", CO2=500.0)
        self.time += timedelta(seconds=30)
        self.measurement(identity="new", CO2=500.0)
        before = self.service.room_view("101")["light_check"]
        self.assertTrue(self.measurement(identity="start", date=original.isoformat(), CO2=500.0)["duplicate"])
        self.assertFalse(self.measurement(identity="late", date=(original + timedelta(seconds=10)).isoformat(), CO2=500.0)["accepted_as_latest"])
        self.assertFalse(self.measurement(identity="same-time", CO2=500.0)["accepted_as_latest"])
        self.assertEqual(self.service.room_view("101")["light_check"], before)

    def test_service_restart_source_switch_and_simulation_restart_reset_light_continuity(self):
        self.measurement(identity="first", CO2=500.0)
        self.time += timedelta(seconds=30)
        self.measurement(identity="second", CO2=500.0)
        self.service.close()
        self.service = ClassroomService(self.root / "state", self.predict, self.logger, now=lambda: self.time)
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "waiting")
        self.assertEqual(self.service.room_view("101")["light_check"]["duration_seconds"], 0)
        self.time += timedelta(seconds=30)
        self.measurement(identity="after-restart", CO2=500.0)
        self.assertEqual(self.service.room_view("101")["light_check"]["duration_seconds"], 0)
        self.service.set_source("101", "simulation")
        self.assertEqual(self.service.room_view("101")["light_check"]["state"], "waiting")
        self.service.control("restart")
        self.assertFalse(self.service.room_view("101")["light_check"]["requires_check"])
        self.service.set_source("101", "real")
        self.measurement(identity="back-real", CO2=500.0)
        self.assertEqual(self.service.room_view("101")["light_check"]["duration_seconds"], 0)

    def test_simulation_light_check_is_identical_at_all_speeds_and_freezes_when_paused(self):
        reference = None
        for speed in (1, 5, 10, 20):
            with self.subTest(speed=speed):
                self.service.control("restart")
                self.service.simulation.update(minute=720.0, last_emitted=None)
                self.service.emit()
                self.service.control("speed", speed)
                self.service.advance(60 / speed)
                check = self.service.room_view("103")["light_check"]
                self.assertTrue(check["requires_check"])
                current = (check["since"], check["duration_seconds"])
                reference = reference or current
                self.assertEqual(current, reference)
                self.service.control("pause")
                self.service.advance(100)
                paused = self.service.room_view("103")["light_check"]
                self.assertEqual(paused["state"], "paused")
                self.assertFalse(paused["requires_check"])
                self.assertEqual(paused["duration_seconds"], check["duration_seconds"])
        self.time += timedelta(seconds=61)
        self.assertEqual(self.service.room_view("103")["light_check"]["state"], "stale")

    def test_large_virtual_gap_and_rule_change_restart_candidate_time(self):
        session = self.service.simulation["session_id"]
        for minute in (780, 781, 785):
            self.service.ingest("103", measurement_id=f"virtual-{minute}",
                observed_at=(self.time + timedelta(minutes=minute)).isoformat(),
                sensors={"Temperature": 22.0, "Humidity": 30.0, "Light": 450.0, "CO2": 500.0},
                source="simulation", session_id=session)
        self.assertEqual(self.service.room_view("103")["light_check"]["duration_seconds"], 0)
        self.service.light_rules = {**self.service.light_rules, "min_lux": 400}
        self.assertEqual(self.service.room_view("103")["light_check"]["state"], "waiting")

    def test_existing_simulation_session_keeps_original_scenario_until_restart(self):
        self.service.simulation.update(minute=780.0, scenario_id="classroom-day-v1")
        self.service.emit()
        self.assertLess(self.service.room_view("103")["latest"]["sensors"]["Light"], 300)
        self.service.control("restart")
        self.assertEqual(self.service.simulation["scenario_id"], "classroom-day-v2")


class ClassroomApiTests(unittest.TestCase):
    def test_operational_api_is_independent_of_training_options(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(api, "JOBS_DIR", root / "jobs"), TestClient(api.create_app(
                    log_root=root / "logs", state_dir=root / "state", start_simulation=False)) as client:
                self.assertEqual(client.get("/classrooms").json()["counts"]["total"], 6)
                self.assertEqual(client.get("/classrooms").json()["counts"]["light_check"], 0)
                self.assertEqual(client.get("/classrooms/101").json()["light_check"]["rules"]["hold_seconds"], 600)
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
