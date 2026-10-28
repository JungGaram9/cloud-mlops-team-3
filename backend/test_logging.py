"""가짜 예측·콜백으로 실행별 로그를 검증한다. 모델 학습·추론·평가는 실행하지 않는다."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import backend.app as api
from models.model import MOCK_SENSORS, REPORTS_DIR
from scripts.runtime_logging import RuntimeLogger


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.logs = self.root / "logs"
        self.jobs = self.root / "jobs"
        self.addCleanup(patch.stopall)
        patch.object(api, "JOBS_DIR", self.jobs).start()
        patch.object(api, "current_version", return_value="v3").start()
        report = json.loads((REPORTS_DIR / "v3/light_comparison.json").read_text(encoding="utf-8"))

        def fake_model(path):
            mode = "without_light" if "without_light" in path.name else "with_light"
            return SimpleNamespace(metadata=report["modes"][mode], mean=None, scale=None,
                                   network=SimpleNamespace(state_dict=lambda: {}))

        patch.object(api, "load_model", side_effect=fake_model).start()
        patch.object(api, "pair_fingerprint", return_value="f" * 64).start()
        patch.object(api, "predict_one", side_effect=lambda model, sensors: {
            "model_version": model.metadata["version"], "occupancy": 0, "label": "비어 있음", "probability": 0.25}).start()

    def read(self, directory, channel):
        return [json.loads(line) for line in (directory / f"{channel}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_api_and_inference_logs_correlate_without_dumping_headers(self):
        app = api.create_app(log_root=self.logs, enable_classrooms=False)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health?token=do-not-store", headers={"Authorization": "Bearer secret-header"}).status_code, 200)
            prediction = client.post("/predict", json=MOCK_SENSORS)
            self.assertEqual(prediction.status_code, 200)
            self.assertEqual(set(prediction.json()), {"model_version", "occupancy", "label", "probability"})
            self.assertEqual(client.post("/compare", json=MOCK_SENSORS).status_code, 200)
            self.assertEqual(client.post("/predict", json={**MOCK_SENSORS, "CO2": 0}).status_code, 400)
            directory = app.state.runtime_logger.directory
        requests = self.read(directory, "api")
        predictions = self.read(directory, "model")
        self.assertEqual([row["event"] for row in requests if row["event_type"] == "lifecycle"], ["started", "stopped"])
        self.assertEqual(len(predictions), 3)
        request_ids = {row["request_id"] for row in requests if row["event_type"] == "request"}
        for row in predictions:
            self.assertIn(row["request_id"], request_ids)
            self.assertEqual(row["sensors"], MOCK_SENSORS)
            self.assertEqual(row["pair_fingerprint"], "f" * 64)
            self.assertEqual(row["label_source"], "model_prediction")
            self.assertNotIn("ground_truth", row)
            self.assertNotIn("reward", row)
            self.assertTrue(row["timestamp_utc"].endswith("+00:00"))
        bad = next(row for row in requests if row.get("status_code") == 400)
        self.assertEqual(bad["error_type"], "RequestValidationError")
        self.assertTrue(all(row["duration_ms"] >= 0 for row in requests if row["event_type"] == "request"))
        text = (directory / "api.jsonl").read_text() + (directory / "model.jsonl").read_text()
        self.assertNotIn("secret-header", text)
        self.assertNotIn("do-not-store", text)

    def test_api_restart_keeps_old_logs_and_child_keeps_original_run(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        app = api.create_app(log_root=self.logs, enable_classrooms=False)
        with patch.object(api.subprocess, "Popen", return_value=process) as spawn, TestClient(app) as client:
            response = client.post("/training/jobs", json={"epochs": 1})
            self.assertEqual(response.status_code, 202)
            record = response.json()
            old_run = app.state.runtime_logger
            environment = spawn.call_args.kwargs["env"]
            self.assertEqual(environment["CLASSROOM_LOG_RUN_ID"], old_run.run_id)
            self.assertEqual(environment["CLASSROOM_LOG_REQUEST_ID"], record["request_id"])
        old_api = (old_run.directory / "api.jsonl").read_bytes()
        # 실행 중 작업의 소속은 로그 연결만으로 확인하고 학습은 기동하지 않는다.
        self.jobs.joinpath(f"{record['id']}.json").unlink()
        with TestClient(api.create_app(log_root=self.logs, enable_classrooms=False)) as restarted:
            self.assertEqual(restarted.get("/health").status_code, 200)
        new_runs = list(self.logs.iterdir())
        self.assertEqual(len(new_runs), 2)
        self.assertEqual((old_run.directory / "api.jsonl").read_bytes(), old_api)
        with patch.dict(os.environ, {"CLASSROOM_LOG_RUN_ID": old_run.run_id,
                                     "CLASSROOM_LOG_ROOT": str(self.logs), "CLASSROOM_LOG_REQUEST_ID": record["request_id"]}):
            child = RuntimeLogger.for_training(record)
        self.assertEqual(child.run_id, old_run.run_id)
        child.epoch(record["id"], "with_light", None)
        child.epoch(record["id"], "with_light", {"epoch": 1, "training_loss": 0.7, "validation_loss": 0.8})
        child.training_finished({"id": record["id"], "status": "cancelled"}, self.root)
        model_rows = self.read(old_run.directory, "model")
        self.assertEqual([row["event"] for row in model_rows], ["queued", "epoch", "finished"])
        self.assertEqual(model_rows[-1]["status"], "cancelled")
        self.assertTrue(all(row["run_id"] == old_run.run_id for row in model_rows))

    def test_threads_and_processes_write_complete_json_lines(self):
        logger = RuntimeLogger(root=self.logs)

        def emit(index):
            other = RuntimeLogger(root=self.logs, run_id=logger.run_id)
            for number in range(12):
                self.assertTrue(other.write("model", "training", event="fixture", item=f"thread-{index}-{number}"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(emit, range(4)))
        code = ("import sys; from scripts.runtime_logging import RuntimeLogger; "
                "log=RuntimeLogger(root=sys.argv[1],run_id=sys.argv[2]); "
                "[log.write('model','training',event='fixture',item=f'process-{sys.argv[3]}-{i}') for i in range(12)]")
        children = [subprocess.Popen([sys.executable, "-c", code, str(self.logs), logger.run_id, str(index)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE) for index in range(3)]
        for child in children:
            _, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, error)
        rows = self.read(logger.directory, "model")
        self.assertEqual(len(rows), 84)
        self.assertEqual(len({row["item"] for row in rows}), 84)
        self.assertEqual(self.read(logger.directory, "api"), [])

    def test_write_failure_is_visible_and_does_not_block_prediction(self):
        blocked = self.root / "blocked"
        blocked.write_text("file")
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured), TestClient(api.create_app(log_root=blocked, enable_classrooms=False)) as client:
            response = client.post("/predict", json=MOCK_SENSORS)
            self.assertEqual(response.status_code, 200)
        self.assertIn("runtime_logging", captured.getvalue())
        self.assertIn("로그 저장 실패", captured.getvalue())

    def test_training_config_callback_and_result_use_fake_events_only(self):
        logger = RuntimeLogger(root=self.logs, request_id="request-fixture")
        logger.write("model", "training", event="started", job_id="job-fixture", config={"seed": 42}, seed=42)
        logger.write("model", "training", event="data_prepared", job_id="job-fixture",
                     training_data={"sha256": "a" * 64}, data_plan={"split_strategy": "source_chronological"})
        logger.epoch("job-fixture", "with_light", {"epoch": 1, "training_loss": 0.7, "validation_loss": 0.8})
        report_dir = self.root / "reports/v7"
        report_dir.mkdir(parents=True)
        report = {"pair_fingerprint": "b" * 64, "modes": {"with_light": {
            "features": ["CO2"], "threshold": 0.5, "test_data": {"sha256": "c" * 64, "rows": 1},
            "model_metrics": {"accuracy": 0.5}, "source_metrics": {}}}}
        (report_dir / "light_comparison.json").write_text(json.dumps(report), encoding="utf-8")
        logger.training_finished({"id": "job-fixture", "status": "completed", "version": "v7"}, self.root / "reports")
        logger.training_finished({"id": "failed-fixture", "status": "failed", "error": "fixture", "error_type": "ValueError"}, self.root)
        rows = self.read(logger.directory, "model")
        self.assertEqual([row["event"] for row in rows], ["started", "data_prepared", "epoch", "finished", "finished"])
        self.assertEqual(rows[0]["seed"], 42)
        self.assertEqual(rows[3]["model_version"], "v7")
        self.assertEqual(rows[3]["result"]["pair_fingerprint"], "b" * 64)
        self.assertEqual(rows[4]["error_type"], "ValueError")
        self.assertTrue(all(row["request_id"] == "request-fixture" for row in rows))
