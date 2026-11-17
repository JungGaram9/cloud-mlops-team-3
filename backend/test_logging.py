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
from scripts.simulate import TIME_FEATURE_POLICY


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
            self.assertEqual(row["time_feature_policy"]["id"], "unrecorded-legacy")
            self.assertNotIn("ground_truth", row)
            self.assertNotIn("reward", row)
            self.assertTrue(row["timestamp_utc"].endswith("+00:00"))
        bad = next(row for row in requests if row.get("status_code") == 400)
        self.assertEqual(bad["error_type"], "RequestValidationError")
        self.assertTrue(all(row["duration_ms"] >= 0 for row in requests if row["event_type"] == "request"))
        text = (directory / "api.jsonl").read_text() + (directory / "model.jsonl").read_text()
        self.assertNotIn("secret-header", text)
        self.assertNotIn("do-not-store", text)

    def test_new_model_time_policy_is_logged_without_converting_caller_input(self):
        original = api.load_model.side_effect
        def policy_model(path):
            saved = original(path)
            saved.metadata = {**saved.metadata, "time_feature_policy": TIME_FEATURE_POLICY}
            return saved
        sensors = {**MOCK_SENSORS, "Hour": 23, "DayOfWeek": 6}
        with patch.object(api, "load_model", side_effect=policy_model), TestClient(api.create_app(log_root=self.logs, enable_classrooms=False)) as client:
            response = client.post("/predict", json=sensors)
            self.assertEqual(response.status_code, 200)
            directory = client.app.state.runtime_logger.directory
        record = self.read(directory, "model")[0]
        self.assertEqual(record["time_feature_policy"], TIME_FEATURE_POLICY)
        self.assertEqual(record["sensors"], sensors)
        self.assertEqual(api.predict_one.call_args.args[1], sensors)

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


class LogTailRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.logger = RuntimeLogger(root=Path(temporary.name))
        self.path = self.logger.directory / "model.jsonl"

    def test_partial_utf8_tail_preserves_valid_prefix_and_exact_damaged_bytes(self):
        self.logger.write_once("model", "fixture", "old", value=0)
        prefix = self.path.read_bytes()
        damaged = b'{"deduplication_key":"interrupted","text":"\xe2\x82'
        with self.path.open("ab") as stream:
            stream.write(damaged)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.logger.write_once("model", "fixture", "next", value=1))
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        self.assertEqual([json.loads(line)["deduplication_key"] for line in self.path.read_bytes().splitlines()], ["old", "next"])
        archives = list(self.logger.directory.glob("*.partial-*.bin"))
        self.assertEqual([path.read_bytes() for path in archives], [damaged])
        self.assertFalse(self.logger.write_once("model", "fixture", "next", value=1))
        self.assertEqual(len(self.path.read_bytes().splitlines()), 2)

    def test_valid_unterminated_record_is_completed_and_not_duplicated(self):
        original = json.dumps({"deduplication_key": "already-written", "value": 0}).encode()
        self.path.write_bytes(original)
        self.assertFalse(self.logger.write_once("model", "fixture", "already-written", value=0))
        self.assertEqual(self.path.read_bytes(), original + b"\n")
        self.assertEqual(list(self.logger.directory.glob("*.partial-*.bin")), [])

    def test_general_write_recovers_before_once_and_reverse_order_keeps_zero_and_cache(self):
        self.logger.write_once("model", "fixture", "old", value=0)
        prefix = self.path.read_bytes()
        first_tail = b'{"text":"\xed\x95'
        with self.path.open("ab") as stream:
            stream.write(first_tail)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.logger.write("model", "training", value=0))
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        self.assertFalse(self.logger.write_once("model", "fixture", "old", value=0))
        self.assertTrue(self.logger.write_once("model", "fixture", "next", value=0))
        second_tail = b'{"text":"second interruption'
        with self.path.open("ab") as stream:
            stream.write(second_tail)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.logger.write_once("model", "fixture", "last", value=0))
        self.assertTrue(self.logger.write("model", "training", value=0))
        rows = [json.loads(line) for line in self.path.read_bytes().splitlines()]
        self.assertEqual([row.get("deduplication_key") for row in rows], ["old", None, "next", "last", None])
        self.assertTrue(all(row["value"] == 0 for row in rows))
        self.assertEqual({path.read_bytes() for path in self.logger.directory.glob("*.partial-*.bin")}, {first_tail, second_tail})

    def test_general_write_completes_unterminated_json_without_full_scan_on_healthy_writes(self):
        original = json.dumps({"deduplication_key": "complete", "value": 0}).encode()
        self.path.write_bytes(original)
        self.assertTrue(self.logger.write("model", "fixture", value=0))
        self.assertTrue(self.path.read_bytes().startswith(original + b"\n"))
        self.assertFalse(self.logger.write_once("model", "fixture", "complete", value=0))
        original_open = Path.open
        scans = []
        class ObserveReads:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def __getattr__(self, name): return getattr(self.stream, name)
            def __iter__(self):
                scans.append("full")
                return iter(self.stream)
            def read(self, count=-1):
                scans.append(count)
                return self.stream.read(count)
        def observing_open(target, mode="r", *args, **kwargs):
            stream = original_open(target, mode, *args, **kwargs)
            return ObserveReads(stream) if target == self.path and mode == "a+b" else stream
        with patch.object(Path, "open", observing_open):
            self.assertTrue(self.logger.write("model", "fixture", value=0))
            self.assertTrue(self.logger.write_once("model", "fixture", "cached", value=0))
            self.assertFalse(self.logger.write_once("model", "fixture", "cached", value=0))
        self.assertEqual(scans, [1, 1, 1])
        self.assertEqual(list(self.logger.directory.glob("*.partial-*.bin")), [])

    def test_returned_short_write_keeps_failure_contract_and_can_retry_through_other_writer(self):
        original_open = Path.open
        class ShortWrite:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def __getattr__(self, name): return getattr(self.stream, name)
            def __iter__(self): return iter(self.stream)
            def write(self, value):
                count = self.stream.write(value[:17])
                self.stream.flush()
                return count
        def broken_open(target, mode="r", *args, **kwargs):
            stream = original_open(target, mode, *args, **kwargs)
            return ShortWrite(stream) if target == self.path and mode == "a+b" else stream
        with patch.object(Path, "open", broken_open), contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(self.logger.write("model", "fixture", value=0))
        partial = self.path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.logger.write_once("model", "fixture", "retry", value=0))
        self.assertEqual(len(self.path.read_bytes().splitlines()), 1)
        self.assertEqual(next(self.logger.directory.glob("*.partial-*.bin")).read_bytes(), partial)
        self.assertFalse(self.logger.write("invalid", "fixture"))
        with self.assertRaises(ValueError): self.logger.write_once("invalid", "fixture", "key")

    def test_mixed_process_writers_recover_one_tail_and_deduplicate_shared_keys(self):
        partial = b'{"interrupted":"\xe2\x82'
        self.path.write_bytes(partial)
        code = "\n".join([
            "import sys", "from scripts.runtime_logging import RuntimeLogger",
            "log=RuntimeLogger(root=sys.argv[1],run_id=sys.argv[2])",
            "for i in range(5):",
            "    assert log.write('model','fixture',writer=sys.argv[3],item=i,value=0)",
            "    log.write_once('model','fixture',str(i),value=0)",
        ])
        children = [subprocess.Popen([sys.executable, "-c", code, str(self.logger.root), self.logger.run_id, str(i)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(3)]
        for child in children:
            _, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, error)
        rows = [json.loads(line) for line in self.path.read_bytes().splitlines()]
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["deduplication_key"] for row in rows if "deduplication_key" in row}, {str(i) for i in range(5)})
        self.assertTrue(all(row["value"] == 0 for row in rows))
        self.assertEqual([path.read_bytes() for path in self.logger.directory.glob("*.partial-*.bin")], [partial])

    def test_quarantine_short_write_does_not_truncate_original_and_resumes_preservation(self):
        self.logger.write_once("model", "fixture", "old", value=0)
        prefix = self.path.read_bytes()
        partial = b'{"text":"interrupted utf8 \xed\x95'
        with self.path.open("ab") as output:
            output.write(partial)
        original_open = Path.open
        @contextlib.contextmanager
        def short_archive(stream):
            with stream:
                yield SimpleNamespace(write=lambda data: stream.write(data[:5]), flush=stream.flush, fileno=stream.fileno)
        def broken_open(target, mode="r", *args, **kwargs):
            stream = original_open(target, mode, *args, **kwargs)
            return short_archive(stream) if ".partial-" in target.name and mode == "xb" else stream
        with patch.object(Path, "open", broken_open), contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(self.logger.write("model", "fixture", value=0))
        self.assertEqual(self.path.read_bytes(), prefix + partial)
        self.assertEqual(next(self.logger.directory.glob("*.partial-*.bin")).read_bytes(), partial[:5])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.logger.write("model", "fixture", value=0))
        self.assertEqual(next(self.logger.directory.glob("*.partial-*.bin")).read_bytes(), partial)
        self.assertEqual(len(self.path.read_bytes().splitlines()), 2)

    def test_short_write_exception_then_retry(self):
        original_open = Path.open
        path = self.path
        class ShortWrite:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def __iter__(self):
                return iter(self.stream)
            def write(self, value):
                self.stream.write(value[:17])
                self.stream.flush()
                raise OSError("짧은 쓰기 후 중단")
        def broken_open(target, mode="r", *args, **kwargs):
            stream = original_open(target, mode, *args, **kwargs)
            return ShortWrite(stream) if target == path and mode == "a+b" else stream
        with patch.object(Path, "open", broken_open), self.assertRaises(OSError):
            self.logger.write_once("model", "fixture", "retry", value=0)
        partial = self.path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.logger.write_once("model", "fixture", "retry", value=0))
        self.assertEqual([json.loads(line)["deduplication_key"] for line in self.path.read_bytes().splitlines()], ["retry"])
        self.assertEqual(next(self.logger.directory.glob("*.partial-*.bin")).read_bytes(), partial)
