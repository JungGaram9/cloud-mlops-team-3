"""가짜 예측과 임시 현장 데이터로만 개선 파이프라인 계약을 검증한다."""

import csv
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import pipeline
import backend.test_data as data_fixtures
from models import train
from scripts.runtime_logging import RuntimeLogger
from scripts.simulate import Config, calibration, digest, generate, read_training


class PipelineTests(unittest.TestCase):
    def setUp(self):
        data_fixtures.SyntheticTrainingTests.setUp(self)
        patch.dict(os.environ, {"CLASSROOM_STATE_DIR": str(self.root / "state")}).start()
        self.fit_output, _ = generate(Config(rooms=3, step_seconds=300, calibration_scope="fit_only"),
                                      self.root / "data/simulated", self.root)
        self.files = {version: self.root / f"{version}.pt" for version in ("v1", "v2")}
        for version, path in self.files.items():
            path.write_bytes(version.encode())
        patch.object(pipeline, "weight_path", side_effect=lambda version, mode: self.files[version]).start()
        patch.object(pipeline, "load_model", side_effect=lambda path: SimpleNamespace(
            metadata={"features": ["CO2", "Hour"]}, name=path.stem)).start()
        self.predict = patch.object(pipeline, "predict_one", side_effect=lambda model, sensors:
            {"occupancy": int(model.name == "v2"), "probability": .8 if model.name == "v2" else .2,
             "model_version": model.name, "label": "가짜 예측"}).start()
        self.logger = RuntimeLogger(root=self.root / "logs")
        self.store = pipeline.PipelineStore(self.root, logger=self.logger)

    def create(self, **updates):
        return self.store.create_run(pipeline.ReplayConfig(**{
            "generation_id": self.fit_output.name, "model_version": "v1", "max_rows": 30, **updates}))

    def complete(self, run):
        for _ in range(50):
            run = self.store.process(run["id"], 256)
            if run["status"] in ("completed", "failed", "cancelled"):
                break
        self.assertEqual(run["status"], "completed", run.get("error"))
        return run

    def test_zero_truth_sensor_only_model_input_restart_and_log_retry_are_idempotent(self):
        run = self.complete(self.create())
        self.assertEqual(self.create()["id"], run["id"])
        self.store.process(run["id"])
        restarted = pipeline.PipelineStore(self.root, logger=RuntimeLogger(root=self.root / "logs"))
        self.assertEqual(restarted.create_run(pipeline.ReplayConfig(generation_id=self.fit_output.name, model_version="v1", max_rows=30))["id"], run["id"])
        restarted.process(run["id"])
        rows = pipeline.PipelineStore(self.root).records(run["id"])
        self.assertEqual(len(rows), 30)
        self.assertTrue(any(row["truth"] == 0 for row in rows))
        self.assertEqual(len({row["inference_id"] for row in rows}), 30)
        self.assertTrue(all(row["model_version"] == "v1" for row in rows))
        for call in self.predict.call_args_list:
            self.assertEqual(set(call.args[1]), set(train.FEATURES))
        lines = [json.loads(line) for line in (self.logger.directory / "model.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 30)
        self.assertTrue(all(line["label_source"] == "verified_virtual_people_gt_zero" for line in lines))
        self.assertTrue(all(line["observation"]["truth"] is not None for line in lines))

    def test_checkpoint_pin_cancel_resume_and_corrupted_observation(self):
        run = self.create()
        first = self.store.process(run["id"], 2)
        self.assertEqual(first["processed"], 2)
        self.store.control(run["id"], "cancel")
        self.assertEqual(self.store.process(run["id"])["processed"], 2)
        self.store.control(run["id"], "resume")
        self.files["v1"].write_bytes(b"changed model")
        failed = self.store.process(run["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("모델", failed["error"])
        self.files["v1"].write_bytes(b"v1")
        self.store.control(run["id"], "resume")
        self.complete(first)
        other = self.create(seed=43)
        with self.store.connect() as db:
            db.execute("UPDATE observations SET payload='{}' WHERE run_id=? AND ordinal=0", (other["id"],))
        self.assertEqual(self.store.process(other["id"])["status"], "failed")

    def test_experiment_weights_are_scoped_without_changing_default_model_resolution(self):
        weights = self.root / "experiment-weights"
        private = weights / "v1/model.pt"
        private.parent.mkdir(parents=True)
        private.write_bytes(b"private v1")
        scoped = pipeline.PipelineStore(self.root, state_dir=self.root / "experiment-state", weights_dir=weights)
        path, fingerprint = scoped.checkpoint("v1", "with_light")
        self.assertEqual(path, private)
        self.assertEqual(fingerprint, digest(private))
        self.assertEqual(self.store.checkpoint("v1", "with_light")[0], self.files["v1"])

    def test_fit_only_calibration_ignores_validation_sensor_values_and_never_reads_test(self):
        records, provenance = read_training(self.root, .2)
        before = calibration(records, provenance)["sources"]
        fit, valid = train.split_training(self.source, .2, True)
        self.assertEqual(provenance["real_partition"]["fit_indices_sha256"], fit.summary["selection_indices_sha256"])
        path = self.root / "data/training_dataset.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        valid_dates = set(valid.dates)
        for row in rows:
            if train.datetime.fromisoformat(row["date"]) in valid_dates:
                row["Temperature"] = "40"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        sidecar = path.parent / "dataset_sources.json"
        info = json.loads(sidecar.read_text())
        info[path.name]["sha256"] = digest(path)
        sidecar.write_text(json.dumps(info))
        (path.parent / "test_dataset.csv").write_text("평가 파일은 읽지 않는다")
        after_records, after_provenance = read_training(self.root, .2)
        self.assertEqual(before, calibration(after_records, after_provenance)["sources"])

    def test_feedback_only_train_dedup_against_csv_and_preserve_actual_validation(self):
        run = self.complete(self.create(max_rows=90))
        snapshot = self.store.create_dataset(run["id"], pipeline.FeedbackConfig(max_rows=30))
        self.assertEqual(snapshot["id"], self.store.create_dataset(run["id"], pipeline.FeedbackConfig(max_rows=30))["id"])
        observed, _ = self.store.feedback(snapshot["id"], self.source)
        self.assertTrue(all(key.split(":")[1] == "room_01" for key in observed.row_keys))
        self.assertGreaterEqual(snapshot["uniform_rows"], snapshot["rows"] / 2)
        config = train.TrainingConfig(synthetic_generation=self.fit_output.name, pipeline_dataset=snapshot["id"])
        mixed, validation, plan = train.prepare_training_data(self.source, self.test, config)
        _, baseline_valid = train.split_training(self.source, .2, True)
        self.assertEqual(validation.summary, baseline_valid.summary)
        synthetic_keys = [key for key in mixed.row_keys if key]
        self.assertEqual(len(synthetic_keys), len(set(synthetic_keys)))
        self.assertGreater(plan["log_data"]["included_rows"], 0)
        self.assertLessEqual(plan["synthetic_fit_rows"], plan["real_fit_rows"])
        self.assertEqual(plan["test"], self.test.summary)
        with self.assertRaises(ValueError):
            train.prepare_training_data(self.source, self.test, config.model_copy(update={"validation_ratio": .3}))
        old = self.complete(self.create(generation_id=self.output.name))
        with self.assertRaisesRegex(ValueError, "fit-only"):
            self.store.create_dataset(old["id"], pipeline.FeedbackConfig())

    def test_comparison_uses_same_heldout_rows_and_exposes_regressions(self):
        baseline = self.complete(self.create(max_rows=60))
        candidate = self.complete(self.store.create_comparison(baseline["id"], "v2"))
        report = self.store.comparison(candidate["id"])
        reference = {row["key"] for row in self.store.records(baseline["id"]) if row["partition"] in ("validation", "test")}
        self.assertEqual(reference, {row["key"] for row in self.store.records(candidate["id"])})
        self.assertTrue(report["complete"])
        self.assertTrue(any(change > 0 for change in report["error_changes"].values()))
        self.assertFalse(report["candidate"]["overall"]["final_real_test"])

    def test_unlabelled_and_single_class_metrics_do_not_invent_zero_accuracy(self):
        self.assertIsNone(pipeline.assessment([])["accuracy"])
        record = {"truth": None, "prediction": {"occupancy": 0}}
        self.assertFalse(pipeline.assessment([record])["available"])
        record["truth"] = 0
        result = pipeline.assessment([record])
        self.assertEqual(result["accuracy"], 1)
        self.assertTrue(result["single_class"])
        self.assertIsNone(result["recall"])

    def test_api_fixed_replay_snapshot_and_existing_mock_training_job(self):
        from fastapi.testclient import TestClient
        from unittest.mock import Mock
        import backend.app as api
        patch.object(api, "ROOT", self.root).start()
        patch.object(api, "JOBS_DIR", self.root / "jobs").start()
        process = Mock(pid=12345)
        process.poll.return_value = None
        app = api.create_app(log_root=self.root / "api-logs", state_dir=self.root / "state",
                             enable_classrooms=False, start_simulation=False)
        with patch.object(api.subprocess, "Popen", return_value=process) as spawn, TestClient(app) as client:
            with patch("scripts.simulate.generate", return_value=(self.fit_output, {"passed": True})) as generator:
                created = client.post("/pipeline/generations", json={"seed": 123})
                self.assertEqual(created.status_code, 202, created.text)
                generator.assert_not_called()
                self.assertIn("scripts.simulate", spawn.call_args.args[0])
                job_path = self.root / "state/generation_jobs" / f"{created.json()['id']}.json"
                from scripts.simulate import run_generation_job
                run_generation_job(job_path)
                self.assertEqual(client.get(f"/pipeline/generations/{created.json()['id']}").json()["status"], "completed")
                self.assertEqual(generator.call_args.args[0].calibration_scope, "fit_only")
                self.assertEqual(generator.call_args.args[1], self.root / "state/simulated")
            spawn.reset_mock()
            response = client.post("/pipeline/runs", json={"generation_id": self.fit_output.name, "model_version": "v1", "max_rows": 12})
            self.assertEqual(response.status_code, 200, response.text)
            run_id = response.json()["id"]
            spawn.assert_not_called()
            response = client.post(f"/pipeline/runs/{run_id}/step")
            self.assertEqual(response.json()["status"], "completed", response.text)
            snapshot = client.post(f"/pipeline/runs/{run_id}/dataset", json={"max_rows": 12}).json()
            self.assertIn("id", snapshot)
            config = train.TrainingConfig(pipeline_dataset=snapshot["id"], shared_room_split=True)
            plan = client.get("/training/options", params={"pipeline_dataset": snapshot["id"], "shared_room_split": True}).json()["data_plan"]
            self.assertTrue(plan["ready"], plan)
            response = client.post("/training/jobs", json=config.model_dump(), headers={"X-Data-Plan-SHA256": plan["fingerprint"]})
            self.assertEqual(response.status_code, 202, response.text)
            self.assertGreater(response.json()["data_plan"]["log_data"]["included_rows"], 0)
            spawn.assert_called_once()
            self.assertEqual(client.post(f"/training/jobs/{response.json()['id']}/cancel").status_code, 200)

    def test_state_generated_storage_and_cancel_during_validation(self):
        target = self.root / "state/simulated" / self.fit_output.name
        target.parent.mkdir(parents=True)
        self.fit_output.rename(target)
        self.fit_output = target
        catalog = train.synthetic_catalog(self.source)
        self.assertTrue(any(row["generation_id"] == target.name and row["available"] for row in catalog))
        run = self.create()
        original = self.store.source
        def cancelling_source(generation_id):
            result = original(generation_id)
            self.store.control(run["id"], "cancel")
            return result
        with patch.object(self.store, "source", side_effect=cancelling_source):
            cancelled = self.store.process(run["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["processed"], 0)
        self.store.control(run["id"], "resume")
        self.complete(run)

    def test_source_verification_cache_and_pending_log_delivery(self):
        pipeline.PipelineStore._verified_sources.clear()
        with patch.object(train, "synthetic_generation", wraps=train.synthetic_generation) as verify:
            run = self.create(max_rows=6)
            self.store.process(run["id"], 2)
            with patch.object(self.logger, "write_once", wraps=self.logger.write_once) as delivered:
                self.store.process(run["id"], 2)
                self.assertEqual(delivered.call_count, 2)
            self.assertEqual(verify.call_count, 1)
            path = self.fit_output / "room_01.metadata.jsonl"
            stat = path.stat()
            content = path.read_bytes()
            path.write_bytes(content.replace(b'"synthetic":true', b'"synthetic":null', 1))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(self.store.process(run["id"])["status"], "failed")
            self.assertEqual(verify.call_count, 2)

    def test_dirty_raw_clean_replay_feedback_contract_and_300_row_coverage(self):
        from scripts.simulate import EVENT_KINDS
        target, _ = generate(Config(rooms=3, step_seconds=300, calibration_scope="fit_only", observation_mode="stress",
                                     fault_min_minutes=30, fault_max_minutes=60), self.root / "data/simulated", self.root)
        self.fit_output = target
        run = self.complete(self.create(max_rows=300))
        self.assertGreater(run["sampling"]["excluded_before_replay"], 0)
        self.assertFalse(run["sampling"]["natural_prevalence"])
        for partition in ("train", "validation", "test"):
            for kind in EVENT_KINDS:
                counts = run["sampling"]["coverage"][partition]["types"]["event:" + kind]
                self.assertGreater(counts["selected_rows"], 0, (partition, kind))
                self.assertEqual(counts["eligible_rows"], counts["selected_rows"] + counts["omitted_rows"])
        records = self.store.records(run["id"])
        self.assertTrue(any(row["truth"] == 0 and row["sensor_faults"] for row in records))
        self.assertTrue(all(row["raw_csv_line"] is not None for row in records))
        self.assertTrue(all("missing" not in row["sensor_faults"] and "dropout" not in row["sensor_faults"] for row in records))
        for call in self.predict.call_args_list:
            self.assertEqual(set(call.args[1]), set(train.FEATURES))
            train.SensorInput(**call.args[1])
        snapshot = self.store.create_dataset(run["id"], pipeline.FeedbackConfig(max_rows=60))
        dataset, _ = self.store.feedback(snapshot["id"], self.source)
        self.assertTrue(all(key.split(":")[1] == "room_01" for key in dataset.row_keys))
        self.assertEqual(set(snapshot["coverage"]), {"train"})
        self.assertTrue(any(label == 0 for label in dataset.y))
        _, _, plan = train.prepare_training_data(self.source, self.test,
            train.TrainingConfig(pipeline_dataset=snapshot["id"], pipeline_ratio=.02, shared_room_split=True))
        self.assertEqual(plan["log_data"]["selected_coverage"]["train"]["selected_rows"], plan["log_data"]["included_rows"])
        self.assertTrue(any("빠진 유형" in warning for warning in plan["warnings"]))
        tiny = self.create(max_rows=3)
        self.assertTrue(any(count["eligible_rows"] > 0 and count["selected_rows"] == 0
            for group in tiny["sampling"]["coverage"].values() for count in group["types"].values()))
        # 원시 파일 해시만 고쳐도 감사값과 불일치하면 사용할 수 없다.
        path = target / "room_02.raw.csv"
        path.write_bytes(path.read_bytes().replace(b",", b",1", 1))
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["room_02.csv"]["raw_sha256"] = digest(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.store.source(target.name)

    def test_legacy_fixed_interval_generation_contract_is_still_readable(self):
        target, _ = generate(Config(rooms=3, step_seconds=300, observation_mode="off"), self.root / "data/simulated", self.root)
        path = target / "manifest.json"
        manifest = json.loads(path.read_text())
        for key in ("observation_schema", "observation_policy", "sensor_faults"):
            manifest.pop(key)
        path.write_text(json.dumps(manifest))
        dataset, info = train.synthetic_generation(target.name, self.source, self.root, observations=True)
        self.assertEqual(info["observation_schema"], 1)
        self.assertEqual(len(dataset.y), 7 * 24 * 12)
        self.assertTrue(all("sensor_faults" not in row for row in info["observations"]))

    def test_partial_jsonl_failure_replays_from_sqlite_without_duplicate_rows(self):
        run = self.create(max_rows=2)
        original = self.logger.write_once
        def crash(*args, **kwargs):
            with (self.logger.directory / "model.jsonl").open("ab") as stream:
                stream.write(b'{"event_type":"pipeline_observation",')
            raise OSError("부분 쓰기 후 중단")
        with patch.object(self.logger, "write_once", side_effect=crash):
            self.assertEqual(self.store.process(run["id"])["status"], "failed")
        self.assertEqual(len(self.store.records(run["id"])), 2)
        self.complete(run)
        rows = [json.loads(line) for line in (self.logger.directory / "model.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["inference_id"] for row in rows}), 2)
        self.assertEqual(len(list(self.logger.directory.glob("*.partial-*.bin"))), 1)
        with patch.object(self.logger, "write_once", wraps=original) as delivered:
            self.store.process(run["id"])
            delivered.assert_not_called()

    def test_generation_job_dedup_reuse_incomplete_recovery_and_corruption(self):
        from unittest.mock import Mock
        from scripts.simulate import run_generation_job, write_generation_job
        config = {"rooms": 3, "days": 7, "step_seconds": 300, "seed": 20261111, "validation_ratio": .2}
        with patch.object(pipeline.subprocess, "Popen", return_value=Mock(pid=12345)) as spawn:
            job = self.store.queue_generation(config)
            self.assertEqual(job["id"], self.store.queue_generation(config)["id"])
            self.assertEqual(spawn.call_count, 1)
            path = self.root / "state/generation_jobs" / f"{job['id']}.json"
            result = run_generation_job(path)
            self.assertEqual(result["status"], "completed", result.get("error"))
            output = self.root / "state/simulated" / result["generation_id"]
            csv_path = output / "room_01.csv"
            original, stat = csv_path.read_bytes(), csv_path.stat()
            self.store.queue_generation(config)
            result = run_generation_job(path)
            self.assertTrue(result["reused_output"])
            self.assertEqual(csv_path.stat().st_mtime_ns, stat.st_mtime_ns)
            (output / "INCOMPLETE").write_text("중단된 작업")
            record = json.loads(path.read_text())
            record["status"] = "failed"
            write_generation_job(path, record)
            self.store.queue_generation(config)
            result = run_generation_job(path)
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertFalse((output / "INCOMPLETE").exists())
            archives = list((output.parent / ".incomplete").glob("*/room_01.csv"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), original)
            csv_path.write_bytes(original + b"broken")
            self.store.queue_generation(config)
            result = run_generation_job(path)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(csv_path.read_bytes(), original + b"broken")


class FeedbackSamplingTests(unittest.TestCase):
    def test_small_budget_keeps_general_and_error_boost_and_is_reproducible(self):
        selected = train.select_feedback_rows(list(range(10)), 6, 4, 2, 42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(i >= 6 for i in selected), 1)
        self.assertEqual(selected, train.select_feedback_rows(list(range(10)), 6, 4, 2, 42))

    def test_dedup_removed_general_rows_limits_error_boost(self):
        selected = train.select_feedback_rows([0, 6, 7, 8, 9], 6, 4, 5, 42)
        self.assertEqual(len(selected), 2)
        self.assertIn(0, selected)
        self.assertEqual(train.select_feedback_rows([6, 7], 6, 4, 2, 42), [])

    def test_no_error_and_single_row_budget(self):
        self.assertEqual(len(train.select_feedback_rows(list(range(6)), 6, 0, 4, 42)), 4)
        self.assertLess(train.select_feedback_rows(list(range(10)), 6, 4, 1, 42)[0], 6)
