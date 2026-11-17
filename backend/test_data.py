"""학습·추론 없이 데이터 분할과 모델 메타데이터 계약을 확인한다."""

import copy
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from models import model, train
from scripts.merge_data import COLUMNS, merge_data
from scripts.simulate import Config as SimulationConfig, TIME_FEATURE_POLICY, digest, generate, read_training
from scripts.test_simulate import fixture as simulation_fixture

EVALUATE_CONTRACT = train.evaluate


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        self.addCleanup(patch.stopall)
        patch.object(train, "ROOT", self.root).start()
        # 잘못된 경로로 넘어가도 실제 학습이나 지표 계산을 실행하지 못하게 막는다.
        for name in ("train_network", "train_experiment", "evaluate", "probabilities"):
            patch.object(train, name, side_effect=AssertionError("학습·평가 실행 금지")).start()

    def write_csv(self, path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self):
        rows, groups = [], []
        for group_index, group in enumerate(("legacy", "Room_A", "Room_B")):
            for index in range(10):
                rows.append(dict(zip(COLUMNS, [str(datetime(2023, 1, 1) + timedelta(days=group_index, hours=index)),
                                             22.123456789 if index == 0 else 20 + index,
                                             40 + index, 100 + index, 600 + index,
                                             0 if group == "Room_B" and index == 9 else index % 2])))
                groups.append(group)
        path = self.root / "data/training_dataset.csv"
        self.write_csv(path, COLUMNS, rows)
        metadata = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sources": groups}}
        (path.parent / "dataset_sources.json").write_text(json.dumps(metadata), encoding="utf-8")
        return train.load_dataset(path)

    def test_merge_preserves_existing_roles_and_matches_rooms_by_full_record(self):
        original = dict(zip(COLUMNS, ["2015-01-01 10:00:00", 22, 40, 100, 600, 1]))
        for index, name in enumerate(("training_dataset.csv", "test_dataset.csv")):
            self.write_csv(self.root / "data" / name, COLUMNS, [{**original, "date": f"2015-01-0{index + 1} 10:00:00"}])
        before = {p.name: p.read_bytes() for p in (self.root / "data").glob("*.csv")}
        for offset, room in enumerate(("Room_A", "Room_B")):
            directory = self.root / "data/occupancy-detection-dataset/data/intermi" / room
            sensors, labels = [], []
            for index in range(10):
                date = (datetime(2023, 1, 1) + timedelta(hours=index)).isoformat() + "+00:00"
                sensors.append({"Date": date, "temperature": 20 + offset, "humidity": 40,
                                "light": 200 + offset, "co2": 700, "occupancy": 99})
                labels.append({"Date": date, "Image-based-occupancy": index % 2})
            self.write_csv(directory / "df_features.csv", sensors[0].keys(), sensors)
            self.write_csv(directory / "df_occ.csv", labels[0].keys(), labels)
        first = merge_data(self.root)
        self.assertEqual(first["appended_rows"], [16, 4])
        for name, content in before.items():
            self.assertTrue((self.root / "data" / name).read_bytes().startswith(content))
        metadata = json.loads((self.root / "data/dataset_sources.json").read_text())
        for name, record in metadata.items():
            with (self.root / "data" / name).open(newline="") as source:
                for row, group in zip(csv.DictReader(source), record["sources"]):
                    if group != "legacy":
                        self.assertEqual(float(row["Temperature"]), 20 if group == "Room_A" else 21)
                        self.assertIn(row["Occupancy"], ("0", "1"))
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root / "data").glob("*.*")}
        self.assertEqual(merge_data(self.root)["appended_rows"], [0, 0])
        self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root / "data").glob("*.*")})

    def test_source_splits_preserve_every_row_and_keep_timestamp_groups(self):
        source = self.fixture()
        # 80% 경계에 같은 출처·시각 행 두 개를 배치한다.
        source.dates[8] = source.dates[7]
        for ratio in (0.1, 0.2, 0.3, 0.4):
            fit, validation = train.split_training(source, ratio)
            self.assertEqual(len(fit.y) + len(validation.y), len(source.y))
            for group in set(source.sources):
                fit_dates = [date for name, date in zip(fit.sources, fit.dates) if name == group]
                validation_dates = [date for name, date in zip(validation.sources, validation.dates) if name == group]
                self.assertLess(max(fit_dates), min(validation_dates))
                self.assertTrue(fit_dates and validation_dates)

    def test_fit_limits_survive_csv_removal_and_keep_raw_precision(self):
        source = self.fixture()
        fit, validation = train.split_training(source)
        self.assertTrue(any(min(fit.dates) < date < max(fit.dates) for date in validation.dates))
        limits = train.fit_input_limits(fit, ("Temperature", "Hour", "DayOfWeek"))
        self.assertEqual(limits["features"]["Temperature"]["maximum"], 27)
        self.assertEqual(fit.raw[0]["Temperature"], 22.123456789)
        self.assertNotEqual(float(fit.x[0, 0]), fit.raw[0]["Temperature"])
        self.assertEqual(limits["features"]["Hour"]["values"], list(range(9)))
        self.assertEqual(max(row["Hour"] for row, group in zip(fit.raw, fit.sources) if group == "legacy"), 7)
        self.assertEqual(max(row["Hour"] for row, group in zip(fit.raw, fit.sources) if group == "Room_A"), 8)
        raw_limits = train.fit_input_limits(source.select([0, 1]), ("Temperature",))
        self.assertEqual(raw_limits["features"]["Temperature"]["maximum"], 22.123456789)
        train.write_json(self.root / "report.json", {"input_limits": raw_limits})
        torch.save({"input_limits": raw_limits}, self.root / "fixture.pt")
        checkpoint = torch.load(self.root / "fixture.pt", weights_only=True)
        (self.root / "data/training_dataset.csv").unlink()
        self.assertEqual(train.training_input_limits({"primary": checkpoint})["primary"], raw_limits)
        self.assertEqual(json.loads((self.root / "report.json").read_text())["input_limits"], raw_limits)

    def test_legacy_hash_fallback_normal_path_and_modified_source(self):
        source = self.fixture()
        # 연속 시간구간을 기록한 기존 형식의 고정 fixture를 재현한다.
        single = source.select(list(range(10)))
        path = self.root / "data/training_dataset.csv"
        self.write_csv(path, COLUMNS, [dict(zip(COLUMNS, [date.isoformat(sep=" "),
                        *[row[key] for key in COLUMNS[1:5]], int(label)]))
                        for date, row, label in zip(single.dates, single.raw, single.y)])
        data = {"file": "data/training_dataset.csv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        fit, _ = train.split_training(single)
        recorded = {"training_data": data, "training_split": fit.summary, "features": list(train.FEATURES)}
        restored = train.training_input_limits({"primary": recorded})["primary"]
        self.assertTrue(restored["available"])
        self.assertEqual(restored["features"], train.fit_input_limits(fit, train.FEATURES)["features"])
        path.write_bytes(path.read_bytes() + b"\n")
        rejected = train.training_input_limits({"primary": recorded})["primary"]
        self.assertFalse(rejected["available"])
        self.assertEqual(rejected["features"], {})

    def test_catalog_plan_reports_distribution_without_training(self):
        source = self.fixture()
        test = copy.deepcopy(source)
        test.dates = [date + timedelta(days=10) for date in source.dates]
        test.summary = train.summarize(test.y, test.dates, test.sources)
        plan = train.training_plan(source, test, 0.4)
        self.assertTrue(plan["ready"], plan)
        for group in ("legacy", "Room_A", "Room_B"):
            self.assertEqual(plan["fit"]["sources"][group]["rows"], 6)
            self.assertEqual(plan["validation"]["sources"][group]["rows"], 4)
        source.y[:] = 0
        self.assertFalse(train.training_plan(source, test, 0.2)["ready"])

    def test_sidecar_rejects_modified_csv(self):
        self.fixture()
        path = self.root / "data/training_dataset.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "출처 메타데이터"):
            train.load_dataset(path)

    def test_graz_local_features_use_dst_and_midnight_without_changing_utc_split_axis(self):
        cases = [("Room_A", "2023-01-01T23:30:00", 0, 0, "2023-01-01T23:30:00"),
                 ("Room_B", "2023-07-02T22:30:00", 0, 0, "2023-07-02T22:30:00"),
                 ("Room_A", "2023-10-29T00:30:00", 2, 6, "2023-10-29T00:30:00"),
                 ("Room_A", "2023-10-29T01:30:00", 2, 6, "2023-10-29T01:30:00"),
                 ("Room_B", "2023-03-26T00:30:00", 1, 6, "2023-03-26T00:30:00"),
                 ("Room_B", "2023-03-26T01:30:00", 3, 6, "2023-03-26T01:30:00"),
                 ("legacy", "2023-01-01T23:30:00", 23, 6, "2023-01-01T23:30:00"),
                 ("Room_B", "2023-07-03T01:30:00+02:00", 1, 0, "2023-07-02T23:30:00")]
        path = self.root / "data/training_dataset.csv"
        self.write_csv(path, COLUMNS, [dict(zip(COLUMNS, [stamp, 21, 40, 15, 430, 0])) for _, stamp, *_ in cases])
        (path.parent / "dataset_sources.json").write_text(json.dumps({path.name: {
            "sha256": digest(path), "sources": [case[0] for case in cases]}}))
        before = path.read_bytes()
        dataset = train.load_dataset(path)
        generated_records, provenance = read_training(self.root)
        generated_by_index = {row["canonical_index"]: row for rows in generated_records.values() for row in rows}
        for i, (_, _, hour, weekday, axis) in enumerate(cases):
            self.assertEqual(dataset.dates[i], datetime.fromisoformat(axis))
            self.assertEqual((dataset.raw[i]["Hour"], dataset.raw[i]["DayOfWeek"]), (hour, weekday))
            self.assertEqual(generated_by_index[i]["axis"], dataset.dates[i])
            local = generated_by_index[i]["local"]
            self.assertEqual((local.hour, local.weekday()), (hour, weekday))
            np.testing.assert_array_equal(dataset.x[i], np.array(model.input_vector(model.SensorInput(**dataset.raw[i])), dtype=np.float32))
        self.assertEqual(dataset.summary["time_feature_policy"], TIME_FEATURE_POLICY)
        self.assertEqual(dataset.select([0]).summary["time_feature_policy"], TIME_FEATURE_POLICY)
        self.assertEqual(provenance["time_feature_policy"], TIME_FEATURE_POLICY)
        self.assertEqual(path.read_bytes(), before)

    def test_synthetic_offset_features_keep_local_weekday_while_axis_is_utc(self):
        path = self.root / "synthetic.csv"
        self.write_csv(path, COLUMNS, [dict(zip(COLUMNS, [stamp, 21, 40, 15, 430, 0])) for stamp in
            ("2023-07-03T00:30:00+09:00", "2023-07-02T23:30:00-03:00")])
        dataset = train.load_dataset(path)
        self.assertEqual([(row["Hour"], row["DayOfWeek"]) for row in dataset.raw], [(0, 0), (23, 6)])
        self.assertEqual(dataset.dates, [datetime(2023, 7, 2, 15, 30), datetime(2023, 7, 3, 2, 30)])
        self.assertTrue(all(stamp.tzinfo is None for stamp in dataset.dates))

    def test_deferred_test_evaluation_never_computes_test_scores_during_fit(self):
        source = self.fixture()
        test = copy.deepcopy(source)
        test.dates = [stamp + timedelta(days=40) for stamp in test.dates]
        config = train.TrainingConfig(compare_light=False)
        prepared, validation, plan = train.prepare_training_data(source, test, config)
        size = len(model.encoded_features(model.FEATURES))
        with patch.object(train, "train_network", return_value=(Mock(state_dict=lambda: {}),
                {"mean": torch.zeros(size), "scale": torch.ones(size)}, {})), \
                patch.object(train, "probabilities", return_value=np.full(len(validation.y), .5)) as predict, \
                patch.object(train, "evaluate", return_value={"accuracy": .5}) as score:
            report, _ = train.train_variant("with_light", source, test, {}, config,
                prepared=(prepared, validation), data_plan=plan, defer_test_evaluation=True)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(score.call_count, 1)
        np.testing.assert_array_equal(predict.call_args.args[1], validation.x)
        self.assertIsNone(report["model_metrics"])
        self.assertIsNone(report["quality_gate"]["passed"])
        self.assertEqual(report["test_evaluation_status"], "pending")

    def test_reproduction_uses_same_encoded_column_layout_as_training(self):
        source = self.fixture()
        columns = list(range(source.x.shape[1]))
        selected = train.select_encoded_columns(source, columns)
        original = source.x[:, columns]
        self.assertEqual(selected.x.strides, original.strides)
        np.testing.assert_array_equal(selected.x, original)
        self.assertTrue(torch.equal(torch.from_numpy(selected.x).mean(dim=0), torch.from_numpy(original).mean(dim=0)))

    def test_new_metadata_records_policy_with_mock_training_and_legacy_prediction_keeps_input(self):
        source = self.fixture()
        test = copy.deepcopy(source)
        test.dates = [stamp + timedelta(days=40) for stamp in test.dates]
        test.summary = {**train.summarize(test.y, test.dates, test.sources), "time_feature_policy": TIME_FEATURE_POLICY}
        config = train.TrainingConfig(compare_light=False)
        prepared, validation, plan = train.prepare_training_data(source, test, config)
        self.assertEqual(plan["time_feature_policy"], TIME_FEATURE_POLICY)
        columns = len(model.encoded_features(model.FEATURES))
        fake = Mock(state_dict=lambda: {})
        with patch.object(train, "train_network", return_value=(fake, {"mean": torch.zeros(columns), "scale": torch.ones(columns)}, {})), \
                patch.object(train, "probabilities", side_effect=lambda network, values, scaler: np.full(len(values), .5)), \
                patch.object(train, "evaluate", return_value={"accuracy": .5, "f1": .5}), contextlib.redirect_stdout(io.StringIO()):
            report, checkpoint = train.train_variant("with_light", source, test, {}, config,
                prepared=(prepared, validation), data_plan=plan)
        self.assertEqual(report["time_feature_policy"], TIME_FEATURE_POLICY)
        self.assertEqual(checkpoint["time_feature_policy"], TIME_FEATURE_POLICY)
        # 기존 가중치는 읽기만 한다. 실제 추론은 가짜 network로 대체해 입력 계약만 확인한다.
        saved = model.load_model(model.weight_path("v7"))
        original_metadata = copy.deepcopy(saved.metadata)
        self.assertNotIn("time_feature_policy", saved.metadata)
        saved.network = Mock(return_value=torch.tensor([0.]))
        sensors = {**model.MOCK_SENSORS, "Hour": 23, "DayOfWeek": 6}
        output = model.predict_one(saved, sensors)
        encoding = saved.metadata.get("encoding", "sensor-only-v1")
        expected = (torch.tensor([model.input_vector(model.SensorInput(**sensors), saved.metadata["features"], encoding)]) - saved.mean) / saved.scale
        self.assertTrue(torch.equal(saved.network.call_args.args[0], expected))
        self.assertEqual(set(output), {"model_version", "occupancy", "label", "probability"})
        self.assertEqual(saved.metadata, original_metadata)
        legacy_checkpoint = {**checkpoint, "version": "v1"}
        old = copy.deepcopy(legacy_checkpoint)
        old.pop("time_feature_policy")
        old["state_dict"] = {"fixture": torch.zeros(1)}
        fingerprint = model.pair_fingerprint({"with_light": old})
        renamed = copy.deepcopy(old)
        renamed.update(version="v2", created_at_utc="different")
        self.assertEqual(fingerprint, model.pair_fingerprint({"with_light": renamed}))
        renamed["time_feature_policy"] = TIME_FEATURE_POLICY
        self.assertNotEqual(fingerprint, model.pair_fingerprint({"with_light": renamed}))

    def test_single_class_json_contract_with_stubbed_scores(self):
        # 점수 함수는 모두 대체한다. 실제 지표 계산 없이 null 처리와 JSON 계약만 확인한다.
        for name in ("accuracy_score", "precision_score", "recall_score", "f1_score",
                     "mean_absolute_error", "brier_score_loss"):
            patch.object(train, name, return_value=0.0).start()
        with patch.object(train, "roc_auc_score", side_effect=AssertionError("단일 클래스 ROC 호출")), \
                patch.object(train, "average_precision_score", side_effect=AssertionError("단일 클래스 AP 호출")), \
                patch.object(train, "confusion_matrix", return_value=np.array([[1, 0], [0, 0]])):
            result = EVALUATE_CONTRACT(np.array([0]), np.array([0.1]))
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["average_precision"])
        self.assertEqual(len(result["confusion_matrix"]), 2)
        json.dumps(result, allow_nan=False)

    def test_version_registry_exposes_evaluation_hash_and_rows(self):
        original = model.REPORTS_DIR / "v3/light_comparison.json"
        report = json.loads(original.read_text(encoding="utf-8"))
        for index, digest in enumerate(("a" * 64, "b" * 64), start=1):
            changed = copy.deepcopy(report)
            changed["model_version"] = f"v{index}"
            for variant in changed["modes"].values():
                variant["test_data"]["sha256"] = digest
            train.write_json(self.root / f"reports/v{index}/light_comparison.json", changed)
        with patch.object(model, "REPORTS_DIR", self.root / "reports"), patch.object(model, "WEIGHTS_DIR", self.root / "weights"):
            versions = model.model_versions("v1")["versions"]
        self.assertEqual([item["test_data"]["sha256"] for item in versions], ["b" * 64, "a" * 64])
        self.assertTrue(all(item["test_data"]["rows"] == item["test_rows"] for item in versions))

    def test_v7_weights_and_legacy_report_remain_readable(self):
        report = json.loads((model.REPORTS_DIR / "v7/light_comparison.json").read_text(encoding="utf-8"))
        for mode in report["modes"]:
            saved = model.load_model(model.weight_path("v7", mode))
            self.assertEqual(saved.metadata["version"], "v7")
            self.assertFalse(saved.network.training)
        limits = train.training_input_limits(report["modes"])
        for limit in limits.values():
            self.assertFalse(limit["available"])
            self.assertIn("이전 다중 데이터", limit["error"])


class SyntheticTrainingTests(unittest.TestCase):
    """임시 CSV·가짜 작업으로 입력 연결만 확인한다. 분류 학습·평가는 금지한다."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(train, "ROOT", self.root).start()
        for name in ("train_network", "evaluate", "probabilities"):
            patch.object(train, name, side_effect=AssertionError("분류 학습·평가 실행 금지")).start()
        simulation_fixture(self.root)
        path = self.root / "data/training_dataset.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            row["date"] = (datetime(2026, 10, 1) + timedelta(minutes=index * 30)).isoformat()
        DataTests.write_csv(self, path, COLUMNS, rows)
        test_path = path.with_name("test_dataset.csv")
        DataTests.write_csv(self, test_path, COLUMNS,
                            [{**row, "date": f"2050-01-01T00:{index:02d}:00"} for index, row in enumerate(rows[:4])])
        metadata = {item.name: {"sha256": digest(item), "sources": ["legacy"] * count}
                    for item, count in ((path, len(rows)), (test_path, 4))}
        (path.parent / "dataset_sources.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.source, self.test = train.load_dataset(path), train.load_dataset(test_path)
        self.output, _ = generate(SimulationConfig(rooms=3, step_seconds=300), self.root / "data/simulated", self.root)
        self.config = train.TrainingConfig(synthetic_generation=self.output.name, synthetic_ratio=0.5)

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))

    def write_manifest(self, value):
        (self.output / "manifest.json").write_text(json.dumps(value), encoding="utf-8")

    def test_real_partitions_fixed_only_train_room_added_seed_reproducible_and_logs_excluded(self):
        real, valid, baseline = train.prepare_training_data(self.source, self.test, train.TrainingConfig(shared_room_split=True))
        mixed, other_valid, plan = train.prepare_training_data(self.source, self.test, self.config)
        _, _, again = train.prepare_training_data(self.source, self.test, self.config)
        np.testing.assert_array_equal(mixed.x[:len(real.y)], real.x)
        np.testing.assert_array_equal(other_valid.x, valid.x)
        self.assertEqual(plan["test"], baseline["test"])
        self.assertEqual(plan["validation"], baseline["validation"])
        self.assertEqual(plan["fingerprint"], again["fingerprint"])
        synthetic = plan["synthetic"]
        self.assertEqual(synthetic["selected_rows"], int(len(real.y) * .5))
        self.assertEqual(plan["synthetic_fit_rows"], synthetic["selected_rows"])
        self.assertLessEqual(synthetic["fraction_of_combined_fit"], .5)
        self.assertEqual({name.rsplit(":", 1)[-1] for name in mixed.sources if name.startswith("synthetic:")}, {"room_01"})
        self.assertFalse(any(name.startswith("synthetic:") for name in other_valid.sources))
        self.assertEqual(plan["log_data"]["included_rows"], 0)
        self.assertIn("model_prediction", plan["log_data"]["reason"])
        self.assertIn("validation", synthetic["calibration_scope"])
        changed = self.config.model_copy(update={"seed": 43})
        next_plan = train.training_plan(self.source, self.test, .2, changed)
        self.assertNotEqual(synthetic["selected_indices_sha256"], next_plan["synthetic"]["selected_indices_sha256"])
        self.assertEqual(digest(self.root / "data/test_dataset.csv"), self.test.summary["sha256"])

    def test_catalog_excludes_incomplete_quality_failures_hash_and_metadata_mismatch(self):
        original = {path.name: path.read_bytes() for path in self.output.iterdir()}
        for failure in ("incomplete", "quality", "hash", "label", "heldout_hash", "path"):
            with self.subTest(failure=failure):
                manifest = self.manifest()
                if failure == "incomplete":
                    (self.output / "INCOMPLETE").touch()
                elif failure == "quality":
                    quality_path = self.output / "quality.json"
                    value = json.loads(quality_path.read_text())
                    value["passed"] = False
                    quality_path.write_text(json.dumps(value))
                    manifest["artifacts"]["quality.json"] = digest(quality_path)
                    self.write_manifest(manifest)
                elif failure in ("hash", "heldout_hash"):
                    (self.output / ("room_01.csv" if failure == "hash" else "room_02.csv")).write_text("broken")
                elif failure == "label":
                    path = self.output / "room_01.metadata.jsonl"
                    values = [json.loads(line) for line in path.read_text().splitlines()]
                    values[0]["virtual_people"] = int(not values[0]["virtual_people"])
                    path.write_text("\n".join(json.dumps(row) for row in values) + "\n")
                    manifest["files"]["room_01.csv"]["metadata_sha256"] = digest(path)
                    self.write_manifest(manifest)
                else:
                    manifest["files"]["room_01.csv"]["metadata"] = "../outside.jsonl"
                    self.write_manifest(manifest)
                catalog = train.synthetic_catalog(self.source)
                self.assertFalse(catalog[0]["available"])
                self.assertTrue(catalog[0]["error"])
                self.assertFalse(train.training_plan(self.source, self.test, .2, self.config)["ready"])
                for name, content in original.items():
                    (self.output / name).write_bytes(content)
                if (self.output / "INCOMPLETE").exists():
                    (self.output / "INCOMPLETE").unlink()
        for invalid in ("../outside", "C:\\outside", self.output.name + "/other"):
            with self.assertRaises(ValueError):
                train.synthetic_generation(invalid, self.source)

    def test_shared_rooms_split_whole_days_and_one_hour_gap(self):
        source = train.concatenate_datasets([self.source, self.source])
        source.sources = ["Room_A"] * len(self.source.y) + ["Room_B"] * len(self.source.y)
        fit, validation = train.split_training(source, .2, True)
        self.assertFalse(set(fit.dates) & set(validation.dates))
        self.assertFalse({date.date() for date in fit.dates} & {date.date() for date in validation.dates})
        self.assertGreater(min(validation.dates) - max(fit.dates), timedelta(hours=1))

    def test_worker_prepares_once_and_passes_same_actual_validation_to_both_variants(self):
        def fake_variant(*args, **kwargs):
            self.assertEqual(kwargs["data_plan"]["synthetic_fit_rows"], len(kwargs["prepared"][0].y) - kwargs["data_plan"]["real_fit_rows"])
            self.assertFalse(any(name.startswith("synthetic:") for name in kwargs["prepared"][1].sources))
            return {}, {}
        logger = Mock()
        with patch.object(train, "train_variant", side_effect=fake_variant) as variant, \
                patch.object(train, "save_model_pair", return_value="v99") as save:
            plan = train.training_plan(self.source, self.test, .2, self.config)
            result = train.train_experiment(self.config, weights_dir=self.root / "weights", reports_dir=self.root / "reports",
                                           logger=logger, expected_plan=plan)
            self.assertEqual(result, "v99")
            self.assertEqual(variant.call_count, 2)
            self.assertEqual(save.call_args.kwargs["training_report"]["data_plan"], plan)
            with self.assertRaisesRegex(ValueError, "제출 후"):
                train.train_experiment(self.config, weights_dir=self.root / "weights", logger=logger,
                                       expected_plan={"fingerprint": "0" * 64})
            self.assertEqual(variant.call_count, 2)

    def test_options_and_mock_submission_store_plan_and_reject_stale_preview(self):
        from fastapi.testclient import TestClient
        import backend.app as api
        patch.object(api, "ROOT", self.root).start()
        patch.object(api, "JOBS_DIR", self.root / "jobs").start()
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch.object(api.subprocess, "Popen", return_value=process) as spawn, \
                TestClient(api.create_app(log_root=self.root / "logs", state_dir=self.root / "state",
                                          enable_classrooms=False, start_simulation=False)) as client:
            query = {key: getattr(self.config, key) for key in ("synthetic_generation", "synthetic_ratio", "seed", "shared_room_split")}
            catalog = client.get("/training/options", params=query).json()
            self.assertTrue(catalog["synthetic_datasets"][0]["available"])
            plan = catalog["data_plan"]
            self.assertTrue(plan["ready"], plan)
            response = client.post("/training/jobs", json=self.config.model_dump(), headers={"X-Data-Plan-SHA256": "stale"})
            self.assertEqual(response.status_code, 409)
            spawn.assert_not_called()
            response = client.post("/training/jobs", json=self.config.model_dump(), headers={"X-Data-Plan-SHA256": plan["fingerprint"]})
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["data_plan"], plan)
            spawn.assert_called_once()
            log = list((self.root / "logs").glob("*/model.jsonl"))[0].read_text()
            self.assertIn(plan["fingerprint"], log)

    def test_variant_metadata_limits_and_version_report_use_combined_fit_only(self):
        training, validation, plan = train.prepare_training_data(self.source, self.test, self.config)
        scores = {"accuracy": .5, "precision": .5, "recall": .5, "f1": .5}
        network = Mock()
        info = {"best_epoch": 1, "epochs_run": 1, "best_validation_loss": 1., "history": []}
        with patch.object(train, "train_network", return_value=(network, {}, info)) as fit, \
                patch.object(train, "probabilities", side_effect=lambda network, x, scaler: np.zeros(len(x))), \
                patch.object(train, "evaluate", return_value=scores), patch("builtins.print"):
            report, checkpoint = train.train_variant("with_light", self.source, self.test, {}, self.config,
                features=["CO2", "Hour"], prepared=(training, validation), data_plan=plan)
        self.assertEqual(fit.call_args.args[0].x.shape, (len(training.y), 3))
        self.assertEqual(fit.call_args.args[1].x.shape, (len(validation.y), 3))
        self.assertEqual(checkpoint["input_limits"]["rows"], len(training.y))
        self.assertEqual(checkpoint["data_plan"], plan)
        self.assertEqual(checkpoint["split_strategy"], train.SHARED_SPLIT_STRATEGY)
        self.assertEqual(checkpoint["input_limits"]["split_strategy"], train.SHARED_SPLIT_STRATEGY)
        bundle = {"model_version": "v99", "config": self.config.model_dump(), "models": {"with_light": report}, "data_plan": plan}
        directory = self.root / "reports/v99"
        train.write_training_report(directory, bundle)
        self.assertIn(self.output.name, (directory / "training.md").read_text())
        self.assertIn(plan["synthetic"]["manifest_sha256"], (directory / "training.json").read_text())
