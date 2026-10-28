"""학습·추론 없이 데이터 분할과 모델 메타데이터 계약을 확인한다."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models import model, train
from scripts.merge_data import COLUMNS, merge_data

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
        self.assertEqual(limits["features"]["Hour"]["values"], list(range(8)))
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
