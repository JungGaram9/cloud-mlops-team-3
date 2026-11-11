"""독립 생성기의 학습 경계·동역학·재현·사건 정답·복제 검사를 검증한다."""

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.simulate import (COLUMNS, Config, calibration, digest, generate, quality, read_training,
                              sensor_step, timeline, transitions)


def fixture(root, auxiliary=False):
    data = root / "data"
    data.mkdir(parents=True)
    rows, sources = [], []
    groups = ("legacy", "Room_A") if auxiliary else ("legacy",)
    for group in groups:
        start = datetime(2026, 10, 1, 8)
        co2 = 430.
        group_rows = []
        for index in range(420):
            label = int(index % 100 in range(20, 75))
            co2 += 8 * label - .025 * (co2 - 430)
            stamp = start + timedelta(minutes=index * (5 if group != "legacy" else 1))
            row = [stamp.isoformat(sep=" "), 21 + .3 * np.sin(index / 20) + .2 * label,
                   40 + .5 * np.cos(index / 40) - .3 * label, 10 + 400 * label + .2 * index,
                   co2, label]
            rows.append(row)
            group_rows.append(row)
            sources.append(group)
        if group == "Room_A":
            directory = data / "occupancy-detection-dataset/data/intermi/Room_A"
            directory.mkdir(parents=True)
            with (directory / "df_occ.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Date", "Image-based-occupancy"])
                for row in group_rows:
                    writer.writerow([row[0] + "+00:00", row[-1] * 2])
                writer.writerow(["2099-01-01 00:00:00+00:00", "not selected"])
            with (directory / "df_features.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Date", "temperature", "humidity", "light", "co2", "openDoors", "openWindows", "occupancy"])
                for row in group_rows:
                    writer.writerow([row[0] + "+00:00", *row[1:5], 0, 1, 9999])
                writer.writerow(["2099-01-01 00:00:00+00:00", *(["not selected"] * 7)])
    with (data / "training_dataset.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    (data / "dataset_sources.json").write_text(json.dumps({"training_dataset.csv": {
        "sha256": digest(data / "training_dataset.csv"), "sources": sources}}), encoding="utf-8")
    (data / "test_dataset.csv").write_text("평가 자료는 파싱할 수 없는 내용이어도 열지 않아야 한다.", encoding="utf-8")


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)
        self.records, self.provenance = read_training(self.root)
        self.profile = calibration(self.records, self.provenance)

    def test_training_hash_and_auxiliary_membership_use_image_truth_only(self):
        other = self.root / "aux"
        fixture(other, auxiliary=True)
        records, provenance = read_training(other)
        self.assertEqual({row["count"] for row in records["Room_A"]}, {0, 2})
        self.assertTrue(all(row["open"] == 1 for row in records["Room_A"]))
        self.assertEqual(provenance["auxiliary"]["Room_A"]["matched_image_rows"], 420)
        self.assertEqual(records["Room_A"][0]["local"].hour, 10)
        self.assertIsNone(records["legacy"][0]["local"].tzinfo)
        path = other / "data/training_dataset.csv"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "해시"):
            read_training(other)

    def test_time_units_and_calibration_changes_reach_generation(self):
        source = self.profile["sources"]["legacy"]
        self.assertEqual(source["cadence_seconds"], 60)
        self.assertEqual(source["valid_transition_pairs"], 419)
        original = np.array(source["center"])
        for row in self.records["legacy"]:
            row["x"][0] += 5
        changed = calibration(self.records, self.provenance)
        self.assertAlmostEqual(changed["sources"]["legacy"]["center"][0] - original[0], 5)
        from scripts.simulate import generate_room
        cfg = Config(mode="normal", rooms=1, days=1, step_seconds=300, event_probability=0)
        first = generate_room(cfg, self.profile, 0, timeline(cfg))[0]
        second = generate_room(cfg, changed, 0, timeline(cfg))[0]
        self.assertGreater(np.mean([b[1] - a[1] for a, b in zip(first, second)]), 4)

    def test_light_does_not_create_people_or_co2_and_exit_retains_co2(self):
        source = self.profile["sources"]["legacy"]
        target = np.array([21., 40., 10., 430.])
        initial = np.array([21., 40., 10., 430.])
        dark = sensor_step(initial, target, 0, 0, 0, np.zeros(4), source, 90, 1)
        bright = sensor_step(initial, target, 0, 600, 0, np.zeros(4), source, 90, 1)
        self.assertEqual(dark[3], bright[3])
        self.assertGreater(bright[2], dark[2])
        state = initial
        for _ in range(90):
            state = sensor_step(state, target, 3, 0, 0, np.zeros(4), source, 90, 1)
        after_exit = sensor_step(state, target, 0, 600, 0, np.zeros(4), source, 90, 1)
        self.assertGreater(after_exit[3], 480)
        self.assertLess(after_exit[3], state[3])
        for humidity in (30, 50):
            ambient = target.copy()
            ambient[1] = humidity
            step = sensor_step(initial, ambient, 3, 0, 0, np.zeros(4), source, 90, 1)
            self.assertEqual(np.sign(step[1] - 40), np.sign(humidity - 40))

    def test_reproducibility_no_overwrite_and_all_exception_truths(self):
        cfg = Config(rooms=1, days=7, step_seconds=300)
        first, report = generate(cfg, self.root / "out1", self.root)
        second, _ = generate(cfg, self.root / "out2", self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["actual_exception_counts"]), 7)
        self.assertGreater(report["rooms"]["room_01"]["occupied_weekend_rows"], 0)
        self.assertGreater(report["rooms"]["room_01"]["occupied_night_rows"], 0)
        for name in ("room_01.csv", "room_01.metadata.jsonl", "manifest.json", "profile.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertFalse((first / "INCOMPLETE").exists())
        with self.assertRaises(FileExistsError):
            generate(cfg, self.root / "out1", self.root)

    def test_dst_day_length_and_strict_utc_order(self):
        cfg = Config(mode="normal", rooms=1, days=1, timezone="Europe/Vienna", step_seconds=300)
        for day, hours in (("2026-10-25", 25), ("2026-03-29", 23)):
            stamps = timeline(replace(cfg, start_date=day))
            self.assertEqual(len(stamps), hours * 12)
            self.assertTrue(all((b - a).total_seconds() == 300 for a, b in zip(stamps, stamps[1:])))

    def test_sampling_rates_exclude_midnight_and_gaps(self):
        origin = datetime(2026, 10, 1, 23, 50)
        rows = [{"axis": origin + timedelta(minutes=minute), "local": origin + timedelta(minutes=minute)}
                for minute in (0, 5, 10, 15, 40, 45)]
        cadence, pairs = transitions(rows)
        self.assertEqual(cadence, 300)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(pair[2] == 5 for pair in pairs))
        for index, row in enumerate(self.records["legacy"]):
            row["axis"] = datetime(2026, 10, 1) + timedelta(minutes=index * 5)
            row["local"] = row["axis"]
            row["x"][0] = 20 + index * .05
        profile = calibration(self.records, self.provenance)["sources"]["legacy"]
        self.assertEqual(profile["cadence_seconds"], 300)
        self.assertAlmostEqual(profile["minute_change"]["mean"][0], .01)

    def test_shifted_timestamps_cannot_hide_exact_or_near_sensor_sequence_copy(self):
        cfg = Config(mode="normal", rooms=1, days=1)
        self.profile["room_traits"] = {"room_01": {"calibration_source": "legacy"}}
        for jitter in (0., .001):
            rows, meta = [], []
            for index, original in enumerate(self.records["legacy"][:120]):
                stamp = original["axis"].replace(year=2050, tzinfo=timezone.utc)
                rows.append((stamp.isoformat(), *(original["x"] + jitter), original["y"]))
                meta.append({"row_index": index, "csv_line": index + 2, "csv_file": "room_01.csv", "row_id": str(index),
                    "synthetic": True, "virtual_people": original["y"] * 2, "event_ids": []})
            report = quality({"room_01": rows}, {"room_01": meta}, {"room_01": []}, self.records, self.profile, cfg)
            self.assertFalse(report["passed"])
            self.assertTrue(any("복사" in error or "복제" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
