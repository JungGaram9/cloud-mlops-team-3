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
                              FAULT_KINDS, EVENT_KINDS, generate_room, observe_room, preprocess_observation,
                              read_export, read_raw_csv, validate_observation_exports, sensor_step, timeline, transitions)


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

    def test_dirty_observations_are_reproducible_audited_and_never_change_virtual_truth(self):
        cfg = Config(rooms=1, days=7, step_seconds=300, observation_mode="stress", fault_min_minutes=30, fault_max_minutes=60)
        path, report = generate(cfg, self.root / "dirty", self.root)
        raw = read_raw_csv((path / "room_01.raw.csv").read_text())
        clean, meta = read_export(path / "room_01.csv", path / "room_01.metadata.jsonl")
        audit = [json.loads(line) for line in (path / "room_01.observations.jsonl").read_text().splitlines()]
        summary = validate_observation_exports(raw, clean, meta, audit, 300)
        self.assertEqual(set(summary["fault_rows"]), set(FAULT_KINDS))
        self.assertGreater(summary["collection_interruption_rows"], 0)
        self.assertGreater(summary["excluded_rows"], summary["collection_interruption_rows"])
        self.assertGreater(summary["clean"]["gap_pairs"], 0)
        self.assertTrue(any(item["status"] == "excluded" and item["metadata"]["virtual_people"] == 0 for item in audit))
        self.assertTrue(any(item["sensor_faults"] and row[-1] == 0 for item, row in zip(meta, clean)))
        self.assertTrue(all(np.isfinite(row[1:5]).all() for row in clean))
        self.assertTrue(any("freeze" in item["sensor_faults"] for item in meta))
        self.assertTrue(any("drift" in item["sensor_faults"] for item in meta))
        self.assertTrue(report["passed"])
        self.assertFalse(report["realism_guarantee"])
        self.assertEqual(report["rooms"]["room_01"]["rows"], len(clean))
        self.assertEqual(report["observation"]["room_01"]["raw"], summary["raw"])
        physical, truth_meta, _, traits = generate_room(cfg, self.profile, 0, timeline(cfg))
        off = generate_room(replace(cfg, observation_mode="off"), self.profile, 0, timeline(cfg))
        self.assertEqual(physical, off[0])
        self.assertEqual([item["virtual_people"] for item in truth_meta], [item["metadata"]["virtual_people"] for item in audit])
        source = self.profile["sources"][traits["calibration_source"]]
        first = observe_room(physical, truth_meta, cfg, source, 0)
        self.assertEqual(first, observe_room(physical, truth_meta, cfg, source, 0))
        self.assertNotEqual(first[0], observe_room(physical, truth_meta, replace(cfg, seed=cfg.seed + 1), source, 0)[0])
        audit[0]["clean_values"][0] += 1
        with self.assertRaisesRegex(ValueError, "전처리"):
            validate_observation_exports(raw, clean, meta, audit, 300)

    def test_preprocessing_excludes_only_explicitly_unusable_values_without_imputation(self):
        self.assertEqual(preprocess_observation([21, 40, 0, 430]), ([21., 40., 0., 430.], []))
        for values, reason in ((["", 40, 0, 430], "missing"), ([21, "nan", 0, 430], "nonfinite"),
                               ([21, 40, "inf", 430], "nonfinite"), ([21, 40, -1, 430], "outside_envelope")):
            result, reasons = preprocess_observation(values)
            self.assertIsNone(result)
            self.assertIn(reason, reasons[0])
        self.assertEqual(preprocess_observation([], False), (None, ["collection_interruption"]))
        self.assertEqual(preprocess_observation([21, 40, 900, 900])[1], [])

    def test_every_room_has_independent_exception_episodes_and_fit_observation_statistics(self):
        cfg = Config(rooms=3, days=7, step_seconds=300)
        episode_ids, sensor_traces = set(), []
        for room_index in range(3):
            rows, metadata, events, _ = generate_room(cfg, self.profile, room_index, timeline(cfg))
            special = [event for event in events if event["kind"] in EVENT_KINDS]
            self.assertEqual({event["kind"] for event in special}, set(EVENT_KINDS))
            self.assertFalse(episode_ids & {event["id"] for event in special})
            episode_ids.update(event["id"] for event in special)
            self.assertTrue(all(any(event["id"] in item["event_ids"] for item in metadata) for event in special))
            sensor_traces.append([row[1:5] for row in rows])
        self.assertFalse(np.array_equal(sensor_traces[0], sensor_traces[1]))
        stats = self.profile["sources"]["legacy"]["observation_calibration"]
        self.assertEqual(stats["measured"]["rows"], len(self.records["legacy"]))
        self.assertIn("assumptions", stats["fault_rates"])


if __name__ == "__main__":
    unittest.main()
