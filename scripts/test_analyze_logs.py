import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_logs import analyze


class LogAnalysisTests(unittest.TestCase):
    def test_model_prediction_and_claimed_truth_are_not_training_labels_and_raw_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.jsonl"
            row = {"event_type": "inference", "source": "simulation", "label_source": "model_prediction",
                   "inference_id": "same", "timestamp_utc": "2026-11-17T00:00:00Z", "ground_truth": 0,
                   "sensors": {"Temperature": 22, "Humidity": 40, "Light": 500, "CO2": 800}, "secret": "must-not-appear"}
            path.write_bytes((json.dumps(row) + "\n" + json.dumps(row) + '\n{"partial":').encode())
            original = path.read_bytes()
            result = analyze(temporary, "test_fixture")
            self.assertEqual(result["training_candidate_rows"], 0)
            self.assertEqual(result["counts"]["unlabelled_prediction_rows"], 2)
            self.assertEqual(result["counts"]["duplicate_inference_ids"], 1)
            self.assertEqual(result["counts"]["parse_errors"], 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertNotIn("must-not-appear", json.dumps(result))

    def test_only_verified_zero_truth_with_matching_time_input_and_train_partition_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            canonical = {"key": "one", "row_id": "one", "room": "room_01", "partition": "train",
                "observed_at": "2026-11-17T09:00:00+09:00", "truth": 0, "virtual_people": 0,
                "sensors": {"Temperature": 22, "Humidity": 40, "Light": 500, "CO2": 800}}
            row = {"event_type": "pipeline_observation", "source": "simulation_pipeline",
                "label_source": "verified_virtual_people_gt_zero", "inference_id": "id",
                "timestamp_utc": "2026-11-17T00:00:00Z", "observation": {**canonical, "inference_id": "id"}}
            path = Path(temporary) / "model.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(analyze(temporary, "test_fixture")["training_candidate_rows"], 0)
            result = analyze(temporary, "test_fixture", {"one": canonical})
            self.assertEqual(result["training_candidate_rows"], 1)
            self.assertEqual(result["counts"]["verified_zero_truth_rows"], 1)
            row["observation"]["observed_at"] = "2026-11-17T10:00:00+09:00"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(analyze(temporary, "test_fixture", {"one": canonical})["training_candidate_rows"], 0)
            row["observation"] = {**canonical, "partition": "test", "inference_id": "id"}
            path.write_text(json.dumps(row) + "\n")
            heldout = {**canonical, "partition": "test"}
            result = analyze(temporary, "test_fixture", {"one": heldout})
            self.assertEqual(result["training_candidate_rows"], 0)
            self.assertEqual(result["counts"]["heldout_excluded_rows"], 1)
