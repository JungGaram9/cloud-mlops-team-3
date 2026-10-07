"""PyTorch 모델의 품질·재현성과 Swagger·HTTP 계약을 함께 검증한다."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from backend.app import create_app
from models.model import (FEATURES, FEATURE_SETS, MIN_ACCURACY, MOCK_SENSORS, REPORTS_DIR,
                          ROOT, SENSOR_FEATURES, WEIGHTS_DIR, SensorInput, current_version,
                          input_vector, load_model, pair_fingerprint, predict_one, weight_path)
from models.train import load_dataset, require_accuracy, save_model_pair, split_training

VERSION = current_version()
MODEL_PATH = weight_path(VERSION)
NO_LIGHT_PATH = weight_path(VERSION, "without_light")
REPORT_PATH = REPORTS_DIR / VERSION / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / VERSION / "light_comparison.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ModelTests(unittest.TestCase):
    def test_without_light_ignores_changed_light(self):
        model = load_model(NO_LIGHT_PATH)
        self.assertNotIn("Light", model.metadata["features"])
        self.assertEqual(predict_one(model, {**MOCK_SENSORS, "Light": 0.0}),
                         predict_one(model, {**MOCK_SENSORS, "Light": 1000.0}))

    def test_comparison_uses_same_data_and_split(self):
        modes = read_json(COMPARISON_PATH)["modes"]
        for key in ("training_data", "training_split", "validation_split", "test_data"):
            self.assertEqual(modes["with_light"][key], modes["without_light"][key])
        self.assertEqual(modes["with_light"]["features"], list(FEATURES))
        self.assertEqual(modes["without_light"]["features"], list(FEATURE_SETS["without_light"]))

    def test_saved_pytorch_model_returns_same_prediction(self):
        first = predict_one(load_model(), MOCK_SENSORS)
        self.assertEqual(first, predict_one(load_model(), MOCK_SENSORS))
        self.assertEqual(first["model_version"], VERSION)
        self.assertGreaterEqual(first["probability"], 0)
        self.assertLessEqual(first["probability"], 1)
        self.assertEqual(first["occupancy"], int(first["probability"] >= 0.5))
        self.assertEqual(load_model().metadata["framework"], "pytorch")
        self.assertFalse(load_model().network.training)

    def test_scaler_uses_training_partition_only(self):
        source = load_dataset(ROOT / "data/training_dataset.csv")
        training, validation = split_training(source)
        self.assertLess(max(training.dates), min(validation.dates))
        self.assertEqual(len(training.y) + len(validation.y), len(source.y))
        self.assertTrue(torch.allclose(load_model().mean, torch.from_numpy(training.x).mean(dim=0)))

    def test_evaluation_meets_85_percent(self):
        report = read_json(REPORT_PATH)
        self.assertGreaterEqual(report["model_metrics"]["accuracy"], MIN_ACCURACY)
        self.assertTrue(report["quality_gate"]["passed"])
        self.assertEqual(report["data_audit"]["shared_timestamps"], 0)

    def test_quality_gate_rejects_bad_accuracy(self):
        for accuracy in (0.849, float("nan")):
            with self.subTest(accuracy=accuracy), self.assertRaises(ValueError):
                require_accuracy({"accuracy": accuracy})
        require_accuracy({"accuracy": 0.85})

    def test_missing_model_explains_how_to_train(self):
        with self.assertRaisesRegex(FileNotFoundError, "먼저"):
            load_model(MODEL_PATH.parent / "missing.pt")

    def test_invalid_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pt"
            checkpoint = torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
            checkpoint["scale"][0] = 0
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "표준화"):
                load_model(path)

    def test_data_rejects_invalid_labels_and_sensor_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for label, temperature in (("0.5", "22"), ("nan", "22"), ("1", "inf"), ("1", "100")):
                with self.subTest(label=label, temperature=temperature):
                    with path.open("w", encoding="utf-8", newline="") as source:
                        writer = csv.writer(source)
                        writer.writerow(["date", *SENSOR_FEATURES, "Occupancy"])
                        writer.writerow(["2015-02-02 10:00:00", temperature, 45, 450, 900, label])
                    with self.assertRaises(ValueError):
                        load_dataset(path)

    def test_time_fields_from_csv_use_same_inference_transform(self):
        source = load_dataset(ROOT / "data/training_dataset.csv")
        with (ROOT / "data/training_dataset.csv").open(encoding="utf-8-sig", newline="") as file:
            row = next(csv.DictReader(file))
        date = source.dates[0]
        sensors = SensorInput(**{name: float(row[name]) for name in SENSOR_FEATURES},
                              Hour=date.hour, DayOfWeek=date.weekday())
        expected = torch.tensor(input_vector(sensors), dtype=torch.float32)
        self.assertTrue(torch.equal(torch.from_numpy(source.x[0]), expected))
        self.assertIn("Hour", load_model().metadata["features"])
        self.assertIn("DayOfWeek", load_model().metadata["features"])

    def test_time_and_weekday_change_current_model_prediction(self):
        # 범위를 벗어난 분포에서는 ReLU 출력이 같을 수 있으므로 실제 측정값으로 확인한다.
        model = load_model(NO_LIGHT_PATH)
        with (ROOT / "data/training_dataset.csv").open(encoding="utf-8-sig", newline="") as source:
            row = next(csv.DictReader(source))
        sensors = {**MOCK_SENSORS, **{name: float(row[name]) for name in SENSOR_FEATURES}}
        hourly = {predict_one(model, {**sensors, "Hour": hour})["probability"] for hour in range(24)}
        daily = {predict_one(model, {**sensors, "DayOfWeek": day})["probability"] for day in range(7)}
        self.assertGreater(len(hourly), 1)
        self.assertGreater(len(daily), 1)

    def test_model_identity_ignores_version_name_and_timestamp(self):
        pair = {mode: torch.load(weight_path(VERSION, mode), weights_only=True, map_location="cpu") for mode in FEATURE_SETS}
        signature = pair_fingerprint(pair)
        for checkpoint in pair.values():
            checkpoint["version"] = "v100"
            checkpoint["created_at_utc"] = "다른 시각"
        self.assertEqual(pair_fingerprint(pair), signature)
        pair["with_light"]["mean"][0] += 0.01
        self.assertNotEqual(pair_fingerprint(pair), signature)

    def test_registering_pair_preserves_history_and_reuses_same_version(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, reports = Path(directory) / "weights", Path(directory) / "reports"
            pair = {mode: torch.load(weight_path(VERSION, mode), weights_only=True, map_location="cpu") for mode in FEATURE_SETS}
            evaluations = read_json(COMPARISON_PATH)["modes"]
            first = save_model_pair(pair, evaluations, weights, reports)
            path = weights / first / "model.pt"
            original, modified_at = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(save_model_pair(pair, evaluations, weights, reports), first)
            self.assertEqual(path.stat().st_mtime_ns, modified_at)
            pair["with_light"]["mean"][0] += 0.01
            second = save_model_pair(pair, evaluations, weights, reports)
            self.assertNotEqual(first, second)
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue((weights / second / "model_without_light.pt").is_file())


class ApiTests(unittest.TestCase):
    def test_versions_group_two_models_under_current_version(self):
        with TestClient(create_app()) as client:
            response = client.get("/versions")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["current_version"], VERSION)
            pair = next(item for item in response.json()["versions"] if item["current"])
            self.assertEqual(pair["version"], VERSION)
            self.assertEqual([item["mode"] for item in pair["models"]], ["with_light", "without_light"])
            self.assertTrue(all(item["checkpoint_exists"] for item in pair["models"]))

    def test_compare_returns_actual_two_model_predictions(self):
        with TestClient(create_app()) as client:
            response = client.post("/compare", json=MOCK_SENSORS)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["with_light"], predict_one(load_model(), MOCK_SENSORS))
            self.assertEqual(response.json()["without_light"], predict_one(load_model(NO_LIGHT_PATH), MOCK_SENSORS))

    def test_comparison_endpoint_matches_report(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get("/comparison").json(), read_json(COMPARISON_PATH))

    def test_api_prediction_matches_saved_model(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get("/health").json(), {
                "status": "ok", "model_version": VERSION, "framework": "pytorch",
            })
            response = client.post("/predict", json=MOCK_SENSORS)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), predict_one(load_model(), MOCK_SENSORS))

    def test_swagger_and_assets_are_served_locally(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get("/", follow_redirects=False).headers["location"], "/docs")
            docs = client.get("/docs")
            self.assertEqual(docs.status_code, 200)
            self.assertIn("/static/swagger/swagger-ui-bundle.js", docs.text)
            self.assertNotIn("cdn.jsdelivr.net", docs.text)
            for asset in ("swagger-ui-bundle.js", "swagger-ui.css", "favicon-32x32.png"):
                with self.subTest(asset=asset):
                    self.assertEqual(client.get("/static/swagger/" + asset).status_code, 200)

    def test_openapi_example_and_error_status_match_api(self):
        with TestClient(create_app()) as client:
            schema = client.get("/openapi.json").json()
            self.assertEqual(schema["components"]["schemas"]["SensorInput"]["examples"][0], MOCK_SENSORS)
            responses = schema["paths"]["/predict"]["post"]["responses"]
            self.assertIn("400", responses)
            self.assertNotIn("422", responses)

    def test_metrics_endpoint_matches_report(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get("/metrics").json(), read_json(REPORT_PATH))

    def test_input_rejects_invalid_types_and_ranges(self):
        cases = [("Temperature", "22.5"), ("Temperature", True), ("Temperature", 61),
                 ("Humidity", -1), ("Humidity", 101), ("Light", -1),
                 ("CO2", 0), ("CO2", None), ("Light", 1e100), ("CO2", 1e100),
                 ("Hour", -1), ("Hour", 24), ("Hour", 14.5), ("Hour", True),
                 ("DayOfWeek", -1), ("DayOfWeek", 7), ("DayOfWeek", "2")]
        with TestClient(create_app()) as client:
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    response = client.post("/predict", json={**MOCK_SENSORS, field: value})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.json())

    def test_missing_and_extra_fields_are_rejected(self):
        with TestClient(create_app()) as client:
            missing = {name: value for name, value in MOCK_SENSORS.items() if name != "CO2"}
            for body in (missing, {**MOCK_SENSORS, "Occupancy": 1}, [MOCK_SENSORS]):
                with self.subTest(body=body):
                    self.assertEqual(client.post("/predict", json=body).status_code, 400)

    def test_malformed_and_nonfinite_json_are_rejected(self):
        with TestClient(create_app()) as client:
            bodies = ["{", json.dumps({**MOCK_SENSORS, "CO2": float("nan")}),
                      json.dumps({**MOCK_SENSORS, "CO2": float("inf")})]
            for body in bodies:
                with self.subTest(body=body):
                    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.json())

    def test_missing_model_stops_startup(self):
        with self.assertRaises(FileNotFoundError):
            with TestClient(create_app(MODEL_PATH.parent / "missing.pt")):
                pass

    def test_historical_pair_uses_its_own_weights_and_ignores_unused_time(self):
        if not (WEIGHTS_DIR / "v2/model.pt").is_file():
            self.skipTest("이전 v2 가중치가 없는 독립 실행 환경")
        with TestClient(create_app()) as client:
            response = client.post("/compare?version=v2", json=MOCK_SENSORS)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["with_light"], predict_one(load_model(weight_path("v2")), MOCK_SENSORS))
            changed = client.post("/compare?version=v2", json={**MOCK_SENSORS, "Hour": 2, "DayOfWeek": 6})
            self.assertEqual(response.json(), changed.json())
            self.assertEqual(response.json()["without_light"]["model_version"], "v2")
            self.assertEqual(client.get("/comparison?version=v2").json()["model_version"], "v2")

    def test_invalid_or_unavailable_version_is_rejected(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.post("/compare?version=v999999", json=MOCK_SENSORS).status_code, 404)
            self.assertEqual(client.get("/comparison?version=invalid").status_code, 400)


if __name__ == "__main__":
    unittest.main()
