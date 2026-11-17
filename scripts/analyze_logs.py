"""원시 로그를 변경하지 않고 진단한다. 외부 정답 대조 없는 예측값은 학습 후보가 아니다."""

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SENSORS = ("Temperature", "Humidity", "Light", "CO2")
SOURCES = {"real", "simulation", "experiment", "simulation_pipeline"}
EVENTS = {"request", "lifecycle", "inference", "pipeline_observation", "training", "measurement", "patrol"}


def analyze(directory, origin, verified=None):
    if origin not in ("ec2_runtime", "local_runtime", "synthetic_experiment", "test_fixture"):
        raise ValueError("로그 출처를 명시하세요.")
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("로그 폴더가 없습니다.")
    verified = verified or {}
    counts, events, sources, labels, missing = Counter(), Counter(), Counter(), Counter(), Counter()
    seen, selected, timestamps, inputs = set(), set(), [], []
    for path in sorted(directory.rglob("*.jsonl")):
        content = path.read_bytes()
        inputs.append({"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
        for line in content.splitlines():
            if not line.strip():
                continue
            counts["nonempty_lines"] += 1
            try:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
                if not isinstance(row, dict):
                    raise ValueError("object required")
            except (ValueError, UnicodeError):
                counts["parse_errors"] += 1
                continue
            counts["parsed_rows"] += 1
            event = row.get("event_type")
            event = event if isinstance(event, str) and event in EVENTS else "other"
            events[event] += 1
            try:
                stamp = datetime.fromisoformat(row.get("timestamp_utc"))
                if stamp.tzinfo is None:
                    counts["timezone_unknown_rows"] += 1
                else:
                    timestamps.append(stamp.astimezone(timezone.utc))
            except (ValueError, TypeError):
                counts["invalid_log_timestamps"] += 1
            if event not in ("inference", "pipeline_observation"):
                continue
            counts["prediction_rows"] += 1
            source = row.get("source")
            sources[source if isinstance(source, str) and source in SOURCES else "other"] += 1
            label = row.get("label_source")
            labels[label if isinstance(label, str) and label in ("model_prediction", "verified_virtual_people_gt_zero") else "other"] += 1
            observation = row.get("observation", {}) if event == "pipeline_observation" else {}
            if not isinstance(observation, dict):
                observation = {}
            sensors = observation.get("sensors", row.get("sensors", {}))
            sensors = sensors if isinstance(sensors, dict) else {}
            unusable = False
            for name in SENSORS:
                value = sensors.get(name)
                if value is None or value == "":
                    missing[name] += 1
                    unusable = True
                else:
                    try:
                        if isinstance(value, bool) or not math.isfinite(float(value)):
                            raise ValueError("invalid sensor")
                    except (TypeError, ValueError, OverflowError):
                        counts["invalid_sensor_values"] += 1
                        unusable = True
            identifier = row.get("inference_id")
            if not isinstance(identifier, str) or not identifier:
                counts["missing_inference_ids"] += 1
                duplicate = False
            else:
                duplicate = identifier in seen
                counts["duplicate_inference_ids"] += int(duplicate)
                seen.add(identifier)
            if label == "model_prediction":
                counts["unlabelled_prediction_rows"] += 1
                continue
            observation_key = observation.get("key")
            canonical = verified.get(observation_key) if isinstance(observation_key, str) else None
            matched = (event == "pipeline_observation" and label == "verified_virtual_people_gt_zero"
                and canonical is not None and identifier == observation.get("inference_id")
                and all(observation.get(key) == canonical.get(key) for key in
                        ("row_id", "room", "partition", "observed_at", "sensors", "truth", "virtual_people")))
            if not matched:
                counts["unverified_label_rows"] += 1
                continue
            counts["verified_truth_rows"] += 1
            counts["verified_zero_truth_rows"] += int(canonical["truth"] == 0)
            if canonical["partition"] not in ("train", "train_only"):
                counts["heldout_excluded_rows"] += 1
            elif not unusable and not duplicate and identifier and canonical["key"] not in selected:
                selected.add(canonical["key"])
    return {"origin": origin, "raw_modified": False, "files": len(inputs), "input_hashes": inputs,
        "counts": dict(counts), "events": dict(events), "prediction_sources": dict(sources), "label_sources": dict(labels),
        "missing_sensors": dict(missing), "first_log_timestamp_utc": min(timestamps).isoformat() if timestamps else None,
        "last_log_timestamp_utc": max(timestamps).isoformat() if timestamps else None,
        "training_candidate_rows": len(selected),
        "policy": "external canonical ID/time/input/independent truth match; train-only; no pseudo labels",
        "privacy": "aggregate counts and file digests only; raw fields and raw paths omitted"}


if __name__ == "__main__":
    # CLI 단독 진단에서는 외부 정답 검증 맵이 없으므로 학습 후보를 자동 승인하지 않는다.
    result = analyze(os.environ["LOG_AUDIT_ROOT"], os.environ["LOG_AUDIT_ORIGIN"])
    output = Path(os.environ["LOG_AUDIT_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
