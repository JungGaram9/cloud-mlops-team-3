"""추가 센서·이미지 정답을 기존 CSV에 반영한다. 모델 학습은 실행하지 않는다."""

import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("date", "Temperature", "Humidity", "Light", "CO2", "Occupancy")
SENSORS = dict(zip(COLUMNS[1:5], ("temperature", "humidity", "light", "co2")))


def merge_data(root=ROOT):
    """동일 시각의 두 방을 같은 분할에 배치하고 기존 행과 파일별 역할을 보존한다."""
    paths = [root / "data" / name for name in ("training_dataset.csv", "test_dataset.csv")]
    originals = [path.read_bytes() for path in paths]
    existing = [list(csv.DictReader(io.StringIO(content.decode("utf-8-sig")))) for content in originals]
    metadata_path = root / "data/dataset_sources.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    origins = []
    for path, content, rows in zip(paths, originals, existing):
        recorded = metadata.get(path.name)
        if recorded and (recorded["sha256"] != hashlib.sha256(content).hexdigest()
                         or len(recorded["sources"]) != len(rows)):
            raise ValueError(f"{path.name}: 출처 메타데이터와 CSV가 일치하지 않습니다.")
        origins.append(list(recorded["sources"]) if recorded else ["legacy"] * len(rows))

    def timestamp(value):
        date = datetime.fromisoformat(value)
        return date.astimezone(timezone.utc).replace(tzinfo=None) if date.tzinfo else date

    def key(row):
        return (timestamp(row["date"]), *(float(row[name]) for name in COLUMNS[1:]))

    records, skipped = [], 0
    for room in ("Room_A", "Room_B"):
        directory = root / "data/occupancy-detection-dataset/data/intermi" / room
        with (directory / "df_occ.csv").open(encoding="utf-8-sig", newline="") as source:
            labels = {}
            for row in csv.DictReader(source):
                date = timestamp(row["Date"])
                if date in labels:
                    raise ValueError(f"{room}: 이미지 정답 시각이 중복됩니다: {date}")
                labels[date] = row["Image-based-occupancy"]
        seen_dates = set()
        with (directory / "df_features.csv").open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                date = timestamp(row["Date"])
                if date in seen_dates:
                    raise ValueError(f"{room}: 센서 시각이 중복됩니다: {date}")
                seen_dates.add(date)
                try:
                    sensors = {name: float(row[column]) for name, column in SENSORS.items()}
                    count = float(labels[date])
                    if (not all(math.isfinite(value) for value in (*sensors.values(), count))
                            or not -50 <= sensors["Temperature"] <= 60
                            or not 0 <= sensors["Humidity"] <= 100
                            or not 0 <= sensors["Light"] <= 1e6
                            or not 0 < sensors["CO2"] <= 1e6
                            or count < 0 or not count.is_integer()):
                        raise ValueError("유효하지 않은 센서 또는 정답")
                except (KeyError, ValueError, TypeError):
                    skipped += 1
                    continue
                records.append((room, {"date": date.isoformat(sep=" "), **sensors, "Occupancy": int(count > 0)}))
    records.sort(key=lambda item: timestamp(item[1]["date"]))
    split = int(len(records) * 0.8)
    while 0 < split < len(records) and records[split - 1][1]["date"] == records[split][1]["date"]:
        split -= 1
    if not 0 < split < len(records):
        raise ValueError("추가 데이터를 시간순으로 분할할 수 없습니다.")

    merged, added = [], []
    # 전체 센서·정답과 발생 순번으로 연결하여 같은 시각의 두 방을 혼동하지 않는다.
    for old, source_names, extra in zip(existing, origins, (records[:split], records[split:])):
        remaining = defaultdict(deque)
        for index, row in enumerate(old):
            remaining[key(row)].append(index)
        appended = []
        for room, row in extra:
            identity = key(row)
            if remaining[identity]:
                index = remaining[identity].popleft()
                if source_names[index] not in ("legacy", room):
                    raise ValueError("기존 행의 출처가 추가 자료와 다릅니다.")
                source_names[index] = room
            else:
                appended.append(row)
                source_names.append(room)
        merged.append(old + appended)
        added.append(len(appended))
    if {timestamp(row["date"]) for row in merged[0]} & {timestamp(row["date"]) for row in merged[1]}:
        raise ValueError("학습·평가 시각이 겹칩니다. 기존 분할을 확인하세요.")
    if set(key(row) for row in merged[0]) & set(key(row) for row in merged[1]):
        raise ValueError("학습·평가에 동일 기록이 있습니다.")
    for path, content in zip(paths, originals):
        if path.read_bytes() != content:
            raise ValueError("CSV가 병합 도중 변경되었습니다.")
    for path, rows, count in zip(paths, merged, added):
        if not count:
            continue
        temporary = path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    metadata = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sources": names}
                for path, names in zip(paths, origins)}
    serialized = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
    if not metadata_path.is_file() or metadata_path.read_text(encoding="utf-8") != serialized:
        temporary = metadata_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"additional_rows": len(records), "skipped_rows": skipped,
            "additional_training_rows": split, "additional_test_rows": len(records) - split,
            "training_rows": len(merged[0]), "test_rows": len(merged[1]),
            "appended_rows": added, "test_starts_at_utc": records[split][1]["date"]}


if __name__ == "__main__":
    print(json.dumps(merge_data(), ensure_ascii=False, indent=2))
