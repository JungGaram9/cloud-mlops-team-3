"""Graz LoRa 재실 데이터(interim)를 프로젝트 CSV 스키마로 변환한다.

원본 시각은 UTC이고 영상 정답은 Europe/Vienna 현지 시각 기준이므로, 현지 시각으로 바꿔 Hour·DayOfWeek가
실제 생활 시간과 일치하게 한다. 영상으로 센 인원이 1명 이상이면 사용 중(1)으로 판별한다.
Room A는 12월 이전을 학습, 12월을 미래 평가로 나누고, Room B는 학습 기간과 겹치지 않는 12월만 새 공간 평가로 쓴다.

실행: python -m models.prepare_data  (data 디렉터리에 쓰기 권한이 필요하다)
"""

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from models.model import ROOT

SOURCE_DIR = ROOT / "data/occupancy-detection-dataset/data/intermi"
OUTPUT_DIR = ROOT / "data"
TIMEZONE = ZoneInfo("Europe/Vienna")
HOLDOUT_START = datetime(2023, 12, 1)
COLUMNS = ("date", "Temperature", "Humidity", "Light", "CO2", "Occupancy")
OUTPUTS = {
    "graz_room_a_train.csv": ("Room_A", False),
    "graz_room_a_test.csv": ("Room_A", True),
    "graz_room_b_test.csv": ("Room_B", True),
}


def convert_room(directory: Path) -> list[dict]:
    with (directory / "df_features.csv").open(encoding="utf-8-sig", newline="") as source:
        features = list(csv.DictReader(source))
    with (directory / "df_occ.csv").open(encoding="utf-8-sig", newline="") as source:
        labels = list(csv.DictReader(source))
    if [row["Date"] for row in features] != [row["Date"] for row in labels]:
        raise ValueError(f"{directory.name}: 센서와 정답의 시각이 일치하지 않습니다.")
    rows = []
    for sensors, label in zip(features, labels):
        local = datetime.fromisoformat(sensors["Date"]).astimezone(TIMEZONE).replace(tzinfo=None)
        count = float(label["Image-based-occupancy"])
        if count < 0 or count != int(count):
            raise ValueError(f"{directory.name} {sensors['Date']}: 인원 수가 0 이상의 정수가 아닙니다.")
        rows.append({"date": local.isoformat(sep=" "), "Temperature": sensors["temperature"],
                     "Humidity": sensors["humidity"], "Light": sensors["light"], "CO2": sensors["co2"],
                     "Occupancy": int(count > 0)})
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, int]:
    rooms = {room: convert_room(source_dir / room) for room, _ in set(OUTPUTS.values())}
    counts = {}
    for filename, (room, holdout) in OUTPUTS.items():
        rows = [row for row in rooms[room]
                if (datetime.fromisoformat(row["date"]) >= HOLDOUT_START) == holdout]
        write_rows(output_dir / filename, rows)
        counts[filename] = len(rows)
    return counts


if __name__ == "__main__":
    for name, rows in prepare().items():
        print(f"{name}: {rows}행")
