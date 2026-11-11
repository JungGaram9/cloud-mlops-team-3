"""실제 학습 구간으로 보정한 독립 센서 CSV 생성기. 분류 모델은 읽거나 학습하지 않는다."""

import csv
import hashlib
import json
import math
import os
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from scipy.optimize import nnls
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("date", "Temperature", "Humidity", "Light", "CO2", "Occupancy")
SENSORS = COLUMNS[1:5]
VERSION = "dynamic-sensors-v1"
EVENT_KINDS = ("weekend_empty_light", "night_empty_light", "forgotten_lights", "night_occupied",
               "occupied_dark", "ventilated_occupied", "co2_residual")
BOUNDS = np.array([[-20, 45], [1, 99], [0, 20000], [250, 10000]], dtype=float)


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Config:
    seed: int = 20261111
    rooms: int = 3
    days: int = 7
    start_date: str = "2026-11-09"
    timezone: str = "Asia/Seoul"
    step_seconds: int = 60
    mode: str = "exceptions"
    capacity: int = 8
    volume_m3: float = 90
    event_probability: float = 0.03
    latitude: float = 37.5

    def __post_init__(self):
        date.fromisoformat(self.start_date)
        ZoneInfo(self.timezone)
        if not (0 <= self.seed < 2**32 and 1 <= self.rooms <= 12 and 1 <= self.days <= 31
                and self.step_seconds in (30, 60, 300) and 1 <= self.capacity <= 100
                and math.isfinite(self.volume_m3) and 20 <= self.volume_m3 <= 2000
                and 0 <= self.event_probability <= 1 and -65 <= self.latitude <= 65
                and self.mode in ("normal", "exceptions")):
            raise ValueError("생성 설정의 범위를 확인하세요.")
        if self.mode == "exceptions" and self.days < 7:
            raise ValueError("예외 집중 모드는 평일·주말을 포함하도록 7일 이상 지정하세요.")

    @classmethod
    def environment(cls):
        defaults = asdict(cls())
        return cls(**{key: type(value)(os.environ.get(f"SIM_{key.upper()}", value)) for key, value in defaults.items()})


def read_training(root=ROOT):
    """canonical training만 읽고 평가 CSV를 열지 않는다. 중복 파생본도 합치지 않는다."""
    path, sidecar = root / "data/training_dataset.csv", root / "data/dataset_sources.json"
    info = json.loads(sidecar.read_text(encoding="utf-8"))[path.name]
    if info["sha256"] != digest(path):
        raise ValueError("학습 CSV와 출처 해시가 다릅니다.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("학습 CSV의 6개 열을 확인하세요.")
        raw = list(reader)
    if len(raw) != len(info["sources"]):
        raise ValueError("학습 행과 출처 개수가 다릅니다.")
    records, seen = defaultdict(list), set()
    for row, source in zip(raw, info["sources"]):
        if source not in ("legacy", "Room_A", "Room_B"):
            raise ValueError(f"알 수 없는 학습 출처: {source}")
        stamp = datetime.fromisoformat(row["date"])
        axis = stamp.astimezone(timezone.utc).replace(tzinfo=None) if stamp.tzinfo else stamp
        local = stamp if source == "legacy" else axis.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Vienna"))
        values = np.array([float(row[key]) for key in SENSORS])
        label = float(row["Occupancy"])
        if not np.isfinite(values).all() or not (-50 <= values[0] <= 60 and 0 <= values[1] <= 100
                and 0 <= values[2] <= 1e6 and 0 < values[3] <= 1e6) or label not in (0, 1):
            raise ValueError("학습 자료의 센서 또는 정답이 유효하지 않습니다.")
        identity = (source, axis)
        if identity in seen:
            raise ValueError("같은 출처의 학습 시각이 중복됩니다. 보정 가중을 중복 적용하지 않습니다.")
        seen.add(identity)
        records[source].append({"axis": axis, "local": local, "x": values, "y": int(label), "count": None, "open": None})
    provenance = {"canonical_file": "data/training_dataset.csv", "canonical_sha256": digest(path),
                  "source_metadata_sha256": digest(sidecar), "canonical_rows": len(raw), "role": "training_only",
                  "excluded": ["test_dataset.csv", "graz_*.csv", "raw copies", "feature occupancy sensor label"],
                  "timezones": {"legacy": "unknown local wall clock, no assumed UTC/Vienna/KST conversion",
                                "Room_A": "canonical UTC -> Europe/Vienna", "Room_B": "canonical UTC -> Europe/Vienna"},
                  "auxiliary": {}}
    for source, rows in records.items():
        rows.sort(key=lambda row: row["axis"])
        if source == "legacy":
            continue
        directory = root / "data/occupancy-detection-dataset/data/intermi" / source
        selected = {row["axis"]: row for row in rows}
        paths = [directory / "df_occ.csv", directory / "df_features.csv"]
        if not all(path.is_file() for path in paths):
            provenance["auxiliary"][source] = {"available": False, "reason": "보조 CSV 없음, 인원수·개구부는 가정 사용"}
            continue
        for path in paths:
            matched_times = set()
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for extra in csv.DictReader(stream):
                    stamp = datetime.fromisoformat(extra["Date"])
                    key = stamp.astimezone(timezone.utc).replace(tzinfo=None) if stamp.tzinfo else stamp
                    if key not in selected:
                        continue
                    if key in matched_times:
                        raise ValueError(f"{source}: 선택한 보조 자료의 시각이 중복됩니다.")
                    matched_times.add(key)
                    row = selected[key]
                    if path.name == "df_occ.csv":
                        count = float(extra["Image-based-occupancy"])
                        if not math.isfinite(count) or count < 0 or not count.is_integer() or int(count > 0) != row["y"]:
                            raise ValueError(f"{source}: 학습 행과 영상 정답이 다릅니다.")
                        row["count"] = int(count)
                    else:
                        sensors = np.array([float(extra[name]) for name in ("temperature", "humidity", "light", "co2")])
                        if not np.allclose(sensors, row["x"], atol=1e-5, rtol=0):
                            raise ValueError(f"{source}: 보조 센서와 canonical 행이 다릅니다.")
                        opening = [float(extra[name]) for name in ("openDoors", "openWindows")]
                        if all(math.isfinite(value) and 0 <= value <= 1 for value in opening):
                            row["open"] = max(opening)
        if any(row["count"] is None for row in rows):
            raise ValueError(f"{source}: 학습 행에 연결할 영상 정답이 일부 누락됐습니다.")
        provenance["auxiliary"][source] = {"available": True, "matched_image_rows": sum(r["count"] is not None for r in rows),
            "matched_opening_rows": sum(r["open"] is not None for r in rows),
            "files": {str(path.relative_to(root)): digest(path) for path in paths},
            "selection": "canonical training source + timestamp + sensor/binary-label agreement"}
    return dict(records), provenance


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    return {"rows": len(x), "mean": np.mean(x, axis=0).tolist(), "std": np.std(x, axis=0).tolist(),
            "q01": np.quantile(x, .01, axis=0).tolist(), "q50": np.quantile(x, .5, axis=0).tolist(),
            "q99": np.quantile(x, .99, axis=0).tolist()}


def correlation(values):
    x = np.asarray(values, dtype=float)
    scale = np.std(x, axis=0)
    z = (x - x.mean(axis=0)) / np.maximum(scale, 1e-8)
    return (z.T @ z / max(len(z), 1)).tolist()


def transitions(rows):
    gaps = [(b["axis"] - a["axis"]).total_seconds() for a, b in zip(rows, rows[1:])]
    positive = [gap for gap in gaps if gap > 0]
    cadence = float(np.median(positive)) if positive else 60.
    pairs = [(a, b, gap / 60) for a, b, gap in zip(rows, rows[1:], gaps)
             if 0 < gap <= cadence * 1.6 and a["local"].date() == b["local"].date()]
    return cadence, pairs


def calibration(records, provenance):
    profiles = {}
    for source, rows in sorted(records.items()):
        values = np.array([r["x"] for r in rows])
        labels = np.array([r["y"] for r in rows])
        cadence, pairs = transitions(rows)
        if len(pairs) < 20 or not labels.any() or labels.all():
            raise ValueError(f"{source}: 연속 관측 또는 두 클래스가 부족합니다.")
        exposures = np.zeros((7, 24))
        occupied = np.zeros((7, 24))
        starts = np.zeros((7, 24))
        day_minutes, day_used, entries = Counter(), set(), Counter()
        for a, b, dt in pairs:
            weekday, hour, day = a["local"].weekday(), a["local"].hour, a["local"].date().isoformat()
            exposures[weekday, hour] += dt
            occupied[weekday, hour] += dt * a["y"]
            day_minutes[day] += dt
            if a["y"]:
                day_used.add(day)
            if not a["y"] and b["y"]:
                starts[b["local"].weekday(), b["local"].hour] += 1
                entries[day] += 1
        # 공백·날짜를 가로지른 run을 쓰지 않도록 별도의 연속 구간에서 지속 시간을 계산한다.
        complete_durations, run_start, previous = [], None, None
        for a, b, dt in pairs:
            if previous is not a:
                run_start = None
            if not a["y"] and b["y"]:
                run_start = b["axis"]
            if a["y"] and not b["y"] and run_start is not None:
                complete_durations.append((b["axis"] - run_start).total_seconds() / 60)
                run_start = None
            previous = b
        usage = []
        for weekday in range(7):
            days = [day for day, minutes in day_minutes.items() if date.fromisoformat(day).weekday() == weekday and minutes >= 360]
            broader = [day for day, minutes in day_minutes.items() if (date.fromisoformat(day).weekday() >= 5) == (weekday >= 5) and minutes >= 360]
            prior = sum(day in day_used for day in broader) / max(len(broader), 1)
            usage.append(float(np.clip((sum(day in day_used for day in days) + .25 * prior) / (len(days) + .25), .01, .99)))
        hour_prior = occupied.sum(axis=0) / np.maximum(exposures.sum(axis=0), 1)
        hourly = (occupied + 60 * hour_prior) / (exposures + 60)
        weights = starts / np.maximum(exposures / 60, 1) + .2 * hourly + .001
        occupied_values, vacant_values = values[labels == 1], values[labels == 0]
        center = np.median(vacant_values, axis=0)
        effect = np.median(occupied_values, axis=0) - center
        daylight = np.array([max(0., math.sin(math.pi * (r["local"].hour + r["local"].minute / 60 - 6) / 12)) for r in rows])
        light_fit, _ = nnls(np.column_stack([np.ones(len(rows)), daylight, labels]), values[:, 2])
        floor = float(np.quantile(values[:, 3], .01))
        counts = [r["count"] for r in rows if r["count"] is not None and r["count"] > 0]
        mean_people = float(np.mean(counts)) if counts else 2.
        co2_x, co2_y, rates, innovations = [], [], [], []
        for a, b, dt in pairs:
            count = a["count"] if a["count"] is not None else a["y"]
            excess = max(0., a["x"][3] - floor)
            co2_x.append([count, -excess, -excess * (a["open"] or 0)])
            co2_y.append((b["x"][3] - a["x"][3]) / dt)
            rates.append((b["x"] - a["x"]) / dt)
            innovations.append((b["x"] - a["x"]) / math.sqrt(dt))
        co2_y = np.array(co2_y)
        raw_co2, _ = nnls(np.array(co2_x), np.clip(co2_y, *np.quantile(co2_y, [.01, .99])))
        emission = raw_co2[0] / (1 if counts else mean_people)
        coefficients = [float(np.clip(emission, .3, 80)), float(np.clip(raw_co2[1], .002, .2)), float(np.clip(raw_co2[2], 0, .2))]
        innovations = np.array(innovations)
        clipped = np.clip(innovations, np.quantile(innovations, .01, axis=0), np.quantile(innovations, .99, axis=0))
        covariance = np.cov(clipped, rowvar=False) * .25 + np.diag([.0001, .0001, .01, .01])
        acf = []
        for column in range(4):
            a = np.array([p[0]["x"][column] for p in pairs])
            b = np.array([p[1]["x"][column] for p in pairs])
            acf.append(float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0.)
        tau = np.clip(-(cadence / 60) / np.log(np.clip(acf, .01, .9999)), 5, 180)
        durations = np.array(complete_durations or [60.])
        profiles[source] = {"rows": len(rows), "first_axis": rows[0]["axis"].isoformat(), "last_axis": rows[-1]["axis"].isoformat(),
            "first_local": rows[0]["local"].isoformat(), "last_local": rows[-1]["local"].isoformat(),
            "cadence_seconds": cadence, "valid_transition_pairs": len(pairs), "summary": describe(values),
            "occupancy_fraction": float(labels.mean()), "bright_vacant_rows": int(np.sum((labels == 0) & (values[:, 2] >= 300))),
            "dark_occupied_rows": int(np.sum((labels == 1) & (values[:, 2] < 300))),
            "occupied_rows": int(labels.sum()), "correlation": correlation(values), "lag_cadence_acf": acf,
            "minute_change": describe(rates), "day_usage_probability": usage, "observed_day_minutes": dict(day_minutes),
            "hourly_occupancy": hourly.tolist(), "start_weights": weights.tolist(),
            "sessions_per_used_day": float(np.mean([entries[day] for day in day_used])) if day_used else 1.,
            "complete_occupied_durations_minutes": describe(durations),
            "complete_duration_count": len(complete_durations), "duration_fit_used_fallback": not bool(complete_durations),
            "duration_log_mean": float(np.log(np.maximum(durations, 1)).mean()),
            "duration_log_std": float(np.clip(np.log(np.maximum(durations, 1)).std(), .25, 1.)),
            "center": center.tolist(), "occupancy_association": effect.tolist(), "light_regression": light_fit.tolist(),
            "co2_floor_proxy": floor, "co2_nnls_raw": raw_co2.tolist(), "co2_coefficients": coefficients,
            "co2_count_basis": "image people" if counts else "binary occupied divided by assumed 2 people",
            "people_distribution": dict(sorted(Counter(counts).items())), "people_mean": mean_people,
            "opening_mean": {str(label): float(np.mean([r["open"] for r in rows if r["y"] == label and r["open"] is not None]))
                             if any(r["y"] == label and r["open"] is not None for r in rows) else (.25 if label else .08)
                             for label in (0, 1)},
            "max_observed_people": max(counts) if counts else None, "relaxation_minutes": tau.tolist(),
            "noise_covariance": covariance.tolist(), "noise_tau_minutes": float(np.clip(np.median(tau) / 6, 2, 30))}
    result = {"version": VERSION, "provenance": provenance, "sources": profiles,
        "assumptions": {"training_scope": "canonical training only; auxiliary selected by training membership",
            "day_probability_floor": .01, "day_probability_ceiling": .99, "weekday_prior_weight": .25,
            "unobserved_duration_fallback_minutes": 60, "duration_limits_minutes": [10, 360],
            "reference_volume_m3": 90, "volume_and_capacity": "not identifiable from sensors; configurable transfer assumptions",
            "light_separation": "nonnegative daylight/occupancy association, not observed lamp truth",
            "co2_floor": "training q01 proxy, not a measured outdoor concentration",
            "co2_bounds": {"emission_ppm_person_min": [.3, 80], "decay_per_min": [.002, .2]},
            "temperature_humidity": "occupancy associations are not causal effects; outdoor/HVAC can reverse changes",
            "noise": "bounded per-source change covariance + correlated OU bias, not independent uniform noise",
            "forced_ventilation_extra_ach": [8, 14], "lamp_on_probability_assumed": .92,
            "forgotten_lamp_probability_assumed": .06, "forced_lamp_lux_assumed": [360, 850],
            "forced_exception_frequency": "coverage configuration, not an empirical occurrence estimate",
            "daylight": "solar altitude at configured latitude; local civil noon approximation, stochastic cloud and room orientation",
            "co2_sensor_bounds_ppm": [250, 10000], "sensor_bounds": BOUNDS.tolist(),
            "challenge_envelopes": {"occupied_dark_max_lux": 300, "ventilated_occupied_max_ppm": 650,
                                    "minimum_residual_excess_ppm": 20, "settling_minutes": 30},
            "office_transfer": "two offices + legacy; large classrooms, actual weather and ventilation need external validation"}}
    result["profile_hash"] = hashlib.sha256(serialized(result).encode()).hexdigest()
    return result


def timeline(config):
    zone = ZoneInfo(config.timezone)
    day = date.fromisoformat(config.start_date)
    first = datetime.combine(day, time(), zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=config.days), time(), zone).astimezone(timezone.utc)
    return [first + timedelta(seconds=i * config.step_seconds) for i in range(int((end - first).total_seconds() // config.step_seconds))]


def solar(local, latitude):
    declination = math.radians(23.44 * math.sin(2 * math.pi * (local.timetuple().tm_yday - 81) / 365.25))
    hour_angle = math.radians(15 * (local.hour + local.minute / 60 - 12))
    lat = math.radians(latitude)
    return max(0., math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(declination) * math.cos(hour_angle))


def person_count(rng, source, config):
    observed = source["people_distribution"]
    if observed:
        counts = np.array([int(key) for key in observed])
        weights = np.array(list(observed.values()), dtype=float)
        return min(config.capacity, int(rng.choice(counts, p=weights / weights.sum())))
    return int(np.clip(1 + rng.poisson(1), 1, config.capacity))


def events_for_room(config, source, room_index, rng):
    zone = ZoneInfo(config.timezone)
    days = [date.fromisoformat(config.start_date) + timedelta(days=i) for i in range(config.days)]
    events = []

    def add(kind, day, minute, duration, people=0):
        grid = config.step_seconds / 60
        minute = math.floor(minute / grid) * grid
        start = datetime.combine(day, time(), zone).astimezone(timezone.utc) + timedelta(minutes=minute)
        stop = start + timedelta(minutes=max(grid, round(duration / grid) * grid))
        event = {"id": f"r{room_index + 1:02}-e{len(events):04}", "kind": kind,
                 "start": start.isoformat(), "end": stop.isoformat(), "people": people,
                 "lamp_lux": float(rng.uniform(360, 850)), "opening": float(rng.uniform(.7, 1.)),
                 "extra_ventilation_ach": float(rng.uniform(8, 14)) if kind == "ventilated_occupied" else 0.}
        if kind in ("forgotten_lights", "co2_residual"):
            event["departure"] = (start + timedelta(minutes=round(rng.uniform(60, 100) / grid) * grid)).isoformat()
        events.append(event)

    for day in days:
        weekday = day.weekday()
        if rng.random() < source["day_usage_probability"][weekday]:
            sessions = min(8, max(1, int(rng.poisson(max(.3, source["sessions_per_used_day"])))))
            weights = np.array(source["start_weights"][weekday], dtype=float)
            for _ in range(sessions):
                hour = int(rng.choice(24, p=weights / weights.sum()))
                duration = float(np.clip(rng.lognormal(source["duration_log_mean"], source["duration_log_std"]), 10, 360))
                add("ordinary_use", day, hour * 60 + rng.uniform(0, 60), duration, person_count(rng, source, config))
        if config.mode == "normal" and rng.random() < config.event_probability:
            kinds = [kind for kind in EVENT_KINDS if kind != "weekend_empty_light" or weekday >= 5]
            kind = str(rng.choice(kinds))
            hour = 22 if "night" in kind else int(rng.integers(9, 17))
            add(kind, day, hour * 60, rng.uniform(150, 220), min(config.capacity, max(3, person_count(rng, source, config))))
    if config.mode == "exceptions":
        # 강제 사건 비중은 관측 발생률이 아니다. 각 사건을 주/방별로 분산한다.
        for index, kind in enumerate(EVENT_KINDS):
            if index % config.rooms != room_index:
                continue
            weekday = (5, 1, 2, 3, 4, 0, 6)[index]
            day = next(day for day in days if day.weekday() == weekday)
            minute = 22 * 60 if kind == "night_empty_light" else 21 * 60 + rng.uniform(0, 60) if kind == "night_occupied" else rng.uniform(9 * 60, 16 * 60)
            add(kind, day, minute, rng.uniform(160, 230), min(config.capacity, max(3, person_count(rng, source, config))))
    return sorted(events, key=lambda event: (event["start"], event["id"]))


def sensor_step(previous, target, people, lamp_lux, opening, noise, source, volume, dt, extra_ach=0.):
    """CO₂ 질량수지는 전등 상태와 무관하다. 다른 센서는 환경 목표로 완만하게 이동한다."""
    emission, decay, open_decay = source["co2_coefficients"]
    k = decay + opening * max(open_decay, .015) + extra_ach / 60
    equilibrium = target[3] + emission * people * 90 / volume / k
    result = previous + (target - previous) * (1 - np.exp(-dt / np.array(source["relaxation_minutes"])))
    result[3] = equilibrium + (previous[3] - equilibrium) * math.exp(-k * dt)
    result[2] = previous[2] + (target[2] + lamp_lux - previous[2]) * (1 - math.exp(-dt / .7))
    return np.clip(result + noise, BOUNDS[:, 0], BOUNDS[:, 1])


def generate_room(config, profile, room_index, stamps):
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, room_index]))
    source_name = sorted(profile["sources"])[room_index % len(profile["sources"])]
    source = profile["sources"][source_name]
    room_id = f"room_{room_index + 1:02}"
    volume = float(config.volume_m3 * rng.uniform(.75, 1.3))
    orientation = float(rng.uniform(.45, 1.6))
    gain = float(max(20, source["light_regression"][2]) * rng.uniform(.7, 1.4))
    center = np.array(source["center"])
    effect = np.array(source["occupancy_association"])
    covariance = np.array(source["noise_covariance"])
    chol = np.linalg.cholesky(covariance)
    noise_state = np.zeros(4)
    previous = center.copy()
    previous[3] = source["co2_floor_proxy"]
    events = events_for_room(config, source, room_index, rng)
    parsed = [(event, datetime.fromisoformat(event["start"]), datetime.fromisoformat(event["end"])) for event in events]
    lamp, lamp_off_at, prior_people = False, None, 0
    vent_state = source["opening_mean"]["0"]
    cloud, outside_shift, hvac, sensor_bias = .65, 0., center[0], rng.normal(0, [.15, .5, 2, 5])
    rows, metadata = [], []
    zone, dt = ZoneInfo(config.timezone), config.step_seconds / 60
    current_day = None
    rho = math.exp(-dt / source["noise_tau_minutes"])
    for index, stamp in enumerate(stamps):
        local = stamp.astimezone(zone)
        if local.date() != current_day:
            current_day = local.date()
            outside_shift = float(rng.normal(-3, 3))
            hvac = float(center[0] + rng.normal(0, .8))
            humidity_outside = float(center[1] + rng.normal(0, 10))
        cloud = float(np.clip(cloud + .05 * dt * (.6 - cloud) + rng.normal(0, .04 * math.sqrt(dt)), .12, 1.))
        sun = solar(local, config.latitude) * orientation * cloud
        active = [(event, start, end) for event, start, end in parsed if start <= stamp < end]
        prep = [event for event, start, _ in parsed if event["kind"] == "night_empty_light" and start - timedelta(minutes=10) <= stamp < start]
        ordinary = [event for event, _, _ in active if event["kind"] == "ordinary_use"]
        special = [event for event, _, _ in active if event["kind"] != "ordinary_use"]
        people = max((event["people"] for event in ordinary), default=0)
        vent_state = float(np.clip(vent_state + (source["opening_mean"][str(int(people > 0))] - vent_state)
                                  * (1 - math.exp(-dt / 20)) + rng.normal(0, .02 * math.sqrt(dt)), 0, 1))
        opening = vent_state
        extra_ach = 0.
        if special:
            event = special[-1]
            kind = event["kind"]
            if kind in ("weekend_empty_light", "night_empty_light"):
                people = 0
            elif kind in ("forgotten_lights", "co2_residual"):
                people = event["people"] if stamp < datetime.fromisoformat(event["departure"]) else 0
                opening = 0.
            else:
                people = event["people"]
            if kind == "ventilated_occupied":
                opening = event["opening"]
                extra_ach = event["extra_ventilation_ach"]
        if prep:
            people = 0
        if people and not prior_people:
            lamp = bool(rng.random() < .92)
            lamp_on_at = stamp + timedelta(minutes=float(rng.uniform(0, 6)))
            lamp_off_at = None
        if not people and prior_people:
            lamp_off_at = stamp + timedelta(minutes=float(rng.uniform(1, 10) if rng.random() > .06 else rng.uniform(60, 180)))
        if lamp_off_at is not None and stamp >= lamp_off_at:
            lamp = False
        lamp_lux = gain if lamp and (not people or stamp >= lamp_on_at) else 0.
        if prep:
            lamp_lux = 0.
        for event in special:
            if event["kind"] in ("weekend_empty_light", "night_empty_light", "forgotten_lights"):
                lamp_lux = event["lamp_lux"]
            elif event["kind"] in ("occupied_dark", "co2_residual"):
                lamp_lux = 0.
            elif event["kind"] == "night_occupied":
                lamp_lux = max(gain, event["lamp_lux"] * .7)
        daylight = source["light_regression"][1] * sun
        if any(event["kind"] == "occupied_dark" for event in special):
            daylight *= .02
        relative_people = people / max(source["people_mean"], 1)
        target = center.copy()
        target[0] = .65 * hvac + .35 * (center[0] + outside_shift + 3 * sun) + float(np.clip(effect[0], -2, 2)) * relative_people * .3
        target[1] = center[1] + .3 * (humidity_outside - center[1]) - .7 * (previous[0] - center[0]) + float(np.clip(effect[1], -5, 5)) * relative_people * .3
        target[2] = source["light_regression"][0] + daylight
        target[3] = source["co2_floor_proxy"] + 8 * cloud
        next_noise = rho * noise_state + math.sqrt(1 - rho**2) * (chol @ rng.normal(size=4))
        # 센서 바이어스 변화만 더해 랜덤 워크의 무한 누적을 막는다.
        previous = sensor_step(previous, target + sensor_bias, people, lamp_lux, opening, next_noise - noise_state,
                               source, volume, dt, extra_ach)
        noise_state = next_noise
        truth = int(people > 0)
        rows.append((local.isoformat(), *(round(float(value), 6) for value in previous), truth))
        row_id = hashlib.sha256(f"{config.seed}:{room_id}:{stamp.isoformat()}:{profile['profile_hash']}".encode()).hexdigest()[:24]
        active_ids = [event["id"] for event, _, _ in active]
        metadata.append({"row_index": index, "csv_line": index + 2, "row_id": row_id, "room_id": room_id,
            "csv_file": f"{room_id}.csv", "schema_version": 1,
            "local_date": local.date().isoformat(), "synthetic": True, "virtual_people": people,
            "lamp_on": lamp_lux > 0, "lamp_target_lux": round(lamp_lux, 6), "opening_fraction": round(opening, 6),
            "ventilation_ach": round(60 * (source["co2_coefficients"][1] + opening * max(source["co2_coefficients"][2], .015)) + extra_ach, 6),
            "event_ids": active_ids, "events": [event["kind"] for event, _, _ in active],
            "prelude_event_ids": [event["id"] for event in prep], "split_group": room_id,
            "room_day_group": f"{room_id}:{local.date()}", "scenario": config.mode})
        prior_people = people
    traits = {"calibration_source": source_name, "volume_m3_assumed": volume, "orientation_assumed": orientation,
              "lamp_gain_lux": gain, "sensor_bias": sensor_bias.tolist(), "capacity_assumed": config.capacity}
    return rows, metadata, events, traits


def sensor_windows(values, stamps, days, cadence, stride=1, sample_every=1, size=8):
    windows = []
    for start in range(0, len(values) - (size - 1) * stride, sample_every):
        indices = start + np.arange(size) * stride
        if days[indices[0]] != days[indices[-1]]:
            continue
        gaps = [(stamps[b] - stamps[a]).total_seconds() for a, b in zip(indices, indices[1:])]
        if all(.5 * cadence <= gap <= 1.6 * cadence for gap in gaps):
            windows.append(values[indices].reshape(-1))
    return np.array(windows).reshape(-1, size * 4)


def quality(rows_by_room, metadata_by_room, events_by_room, records, profile, config):
    errors, reports, kind_counts = [], {}, Counter()
    raw = np.array([row["x"] for source in records.values() for row in source])
    scale = np.maximum(np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0), [1., 1., 10., 10.])
    tree = cKDTree(raw / scale)
    exact = {tuple(np.round(row, 6)) for row in raw}
    actual_windows, window_trees = set(), {}
    for source_name, source_rows in records.items():
        cadence, _ = transitions(source_rows)
        windows = sensor_windows(np.array([r["x"] for r in source_rows]), [r["axis"] for r in source_rows],
                                 [r["local"].date() for r in source_rows], cadence)
        actual_windows.update(hashlib.sha256(np.round(window, 3).tobytes()).hexdigest() for window in windows)
        if len(windows):
            window_trees[source_name] = cKDTree(windows / np.tile(scale, 8))
    for room, rows in rows_by_room.items():
        meta, events = metadata_by_room[room], events_by_room[room]
        x = np.array([row[1:5] for row in rows])
        y = np.array([row[5] for row in rows])
        stamps = [datetime.fromisoformat(row[0]) for row in rows]
        gaps = np.array([(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])])
        # 파싱한 고정 offset datetime을 UTC에서 비교해 DST 반복 시각도 구분한다.
        if not np.isfinite(x).all() or not np.all((x >= BOUNDS[:, 0]) & (x <= BOUNDS[:, 1])):
            errors.append(f"{room}: 센서 범위/유한값 위반")
        if not np.all(gaps == config.step_seconds) or len(rows) != len(meta):
            errors.append(f"{room}: 시간 순서/메타데이터 개수 위반")
        if not np.array_equal(y, [int(item["virtual_people"] > 0) for item in meta]):
            errors.append(f"{room}: 가상 정답 불일치")
        if any(item["row_index"] != index or item["csv_line"] != index + 2 or not item["synthetic"]
               or item["csv_file"] != f"{room}.csv" or not 0 <= item["virtual_people"] <= config.capacity for index, item in enumerate(meta)):
            errors.append(f"{room}: 행 연결/합성 표시 위반")
        if len({item["row_id"] for item in meta}) != len(meta):
            errors.append(f"{room}: 행 식별값 중복")
        distances, _ = tree.query(x / scale)
        exact_rows = sum(tuple(np.round(row, 6)) in exact for row in x)
        copied_windows, near_windows = 0, {}
        local_days = [stamp.date() for stamp in stamps]
        for source_name, training_tree in window_trees.items():
            cadence = profile["sources"][source_name]["cadence_seconds"]
            stride = max(1, round(cadence / config.step_seconds))
            if not math.isclose(stride * config.step_seconds, cadence, abs_tol=1):
                continue
            windows = sensor_windows(x, stamps, local_days, cadence, stride=stride,
                                     sample_every=max(1, round(900 / config.step_seconds)))
            if not len(windows):
                continue
            distances_window, _ = training_tree.query(windows / np.tile(scale, 8))
            distances_window /= math.sqrt(32)
            near_fraction = float(np.mean(distances_window < .02))
            near_windows[source_name] = {"sampled_windows": len(windows), "window_observations": 8,
                "window_span_minutes": cadence * 7 / 60, "q01_distance_iqr_rms": float(np.quantile(distances_window, .01)),
                "median_distance_iqr_rms": float(np.median(distances_window)), "fraction_below_0_02": near_fraction}
            copied_windows += sum(hashlib.sha256(np.round(window, 3).tobytes()).hexdigest() in actual_windows for window in windows)
            if near_fraction > .8:
                errors.append(f"{room}: {source_name} 센서 윈도우의 근접 복사가 지배적입니다.")
        if exact_rows or copied_windows:
            errors.append(f"{room}: 원본 센서 행/시계열 복제 감지")
        reference = profile["sources"][profile["room_traits"][room]["calibration_source"]]
        lag = max(1, round(reference["cadence_seconds"] / config.step_seconds))
        mask = np.array([a == b for a, b in zip(local_days[:-lag], local_days[lag:])])
        before, after = x[:-lag][mask], x[lag:][mask]
        delta = (after - before) / (lag * config.step_seconds / 60)
        acf = [float(np.corrcoef(before[:, column], after[:, column])[0, 1])
               if np.std(before[:, column]) and np.std(after[:, column]) else 0. for column in range(4)]
        event_report = []
        for event in events:
            if event["kind"] == "ordinary_use":
                continue
            indices = [index for index, item in enumerate(meta) if event["id"] in item["event_ids"]]
            if not indices:
                errors.append(f"{room}: 사건 {event['kind']}가 출력에 없음")
                continue
            kind_counts[event["kind"]] += 1
            people = [meta[index]["virtual_people"] for index in indices]
            lamps = [meta[index]["lamp_on"] for index in indices]
            if event["kind"] in ("weekend_empty_light", "night_empty_light") and (any(people) or not all(lamps)):
                errors.append(f"{room}: 무인 점등 정답/전등 상태 위반")
            if event["kind"] == "weekend_empty_light" and any(stamps[index].weekday() < 5 for index in indices):
                errors.append(f"{room}: 주말 사건 날짜 위반")
            if event["kind"] in ("night_occupied", "occupied_dark", "ventilated_occupied") and not all(person > 0 for person in people):
                errors.append(f"{room}: 실제 사용 사건의 인원수 위반")
            first = indices[0]
            if event["kind"] == "night_empty_light" and first and (y[first - 1] != 0 or meta[first - 1]["lamp_on"]):
                errors.append(f"{room}: 야간 무인 점등 전 빈 방·소등 준비 상태 위반")
            departure = datetime.fromisoformat(event["departure"]) if "departure" in event else None
            after_exit = [index for index in indices if departure and stamps[index] >= departure]
            if event["kind"] in ("forgotten_lights", "co2_residual"):
                if not after_exit or any(meta[index]["virtual_people"] for index in after_exit):
                    errors.append(f"{room}: 퇴실 이후 가상 정답 위반")
                if event["kind"] == "co2_residual" and after_exit and x[after_exit[0], 3] <= reference["co2_floor_proxy"] + 20:
                    errors.append(f"{room}: 퇴실 직후 CO2 잔류가 충분히 생성되지 않았습니다.")
                if event["kind"] == "forgotten_lights" and not all(meta[index]["lamp_on"] for index in after_exit):
                    errors.append(f"{room}: 퇴실 후 소등 누락 사건의 전등 상태 위반")
            settled = [index for index in indices if (stamps[index] - stamps[first]).total_seconds() >= 1800]
            settled = settled or indices
            if event["kind"] == "occupied_dark" and np.median(x[settled, 2]) >= 300:
                errors.append(f"{room}: 사람이 있지만 어두운 사건의 밝기 확인 필요")
            if event["kind"] == "ventilated_occupied" and np.median(x[settled, 3]) > 650:
                errors.append(f"{room}: 환기 재실 사건의 낮은 CO2가 생성되지 않았습니다.")
            event_report.append({"id": event["id"], "kind": event["kind"], "rows": len(indices),
                "people_min_max": [min(people), max(people)], "sensor_summary": describe(x[indices]),
                "co2_start_end": [float(x[indices[0], 3]), float(x[indices[-1], 3])],
                "settled_median_light_co2": np.median(x[settled][:, [2, 3]], axis=0).tolist(),
                "light_onset_delta_lux": float(x[first, 2] - x[first - 1, 2]) if first else None,
                "co2_after_exit_first_last": [float(x[after_exit[0], 3]), float(x[after_exit[-1], 3])] if after_exit else None,
                "right_censored": datetime.fromisoformat(event["end"]) > stamps[-1] + timedelta(seconds=config.step_seconds)})
        complete_runs, start = [], None
        for index in range(1, len(rows)):
            if local_days[index] != local_days[index - 1]:
                start = None
                continue
            if not y[index - 1] and y[index]:
                start = index
            if y[index - 1] and not y[index] and start is not None:
                complete_runs.append((index - start) * config.step_seconds / 60)
                start = None
        reports[room] = {"rows": len(rows), "occupancy_fraction": float(y.mean()), "synthetic": describe(x),
            "reference_training": reference["summary"], "mean_difference": (x.mean(axis=0) - reference["summary"]["mean"]).tolist(),
            "minute_change": describe(delta), "reference_minute_change": reference["minute_change"],
            "comparison_interval_seconds": lag * config.step_seconds,
            "reference_interval_seconds": reference["cadence_seconds"],
            "rate_q99_difference": (np.quantile(delta, .99, axis=0) - reference["minute_change"]["q99"]).tolist(),
            "lag_cadence_acf": acf, "reference_lag_cadence_acf": reference["lag_cadence_acf"],
            "acf_difference": (np.array(acf) - reference["lag_cadence_acf"]).tolist(),
            "correlation": correlation(x), "reference_correlation": reference["correlation"],
            "correlation_difference": (np.array(correlation(x)) - reference["correlation"]).tolist(),
            "complete_occupied_durations_minutes": describe(complete_runs),
            "reference_complete_occupied_durations_minutes": reference["complete_occupied_durations_minutes"],
            "occupied_weekend_rows": int(sum(label and stamp.weekday() >= 5 for label, stamp in zip(y, stamps))),
            "occupied_night_rows": int(sum(label and (stamp.hour >= 21 or stamp.hour < 6) for label, stamp in zip(y, stamps))),
            "outside_training_q01_q99_fraction": np.mean((x < reference["summary"]["q01"]) | (x > reference["summary"]["q99"]), axis=0).tolist(),
            "exact_sensor_rows": exact_rows, "copied_eight_sensor_windows": copied_windows,
            "near_sensor_windows": near_windows,
            "nearest_training_distance_iqr": {"q01": float(np.quantile(distances, .01)), "median": float(np.median(distances)),
                "near_fraction_below_0_02": float(np.mean(distances < .02)), "definition": "four-sensor Euclidean distance / training IQR"},
            "events": event_report}
    if config.mode == "exceptions" and any(not kind_counts[kind] for kind in EVENT_KINDS):
        errors.append("대표 예외 중 출력에 포함되지 않은 사건이 있습니다.")
    return {"passed": not errors, "errors": errors, "sensor_order": list(SENSORS), "rooms": reports,
            "actual_exception_counts": dict(kind_counts), "realism_guarantee": False,
            "copy_check": {"timestamp_ignored": True, "near_window_iqr_rms_threshold": .02,
                           "near_copy_dominance_reject_fraction": .8, "definition": "copy guard, not realism score"},
            "limitations": ["사무실 관측의 강의실 외삽이며 시설 현실성/분류 성능을 보장하지 않음",
                "예외 집중 모드의 강제 사건 비율은 실제 발생률이 아님", "근접 거리만으로 실제 재현 여부를 증명하지 않음"]}


def read_export(csv_path, meta_path):
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("내보낸 CSV의 센서 전용 6열 계약이 다릅니다.")
        rows = [(row["date"], *(float(row[key]) for key in SENSORS), int(row["Occupancy"])) for row in reader]
    metadata = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    return rows, metadata


def quality_markdown(report, profile, config):
    lines = ["# 합성 센서 생성 품질", "", f"검사 통과: {report['passed']} · 모드: {config.mode} · seed: {config.seed}", "",
             "모든 정답은 가상 인원수에서만 계산했습니다. 실제 정답이나 실제 시설의 분류 성능을 뜻하지 않습니다.", "",
             "| 방 | 보정 출처 | 행수 | 가상 사용 비율 | 평균 CO₂ (ppm) | 원본 센서 행 복제 |", "|---|---|---|---|---|---|"]
    for room, item in report["rooms"].items():
        lines.append(f"| {room} | {profile['room_traits'][room]['calibration_source']} | {item['rows']} | {item['occupancy_fraction']:.1%} | {item['synthetic']['mean'][3]:.1f} | {item['exact_sensor_rows']} |")
    lines += ["", "사건 발생 건수: " + ", ".join(f"{kind} {count}" for kind, count in sorted(report["actual_exception_counts"].items())), "",
              "quality.json에는 센서 4종의 분포·분당 변화·자기상관·센서 상관·사용 지속시간 차이와 원본 센서 8시점 윈도우의 근접 거리를 기록했습니다. 시간표시를 바꿔도 센서 자체의 복제 검사는 유지됩니다.", "",
              "## 보정과 가정", "",
              "canonical training만 보정했으며 기존 평가 CSV는 열지 않았습니다. Graz 보조정보도 해당 학습 출처·시각·센서/영상정답이 일치한 부분만 연결했습니다. 보조정보의 occupancy 센서는 영상 정답으로 사용하지 않았습니다.", "",
              "Graz는 UTC에서 Europe/Vienna로 변환했습니다. legacy는 시간대를 확인할 수 없어 원래 현지 시각 그대로 분석했습니다. 생성 시각은 설정한 시간대의 offset을 포함하며 일광절약시간은 UTC 경과로 처리합니다.", "",
              f"가정: 방 기준 부피 {config.volume_m3}m³, 최대 가상 인원 {config.capacity}명, 외기·날씨·냉난방·전등 지연 및 사건 빈도. 보정된 사무실의 실제 부피와 전등 상태는 식별되지 않습니다. 큰 강의실과 다른 계절의 현실성을 보장하지 않습니다. 예외 집중 모드의 강제 비율은 관측 발생률이 아닙니다.", "",
              "manifest.json의 전체 방 단위 분할 권고를 사용하면 날짜·사건·이웃 행이 분할을 넘지 않습니다. 방 3개 미만은 train_only로만 표시합니다. 방마다 보정 출처가 다를 수 있어 이 분할은 출처 차이도 포함합니다. 자동 학습·평가·원본 병합은 수행하지 않았습니다.", ""]
    return "\n".join(lines)


def generate(config, output_root=None, root=ROOT):
    records, provenance = read_training(root)
    profile = calibration(records, provenance)
    code_hash = digest(Path(__file__))
    identity = hashlib.sha256(serialized({"config": asdict(config), "profile": profile["profile_hash"], "code": code_hash}).encode()).hexdigest()[:12]
    generation_id = f"{VERSION}-{identity}"
    target = Path(output_root or root / "data/simulated") / generation_id
    target.mkdir(parents=True, exist_ok=False)
    (target / "INCOMPLETE").write_text("생성 중, manifest와 quality의 passed를 확인하세요.\n", encoding="utf-8")
    rows_by_room, metadata_by_room, events_by_room, traits = {}, {}, {}, {}
    stamps = timeline(config)
    manifest = {"generation_id": generation_id, "generator_version": VERSION, "generator_code_sha256": code_hash, "config": asdict(config),
                "profile_hash": profile["profile_hash"], "provenance": provenance, "files": {}, "events": {},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                "suggested_split": {}, "split_policy": "whole room; date/event groups stay within room; no random adjacent-row split",
                "synthetic": True, "automatic_training_merge": False}
    for room_index in range(config.rooms):
        room = f"room_{room_index + 1:02}"
        rows, metadata, events, room_traits = generate_room(config, profile, room_index, stamps)
        csv_path, meta_path = target / f"{room}.csv", target / f"{room}.metadata.jsonl"
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        meta_path.write_text("".join(serialized(item) + "\n" for item in metadata), encoding="utf-8")
        partition = ("train" if room_index % 3 == 0 else "validation" if room_index % 3 == 1 else "test") if config.rooms >= 3 else "train_only"
        manifest["suggested_split"][room] = partition
        manifest["files"][csv_path.name] = {"sha256": digest(csv_path), "rows": len(rows), "metadata": meta_path.name,
            "metadata_sha256": digest(meta_path), "room_id": room, "partition": partition}
        manifest["events"][room] = events
        # 품질 검사는 메모리 원본이 아니라 실제 기록한 CSV/메타데이터를 다시 읽어 수행한다.
        rows_by_room[room], metadata_by_room[room] = read_export(csv_path, meta_path)
        events_by_room[room], traits[room] = events, room_traits
    profile["room_traits"] = traits
    report = quality(rows_by_room, metadata_by_room, events_by_room, records, profile, config)
    manifest["actual_exception_counts"] = report["actual_exception_counts"]
    manifest["quality_passed"] = report["passed"]
    for name, content in (("profile.json", profile), ("quality.json", report)):
        (target / name).write_text(json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    (target / "quality.md").write_text(quality_markdown(report, profile, config), encoding="utf-8")
    manifest["artifacts"] = {name: digest(target / name) for name in ("profile.json", "quality.json", "quality.md")}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise ValueError(f"생성 검증 실패: {report['errors']}. 출력은 {target}에 보존됩니다.")
    if (digest(root / "data/training_dataset.csv") != provenance["canonical_sha256"]
            or digest(root / "data/dataset_sources.json") != provenance["source_metadata_sha256"]
            or any(digest(root / name) != expected
                   for auxiliary in provenance["auxiliary"].values()
                   for name, expected in auxiliary.get("files", {}).items())
            or digest(Path(__file__)) != code_hash):
        raise ValueError("생성 도중 보정 자료 또는 코드가 바뀌었습니다. 미완료 출력을 확인하세요.")
    (target / "INCOMPLETE").unlink()
    return target, report


if __name__ == "__main__":
    target, report = generate(Config.environment(), Path(os.environ.get("SIM_OUTPUT_ROOT", ROOT / "data/simulated")))
    print(serialized({"output": str(target), "passed": report["passed"], "rows": sum(r["rows"] for r in report["rooms"].values()),
                      "events": report["actual_exception_counts"]}))
