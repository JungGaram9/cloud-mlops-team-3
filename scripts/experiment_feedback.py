"""실제 데이터 기준 학습 1회와 독립 가상 정답 피드백 재학습 1회를 고정 조건으로 비교한다."""

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from backend.pipeline import FeedbackConfig, PipelineStore, ReplayConfig, assessment
from models import model, train
from scripts.runtime_logging import RuntimeLogger
from scripts.simulate import Config, digest, generate


def finish_replay(store, run):
    while run["status"] != "completed":
        if run["status"] in ("failed", "cancelled"):
            raise RuntimeError(run.get("error") or run["status"])
        run = store.process(run["id"], 256)
        print(f"관측 {run['config']['model_version']}: {run['processed']}/{run['rows']} · {run['status']}", flush=True)
    return run


def fit_version(config, source, test, weights, reports, logger):
    started = time.monotonic()
    training, validation, plan = train.prepare_training_data(source, test, config)
    if not plan["ready"]:
        raise ValueError(plan["errors"])
    logger.write("model", "training", event="data_prepared", config=config.model_dump(), data_plan=plan)
    report, checkpoint = train.train_variant("with_light", source, test, train.audit_split(source, test), config,
        features=config.features, prepared=(training, validation), data_plan=plan, defer_test_evaluation=True,
        progress=lambda point: logger.epoch(config.name, "with_light", point))
    version = train.save_model_pair({"with_light": checkpoint}, {"with_light": report}, weights, reports,
                                   force_new=True, make_default=False)
    logger.write("model", "training", event="fit_completed_test_pending", model_version=version,
                 data_plan=plan, epochs_run=report["training"]["epochs_run"])
    return version, {"job_id": config.name, "config": config.model_dump(), "models": {"with_light": report},
                     "model_version": version, "duration_seconds": time.monotonic() - started,
                     "data_plan": plan, "time_feature_policy": plan["time_feature_policy"]}


def finalize_test(version, training_report, source, test, weights, reports):
    saved = model.load_model(weights / version / "model.pt")
    report = training_report["models"]["with_light"]
    names = model.encoded_features(model.FEATURES)
    columns = [names.index(name) for name in report["encoded_features"]]
    selected = test.select(list(range(len(test.y))), columns)
    train.evaluate_variant_test(report, saved.network, {"mean": saved.mean, "scale": saved.scale}, selected, source)
    path = reports / version / "light_comparison.json"
    comparison = json.loads(path.read_text(encoding="utf-8"))
    comparison["modes"]["with_light"] = report
    train.write_json(path, comparison)
    train.write_json(reports / version / "metrics.json", report)
    training_report["pair_fingerprint"] = comparison["pair_fingerprint"]
    train.write_training_report(reports / version, training_report)
    return {"version": version, "metrics": report["model_metrics"], "source_metrics": report["source_metrics"],
            "quality_gate": report["quality_gate"], "checkpoint_sha256": digest(weights / version / "model.pt"),
            "epochs_run": report["training"]["epochs_run"], "best_epoch": report["training"]["best_epoch"],
            "fit_rows": training_report["data_plan"]["fit"]["rows"],
            "feedback_rows": training_report["data_plan"]["log_data"]["included_rows"]}


def slices(records):
    return {"empty_light_on": assessment([row for row in records if row["truth"] == 0 and row["sensors"]["Light"] >= 300]),
            "occupied_dark": assessment([row for row in records if row["truth"] == 1 and row["sensors"]["Light"] < 50]),
            "occupied": assessment([row for row in records if row["truth"] == 1]),
            "empty": assessment([row for row in records if row["truth"] == 0])}


def write_summary(directory, result):
    train.write_json(directory / "experiment.json", result)
    if result["status"] != "completed":
        return
    lines = ["# 고정 조건 피드백 재학습 비교", "",
        "기준 1회 + 후보 1회만 학습했습니다. 실제 test는 두 학습 완료 후 평가했고 결과로 추가 튜닝하지 않았습니다.", "",
        "| 평가 자료 | 모델 | 행수 | 정확도 | 정밀도 | 재현율 | F1 | 혼동행렬 [[TN, FP], [FN, TP]] |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for label, section in (("고정 실제 test", result["actual_test"]), ("고정 합성 heldout", result["synthetic_heldout"])):
        for key in ("baseline", "candidate"):
            metrics = section[key]["metrics"]
            scores = " | ".join(f"{metrics[name]:.4f}" if metrics[name] is not None else "계산 불가" for name in ("precision", "recall", "f1"))
            lines.append(f"| {label} | {key} | {section['rows']} | {metrics['accuracy']:.2%} | {scores} | {metrics['confusion_matrix']} |")
    lines += ["", f"실제 test 정확도 변화: {result['actual_test']['accuracy_change_pp']:+.4f}%p.",
        f"합성 heldout 정확도 변화: {result['synthetic_heldout']['accuracy_change_pp']:+.4f}%p.", "",
        f"피드백 포함: {result['actual_test']['candidate']['feedback_rows']}행. 모델은 동일 seed로 각각 처음부터 학습했습니다.", ""]
    before, after = result["synthetic_heldout"]["baseline"]["slices"], result["synthetic_heldout"]["candidate"]["slices"]
    real_before = result["actual_test"]["baseline"]["metrics"]["confusion_matrix"]
    real_after = result["actual_test"]["candidate"]["metrics"]["confusion_matrix"]
    lines += ["## 오탐·재실 누락과 미해결 문제", "",
        f"- 실제 test: 빈방 오탐 {real_before[0][1]} → {real_after[0][1]}건, 재실 누락 {real_before[1][0]} → {real_after[1][0]}건.",
        f"- 합성 빈방 점등 {before['empty_light_on']['rows']}건 중 오탐 {before['empty_light_on']['errors']} → {after['empty_light_on']['errors']}건.",
        f"- 합성 재실 {before['occupied']['rows']}건 중 누락 {before['occupied']['errors']} → {after['occupied']['errors']}건.",
        f"- 합성 어두운 재실 {before['occupied_dark']['rows']}건 중 누락 {before['occupied_dark']['errors']} → {after['occupied_dark']['errors']}건.", "",
        "전체 정확도 상승과 상황별 악화는 함께 판단해야 합니다. 이 실험은 모델 승격을 수행하지 않으며, 해결되지 않은 재실 누락이나 빈방 점등 오탐을 성공으로 간주하지 않습니다.", "",
        "## 해석 범위", "",
        "지도학습 재학습이며 강화학습 테스트는 미실행입니다. 행동·보상·정책 갱신이 구현되어 있지 않습니다.",
        "합성 heldout은 동일 생성 프로필의 별도 방/사건이며 희귀 사건을 우선 포함한 개발 표본입니다. 실제 강의실 성능이나 자연 발생 비율의 성능을 보장하지 않습니다.",
        "실제 test는 시간적으로 학습 이전·이후가 혼재한 기존 고정 CSV이므로 순수 미래 성능도 아닙니다.",
        "기본 모델 포인터는 변경하지 않았습니다. 재시도·추가 튜닝 없이 결과를 보존합니다.", "",
        "유형별·빈방 점등·재실 누락 지표와 provenance는 experiment.json 및 synthetic_comparison.json, 버전별 training.json에 기록합니다."]
    result["report_rendering"] = {"renderer_sha256": digest(model.ROOT / "scripts/experiment_feedback.py"),
                                  "rendered_at_utc": datetime.now(timezone.utc).isoformat()}
    train.write_json(directory / "experiment.json", result)
    (directory / "experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(experiment_id):
    if not re.fullmatch(r"[a-z0-9-]{1,80}", experiment_id):
        raise ValueError("실험 ID는 영문 소문자·숫자·하이픈 1~80자로 지정하세요.")
    root = model.ROOT
    directory = root / "reports/experiments" / experiment_id
    weights = root / "models/weights/experiments" / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "experiment.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "initializing"}, stream)
    if weights.exists() and any(weights.iterdir()):
        raise ValueError("기존 실험 가중치를 덮어쓰지 않습니다. 새 실험 ID를 사용하세요.")
    os.environ["CLASSROOM_STATE_DIR"] = str(directory / "state")
    protected = [root / "data" / name for name in ("training_dataset.csv", "test_dataset.csv", "dataset_sources.json")]
    protected += [model.CURRENT_PATH, *model.WEIGHTS_DIR.glob("v*/model*.pt")]
    protected += [path for path in model.REPORTS_DIR.glob("v*/*") if path.is_file()]
    original_hashes = {str(path.relative_to(root)): digest(path) for path in protected}
    config = train.TrainingConfig(name="실제 데이터 기준", seed=42, epochs=200, patience=0, compare_light=False,
                                  shared_room_split=True, make_default=False)
    generation_config = Config(seed=20261117, rooms=3, days=7, step_seconds=60,
                               calibration_scope="fit_only", observation_mode="stress", validation_ratio=config.validation_ratio)
    result = {"status": "running", "experiment_id": experiment_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"training_runs": 2, "config": config.model_dump(), "generation": asdict(generation_config),
                     "replay_max_rows": 3000, "feedback_max_rows": 600, "replay_seed": 42,
                     "test_metrics_after_both_fits": True, "automatic_tuning": False, "from_scratch": True},
        "reinforcement_learning": {"executed": False, "reason": "행동·보상·정책 갱신 미구현. BCE 이진분류 지도학습 재학습만 실행."},
        "source_code_sha256": {name: digest(root / name) for name in
            ("models/model.py", "models/train.py", "scripts/simulate.py", "scripts/runtime_logging.py", "backend/pipeline.py", "scripts/experiment_feedback.py")},
        "protected_files_before": original_hashes, "weights_directory": str(weights.relative_to(root)),
        "resources": {"torch_threads": 1, "cpu_only": True}}
    write_summary(directory, result)
    logger = RuntimeLogger(root=directory / "logs")
    result["logs_directory"] = str(logger.directory.relative_to(root))
    try:
        source = train.load_dataset(root / "data/training_dataset.csv")
        test = train.load_dataset(root / "data/test_dataset.csv")
        print("기준 모델 학습 1/2, test 지표는 보류합니다.", flush=True)
        baseline, baseline_report = fit_version(config, source, test, weights, directory, logger)
        result["completed_training_runs"] = 1
        write_summary(directory, result)
        print("fit-only 생활 예외·센서 오류 생성", flush=True)
        generated, quality = generate(generation_config, directory / "state/simulated", root)
        result["generation"] = {"id": generated.name, "manifest_sha256": digest(generated / "manifest.json"),
                                "observation": quality["observation"], "quality_passed": quality["passed"]}
        store = PipelineStore(root, state_dir=directory / "state", logger=logger, weights_dir=weights)
        replay = finish_replay(store, store.create_run(ReplayConfig(generation_id=generated.name, model_version=baseline, max_rows=3000, seed=42)))
        snapshot = store.create_dataset(replay["id"], FeedbackConfig(max_rows=600, seed=42))
        result.update(baseline_run_id=replay["id"], sampling=replay["sampling"], feedback=snapshot)
        candidate_config = config.model_copy(update={"name": "독립 가상 정답 로그 피드백", "pipeline_dataset": snapshot["id"]})
        print("피드백 모델 학습 2/2, test 지표는 보류합니다.", flush=True)
        candidate, candidate_report = fit_version(candidate_config, source, test, weights, directory, logger)
        if baseline_report["data_plan"]["validation"] != candidate_report["data_plan"]["validation"]:
            raise RuntimeError("두 모델의 실제 validation 분할이 다릅니다.")
        if candidate_report["data_plan"]["log_data"]["included_rows"] < 1:
            raise RuntimeError("피드백 학습 행이 없습니다.")
        result["completed_training_runs"] = 2
        write_summary(directory, result)
        compared = finish_replay(store, store.create_comparison(replay["id"], candidate))
        comparison = store.comparison(compared["id"])
        train.write_json(directory / "synthetic_comparison.json", comparison)
        baseline_rows = [row for row in store.records(replay["id"]) if row["partition"] in ("validation", "test")]
        candidate_rows = store.records(compared["id"])
        if {row["key"] for row in baseline_rows} != {row["key"] for row in candidate_rows}:
            raise RuntimeError("동일 heldout 관측 비교가 아닙니다.")
        print("두 학습 완료. 고정 실제 test를 최종 비교합니다.", flush=True)
        actual_baseline = finalize_test(baseline, baseline_report, source, test, weights, directory)
        actual_candidate = finalize_test(candidate, candidate_report, source, test, weights, directory)
        result["actual_test"] = {"rows": len(test.y), "sha256": test.summary["sha256"],
            "baseline": actual_baseline, "candidate": actual_candidate,
            "accuracy_change_pp": (actual_candidate["metrics"]["accuracy"] - actual_baseline["metrics"]["accuracy"]) * 100}
        result["synthetic_heldout"] = {"rows": len(baseline_rows), "baseline": {"metrics": comparison["baseline"]["overall"], "slices": slices(baseline_rows)},
            "candidate": {"metrics": comparison["candidate"]["overall"], "slices": slices(candidate_rows)},
            "accuracy_change_pp": (comparison["candidate"]["overall"]["accuracy"] - comparison["baseline"]["overall"]["accuracy"]) * 100,
            "natural_prevalence": False, "candidate_run_id": compared["id"]}
        result["protected_files_unchanged"] = all(digest(root / name) == value for name, value in original_hashes.items())
        if not result["protected_files_unchanged"]:
            raise RuntimeError("보존 대상 파일이 바뀌었습니다.")
        result.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_summary(directory, result)
        print(json.dumps({"report": str(directory.relative_to(root)), "actual_test_change_pp": result["actual_test"]["accuracy_change_pp"],
                          "synthetic_change_pp": result["synthetic_heldout"]["accuracy_change_pp"], "training_runs": 2}, ensure_ascii=False), flush=True)
        return result
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_summary(directory, result)
        raise


def verify_reproducibility(experiment_id):
    """설정 변경이나 test 평가 없이 두 모델의 학습을 각각 1회 재현해 텐서를 대조한다."""
    if not re.fullmatch(r"[a-z0-9-]{1,80}", experiment_id):
        raise ValueError("실험 ID 형식이 잘못되었습니다.")
    root = model.ROOT
    directory = root / "reports/experiments" / experiment_id
    result = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if result["status"] != "completed":
        raise ValueError("완료된 실험만 재현 검증할 수 있습니다.")
    output_name = os.environ.get("FEEDBACK_REPRO_OUTPUT", "reproducibility.json")
    if not re.fullmatch(r"reproducibility(?:-[a-z0-9]+)?\.json", output_name):
        raise ValueError("재현 검증 파일명 형식이 잘못되었습니다.")
    output = directory / output_name
    if output.exists():
        raise FileExistsError("이미 재현 검증 결과가 있습니다. 추가 학습을 자동 반복하지 않습니다.")
    checks = []
    with tempfile.TemporaryDirectory(prefix="feedback-reproduce-") as temporary:
        state = Path(temporary) / "state"
        state.mkdir()
        shutil.copy2(directory / "state/pipeline.sqlite3", state / "pipeline.sqlite3")
        generation = result["generation"]["id"]
        shutil.copytree(directory / "state/simulated" / generation, state / "simulated" / generation)
        os.environ["CLASSROOM_STATE_DIR"] = str(state)
        source = train.load_dataset(root / "data/training_dataset.csv")
        test = train.load_dataset(root / "data/test_dataset.csv")
        for version in ("v1", "v2"):
            record = json.loads((directory / version / "training.json").read_text(encoding="utf-8"))
            config = train.TrainingConfig(**record["config"])
            fit, validation, plan = train.prepare_training_data(source, test, config)
            if plan["fingerprint"] != record["data_plan"]["fingerprint"]:
                raise ValueError("재현 입력 계획이 원래 학습과 다릅니다.")
            names = model.encoded_features(model.FEATURES)
            columns = [names.index(name) for name in model.encoded_features(config.features)]
            fit = train.select_encoded_columns(fit, columns)
            validation = train.select_encoded_columns(validation, columns)
            print(f"동일 조건 재현 검증: {version}, test 평가 없음", flush=True)
            network, scaler, training = train.train_network(fit, validation, config)
            checkpoint = torch.load(root / result["weights_directory"] / version / "model.pt", weights_only=True, map_location="cpu")
            tensors_equal = all(torch.equal(value, checkpoint["state_dict"][key]) for key, value in network.state_dict().items())
            scalers_equal = all(torch.equal(value, checkpoint[key]) for key, value in scaler.items())
            checks.append({"version": version, "weights_bitwise_equal": tensors_equal, "scaler_bitwise_equal": scalers_equal,
                "history_equal": training["history"] == checkpoint["training"]["history"], "epochs_run": training["epochs_run"],
                "best_epoch_equal": training["best_epoch"] == checkpoint["training"]["best_epoch"],
                "data_plan_fingerprint_equal": True, "original_checkpoint_sha256": digest(root / result["weights_directory"] / version / "model.pt")})
    prior_runs = sum(json.loads(path.read_text(encoding="utf-8")).get("reproduction_training_runs", 0)
                     for path in directory.glob("reproducibility*.json"))
    report = {"experiment_id": experiment_id, "reproduction_training_runs": len(checks), "hyperparameter_changes": 0,
              "all_reproduction_training_runs": prior_runs + len(checks),
              "original_comparison_training_runs": result["completed_training_runs"],
              "total_training_runs_including_reproduction": result["completed_training_runs"] + prior_runs + len(checks),
              "verification_input_layout": "same encoded column selection as train_variant; column-major strides preserved",
              "environment": {"torch": str(torch.__version__), "threads": torch.get_num_threads(),
                              "scope": "same container dependencies and CPU; cross-device bitwise identity is not guaranteed"},
              "test_evaluation_executed": False, "checks": checks,
              "passed": all(all(row[key] for key in ("weights_bitwise_equal", "scaler_bitwise_equal", "history_equal", "best_epoch_equal")) for row in checks)}
    train.write_json(output, report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    experiment_id = os.environ.get("FEEDBACK_EXPERIMENT_ID", "feedback-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    if os.environ.get("FEEDBACK_VERIFY_ONLY") == "1":
        if not verify_reproducibility(experiment_id)["passed"]:
            raise SystemExit(1)
    else:
        run(experiment_id)
