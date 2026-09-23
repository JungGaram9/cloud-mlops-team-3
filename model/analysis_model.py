import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "웹 데이터.csv"
SPLIT_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "model" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "purchase_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

TARGET = "Revenue"
SKEWED_FEATURES = ["PageValues", "ProductRelated", "ProductRelated_Duration"]
SCALED_FEATURES = ["ExitRates"]
CATEGORICAL_FEATURES = ["Month", "TrafficType"]
FEATURES = SKEWED_FEATURES + SCALED_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def load_raw_data(path=DATA_PATH):
    """원본 데이터를 읽는다. 확장자가 .csv여도 실제 포맷이 xlsx이면 엑셀로 읽는다."""
    with open(path, "rb") as f:
        is_xlsx = f.read(2) == b"PK"
    if is_xlsx:
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def clean_features(df):
    """모델 입력 피쳐만 추출하고 Month 표기를 3글자 약어(June -> Jun)로 통일한다."""
    X = df[FEATURES].copy()
    X["Month"] = X["Month"].astype(str).str.strip().str[:3].str.title()
    return X


def clean_data(df):
    """피쳐를 정리하고 타겟을 0/1 정수로 변환해 붙인다."""
    data = clean_features(df)
    data[TARGET] = df[TARGET].astype(int)
    return data


def split_data(df, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    """타겟 비율을 유지(stratify)하며 훈련/검증/테스트 세트로 분할한다. 기본 비율은 60/20/20."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET],
        random_state=random_state,
    )
    return train, val, test


def exclude_suspect_purchases(df):
    """PageValues 로그 누락이 의심되는 행(Revenue=1 이면서 PageValues=0)을 제외한다.

    학습 세트에만 적용하고 검증/테스트 세트는 실제 분포 그대로 평가하기 위해 원본을 유지한다.
    """
    suspect = (df[TARGET] == 1) & (df["PageValues"] == 0)
    return df[~suspect], int(suspect.sum())


def save_splits(train_df, val_df, test_df, split_dir=SPLIT_DIR):
    """분할된 세트를 CSV로 저장해 이후 동일한 세트로 재평가할 수 있게 한다."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train_df, "val": val_df, "test": test_df}.items():
        part.to_csv(split_dir / f"{name}.csv", index=False, encoding="utf-8")


def build_preprocessor():
    """치우친 수치형은 log1p 후 표준화, ExitRates는 표준화, 범주형은 원-핫 인코딩한다.

    표본이 30건 미만인 범주(희귀 TrafficType 등)와 학습 시 없던 범주는 'infrequent' 그룹으로 처리한다.
    """
    skewed = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()
    )
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False
    )
    return ColumnTransformer(
        [
            ("skewed", skewed, SKEWED_FEATURES),
            ("scaled", StandardScaler(), SCALED_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_candidates():
    """클래스 불균형을 class_weight로 보정한 후보 모델들을 전처리기와 결합해 반환한다."""
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        for name, model in models.items()
    }


def find_best_threshold(y_true, proba):
    """검증 세트에서 F1 점수를 최대화하는 분류 임계값을 찾는다."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def compute_metrics(y_true, proba, threshold):
    """임계값 기준 분류 지표, 확률 기반 지표, 다수 클래스 기준선 정확도를 계산한다."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    positive_rate = float(np.mean(y_true))
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "baseline_accuracy": round(max(positive_rate, 1 - positive_rate), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def predict(bundle, df):
    """저장된 모델 번들로 구매 확률과 임계값 기반 예측 라벨(0/1)을 반환한다."""
    X = clean_features(df)[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    return proba, (proba >= bundle["threshold"]).astype(int)


def evaluate(bundle, df):
    """타겟이 포함된 데이터프레임에 대해 모델 성능 지표를 계산한다."""
    proba, _ = predict(bundle, df)
    return compute_metrics(df[TARGET].astype(int).to_numpy(), proba, bundle["threshold"])


def train(data_path=DATA_PATH):
    """분할 → 학습 세트 이상치 제외 → 후보 모델 학습(train) → 모델/임계값 선택(val) → 최종 평가(test) → 산출물 저장.

    테스트 세트는 모델 선택과 임계값 결정에 사용하지 않고 마지막 평가에만 한 번 사용한다.
    """
    df = clean_data(load_raw_data(data_path))
    train_df, val_df, test_df = split_data(df)
    train_df, excluded = exclude_suspect_purchases(train_df)
    save_splits(train_df, val_df, test_df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    validation, fitted = {}, {}
    for name, pipeline in build_candidates().items():
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, proba)
        validation[name] = {"threshold": round(threshold, 4), **compute_metrics(y_val, proba, threshold)}
        fitted[name] = pipeline

    best_name = max(validation, key=lambda name: validation[name]["pr_auc"])
    bundle = {
        "model_name": best_name,
        "pipeline": fitted[best_name],
        "threshold": validation[best_name]["threshold"],
        "features": FEATURES,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    report = {
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "excluded_from_train": {"rule": "Revenue=1 & PageValues=0", "count": excluded},
        "selection_metric": "val pr_auc",
        "selected_model": best_name,
        "threshold": bundle["threshold"],
        "validation": validation,
        "test": evaluate(bundle, test_df),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def evaluate_saved(model_path=MODEL_PATH, test_path=SPLIT_DIR / "test.csv"):
    """저장된 모델과 저장된 테스트 세트를 불러와 성능을 재평가한다."""
    bundle = joblib.load(model_path)
    return bundle["model_name"], evaluate(bundle, pd.read_csv(test_path))


def print_metrics(title, metrics):
    """지표 딕셔너리를 보기 좋게 출력한다."""
    print(f"\n[{title}]")
    for key, value in metrics.items():
        print(f"  {key:<18} {value}")


def main():
    parser = argparse.ArgumentParser(description="웹사이트 구매 전환 예측 모델")
    parser.add_argument("mode", nargs="?", choices=["train", "evaluate"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        report = train()
        print(f"분할 크기: {report['split_sizes']}")
        print(f"학습 세트 제외: {report['excluded_from_train']}")
        for name, metrics in report["validation"].items():
            print_metrics(f"검증 세트 - {name}", metrics)
        print(f"\n선택된 모델: {report['selected_model']} (임계값 {report['threshold']})")
        print_metrics("테스트 세트 - 최종 평가", report["test"])
        print(f"\n모델 저장: {MODEL_PATH}\n지표 저장: {METRICS_PATH}")
    else:
        model_name, metrics = evaluate_saved()
        print_metrics(f"테스트 세트 - {model_name}", metrics)


if __name__ == "__main__":
    main()
