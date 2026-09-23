"""사용자 입력을 받아 저장된 모델로 구매 전환 여부를 예측하는 대화형 스크립트.

사용법:
    python model/predict.py
"""

import sys

import joblib
import pandas as pd

from analysis_model import MODEL_PATH, predict

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_number(text, cast, minimum=None, maximum=None):
    """문자열을 숫자로 변환하고 허용 범위를 벗어나면 ValueError를 발생시킨다."""
    try:
        value = cast(text.strip())
    except ValueError:
        kind = "정수" if cast is int else "숫자"
        raise ValueError(f"{kind}를 입력하세요.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{minimum} 이상이어야 합니다.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{maximum} 이하여야 합니다.")
    return value


def parse_month(text):
    """1~12 숫자, '11월', 영문 월 이름(Nov, November)을 3글자 약어로 변환한다."""
    text = text.strip().removesuffix("월")
    if text.isdigit() and 1 <= int(text) <= 12:
        return MONTHS[int(text) - 1]
    abbr = text[:3].title()
    if abbr in MONTHS:
        return abbr
    raise ValueError("1~12 숫자 또는 영문 월 이름(예: Nov)을 입력하세요.")


FIELDS = [
    ("PageValues", "페이지 가치 (0 이상, 예: 0 ~ 360)", lambda s: parse_number(s, float, 0)),
    ("ExitRates", "이탈률 (0 ~ 1, 예: 0.02)", lambda s: parse_number(s, float, 0, 1)),
    ("ProductRelated", "상품 관련 페이지 방문 수 (0 이상 정수)", lambda s: parse_number(s, int, 0)),
    ("ProductRelated_Duration", "상품 페이지 체류 시간 (초, 0 이상)", lambda s: parse_number(s, float, 0)),
    ("Month", "방문 월 (1~12 또는 Nov 등)", parse_month),
    ("TrafficType", "유입 경로 유형 (1 ~ 20 정수)", lambda s: parse_number(s, int, 1, 20)),
]


def ask(label, parser):
    """유효한 값이 입력될 때까지 반복해서 입력을 받는다."""
    while True:
        try:
            return parser(input(f"  {label}: "))
        except ValueError as error:
            print(f"    -> {error}")


def known_months(bundle):
    """모델이 학습 때 본 Month 범주 목록을 반환한다."""
    encoder = bundle["pipeline"].named_steps["preprocess"].named_transformers_["categorical"]
    return set(encoder.categories_[0])


def predict_one(bundle, values):
    """입력값 딕셔너리 하나로 구매 확률과 예측 라벨(0/1)을 반환한다."""
    proba, label = predict(bundle, pd.DataFrame([values]))
    return float(proba[0]), int(label[0])


def main():
    if not MODEL_PATH.exists():
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        print("먼저 python model/analysis_model.py train 을 실행하세요.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    months = known_months(bundle)
    print(f"구매 전환 예측 (모델: {bundle['model_name']}, 임계값: {bundle['threshold']:.1%})")
    print("종료하려면 Ctrl+C 를 누르세요.")

    try:
        while True:
            print("\n세션 정보를 입력하세요.")
            values = {name: ask(label, parser) for name, label, parser in FIELDS}
            proba, label = predict_one(bundle, values)
            print(f"\n  구매 확률: {proba:.1%}")
            print(f"  예측 결과: {'구매' if label else '미구매'}")
            if values["Month"] not in months:
                print(f"  참고: {values['Month']}은(는) 학습 데이터에 없는 월이라 월 정보 없이 예측했습니다.")
            if input("\n계속 예측할까요? (y/n): ").strip().lower() != "y":
                break
    except (KeyboardInterrupt, EOFError):
        print()
    print("종료합니다.")


if __name__ == "__main__":
    main()
