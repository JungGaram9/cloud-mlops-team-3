# v7 학습 보고서

실험: Graz 보강 · 요일 제외 · tanh
학습 작업: 9fbb82768f124dbe92b2b7240eef96d2

## 학습 설정

```json
{
  "name": "Graz 보강 · 요일 제외 · tanh",
  "features": [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "Hour"
  ],
  "epochs": 200,
  "learning_rate": 0.01,
  "batch_size": 0,
  "hidden_layers": [
    16,
    8
  ],
  "activation": "tanh",
  "optimizer": "adam",
  "dropout": 0.0,
  "weight_decay": 0.0,
  "validation_ratio": 0.2,
  "patience": 30,
  "threshold": 0.5,
  "seed": 42,
  "normalize": true,
  "balance_classes": false,
  "compare_light": true,
  "make_default": false,
  "datasets": [
    "uci",
    "graz_room_a"
  ]
}
```

## 평가 (UCI 평가 파일)

| 모델 | 피처 | 정확도 | 정밀도 | 재현율 | F1 | 최적 에포크 |
|---|---|---:|---:|---:|---:|---:|
| 선택 모델 | Temperature, Humidity, Light, CO2, Hour | 87.43% | 66.24% | 98.54% | 0.7923 | 29 |
| 조도 제외 비교 모델 | Temperature, Humidity, CO2, Hour | 78.16% | 53.22% | 84.57% | 0.6533 | 28 |

## 장소·기간별 평가 (평일 업무시간)

| 모델 | 평가 데이터 | 전체 정확도 | 업무시간 정확도 | 다수 클래스 기준선 | 업무시간 F1 |
|---|---|---:|---:|---:|---:|
| 선택 모델 | UCI 사무실 평가 | 87.43% | 85.52% | 67.18% | 0.9014 |
| 선택 모델 | Graz A 12월 · 미래 기간 | 96.62% | 89.33% | 54.75% | 0.9035 |
| 선택 모델 | Graz B 12월 · 학습하지 않은 공간 | 91.73% | 76.90% | 51.19% | 0.7563 |
| 조도 제외 비교 모델 | UCI 사무실 평가 | 78.16% | 68.20% | 67.18% | 0.7813 |
| 조도 제외 비교 모델 | Graz A 12월 · 미래 기간 | 95.95% | 86.90% | 54.75% | 0.8840 |
| 조도 제외 비교 모델 | Graz B 12월 · 학습하지 않은 공간 | 91.93% | 77.79% | 51.19% | 0.7799 |

검증 손실이 가장 낮은 가중치를 선택했습니다. 손실 곡선과 데이터 해시는 training.json에 보존합니다.
전체 평가 파일은 학습 기간 이전·이후 기록을 포함하므로 순수한 미래 예측 성능이 아닙니다.
피처 제거·설정 변경으로 85% 미만이 되어도 실험 모델을 보존합니다.
