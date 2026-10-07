# v5 학습 보고서

실험: 조도 제외 · 피처 영향 검증
학습 작업: 8fb40b62711f47ddb8637eea2b3850f1

## 학습 설정

```json
{
  "name": "조도 제외 · 피처 영향 검증",
  "features": [
    "Temperature",
    "Humidity",
    "CO2",
    "Hour",
    "DayOfWeek"
  ],
  "epochs": 200,
  "learning_rate": 0.01,
  "batch_size": 0,
  "hidden_layers": [
    16,
    8
  ],
  "activation": "relu",
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
  "make_default": false
}
```

## 평가

| 모델 | 피처 | 정확도 | 정밀도 | 재현율 | F1 | 최적 에포크 |
|---|---|---:|---:|---:|---:|---:|
| 선택 모델 | Temperature, Humidity, CO2, Hour, DayOfWeek | 85.17% | 63.34% | 92.75% | 0.7527 | 200 |

검증 손실이 가장 낮은 가중치를 선택했습니다. 손실 곡선과 데이터 해시는 training.json에 보존합니다.
전체 평가 파일은 학습 기간 이전·이후 기록을 포함하므로 순수한 미래 예측 성능이 아닙니다.
피처 제거·설정 변경으로 85% 미만이 되어도 실험 모델을 보존합니다.
