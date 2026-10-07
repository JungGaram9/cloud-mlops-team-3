# 강의실 센서 실험 · cloud-mlops-team-3

기온·습도·조도·CO₂·시간·요일로 측정 시점의 강의실 사용 여부를 판별하는 PyTorch 프로젝트입니다. React 화면에서 측정값 한 건을 입력하고 조도 포함·제외 모델을 비교합니다. FastAPI는 Swagger UI, React는 **Vite 개발 서버**로 실행합니다.

## 실행하기

Docker Desktop의 Linux 컨테이너 엔진과 [Docker Compose 2.24 이상](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/)이 필요합니다. Python·Node·가상환경을 별도로 설치하지 않습니다. 저장소 루트의 CMD에서 실행합니다.

```cmd
docker compose up -d --build
```

모델 이미지 준비 → **trainer가 두 모델을 학습·등록 → API healthy → 웹 실행** 순서입니다. 조도 포함 모델의 정확도가 85% 미만이면 학습 명령이 실패해 API 시작을 막습니다. 프런트는 프로덕션 빌드하지 않고 `npm ci` 후 `npm run dev`를 실행합니다.

- [실험 웹사이트](http://localhost:5173): 예시 측정값, 시간·요일 선택, 단건 예측, 조도 비교, 모델 버전 선택.
- [Swagger UI](http://localhost:8000/docs): 입력 규격과 실제 요청 실행.

처음에는 이미지·패키지 다운로드를 위한 인터넷 연결이 필요합니다. 실행 진행 상황과 준비 상태는 다음 명령으로 확인합니다. trainer의 `Exited (0)`은 학습이 성공해 정상 종료된 상태입니다.

```cmd
docker compose ps -a
docker compose logs --tail 50 trainer api web
```

## 환경 변수

Compose는 루트 `.env`를 자동으로 읽습니다. `.env`가 없으면 기본 포트로 실행하며, 서비스 환경 변수는 `.env.example` 기본값 위에 선택적인 `.env` 값을 적용합니다. 포트를 변경하려면 사용자가 직접 예제를 복사하고 `API_PORT` 또는 `WEB_PORT`를 수정합니다.

```cmd
copy .env.example .env
docker compose up -d --build
```

`.env`는 Git에서 제외합니다. 브라우저에서도 변경한 포트로 접속합니다. 웹의 `/api` 요청은 컨테이너 내부 API로 전달하므로 API 호스트 포트를 바꿔도 프런트 수정이 필요 없습니다. 개발 서버는 로컬 주소에만 공개합니다.

## 입력하고 예측하기

1. **밝은 실내 / 불 꺼진 실내 / CO₂가 높은 실내**를 선택하면 센서값 4개와 시간·요일을 채우고 예측합니다.
2. 시간·요일을 선택하고, 센서 숫자나 슬라이더를 조절합니다. 센서마다 단위·범위·체감 수준이 표시됩니다.
3. **이 센서값으로 예측하기**를 누르면 사용 여부와 **사용 중일 확률**을 받습니다.
4. **조도 피처 사용** 스위치로 같은 버전 안의 조도 포함·제외 모델을 전환합니다.
5. 왼쪽 **모델 버전 관리**에서 버전을 선택하고 **이 모델 쌍으로 실험하기**를 누르면 해당 버전의 실제 가중치로 실험합니다.

조도 제외 모델은 `Light=0`으로 바꾸는 방식이 아니라 **Light 컬럼 없이 따로 학습한 모델**입니다. 두 모델은 같은 학습·검증·평가 행과 학습 설정을 사용합니다. 입력을 바꾸면 단건 예측이 바뀔 수 있지만, 고정된 평가 파일의 정확도는 변하지 않습니다. 예시값은 가상의 환경이고 실제 재실 정답은 없습니다. 체감 수준은 입력을 돕는 대략적인 설명입니다.

v1·v2는 시간·요일을 학습하지 않은 기존 센서 모델입니다. 해당 버전으로 실험하면 시간·요일 입력을 사용하지 않으며 웹에서 비활성화합니다. v3부터 시간·요일도 실제 학습·추론에 사용합니다. 입력이 달라도 신경망의 특정 구간에서 같은 결과가 나올 수 있습니다.

## 입력 규격

| 필드 | 의미·단위 | 허용 범위 |
|---|---|---|
| `Temperature` | 기온, °C | -50~60 |
| `Humidity` | 상대습도, % | 0~100 |
| `Light` | 밝기, lux | 0~1,000,000 |
| `CO2` | 이산화탄소 농도, ppm | 0 초과~1,000,000 |
| `Hour` | 측정 시간, 시 | 정수 0~23 |
| `DayOfWeek` | 측정 요일 | 정수 0~6: 월 0, 화 1, 수 2, 목 3, 금 4, 토 5, 일 6 |

시간·요일은 측정한 공간의 기준으로 입력합니다. 예: **수요일 오후 2시 → Hour 14, DayOfWeek 2**. 분·초·날짜·예약은 입력 피처가 아닙니다. 학습도 CSV의 `date`에서 동일하게 시와 요일을 추출합니다.

웹 슬라이더는 흔히 실험할 범위인 기온 15~35°C, 습도 0~100%, 조도 0~1,000 lux, CO₂ 350~2,500 ppm을 제공합니다. 센서 숫자 칸은 소수와 슬라이더보다 넓은 허용 범위를 입력할 수 있습니다. 허용 범위와 실제 학습 데이터의 분포는 서로 다릅니다.

```json
{
  "Temperature": 22.5,
  "Humidity": 45.0,
  "Light": 450.0,
  "CO2": 900.0,
  "Hour": 14,
  "DayOfWeek": 2
}
```

```cmd
curl http://localhost:8000/compare --json @backend/examples/mock_sensor.json
curl http://localhost:8000/compare?version=v2 --json @backend/examples/mock_sensor.json
```

모든 요청에 위 6개 필드를 보냅니다. 이전 모델은 자신의 피처 목록에 없는 시간·요일을 무시합니다. 응답의 `with_light`와 `without_light`는 같은 버전의 두 모델입니다. 각 결과에는 실제 `model_version`, `occupancy`(0 비어 있음 / 1 사용 중), 한글 `label`, 사용 중일 `probability`가 있습니다. 확률 0.5 이상을 사용 중으로 판별합니다.

누락·추가 필드, 문자열 숫자, 참/거짓, 범위 오류, NaN·무한대는 HTTP 400으로 거부합니다. 없는 모델 쌍은 HTTP 404입니다. Swagger 자산도 API에서 직접 제공합니다.

| API | 용도 |
|---|---|
| `GET /health` | API 기본 모델 준비 상태 |
| `GET /versions` | 역대 모델 버전·피처·평가·두 가중치 상태 |
| `GET /metrics?version=v3` | 해당 버전 조도 포함 평가 |
| `GET /comparison?version=v3` | 해당 버전 두 모델의 평가 비교 |
| `POST /predict?version=v3` | 해당 버전 조도 포함 예측 |
| `POST /compare?version=v3` | 해당 버전 두 모델 예측 |

`version`을 생략하면 API의 기본 버전을 사용합니다. 웹 버전 선택은 **그 브라우저의 실험 대상**을 바꾸며 다른 팀원의 기본 모델을 변경하지 않습니다.

## 모델 버전과 가중치

**화면이나 폴더 변경은 모델 버전이 아닙니다.** 두 모델의 피처·입력 변환 방식·표준화·가중치·임계값으로 모델 쌍의 식별값을 계산합니다.

- 기존 모델 쌍과 같으면 해당 버전을 재사용하고 가중치·보고서를 덮어쓰지 않습니다.
- 예측을 결정하는 내용이 달라지면 다음 번호의 `vN` 폴더를 만들어 두 가중치를 함께 저장합니다.
- 시간·요일을 추가한 이번 모델은 **v3**입니다. 기존 v1·v2 가중치는 그대로 보존했습니다. 두 기존 버전의 가중치가 같아 버전 목록에 같은 모델의 기존 이력임을 표시합니다.
- `models/weights/current.json`이 기본 모델 쌍을 가리킵니다. 학습 성공 후 두 가중치와 평가 보고서를 완성한 다음 현재 정보를 교체합니다.
- 작은 CPU 체크포인트를 저장소에서 공유할 수 있도록 `models/weights/`를 Git 제외하지 않습니다. 이 작업에서 add·commit·push는 실행하지 않습니다.

```text
frontend/
  src/main.jsx                 화면·예시값·시간/요일 입력·버전 선택·API 요청
  src/styles.css               디자인·반응형 전체
  vite.config.js / package.json / package-lock.json
  index.html / public/favicon.svg
backend/
  app.py                       FastAPI·Swagger·버전별 모델 쌍 요청
  test_app.py                  입력·모델·API·버전 보존 테스트
  Dockerfile / Dockerfile.dockerignore
  requirements.txt / requirements.lock.txt
  examples/mock_sensor.json
models/
  model.py                     입력 규격·공통 변환·신경망·로드·추론·모델 식별값
  train.py                     CSV 검증·시간순 분할·학습·평가·버전 등록
  weights/
    current.json               API 기본 모델 쌍
    v1/                        기존 가중치·메타데이터
    v2/                        기존 가중치·메타데이터
    v3/
      model.pt                 조도 포함 가중치·표준화·입력 규격
      model_without_light.pt   조도 제외 가중치·표준화·입력 규격
  reports/v1/ / v2/ / v3/       버전별 실제 평가 보고서
data/                          원본 학습·평가 CSV
compose.yaml / .env.example     루트 실행·포트 예제
```

기능은 프런트 화면 한 파일, API 한 파일, 모델 한 파일, 학습 한 파일 중심으로 관리합니다. 학습과 추론이 `models/model.py`의 동일한 입력 변환을 사용합니다. JSX·CSS는 Vite, API 코드는 Uvicorn이 변경을 반영합니다. 학습 코드를 바꾸면 시작 명령으로 재학습합니다.

기존 이미지를 사용해 재학습만 하려면:

```cmd
docker compose run --no-deps trainer
docker compose restart api
```

## v3 평가 결과

| 모델 | 원본 입력 | 정확도 | 정밀도 | 재현율 | F1 |
|---|---|---:|---:|---:|---:|
| 조도 포함 | 센서 4개 + 시간·요일 | **98.96%** | 96.83% | 98.97% | 0.9789 |
| 조도 제외 | 센서 3개 + 시간·요일 | **85.17%** | 63.34% | 92.75% | 0.7527 |

조도 포함 시 정확도가 **13.79%p** 높았고 이번 두 모델은 모두 85%를 충족했습니다. 전체 평가 파일을 기준으로 한 결과입니다. 조도 제외 모델의 정밀도가 더 낮다는 점도 함께 확인해야 합니다.

- 학습 파일 `data/training_dataset.csv`: 8,143건, 시간순 앞 6,514건 학습 / 뒤 1,629건 검증.
- 별도 평가 파일 `data/test_dataset.csv`: 12,417건. 평가 정답으로 가중치·임계값을 조정하지 않습니다.
- 시간은 24시간 주기의 sin/cos, 요일은 7일 주기의 sin/cos와 주말 여부로 변환합니다. 실제 신경망 입력은 조도 포함 9개 / 제외 8개입니다.
- PyTorch MLP 입력 → 16 → 8 → 1, ReLU, CPU, 시드 42, Adam, BCE 손실, 최대 200회.
- 학습 구간에서만 표준화를 계산하고, 검증 손실 최소 가중치를 선택합니다. 30회 개선이 없으면 종료합니다.
- 평가 파일에는 학습 기간 이전 2,665건과 이후 9,752건이 함께 있습니다. 전체 정확도를 순수한 미래 예측 성능으로 해석할 수 없습니다. 이후 구간 지표도 보고서에 기록합니다.

이 모델은 측정 시점의 재실 판별이며 미래 예약이나 미래 재실 예측이 아닙니다. 원본은 [v3 평가](models/reports/v3/metrics.json)와 [v3 조도 비교](models/reports/v3/light_comparison.json)에서 확인합니다.

## 검증과 중지

```cmd
docker compose exec -T api python -m unittest backend.test_app -v
docker compose ps -a
docker compose logs --tail 50 trainer api web
docker compose stop api web
```

API 의존성의 전체 버전과 프런트 npm 잠금 파일을 고정했습니다. 체크포인트는 `torch.load(weights_only=True, map_location="cpu")`로 로드하고 `eval()`·`inference_mode()`로 추론합니다. 별도 가상환경과 프런트 프로덕션 빌드는 사용하지 않습니다.
