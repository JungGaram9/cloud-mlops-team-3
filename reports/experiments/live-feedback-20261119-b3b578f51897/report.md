# EC2 v7 → v8 피드백 학습 실측 결과

2026-11-19 배포 API에서 생성·v7 예측·피드백 확정·후보 학습 1회·동일 관측 비교를 완료했다. **전체 정확도는 84.70% → 84.60%로 0.10%p 하락했다. v8은 활성화하지 않고 기본 v7을 유지했다.**

## 실험 조건

- 배포 코드: `dadd44f2ecafefc26d681e836f714179480211c7`. 로컬 과거 v1/v2 실험과 별개인 EC2 실측이다.
- 3개 방 × 7일(2026-11-16~22, Asia/Seoul) × 24시간, **300초 간격**, 생성 seed `20261119`, exceptions + stress, fault_probability `0.03`, 오류 지속 30~180분. 이는 시뮬레이터 가정이며 실제 고장률이 아니다.
- `fit_only`: 실제 fit 25,169행만 보정에 사용했다. 생성 예정 6,048행 중 결측·비정상·수집 중단 214행을 제외해 5,834행을 보존했다. 값을 보간하지 않았다.
- 방별 파티션은 room_01=train, room_02=validation, room_03=test. 7일 전체 시간대를 각 방에서 생성했으며 시간으로 3개 조각을 나눈 실험이 아니다.
- 희소 이벤트·센서 오류를 우선 포함한 파티션 균형 추출로 1,500행(각 500행)을 v7으로 예측했다. 자연 발생 빈도에 따른 정확도가 아니다. 정답은 예측값과 독립된 `virtual_people > 0`으로 검증했다.
- train 관측 500행에서 피드백 300행을 확정했다. 최초 균등 150행, 오판 보강 49행, 일반 보충 101행으로 최종 일반 251행 + 오판 보강 49행이다. 중복 제외 0행, 실제 fit 대비 1.192%이며 설정 상한 25%보다 작다. validation/test는 학습에 넣지 않았다.

## 학습

- `Temperature, Humidity, Light, CO2, Hour`, cyclic hour encoding, MLP `6 → 16 → 8 → 1`, tanh. v7과 피처·구조·주요 하이퍼파라미터를 맞췄다.
- Adam, learning_rate=0.01, full batch, 최대 200 epoch, patience=30, seed=42, threshold=0.5, normalize=true, balance_classes=false, dropout=0, weight_decay=0.
- 실제 fit 25,169 + 피드백 300 = **25,469행**, 실제 validation **7,135행**, 날짜 경계의 1시간 간격으로 실제 83행 제외. `shared_room_split=true`, `source-local-time-v1`.
- 초기화부터 후보 1개를 학습했다. 기존 가중치 미세조정이나 강화학습이 아니다. `compare_light=false`, `make_default=false`. v7에 있던 조도 제외 모델의 추가 학습은 수행하지 않았다.
- 후보 **v8**, 57 epoch 실행, best epoch 27, best validation loss 0.2133902907371521, 학습 6.145초. 실제 validation 정확도 92.1654%.

## 동일 고정 관측 1,000행 비교

- v7: 84.70%, precision 30.77%, recall 8.70%, F1 0.135593, FP 27, FN 126.
- v8: 84.60%, precision 32.61%, recall 10.87%, F1 0.163043, FP 31, FN 123.
- 혼동행렬은 실제값 행·예측값 열, 순서는 [비어 있음, 사용 중]이다. v7 `[[835,27],[126,12]]`, v8 `[[831,31],[123,15]]`.
- validation 500행 정확도 81.20% → 81.00%, test 500행 88.20% → 88.20%. 비교 응답의 `complete=true`, `same_observations=true`를 확인했다.
- 항상 비어 있다고 예측해도 이 표본에서는 86.20%다. 두 모델 모두 이에 못 미치며, 사용 중 recall이 매우 낮다. 배포 개선 후보로 채택할 근거가 없다.

### 생활 이벤트별 오류

- weekend_empty_light: 15행, FP 11 → 15, FN 0 → 0.
- night_empty_light: 17행, FP 0 → 0, FN 0 → 0.
- forgotten_lights: 21행, FP 14 → 14, FN 0 → 0.
- night_occupied: 24행, FP 0 → 0, FN 19 → 19.
- occupied_dark: 14행, FP 0 → 0, FN 14 → 14.
- ventilated_occupied: 20행, FP 0 → 0, FN 20 → 20.
- co2_residual: 24행, FP 0 → 0, FN 5 → 5.
- ordinary_use: 71행, FP 0 → 0, FN 71 → 68.

주말 빈 방 점등의 오탐이 11 → 15건으로 악화됐다. 어두운 재실 14건과 환기 중 재실 20건은 두 모델 모두 전부 미탐했다. 이벤트 그룹에는 중복이 있으며, forgotten_lights/co2_residual은 재실과 빈 방이 섞여 있다. “조도 300 이상인 모든 빈 방” 같은 수치 임계값 전체 집계로 해석하지 않는다.

### 센서 오류

- fault:freeze: 17행, 오류 0 → 0건.
- fault:spike: 2행, 오류 1 → 1건.
- fault:drift: 13행, 오류 5 → 5건.
- missing/dropout/nonfinite는 예측 전 제외되어 비교 표본 0행이다. 이 그룹의 정확도를 100%로 해석하지 않는다.

## 실제 test와 해석 한계

- v8의 현재 실제 test 18,479행 정확도는 **89.3284%**, F1 0.797661이다. v7의 과거 87.4285%는 UCI 12,417행에서 산출됐다.
- 평가 파일 SHA256: v7 `b906f685ef1140854169af240458f556b0cecc832e8d03fc9a468aaf2b25a698`, v8 `ff089d006a2b8b8aac85e9e4e9023ebc6d63b2977c7981f644039c56dc094309`. **코호트와 파일이 달라 둘의 차이를 개선폭으로 제시하지 않는다.**
- 기존 v7의 실제 데이터 구성(UCI+Graz A), 후보의 현재 구성(UCI+Graz A/B), 시간 피처 정책·분할도 다르다. 이번 결과는 고정 관측에 대한 모델 동작 비교이며 피드백만의 인과 효과를 분리한 대조 실험이 아니다.
- 개발용 합성 관측 결과로 실제 현장 성능을 주장하지 않는다. 결과 확인 후 추가 튜닝·재학습을 하지 않았다.

## 재확인 식별자

- 생성 ID: `dynamic-sensors-v1-b3b578f51897`, 생성 job: `8365bdcd688e7e077d8f02d21dbe87641f596c73fa61c11e5adae5362a9497fd`.
- v7 실행: `b3fda143109d9eb7c453a001e747c5164307b0a140ff8b0888b95386a2e4d0b2`, 전체 관측 SHA256: `f1a56dfbba4586be450e3dcee3d15998973987d3ddd67f0c24512a80ee8eacd5`.
- 피드백 snapshot: `a472eebd4be1f313a9d920e368492fdd168c19f63c077d7f205b392d6f90e35c`, 선택 행 키 SHA256: `b3b76d43566afeb1dc6c4145c0f895d0b8eb5300c366b70fb8499df08cda284d`.
- 후보 학습 job: `9c44793f54d04c7e874ce436431fb190`, 데이터플랜 SHA256: `b98c47bf7837f11011defc452135ddd190065b3166a88ce16af2410c8703e3fd`.
- 비교 실행: `b16fd4034b4485f15aa3b65de6a3c12de8ad78cf3e3c51c5690df08428d60c7b`, 고정 heldout 관측 SHA256: `2360a0108f43e05114ab43f1ac094227c227bdcf574e11205b20867b5a0c2f7b`.
- 가중치 SHA256: v7 `337d48233fefe1ea5fcc9a5fb51a3d082f786c4aa818441e2eb0bff7c0d76194`, v8 `525594415c70bd9095a023abf16de858d07f6e5df212e1a9ccef349282b0f92a`.
- 로그 run: `20261119T023824_516583Z-ef0acbe82bbc`, training request: `978cd4a1b0184e708613942428521110`. 추론별 식별자 예시와 생성물 원본·audit 해시는 summary.json에 포함했다.
- 전체 관측/예측 정본은 EC2 SQLite, 로그는 `/app/logs`에 보존된다. HTTP API의 오판 응답은 최대 50행이다. SSH 접근 불가로 전체 JSONL/SQLite/원본 CSV를 내려받아 대조하지 않았으며, API 집계·품질검사·해시 검증 결과를 기록했다.
- 실험 종료 시 `/api/health` 정상, 기본 v7, 학습 작업 총 1건 완료. v7 모델 쌍 지문은 전후 `0ded08daed4d294df913bc803208912209dd8a28815076270a6662702a961e79`로 동일하다.

세부 수치·설정·행 수·코호트·상황별 혼동행렬은 [summary.json](summary.json)을 참조한다.
