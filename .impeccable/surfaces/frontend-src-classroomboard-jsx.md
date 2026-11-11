---
version: 1
slug: "frontend-src-classroomboard-jsx"
primary_target: "frontend/src/ClassroomBoard.jsx"
related_targets: ["frontend/src/styles.css","frontend/src/main.jsx"]
---

# 강의실 현황 (모니터링 화면)

Scope: 첫 화면 "강의실 현황", 공통 상단 바, 모델 관리 영역의 색·글꼴·모서리. 모델 관리의 구조와 문구는 유지.
Visitor mode: Operate.

## Brief
- 사용자: 40~50대 시설 관리인과 20~30대 조교. 큰 모니터링 화면이되 구조가 너무 단순해 보이지 않아야 한다.
- 할 일: 강의실별 사용 여부 예측과 센서 값(밝기, CO₂, 온도, 습도, 시간대)을 보고, 불 확인이 필요한 방을 먼저 찾는다.
- 제약: 기존 API, 필터 로직, 팀 렌더 테스트(scripts/check_frontend.mjs)의 문구 계약 유지.

## Direction contract
THESIS: Supabase 대시보드 문법의 강의실 모니터링. 상단 바, 왼쪽 사이드바, 요약 카드, 강의실 카드 격자, 오른쪽 상세 패널의 3단 구조.
OWN-WORLD: 바탕 #171717, 면 #1f1f1f/#242424/#292929, 1px 테두리 #2e2e2e, 모서리 6~8px, 브랜드 녹색 #3ecf8e(버튼 #006239), 경고 #e9a23b, 위험 #f0705a. Inter + Noto Sans KR, 굵기 400~600, 고정폭 숫자. 상태는 아이콘 달린 알약형 배지.
STORY: 요약 카드에서 상태별 개수를 보고, 카드에서 방마다 예측과 센서 4종을 읽고, 오른쪽 패널에서 선택한 방의 사용 확률·근거·추이를 확인한다.
FIRST VIEWPORT: 56px 상단 바(브랜드, 영역 탭). 왼쪽 256px 사이드바(층, 찾기, 전체 센서 평균, 상태 안내). 본문: 제목과 연결 상태, 시연 제어 바, 요약 카드 5개, 강의실 카드 격자. 오른쪽 440px 고정 상세 패널(사용 확률, 센서 값, 순찰 참고 신호, 추이 4종).
FORM: Supabase 대시보드 참고. seed: none (pinned) — "현재 프로젝트의 디자인을 supabase 대시보드 디자인 참고해서 디자인 반영"
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
