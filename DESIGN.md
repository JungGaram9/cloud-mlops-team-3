---
name: 강의실 지킴이
description: 관리실 모니터에 띄워 두고 읽는 강의실 모니터링 대시보드(어두운 테마와 밝은 테마)
colors:
  brand: "#3ecf8e"
  brand-btn: "#006239"
  brand-btn-hover: "#00754a"
  brand-line: "#3ecf8e4d"
  brand-bg: "#12261e"
  brand-tint: "#3ecf8e1a"
  warn: "#e9a23b"
  warn-bg: "#2b2112"
  warn-line: "#5c4318"
  warn-tint: "#e9a23b1f"
  danger: "#f0705a"
  danger-solid: "#e5484d"
  danger-bg: "#2b1614"
  danger-line: "#5c2a22"
  danger-tint: "#e5484d24"
  danger-strong: "#ffb4a6"
  series-blue: "#5fa8ff"
  series-temp: "#f0705a"
  bg: "#171717"
  surface: "#1f1f1f"
  surface-2: "#242424"
  surface-3: "#292929"
  line: "#2e2e2e"
  line-strong: "#3e3e3e"
  line-hover: "#4d4d4d"
  fg: "#ededed"
  fg-light: "#b4b4b4"
  fg-muted: "#898989"
  on-brand: "#fafafa"
  selection-fg: "#0f0f0f"
  light-brand: "#007a52"
  light-brand-btn: "#007a52"
  light-brand-btn-hover: "#006643"
  light-brand-line: "#007a5240"
  light-brand-bg: "#e7f5ee"
  light-brand-tint: "#007a5214"
  light-warn: "#9a5b00"
  light-warn-bg: "#fff6e6"
  light-warn-line: "#efc98a"
  light-warn-tint: "#9a5b0014"
  light-danger: "#c0321f"
  light-danger-solid: "#d93a3a"
  light-danger-bg: "#fdefed"
  light-danger-line: "#f1bdb5"
  light-danger-tint: "#c0321f14"
  light-danger-strong: "#8f2416"
  light-series-blue: "#2563c9"
  light-series-temp: "#c0321f"
  light-bg: "#f8f8f8"
  light-surface: "#ffffff"
  light-surface-2: "#f3f3f3"
  light-surface-3: "#ebebeb"
  light-line: "#e0e0e0"
  light-line-strong: "#cdcdcd"
  light-line-hover: "#a8a8a8"
  light-fg: "#171717"
  light-fg-light: "#474747"
  light-fg-muted: "#636363"
  light-on-brand: "#ffffff"
  light-selection-fg: "#ffffff"
typography:
  display:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "34px"
    fontWeight: 500
    lineHeight: 1
    fontFeature: "tnum"
  stat-number:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "34px"
    fontWeight: 500
    lineHeight: 1.1
    fontFeature: "tnum"
  headline:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "26px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  room-name:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "24px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.5
  body:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: 1.5
  caption:
    fontFamily: "Inter, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  mono:
    fontFamily: "'Source Code Pro', ui-monospace, monospace"
    fontWeight: 400
rounded:
  control: "6px"
  panel: "8px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "20px"
  2xl: "24px"
  page: "32px"
components:
  button-primary:
    backgroundColor: "{colors.brand-btn}"
    textColor: "{colors.on-brand}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "36px"
  button-primary-hover:
    backgroundColor: "{colors.brand-btn-hover}"
  button-secondary:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.fg}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "36px"
  button-secondary-hover:
    backgroundColor: "{colors.surface-3}"
  theme-switch:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
    padding: "3px"
  theme-switch-option:
    textColor: "{colors.fg-muted}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "32px"
  theme-switch-option-pressed:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.fg}"
  input-search:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.fg}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "0 10px"
  nav-tab:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.fg-muted}"
    padding: "0 12px"
    height: "56px"
  nav-tab-active:
    textColor: "{colors.fg}"
  side-item:
    textColor: "{colors.fg-light}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
  side-item-hover:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.fg}"
  side-item-selected:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.fg}"
  badge-status:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.fg-light}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "3px 11px"
  badge-status-occupied:
    backgroundColor: "{colors.brand-tint}"
    textColor: "{colors.brand}"
  badge-status-light-check:
    backgroundColor: "{colors.warn-tint}"
    textColor: "{colors.warn}"
  badge-status-attention:
    backgroundColor: "{colors.danger-tint}"
    textColor: "{colors.danger}"
  badge-status-waiting:
    textColor: "{colors.danger}"
  live-badge:
    backgroundColor: "{colors.brand-tint}"
    textColor: "{colors.brand}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "6px 12px"
  sim-inline:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
    padding: "4px 4px 4px 12px"
  stat-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.fg}"
    rounded: "{rounded.panel}"
    padding: "16px"
  stat-card-light-check:
    backgroundColor: "{colors.warn-bg}"
    textColor: "{colors.warn}"
  stat-card-attention:
    backgroundColor: "{colors.danger-bg}"
    textColor: "{colors.danger}"
  room-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.fg}"
    rounded: "{rounded.panel}"
    padding: "16px"
  room-card-light-check:
    backgroundColor: "{colors.warn-bg}"
  detail-panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.fg}"
    rounded: "{rounded.panel}"
    padding: "20px"
    width: "420px"
  detail-reason-band-light-check:
    backgroundColor: "{colors.warn-bg}"
    padding: "16px 20px"
  detail-reason-band-attention:
    backgroundColor: "{colors.danger-bg}"
    padding: "16px 20px"
  inset-well:
    backgroundColor: "{colors.bg}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
---

# Design System: 강의실 지킴이

## Overview

**Creative North Star: "밤 근무 관제판"**

Supabase 대시보드를 참고한 대시보드 체계이고, 어두운 테마와 밝은 테마 두 가지로 나간다. 어두운 테마가 기본 정의(`:root`)이고, 밝은 테마는 `:root[data-theme="light"]`가 모든 색 토큰을 같은 이름으로 덮어쓴다. 페이지가 그려지기 전에 `index.html`의 인라인 스크립트가 테마를 정한다. 저장된 선택(`localStorage`의 `classroom-theme`)이 있으면 그것을, 없으면 운영체제의 `prefers-color-scheme`을 따른다. 어느 테마든 면은 한 단계씩만 밝기가 달라지고, 모든 구획은 1px 선으로 나뉜다. 색은 상태를 말할 때만 쓴다. 녹색은 사용 중, 주황은 불 확인 필요, 붉은색은 점검 필요다.

40~50대 관리인이 1~2m 떨어져서 읽고, 조교는 같은 화면에서 센서 값과 예측 근거까지 본다. 그래서 글자와 숫자는 일반 대시보드보다 크고(본문 15px, 방 이름 24px, 요약 숫자 34px, 사용 확률 34px), 구조는 상단 바, 왼쪽 사이드바, 두 묶음의 요약 카드, 카드 격자, 오른쪽 상세 패널로 촘촘하게 짠다.

모니터링 화면("강의실 현황")이 이 체계의 기준이고, 모델 관리 영역은 같은 색 토큰과 모서리 값을 따르되 자기 사이드바와 상단 줄을 유지한다. 두 영역 모두 색은 토큰으로만 참조하므로 테마 전환이 화면 전체에 한 번에 적용된다.

**Key Characteristics:**
- 어두운 테마와 밝은 테마. 같은 토큰 이름을 테마별 값으로 덮어쓴다.
- 바탕 위에 면 3단계, 구획은 1px 선.
- 색은 상태 전용. 녹색·주황·붉은색이 각각 한 가지 뜻만 가진다. 선택은 색이 아니라 글자색 고리다.
- 그림자 없음. 선택은 1px 고리로만 표시한다(떠 있는 시연 설정 팝오버만 예외).
- 모서리는 조작 요소 6px, 패널 8px, 배지 999px 세 가지.
- 숫자는 모두 고정폭(`tabular-nums`)이라 1초 갱신에도 자리가 흔들리지 않는다.
- 상태는 색, 글자, 아이콘(그리고 테두리 모양) 세 가지 이상으로 함께 구분된다.

## Colors

무채색 회색 계단 위에 녹색 하나를 강조로 쓰고, 주황과 붉은색은 확인이 필요한 상태에만 쓴다. 각 토큰은 어두운 값(접두어 없음)과 밝은 값(`light-` 접두어)을 짝으로 가진다. 밝은 테마의 상태색은 흰 바탕 위에서 글자로 읽히도록 훨씬 짙다.

### Primary
- **신호 녹색** (`brand` / `light-brand`): 사용 중 예상 배지와 숫자, 연결 표시 알약, 현재 탭 밑줄, 포커스 외곽선, 사용 확률 막대, CO₂ 추이선, 글자 선택 바탕.
- **버튼 녹색** (`brand-btn`, hover `brand-btn-hover`): 주 버튼의 채움. `on-brand` 글자를 올린다. 어두운 테마에서는 신호 녹색보다 훨씬 어둡고, 밝은 테마에서는 신호 녹색과 같은 값이다.
- **녹색 선** (`brand-line`): 녹색 배지와 주 버튼의 테두리, 찾기 입력 포커스 고리.
- **녹색 옅은 칠** (`brand-tint`): 사용 중 배지, 연결 표시 알약, 성공 배지의 바탕.
- **녹색 바닥** (`brand-bg`): 모델 관리 영역의 브랜드 아이콘 바탕.

### Secondary
- **확인 주황** (`warn`, 바닥 `warn-bg`, 선 `warn-line`, 옅은 칠 `warn-tint`): "불 확인 필요" 한 가지 상태 전용. 카드 전체 바탕과 테두리, 배지, 요약 카드 숫자, 상세 패널의 사유 띠, 밝기 추이선. 연결이 끊겼을 때의 연결 표시 알약도 이 색이다.

### Tertiary
- **경보 붉은색** (`danger`, 진한 점 `danger-solid`, 바닥 `danger-bg`, 선 `danger-line`, 옅은 칠 `danger-tint`, 진한 글자 `danger-strong`): "점검 필요" 묶음 전체(수신 지연, 수신 대기, 판별 불가, 측정 시각 이상, 서버 연결 끊김)와 오류 알림. 범례 점은 `danger-solid`, 오류 알림 글자는 `danger-strong`, 다시 시작 확인 단계의 바탕은 `danger-bg`.
- **계열 색** (`series-blue`, `series-temp`): 차트 계열 전용. 파랑은 습도 추이선·검증 손실 곡선·학습 진행 중 글자, `series-temp`는 온도 추이선. 화면 요소의 강조색으로는 쓰지 않는다.

### Neutral
- **바탕** (`bg`): 페이지, 상단 바, 사이드바, 카드 안에 파인 센서 칸.
- **면 1** (`surface`): 카드, 패널, 찾기 입력, 테마 스위치와 시연 줄의 틀.
- **면 2** (`surface-2`): 보조 버튼, 기본 배지, 사이드바 항목 hover.
- **면 3** (`surface-3`): 선택된 사이드바 항목, 눌린 테마 버튼, 보조 버튼 hover, 확률 막대의 빈 부분.
- **선** (`line`): 모든 기본 테두리와 구분선.
- **진한 선** (`line-strong`): 조작 요소 테두리, 스크롤바, 추이 기준선.
- **hover 선** (`line-hover`): 카드와 버튼 hover 테두리.
- **글자** (`fg`): 본문과 제목. 선택 고리(`--select`)도 이 값이다.
- **보조 글자** (`fg-light`): 카드 안 설명, 사이드바 항목과 개수, 라벨.
- **흐린 글자** (`fg-muted`): 단위, 시각, 안내문. `bg`와 `surface` 위에서만 쓴다.
- **버튼 위 글자** (`on-brand`), **선택 글자** (`selection-fg`): 녹색 채움 위의 글자.

`--accent`, `--accent-dark`, `--soft`, `--muted`, `--border`는 모델 관리 영역의 옛 이름이며 각각 `brand-btn`, `brand-btn-hover`, `surface-3`, `fg-muted`, `line`을 가리킨다. 새 코드는 원래 토큰 이름을 쓴다.

### Named Rules
**상태색 전용 규칙.** 녹색·주황·붉은색은 상태를 말할 때만 쓴다. 장식, 구분, 강조, 선택 표시 목적으로 색을 칠하지 않는다. 녹색은 "사용 중", 연결 표시 알약, 주 버튼에만 남긴다.

**옅은 바닥 규칙.** 상태색을 넓은 면에 채울 때는 원색이 아니라 전용 바닥색(`warn-bg`, `danger-bg`)이나 옅은 칠(`brand-tint`, `warn-tint`, `danger-tint`)을 쓰고, 같은 계열의 선(`brand-line`, `warn-line`, `danger-line`)으로 테두리를 두른다.

**토큰 전용 규칙.** 색은 언제나 `var(--토큰)`으로 참조한다. 값을 직접 적으면 밝은 테마에서 그 자리만 어두운 값으로 남는다.

## Typography

**Display Font:** Inter (한글은 Noto Sans KR, 이어서 시스템 산세리프)
**Body Font:** Inter + Noto Sans KR
**Label/Mono Font:** Source Code Pro (측정 식별값 전용)

**Character:** 한 가지 산세리프로 전부 쓴다. 위계는 서체가 아니라 크기와 회색 단계로 만든다. 숫자는 크고 중간 굵기(500), 이름과 제목은 600, 나머지는 400이다. 글꼴은 `index.html`의 `<link>`로 Inter 400/500/600, Noto Sans KR 400/500/600, Source Code Pro 400만 불러온다.

### Hierarchy
- **Display** (500, 34px, 1): 상세 패널의 사용 확률.
- **Stat number** (500, 34px, 1.1): 요약 카드의 개수. 1500px 이하에서 30px, 560px 이하에서 22px.
- **Headline** (600, 26px, 1.3, -0.02em): 화면 제목. 560px 이하에서 23px.
- **Room name** (600, 24px, 1.25, -0.02em): 카드의 강의실 이름. 상세 패널 제목은 22px.
- **Title** (600, 18px): 구역 제목(강의실), 상세 패널의 사유 문장.
- **Body** (400, 15px, 1.55): 본문, 카드 설명, 사이드바 항목, 입력값. 안내문은 최대 90ch. 점검·불 확인 카드의 할 일 문장은 16px/600.
- **Label** (500, 14px): 배지, 버튼, 묶음 제목, 칸 제목.
- **Caption** (400, 13px, 1.5): 단위, 추이 축, 범례 설명. 모니터링 화면의 최소 크기.
- **Mono** (400, Source Code Pro): 접힌 측정 기록 안의 식별값.

카드 안 센서 값은 19px, 상세 패널 센서 값은 26px, 판단 근거 값은 20px, 시연 시계는 20px로 모두 500 굵기 고정폭 숫자다.

### Named Rules
**고정폭 숫자 규칙.** 갱신되는 숫자(개수, 확률, 센서 값, 시각)는 모두 `font-variant-numeric: tabular-nums`를 쓴다.

**굵기 세 단계 규칙.** 400, 500, 600만 쓴다. 숫자는 500, 이름과 제목은 600.

**한글 줄바꿈 규칙.** 모니터링 화면 전체에 `word-break: keep-all`을 걸어 단어 중간에서 줄이 바뀌지 않게 한다.

**시각 표기 규칙.** 실제 수신 시각은 한국 시간 24시간제 `HH:MM`(초 없음)으로 쓴다. 시연 시각은 가상 하루의 시계이므로 시간대를 붙이지 않는다.

## Layout

화면은 세 겹이다. 위에 56px 고정 상단 바(브랜드, 영역 탭, 오른쪽 한 줄 설명, 테마 스위치), 왼쪽에 256px 사이드바(층, 찾기, 상태 안내, 전체 센서 평균), 나머지가 본문이다. 문서 순서는 본문이 먼저, 사이드바가 나중이고 격자가 사이드바를 왼쪽 열에 놓는다.

본문은 위에서부터 제목 줄(제목과 설명, 오른쪽에 연결 표시와 시연 줄), 두 묶음의 요약 카드, 그 아래 카드 격자와 420px 상세 패널이 나란히 놓인다. 요약 카드는 "가 볼 방"(불 확인 필요, 점검 필요)과 "사용 현황"(전체, 사용 중 예상, 비어 있음 예상) 두 묶음이 2:3 비율로 나뉘고, 묶음마다 14px 제목이 붙는다.

- 본문 여백은 24px 32px, 카드 사이 간격은 12px, 묶음과 구역 사이는 16~24px, 카드 안쪽 여백은 16px, 상세 패널은 20px.
- 강의실 카드 격자는 `repeat(auto-fill, minmax(290px, 1fr))`이라 폭에 따라 열 수가 스스로 정해진다.
- 사이드바만 스크롤해도 제자리에 머문다(sticky). 상세 패널은 고정되지 않고 안쪽 스크롤도 없이 본문과 함께 흐른다.
- 카드 안 센서 값 4종은 1px 간격의 2×2 칸으로 묶는다. 칸 사이 선은 간격 사이로 비치는 `line` 색이다.

반응형 단계:
- **1500px 이하**: 상세 패널 380px, 요약 숫자 30px.
- **1280px 이하**: 상세 패널이 격자 아래로 내려가고 "강의실 목록으로" 버튼이 나타난다. 두 요약 묶음은 위아래로 쌓인다.
- **900px 이하**: 사이드바가 위로 올라가 층 선택과 찾기, 상태 안내만 남는다(센서 평균과 하단 안내는 숨김). 상단 바 설명을 숨기고 테마 스위치가 오른쪽 끝으로 간다. 본문 여백 20px 16px.
- **560px 이하**: 요약 카드는 라벨 왼쪽·숫자 오른쪽의 가로 줄이 되어 2열(전체는 한 줄 전체), 강의실 카드 1열, 추이 1열. 시연 줄이 폭 전체로 늘어나 줄바꿈되고, 테마 스위치는 아이콘만 보인다.

## Elevation & Depth

그림자를 쓰지 않는다. 깊이는 면의 밝기 단계(`bg` → `surface` → `surface-2` → `surface-3`)와 1px 선으로만 만든다. 카드 안 센서 칸은 반대로 `bg`로 한 단계 내려가 파인 것처럼 보인다. 상세 패널 안에는 카드를 겹치지 않고, 구획을 1px 위 구분선으로만 나눈다.

### Shadow Vocabulary
- **선택 고리** (`box-shadow: 0 0 0 1px var(--select)`, `--select: var(--fg)`): 선택된 요약 카드와 강의실 카드. 테두리 색도 같이 바꿔 2px 두께처럼 보인다. 불 확인 카드·요약 카드는 `--warn`, 점검 필요 카드·요약 카드는 `--danger` 고리를 쓴다.
- **입력 포커스 고리** (`box-shadow: 0 0 0 2px var(--brand-line)`): 찾기 입력에 포커스가 들어갔을 때.
- **탭 밑줄** (`box-shadow: inset 0 -2px 0 var(--brand)`): 상단 바의 현재 탭.
- **팝오버 그림자** (`box-shadow: 0 8px 24px` 25% 검정): 화면 위에 떠서 다른 요소를 가리는 시연 설정 팝오버 한 곳.

### Named Rules
**고리만 허용 규칙.** `box-shadow`는 번짐 없는 고리와 밑줄로만 쓴다. 번지는 그림자는 실제로 위에 떠서 내용을 가리는 팝오버에만 허용하고, 카드와 패널은 띄우지 않는다. 흐림 효과는 없다.

**선택은 글자색 규칙.** 일반 선택 고리는 상태색이 아니라 글자색(`--select` = `fg`)이다. 녹색 고리는 "사용 중"과 헷갈리므로 쓰지 않는다.

## Shapes

모서리는 세 가지뿐이다. 버튼·입력·사이드바 항목·테마 버튼 같은 조작 요소는 6px, 카드·패널·테마 스위치 틀·시연 줄은 8px, 배지와 막대는 999px 알약형이다. 패널 안에 들어간 작은 칸은 6px로 한 단계 줄인다.

테두리는 기본 1px 실선이다. 점선은 "아직 값이 없음"을 뜻한다. 빈 상태 안내, 수신 대기 카드와 배지, 범례의 "확인 중" 점이 점선이다. 상태 점은 10px 원이고, "비어 있음"은 채우지 않은 1.5px 테두리 원, "확인 중"은 1.5px 점선 원이라 색 없이도 구분된다. 아이콘은 24px 격자에 그린 선 아이콘(선 굵기 1.75, 둥근 끝)이고 기본 18px, 테마 스위치에서는 16px로 쓴다.

## Components

### Buttons
- **Shape:** 6px 모서리, 최소 높이 36px, 좌우 여백 12px, 14px/500 글자, 아이콘과 7px 간격.
- **Primary:** `brand-btn` 채움에 `on-brand` 글자, `brand-line` 테두리. 화면마다 주 동작 하나(시연 재생·일시정지)에만 쓴다.
- **Secondary:** `surface-2` 채움, `line-strong` 테두리, `fg` 글자. 셀렉트와 "강의실 목록으로" 버튼도 같은 모양이다.
- **Danger:** `danger-solid` 채움. 시연을 처음부터 다시 시작하는 확인 단계의 실행 버튼에만 쓴다.
- **Hover / Focus:** hover에서 면이 한 단계 밝아지고 테두리가 `line-hover`로 진해진다(주 버튼은 `brand-btn-hover`와 `brand` 테두리). 전환 .15s. 포커스는 2px `brand` 외곽선, 2px 띄움.
- **Disabled:** 불투명도 .45, 금지 커서.

### Chips
- **Style:** 상태 배지는 알약형(999px), 3px 11px 여백, 14px/500, 앞에 선 아이콘. 기본은 `surface-2` 바탕에 `line-strong` 테두리, `fg-light` 글자.
- **State:** 사용 중 예상은 `brand-tint`·`brand-line`·`brand`, 불 확인 필요는 `warn-tint`·`warn-line`·`warn`, 점검 필요는 `danger-tint`·`danger-line`·`danger`. 수신 대기는 바탕 없이 `danger-line` 점선 테두리와 `danger` 글자다.
- **점검 필요의 사유:** 배지 글자는 묶음 이름이 아니라 사유 하나다. 수신 지연, 수신 대기, 판별 불가, 측정 시각 이상, 서버 연결 끊김. 필터와 요약 카드에서는 이들을 "점검 필요" 하나로 묶는다.
- **확인 중:** 밝기·CO₂ 조건이 이어지는지 지켜보는 상태에만 쓰는 말이다. 연결이나 수신 문제에는 쓰지 않는다.
- **연결 표시:** 제목 옆 알약. 8px 점과 글자가 같은 색(정상 `brand`와 `brand-tint`, 끊김 `warn`과 `warn-tint`).

### Cards / Containers
- **Corner Style:** 8px.
- **Background:** `surface`. hover에서는 테두리만 `line-hover`로 진해진다.
- **Shadow Strategy:** 없음. 선택 시 1px 고리(Elevation 참고).
- **Border:** 1px `line`. 불 확인 카드는 `warn-line` 테두리에 `warn-bg` 바탕, 점검 필요 카드는 `danger-line` 테두리, 수신 대기 카드는 `line-strong` 점선 테두리.
- **Internal Padding:** 16px, 내부 간격 10px.
- **요약 카드:** 카드 전체가 필터 버튼이다. 라벨(아이콘 + 15px) 위, 34px 숫자 아래. 개수가 0이면 상태색을 빼고 무채색으로 둔다. 불 확인 필요는 주황 바닥, 점검 필요는 붉은 바닥.
- **강의실 카드:** 위에서부터 층·센서 종류, 24px 이름과 상태 배지, 할 일 한 줄, 예측 한 줄, 센서 2×2 칸, 마지막 수신 시각.

### Inputs / Fields
- **Style:** 찾기 입력은 6px 모서리, `surface` 바탕, `line-strong` 테두리, 앞에 돋보기 아이콘, 15px 글자.
- **Focus:** 테두리가 `brand`로 바뀌고 2px `brand-line` 고리가 생긴다.

### Navigation
- **상단 바:** 56px, `bg` 바탕, 아래 1px `line`, 스크롤해도 위에 고정. 탭은 15px/500 `fg-muted`, hover와 현재 탭은 `fg`, 현재 탭은 아래 2px 녹색 밑줄. 900px 이하에서 오른쪽 설명을 숨기고, 560px 이하에서 브랜드 아이콘을 숨긴다.
- **사이드바 항목:** 6px 모서리, 8px 10px 여백, 15px `fg-light`, 오른쪽에 고정폭 개수. hover `surface-2`, 선택 `surface-3`과 `fg` 글자, 500 굵기. 900px 이하에서는 테두리 있는 가로 버튼 줄이 된다.
- **상태 안내:** 사이드바의 접을 수 있는 묶음(`<details>`). 넓은 화면에서는 펼친 채로 시작한다.

### 테마 스위치
상단 바 오른쪽 끝의 "라이트 | 다크" 두 칸 분할 버튼. 틀은 8px 모서리, `surface` 바탕, 1px `line`, 3px 안쪽 여백. 칸은 6px 모서리, 32px 높이, 16px 아이콘과 14px/500 글자, 기본 `fg-muted`. 눌린 칸은 `surface-3` 바탕에 `fg` 글자이고 `aria-pressed`로 상태를 알린다. 560px 이하에서는 글자를 시각적으로 숨기고 아이콘만 남긴다(글자는 보조기기용으로 유지).

### 시연 줄
제목 줄 오른쪽에 붙는 한 줄짜리 시연 제어. 8px 모서리 `surface` 틀 안에 "시연" 꼬리표, 시연 시계(20px 고정폭 숫자, 시간대 없음), 재생·일시정지 주 버튼, "시연 설정" 펼침이 들어간다. 펼침은 아래로 떠오르는 팝오버(8px 모서리, `line-strong` 테두리)이고 속도 선택과 처음부터 다시 시작이 있다. 다시 시작은 바로 실행하지 않고 `danger-bg` 확인 칸에서 한 번 더 묻는다.

### 상세 패널
420px 폭, 20px 여백, `surface` 바탕의 한 장짜리 패널. 안에 카드를 겹치지 않고, 머리말 아래 구획(사유, 사용 확률, 센서 값, 추이, 측정 기록)을 1px `line` 위 구분선으로만 나눈다. 불 확인 사유와 점검 사유는 패널 좌우 여백까지 꽉 채운 띠(`warn-bg`/`danger-bg`, 위아래 같은 계열 선)로 칠한다. 기준과 한계 설명은 "판단 기준과 한계" 펼침 안에 접어 둔다.

### 사용 확률 칸
34px 숫자와 8px 알약형 막대. 사용 중으로 예상되면 숫자와 막대가 녹색, 아니면 무채색(`fg`, `fg-muted`)이다. 막대는 .3s로 늘어나고 줄어든다.

### 센서 추이
2×2 칸에 높이 72px 선 그래프. 선 굵기 2, 둥근 끝, 기준선 1px `line-strong`. 밝기 `warn`, CO₂ `brand`, 온도 `series-temp`, 습도 `series-blue`로 센서마다 색이 고정된다.

## Do's and Don'ts

### Do:
- **Do** 색 값은 `:root`의 토큰(`--bg`, `--surface`, `--line`, `--brand` 등)으로 참조하고, 새 토큰을 만들면 `:root[data-theme="light"]`에도 같은 이름으로 값을 둔다.
- **Do** 두 테마에서 모두 확인한다. 밝은 테마의 상태색 글자는 흰 바탕에서 읽혀야 한다.
- **Do** 상태는 색, 글자, 아이콘 세 가지로 함께 표시한다. 범례 점도 채움, 빈 원, 점선 원으로 모양을 달리한다.
- **Do** 갱신되는 숫자에는 `tabular-nums`를 쓴다.
- **Do** 모서리는 6px(조작 요소), 8px(패널), 999px(배지) 중에서 고른다.
- **Do** 선택은 테두리 색과 1px `--select` 고리로, hover는 테두리를 `line-hover`로 진하게 하는 것으로 표시한다.
- **Do** 모니터링 화면의 글자는 본문 15px, 보조 13px 이상으로 유지한다.
- **Do** 흐린 글자(`fg-muted`)는 `bg`와 `surface` 위에서 쓰고, 더 밝은 면 위에서는 `fg-light`를 쓴다.
- **Do** 아이콘은 인라인 SVG 선 아이콘(24px 격자, 선 굵기 1.75)으로 그린다.

### Don't:
- **Don't** 색 값을 직접 적지 않는다. 직접 적은 값은 테마가 바뀌어도 따라가지 않는다.
- **Don't** 번지는 그림자나 흐림 효과로 카드와 패널의 깊이를 만들지 않는다. `box-shadow`는 고리와 밑줄, 떠 있는 팝오버만.
- **Don't** 녹색·주황·붉은색을 장식, 구분, 선택 표시에 쓰지 않는다.
- **Don't** 상태색 원색을 넓은 면에 채우지 않는다. 전용 바닥색이나 옅은 칠을 쓴다.
- **Don't** 상세 패널 안에 카드를 다시 겹치지 않는다. 구획은 1px 구분선으로 나눈다.
- **Don't** "확인 중"을 연결·수신 문제에 쓰지 않는다. 그 경우는 점검 필요의 사유 이름을 쓴다.
- **Don't** 굵기 700 이상을 쓰지 않는다.
- **Don't** Source Code Pro를 측정 식별값 외의 글자에 쓰지 않는다.
- **Don't** Supabase의 이름이나 로고를 화면에 쓰지 않는다. 참고한 것은 문법이지 브랜드가 아니다.
