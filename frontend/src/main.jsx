import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

// 화면·예시 데이터·입력 설명을 한곳에서 관리하고 디자인은 styles.css에 둔다.
const PRESETS = [
  { id: 'bright', title: '밝은 실내', description: '수요일 오후 2시 · 조명이 켜진 실내', values: { Temperature: 23.1, Humidity: 27.2, Light: 430, CO2: 720, Hour: 14, DayOfWeek: 2 } },
  { id: 'dark', title: '불 꺼진 실내', description: '일요일 새벽 2시 · 어두운 실내', values: { Temperature: 20.4, Humidity: 30, Light: 0, CO2: 450, Hour: 2, DayOfWeek: 6 } },
  { id: 'co2', title: 'CO₂가 높은 실내', description: '금요일 오후 4시 · 밝고 높은 CO₂', values: { Temperature: 23, Humidity: 35, Light: 450, CO2: 1500, Hour: 16, DayOfWeek: 4 } },
];
const DAYS = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'];

const SENSORS = [
  { key: 'Temperature', label: '기온', unit: '°C', icon: 'temperature', min: 15, max: 35, step: 0.1,
    inputMin: -50, inputMax: 60, range: '입력 −50~60°C · 슬라이더 15~35°C',
    references: ['18°C 서늘한 실내', '23°C 보통 실내', '28°C 따뜻한 실내'],
    hint: '온도계로 측정한 공기 온도입니다. 개인마다 느끼는 정도는 다를 수 있어요.' },
  { key: 'Humidity', label: '습도', unit: '%', icon: 'humidity', min: 0, max: 100, step: 0.1,
    inputMin: 0, inputMax: 100, range: '입력 0~100%',
    references: ['30% 건조한 편', '45% 보통 수준', '70% 습한 편'],
    hint: '공기가 얼마나 건조하거나 습한지 나타내는 상대습도입니다.' },
  { key: 'Light', label: '조도', unit: 'lux', icon: 'light', min: 0, max: 1000, step: 1,
    inputMin: 0, inputMax: 1e6, range: '허용 0~100만 lux · 슬라이더 0~1,000',
    references: ['0 lux 거의 어두움', '150 lux 은은한 밝기', '500 lux 밝은 실내'],
    hint: '센서 위치에서 측정한 밝기입니다. 조명을 켜도 사람이 없을 수 있어요.' },
  { key: 'CO2', label: 'CO₂', unit: 'ppm', icon: 'air', min: 350, max: 2500, step: 1,
    inputMin: 1, inputMax: 1e6, range: '허용 0 초과~100만 ppm · 슬라이더 350~2,500',
    references: ['450 ppm 낮은 농도', '900 ppm 중간 농도', '1,500 ppm 높은 농도'],
    hint: '사람의 호흡과 환기에 영향을 받습니다. 직접 체감하기보다 센서로 확인하세요.' },
];

function Icon({ name, className = '' }) {
  const paths = {
    room: <><path d="M5 20V4h14v16M2 20h20M9 9h6M9 13h6" /><path d="M9 20v-3h6v3" /></>,
    chart: <><path d="M4 4v16h16M8 16v-4M12 16V7M16 16v-6" /></>,
    temperature: <><path d="M10 14.5V5a2 2 0 0 1 4 0v9.5a4 4 0 1 1-4 0Z" /><path d="M12 8v9" /></>,
    humidity: <path d="M12 3s-6 7-6 11a6 6 0 0 0 12 0c0-4-6-11-6-11Z" />,
    light: <><path d="M9 18h6M10 21h4M8 12a5 5 0 1 1 8 0c-1.5 1-1.5 2-1.5 3h-5c0-1 0-2-1.5-3Z" /><path d="M12 1v1M2 8h2M20 8h2M4 2l2 2M20 2l-2 2" /></>,
    air: <><path d="M3 8h12a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h6" /></>,
    arrow: <path d="m8 5 7 7-7 7M3 12h12" />,
    check: <path d="m5 12 4 4L19 6" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7h.01" /></>,
    reset: <><path d="M4 9a8 8 0 1 1 0 6M4 3v6h6" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 6v6l4 2" /></>,
  };
  return <svg className={`icon ${className}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.room}</svg>;
}

const percent = (value, digits = 2) => Number.isFinite(value) ? (value * 100).toFixed(digits) : '—';
const featureLabel = (name) => ({ Hour: '시간', DayOfWeek: '요일' })[name] || SENSORS.find((sensor) => sensor.key === name)?.label || name;
const featureNames = (features) => features?.map(featureLabel).join(' · ') || '피처 확인 중';

function reading(key, value) {
  if (value === '') return '숫자를 입력해 주세요';
  if (key === 'Temperature') return value < 20 ? '서늘한 편' : value < 25 ? '보통 실내 온도' : '따뜻한 편';
  if (key === 'Humidity') return value < 35 ? '건조한 편' : value <= 60 ? '보통 습도' : '습한 편';
  if (key === 'Light') return value < 10 ? '거의 어두움' : value < 200 ? '은은한 밝기' : value < 600 ? '밝은 실내' : '매우 밝은 편';
  return value < 600 ? '비교적 낮은 농도' : value < 1000 ? '중간 농도' : value < 1500 ? '비교적 높은 농도' : '높은 농도';
}

async function api(path, body, signal) {
  const response = await fetch(`/api${path}`, {
    method: body ? 'POST' : 'GET', signal,
    ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || '서버에 연결하지 못했습니다. API 실행 상태를 확인해 주세요.');
  }
  return response.json();
}

function Sensor({ sensor, value, excluded, onChange }) {
  return <fieldset className={`sensor ${excluded ? 'sensor--excluded' : ''}`}>
    <legend className="sr-only">{sensor.label} 입력</legend>
    <div className="sensor-heading">
      <div className="sensor-name"><span className="sensor-icon"><Icon name={sensor.icon} /></span><div><label htmlFor={sensor.key}>{sensor.label}</label><small>{sensor.range}</small></div></div>
      <div className="number-control"><input id={sensor.key} type="number" required aria-label={`${sensor.label} 숫자 입력`} min={sensor.inputMin} max={sensor.inputMax} step="any" value={value} disabled={excluded} onChange={(event) => onChange(event.target.value === '' ? '' : Number(event.target.value))} /><span>{sensor.unit}</span></div>
    </div>
    <div className="sensor-reading"><span>{reading(sensor.key, value)}</span>{excluded && <b>이 모델에서는 제외</b>}</div>
    <input className="sensor-slider" type="range" aria-label={`${sensor.label} 슬라이더`} min={sensor.min} max={sensor.max} step={sensor.step} value={Math.min(sensor.max, Math.max(sensor.min, Number(value)))} disabled={excluded} onChange={(event) => onChange(Number(event.target.value))} />
    <div className="sensor-references">{sensor.references.map((reference) => <span key={reference}>{reference}</span>)}</div>
    <p className="sensor-hint">{sensor.hint}</p>
  </fieldset>;
}

function AccuracyCard({ title, description, report, selected, onClick }) {
  const accuracy = report?.model_metrics.accuracy;
  return <button className={`accuracy-card ${selected ? 'accuracy-card--selected' : ''}`} onClick={onClick} aria-pressed={selected}>
    <span className="accuracy-card-title"><span>{title}</span><span className="selection-dot">{selected && <Icon name="check" />}</span></span>
    <span className="accuracy-number">{percent(accuracy)}<small>%</small></span>
    <span className="accuracy-description">{description}</span>
    <progress value={(accuracy || 0) * 100} max="100" aria-label={`${title} 평가 정확도`} />
    <span className="accuracy-bottom"><span>평가 데이터 기준</span><b>{report ? (report.quality_gate.passed ? '85% 기준 충족' : '85% 기준 미달') : '불러오는 중'}</b></span>
  </button>;
}

function App() {
  const [mode, setMode] = useState('with_light');
  const [page, setPage] = useState('experiment');
  const [versions, setVersions] = useState(null);
  const [selectedVersion, setSelectedVersion] = useState('');
  const [preset, setPreset] = useState(PRESETS[0].id);
  const [sensors, setSensors] = useState({ ...PRESETS[0].values });
  const [comparison, setComparison] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const requestRef = useRef(null);
  const generation = useRef(0);
  const lightEnabled = mode === 'with_light';
  const report = comparison?.modes[mode];
  const prediction = predictions?.[mode];
  const currentVersion = comparison?.model_version || versions?.current_version || '—';
  const versionPair = versions?.versions.find((item) => item.version === selectedVersion);
  const timeEnabled = Boolean(report?.features.includes('Hour'));

  function validInput(values) {
    return SENSORS.every((sensor) => Number.isFinite(values[sensor.key]) && values[sensor.key] >= sensor.inputMin && (sensor.inputMax === undefined || values[sensor.key] <= sensor.inputMax))
      && Number.isInteger(values.Hour) && values.Hour >= 0 && values.Hour <= 23
      && Number.isInteger(values.DayOfWeek) && values.DayOfWeek >= 0 && values.DayOfWeek <= 6;
  }

  async function predict(values) {
    if (!validInput(values) || !comparison) {
      setError('모델 연결과 센서값·시간·요일의 허용 범위를 확인해 주세요.');
      return;
    }
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const current = ++generation.current;
    setLoading(true);
    setError('');
    try {
      const result = await api(`/compare?version=${comparison.model_version}`, values, controller.signal);
      if (current === generation.current) setPredictions(result);
    } catch (failure) {
      if (failure.name !== 'AbortError' && current === generation.current) setError(failure.message);
    } finally {
      if (current === generation.current) setLoading(false);
    }
  }

  async function useModelPair(version, values, openExperiment = true) {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const current = ++generation.current;
    setLoading(true);
    setError('');
    setPredictions(null);
    try {
      const [evaluation, result] = await Promise.all([
        api(`/comparison?version=${version}`, null, controller.signal),
        api(`/compare?version=${version}`, values, controller.signal),
      ]);
      if (current === generation.current) {
        setComparison(evaluation);
        setPredictions(result);
        setSelectedVersion(version);
        if (openExperiment) setPage('experiment');
      }
    } catch (failure) {
      if (failure.name !== 'AbortError' && current === generation.current) setError(failure.message);
    } finally {
      if (current === generation.current) setLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    api('/versions', null, controller.signal).then((result) => {
      setVersions(result);
      setSelectedVersion(result.current_version);
      useModelPair(result.current_version, PRESETS[0].values, false);
    }).catch((failure) => {
      if (failure.name !== 'AbortError') setError(failure.message);
    });
    return () => { controller.abort(); requestRef.current?.abort(); };
  }, []);

  function editSensor(key, value) {
    requestRef.current?.abort();
    generation.current += 1;
    setLoading(false);
    setSensors((previous) => ({ ...previous, [key]: value }));
    setPreset('');
    setPredictions(null);
    setError('');
  }

  function choosePreset(item) {
    setPreset(item.id);
    setSensors({ ...item.values });
    setPredictions(null);
    predict(item.values);
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#top"><span className="brand-mark"><Icon name="room" /></span><span>공간 실험실<small>강의실 센서 프로젝트</small></span></a>
      <div className="sidebar-section-label">실험 워크스페이스</div>
      <nav aria-label="화면 탐색"><button className={`nav-item ${page === 'experiment' ? 'nav-item--active' : ''}`} onClick={() => setPage('experiment')} aria-current={page === 'experiment' ? 'page' : undefined}><Icon name="room" />센서 실험</button><button className={`nav-item ${page === 'versions' ? 'nav-item--active' : ''}`} onClick={() => setPage('versions')} aria-current={page === 'versions' ? 'page' : undefined}><Icon name="chart" />모델 버전 관리</button><a className="nav-item" href="/api/docs" target="_blank" rel="noreferrer"><Icon name="info" />API 사용 안내</a></nav>
      <div className="sidebar-note"><span className="tiny-dot" />PyTorch {currentVersion}<p>같은 공간, 다른 입력.<br />센서가 결과에 미치는 영향을<br />직접 확인해 보세요.</p></div>
      <div className="sidebar-footer">클라우드 MLOps · 3팀</div>
    </aside>

    <main id="top">
      <header className="topbar"><span>강의실 관리 <span className="breadcrumb-separator">/</span> <b>{page === 'versions' ? '모델 버전 관리' : '센서 실험'}</b></span><span className={`connection ${comparison ? 'connection--ready' : ''}`}><span className="tiny-dot" />{comparison ? '모델 연결됨' : '모델 연결 확인 중'}</span></header>
      <div className="workspace">
        {error && <div className="error-banner" role="alert"><Icon name="info" />{error}<button onClick={() => window.location.reload()}>다시 연결</button></div>}
        {page === 'versions' ? <>
          <div className="page-heading"><div><div className="eyebrow">모델 변경이 버전을 만듭니다</div><h1>모델 버전 관리</h1><p>각 모델 버전은 조도 포함·제외 두 가중치를 한 쌍으로 보존합니다.</p></div><span className="version-tag">현재 실험 <b>{currentVersion}</b></span></div>
          <div className="version-overview"><Icon name="room" /><div><b>{currentVersion} 모델 쌍으로 실험하고 있습니다</b><p>모델이 같으면 재학습해도 버전을 재사용합니다. 화면·폴더 변경은 모델 버전을 만들지 않습니다.</p></div><span className="pair-badge">모델 2개 / 쌍</span></div>
          <section className="version-list" aria-label="모델 쌍 버전 목록">{versions?.versions.map((item) => <button key={item.version} className={`version-button ${selectedVersion === item.version ? 'version-button--selected' : ''}`} aria-pressed={selectedVersion === item.version} onClick={() => setSelectedVersion(item.version)}><span className="version-symbol">{item.version}</span><span><b>PyTorch {item.version} 모델 쌍</b><small>{new Date(item.created_at_utc).toLocaleString('ko-KR')} · 평가 {item.test_rows.toLocaleString('ko-KR')}건{item.same_model_as && ` · ${item.same_model_as}과 동일한 모델의 기존 이력`}</small></span><span className={`version-status ${item.version === currentVersion ? 'version-status--current' : ''}`}>{item.version === currentVersion ? '실험 중' : item.current ? '기본 모델' : '저장됨'}</span></button>) || <p>버전 목록을 불러오는 중입니다.</p>}</section>
          {versionPair && <section className="version-details" aria-labelledby="version-heading"><div className="section-heading"><div><div className="section-eyebrow">선택한 모델 쌍</div><h2 id="version-heading">{versionPair.version} · 조도 포함 + 조도 제외</h2></div><button className="use-version-button" disabled={!versionPair.ready || loading || !validInput(sensors)} onClick={() => useModelPair(versionPair.version, sensors)}>이 모델 쌍으로 실험하기<Icon name="arrow" /></button></div><div className="version-model-grid">{versionPair.models.map((item) => <article className="version-model" key={item.mode}><div className="section-eyebrow">{item.mode === 'with_light' ? '모델 01' : '모델 02'}</div><h3>{item.mode === 'with_light' ? '조도 피처 포함' : '조도 피처 제외'}</h3><p className="version-features">{featureNames(item.features)}</p><dl><div><dt>평가 정확도</dt><dd>{percent(item.metrics.accuracy)}%</dd></div><div><dt>F1 점수</dt><dd>{item.metrics.f1.toFixed(4)}</dd></div><div><dt>85% 기준</dt><dd>{item.quality_gate.passed ? '충족' : '미달'}</dd></div><div><dt>체크포인트</dt><dd>{item.checkpoint_exists ? '저장됨' : '가중치 없음'}</dd></div></dl><p className="version-architecture">{item.architecture}</p></article>)}</div><div className="pair-evaluation"><span>같은 평가 데이터에서 조도 포함 시 정확도 차이</span><b>{versionPair.accuracy_difference_percentage_points >= 0 ? '+' : ''}{versionPair.accuracy_difference_percentage_points.toFixed(2)}%p</b></div><p className="version-footnote">피처·입력 변환·표준화·가중치·임계값이 달라지면 새 버전으로 저장합니다. 기존 버전의 두 가중치를 덮어쓰지 않습니다. 위 버튼은 이 브라우저의 실험 모델 쌍을 선택합니다.</p></section>}
        </> : <>
        <div className="page-heading"><div><div className="eyebrow">센서로 읽는 공간</div><h1>강의실 사용 여부 실험</h1><p>센서값을 바꿔 예측하고, 조도 피처가 정확도에 미치는 영향을 비교해 보세요.</p></div><span className="version-tag">PyTorch <b>{currentVersion}</b></span></div>
        <section className="feature-panel" aria-labelledby="feature-heading">
          <div className="feature-icon"><Icon name="light" /></div><div className="feature-copy"><div className="section-eyebrow">이번 실험의 변수</div><h2 id="feature-heading">조도 피처를 사용할까요?</h2><p>스위치를 바꾸면 조도를 포함하거나 제외해 학습한 모델로 전환됩니다.</p></div>
          <div className="feature-control"><span>{lightEnabled ? '조도 피처 포함' : '조도 피처 제외'}</span><button className="switch" role="switch" aria-label="조도 피처 사용" aria-checked={lightEnabled} onClick={() => setMode(lightEnabled ? 'without_light' : 'with_light')}><span /></button></div>
        </section>

        <section id="evaluation" className="evaluation-section" aria-labelledby="evaluation-heading">
          <div className="section-heading"><h2 id="evaluation-heading">조도의 차이, 정확도로 확인하기</h2><span>동일한 평가 데이터 <b>{comparison ? comparison.test_rows.toLocaleString('ko-KR') : '—'}건</b></span></div>
          <div className="accuracy-grid"><AccuracyCard title="조도 피처 포함" description={featureNames(comparison?.modes.with_light.features)} report={comparison?.modes.with_light} selected={lightEnabled} onClick={() => setMode('with_light')} /><AccuracyCard title="조도 피처 제외" description={featureNames(comparison?.modes.without_light.features)} report={comparison?.modes.without_light} selected={!lightEnabled} onClick={() => setMode('without_light')} /><div className="difference-card"><span className="section-eyebrow">조도를 포함했을 때</span><div className="difference-number">{comparison ? (comparison.accuracy_difference_percentage_points >= 0 ? '+' : '') + comparison.accuracy_difference_percentage_points.toFixed(2) : '—'}<small>%p</small></div><p>두 모델의 평가 정확도 차이</p><span className="difference-note"><Icon name="info" />센서 입력을 바꿔도<br />평가 정확도는 변하지 않아요.</span></div></div>
        </section>

        <section id="experiment" className="experiment-section">
          <div className="sensor-panel">
            <div className="section-heading"><div><div className="section-eyebrow">직접 입력해 보기</div><h2>지금 강의실은 어떤 환경인가요?</h2></div><button className="reset-button" disabled={loading || !comparison} onClick={() => choosePreset(PRESETS[0])} aria-label="센서값 초기화"><Icon name="reset" /></button></div>
            <div className="preset-label">처음이라면, 예시 데이터로 시작하세요</div><div className="preset-grid">{PRESETS.map((item) => <button className={`preset-button ${preset === item.id ? 'preset-button--selected' : ''}`} disabled={loading || !comparison} key={item.id} onClick={() => choosePreset(item)} aria-pressed={preset === item.id}><Icon name={item.id === 'co2' ? 'air' : 'light'} /><b>{item.title}</b><small>{item.description}</small></button>)}</div>
            <form onSubmit={(event) => { event.preventDefault(); predict(sensors); }}>
              <fieldset className="input-fields" disabled={loading || !comparison}>
              <div className={`time-inputs ${!timeEnabled ? 'time-inputs--excluded' : ''}`}><div className="time-heading"><Icon name="clock" /><b>언제 측정한 값인가요?</b><span>{timeEnabled ? '학습·예측에 사용' : '이전 모델에서는 제외'}</span></div><div className="time-controls"><label htmlFor="Hour">시간 <small>0~23시</small><select id="Hour" aria-label="측정 시간" value={sensors.Hour} disabled={!timeEnabled} onChange={(event) => editSensor('Hour', Number(event.target.value))}>{Array.from({length:24}, (_, hour) => <option value={hour} key={hour}>{String(hour).padStart(2,'0')}시 · {hour < 12 ? '오전' : '오후'} {hour % 12 || 12}시</option>)}</select></label><label htmlFor="DayOfWeek">요일 <small>월 0~일 6</small><select id="DayOfWeek" aria-label="측정 요일" value={sensors.DayOfWeek} disabled={!timeEnabled} onChange={(event) => editSensor('DayOfWeek', Number(event.target.value))}>{DAYS.map((day,index) => <option value={index} key={day}>{day} · {index}</option>)}</select></label></div><p>측정한 공간의 시간과 요일을 선택하세요. 예: 수요일 오후 2시 = 시간 14, 요일 2. 날짜·예약 정보는 입력하지 않습니다.</p></div>
              <div className="sensor-list">{SENSORS.map((sensor) => <Sensor key={sensor.key} sensor={sensor} value={sensors[sensor.key]} excluded={sensor.key === 'Light' && !lightEnabled} onChange={(value) => editSensor(sensor.key, value)} />)}</div>
              </fieldset>
              <button className="predict-button" disabled={loading || !comparison} type="submit">{loading ? '예측하고 있어요…' : '이 센서값으로 예측하기'}<Icon name="arrow" /></button>
              <p className="input-footnote">예시 데이터는 가상의 측정값입니다. 센서값만으로 실제 사용 정답을 알 수는 없습니다.</p>
            </form>
          </div>

          <aside className="result-column" aria-label="선택한 모델의 예측 결과">
            <div className="prediction-panel" aria-live="polite"><div className="prediction-heading"><span className="section-eyebrow">센서 한 건의 예측</span><span className="model-mode-label">{lightEnabled ? '조도 포함' : '조도 제외'}</span></div>
              <div className={`room-visual ${lightEnabled ? 'room-visual--light' : ''}`} aria-hidden="true"><div className="room-window" /><div className="room-board" />{Array.from({ length: 6 }, (_, index) => <div className={`room-desk room-desk--${index + 1}`} key={index}><span /><span /></div>)}<span className="room-light room-light--one" /><span className="room-light room-light--two" /><div className="room-door" /></div>
              <div className="prediction-title">{loading ? '분석 중…' : prediction?.label || '예측 대기'}</div><p className="prediction-subtitle">{prediction ? '선택한 모델이 예상한 현재 상태' : '센서값을 선택하고 예측을 실행하세요'}</p>
              <div className="probability-block"><div><span>사용 중일 확률</span><strong>{prediction ? percent(prediction.probability, 1) : '—'}<small>%</small></strong></div><progress value={(prediction?.probability || 0) * 100} max="100" aria-label="사용 중일 확률" /><div className="probability-labels"><span>비어 있음</span><span>사용 중</span></div></div>
              <div className="other-result"><span>다른 모델의 예측</span><b>{predictions ? `${predictions[lightEnabled ? 'without_light' : 'with_light'].label} · 사용 확률 ${percent(predictions[lightEnabled ? 'without_light' : 'with_light'].probability, 1)}%` : '—'}</b></div>
            </div>
            <div className="model-summary"><div className="section-heading"><h2>선택한 모델의 평가</h2><Icon name="chart" /></div><dl><div><dt>정밀도</dt><dd>{percent(report?.model_metrics.precision)}%</dd></div><div><dt>재현율</dt><dd>{percent(report?.model_metrics.recall)}%</dd></div><div><dt>F1 점수</dt><dd>{report ? report.model_metrics.f1.toFixed(4) : '—'}</dd></div></dl><p>평가 정확도는 여러 정답 데이터에서 맞힌 비율이고, 위의 확률은 현재 입력 한 건의 예측값입니다.</p></div>
            <div className="experiment-note"><Icon name="info" /><div><b>조도 제외는 이렇게 작동해요</b><p>조도를 0으로 바꾸는 대신, 조도 입력 없이 학습한 모델을 사용합니다. 나머지 피처는 같은 값으로 두 모델에 입력합니다.</p></div></div>
          </aside>
        </section>
        </>}
        <footer className="workspace-footer"><span>시간순 학습·검증 분할 · 동일한 평가 파일 · 임계값 0.5</span><span>현재 재실 여부를 판별하며, 미래 사용 여부를 예측하지 않습니다.</span></footer>
      </div>
    </main>
  </div>;
}

// 파일 하나로 화면을 유지하되 Vite 갱신 시 React 루트 하나를 재사용한다.
const reactRoot = import.meta.hot?.data.reactRoot || createRoot(document.getElementById('root'));
reactRoot.render(<App />);
if (import.meta.hot) {
  import.meta.hot.dispose((data) => { data.reactRoot = reactRoot; });
}
