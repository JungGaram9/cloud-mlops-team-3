import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import ClassroomBoard from './ClassroomBoard.jsx';

// 기존 모델 관리 기능을 유지하며 운영 현황 화면은 별도 책임으로 분리한다.
const FEATURES = [
  { key: 'Temperature', label: '기온', unit: '°C', hint: '18° 서늘함 · 23° 보통 · 28° 따뜻함', description: '실내 공기 온도' },
  { key: 'Humidity', label: '습도', unit: '%', hint: '30% 건조 · 45% 보통 · 70% 습함', description: '공기의 상대습도' },
  { key: 'Light', label: '조도', unit: 'lux', hint: '0 어두움 · 150 은은함 · 500 밝은 실내', description: '센서 위치의 밝기' },
  { key: 'CO2', label: 'CO₂', unit: 'ppm', hint: '450 낮음 · 900 중간 · 1,500 높음', description: '호흡·환기의 영향을 받는 농도' },
  { key: 'Hour', label: '시간', unit: '시', hint: '0시 자정 · 12시 정오 · 18시 저녁', description: '측정 시각 · 24시간 주기' },
  { key: 'DayOfWeek', label: '요일', unit: '', hint: '월 0 · 화 1 · 수 2 · 목 3 · 금 4 · 토 5 · 일 6', description: '측정 요일 · 주말 여부 포함' },
];
const LABELS = Object.fromEntries(FEATURES.map(item => [item.key, item.label]));
const DAYS = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'];
const PRESETS = [
  { name: '밝은 실내', hint: '수요일 오후 2시', values: { Temperature: 23.1, Humidity: 27.2, Light: 430, CO2: 720, Hour: 14, DayOfWeek: 2 } },
  { name: '불 꺼진 실내', hint: '일요일 새벽 2시', values: { Temperature: 20.4, Humidity: 30, Light: 0, CO2: 450, Hour: 2, DayOfWeek: 6 } },
  { name: 'CO₂가 높은 실내', hint: '금요일 오후 4시', values: { Temperature: 23, Humidity: 35, Light: 450, CO2: 1500, Hour: 16, DayOfWeek: 4 } },
];
const DEFAULT_CONFIG = { name: '새 학습 실험', features: FEATURES.map(item => item.key), epochs: 200, learning_rate: 0.01,
  batch_size: 0, hidden_layers: [16, 8], activation: 'relu', optimizer: 'adam', dropout: 0, weight_decay: 0,
  validation_ratio: 0.2, patience: 30, threshold: 0.5, seed: 42, normalize: true, balance_classes: false,
  compare_light: true, make_default: false };
const STATUS = { queued: '준비 중', running: '학습 중', completed: '완료', failed: '실패', cancelled: '중지됨' };
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—';
const decimal = value => Number.isFinite(value) ? value.toFixed(4) : '—';
const date = value => value ? new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value)) : '—';
const featureText = features => features.map(key => LABELS[key] || key).join(' · ');
const modelLabel = mode => mode === 'with_light' ? '선택 모델' : '조도 제외 비교';
const activeJob = job => job && ['queued', 'running'].includes(job.status);

function predictionError(values, limits) {
  if (!limits?.available) return limits?.error || '선택 모델의 입력 제한을 준비할 수 없습니다.';
  for (const [key, bounds] of Object.entries(limits.features)) {
    const value = values[key];
    if (!Number.isFinite(value)) return `${LABELS[key]} 값을 입력하세요.`;
    if (bounds.values) {
      if (!bounds.values.includes(value)) return `${LABELS[key]}은 선택 모델이 학습에서 관측한 값만 선택하세요.`;
    } else if (value < bounds.minimum || value > bounds.maximum) {
      return `${LABELS[key]}은 실제 학습 관측 범위 ${bounds.minimum}~${bounds.maximum} 안에 입력하세요.`;
    }
  }
  return '';
}

async function request(path, { signal, body, method } = {}) {
  const response = await fetch(`/api${path}`, { signal, method: method || (body !== undefined ? 'POST' : 'GET'),
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.details?.map(item => `${item.field}: ${item.reason}`).join(', ');
    throw new Error(`${data.error || '요청을 처리하지 못했습니다.'}${detail ? ` (${detail})` : ''}`);
  }
  return data;
}

function Icon({ name, size = 20 }) {
  const paths = {
    lab: <><path d="M9 3h6m-5 0v7l-5 8a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-8V3M8 15h8" /><path d="M10 18h.01M14 17h.01" /></>,
    dashboard: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="11" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="18" width="7" height="3" rx="1" /></>,
    sliders: <><path d="M4 7h5m5 0h6M4 17h9m5 0h2" /><circle cx="11.5" cy="7" r="2.5" /><circle cx="15.5" cy="17" r="2.5" /></>,
    train: <><path d="M4 19V5m0 14h16M8 14l4-5 4 3 4-7" /><path d="M16 5h4v4" /></>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    play: <path d="m8 5 11 7-11 7Z" />,
    refresh: <><path d="M20 11a8 8 0 0 0-14-5L3 9m0-5v5h5M4 13a8 8 0 0 0 14 5l3-3m0 5v-5h-5" /></>,
    file: <><path d="M14 3H5v18h14V8Zm0 0v5h5M8 12h8M8 16h6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.lab}</svg>;
}

function Switch({ label, checked, onChange, disabled, description }) {
  return <div className={`switch-row ${disabled ? 'muted' : ''}`}><div><b>{label}</b>{description && <small>{description}</small>}</div>
    <button type="button" role="switch" aria-label={label} aria-checked={checked} disabled={disabled} className={`switch ${checked ? 'on' : ''}`} onClick={() => onChange(!checked)}><span /></button></div>;
}

function RangeControl({ label, value, onChange, min, max, step = 1, numberStep = step, hint, suffix = '', disabled = false, format, form }) {
  return <div className="range-control"><div className="control-heading"><label>{label}</label>
    <div className="numeric"><input type="number" form={form} aria-label={`${label} 숫자 입력`} min={min} max={max} step={numberStep} required value={value} disabled={disabled}
      onChange={event => onChange(event.target.value === '' ? '' : Number(event.target.value))} /><span>{suffix}</span></div></div>
    <input type="range" form={form} aria-label={`${label} 슬라이더`} min={min} max={max} step={step} value={value === '' ? min : value} disabled={disabled} onChange={event => onChange(Number(event.target.value))} />
    <div className="range-caption"><span>{hint}</span>{format && <b>{format(value)}</b>}</div></div>;
}

function SensorControl({ feature, value, onChange, bounds, disabled, preview = false }) {
  const label = `${feature.label}${preview ? ' 미리보기' : ''}`;
  // 미리보기 입력의 유효성은 CSV 학습 폼이 아닌 별도 예측 폼에 적용한다.
  const form = preview ? 'sensor-preview-form' : undefined;
  if (!bounds) return <div className="range-control">
    <div className="control-heading"><label>{label}</label>
      <div className="numeric"><input type="number" aria-label={`${label} 숫자 입력`} value={value} disabled /><span>{feature.unit}</span></div>
    </div>
    <small>선택 모델에서 사용하지 않거나 관측 범위를 확인할 수 없는 피처</small>
  </div>;
  if (bounds.values) return <label className="select-control">{label}
    <select form={form} aria-label={`${label} 관측값 선택`} value={value} disabled={disabled} required
      onChange={event => onChange(Number(event.target.value))}>
      {!bounds.values.includes(value) && <option value={value} disabled>관측값을 선택하세요{value === '' ? '' : ` (현재 ${value})`}</option>}
      {bounds.values.map(item => <option key={item} value={item}>{feature.key === 'DayOfWeek' ? DAYS[item] : `${item}시`}</option>)}
    </select>
    <small>실제 학습에 관측된 값만 선택할 수 있습니다.</small>
  </label>;
  return <RangeControl label={label} value={value} onChange={onChange}
    form={form}
    min={bounds.minimum} max={bounds.maximum} step="any" numberStep="any"
    suffix={feature.unit} hint={feature.hint} disabled={disabled} />;
}

function LossChart({ models = {}, curves = {}, compact = false, emptyMessage = '학습이 시작되면 손실 곡선을 표시합니다.' }) {
  const [mode, setMode] = useState('with_light');
  const modes = [...new Set([...Object.keys(models), ...Object.keys(curves)])];
  const selected = modes.includes(mode) ? mode : modes[0];
  const points = curves[selected] || models[selected]?.training?.history || [];
  const finite = points.filter(point => Number.isFinite(point.training_loss) && Number.isFinite(point.validation_loss));
  const maxLoss = Math.max(0.01, ...finite.flatMap(point => [point.training_loss, point.validation_loss]));
  const maxEpoch = Math.max(1, finite.at(-1)?.epoch || 1);
  const x = point => 42 + point.epoch / maxEpoch * 510;
  const y = value => 190 - value / maxLoss * 155;
  return <div className={`loss-chart ${compact ? 'compact' : ''}`}><div className="section-heading"><div><h3>학습 곡선</h3><p>검증 손실이 낮을수록 정답에 가까운 확률입니다.</p></div>
    {modes.length > 1 && <select aria-label="손실 곡선 모델" value={selected} onChange={event => setMode(event.target.value)}>{modes.map(key => <option key={key} value={key}>{modelLabel(key)}</option>)}</select>}</div>
    {finite.length ? <><div className="chart-legend"><span><i className="dot green" />학습 손실</span><span><i className="dot blue" />검증 손실</span></div>
      <svg viewBox="0 0 580 225" role="img" aria-label="학습 및 검증 손실 곡선">
        {[0, 0.5, 1].map(fraction => <g key={fraction}><line x1="42" y1={y(maxLoss * fraction)} x2="552" y2={y(maxLoss * fraction)} className="grid-line" /><text x="32" y={y(maxLoss * fraction) + 4} textAnchor="end">{(maxLoss * fraction).toFixed(2)}</text></g>)}
        <polyline points={finite.map(point => `${x(point)},${y(point.training_loss)}`).join(' ')} className="curve training" />
        <polyline points={finite.map(point => `${x(point)},${y(point.validation_loss)}`).join(' ')} className="curve validation" />
        <text x="42" y="215">1회</text><text x="552" y="215" textAnchor="end">{maxEpoch}회</text>
      </svg></> : <div className="empty-chart"><Icon name="train" size={28} /><p>{emptyMessage}</p></div>}</div>;
}

function MetricGrid({ metrics }) {
  return <div className="metric-grid">{[['정확도', metrics?.accuracy], ['정밀도', metrics?.precision], ['재현율', metrics?.recall], ['F1', metrics?.f1]].map(([name, value]) => <div key={name}><small>{name}</small><strong>{name === 'F1' ? decimal(value) : percent(value)}</strong></div>)}</div>;
}

function FeatureTags({ features = [] }) {
  return <div className="feature-tags">{features.map(key => <span key={key}>{LABELS[key] || key}</span>)}</div>;
}

const SOURCE_LABELS = { legacy: '기존 자료', Room_A: 'Room A', Room_B: 'Room B' };
const splitLabel = strategy => strategy === 'source_chronological' ? '출처별 시간순' : '전체 시간순 (기존 방식)';

function evaluationGroups(versions) {
  const groups = new Map();
  for (const item of versions) {
    const data = item.test_data;
    const key = data?.sha256 && data?.rows ? `${data.sha256}:${data.rows}` : `unknown:${item.version}`;
    if (!groups.has(key)) groups.set(key, { key, data, versions: [] });
    groups.get(key).versions.push(item);
  }
  return [...groups.values()];
}

function SourceSplitTable({ fit, validation, test }) {
  const names = [...new Set([fit, validation, test].flatMap(part => Object.keys(part?.sources || {})))];
  if (!names.length) return <p className="small-note">출처별 분할 정보가 저장되지 않은 버전입니다.</p>;
  const cell = stats => stats ? <>{stats.rows.toLocaleString()}건<small>비어 있음 {stats.class_counts['0'] || 0} · 사용 중 {stats.class_counts['1'] || 0}</small></> : '0건';
  return <div className="table-scroll source-splits"><table>
    <caption>출처별 행수와 정답 분포</caption><thead><tr><th>출처</th><th>실제 학습 (fit)</th><th>검증</th><th>평가</th></tr></thead>
    <tbody>{names.map(name => <tr key={name}><th>{SOURCE_LABELS[name] || name}</th>
      {[fit, validation, test].map((part, index) => <td key={index}>{cell(part?.sources?.[name])}</td>)}
    </tr>)}</tbody></table></div>;
}

function downloadReport(report) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = `${report.model_version}-training.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ReportPanel({ report, version, loading, onExperiment, onReuse, onActivate, isDefault, activating }) {
  const modes = Object.entries(report?.models || {});
  const primaryModel = report?.models.with_light;
  const config = report?.config || primaryModel?.training;
  const parameters = report ? [
    ['옵티마이저', config.optimizer?.toUpperCase()],
    ['학습률', config.learning_rate],
    ['최대 에포크', config.epochs ?? config.max_epochs],
    ['실행 / 최적 에포크', `${primaryModel.training.epochs_run} / ${primaryModel.training.best_epoch}`],
    ['배치 크기', config.batch_size === 0 ? '전체 배치' : config.batch_size],
    ['은닉층', config.hidden_layers?.join(' → ')],
    ['활성화', config.activation],
    ['드롭아웃', config.dropout],
    ['가중치 감쇠', config.weight_decay],
    ['검증 비율', config.validation_ratio == null ? null : percent(config.validation_ratio)],
    ['조기 종료 대기', config.patience === 0 ? '끄기' : config.patience],
    ['시드', config.seed ?? primaryModel.random_state],
    ['표준화', config.normalize == null ? null : config.normalize ? '사용' : '제외'],
    ['클래스 균형', config.balance_classes == null ? null : config.balance_classes ? '보정' : '원본'],
    ['학습 / 검증 / 평가 행', `${primaryModel.training_split.rows.toLocaleString()} / ${primaryModel.validation_split.rows.toLocaleString()} / ${primaryModel.test_data.rows.toLocaleString()}`],
    ['검증 최저 손실', decimal(primaryModel.training.best_validation_loss)],
  ] : [];
  return <section className="panel report-panel"><div className="section-heading"><div><span className="eyebrow">모델 기록</span><h2>{version || '모델 선택'} 학습 보고서</h2></div><Icon name="file" /></div>
    {loading ? <p className="empty">보고서를 불러오는 중입니다.</p> : !report ? <p className="empty">버전을 선택하면 학습 설정과 실제 평가 결과를 볼 수 있습니다.</p> : <>
      {report.legacy && <p className="notice">{report.note}</p>}
      {report.config && <div className="report-title"><h3>{report.config.name}</h3><small>학습 {report.duration_seconds?.toFixed(1)}초 · {date(report.created_at_utc)}</small></div>}
      <div className="report-actions"><button className="button secondary" onClick={onExperiment}>예측 실험 <Icon name="arrow" size={16} /></button><button className="button secondary" onClick={onReuse}>설정 불러오기</button>
        <button className="button secondary" disabled={isDefault || activating} onClick={onActivate}>{isDefault ? '기본 모델 사용 중' : activating ? '적용 중…' : '기본 모델로 사용'}</button>
        <button className="text-button" onClick={() => downloadReport(report)}>JSON 내려받기</button></div>
      {modes.map(([mode, model]) => <div className="model-report" key={mode}><div className="section-heading"><h3>{modelLabel(mode)}</h3><span className={`badge ${model.quality_gate?.passed ? 'success' : 'neutral'}`}>{model.quality_gate?.passed ? '85% 기준 충족' : '85% 기준 미달'}</span></div>
        <FeatureTags features={model.features} /><MetricGrid metrics={model.model_metrics} /><p className="architecture">{model.architecture} · 임계값 {model.threshold}</p>
        <div className="confusion"><b>혼동 행렬</b><span>실제 비어 있음 → 예측 비어 있음 / 사용 중: {model.model_metrics.confusion_matrix[0].join(' / ')}</span><span>실제 사용 중 → 예측 비어 있음 / 사용 중: {model.model_metrics.confusion_matrix[1].join(' / ')}</span></div>
        <details className="source-evaluation" open><summary>출처별 평가</summary>
          {model.source_metrics ? Object.entries(model.source_metrics).map(([name, result]) => <div key={name} className="source-result">
            <h4>{SOURCE_LABELS[name] || name} · {result.rows.toLocaleString()}건</h4>
            <p className="small-note">비어 있음 {result.class_counts['0'] || 0} · 사용 중 {result.class_counts['1'] || 0}</p>
            {result.metrics ? <><MetricGrid metrics={result.metrics} />
              <p className="small-note">혼동 행렬 [실제 0: 예측 0 / 1, 실제 1: 예측 0 / 1]<br />
                {result.metrics.confusion_matrix.map(row => `[${row.join(' / ')}]`).join(' · ')}</p>
              {result.metrics.roc_auc == null && <p className="small-note">단일 클래스이므로 ROC-AUC·평균 정밀도는 계산하지 않습니다.</p>}
            </> : <p className="small-note">평가 행이 없습니다.</p>}
          </div>) : <p className="small-note">기존 보고서에는 출처별 평가가 없습니다.</p>}
        </details>
        {model.future_subset_metrics && <p className="small-note">학습 기간 이후 구간 정확도 {percent(model.future_subset_metrics.accuracy)} · 전체 평가 정확도와 별도로 확인하세요.</p>}
      </div>)}
      <p className="dataset-identity">평가 {primaryModel.test_data.rows.toLocaleString()}건 · SHA-256 {primaryModel.test_data.sha256 || '기록 없음'}</p>
      <p className="small-note">분할: {splitLabel(primaryModel.split_strategy)}</p>
      <SourceSplitTable fit={primaryModel.training_split} validation={primaryModel.validation_split} test={primaryModel.test_data} />
      <LossChart models={report.models} emptyMessage={report.legacy ? '기존 버전에는 손실 곡선이 저장되어 있지 않습니다.' : undefined} />
      <div className="section-heading"><h3>하이퍼파라미터</h3><span className="small-note">선택 모델 기준</span></div>
      <dl className="parameter-grid">{parameters.map(([label, value]) => (
        <div key={label}><dt>{label}</dt><dd>{value ?? '기록 없음'}</dd></div>
      ))}</dl>
      <p className="small-note">새 실험은 출처마다 시간순으로 fit/validation을 나눕니다. 출처가 다른 행끼리는 시각이 겹칠 수 있습니다. 같은 평가 자료로 반복 실험한 결과는 독립적인 최종 검증이 아닙니다.</p>
    </>}
  </section>;
}

function PredictionForm({ values, setValues, enabled, limits, presets = true, disabled = false }) {
  return <><div className="section-heading"><div><h3>센서 입력</h3><p>값을 조절하고 해당 모델의 예측을 확인하세요.</p></div></div>
    <p className="small-note">선택 모델의 실제 학습 관측 범위</p>
    {presets && <div className="presets">{PRESETS.map(preset => {
      const reason = predictionError(preset.values, limits);
      return <button type="button" className="preset" key={preset.name} disabled={disabled || Boolean(reason)} title={reason}
        onClick={() => { if (!disabled && !predictionError(preset.values, limits)) setValues({ ...preset.values }); }}>
        <b>{preset.name}</b><small>{preset.hint}</small>{reason && <small>{reason}</small>}
      </button>;
    })}</div>}
    <div className="sensor-grid">{FEATURES.map(feature => {
      const excluded = enabled && !enabled.includes(feature.key);
      return <div className={`sensor-field ${excluded ? 'excluded' : ''}`} key={feature.key}><SensorControl feature={feature} value={values[feature.key]}
        bounds={limits?.features[feature.key]} disabled={disabled || excluded || !limits?.available}
        onChange={value => setValues(previous => ({ ...previous, [feature.key]: value }))} /></div>;
    })}</div></>;
}

function Dashboard({ registry, selected, onSelect, job }) {
  const totalVersions = registry.versions.length;
  const latest = registry.versions[0];
  const groups = evaluationGroups(registry.versions);

  return <>
    <div className="summary-grid">
      <div className="summary-card"><small>저장된 모델 버전</small><strong>{totalVersions}<span>개</span></strong><p>학습 설정과 평가를 함께 보존</p></div>
      <div className="summary-card"><small>평가 데이터 그룹</small><strong>{groups.length}<span>개</span></strong><p>같은 SHA-256·행수 안에서만 비교</p></div>
      <div className="summary-card"><small>기본 예측 모델</small><strong>{registry.current_version || '—'}</strong><p>다른 실험과 독립적으로 선택</p></div>
      <div className="summary-card"><small>최근 학습</small><strong className="summary-word">{activeJob(job) ? STATUS[job.status] : latest?.version || '—'}</strong><p>{activeJob(job) ? job.phase : latest?.name || '모델 준비 중'}</p></div>
    </div>
    <section className="panel">
      <div className="section-heading">
        <div><span className="eyebrow">성능 비교</span><h2>버전마다 달라지는 정확도</h2></div>
        <div className="chart-legend"><span><i className="dot green" />선택 모델</span><span><i className="dot blue" />조도 제외 비교</span></div>
      </div>
      {groups.map(group => {
        const best = group.versions.reduce((previous, item) => !previous || item.models[0].metrics.accuracy > previous.models[0].metrics.accuracy ? item : previous, null);
        return <div className="evaluation-group" key={group.key}>
        <h3>{group.data?.rows?.toLocaleString() || '행수 미상'}건 · 그룹 최고 {best.version} {percent(best.models[0].metrics.accuracy)}</h3>
        <p className="dataset-identity">SHA-256 {group.data?.sha256 || '기록 없음, 다른 버전과 비교하지 않음'}</p>
        <div className="performance-chart">
        {[...group.versions].reverse().map(item => (
          <button className={`performance-row ${selected === item.version ? 'selected' : ''}`} key={item.version}
            onClick={() => onSelect(item.version)} aria-label={`${item.version} 보고서 선택`}>
            <span>{item.version}</span>
            <div className="performance-track">
              <div className="performance-bar primary-bar" style={{ width: percent(item.models[0].metrics.accuracy) }} />
              <b>{percent(item.models[0].metrics.accuracy)}</b>
              {item.models[1] && <div className="performance-bar secondary-bar" style={{ width: percent(item.models[1].metrics.accuracy) }} />}
            </div>
            <span className="feature-count">피처 {item.models[0].features.length}개</span>
          </button>
        ))}
        </div></div>;
      })}
      <p className="small-note">평가 해시와 행수가 같은 그룹 안에서만 비교합니다. 다른 그룹의 정확도로 모델 간 우열을 판단하지 않습니다. 검증 비율·임계값도 함께 확인하세요.</p>
    </section>
    <section className="panel history-panel">
      <div className="section-heading">
        <div><span className="eyebrow">실험 이력</span><h2>모델 버전</h2></div>
        <span className="badge neutral">{totalVersions}개 보존됨</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>버전 / 실험</th><th>사용한 피처</th><th>정확도</th><th>F1</th><th>학습 설정</th><th>기록 시각</th></tr></thead>
          <tbody>{registry.versions.map(item => (
            <tr key={item.version} className={selected === item.version ? 'selected' : ''}>
              <td>
                <button className="version-link" onClick={() => onSelect(item.version)}>{item.version}<Icon name="arrow" size={14} /></button>
                {item.current && <span className="badge success">기본</span>}
                <small>{item.name}</small>
                <small>평가 {item.test_data?.rows?.toLocaleString() || '—'}건</small>
                <small className="dataset-identity" title={item.test_data?.sha256}>SHA-256 {item.test_data?.sha256?.slice(0, 12) || '기록 없음'}</small>
                {item.same_model_as && <small>{item.same_model_as}와 같은 가중치</small>}
              </td>
              <td><FeatureTags features={item.models[0].features} /></td>
              <td className="metric-cell">{percent(item.models[0].metrics.accuracy)}</td>
              <td>{decimal(item.models[0].metrics.f1)}</td>
              <td>{item.config ? <>
                <b>{item.config.optimizer.toUpperCase()} · {item.config.learning_rate}</b>
                <small>{item.config.epochs}회 · {item.config.hidden_layers.join(' → ')}</small>
              </> : <small>기존 저장 정보</small>}</td>
              <td>{date(item.created_at_utc)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  </>;
}

function TrainingForm({ config, options, previewLimits, values, setValues, busy, onSubmit, onUpdate, onToggleFeature, onReset }) {
  const plan = options?.data_plan;
  const planMatches = plan?.validation_ratio === config.validation_ratio;
  return (
    <form className="training-form" onSubmit={onSubmit}>
      <fieldset disabled={busy || !options}>
        <section className="panel">
          <div className="section-heading">
            <div><span className="step-label">01 · 실험 설계</span><h2>이름과 데이터</h2></div>
            <span className="badge neutral">출처별 시간순 분할</span>
          </div>
          <p className="small-note">기존 자료·Room A·Room B 각각의 앞부분을 학습하고 뒷부분을 검증합니다. 출처는 입력 피처에 추가하지 않습니다.</p>
          {planMatches ? <><SourceSplitTable fit={plan.fit} validation={plan.validation} test={plan.test} />
            {plan.errors.map(value => <p className="inline-error" key={value}>{value}</p>)}
            {plan.warnings.map(value => <p className="small-note" key={value}>{value}</p>)}
          </> : <p className="small-note" role="status">설정에 맞는 분할을 확인하는 중입니다.</p>}
          <label className="field-label" htmlFor="run-name">실험 이름</label>
          <input id="run-name" className="wide-input" value={config.name} maxLength={80} required
            onChange={event => onUpdate('name', event.target.value)} placeholder="예: 시간과 요일을 제외한 실험" />
          <div className="dataset-card">
            <Icon name="file" />
            <div>
              <b>학습 파일 {options?.training_data.rows.toLocaleString() || '—'}건 (분할 전) · 평가 {options?.test_data.rows.toLocaleString() || '—'}건</b>
              <small>training_dataset.csv / test_dataset.csv</small>
              <small>표준화는 학습 구간에서만 계산 · 평가 정답은 학습에 사용하지 않음</small>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div><span className="step-label">02 · 피처 선택</span><h2>무엇을 학습할까요?</h2></div>
            <span className="badge success">{config.features.length} / 6개 사용</span>
          </div>
          <p className="section-description">스위치는 실제 학습 입력을 바꿉니다. 아래 슬라이더 값은 완료한 모델의 예측 미리보기에 사용하며 CSV 학습값을 바꾸지 않습니다.</p>
          <p className="small-note">선택 모델의 실제 학습 관측 범위{previewLimits?.available ? ` · ${previewLimits.rows.toLocaleString()}건` : ''}</p>
          <div className="training-feature-grid">
            {FEATURES.map(feature => {
              const enabled = config.features.includes(feature.key);
              const profile = previewLimits?.features[feature.key];
              return (
                <div className={`training-feature ${enabled ? 'enabled' : ''}`} key={feature.key}>
                  <Switch label={`${feature.label} 피처 사용`} checked={enabled}
                    onChange={() => onToggleFeature(feature.key)} description={feature.description} />
                  <SensorControl feature={feature} preview value={values[feature.key]} bounds={profile}
                    disabled={!previewLimits?.available || !profile}
                    onChange={value => setValues(previous => ({ ...previous, [feature.key]: value }))} />
                  <small className="data-range">
                    {!profile ? '선택 모델의 관측 범위를 사용할 수 없습니다.'
                      : feature.key === 'DayOfWeek' ? `학습 관측값 ${profile.values.map(value => DAYS[value]).join(' · ')}`
                      : profile.values ? `관측값 ${profile.values.length}개 (선택 목록에서 확인)`
                      : `학습 관측값 ${profile.minimum}~${profile.maximum}${feature.unit}`}
                  </small>
                </div>
              );
            })}
          </div>
          {!config.features.length && <p className="inline-error">피처를 한 개 이상 선택하세요.</p>}
          <Switch label="조도 제외 모델도 비교 학습" checked={config.compare_light}
            disabled={!config.features.includes('Light') || config.features.length < 2}
            onChange={value => onUpdate('compare_light', value)}
            description="조도와 다른 피처를 선택한 경우, 같은 설정으로 조도만 제외한 모델을 함께 보존합니다." />
        </section>

        <section className="panel">
          <div className="section-heading">
            <div><span className="step-label">03 · 학습 설정</span><h2>모델의 학습 방식을 조절하세요.</h2></div>
            <button type="button" className="text-button" onClick={onReset}>설정 초기화</button>
          </div>
          <div className="control-grid">
            <RangeControl label="최대 에포크" value={config.epochs} min={1} max={500}
              hint="데이터를 반복해서 학습하는 횟수" suffix="회" onChange={value => onUpdate('epochs', value)} />
            <RangeControl label="학습률" value={config.learning_rate} min={0.00001} max={0.1} step={0.00001}
              hint="가중치 변경 폭 · 기본 0.01" onChange={value => onUpdate('learning_rate', value)} />
            <label className="select-control">옵티마이저
              <select aria-label="옵티마이저" value={config.optimizer} onChange={event => onUpdate('optimizer', event.target.value)}>
                <option value="adam">Adam · 적응형 학습률</option>
                <option value="sgd">SGD · 경사 하강법</option>
              </select>
            </label>
            <label className="select-control">배치 크기
              <select aria-label="배치 크기" value={config.batch_size} onChange={event => onUpdate('batch_size', Number(event.target.value))}>
                {[0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].map(value => (
                  <option key={value} value={value}>{value === 0 ? '전체 배치' : `${value}행 / 배치`}</option>
                ))}
              </select>
            </label>
            <RangeControl label="검증 비율" value={config.validation_ratio} min={0.1} max={0.4} step={0.01}
              hint="출처마다 시간순 뒤쪽 데이터를 검증에 사용" format={percent} onChange={value => onUpdate('validation_ratio', value)} />
            <RangeControl label="조기 종료 대기" value={config.patience} min={0} max={100} suffix="회"
              hint="검증 손실이 개선되지 않는 횟수 · 0은 끄기" onChange={value => onUpdate('patience', value)} />
          </div>
          <div className="architecture-controls">
            <div className="section-heading"><h3>신경망 구조</h3><span className="architecture-badge">{config.hidden_layers.join(' → ')} → 1</span></div>
            <div className="control-grid">
              <label className="select-control">은닉층 수
                <select aria-label="은닉층 수" value={config.hidden_layers.length}
                  onChange={event => onUpdate('hidden_layers', Array.from({ length: Number(event.target.value) }, (_, index) => config.hidden_layers[index] || 8))}>
                  {[1, 2, 3].map(value => <option key={value} value={value}>{value}개</option>)}
                </select>
              </label>
              <label className="select-control">활성화 함수
                <select aria-label="활성화 함수" value={config.activation} onChange={event => onUpdate('activation', event.target.value)}>
                  <option value="relu">ReLU</option>
                  <option value="tanh">Tanh</option>
                </select>
              </label>
              {config.hidden_layers.map((width, index) => (
                <RangeControl key={index} label={`은닉층 ${index + 1} 너비`} value={width} min={4} max={128}
                  hint="층의 뉴런 개수"
                  onChange={value => onUpdate('hidden_layers', config.hidden_layers.map((item, position) => position === index ? value : item))} />
              ))}
              <RangeControl label="드롭아웃" value={config.dropout} min={0} max={0.5} step={0.01}
                hint="학습 중 일부 뉴런을 무작위 제외" format={percent} onChange={value => onUpdate('dropout', value)} />
              <RangeControl label="가중치 감쇠" value={config.weight_decay} min={0} max={0.1} step={0.0001}
                hint="가중치가 과도하게 커지는 것을 줄임" onChange={value => onUpdate('weight_decay', value)} />
            </div>
          </div>
          <div className="control-grid advanced-controls">
            <RangeControl label="분류 임계값" value={config.threshold} min={0.1} max={0.9} step={0.01}
              hint="이 확률 이상이면 사용 중으로 판별" onChange={value => onUpdate('threshold', value)} />
            <label className="select-control">랜덤 시드
              <input aria-label="랜덤 시드" type="number" min={0} max={2147483647} step={1} value={config.seed}
                onChange={event => onUpdate('seed', event.target.value === '' ? '' : Number(event.target.value))} />
              <small>동일 설정을 재현하기 위한 시작값</small>
            </label>
          </div>
          <Switch label="입력 표준화" checked={config.normalize} onChange={value => onUpdate('normalize', value)}
            description="서로 다른 센서 단위의 크기를 맞춥니다." />
          <Switch label="사용 중 클래스 균형 보정" checked={config.balance_classes}
            onChange={value => onUpdate('balance_classes', value)} description="사용 중 데이터가 적을 때 학습 손실의 가중치를 조정합니다." />
          <Switch label="완료 모델을 기본 예측에 사용" checked={config.make_default}
            onChange={value => onUpdate('make_default', value)} description="끄면 기존 기본 모델을 유지하고 새 모델을 실험 이력에만 추가합니다." />
        </section>
      </fieldset>
      <div className="training-submit">
        <div><b>{busy ? '학습을 진행하고 있습니다.' : '준비된 설정으로 새 버전을 학습합니다.'}</b><small>완료된 모델만 등록 · 실패와 중지 이력은 별도 보존</small></div>
        <button type="submit" className="button primary" disabled={busy || !options || !config.features.length || !planMatches || !plan?.ready}>
          <Icon name="play" size={17} />{busy ? '학습 중…' : '학습 시작'}
        </button>
      </div>
    </form>
  );
}

function TrainingMonitor({ job, jobs, cancelling, onCancel, onViewReport, onInspect, selected,
  report, reportLoading, values, predicting, onPredict, predictions, previewVersion }) {
  const primaryModel = report?.models.with_light;
  const primaryPrediction = predictions?.with_light;
  const inputError = predictionError(values, report?.input_limits?.with_light);
  return (
    <aside className="training-aside">
      <section className="panel live-panel">
        <div className="section-heading">
          <div><span className="eyebrow">실시간 학습</span><h2>{job ? STATUS[job.status] : '학습을 기다리고 있어요.'}</h2></div>
          {job && <span className={`badge ${job.status === 'completed' ? 'success' : 'neutral'}`}>{job.version || '새 실험'}</span>}
        </div>
        <p className="job-name">{job?.config.name || '설정을 조정한 뒤 학습을 시작하세요.'}</p>
        {job && <>
          <div className="progress-heading">
            <span>{job.cancel_requested && activeJob(job) ? '중지 요청 처리 중' : job.phase}</span>
            <b>{job.progress || 0}%</b>
          </div>
          <progress max="100" value={job.progress || 0} aria-label="학습 진행률" />
          <p className="small-note">{job.epoch ? `${job.epoch} / ${job.config.epochs} 에포크 · ` : ''}{date(job.created_at_utc)}</p>
          {activeJob(job) && <button className="button secondary full-width"
            disabled={cancelling || job.cancel_requested} onClick={onCancel}>
            {cancelling || job.cancel_requested ? '중지 요청됨' : '학습 중지'}
          </button>}
          {job.error && <p className="inline-error">{job.error}</p>}
          <LossChart curves={job.curves} compact />
          {job.status === 'completed' && <button className="button secondary full-width" onClick={() => onViewReport(job.version)}>
            완료 모델 보고서 보기 <Icon name="arrow" size={15} />
          </button>}
        </>}
        <p className="small-note">탭을 바꾸거나 새로고침해도 서버에서 학습은 계속됩니다. 동시에 한 실험을 실행합니다.</p>
      </section>
      <form id="sensor-preview-form" className="panel preview-panel" onSubmit={onPredict}>
        <div className="section-heading"><div><span className="eyebrow">입력 미리보기</span><h3>{selected || '—'} 예측</h3></div></div>
        <p className="small-note">피처 카드의 슬라이더 값으로 저장된 모델을 테스트합니다. 아직 학습하지 않은 설정의 예측은 아닙니다.</p>
        <FeatureTags features={primaryModel?.features} />
        {inputError && <p className="inline-error">{inputError} 버전 전환 후 기존 값이 범위 밖이면 값을 수정하세요.</p>}
        <button type="submit" className="button secondary full-width" disabled={predicting || reportLoading || !report || Boolean(inputError)}>
          {predicting ? '예측 중…' : '미리보기 값으로 예측'}
        </button>
        {primaryPrediction && previewVersion === selected && <div className="preview-result">
          <b>{primaryPrediction.label}</b><strong>{percent(primaryPrediction.probability)}</strong>
          <small>사용 중일 확률 · {previewVersion}</small>
        </div>}
      </form>
      <section className="panel">
        <div className="section-heading"><h3>최근 학습 요청</h3><span className="badge neutral">{jobs.length}</span></div>
        <div className="job-history">
          {jobs.length ? jobs.slice(0, 8).map(item => (
            <button key={item.id} onClick={() => onInspect(item)}>
              <span><b>{item.version || STATUS[item.status]}</b><small>{item.config.name}</small></span>
              <span className={`status-label ${item.status}`}>{STATUS[item.status]}</span>
            </button>
          )) : <p className="empty">아직 실행한 학습이 없습니다.</p>}
        </div>
      </section>
    </aside>
  );
}

function PredictionExperiment({ registry, selected, onSelect, report, reportLoading, values, setValues,
  predicting, onPredict, predictions, previewVersion, activeMode, onModeChange }) {
  const primaryModel = report?.models.with_light;
  const selectedModel = report?.models[activeMode] || primaryModel;
  const selectedPrediction = predictions?.[activeMode];
  const availableComparison = Boolean(report?.models.without_light);
  const limits = report?.input_limits?.[activeMode];
  const inputError = predictionError(values, limits);
  return <>
    <div className="model-selector">
      <label htmlFor="experiment-version">실험할 모델</label>
      <select id="experiment-version" value={selected} disabled={!registry.versions.length} onChange={event => onSelect(event.target.value)}>
        {registry.versions.map(item => (
          <option key={item.version} value={item.version}>{item.version} · {item.name}{item.current ? ' · 기본' : ''}</option>
        ))}
      </select>
    </div>
    <div className="experiment-layout">
      <form className="panel" onSubmit={onPredict}>
        <PredictionForm values={values} setValues={setValues} enabled={selectedModel?.features} limits={limits} disabled={!report || reportLoading} />
        {inputError && <p className="inline-error">{inputError} 버전 전환 후 기존 값이 범위 밖이면 값을 수정하세요.</p>}
        <button type="submit" className="button primary full-width" disabled={predicting || !report || reportLoading || Boolean(inputError)}>
          {predicting ? '예측 중…' : '이 값으로 예측하기'}<Icon name="arrow" size={17} />
        </button>
        <p className="small-note">예시는 가상 측정값입니다. 입력 한 건의 확률과 여러 정답에서 계산한 평가 정확도는 다릅니다.</p>
      </form>
      <section className="panel prediction-panel">
        <div className="section-heading"><h2>예측 결과</h2><span className="badge neutral">{selected}</span></div>
        <div className="mode-buttons">
          {Object.keys(report?.models || {}).map(mode => (
            <button key={mode} aria-pressed={activeMode === mode} className={activeMode === mode ? 'active' : ''}
              onClick={() => onModeChange(mode)}>{modelLabel(mode)}</button>
          ))}
        </div>
        <FeatureTags features={selectedModel?.features} />
        {selectedPrediction && previewVersion === selected ? (
          <div className={`occupancy-result ${selectedPrediction?.occupancy ? 'occupied' : ''}`}>
            <span className="result-icon"><Icon name="lab" size={36} /></span>
            <h3>{selectedPrediction?.label}</h3><small>사용 중일 확률</small>
            <strong>{percent(selectedPrediction?.probability)}</strong>
            <progress max="1" value={selectedPrediction?.probability || 0} aria-label="사용 중일 확률" />
            <small>모델 {previewVersion} · 임계값 {selectedModel?.threshold}</small>
          </div>
        ) : (
          <div className="empty-chart"><Icon name="sliders" size={32} /><p>측정값을 입력하고 예측 버튼을 누르세요.</p></div>
        )}
        <h3>선택 모델의 평가</h3><MetricGrid metrics={selectedModel?.model_metrics} />
        {availableComparison && <p className="small-note">조도 제외 비교는 Light를 0으로 바꾸는 방식이 아니라 해당 입력 없이 따로 학습한 모델입니다.</p>}
      </section>
    </div>
  </>;
}

function ModelManagement() {
  const [tab, setTab] = useState('dashboard');
  const [registry, setRegistry] = useState({ versions: [], current_version: '' });
  const [options, setOptions] = useState(null);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [values, setValues] = useState({ ...PRESETS[0].values });
  const [selected, setSelected] = useState('');
  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [job, setJob] = useState(null);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [activating, setActivating] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [previewVersion, setPreviewVersion] = useState('');
  const [activeMode, setActiveMode] = useState('with_light');
  const handledJob = useRef(new Set());
  const predictionController = useRef(null);
  const busy = starting || activeJob(job);

  async function refresh() {
    const [versions, history] = await Promise.all([request('/versions'), request('/training/jobs')]);
    setRegistry(versions);
    setJobs(history.jobs);
    const running = history.jobs.find(activeJob);
    if (running) {
      setJob(previous => ({ ...running, curves: previous?.id === running.id ? previous.curves : {} }));
    }
    return { versions, history: history.jobs };
  }

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([request('/versions', { signal: controller.signal }), request('/training/jobs', { signal: controller.signal }), request('/training/options', { signal: controller.signal })])
      .then(async ([versions, history, catalog]) => {
        setRegistry(versions);
        setSelected(versions.current_version);
        setJobs(history.jobs);
        setOptions(catalog);
        setConfig(catalog.defaults);
        const recent = history.jobs.find(activeJob) || history.jobs[0];
        setJob(recent || null);
        if (activeJob(recent)) setConfig(recent.config);
        else if (recent) {
          handledJob.current.add(recent.id);
          const detail = await request(`/training/jobs/${recent.id}`, { signal: controller.signal });
          setJob(previous => previous?.id === recent.id ? detail : previous);
        }
      }).catch(cause => { if (cause.name !== 'AbortError') setError(cause.message); });
    return () => { controller.abort(); predictionController.current?.abort(); };
  }, []);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    predictionController.current?.abort();
    setPredictions(null);
    setReport(null);
    setReportLoading(true);
    setActiveMode('with_light');
    request(`/versions/${selected}/training`, { signal: controller.signal }).then(setReport)
      .catch(cause => { if (cause.name !== 'AbortError') setError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setReportLoading(false); });
    return () => controller.abort();
  }, [selected]);

  useEffect(() => {
    predictionController.current?.abort();
    setPredictions(null);
  }, [values, activeMode]);

  useEffect(() => {
    if (!options || typeof config.validation_ratio !== 'number' || config.validation_ratio < 0.1 || config.validation_ratio > 0.4) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      request(`/training/options?validation_ratio=${config.validation_ratio}`, { signal: controller.signal })
        .then(setOptions).catch(cause => { if (cause.name !== 'AbortError') setError(`분할을 확인하지 못했습니다. ${cause.message}`); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [config.validation_ratio, options !== null]);

  useEffect(() => {
    if (!activeJob(job)) return;
    const controller = new AbortController();
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      try {
        const next = await request(`/training/jobs/${job.id}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setJob(next);
        if (!activeJob(next) && !handledJob.current.has(next.id)) {
          handledJob.current.add(next.id);
          const data = await refresh();
          if (next.status === 'completed') {
            setSelected(next.version);
            setMessage(`${next.version} 학습을 완료했습니다. 보고서와 예측 실험에서 확인하세요.`);
          } else if (next.status === 'failed') {
            setError(next.error || '학습에 실패했습니다. 설정을 확인하고 다시 실행하세요.');
          } else {
            setMessage('학습을 중지했습니다. 새 모델 버전은 만들지 않았습니다.');
          }
          setRegistry(data.versions);
        }
      } catch (cause) { if (cause.name !== 'AbortError') setError(`학습 상태를 불러오지 못했습니다. ${cause.message}`); }
      finally { pending = false; }
    };
    const timer = setInterval(poll, 1200);
    poll();
    return () => { controller.abort(); clearInterval(timer); };
  }, [job?.id, job?.status]);

  function update(key, value) {
    setConfig(previous => ({ ...previous, [key]: value }));
  }

  function toggleFeature(key) {
    setConfig(previous => {
      const features = FEATURES.map(item => item.key).filter(name => (
        name === key ? !previous.features.includes(name) : previous.features.includes(name)
      ));
      return { ...previous, features };
    });
  }

  async function train(event) {
    event.preventDefault();
    if (busy || !config.features.length || !options?.data_plan?.ready
        || options.data_plan.validation_ratio !== config.validation_ratio) return;
    setStarting(true);
    setError('');
    setMessage('');
    try {
      const next = await request('/training/jobs', { body: config });
      setJob(next);
      setJobs(previous => [next, ...previous]);
    } catch (cause) {
      setError(cause.message);
      try {
        const data = await refresh();
        const running = data.history.find(activeJob);
        if (running) setJob(running);
      } catch {
        // 중복 요청 등의 최초 오류 메시지를 유지한다.
      }
    } finally { setStarting(false); }
  }

  async function cancel() {
    if (!activeJob(job)) return;
    setCancelling(true);
    setError('');
    try {
      await request(`/training/jobs/${job.id}/cancel`, { method: 'POST' });
      setJob(previous => ({ ...previous, cancel_requested: true }));
    }
    catch (cause) { setError(cause.message); }
    finally { setCancelling(false); }
  }

  async function predict(event) {
    event?.preventDefault();
    if (!selected || reportLoading || !report) return;
    const modelKey = tab === 'training' ? 'with_light' : activeMode;
    const limits = report.input_limits?.[modelKey];
    const inputError = predictionError(values, limits);
    if (inputError) {
      setError(inputError);
      return;
    }
    // API는 여섯 필드를 받는다. 선택 모델이 쓰지 않는 필드는 유효한 기본값으로 채운다.
    const body = { ...PRESETS[0].values };
    for (const key of Object.keys(limits.features)) body[key] = values[key];
    predictionController.current?.abort();
    const controller = new AbortController();
    predictionController.current = controller;
    setPredicting(true);
    setError('');
    try {
      const result = await request(`/compare?version=${selected}`, { body, signal: controller.signal });
      if (!controller.signal.aborted) {
        // 비활성 필드의 대체값으로 계산된 다른 모델의 결과는 표시하지 않는다.
        setPredictions({ [modelKey]: result[modelKey] });
        setPreviewVersion(selected);
      }
    } catch (cause) { if (cause.name !== 'AbortError') setError(cause.message); }
    finally { if (predictionController.current === controller) setPredicting(false); }
  }

  async function activate() {
    setActivating(true);
    setError('');
    try {
      await request(`/versions/${selected}/activate`, { method: 'POST' });
      await refresh();
      setMessage(`${selected}를 기본 예측 모델로 적용했습니다.`);
    }
    catch (cause) { setError(cause.message); }
    finally { setActivating(false); }
  }

  function reuse() {
    const next = report?.config || { ...options.defaults, features: report.models.with_light.features };
    setConfig({ ...next, name: `${selected} 기반 실험`, make_default: false });
    setTab('training');
    setMessage(report.config ? `${selected}의 학습 설정을 불러왔습니다.` : `${selected}의 피처를 불러왔습니다. 저장되지 않은 설정은 기본값으로 시작합니다.`);
  }

  async function inspectJob(item) {
    if (!activeJob(job) || item.id === job.id) {
      try {
        const detail = await request(`/training/jobs/${item.id}`);
        setJob(previous => activeJob(previous) && previous.id !== item.id ? previous : detail);
      } catch (cause) { setError(cause.message); }
    }
    if (item.version) {
      setSelected(item.version);
      setTab('dashboard');
    }
  }

  const reportPanel = <ReportPanel report={report} version={selected} loading={reportLoading}
    onExperiment={() => setTab('experiment')} onReuse={reuse} onActivate={activate}
    isDefault={selected === registry.current_version} activating={activating} />;

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={event => { event.preventDefault(); setTab('dashboard'); }}>
        <span className="brand-icon"><Icon name="lab" size={25} /></span>
        <span><b>공간 학습실</b><small>강의실 모델 워크스페이스</small></span>
      </a>
      <div className="nav-label">워크스페이스</div>
      <nav aria-label="화면 탐색">
        {[
          ['dashboard', '대시보드', 'dashboard'],
          ['experiment', '센서 실험', 'sliders'],
          ['training', '모델 학습', 'train'],
        ].map(([key, label, icon]) => (
          <button key={key} aria-current={tab === key ? 'page' : undefined}
            className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
            <Icon name={icon} /><span>{label}</span>{key === 'training' && busy && <i className="live-dot" />}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="engine-card"><span className="dot green" /><b>PyTorch · CPU</b><small>기본 모델 {registry.current_version || '연결 중'}</small></div>
        <a href="/api/docs" target="_blank" rel="noreferrer"><Icon name="file" size={16} />Swagger UI <Icon name="arrow" size={14} /></a>
        <p>학습 · 평가 · 버전 기록</p>
      </div>
    </aside>
    <main>
      <header className="topbar">
        <span>워크스페이스 <span className="breadcrumb">/</span> {tab === 'dashboard' ? '대시보드' : tab === 'training' ? '모델 학습' : '센서 실험'}</span>
        <div>
          <span className={`connection ${options ? 'online' : ''}`}><i />{options ? '모델 연결됨' : '연결 확인 중'}</span>
          <button className="icon-button" aria-label="데이터 새로고침"
            onClick={() => refresh().then(() => setMessage('모델 기록을 새로고침했습니다.')).catch(cause => setError(cause.message))}>
            <Icon name="refresh" size={17} />
          </button>
        </div>
      </header>
      <div className="workspace">
        <div className="page-heading">
          <div>
            <span className="eyebrow">강의실 사용 여부 · 모델 실험</span>
            <h1>{tab === 'dashboard' ? '모델을 한눈에, 실험을 차곡차곡.' : tab === 'training' ? '다음 모델을 만들어 보세요.' : '센서로 읽는 강의실.'}</h1>
            <p>{tab === 'dashboard' ? '피처와 학습 설정이 성능에 어떤 차이를 만드는지 비교하세요.' : tab === 'training' ? '피처를 선택하고 학습 설정을 조절하면, 새 모델과 보고서를 함께 기록합니다.' : '저장된 모델을 선택하고 측정값 한 건으로 현재 사용 여부를 확인하세요.'}</p>
          </div>
          {tab === 'dashboard' ? (
            <button className="button primary" onClick={() => setTab('training')}><Icon name="play" size={16} />새 모델 학습</button>
          ) : (
            <span className="badge neutral">{tab === 'training' ? 'PyTorch MLP' : `실험 ${selected || '—'}`}</span>
          )}
        </div>
        {error && <div className="alert error" role="alert"><span>{error}</span><button aria-label="오류 닫기" onClick={() => setError('')}>×</button></div>}
        {message && <div className="alert info" role="status"><span>{message}</span><button aria-label="안내 닫기" onClick={() => setMessage('')}>×</button></div>}

        {tab === 'dashboard' && <>
          <Dashboard registry={registry} selected={selected} onSelect={setSelected} job={job} />
          {reportPanel}</>}

        {tab === 'training' && <div className="training-layout">
          <TrainingForm config={config} options={options} previewLimits={report?.input_limits?.with_light}
            values={values} setValues={setValues} busy={busy}
            onSubmit={train} onUpdate={update} onToggleFeature={toggleFeature}
            onReset={() => setConfig(previous => ({ ...options.defaults, name: previous.name, features: previous.features }))} />
          <TrainingMonitor job={job} jobs={jobs} cancelling={cancelling} onCancel={cancel} onInspect={inspectJob}
            onViewReport={version => { setSelected(version); setTab('dashboard'); }} selected={selected}
            report={report} reportLoading={reportLoading} values={values} predicting={predicting} onPredict={predict}
            predictions={predictions} previewVersion={previewVersion} />
        </div>}

        {tab === 'experiment' && <>
          <PredictionExperiment registry={registry} selected={selected} onSelect={setSelected}
            report={report} reportLoading={reportLoading} values={values} setValues={setValues}
            predicting={predicting} onPredict={predict} predictions={predictions} previewVersion={previewVersion}
            activeMode={activeMode} onModeChange={setActiveMode} />
          {reportPanel}
        </>}
        <footer>실제 저장된 가중치로 예측 · 평가 파일 {options?.test_data.rows.toLocaleString() || '—'}건 · 현재 재실 여부 판별</footer>
      </div>
    </main>
  </div>;
}

function App() {
  const [area, setArea] = useState('classrooms');
  return <div className={`service-shell ${area === 'management' ? 'management-area' : ''}`}>
    <header className="service-nav"><a href="#" onClick={event => { event.preventDefault(); setArea('classrooms'); }} className="service-brand"><span aria-hidden="true">▥</span>강의실 지킴이</a>
      <nav aria-label="서비스 영역"><button aria-current={area === 'classrooms' ? 'page' : undefined} onClick={() => setArea('classrooms')}>강의실 현황</button>
        <button aria-current={area === 'management' ? 'page' : undefined} onClick={() => setArea('management')}>모델 관리</button></nav>
      <span className="service-purpose">센서로 확인하는 공간의 현재</span></header>
    {area === 'classrooms' ? <ClassroomBoard /> : <ModelManagement />}
  </div>;
}

// Vite 갱신에서도 같은 React 루트를 사용한다.
const reactRoot = import.meta.hot?.data.reactRoot || createRoot(document.getElementById('root'));
reactRoot.render(<App />);
if (import.meta.hot) import.meta.hot.dispose(data => { data.reactRoot = reactRoot; });
