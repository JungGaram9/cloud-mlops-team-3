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
  compare_light: true, make_default: false, synthetic_generation: null, synthetic_ratio: 0.25, shared_room_split: false,
  pipeline_dataset: null, pipeline_ratio: 0.25 };
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

async function request(path, { signal, body, method, headers } = {}) {
  const response = await fetch(`/api${path}`, { signal, method: method || (body !== undefined ? 'POST' : 'GET'),
    headers: { ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}), ...headers },
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
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
    moon: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.lab}</svg>;
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
const splitLabel = strategy => strategy === 'source_chronological' ? '출처별 시간순'
  : strategy === 'shared_graz_calendar_day_1h_gap' ? '공통 날짜 분할 · 1시간 간격' : '전체 시간순 (기존 방식)';

function dataPlanMatches(plan, config) {
  if (!plan?.selection) return !config.synthetic_generation && !config.pipeline_dataset && !config.shared_room_split && plan?.validation_ratio === config.validation_ratio;
  return ['validation_ratio', 'synthetic_generation', 'synthetic_ratio', 'shared_room_split', 'seed', 'pipeline_dataset', 'pipeline_ratio']
    .every(key => plan.selection[key] === config[key]);
}

function reusableTrainingConfig(stored, defaults) {
  // 이전 버전의 전용 데이터 옵션을 새 학습 API로 다시 전달하지 않는다.
  return Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, Object.hasOwn(stored, key) ? stored[key] : value]));
}

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
    <caption>출처별 행수와 정답 분포</caption><thead><tr><th>출처</th><th>학습 (fit)</th><th>실제 검증</th><th>실제 평가</th></tr></thead>
    <tbody>{names.map(name => <tr key={name}><th title={name}>{SOURCE_LABELS[name] || (name.startsWith('synthetic:') ? `합성 ${name.split(':').at(-1)}` : name)}</th>
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
  return <section className="panel report-panel"><div className="section-heading"><div><h2>{version || '모델 선택'} 학습 보고서</h2></div><Icon name="file" /></div>
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
      {report.data_plan && <div className="data-plan-summary">
        <b>실제 fit {report.data_plan.real_fit_rows?.toLocaleString()}행 + 합성 {report.data_plan.synthetic_fit_rows?.toLocaleString()}행 · 정답 관측 {report.data_plan.log_data?.included_rows || 0}행 포함</b>
        <span>분할 정책: {report.data_plan.partition_policy || report.data_plan.split_strategy}</span>
        {report.data_plan.synthetic && <><span>생성 {report.data_plan.synthetic.generation_id}</span>
          <span>실제 fit 대비 {percent(report.data_plan.synthetic.actual_ratio_to_real_fit)} · 전체 fit 중 합성 {percent(report.data_plan.synthetic.fraction_of_combined_fit)}</span>
          <small className="dataset-identity">manifest SHA-256 {report.data_plan.synthetic.manifest_sha256}</small></>}
        {report.data_plan.warnings?.map(value => <p className="data-warning" key={value}>{value}</p>)}
      </div>}
      <SourceSplitTable fit={primaryModel.training_split} validation={primaryModel.validation_split} test={primaryModel.test_data} />
      <LossChart models={report.models} emptyMessage={report.legacy ? '기존 버전에는 손실 곡선이 저장되어 있지 않습니다.' : undefined} />
      <div className="section-heading"><h3>하이퍼파라미터</h3><span className="small-note">선택 모델 기준</span></div>
      <dl className="parameter-grid">{parameters.map(([label, value]) => (
        <div key={label}><dt>{label}</dt><dd>{value ?? '기록 없음'}</dd></div>
      ))}</dl>
      <p className="small-note">저장된 분할 정책과 실제 평가 CSV 해시를 함께 확인하세요. 같은 평가 자료로 반복 실험한 결과는 독립적인 최종 검증이 아닙니다.</p>
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
        <div><h2>버전마다 달라지는 정확도</h2></div>
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
        <div><h2>모델 버전</h2></div>
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

function TrainingForm({ config, options, previewLimits, values, setValues, busy, onSubmit, onUpdate, onToggleFeature, onReset, onRefreshData }) {
  const plan = options?.data_plan;
  const planMatches = dataPlanMatches(plan, config);
  const datasets = options?.synthetic_datasets || [];
  return (
    <form className="training-form" onSubmit={onSubmit}>
      <fieldset disabled={busy || !options}>
        <section className="panel">
          <div className="section-heading">
            <div><h2>1. 이름과 데이터</h2></div>
            <span className="badge neutral">{config.shared_room_split || config.synthetic_generation ? '공통 날짜 분할 · 1시간 간격' : '출처별 시간순 분할'}</span>
          </div>
          <p className="small-note">실제 CSV를 먼저 fit·검증으로 분할하고, 선택한 합성 데이터는 fit에만 추가합니다. 출처는 입력 피처에 추가하지 않습니다.</p>
          <div className="synthetic-controls">
            <button type="button" className="text-button" onClick={onRefreshData}>데이터 다시 확인</button>
            {config.pipeline_dataset && <div className="data-plan-summary"><b>정답 연결 관측 데이터셋 선택됨</b>
              <small className="dataset-identity">{config.pipeline_dataset}</small>
              <RangeControl label="정답 관측 추가량 / 실제 fit" value={config.pipeline_ratio * 100} min={1} max={100} step={1} suffix="%"
                hint="CSV 보충과 합쳐 실제 fit의 100%를 넘지 않으며 같은 행은 한 번만 사용"
                onChange={value => onUpdate('pipeline_ratio', value / 100)} />
              <button type="button" className="text-button" onClick={() => onUpdate('pipeline_dataset', null)}>관측 데이터셋 제외</button>
            </div>}
            <label className="field-label" htmlFor="synthetic-generation">학습 데이터 구성</label>
            <select id="synthetic-generation" className="wide-input" value={config.synthetic_generation || ''}
              onChange={event => onUpdate('synthetic_generation', event.target.value || null)}>
              <option value="">실제 CSV만 사용 (기본)</option>
              {config.synthetic_generation && !datasets.some(item => item.generation_id === config.synthetic_generation) &&
                <option value={config.synthetic_generation} disabled>{config.synthetic_generation} · 현재 없음</option>}
              {datasets.map(item => <option key={item.generation_id} value={item.generation_id} disabled={!item.available}>
                {item.generation_id} · {item.available ? `학습 후보 ${item.eligible_rows.toLocaleString()}행` : '사용 불가'}
              </option>)}
            </select>
            {!datasets.length && <p className="small-note">생성 데이터가 없습니다. 생성 스크립트를 실행한 뒤 데이터 다시 확인을 눌러주세요.</p>}
            {config.synthetic_generation && <RangeControl label="합성 추가량 / 실제 fit 행수" value={config.synthetic_ratio * 100}
              min={1} max={100} step={1} suffix="%" hint="25% = 실제 fit 100행당 합성 최대 25행. 후보 부족 시 가능한 만큼만 사용"
              onChange={value => onUpdate('synthetic_ratio', value / 100)} />}
            <Switch label="Graz 두 방 공통 시간 분할" checked={config.shared_room_split || Boolean(config.synthetic_generation || config.pipeline_dataset)}
              disabled={Boolean(config.synthetic_generation || config.pipeline_dataset)} onChange={value => onUpdate('shared_room_split', value)}
              description="같은 날짜·시각의 두 방을 함께 나누고 경계 앞 1시간은 제외합니다. 합성 보충 시 필수이며, 실제 데이터만 비교할 때도 켜두세요." />
            {datasets.some(item => !item.available) && <details className="dataset-exclusions"><summary>사용 불가 생성물과 제외 이유</summary>
              {datasets.filter(item => !item.available).map(item => <p key={item.generation_id}>{item.generation_id}: {item.error}</p>)}
            </details>}
          </div>
          {planMatches ? <><SourceSplitTable fit={plan.fit} validation={plan.validation} test={plan.test} />
            {Number.isInteger(plan.real_fit_rows) && <div className="data-plan-summary" role="status">
              <b>실제 fit {plan.real_fit_rows.toLocaleString()}행 + 합성 {plan.synthetic_fit_rows.toLocaleString()}행</b>
              <span>CSV 보충 {plan.csv_synthetic_fit_rows || 0}행 · 정답 관측 {plan.log_data?.included_rows || 0}행 · 중복 제외 {plan.log_data?.duplicates_excluded || 0}행</span>
              <span>실제 검증 {plan.validation.rows.toLocaleString()}행 · 고정 실제 평가 {plan.test.rows.toLocaleString()}행 · 경계 제외 {plan.real_excluded_rows.toLocaleString()}행</span>
              {plan.synthetic && <><span>실제 fit 대비 {percent(plan.synthetic.actual_ratio_to_real_fit)} · 전체 fit 중 합성 {percent(plan.synthetic.fraction_of_combined_fit)}</span>
                <span>선택 seed {plan.synthetic.sampling_seed} · train/train_only 방만 보충 · validation/test 방 제외</span>
                <small className="dataset-identity">manifest SHA-256 {plan.synthetic.manifest_sha256}</small>
                <details><summary>생성 출처와 방별 파티션</summary>{plan.synthetic.rooms.map(room => <p key={room.room}>
                  {room.room} · {room.partition} · {room.rows.toLocaleString()}행 · 보정 {SOURCE_LABELS[room.calibration_source] || room.calibration_source}
                </p>)}</details></>}
            </div>}
            {plan.errors.map(value => <p className="inline-error" key={value}>{value}</p>)}
            {plan.warnings.map(value => <p className="data-warning" key={value}>{value}</p>)}
          </> : <p className="small-note" role="status">설정에 맞는 분할을 확인하는 중입니다.</p>}
          <p className="small-note">기존 예측 로그는 실제 정답이 없어 학습 0행입니다. 별도 검증된 가상 현장 정답만 관측 보충에 사용하며 모델 예측값을 정답으로 쓰지 않습니다.</p>
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
            <div><h2>2. 무엇을 학습할까요?</h2></div>
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
            <div><h2>3. 모델의 학습 방식을 조절하세요.</h2></div>
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
  const unreadableCount = jobs.filter(item => item.history_error).length;
  const primaryModel = report?.models.with_light;
  const primaryPrediction = predictions?.with_light;
  const inputError = predictionError(values, report?.input_limits?.with_light);
  return (
    <aside className="training-aside">
      <section className="panel live-panel">
        <div className="section-heading">
          <div><h2>{job ? STATUS[job.status] : '학습을 기다리고 있어요.'}</h2></div>
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
          {job.data_plan && <p className="small-note">제출 입력: 실제 fit {job.data_plan.real_fit_rows?.toLocaleString()}행 + 합성 {job.data_plan.synthetic_fit_rows?.toLocaleString()}행
            {job.data_plan.synthetic ? ` · ${job.data_plan.synthetic.generation_id}` : ''}</p>}
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
        <div className="section-heading"><div><h3>{selected || '—'} 예측</h3></div></div>
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
        {unreadableCount > 0 && <p className="inline-error" role="alert">
          학습 이력 {unreadableCount}개를 읽지 못했습니다. 해당 원본 파일은 보존되며 정상 이력은 계속 조회할 수 있습니다.
        </p>}
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

function PipelinePanel({ options, registry, onRefresh, onTraining }) {
  const [catalog, setCatalog] = useState({ runs: [], datasets: [] });
  const [generation, setGeneration] = useState('');
  const [version, setVersion] = useState(registry.current_version || '');
  const [candidateVersion, setCandidateVersion] = useState(registry.current_version || '');
  const [mode, setMode] = useState('with_light');
  const [candidateMode, setCandidateMode] = useState('with_light');
  const [seed, setSeed] = useState(42);
  const [maxRows, setMaxRows] = useState(300);
  const [validationRatio, setValidationRatio] = useState(.2);
  const [observationMode, setObservationMode] = useState('normal');
  const [detail, setDetail] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [generationJob, setGenerationJob] = useState(null);
  const stop = useRef(false);
  const execution = useRef(null);
  async function refresh() { setCatalog(await request('/pipeline')); }
  useEffect(() => {
    stop.current = false;
    refresh().catch(error => setNotice(error.message));
    return () => { stop.current = true; };
  }, []);
  async function perform(action) {
    setBusy(true); setNotice('');
    try { await action(); await refresh(); } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }
  async function inspect(id) {
    await perform(async () => { setDetail(await request(`/pipeline/runs/${id}`)); setComparison(null); });
  }
  async function watchGeneration(job) {
    stop.current = false;
    setGenerationJob(job);
    while (!stop.current && ['queued', 'running'].includes(job.status)) {
      await new Promise(resolve => setTimeout(resolve, 750));
      if (stop.current) return;
      job = await request(`/pipeline/generations/${job.id}`);
      setGenerationJob(job);
    }
    if (job.status === 'completed') {
      setGeneration(job.generation_id); onRefresh();
      setNotice(`${job.generation_id} ${job.reused_output ? '완료 산출물 검증·재사용' : '생성 완료'}. 데이터 다시 확인 후 선택할 수 있습니다.`);
    } else if (job.status === 'failed') throw new Error(job.error || '생성 작업에 실패했습니다.');
  }
  async function advance(id, compare = false) {
    execution.current = id;
    await perform(async () => {
      stop.current = false;
      await request(`/pipeline/runs/${id}/control?action=resume`, { method: 'POST' });
      let status;
      do {
        status = await request(`/pipeline/runs/${id}/step`, { method: 'POST' });
        if (compare) setComparison(await request(`/pipeline/comparisons/${id}`));
        else setDetail(await request(`/pipeline/runs/${id}`));
      } while (!stop.current && !['completed', 'cancelled', 'failed'].includes(status.status));
      if (status.error) setNotice(status.error);
    });
    execution.current = null;
  }
  const run = detail?.run;
  const controls = <><label>관측 선택 seed<input type="number" min="0" max="2147483647" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label>
    <label>고정 관측 최대 행수<input type="number" min="1" max="3000" value={maxRows} onChange={event => setMaxRows(Number(event.target.value))} /></label></>;
  const score = result => result?.available ? <>
    <span>정답 {result.labelled_rows}건 · 무정답 {result.unlabelled_rows}건 · 오판 {result.errors}건</span>
    <span>정확도 {percent(result.accuracy)} · 정밀도 {percent(result.precision)} · 재현율 {percent(result.recall)} · F1 {decimal(result.f1)}</span>
    <small>혼동행렬 {JSON.stringify(result.confusion_matrix)}{result.single_class ? ' · 단일 클래스. 분모가 없는 지표는 —로 표시합니다.' : ''}</small>
  </> : <span>아직 정답 연결 예측이 없어 평가할 수 없습니다.</span>;
  return <div className="pipeline-layout">
    <section className="panel"><h2>1. 가상 현장 데이터 준비</h2>
      <p className="small-note">생성기는 현장 센서·독립 인원수 정답의 역할입니다. 실제 fit만 보정에 사용하며 실제 test는 읽지 않습니다. 이 과정은 정답을 이용한 지도학습 재학습이며 강화학습은 아닙니다.</p>
      <fieldset disabled={busy}><div className="pipeline-controls">{controls}
        <label>실제 검증 비율<input type="number" min="0.1" max="0.4" step="0.01" value={validationRatio} onChange={event => setValidationRatio(Number(event.target.value))} /></label>
        <label>관측 센서 오류<select value={observationMode} onChange={event => setObservationMode(event.target.value)}>
          <option value="normal">일반 · 가정 확률 3% / 유형·방·날짜</option><option value="stress">스트레스 · 모든 오류 유형 강제 포함</option><option value="off">오류 계층 끄기</option></select></label>
      </div><button className="button secondary" onClick={() => perform(async () => {
        await watchGeneration(await request('/pipeline/generations', { body: { seed, validation_ratio: validationRatio, observation_mode: observationMode } }));
      })}>fit-only 현장 데이터 생성 · 3개 방 / 7일</button>
      <button className="text-button" onClick={onRefresh}>데이터 다시 확인</button></fieldset>
      {generationJob && <p role="status" className="small-note">생성 작업 {generationJob.id.slice(0, 12)} · {generationJob.status}
        {generationJob.error ? ` · ${generationJob.error}` : ''} · 페이지를 나가도 접수된 생성 작업은 계속됩니다.</p>}
      {(catalog.generation_jobs || []).map(job => <button key={job.id} className="text-button" disabled={busy}
        onClick={() => perform(async () => watchGeneration(await request(`/pipeline/generations/${job.id}`)))}>
        생성 작업 상태 보기 · {job.id.slice(0, 12)} · {job.status}</button>)}
      <p className="data-warning">기존 canonical 전체 보정 생성물은 탐색용 예측·분석만 허용합니다. 피드백 데이터 확정에는 fit-only 생성물이 필요합니다. 모든 지표는 반복 개발용 시뮬레이션 지표입니다.</p>
      <p className="small-note">원시 관측은 별도 보존합니다. 결측·비유한값·범위 밖 값·수집 중단은 보간 없이 제외하며, 유효 범위의 고정값·스파이크·편향은 유지합니다. 일반 오류 확률과 지속 시간은 실제 측정된 고장률이 아닌 가정입니다.</p>
    </section>
    <section className="panel"><h2>2. 모델과 관측 고정 → 예측</h2>
      <fieldset disabled={busy}><div className="pipeline-controls">
        <label>생성물<select value={generation} onChange={event => setGeneration(event.target.value)}><option value="">생성 ID 선택</option>
          {(options?.synthetic_datasets || []).map(item => <option key={item.generation_id} value={item.generation_id} disabled={!item.available}>
            {item.generation_id} · {item.calibration_scope === 'real_fit_only' ? 'fit-only' : '기존 보정 / 제한 있음'}</option>)}</select></label>
        <label>기준 모델<select value={version} onChange={event => setVersion(event.target.value)}><option value="">모델 선택</option>
          {registry.versions.map(item => <option key={item.version} value={item.version}>{item.version} · {item.name}</option>)}</select></label>
        <label>기준 모델 종류<select value={mode} onChange={event => setMode(event.target.value)}><option value="with_light">선택 피처 모델</option><option value="without_light">조도 제외 비교 모델 · 저장된 버전만</option></select></label>
      </div><button className="button primary" disabled={!generation || !version} onClick={() => perform(async () => {
        const created = await request('/pipeline/runs', { body: { generation_id: generation, model_version: version, mode, seed, max_rows: maxRows } });
        setDetail(await request(`/pipeline/runs/${created.id}`)); setComparison(null);
      })}>관측 묶음 고정</button></fieldset>
      <div className="pipeline-runs">{catalog.runs.filter(item => !item.baseline).map(item => <button key={item.id} className="text-button" disabled={busy} onClick={() => inspect(item.id)}>
        {item.config.model_version} · {item.processed}/{item.rows} · {item.status} · {item.id.slice(0, 10)}</button>)}</div>
      {run && <div className="data-plan-summary"><b>{run.config.model_version} · {run.processed}/{run.rows}행 · {run.status}</b>
        <small className="dataset-identity">모델 SHA-256 {run.model_fingerprint}</small><small className="dataset-identity">관측 SHA-256 {run.observation_hash}</small>
        <small>정답은 가상 인원수 &gt; 0. 인원수·정답·사건 ID는 모델 입력에 포함하지 않습니다.</small>
        {run.sampling && <><small>추론 전 불능 관측 제외 {run.sampling.excluded_before_replay}행. 희귀 유형을 우선 포함한 표본이므로 자연 발생 비율의 성능이 아닙니다.</small>
          <details><summary>파티션·유형별 포함 / 미포함 행수와 제외 사유</summary><pre>{JSON.stringify(run.sampling, null, 2)}</pre></details></>}
        <button className="button primary" disabled={busy || run.status === 'completed'} onClick={() => advance(run.id)}>예측 시작 / 재개</button>
        {busy && execution.current && <button className="button secondary" onClick={async () => { stop.current = true;
          await request(`/pipeline/runs/${execution.current}/control?action=cancel`, { method: 'POST' }).catch(error => setNotice(error.message)); }}>예측 중지</button>}
      </div>}
    </section>
    {detail && <section className="panel"><h2>3. 정답 연결과 오판 분석</h2><div className="data-plan-summary">{score(detail.overall)}</div>
      <div className="table-scroll"><table><thead><tr><th>상황 / 파티션</th><th>정답 행수</th><th>오판</th><th>정확도</th><th>F1</th></tr></thead>
        <tbody>{Object.entries(detail.groups).map(([key, value]) => <tr key={key}><td>{key}</td><td>{value.labelled_rows}</td><td>{value.errors}</td><td>{percent(value.accuracy)}</td><td>{decimal(value.f1)}</td></tr>)}</tbody></table></div>
      <details className="pipeline-evidence"><summary>오판 사례 · 관측 ID / 예측 확률 / 피처 / 별도 가상 정답 (최대 50건)</summary><pre>{JSON.stringify(detail.mistakes, null, 2)}</pre></details>
      <button className="button secondary" disabled={busy || run.status !== 'completed' || run.generation.calibration_scope !== 'real_fit_only'} onClick={() => perform(async () => {
        const snapshot = await request(`/pipeline/runs/${run.id}/dataset`, { body: { max_rows: maxRows, seed } });
        setNotice(`데이터셋 ${snapshot.id.slice(0, 12)} 확정. 일반 표본 ${snapshot.uniform_rows}행, 오류 보충 ${snapshot.error_boost_rows}행.`);
      })}>학습용 정답 관측 데이터셋 확정</button>
      <p className="small-note">train/train_only 관측만 사용합니다. 절반 이상은 일반 표본이며 나머지는 오판을 우선 보충합니다. validation/test 오판은 학습에 넣지 않습니다.</p>
    </section>}
    <section className="panel"><h2>4. 확정 데이터로 기존 학습 작업 열기</h2><p className="small-note">학습 화면에서 실제·CSV·정답 관측 행수와 중복 제외량을 확인한 뒤 직접 학습 시작을 누릅니다. 기본 모델은 자동 교체하지 않습니다.</p>
      {catalog.datasets.map(item => <div className="data-plan-summary" key={item.id}><b>{item.rows}행 · {item.id.slice(0, 12)}</b>
        <span>일반 표본 {item.uniform_rows}행 / 오류 보충 {item.error_boost_rows}행 · seed {item.seed}</span>
        {item.coverage && <details><summary>학습 후보의 유형별 포함 / 미포함</summary><pre>{JSON.stringify(item.coverage, null, 2)}</pre></details>}
        <button className="button secondary" disabled={busy} onClick={() => onTraining(item)}>재학습 설정으로 가져오기</button></div>)}
    </section>
    {run && <section className="panel"><h2>5. 같은 고정 관측으로 후보 모델 비교</h2>
      <p className="small-note">기준 실행의 validation/test 관측을 그대로 사용합니다. 최종 실제 평가 CSV로 후보를 선택하지 않습니다. 양수인 오류 변화는 성능 저하이며 숨기지 않습니다.</p>
      {catalog.runs.filter(item => item.baseline === run.id).map(item => <button key={item.id} className="text-button" disabled={busy}
        onClick={() => perform(async () => setComparison(await request(`/pipeline/comparisons/${item.id}`)))}>
        {item.config.model_version} 비교 이어보기 · {item.processed}/{item.rows} · {item.status}</button>)}
      <label>후보 모델<select value={candidateVersion} onChange={event => setCandidateVersion(event.target.value)}>
        {registry.versions.map(item => <option key={item.version} value={item.version}>{item.version} · {item.name}</option>)}</select></label>
      <label>후보 모델 종류<select value={candidateMode} onChange={event => setCandidateMode(event.target.value)}><option value="with_light">선택 피처 모델</option><option value="without_light">조도 제외 비교 모델 · 저장된 버전만</option></select></label>
      <button className="button secondary" disabled={busy || run.status !== 'completed' || !candidateVersion} onClick={() => perform(async () => {
        const created = await request(`/pipeline/runs/${run.id}/compare`, { body: { model_version: candidateVersion, mode: candidateMode } });
        setComparison(await request(`/pipeline/comparisons/${created.id}`));
      })}>동일 관측 비교 준비</button>
      {comparison && <><button className="button primary" disabled={busy || comparison.complete} onClick={() => advance(comparison.run.id, true)}>후보 예측 / 비교 진행</button>
        <div className="data-plan-summary"><b>{comparison.complete ? '비교 완료' : '진행 중인 공통 관측만 집계'}</b><span>기준 모델</span>{score(comparison.baseline.overall)}<span>후보 모델</span>{score(comparison.candidate.overall)}</div>
        {Object.entries(comparison.error_changes).map(([key, value]) => <p className={value > 0 ? 'data-warning' : 'small-note'} key={key}>{key}: 오류 {value > 0 ? '+' : ''}{value}건</p>)}
        <p className="small-note">{comparison.note}</p></>}
    </section>}
    {notice && <p className="notice" role="status">{notice}</p>}
  </div>;
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
  const [dataRevision, setDataRevision] = useState(0);
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
        if (activeJob(recent)) setConfig({ ...catalog.defaults, ...recent.config });
        else if (recent && !recent.history_error) {
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
    if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 2147483647) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      const query = new URLSearchParams({ validation_ratio: config.validation_ratio, synthetic_ratio: config.synthetic_ratio,
        shared_room_split: config.shared_room_split, seed: config.seed, pipeline_ratio: config.pipeline_ratio });
      if (config.synthetic_generation) query.set('synthetic_generation', config.synthetic_generation);
      if (config.pipeline_dataset) query.set('pipeline_dataset', config.pipeline_dataset);
      request(`/training/options?${query}`, { signal: controller.signal })
        .then(setOptions).catch(cause => { if (cause.name !== 'AbortError') setError(`분할을 확인하지 못했습니다. ${cause.message}`); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [config.validation_ratio, config.synthetic_generation, config.synthetic_ratio, config.shared_room_split, config.seed, config.pipeline_dataset, config.pipeline_ratio, options !== null, tab, dataRevision]);

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
    setConfig(previous => ({ ...previous, [key]: value,
      ...(key === 'synthetic_generation' && value ? { shared_room_split: true } : {}) }));
  }

  function refreshData() {
    setOptions(previous => previous ? { ...previous, data_plan: null } : previous);
    setDataRevision(previous => previous + 1);
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
    if (busy || !config.features.length || !options?.data_plan?.ready || !dataPlanMatches(options.data_plan, config)) return;
    setStarting(true);
    setError('');
    setMessage('');
    try {
      const next = await request('/training/jobs', { body: config,
        headers: options.data_plan.fingerprint ? { 'X-Data-Plan-SHA256': options.data_plan.fingerprint } : {} });
      setJob(next);
      setJobs(previous => [next, ...previous]);
    } catch (cause) {
      setError(cause.message);
      refreshData();
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
    const stored = report?.config || { features: report.models.with_light.features };
    const next = reusableTrainingConfig(stored, options.defaults);
    setConfig({ ...next, name: `${selected} 기반 실험`, make_default: false });
    setTab('training');
    const legacyOptions = Object.keys(stored).filter(key => !Object.hasOwn(options.defaults, key));
    setMessage(legacyOptions.length ? `${selected}의 설정을 불러왔습니다. 이전 전용 옵션(${legacyOptions.join(', ')})은 제외하고 현재 통합 CSV의 출처별 분할을 사용합니다.`
      : report.config ? `${selected}의 학습 설정을 불러왔습니다.` : `${selected}의 피처를 불러왔습니다. 저장되지 않은 설정은 기본값으로 시작합니다.`);
  }

  async function inspectJob(item) {
    if (!activeJob(job) || item.id === job.id) {
      try {
        const detail = item.history_error ? item : await request(`/training/jobs/${item.id}`);
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
          ['pipeline', '현장 개선 파이프라인', 'refresh'],
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
        <span>워크스페이스 <span className="breadcrumb">/</span> {tab === 'pipeline' ? '현장 개선 파이프라인' : tab === 'dashboard' ? '대시보드' : tab === 'training' ? '모델 학습' : '센서 실험'}</span>
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
            <h1>{tab === 'pipeline' ? '관측에서 다음 모델까지.' : tab === 'dashboard' ? '모델을 한눈에, 실험을 차곡차곡.' : tab === 'training' ? '다음 모델을 만들어 보세요.' : '센서로 읽는 강의실.'}</h1>
            <p>{tab === 'pipeline' ? '고정한 가상 현장 정답으로 오판을 분석하고, 데이터를 확정한 뒤 같은 관측으로 비교합니다.' : tab === 'dashboard' ? '피처와 학습 설정이 성능에 어떤 차이를 만드는지 비교하세요.' : tab === 'training' ? '피처를 선택하고 학습 설정을 조절하면, 새 모델과 보고서를 함께 기록합니다.' : '저장된 모델을 선택하고 측정값 한 건으로 현재 사용 여부를 확인하세요.'}</p>
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

        {tab === 'pipeline' && <PipelinePanel options={options} registry={registry} onRefresh={refreshData}
          onTraining={snapshot => { setConfig(previous => ({ ...previous, pipeline_dataset: snapshot.id,
            synthetic_generation: null, shared_room_split: true, validation_ratio: snapshot.validation_ratio, make_default: false }));
            setTab('training'); refreshData(); }} />}

        {tab === 'training' && <div className="training-layout">
          <TrainingForm config={config} options={options} previewLimits={report?.input_limits?.with_light}
            values={values} setValues={setValues} busy={busy}
            onSubmit={train} onUpdate={update} onToggleFeature={toggleFeature}
            onReset={() => setConfig(previous => ({ ...options.defaults, name: previous.name, features: previous.features }))}
            onRefreshData={refreshData} />
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

const THEME_KEY = 'classroom-theme';

function ThemeSwitch() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  function choose(next) {
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch { }
  }
  return <div className="theme-switch" role="group" aria-label="화면 밝기">
    {[['light', '라이트', 'sun'], ['dark', '다크', 'moon']].map(([value, label, icon]) =>
      <button type="button" key={value} aria-pressed={theme === value} aria-label={`${label} 모드`} title={`${label} 모드`} onClick={() => choose(value)}><Icon name={icon} size={16} /><span>{label}</span></button>)}
  </div>;
}

function App() {
  const [area, setArea] = useState('classrooms');
  return <div className={`service-shell ${area === 'management' ? 'management-area' : ''}`}>
    <header className="service-nav"><a href="#" onClick={event => { event.preventDefault(); setArea('classrooms'); }} className="service-brand"><img src="/icon.svg" width="26" height="26" alt="" />강의실 지킴이</a>
      <nav aria-label="서비스 영역"><button aria-current={area === 'classrooms' ? 'page' : undefined} onClick={() => setArea('classrooms')}>강의실 현황</button>
        <button aria-current={area === 'management' ? 'page' : undefined} onClick={() => setArea('management')}>모델 관리</button></nav>
      <span className="service-purpose">센서로 확인하는 공간의 현재</span><ThemeSwitch /></header>
    {area === 'classrooms' ? <ClassroomBoard /> : <ModelManagement />}
  </div>;
}

// Vite 갱신에서도 같은 React 루트를 사용한다.
const reactRoot = import.meta.hot?.data.reactRoot || createRoot(document.getElementById('root'));
reactRoot.render(<App />);
if (import.meta.hot) import.meta.hot.dispose(data => { data.reactRoot = reactRoot; });
