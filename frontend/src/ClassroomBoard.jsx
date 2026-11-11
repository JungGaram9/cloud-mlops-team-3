import React, { useEffect, useRef, useState } from 'react';

const STATES = { occupied: '사용 중', vacant: '비어 있음', waiting: '수신 대기', stale: '연결 확인 필요', unavailable: '예상 상태 확인 불가' };
const SENSOR_LABELS = { Temperature: ['실내 온도', '°C'], Humidity: ['습도', '%'], Light: ['실내 밝기', 'lux'], CO2: ['이산화탄소', 'ppm'] };
const SENSOR_ICONS = { Light: 'sun', CO2: 'cloud', Temperature: 'temp', Humidity: 'drop' };
const FILTERS = [['all', '전체 강의실', 'grid'], ['light_check', '불 확인 필요', 'bulb'], ['occupied', '사용 중 예상', 'users'],
  ['vacant', '비어 있음 예상', 'disc'], ['attention', '점검 필요', 'alert']];
const KIND_ICONS = { 'light-check': 'bulb', connection: 'alert', waiting: 'hourglass', paused: 'pause', monitoring: 'clock', occupied: 'users', vacant: 'disc' };
const LEGEND = [['light-check', '불 확인 필요', '밝은데 사람 흔적이 없는 상태가 이어짐'], ['occupied', '사용 중 예상', '모델이 사람이 있다고 예측'],
  ['vacant', '비어 있음 예상', '모델이 빈 방으로 예측'], ['monitoring', '확인 중', '밝기·CO₂ 조건이 이어지는지 지켜보는 중'],
  ['connection', '점검 필요', '수신 지연·수신 대기·판별 불가·측정 시각 이상. 지금 값을 믿기 어려운 방']];
const CONNECTION_LABELS = { stale: '수신 지연', unavailable: '판별 불가' };
const DAYS = ['월', '화', '수', '목', '금', '토', '일'];
const hhmm = (value, timeZone) => value ? new Intl.DateTimeFormat('en-GB', { timeZone, hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value)) : '—';
const clock = value => hhmm(value, 'UTC');
const receivedTime = value => value ? hhmm(value, 'Asia/Seoul') : '아직 없음';
const measured = (value, source) => source === 'simulation' ? clock(value) : hhmm(value, 'Asia/Seoul');
const reading = (value, digits = 0) => value == null ? '—' : (Math.round(value * 10 ** digits) / 10 ** digits).toLocaleString('ko-KR');
const VISIT_FILTERS = ['light_check', 'attention'];
const USAGE_FILTERS = ['all', 'occupied', 'vacant'];

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { ...options,
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined });
  let data;
  try { data = await response.json(); }
  catch { throw new Error('현황 서버의 응답을 확인하지 못했습니다.'); }
  if (!response.ok) throw new Error(data.error || '현황을 가져오지 못했습니다.');
  return data;
}

function Icon({ name, size = 18 }) {
  const paths = {
    grid: <path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" />,
    bulb: <path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z" />,
    users: <path d="M16 20v-1.5a3.5 3.5 0 0 0-3.5-3.5h-5A3.5 3.5 0 0 0 4 18.5V20M10 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM20 20v-1.5a3.5 3.5 0 0 0-2.5-3.3M15.5 4.2a3.5 3.5 0 0 1 0 6.6" />,
    circle: <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z" />,
    disc: <><path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z" /><circle cx="12" cy="12" r="5" fill="currentColor" stroke="none" /></>,
    clock: <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2" />,
    alert: <path d="M12 4 3 20h18L12 4zM12 10v4M12 17h.01" />,
    pause: <path d="M8 5v14M16 5v14" />,
    play: <path d="M8 5l11 7-11 7V5z" />,
    restart: <path d="M4 12a8 8 0 1 0 2.5-5.8M4 4v4.5h4.5" />,
    search: <path d="M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-4-4" />,
    sun: <path d="M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />,
    cloud: <path d="M7 18a4 4 0 0 1-.5-8A5.5 5.5 0 0 1 17 9.5a4.2 4.2 0 0 1 0 8.5H7z" />,
    temp: <path d="M10 13.5V5a2 2 0 1 1 4 0v8.5a4 4 0 1 1-4 0z" />,
    drop: <path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z" />,
    hourglass: <path d="M7 3h10M7 21h10M8 3v3a4 4 0 0 0 1.6 3.2L12 12l2.4-2.8A4 4 0 0 0 16 6V3M8 21v-3a4 4 0 0 1 1.6-3.2L12 12l2.4 2.8A4 4 0 0 1 16 18v3" />,
  };
  return <svg className="glyph" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.circle}</svg>;
}

function Badge({ room, disconnected = false }) {
  const view = roomPresentation(room, disconnected);
  return <span className={`room-status k-${view.kind}`}><Icon name={KIND_ICONS[view.kind] || 'alert'} size={15} />{view.label}</span>;
}

export function roomPresentation(room, disconnected = false) {
  const check = room.light_check;
  const offline = disconnected || room.offline;
  const demoStopped = !offline && room.source === 'simulation' && room.simulation_paused && room.status === 'stale';
  if (!demoStopped && (offline || ['stale', 'unavailable'].includes(room.status) || check?.state === 'stale')) {
    return { kind: 'connection', label: offline ? '서버 연결 끊김' : CONNECTION_LABELS[room.status] || '측정 시각 이상', priority: 1,
      message: offline ? '현황 서버에 연결되지 않습니다. 화면의 값은 마지막으로 받은 것입니다.'
        : room.status === 'stale' ? `${receivedTime(room.latest?.received_at)} 이후 새 측정이 없습니다. 센서 전원과 네트워크를 확인해 주세요.`
          : room.status === 'unavailable' ? `${room.message || '저장 모델로 판별하지 못했습니다.'} 모델 관리 담당자에게 알려 주세요.`
            : '센서가 보낸 측정 시각이 지금과 맞지 않습니다. 센서의 시계 설정을 확인해 주세요.' };
  }
  if (demoStopped || room.simulation_paused || check?.state === 'paused') {
    return { kind: 'paused', label: '시연 정지', priority: 4, message: '마지막 측정값입니다. 현재 실시간 상태가 아닙니다.' };
  }
  if (room.status === 'waiting') {
    return { kind: 'waiting', label: '수신 대기', priority: 2, message: '아직 받은 센서 측정이 없습니다. 센서가 켜져 있는지 확인해 주세요.' };
  }
  if (!check || ['waiting', 'unavailable'].includes(check.state)) {
    return { kind: 'monitoring', label: '확인 중', priority: 2,
      message: check?.message || '새로운 센서 측정을 기다리고 있습니다.' };
  }
  if (check.requires_check && check.state === 'attention') {
    return { kind: 'light-check', label: '불 확인 필요', priority: 0, message: check.message };
  }
  if (check.state === 'monitoring') {
    return { kind: 'monitoring', label: '확인 중', priority: 2, message: check.message };
  }
  return { kind: room.status, label: `${STATES[room.status] || STATES.unavailable} 예상`, priority: 3, message: '현재 불 확인 참고 조건에 해당하지 않습니다.' };
}

export function filterRooms(rooms, { floor = 'all', filter = 'all', search = '' } = {}) {
  return rooms.filter(room => {
    const view = roomPresentation(room);
    const usageAvailable = !['connection', 'paused'].includes(view.kind) && room.status !== 'waiting';
    return (floor === 'all' || String(room.floor) === floor)
      && (filter === 'all' || filter === 'light_check' && view.kind === 'light-check'
        || filter === 'attention' && (view.kind === 'connection' || room.status === 'waiting')
        || ['occupied', 'vacant'].includes(filter) && usageAvailable && room.status === filter)
      && `${room.name} ${room.id}`.toLowerCase().includes(search.trim().toLowerCase());
  }).sort((first, second) => roomPresentation(first).priority - roomPresentation(second).priority
    || first.id.localeCompare(second.id, undefined, { numeric: true }));
}

const duration = seconds => {
  const total = Math.max(0, Math.floor(seconds || 0));
  return total >= 60 ? `${Math.floor(total / 60)}분${total % 60 ? ` ${total % 60}초` : ''}` : `${total}초`;
};


function Measure({ value, unit, digits = 0 }) {
  return value == null ? <b className="empty-value">—</b> : <b>{reading(value, digits)}<i>{unit}</i></b>;
}

export function RoomCard({ room, selected, onSelect }) {
  const view = roomPresentation(room);
  const sensors = room.latest?.sensors;
  const estimate = room.latest?.prediction && (STATES[room.status] && ['occupied', 'vacant'].includes(room.status)
    ? STATES[room.status] : room.latest.prediction.occupancy ? '사용 중' : '비어 있음');
  return <button type="button" className={`room-card k-${view.kind} ${selected ? 'selected' : ''}`}
    onClick={() => onSelect(room.id)} aria-pressed={Boolean(selected)}>
    <span className="room-card-top"><span>{room.floor}층 · {room.source === 'simulation' ? '가상 센서' : '실센서'}</span>
      {['light-check', 'monitoring'].includes(view.kind) && room.light_check?.since && <b>{duration(room.light_check.duration_seconds)} 관찰</b>}</span>
    <span className="room-card-title"><strong>{room.name}</strong><Badge room={room} /></span>
    <span className="room-action">{view.message}</span>
    {estimate && <span className={`room-estimate ${view.kind === 'light-check' ? 'secondary' : ''}`}>
      {['connection', 'paused'].includes(view.kind) ? '마지막 예상' : '사용 여부 예상'}: <b>{estimate}</b>
      {view.kind === 'light-check' && <em> · 센서 흐름과 달라 현장 확인 우선</em>}</span>}
    <span className="room-card-sensors">{['Light', 'CO2', 'Temperature', 'Humidity'].map(key => <span key={key}>
      <small><Icon name={SENSOR_ICONS[key]} size={14} />{SENSOR_LABELS[key][0]}</small>
      <Measure value={sensors?.[key]} unit={SENSOR_LABELS[key][1]} digits={key === 'Temperature' ? 1 : 0} /></span>)}</span>
    <span className="room-card-bottom">마지막 수신 {receivedTime(room.latest?.received_at)}</span>
  </button>;
}

export function LightCheckPanel({ room, disconnected = false }) {
  const check = room.light_check;
  const view = roomPresentation(room, disconnected);
  const rules = check?.rules;
  const basis = room.source === 'simulation' ? ' (시연 시각)' : '';
  return <section className={`light-reason k-${view.kind}`} aria-label="불 확인 참고 이유">
    <div className="light-reason-heading"><h3>순찰 참고 신호</h3><Badge room={room} disconnected={disconnected} /></div>
    <p className="light-message">{view.message}</p>
    {view.kind === 'light-check' && check?.model_disagrees && <p className="light-conflict">모델은 사용 중으로 예상했지만 센서 흐름이 달라 현장 확인이 필요합니다. 사용 여부 예상은 그대로 유지합니다.</p>}
    <dl className="light-evidence">
      {check?.duration_seconds > 0 && <div><dt>{['connection', 'paused'].includes(view.kind) ? '멈추기 전 지속 시간' : '이 상태로 지속'}</dt><dd>{duration(check.duration_seconds)}{check.time_basis === 'simulation' && <small>시연 경과 시간</small>}</dd></div>}
      <div><dt>마지막 수신</dt><dd>{receivedTime(room.latest?.received_at)}</dd></div>
    </dl>
    <details className="reason-more"><summary>판단 기준과 한계</summary>
      <p>{rules ? `시연용 가정: 실내 밝기 ${rules.min_lux} lux 이상 · CO₂ ${rules.max_co2_ppm} ppm 이하 · ${duration(rules.hold_seconds)} 지속`
        : '서버의 참고 기준을 확인 중입니다.'}</p>
      <p>햇빛 때문에 밝을 수 있고 CO₂는 환기의 영향을 받습니다. 사람 없음이나 불 켜짐을 확정하지 않습니다. 실제 시설에서는 기준 교정이 필요합니다.</p>
      <p>관찰 시작 {check?.since ? `${measured(check.since, room.source)}${basis}` : '—'} · 마지막 측정 {room.latest ? `${measured(room.latest.observed_at, room.source)}${basis}` : '—'}. 지속 시간은 새 측정 사이의 시간만 셉니다.</p>
    </details>
  </section>;
}

export function SensorTrend({ history, sensor, maxGap = 120, source }) {
  const points = history.filter(row => row.accepted_as_latest && row.sensors?.[sensor] != null).slice(-30);
  const values = points.map(row => row.sensors[sensor]);
  const low = values.length ? Math.min(...values) : 0, high = values.length ? Math.max(...values) : 1;
  const pad = Math.max((high - low) * 0.15, Math.abs(high) * 0.02, 0.5);
  const min = low - pad, max = high + pad;
  const times = points.map(row => new Date(row.observed_at).getTime());
  const firstTime = times.length ? Math.min(...times) : 0;
  const span = Math.max(...times, firstTime) - firstTime || 1;
  const path = values.map((value, index) => {
    const gap = index && (times[index] <= times[index - 1] || (times[index] - times[index - 1]) / 1000 > maxGap
      || points[index].light_check?.continuity_id !== points[index - 1].light_check?.continuity_id);
    return `${index && !gap ? 'L' : 'M'}${16 + (times[index] - firstTime) * 268 / span},${76 - (value - min) / (max - min || 1) * 57}`;
  }).join(' ');
  const change = values.length > 1 ? Math.round((values.at(-1) - values[0]) * 10) / 10 : null;
  const unit = SENSOR_LABELS[sensor][1];
  return <div className="room-trend"><div className="trend-head"><b><Icon name={SENSOR_ICONS[sensor]} size={15} />{SENSOR_LABELS[sensor][0]} 변화</b>
    <small>{change === null ? `최근 ${points.length}회` : `표시 구간 처음 대비 ${change > 0 ? '+' : ''}${change}\u00a0${unit}`}</small></div>
    {points.length ? <><svg viewBox="0 0 300 90" role="img" aria-label={`${SENSOR_LABELS[sensor][0]} 최근 센서 변화`}>
      <path d="M16 76H284" className="trend-baseline" /><path d={path} className={`trend-line ${sensor.toLowerCase()}`} />
    </svg><div className="trend-axis"><span>{measured(points[0].observed_at, source)}</span><span>{measured(points.at(-1).observed_at, source)}</span></div>
      <p className="trend-range"><span>최저 {reading(low, 1)}</span><span>{`최고 ${reading(high, 1)}\u00a0${unit}`}</span></p>
      {change === null && <p className="trend-change">변화를 보려면 새 측정이 필요합니다.</p>}</>
      : <p className="guard-muted">새 측정이 들어오면 표시합니다.</p>}
  </div>;
}

export default function ClassroomBoard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [floor, setFloor] = useState('all');
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState('101');
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState('');
  const [controlError, setControlError] = useState('');
  const [busy, setBusy] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [confirmRestart, setConfirmRestart] = useState(false);
  const [legendOpen] = useState(() => typeof window === 'undefined' || !window.matchMedia('(max-width: 900px)').matches);
  const detailRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const next = await api('/classrooms', { signal: controller.signal });
        if (!controller.signal.aborted) { setData(next); setError(''); }
      } catch (cause) { if (cause.name !== 'AbortError') setError(cause.message); }
      finally { if (!controller.signal.aborted) timer = setTimeout(poll, 1000); }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refreshKey]);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    setDetail(null);
    setDetailError('');
    async function poll() {
      try {
        const next = await api(`/classrooms/${selected}`, { signal: controller.signal });
        if (!controller.signal.aborted) { setDetail(next); setDetailError(''); }
      } catch (cause) { if (cause.name !== 'AbortError') setDetailError(cause.message); }
      finally { if (!controller.signal.aborted) timer = setTimeout(poll, 1000); }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [selected, refreshKey]);

  async function control(body, roomId = null) {
    setBusy(true); setControlError('');
    try {
      await api(roomId ? `/classrooms/${roomId}/source` : '/simulation/control', { method: roomId ? 'PUT' : 'POST', body });
      setRefreshKey(previous => previous + 1);
    } catch (cause) { setControlError(cause.message); }
    finally { setBusy(false); }
  }

  function jump(ref) {
    if (window.matchMedia('(max-width: 1280px)').matches) {
      const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      requestAnimationFrame(() => ref.current?.scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'start' }));
    }
  }

  function choose(roomId) {
    setSelected(roomId);
    jump(detailRef);
  }

  const rooms = (data?.classrooms || []).map(room => error ? { ...room, status: 'unavailable', message: '서버 연결 확인 필요', offline: true } : room);
  const counts = Object.fromEntries(FILTERS.map(([key]) => [key, filterRooms(rooms, { filter: key }).length]));
  const visible = filterRooms(rooms, { floor, filter, search });
  const floors = [...new Set(rooms.map(room => String(room.floor)))].sort();
  const average = key => {
    const values = rooms.map(room => room.latest?.sensors?.[key]).filter(value => value != null);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const sim = data?.simulation;
  const simCount = rooms.filter(room => room.source === 'simulation').length;
  const latest = detail?.latest;
  const prediction = latest?.prediction;
  const inputs = latest?.model_inputs;
  const history = (detail?.history || []).filter(row => row.source === detail.source
    && (detail.source !== 'simulation' || row.session_id === latest?.session_id));
  const stat = key => {
    const [, label, icon] = FILTERS.find(([name]) => name === key);
    return <button type="button" key={key} className={`stat s-${key} ${counts[key] ? '' : 'zero'}`} onClick={() => setFilter(key)} aria-pressed={filter === key}>
      <span><Icon name={icon} size={17} />{label}</span><strong>{counts[key]}<small>개</small></strong></button>;
  };

  return <div className="board">
    <main className="guard-main">
      <header className="board-head">
        <div className="board-title"><h1>강의실 현황</h1>
          <p>불을 확인할 강의실을 먼저 보세요. 밝기·이산화탄소·온도·습도와 시간대로 사용 여부를 예측합니다.</p></div>
        <div className="board-tools">
          <span className={`live ${error ? 'off' : ''}`} role="status"><i />{error ? '연결 끊김 · 다시 연결 중' : data ? '연결됨 · 1초마다 갱신' : '현황 연결 중…'}</span>
          <div className="sim-inline" role="group" aria-label="시연 제어">
            <span className="sim-tag">{!data ? '연결 중' : simCount ? `시연 ${simCount}곳` : '실센서'}</span>
            <span className="sim-time"><small>시연 시각</small><strong>{clock(sim?.observed_at)}</strong></span>
            <button type="button" className="primary" disabled={!sim || busy || sim.finished && !sim.playing} onClick={() => control({ action: sim.playing ? 'pause' : 'play' })}>
              <Icon name={sim?.playing ? 'pause' : 'play'} size={16} />{sim?.playing ? '일시정지' : sim?.finished ? '시연 종료' : '재생'}</button>
            <details className="sim-more" onToggle={event => { if (!event.currentTarget.open) setConfirmRestart(false); }}>
              <summary>시연 설정</summary>
              <div className="sim-pop">
                <p className="sim-range">가상 하루 {sim?.start || '05:00'}~{sim?.end || '24:00'} · 시연은 실센서 방에 영향을 주지 않습니다.</p>
                <div className="sim-speed"><span id="sim-speed-label">시간 흐름 <small>실제 1초에</small></span>
                  <div className="segmented" role="group" aria-labelledby="sim-speed-label">
                    {[1, 5, 10, 20].map(speed => <button type="button" key={speed} aria-pressed={(sim?.speed || 5) === speed} disabled={!sim || busy}
                      onClick={() => control({ action: 'speed', speed })}>{speed}분</button>)}
                  </div></div>
                {confirmRestart
                  ? <div className="sim-confirm" role="alert"><p>처음부터 다시 시작할까요? 이 화면을 보는 모든 사람의 시연 시계가 함께 05:00으로 돌아갑니다.</p>
                    <span><button type="button" className="danger" disabled={busy} onClick={() => { setConfirmRestart(false); control({ action: 'restart' }); }}>다시 시작</button>
                      <button type="button" onClick={() => setConfirmRestart(false)}>취소</button></span></div>
                  : <button type="button" disabled={!sim || busy} onClick={() => setConfirmRestart(true)}><Icon name="restart" size={16} />처음부터…</button>}
                <p className="sim-note">{!sim ? '시연 상태를 확인하고 있습니다.' : sim.playing ? '시연의 분 단위 센서 흐름을 재생 중입니다.' : sim.finished ? `${sim.end || '24:00'}까지 도달했습니다. 처음부터 다시 볼 수 있습니다.` : '시연 일시정지 · 마지막 예상만 보존합니다.'} 실제 시설이 아닌 예시 강의실입니다.
                  {sim?.scenario_id === 'classroom-day-v2' && ' 103호에는 수업 뒤 밝기가 남는 예시가 포함됩니다. 확인 신호는 시간표가 아닌 센서값으로 계산합니다.'}
                  {sim?.scenario_id === 'classroom-day-v1' && ' 이전 시연이 진행 중입니다. ‘처음부터’를 누르면 밝기가 남는 예시를 볼 수 있습니다.'}</p>
              </div>
            </details>
          </div>
        </div>
      </header>
      {sim?.error && <p className="guard-alert" role="alert">{sim.error}</p>}
      {error && <div className="guard-alert" role="alert">현황 서버와 통신하지 못했습니다. 마지막 수신값만 보존하고 재연결 중입니다. {error}</div>}
      {controlError && <div className="guard-alert" role="alert">{controlError}</div>}

      <section className="stat-groups" aria-label="강의실 요약">
        <div className="stat-group visit" role="group" aria-labelledby="group-visit"><h2 id="group-visit">가 볼 방</h2>
          <div className="stat-row">{VISIT_FILTERS.map(stat)}</div></div>
        <div className="stat-group usage" role="group" aria-labelledby="group-usage"><h2 id="group-usage">사용 현황 <small>사용 여부를 예상할 수 있는 방 기준</small></h2>
          <div className="stat-row">{USAGE_FILTERS.map(stat)}</div></div>
      </section>

      <div className="board-body">
        <section className="room-overview" ref={listRef} aria-labelledby="rooms-title">
          <div className="room-toolbar"><h2 id="rooms-title">강의실 <small>{visible.length}개 표시</small></h2>
            {(filter !== 'all' || floor !== 'all' || search) && <button type="button" className="reset" onClick={() => { setFilter('all'); setFloor('all'); setSearch(''); }}>조건 지우기</button>}</div>
          <div className="room-grid">{visible.map(room => <RoomCard key={room.id} room={room} selected={selected === room.id} onSelect={choose} />)}</div>
          {!visible.length && <p className="guard-empty">{!data && !error ? '강의실을 불러오는 중입니다.' : '조건에 맞는 강의실이 없습니다. ‘조건 지우기’를 눌러 전체를 다시 볼 수 있습니다.'}</p>}
          <p className="guard-explanation">‘불 확인 필요’는 별도 순찰 참고 신호입니다. 사용 여부 예상과 중복될 수 있으며 예약·출입 가능 여부를 뜻하지 않습니다. 확인 대상부터 정렬합니다.</p>
        </section>

        <aside className="room-detail" ref={detailRef} aria-label="선택 강의실 상세">
          <header><div><h2>{detail?.name || `시연 ${selected}호`}</h2>
            <p>{detail ? `${detail.source === 'simulation' ? '가상 센서' : '실센서'} · 마지막 수신 ${receivedTime(latest?.received_at)}` : '선택한 강의실'}</p></div>
            {detail && <Badge room={detail} disconnected={Boolean(error || detailError)} />}</header>
          <button type="button" className="back-to-list" onClick={() => jump(listRef)}>강의실 목록으로</button>
          {detailError && <p className="guard-alert" role="alert">상세 정보를 갱신하지 못했습니다. {detailError}</p>}
          {detail ? <>
            <LightCheckPanel room={detail} disconnected={Boolean(error || detailError)} />
            <section className={`predict ${prediction?.occupancy ? 'on' : ''}`} aria-label="모델 예측">
              <div><h3>사용 중일 확률</h3><strong>{prediction ? `${Math.round(prediction.probability * 100)}%` : '—'}</strong></div>
              <div className="predict-bar" role="presentation"><i style={{ transform: `scaleX(${prediction ? prediction.probability : 0})` }} /></div>
              <p>{prediction ? `모델 예상: ${prediction.occupancy ? '사용 중' : '비어 있음'}` : latest?.error?.message || '새 측정을 기다리고 있습니다.'}
                {inputs && ` · ${inputs.Hour}시 ${DAYS[inputs.DayOfWeek]}요일 기준`}. 모델 예상은 실제 관측 정답이 아닙니다.</p>
            </section>
            <div className="room-sensor-values">{['Light', 'CO2', 'Temperature', 'Humidity'].map(key => <div key={key}>
              <span><Icon name={SENSOR_ICONS[key]} size={15} />{SENSOR_LABELS[key][0]}</span>
              <strong>{reading(latest?.sensors?.[key], 1)}{latest?.sensors?.[key] != null && <small>{SENSOR_LABELS[key][1]}</small>}</strong></div>)}</div>
            <div className="room-trends">{['Light', 'CO2', 'Temperature', 'Humidity'].map(sensor => <SensorTrend key={sensor} history={history} sensor={sensor}
              source={detail.source} maxGap={detail.light_check?.rules?.max_gap_seconds} />)}</div>
            <p className="guard-muted">그래프 시각은 {detail.source === 'simulation' ? '시연 시계' : '한국 시간'} 기준입니다. 습도는 참고 정보이며 낮은 습도를 빈 방의 증거로 쓰지 않습니다.</p>
            <details className="room-history">
              <summary>측정 기록과 데이터 소스 ({history.length}건)</summary>
              <div className="source-switch"><span>모델 {prediction?.model_version || '—'} · 마지막 측정 {latest ? measured(latest.observed_at, detail.source) : '아직 없음'}</span>
                <button type="button" disabled={busy} onClick={() => control({ source: detail.source === 'real' ? 'simulation' : 'real' }, detail.id)}>
                  {detail.source === 'real' ? '시연으로 전환' : '실센서 수신 대기로 전환'}</button></div>
              {detail.source === 'simulation' && <p className="guard-muted">가상 센서 변화 시간표: {detail.scenario_schedule.map(period => `${period.start}~${period.end}`).join(' · ')} (시연 시각). 실제 예약이나 사람 유무의 정답을 뜻하지 않습니다.</p>}
              <div className="table-scroll"><table>
                <thead><tr><th>측정 시각</th><th>판별</th><th>모델</th><th>연결 식별값</th></tr></thead>
                <tbody>{history.slice(-12).reverse().map(row => <tr key={row.measurement_id}><td>{measured(row.observed_at, row.source)}</td>
                  <td>{!row.accepted_as_latest ? '과거·동일 시각 측정 · 최신 반영 안 됨' : row.error ? '판별 불가' : row.prediction?.occupancy ? '사용 중' : '비어 있음'}</td>
                  <td>{row.prediction?.model_version || '—'}</td><td><small>측정 {row.measurement_id}</small><small>추론 {row.inference_id || '없음'}</small></td></tr>)}</tbody>
              </table></div>
            </details>
          </> : <p className="guard-empty">상세 정보를 불러오는 중입니다.</p>}
        </aside>
      </div>
      <footer className="guard-footer">예시 강의실 6개 · AWS EC2 배포 · 자동 소등과 로그인 기능은 없습니다.</footer>
    </main>

    <aside className="board-side" aria-label="보기 설정">
      <div className="side-group side-floors"><h2>층</h2>
        {[['all', '전체 층'], ...(floors.length ? floors : ['1', '2']).map(value => [value, `${value}층`])].map(([value, label]) =>
          <button type="button" key={value} className="side-item" aria-pressed={floor === value} onClick={() => setFloor(value)}>
            <span>{label}</span><b>{value === 'all' ? rooms.length : rooms.filter(room => String(room.floor) === value).length}</b></button>)}
      </div>
      <div className="side-group side-find"><h2><label htmlFor="room-search">강의실 찾기</label></h2>
        <div className="side-search"><Icon name="search" size={16} />
          <input id="room-search" type="search" placeholder="이름 또는 호수" value={search} onChange={event => setSearch(event.target.value)} /></div>
      </div>
      <details className="side-group side-legend-box" open={legendOpen}><summary>상태 안내</summary>
        <ul className="side-legend">{LEGEND.map(([kind, name, text]) => <li key={kind}><i className={`state-dot k-${kind}`} />
          <span><b>{name}</b><small>{text}</small></span></li>)}</ul>
      </details>
      <div className="side-group side-avg"><h2>전체 센서 평균</h2>
        <dl className="side-average">{['Light', 'CO2', 'Temperature', 'Humidity'].map(key => <div key={key}>
          <dt><Icon name={SENSOR_ICONS[key]} size={15} />{SENSOR_LABELS[key][0]}</dt>
          <dd>{reading(average(key), key === 'Temperature' ? 1 : 0)}{average(key) != null && <small> {SENSOR_LABELS[key][1]}</small>}</dd></div>)}</dl>
      </div>
      <p className="side-foot">시각은 한국 시간 24시간제입니다. 시연 시각은 가상 하루의 시계입니다.<br />수신 후 {data?.stale_after_seconds || 60}초가 지나면 점검 필요로 먼저 보여 줍니다.</p>
    </aside>
  </div>;
}
