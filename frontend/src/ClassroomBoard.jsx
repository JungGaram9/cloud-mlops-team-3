import React, { useEffect, useState } from 'react';

const STATES = {
  occupied: ['사용 중', '●'], vacant: ['비어 있음', '○'], waiting: ['수신 대기', '◷'],
  stale: ['연결 확인 필요', '△'], unavailable: ['예상 상태 확인 불가', '!'],
};
const SENSOR_LABELS = { Temperature: ['실내 온도', '°C'], Humidity: ['습도', '%'], Light: ['실내 밝기', 'lux'], CO2: ['이산화탄소', 'ppm'] };
const clock = value => value ? new Date(value).toLocaleTimeString('ko-KR', { timeZone: 'UTC', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }) : '—';
const receivedTime = value => value ? new Date(value).toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul', hourCycle: 'h23' }) : '아직 없음';

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

function Badge({ room, disconnected = false }) {
  const view = roomPresentation(room, disconnected);
  return <span className={`room-status ${view.kind}`}><span aria-hidden="true">{view.icon}</span>{view.label}</span>;
}

export function roomPresentation(room, disconnected = false) {
  const check = room.light_check;
  if (disconnected || ['stale', 'unavailable'].includes(room.status) || check?.state === 'stale') {
    return { kind: 'connection', label: '연결·측정 확인', icon: '△', priority: 1,
      message: disconnected ? '서버 연결을 확인해 주세요. 마지막 수신값입니다.'
        : ['stale', 'unavailable'].includes(room.status) ? room.message : check?.message };
  }
  if (room.simulation_paused || check?.state === 'paused') {
    return { kind: 'paused', label: '시연 정지', icon: 'Ⅱ', priority: 4, message: '마지막 측정값입니다. 현재 실시간 상태가 아닙니다.' };
  }
  if (room.status === 'waiting' || !check || ['waiting', 'unavailable'].includes(check.state)) {
    return { kind: 'monitoring', label: '확인 중', icon: '◷', priority: 2,
      message: check?.message || '새로운 센서 측정을 기다리고 있습니다.' };
  }
  if (check.requires_check && check.state === 'attention') {
    return { kind: 'light-check', label: '불 확인 필요', icon: '☼', priority: 0, message: check.message };
  }
  if (check.state === 'monitoring') {
    return { kind: 'monitoring', label: '확인 중', icon: '◷', priority: 2, message: check.message };
  }
  const [label, icon] = STATES[room.status] || STATES.unavailable;
  return { kind: room.status, label: `${label} 예상`, icon, priority: 3, message: '현재 불 확인 참고 조건에 해당하지 않습니다.' };
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

export function RoomCard({ room, selected, onSelect }) {
  const view = roomPresentation(room);
  return <button className={`room-card ${room.status} ${view.kind} ${selected ? 'selected' : ''}`}
    onClick={() => onSelect(room.id)} aria-label={`${room.name} · ${view.label} · 상세 보기`} aria-pressed={selected}>
    <div className="room-card-top"><span>{room.floor}층 · {room.source === 'simulation' ? '가상 센서' : '실센서'}</span><span aria-hidden="true">↗</span></div>
    <h3>{room.name}</h3><Badge room={room} />
    <p className="room-action">{view.message}</p>
    {room.latest?.prediction && <p className="room-estimate">{['connection', 'paused'].includes(view.kind) ? '마지막 예상' : '사용 여부 예상'}: {STATES[room.status]?.[0] && ['occupied', 'vacant'].includes(room.status)
      ? STATES[room.status][0] : room.latest.prediction.occupancy ? '사용 중' : '비어 있음'}
      {view.kind === 'light-check' && <span> · 현장 확인 우선</span>}</p>}
    <div className="room-card-sensors"><span>실내 밝기 <b>{room.latest?.sensors ? Math.round(room.latest.sensors.Light) : '—'} lux</b></span>
      <span>이산화탄소 <b>{room.latest?.sensors ? Math.round(room.latest.sensors.CO2) : '—'} ppm</b></span></div>
    <div className="room-card-bottom"><span>마지막 수신 {receivedTime(room.latest?.received_at)} KST</span>
      {['light-check', 'monitoring'].includes(view.kind) && room.light_check?.since && <b>{duration(room.light_check.duration_seconds)} 관찰</b>}</div>
  </button>;
}

export function LightCheckPanel({ room, disconnected = false }) {
  const check = room.light_check;
  const view = roomPresentation(room, disconnected);
  const rules = check?.rules;
  return <section className={`light-reason ${view.kind}`} aria-label="불 확인 참고 이유">
    <div className="light-reason-heading"><span className="guard-eyebrow">순찰 참고 신호</span>
      <Badge room={room} disconnected={disconnected} /></div>
    <h3>{view.message}</h3>
    {view.kind === 'light-check' && check?.model_disagrees && <p className="light-conflict">모델은 사용 중으로 예상했지만 센서 흐름이 달라 현장 확인이 필요합니다. 사용 여부 예상은 그대로 유지합니다.</p>}
    <dl className="light-evidence"><div><dt>{['connection', 'paused'].includes(view.kind) ? '마지막 측정의 지속 시간' : '측정으로 확인한 지속 시간'}</dt><dd>{duration(check?.duration_seconds)}<small>{check?.time_basis === 'simulation' ? '시연 경과 시간' : '실제 측정 시각 기준'}</small></dd></div>
      <div><dt>조건을 관찰하기 시작한 시각</dt><dd>{clock(check?.since)}<small>UTC · 새 측정 사이의 시간만 계산</small></dd></div>
      <div><dt>마지막 갱신</dt><dd>{receivedTime(room.latest?.received_at)}<small>KST · 측정 {clock(room.latest?.observed_at)} UTC</small></dd></div></dl>
    <p className="light-rule">{rules ? `시연용 가정: 실내 밝기 ${rules.min_lux} lux 이상 · CO₂ ${rules.max_co2_ppm} ppm 이하 · ${duration(rules.hold_seconds)} 지속`
      : '서버의 참고 기준을 확인 중입니다.'}</p>
    <p className="light-limit">햇빛 때문에 밝을 수 있고 CO₂는 환기의 영향을 받습니다. 사람 없음이나 불 켜짐을 확정하지 않습니다. 실제 시설에서는 기준 교정이 필요합니다.</p>
  </section>;
}

export function SensorTrend({ history, sensor, maxGap = 120 }) {
  const points = history.filter(row => row.accepted_as_latest && row.sensors?.[sensor] != null).slice(-30);
  const values = points.map(row => row.sensors[sensor]);
  const min = Math.min(...values, 0), max = Math.max(...values, 1);
  const times = points.map(row => new Date(row.observed_at).getTime());
  const firstTime = times.length ? Math.min(...times) : 0;
  const span = Math.max(...times, firstTime) - firstTime || 1;
  const path = values.map((value, index) => {
    const gap = index && (times[index] <= times[index - 1] || (times[index] - times[index - 1]) / 1000 > maxGap
      || points[index].light_check?.continuity_id !== points[index - 1].light_check?.continuity_id);
    return `${index && !gap ? 'L' : 'M'}${16 + (times[index] - firstTime) * 268 / span},${76 - (value - min) / (max - min || 1) * 57}`;
  }).join(' ');
  const change = values.length > 1 ? Math.round((values.at(-1) - values[0]) * 10) / 10 : null;
  return <div className="room-trend"><div><b>{SENSOR_LABELS[sensor][0]} 변화</b><small>최근 {points.length}회 · {SENSOR_LABELS[sensor][1]}</small></div>
    {points.length ? <><svg viewBox="0 0 300 90" role="img" aria-label={`${SENSOR_LABELS[sensor][0]} 최근 센서 변화`}>
      <path d="M16 76H284" className="trend-baseline" /><path d={path} className={`trend-line ${sensor.toLowerCase()}`} />
    </svg><div className="trend-axis"><span>{clock(points[0].observed_at)}</span><span>{clock(points.at(-1).observed_at)} UTC</span></div>
      <p className="trend-change">{change === null ? '변화를 보려면 새 측정이 필요합니다.' : `표시 구간 처음 대비 ${change > 0 ? '+' : ''}${change} ${SENSOR_LABELS[sensor][1]}`}</p></>
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

  const rooms = (data?.classrooms || []).map(room => error ? { ...room, status: 'unavailable', message: '서버 연결 확인 필요' } : room);
  const counts = Object.fromEntries(['all', 'light_check', 'occupied', 'vacant', 'attention'].map(key =>
    [key, filterRooms(rooms, { filter: key }).length]));
  const visible = filterRooms(rooms, { floor, filter, search });
  const sim = data?.simulation;
  const simCount = rooms.filter(room => room.source === 'simulation').length;
  const latest = detail?.latest;
  const history = (detail?.history || []).filter(row => row.source === detail.source
    && (detail.source !== 'simulation' || row.session_id === latest?.session_id));

  return <main className="guard-main">
    <header className="guard-heading"><div><p className="guard-eyebrow">시설 순찰 · 강의실 사용 현황</p>
      <h1>불을 확인할 강의실을 먼저 보세요.</h1><p>밝기와 이산화탄소의 흐름으로 순찰 참고 대상을 알려드립니다. 현장에서 사람이 있는지 확인해 주세요.</p></div>
      <span className={`guard-connection ${error ? 'disconnected' : ''}`} role="status">{error ? '연결 확인 중' : data ? '● 연결됨 · 1초마다 갱신' : '현황 연결 중…'}</span>
    </header>

    <section className="simulation-banner" aria-label="시연 제어">
      <div className="simulation-label"><b>{!data ? '현황 연결 중' : simCount ? '시연 모드 · 가상 센서 데이터' : '실센서 모드'}</b>
        <span>실제 시설이 아닌 예시 강의실 · 시연 {simCount}개 / 실센서 {rooms.length - simCount}개</span></div>
      <div className="simulation-clock"><small>공유 시연 시각 · UTC</small><strong>{clock(sim?.observed_at)}</strong><span>{sim?.start || '08:00'}~{sim?.end || '22:00'}</span></div>
      <div className="simulation-controls">
        <button disabled={!sim || busy || sim.finished && !sim.playing} onClick={() => control({ action: sim.playing ? 'pause' : 'play' })}>
          {sim?.playing ? 'Ⅱ 일시정지' : sim?.finished ? '하루 시연 종료' : '▶ 재생'}</button>
        <button disabled={!sim || busy} onClick={() => control({ action: 'restart' })}>↺ 처음부터</button>
        <label>속도 <select aria-label="시연 속도" value={sim?.speed || 5} disabled={!sim || busy}
          onChange={event => control({ action: 'speed', speed: Number(event.target.value) })}>
          {[1, 5, 10, 20].map(speed => <option value={speed} key={speed}>1초 = {speed}분</option>)}
        </select></label>
      </div>
      <p className="simulation-note">{!sim ? '시연 상태를 확인하고 있습니다.' : sim.playing ? '시연의 분 단위 센서 흐름을 재생 중입니다.' : sim.finished ? '22:00까지 도달했습니다. 처음부터 다시 볼 수 있습니다.' : '시연 일시정지 · 마지막 예상만 보존합니다.'} 시연은 실센서 방에 영향을 주지 않습니다.
        {sim?.scenario_id === 'classroom-day-v2' && ' 103호에는 수업 뒤 밝기가 남는 예시가 포함됩니다. 확인 신호는 시간표가 아닌 센서값으로 계산합니다.'}
        {sim?.scenario_id === 'classroom-day-v1' && ' 이전 시연이 진행 중입니다. ‘처음부터’를 누르면 밝기가 남는 예시를 볼 수 있습니다.'} {sim?.error}</p>
    </section>

    {error && <div className="guard-alert" role="alert">현황 서버와 통신하지 못했습니다. 마지막 수신값만 보존하고 재연결 중입니다. {error}</div>}
    {controlError && <div className="guard-alert" role="alert">{controlError}</div>}
    <section className="guard-totals" aria-label="강의실 요약">
      {[['all', '전체 강의실'], ['light_check', '☼ 불 확인 필요'], ['occupied', '사용 중 예상'], ['vacant', '비어 있음 예상'], ['attention', '연결·측정 확인']].map(([key, label]) =>
        <button key={key} className={`guard-total ${key} ${filter === key ? 'chosen' : ''}`} onClick={() => setFilter(key)} aria-pressed={filter === key}>
          <span>{label}</span><strong>{counts[key]}<small>개</small></strong></button>)}
    </section>
    <p className="guard-explanation">‘불 확인 필요’는 별도 순찰 참고 신호입니다. 사용 여부 예상과 중복될 수 있으며 예약·출입 가능 여부를 뜻하지 않습니다. 확인 대상부터 정렬합니다.</p>

    <section className="room-overview"><div className="room-toolbar"><h2>강의실 현황 <small>{visible.length}개 표시</small></h2>
      <div><input aria-label="강의실 이름 검색" placeholder="강의실 이름 또는 호수 검색" value={search} onChange={event => setSearch(event.target.value)} />
        <select aria-label="층 선택" value={floor} onChange={event => setFloor(event.target.value)}><option value="all">전체 층</option><option value="1">1층</option><option value="2">2층</option></select>
        <select aria-label="상태 선택" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">전체 상태</option><option value="light_check">불 확인 필요</option><option value="occupied">사용 중 예상</option><option value="vacant">비어 있음 예상</option><option value="attention">연결·측정 확인</option></select></div>
    </div>
      <div className="room-grid">{visible.map(room => <RoomCard key={room.id} room={room} selected={selected === room.id} onSelect={setSelected} />)}</div>
      {!visible.length && <p className="guard-empty">{!data && !error ? '강의실을 불러오는 중입니다.' : '조건에 맞는 강의실이 없습니다.'}</p>}
      <p className="guard-muted">실제 수신 후 {data?.stale_after_seconds || 60}초가 지나면 연결 확인을 우선합니다. 시연 정지 중에는 마지막 참고 정보만 보여줍니다.</p>
    </section>

    <section className="room-detail" aria-label="선택 강의실 상세">
      <header><div><p className="guard-eyebrow">선택 강의실</p><h2>{detail?.name || `시연 ${selected}호`} 상세</h2></div>
        {detail && <Badge room={detail} disconnected={Boolean(error || detailError)} />}</header>
      {detailError && <p className="guard-alert" role="alert">상세 정보를 갱신하지 못했습니다. {detailError}</p>}
      {detail ? <><LightCheckPanel room={detail} disconnected={Boolean(error || detailError)} />
        <div className="room-detail-meta"><span>데이터: {detail.source === 'simulation' ? '시연 · 가상 센서' : '실센서'}</span>
        <span>측정 {latest ? `${new Date(latest.observed_at).toISOString().slice(0, 10)} ${clock(latest.observed_at)} UTC` : '아직 없음'}</span>
        <span>실제 수신 {receivedTime(latest?.received_at)} KST</span>
        <button disabled={busy} onClick={() => control({ source: detail.source === 'real' ? 'simulation' : 'real' }, detail.id)}>
          {detail.source === 'real' ? '시연으로 전환' : '실센서 수신 대기로 전환'}</button></div>
        <div className="room-sensor-values">{Object.entries(SENSOR_LABELS).map(([key, [name, unit]]) => <div key={key}><span>{name}</span>
          <strong>{latest?.sensors?.[key] == null ? '—' : Math.round(latest.sensors[key] * 10) / 10}<small>{unit}</small></strong></div>)}</div>
        <p className="guard-muted">습도는 참고 정보이며, 낮은 습도를 빈 방의 증거로 사용하지 않습니다. 아래 그래프는 마지막 측정까지의 변화입니다.</p>
        <div className="room-trends">{['Light', 'CO2', 'Humidity'].map(sensor => <SensorTrend key={sensor} history={history} sensor={sensor} maxGap={detail.light_check?.rules?.max_gap_seconds} />)}</div>
        <details className="room-history room-technical"><summary>모델·시간표·측정 기록 자세히 보기 ({history.length}건)</summary>
          <p className="room-model-note">{latest?.prediction ? `마지막 사용 중 예상 확률 ${Math.round(latest.prediction.probability * 100)}% · 모델 ${latest.prediction.model_version}` : latest?.error?.message || '새 측정을 기다리고 있습니다.'} 모델 예상은 실제 관측 정답이 아닙니다.</p>
          {detail.source === 'simulation' && <p className="guard-muted">가상 센서 변화 시간표: {detail.scenario_schedule.map(period => `${period.start}~${period.end}`).join(' · ')} UTC. 실제 예약이나 사람 유무의 정답을 뜻하지 않습니다.</p>}
          <div className="table-scroll"><table>
          <thead><tr><th>측정 UTC</th><th>판별</th><th>모델</th><th>연결 식별값</th></tr></thead>
          <tbody>{history.slice(-12).reverse().map(row => <tr key={row.measurement_id}><td>{clock(row.observed_at)}</td>
            <td>{!row.accepted_as_latest ? '과거·동일 시각 측정 · 최신 반영 안 됨' : row.error ? '판별 불가' : row.prediction?.occupancy ? '사용 중' : '비어 있음'}</td>
            <td>{row.prediction?.model_version || '—'}</td><td><small>측정 {row.measurement_id}</small><small>추론 {row.inference_id || '없음'}</small></td></tr>)}</tbody>
        </table></div></details>
      </> : <p className="guard-empty">상세 정보를 불러오는 중입니다.</p>}
    </section>
    <footer className="guard-footer">예시 강의실 6개 · 사용 여부와 순찰 참고 서비스 · 자동 소등 기능은 없습니다.</footer>
  </main>;
}
