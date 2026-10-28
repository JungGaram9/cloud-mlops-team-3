import React, { useEffect, useState } from 'react';

const STATES = {
  occupied: ['사용 중', '●'], vacant: ['비어 있음', '○'], waiting: ['수신 대기', '◷'],
  stale: ['확인 필요', '△'], unavailable: ['판별 불가', '!'],
};
const SENSOR_LABELS = { Temperature: ['기온', '°C'], Humidity: ['습도', '%'], Light: ['조도', 'lux'], CO2: ['CO₂', 'ppm'] };
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

function Badge({ status }) {
  const [label, icon] = STATES[status] || STATES.unavailable;
  return <span className={`room-status ${status}`}><span aria-hidden="true">{icon}</span>{label}</span>;
}

function SensorTrend({ history, sensor }) {
  const points = history.filter(row => row.accepted_as_latest && row.sensors?.[sensor] != null).slice(-30);
  const values = points.map(row => row.sensors[sensor]);
  const min = Math.min(...values, 0), max = Math.max(...values, 1);
  const path = values.map((value, index) => `${index ? 'L' : 'M'}${16 + index * 268 / Math.max(values.length - 1, 1)},${76 - (value - min) / (max - min || 1) * 57}`).join(' ');
  return <div className="room-trend"><div><b>{SENSOR_LABELS[sensor][0]} 변화</b><small>최근 {points.length}회 · {SENSOR_LABELS[sensor][1]}</small></div>
    {points.length ? <><svg viewBox="0 0 300 90" role="img" aria-label={`${SENSOR_LABELS[sensor][0]} 최근 센서 변화`}>
      <path d="M16 76H284" className="trend-baseline" /><path d={path} className={`trend-line ${sensor.toLowerCase()}`} />
    </svg><div className="trend-axis"><span>{clock(points[0].observed_at)}</span><span>{clock(points.at(-1).observed_at)} UTC</span></div></>
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
  const counts = { all: rooms.length, occupied: rooms.filter(room => room.status === 'occupied').length,
    vacant: rooms.filter(room => room.status === 'vacant').length, attention: rooms.filter(room => !['occupied', 'vacant'].includes(room.status)).length };
  const visible = rooms.filter(room => (floor === 'all' || String(room.floor) === floor)
    && (filter === 'all' || filter === room.status || filter === 'attention' && !['occupied', 'vacant'].includes(room.status))
    && `${room.name} ${room.id}`.toLowerCase().includes(search.toLowerCase()));
  const sim = data?.simulation;
  const simCount = rooms.filter(room => room.source === 'simulation').length;
  const latest = detail?.latest;
  const history = (detail?.history || []).filter(row => row.source === detail.source
    && (detail.source !== 'simulation' || row.session_id === latest?.session_id));

  return <main className="guard-main">
    <header className="guard-heading"><div><p className="guard-eyebrow">시설 순찰 · 강의실 사용 현황</p>
      <h1>지금, 어느 강의실이 사용 중인가요?</h1><p>센서로 추정한 재실 상태를 한눈에 확인하세요. 예약·출입 가능 여부와는 다릅니다.</p></div>
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
      <p className="simulation-note">{!sim ? '시연 상태를 확인하고 있습니다.' : sim.playing ? '서버가 시연을 진행 중입니다.' : sim.finished ? '22:00까지 도달했습니다. 처음부터 다시 볼 수 있습니다.' : '시연 일시정지 · 마지막 판별을 보존합니다.'} 시연은 실센서 모드의 방을 덮어쓰지 않습니다. {sim?.error}</p>
    </section>

    {error && <div className="guard-alert" role="alert">현황 서버와 통신하지 못했습니다. 마지막 수신값만 보존하고 재연결 중입니다. {error}</div>}
    {controlError && <div className="guard-alert" role="alert">{controlError}</div>}
    <section className="guard-totals" aria-label="강의실 요약">
      {[['all', '전체 강의실'], ['occupied', '사용 중'], ['vacant', '비어 있음'], ['attention', '확인 필요']].map(([key, label]) =>
        <button key={key} className={`guard-total ${key} ${filter === key ? 'chosen' : ''}`} onClick={() => setFilter(key)} aria-pressed={filter === key}>
          <span>{label}</span><strong>{counts[key]}<small>개</small></strong></button>)}
    </section>

    <section className="room-overview"><div className="room-toolbar"><h2>강의실 현황 <small>{visible.length}개 표시</small></h2>
      <div><input aria-label="강의실 이름 검색" placeholder="강의실 이름 또는 호수 검색" value={search} onChange={event => setSearch(event.target.value)} />
        <select aria-label="층 선택" value={floor} onChange={event => setFloor(event.target.value)}><option value="all">전체 층</option><option value="1">1층</option><option value="2">2층</option></select>
        <select aria-label="상태 선택" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">전체 상태</option><option value="occupied">사용 중</option><option value="vacant">비어 있음</option><option value="attention">확인 필요</option></select></div>
    </div>
      <div className="room-grid">{visible.map(room => <button key={room.id} className={`room-card ${room.status} ${selected === room.id ? 'selected' : ''}`}
        onClick={() => setSelected(room.id)} aria-label={`${room.name} 상세`} aria-pressed={selected === room.id}>
        <div className="room-card-top"><span>{room.floor}층 · {room.source === 'simulation' ? '가상 센서' : '실센서'}</span><span aria-hidden="true">↗</span></div>
        <h3>{room.name}</h3><Badge status={room.status} /><p>{room.message}</p>
        <div className="room-card-bottom"><span>실제 수신 {receivedTime(room.latest?.received_at)} KST</span>{room.simulation_paused && <b>시연 정지</b>}</div>
      </button>)}</div>
      {!visible.length && <p className="guard-empty">{!data && !error ? '강의실을 불러오는 중입니다.' : '조건에 맞는 강의실이 없습니다.'}</p>}
      <p className="guard-muted">실제 수신 후 {data?.stale_after_seconds || 60}초가 지나면 확인 필요로 표시합니다. 가상 측정 시각은 연결 상태 판단에 쓰지 않습니다.</p>
    </section>

    <section className="room-detail" aria-label="선택 강의실 상세">
      <header><div><p className="guard-eyebrow">선택 강의실</p><h2>{detail?.name || `시연 ${selected}호`} 상세</h2></div>
        {detail && <Badge status={error || detailError ? 'unavailable' : detail.status} />}</header>
      {detailError && <p className="guard-alert" role="alert">상세 정보를 갱신하지 못했습니다. {detailError}</p>}
      {detail ? <><div className="room-detail-meta"><span>소스: {detail.source === 'simulation' ? '시연 · 가상 센서' : '실센서'}</span>
        <span>측정 {latest ? `${new Date(latest.observed_at).toISOString().slice(0, 10)} ${clock(latest.observed_at)} UTC` : '아직 없음'}</span>
        <span>실제 수신 {receivedTime(latest?.received_at)} KST</span>
        <button disabled={busy} onClick={() => control({ source: detail.source === 'real' ? 'simulation' : 'real' }, detail.id)}>
          {detail.source === 'real' ? '시연으로 전환' : '실센서 수신 대기로 전환'}</button></div>
        <div className="room-sensor-values">{Object.entries(SENSOR_LABELS).map(([key, [name, unit]]) => <div key={key}><span>{name}</span>
          <strong>{latest?.sensors?.[key] == null ? '—' : Math.round(latest.sensors[key] * 10) / 10}<small>{unit}</small></strong></div>)}</div>
        <p className="room-model-note">{latest?.prediction ? `사용 중으로 추정한 확률 ${Math.round(latest.prediction.probability * 100)}% · 사용 모델 ${latest.prediction.model_version}` : latest?.error?.message || '새 측정을 기다리고 있습니다.'} 모델 추정은 실제 관측 정답이 아닙니다.</p>
        <div className="room-trends"><SensorTrend history={history} sensor="CO2" /><SensorTrend history={history} sensor="Light" /></div>
        {detail.source === 'simulation' && <p className="guard-muted">가상 센서 변화 시간표: {detail.scenario_schedule.map(period => `${period.start}~${period.end}`).join(' · ')} UTC. 실제 예약이나 재실 정답을 뜻하지 않습니다.</p>}
        <details className="room-history"><summary>최근 측정·판별 이력 ({history.length}건)</summary><div className="table-scroll"><table>
          <thead><tr><th>측정 UTC</th><th>판별</th><th>모델</th><th>연결 식별값</th></tr></thead>
          <tbody>{history.slice(-12).reverse().map(row => <tr key={row.measurement_id}><td>{clock(row.observed_at)}</td>
            <td>{!row.accepted_as_latest ? '과거·동일 시각 측정 · 최신 반영 안 됨' : row.error ? '판별 불가' : row.prediction?.occupancy ? '사용 중' : '비어 있음'}</td>
            <td>{row.prediction?.model_version || '—'}</td><td><small>측정 {row.measurement_id}</small><small>추론 {row.inference_id || '없음'}</small></td></tr>)}</tbody>
        </table></div></details>
      </> : <p className="guard-empty">상세 정보를 불러오는 중입니다.</p>}
    </section>
    <footer className="guard-footer">시연 명칭의 예시 강의실 6개 · 재실 추정 서비스 · 로그인·권한 관리는 이번 구현에 포함되지 않습니다.</footer>
  </main>;
}
