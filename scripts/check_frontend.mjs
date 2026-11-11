// 빌드나 서버 기동 없이 JSX 렌더링과 평가 그룹·분할 표시 계약을 확인한다.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createServer } from '/app/frontend/node_modules/vite/dist/node/index.js';
import react from '/app/frontend/node_modules/@vitejs/plugin-react/dist/index.js';
import React from '/app/frontend/node_modules/react/index.js';
import { renderToStaticMarkup } from '/app/frontend/node_modules/react-dom/server.node.js';

const cacheDir = await fs.mkdtemp('/tmp/classroom-ui-');
const server = await createServer({
  root: '/app/frontend', cacheDir, configFile: false, envDir: false, server: { middlewareMode: true, hmr: false },
  plugins: [react(), { name: 'verify-components', enforce: 'pre', transform(source, id) {
    if (!id.endsWith('/src/main.jsx')) return;
    return source.slice(0, source.indexOf('const reactRoot ='))
      + '\nexport { evaluationGroups, Dashboard, TrainingForm, TrainingMonitor, ReportPanel, reusableTrainingConfig, DEFAULT_CONFIG, PRESETS };';
  } }],
});
try {
  const ui = await server.ssrLoadModule('/src/main.jsx');
  const version = (name, sha256, rows, accuracy) => ({ version: name, test_data: { sha256, rows },
    models: [{ metrics: { accuracy, f1: 0.4 }, features: ['CO2'] }], name, created_at_utc: '2026-01-01T00:00:00Z' });
  const versions = [version('v3', 'b'.repeat(64), 18479, 0.5),
    version('v2', 'a'.repeat(64), 12417, 0.8), version('v1', 'a'.repeat(64), 12417, 0.7)];
  assert.deepEqual(ui.evaluationGroups(versions).map(group => group.versions.length), [1, 2]);
  assert.equal(ui.evaluationGroups([...versions, version('v4', 'a'.repeat(64), 1, 0.99)]).length, 3);
  assert.equal(ui.evaluationGroups([version('v5', null, 1, 0.8), version('v6', null, 1, 0.9)]).length, 2);
  const dashboard = renderToStaticMarkup(React.createElement(ui.Dashboard, {
    registry: { versions, current_version: 'v2' }, selected: 'v2', onSelect() {},
  }));
  assert.match(dashboard, /그룹 최고 v2/);
  assert.match(dashboard, /그룹 최고 v3/);
  assert.match(dashboard, /다른 그룹의 정확도로 모델 간 우열을 판단하지 않습니다/);
  const summary = { rows: 12, sources: { Room_A: { rows: 6, class_counts: { 0: 3, 1: 3 } },
    Room_B: { rows: 6, class_counts: { 0: 3, 1: 3 } } } };
  const plan = { validation_ratio: 0.2, ready: true, errors: [], warnings: [], fit: summary, validation: summary, test: summary };
  const profiles = { available: true, rows: 12, features: {
    Temperature: { minimum: 10, maximum: 30 }, Humidity: { minimum: 10, maximum: 80 },
    Light: { minimum: 0, maximum: 2000 }, CO2: { minimum: 300, maximum: 2000 },
    Hour: { minimum: 0, maximum: 23, values: [0, 14, 23] }, DayOfWeek: { minimum: 0, maximum: 6, values: [0, 2, 6] },
  } };
  const props = { config: ui.DEFAULT_CONFIG, options: { training_data: summary, test_data: summary, data_plan: plan },
    previewLimits: profiles, values: ui.PRESETS[0].values, setValues() {}, onSubmit() {}, onUpdate() {}, onToggleFeature() {}, onReset() {} };
  const form = renderToStaticMarkup(React.createElement(ui.TrainingForm, props));
  assert.match(form, /출처별 시간순 분할/);
  assert.match(form, /Room A/);
  assert.match(form, /Room B/);
  assert.match(form, /form="sensor-preview-form"/);
  assert.doesNotMatch(form.match(/<button type="submit"[^>]*>/)?.[0] || '', /disabled/);
  const invalid = renderToStaticMarkup(React.createElement(ui.TrainingForm, {
    ...props, options: { ...props.options, data_plan: { ...plan, ready: false, errors: ['fit 행이 없습니다.'] } },
  }));
  assert.match(invalid, /fit 행이 없습니다/);
  assert.match(invalid.match(/<button type="submit"[^>]*>/)?.[0] || '', /disabled/);
  const report = JSON.parse(await fs.readFile('/app/reports/v3/light_comparison.json', 'utf8'));
  const models = structuredClone(report.modes);
  for (const model of Object.values(models)) {
    model.split_strategy = 'source_chronological';
    model.training_split = summary;
    model.validation_split = summary;
    model.test_data = { ...summary, sha256: 'b'.repeat(64) };
    model.source_metrics = { Room_A: { rows: 6, class_counts: { 0: 3, 1: 3 }, metrics: null } };
  }
  const html = renderToStaticMarkup(React.createElement(ui.ReportPanel, { report: { models }, version: 'v7' }));
  assert.match(html, /출처별 평가/);
  assert.match(html, /b{64}/);
  assert.match(html, /출처별 시간순/);
  const oldReport = JSON.parse(await fs.readFile('/app/reports/v7/training.json', 'utf8'));
  const oldHtml = renderToStaticMarkup(React.createElement(ui.ReportPanel, { report: oldReport, version: 'v7' }));
  assert.match(oldHtml, /v7 학습 보고서/);
  const imported = ui.reusableTrainingConfig(oldReport.config, ui.DEFAULT_CONFIG);
  assert.equal(Object.hasOwn(imported, 'datasets'), false);
  assert.deepEqual(imported.features, oldReport.config.features);
  assert.equal(imported.activation, oldReport.config.activation);
  assert.deepEqual(Object.keys(imported).sort(), Object.keys(ui.DEFAULT_CONFIG).sort());
  const damaged = { id: 'invalid-name', status: 'failed', history_error: true, config: { name: '읽을 수 없는 이력' },
    phase: '학습 이력 읽기 오류', error: '이 학습 이력만 읽지 못했습니다. 원본 파일은 보존됩니다.', curves: {} };
  const monitor = renderToStaticMarkup(React.createElement(ui.TrainingMonitor, {
    job: damaged, jobs: [damaged], values: ui.PRESETS[0].values,
  }));
  assert.match(monitor, /학습 이력 읽기 오류/);
  assert.match(monitor, /원본 파일은 보존됩니다/);
  assert.doesNotMatch(monitor, /학습 중지<\/button>/);
  const olderError = renderToStaticMarkup(React.createElement(ui.TrainingMonitor, {
    job: null, jobs: [...Array.from({ length: 8 }, (_, index) => ({ id: String(index), status: 'completed',
      config: { name: '정상 이력' } })), damaged], values: ui.PRESETS[0].values,
  }));
  assert.match(olderError, /role="alert"/);
  assert.match(olderError, /학습 이력 1개를 읽지 못했습니다/);
  const board = await server.ssrLoadModule('/src/ClassroomBoard.jsx');
  const room = { id: '103', name: '시연 103호', floor: 1, source: 'simulation', status: 'occupied',
    latest: { observed_at: '2026-10-29T13:00:00Z', received_at: '2026-10-29T04:00:00Z',
      sensors: { Light: 450, CO2: 500, Humidity: 28, Temperature: 22 },
      prediction: { occupancy: 1, probability: 0.7, model_version: 'v7' } },
    light_check: { state: 'attention', requires_check: true, model_disagrees: true, duration_seconds: 600,
      since: '2026-10-29T12:50:00Z', time_basis: 'simulation',
      message: '밝은 상태가 계속되고 있습니다. 사람이 있는지 확인해 주세요.',
      rules: { min_lux: 300, max_co2_ppm: 650, hold_seconds: 600 } } };
  const normal = { ...room, id: '101', name: '시연 101호', light_check: { ...room.light_check, state: 'clear', requires_check: false } };
  const stale = { ...room, id: '201', name: '시연 201호', floor: 2, status: 'stale', message: '수신 지연 · 연결 확인 필요' };
  const paused = { ...room, id: '202', name: '시연 202호', floor: 2, simulation_paused: true };
  const uncertain = { ...room, id: '102', name: '시연 102호', light_check: { ...room.light_check, state: 'monitoring', requires_check: false } };
  const rooms = [normal, stale, paused, uncertain, room];
  assert.equal(board.filterRooms(rooms)[0].id, '103');
  assert.deepEqual(board.filterRooms(rooms, { filter: 'light_check' }).map(item => item.id), ['103']);
  assert.deepEqual(board.filterRooms(rooms, { filter: 'attention' }).map(item => item.id), ['201']);
  assert.equal(board.filterRooms(rooms, { floor: '2' }).length, 2);
  assert.equal(board.filterRooms(rooms, { search: ' 103 ' }).length, 1);
  assert.equal(board.roomPresentation(room, true).kind, 'connection');
  assert.equal(board.roomPresentation(paused).kind, 'paused');
  assert.equal(board.roomPresentation(uncertain).label, '확인 중');
  const card = renderToStaticMarkup(React.createElement(board.RoomCard, { room, onSelect() {} }));
  assert.match(card, /불 확인 필요/);
  assert.match(card, /사용 여부 예상.*사용 중/);
  assert.doesNotMatch(card, /v7|70%/);
  const reason = renderToStaticMarkup(React.createElement(board.LightCheckPanel, { room }));
  assert.match(reason, /300 lux 이상.*650 ppm 이하.*10분 지속/);
  assert.match(reason, /모델은 사용 중으로 예상했지만/);
  assert.match(reason, /햇빛.*환기/);
  assert.match(reason, /실제 시설에서는 기준 교정/);
  const offline = renderToStaticMarkup(React.createElement(board.LightCheckPanel, { room, disconnected: true }));
  assert.doesNotMatch(offline, /불 확인 필요/);
  const pausedCard = renderToStaticMarkup(React.createElement(board.RoomCard, { room: paused, onSelect() {} }));
  assert.doesNotMatch(pausedCard, /불 확인 필요/);
  assert.match(pausedCard, /현재 실시간 상태가 아닙니다/);
  const history = [0, 60, 300].map((second, index) => ({ accepted_as_latest: true,
    observed_at: new Date(Date.parse('2026-10-29T12:00:00Z') + second * 1000).toISOString(), sensors: { Humidity: 28 + index } }));
  const trend = renderToStaticMarkup(React.createElement(board.SensorTrend, { history, sensor: 'Humidity' }));
  assert.match(trend, /습도 변화/);
  assert.match(trend, /처음 대비 \+2/);
  assert.match(trend, /d="M[^"M]*M/);
  const backward = renderToStaticMarkup(React.createElement(board.SensorTrend, { history: [...history].reverse(), sensor: 'Humidity' }));
  assert.match(backward, /<path d="M284[^"L]*" class="trend-line humidity"/);
  const initial = renderToStaticMarkup(React.createElement(board.default));
  assert.match(initial, /불을 확인할 강의실을 먼저/);
  assert.match(initial, /별도 순찰 참고 신호/);
  assert.match(initial, /05:00~24:00/);
  assert.match(initial, /AWS EC2 배포/);
  console.log('PASS: patrol priority/filter; model coexistence; connection/pause suppression; thresholds/limits; humidity trend gap; initial board');
  console.log('PASS: JSX rendering; SHA-256 + row grouping; per-group best; source distributions; preflight disable; preview form ownership; unreadable job display');
} finally {
  await server.close();
  await fs.rm(cacheDir, { recursive: true, force: true });
}
