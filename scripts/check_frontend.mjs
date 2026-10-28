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
      + '\nexport { evaluationGroups, Dashboard, TrainingForm, ReportPanel, reusableTrainingConfig, DEFAULT_CONFIG, PRESETS };';
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
  console.log('PASS: JSX rendering; SHA-256 + row grouping; per-group best; source distributions; preflight disable; preview form ownership');
} finally {
  await server.close();
  await fs.rm(cacheDir, { recursive: true, force: true });
}
