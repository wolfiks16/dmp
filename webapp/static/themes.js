// ТЕМЫ ОФОРМЛЕНИЯ (см. themes.css): «Лист ЕСКД», «Испытательный стенд», «Привычная тёмная». Переключаются в ⚙
// и запоминаются в этом браузере; тему ставит ещё скрипт в <head>, до первой отрисовки. Холст 2D, графики и вид
// 3D читают цвета при рисовании — после смены темы они перерисовываются, перезагружать страницу не нужно.
// У «Листа ЕСКД» поверх вида — рамка листа и штамп с живыми данными проекта (название, сценарий с температурой,
// сетка, дата); «Снимок вида (PNG)» в этой теме выходит с той же рамкой и штампом.
const $ = id => document.getElementById(id);
const THEMES = ['eskd', 'stand', 'refresh'], DEFAULT_THEME = 'eskd', KEY = 'magfield-theme';
const theme = () => document.documentElement.dataset.theme;

// ---------------------------------------------------------------- лист: рамка и штамп
const vp = document.querySelector('.viewport');
const frame = document.createElement('div');
frame.className = 'sheet-frame';
frame.setAttribute('aria-hidden', 'true');
frame.innerHTML = '<div class="stamp"><span>Проект</span><b id="stp-proj"></b><span>Сценарий</span><b id="stp-scn"></b>'
  + '<span>Сетка</span><b id="stp-mesh"></b><span>Дата</span><b id="stp-date"></b></div>';
vp.appendChild(frame);
const stamp = frame.querySelector('.stamp');

const plural = (n, one, few, many) => {
  const a = n % 100, b = n % 10;
  return (a >= 11 && a <= 14) ? many : b === 1 ? one : (b >= 2 && b <= 4) ? few : many;
};
// Строки штампа: [подпись, значение] — из тех же переменных страницы, по которым идёт расчёт.
function stampRows() {
  const d3 = MODE === 'objects3d', W3 = window.WS3D;
  const proj = ((PROJECT && PROJECT.name) || '').trim() || '—';
  let scn;
  if (d3) {
    const r = W3 && W3.result(), T = (r && r.T != null) ? r.T : (W3 ? W3.temperature() : 20);
    scn = 'магнитостатика 3D, ' + T + ' °C';
  } else if (SCENARIO === 'dynamic') scn = 'нагрев во времени, среда ' + (+$('s3-tamb').value) + ' °C';
  else scn = (SCN_NAME[SCENARIO] || '').toLowerCase() + ', ' + (SCENARIO === 'thermostatic' ? +$('T').value : 20) + ' °C';
  const n = d3 ? (W3 && W3.cells()) : (SCENE ? SCENE.cells.length : null);
  const mesh = n ? n.toLocaleString('ru-RU') + ' ' + plural(n, 'ячейка', 'ячейки', 'ячеек') : 'не построена';
  return [['Проект', proj], ['Сценарий', scn], ['Сетка', mesh], ['Дата', new Date().toLocaleDateString('ru-RU')]];
}
function updateStamp() {
  if (!STEPS_ON || theme() !== 'eskd') return;
  stamp.hidden = noProject();
  const rows = stampRows();
  ['proj', 'scn', 'mesh', 'date'].forEach((k, i) => { const el = $('stp-' + k); el.textContent = rows[i][1]; el.title = rows[i][1]; });
  // кнопки масштаба — над штампом (штамп прижат к рамке, рамка — в 8 px от края вида)
  vp.style.setProperty('--stamp-h', (stamp.hidden || !stamp.offsetHeight ? 18 : stamp.offsetHeight + 16) + 'px');
}
// Штамп обновляется там же, где строка шагов: смена проекта, сетки, сценария, результата.
const updateStepsBase = window.updateSteps;
window.updateSteps = function () { updateStepsBase.apply(this, arguments); updateStamp(); };
window.addEventListener('resize', updateStamp);

// ---------------------------------------------------------------- снимок вида 3D
function save(href, name) {
  const a = document.createElement('a');
  a.href = href; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}
const pngName = () => (((PROJECT && PROJECT.name) || 'model3d').trim() || 'model3d') + '.png';
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
// Кадр вида + рамка листа + штамп теми же строками, что на экране.
async function stampedPng(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const W = img.width, H = img.height, k = W / Math.max(1, $('v3d').clientWidth);   // пикселей кадра на пиксель экрана
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.drawImage(img, 0, 0);
  const ink = css('--text') || '#1e262d', muted = css('--muted') || '#4f5b66';
  const face = "Bahnschrift, 'Arial Narrow', sans-serif", fs = 12 * k, pad = 8 * k, rh = 20 * k, m = 8 * k;
  g.strokeStyle = ink; g.lineWidth = 1.5 * k;
  g.strokeRect(m, m, W - 2 * m, H - 2 * m);                                          // рамка листа
  const rows = stampRows();
  g.font = '500 ' + fs + 'px ' + face;
  const w1 = Math.max(...rows.map(r => g.measureText(r[0]).width)) + 2 * pad;
  g.font = '600 ' + fs + 'px ' + face;
  const w2 = Math.max(150 * k, ...rows.map(r => g.measureText(r[1]).width + 2 * pad));
  const x0 = W - m - w1 - w2, y0 = H - m - rows.length * rh, sh = rows.length * rh;
  g.fillStyle = '#ffffff'; g.fillRect(x0, y0, w1 + w2, sh);
  g.lineWidth = 1.5 * k; g.strokeRect(x0, y0, w1 + w2, sh);
  g.lineWidth = 1 * k; g.beginPath();
  g.moveTo(x0 + w1, y0); g.lineTo(x0 + w1, y0 + sh);
  for (let i = 1; i < rows.length; i++) { g.moveTo(x0, y0 + i * rh); g.lineTo(x0 + w1 + w2, y0 + i * rh); }
  g.stroke();
  g.textBaseline = 'middle';
  rows.forEach((r, i) => {
    const y = y0 + (i + 0.5) * rh;
    g.font = '500 ' + fs + 'px ' + face; g.fillStyle = muted; g.fillText(r[0], x0 + pad, y);
    g.font = '600 ' + fs + 'px ' + face; g.fillStyle = ink; g.fillText(r[1], x0 + w1 + pad, y);
  });
  return c.toDataURL('image/png');
}
$('r3-png').onclick = async () => {
  const url = window.WS3D && WS3D.screenshot();
  if (!url) return;
  save(theme() === 'eskd' ? await stampedPng(url) : url, pngName());
};

// ---------------------------------------------------------------- переключение темы
function applyTheme(name) {
  if (!THEMES.includes(name)) name = DEFAULT_THEME;
  document.documentElement.dataset.theme = name;
  try { localStorage.setItem(KEY, name); } catch (e) { /* без хранилища — тема до перезагрузки */ }
  document.querySelectorAll('input[name="theme"]').forEach(r => { r.checked = r.value === name; });
  readTheme();                                            // цвета холста 2D и графиков — из новой темы
  if (MODE === 'objects3d') { if (window.WS3D) WS3D.applyTheme(); }
  else { resize(); bars(); draw(); }
  if (STAGE === 'result' && TOOL !== 'view') runProbe();   // графики вдоль линии и окружности
  if (STAGE === 'mat') drawCurve();                        // кривая материала
  restorePost();                                           // момент, потери, сценарий нагрева
  updateStamp();
}
document.querySelectorAll('input[name="theme"]').forEach(r => { r.onchange = () => { if (r.checked) applyTheme(r.value); }; });
document.querySelectorAll('input[name="theme"]').forEach(r => { r.checked = r.value === theme(); });
updateStamp();
