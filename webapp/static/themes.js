// ТЕМЫ ОФОРМЛЕНИЯ (см. themes.css): «Белая», «Серая», «Цветная». Выбор в ⚙ «Оформление» применяется сразу;
// при запуске открывается ТЕМА ПО УМОЛЧАНИЮ — она хранится на сервере (одна для всех браузеров, переживает
// перезапуск) и меняется кнопкой «Сделать по умолчанию». Копия её в браузере — только чтобы скрипт в <head>
// поставил тему до первой отрисовки, без мигания. Холст 2D, графики и вид 3D читают цвета при рисовании —
// после смены темы они перерисовываются, перезагружать страницу не нужно.
// У «Белой» поверх вида — рамка листа и штамп с живыми данными проекта (название, сценарий с температурой,
// сетка, дата); «Снимок вида (PNG)» в этой теме выходит с той же рамкой и штампом.
const $ = id => document.getElementById(id);
const THEMES = ['white', 'grey', 'color'], FALLBACK = 'color', CACHE = 'magfield-theme-default';
const NAMES = { white: 'Белая', grey: 'Серая', color: 'Цветная' };
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
  if (!STEPS_ON || theme() !== 'white') return;
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
  save(theme() === 'white' ? await stampedPng(url) : url, pngName());
};

// ---------------------------------------------------------------- переключение темы
let DEFAULT_T = FALLBACK;
try { const c = localStorage.getItem(CACHE); if (THEMES.includes(c)) DEFAULT_T = c; } catch (e) { /* без хранилища */ }
let PICKED = false;                                       // тему уже выбрали в этом сеансе — умолчание её не перебивает
function applyTheme(name) {
  if (!THEMES.includes(name)) name = FALLBACK;
  document.documentElement.dataset.theme = name;
  readTheme();                                            // цвета холста 2D и графиков — из новой темы
  if (MODE === 'objects3d') { if (window.WS3D) WS3D.applyTheme(); }
  else { resize(); bars(); draw(); }
  if (STAGE === 'result' && TOOL !== 'view') runProbe();   // графики вдоль линии и окружности
  if (STAGE === 'mat') drawCurve();                        // кривая материала
  restorePost();                                           // момент, потери, сценарий нагрева
  updateStamp();
  syncThemeUI();
}
// Выбор в настройках: отмечен текущий; «по умолчанию» — у темы, с которой открывается приложение.
function syncThemeUI() {
  document.querySelectorAll('input[name="theme"]').forEach(r => {
    r.checked = r.value === theme();
    r.closest('.theme-opt').classList.toggle('is-default', r.value === DEFAULT_T);
  });
  const b = $('theme-default');
  if (b) {
    b.disabled = theme() === DEFAULT_T;
    b.title = b.disabled ? 'Эта тема уже по умолчанию' : 'Открывать приложение в теме «' + NAMES[theme()] + '»';
  }
}
function rememberDefault(name) {
  DEFAULT_T = name;
  try { localStorage.setItem(CACHE, name); } catch (e) { /* без хранилища — только до перезагрузки */ }
}
document.querySelectorAll('input[name="theme"]').forEach(r => {
  r.onchange = () => { if (r.checked) { PICKED = true; applyTheme(r.value); } };
});
$('theme-default').onclick = async () => {
  const name = theme();
  let d;
  try {
    d = await (await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ default_theme: name }) })).json();
  } catch (e) { toast('Сервер недоступен — тема по умолчанию не сохранена.', 'warn'); return; }
  if (d.error) { toast(d.error, 'warn'); return; }
  rememberDefault(d.default_theme);
  syncThemeUI();
  toast('Тема по умолчанию — «' + NAMES[d.default_theme] + '»');
};
// Тема по умолчанию — с сервера: если в этом браузере другая копия и тему ещё не выбирали — перейти на неё.
(async () => {
  try {
    const d = await (await fetch('/api/settings')).json();
    if (THEMES.includes(d.default_theme)) {
      rememberDefault(d.default_theme);
      if (!PICKED && theme() !== d.default_theme) applyTheme(d.default_theme);
    }
  } catch (e) { /* сервер недоступен — остаётся копия из браузера */ }
  syncThemeUI();
})();
try { localStorage.removeItem('magfield-theme'); } catch (e) { /* прежний ключ (тема сохранялась в браузере) */ }
syncThemeUI();
updateStamp();
