// 3D-РЕЖИМ ИНТЕРФЕЙСА (этап 3D-5, план — docs/plan_3d_2026-09-11.md): свободная геометрия из тел,
// объёмный вид на three.js (viewer3d.js), расчёт на сервере (magcore.fem3d, обработчики /api/3d/*).
// Состояние 3D живёт здесь. Общие части страницы — проект, сохранение в рабочую папку, фоновые
// задачи, библиотека материалов, окно проверки перед расчётом — берутся из основного скрипта
// index.html (его глобальные функции и переменные). Единицы в интерфейсе — мм и градусы.
import { Viewer3D } from './viewer3d.js';

const KIND = { box: 'Параллелепипед', cylinder: 'Цилиндр', tube: 'Труба', tube_sector: 'Сектор трубы', sphere: 'Шар', prism: 'Призма',
  step: 'Тело из STEP' };
const DEF = { box: { lx: 10, ly: 10, lz: 5 }, cylinder: { r: 5, h: 10 }, tube: { r_in: 3, r_out: 6, h: 8 },
  tube_sector: { r_in: 10, r_out: 15, h: 8, a1: 0, a2: 90 }, sphere: { r: 5 },
  prism: { points: [[0, 0], [10, 0], [5, 8]], h: 5 } };
const PAR = { lx: 'Длина по X', ly: 'Ширина по Y', lz: 'Высота по Z', r: 'Радиус', h: 'Высота вдоль оси',
  r_in: 'Радиус внутр.', r_out: 'Радиус внешн.', a1: 'Угол от', a2: 'Угол до' };
const DIRS = [['axial', 'вдоль оси тела'], ['axial-in', 'против оси тела'], ['radial', 'радиально наружу'],
  ['radial-in', 'радиально внутрь'], ['x', 'по +X'], ['y', 'по +Y'], ['z', 'по +Z']];
// У тела из CAD своей оси нет: «осевое» и «радиальное» — относительно оси двигателя, заданной в теле.
const DIRS_STEP = [['axial', 'вдоль оси двигателя'], ['axial-in', 'против оси двигателя'], ['radial', 'радиально от оси'],
  ['radial-in', 'радиально к оси'], ['x', 'по +X'], ['y', 'по +Y'], ['z', 'по +Z']];
const VEC = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const QTY = [['B', 'B — модуль индукции'], ['Bx', 'Bx'], ['By', 'By'], ['Bz', 'Bz'], ['H', 'H — модуль напряжённости'],
  ['mu', 'μ — относительная проницаемость'], ['margin', '⚠ Запас до колена (магниты)'],
  ['loss', 'Потеря B_r (магниты)'], ['Hpar', 'H вдоль оси намагничивания (магниты)']];
const SIGNED = new Set(['Bx', 'By', 'Bz', 'Hpar']);
const FROM_ZERO = new Set(['B', 'H', 'loss']);
const MAGNET_ONLY = new Set(['margin', 'loss', 'Hpar']);
const MATCOL = { magnet: '#9a72d6', steel: '#8391a6', linear: '#4d8bff', air: '#5d6d8a' };
const AIR_RGB = [28, 37, 54], GRAY_RGB = [70, 80, 96], LINE_ON_SECTION = [238, 243, 252];

const fresh = () => ({ objects: [], sel: -1, h: 2, margin: 2, grading: 2, T: 20, bc: 'neumann', H0: [0, 0, 0],
  model: null, modelKey: '', result: null, values: null, range: null, unit: '', q: 'B', pal: 'viridis',   // равномерная палитра
  field3d: null, restoring: false, restoreNote: '',     // сетка и φ решения для файла расчёта (этап 3D-9)
  sec: { axis: 'off', pos: 0, flip: false, lo: -50, hi: 50 }, secData: null, secSeq: 0, secTimer: 0,
  bodies: new Set(), opacity: 1, pvTimer: 0, pvSeq: 0, pvBBox: null, pvAxes: null, pvArrows: null, pvArrowsOn: false,
  cells: {}, regionKey: {}, regionRGB: {}, stepFiles: {}, linesData: null, secLinesData: null });
// Показ осей, стрелок и силовых линий — настройка вида, а не расчёта: живёт вне fresh() и не сбрасывается
// при смене проекта.
const S = Object.assign(fresh(), { viewer: null, active: false, showCoordAxes: true, showBodyAxes: true,
  showArrows: false, showLines: false, showSecLines: false, linesN: 200 });

// ---------------------------------------------------------------- мелочи
const post = async (url, body) => (await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body) })).json();
function b64(s, Ctor) { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Ctor(u.buffer); }
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const kindOf = mid => mid === 'air' ? 'air' : STEELS.some(s => s.id === mid) ? 'steel' : 'magnet';
const matColor = o => MATCOL[kindOf(o.material)];
const hexRgb = h => { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
const keyOf = i => 'o' + i;
const idxOf = k => (k && k[0] === 'o') ? +k.slice(1) : -1;
const hasH0 = () => S.H0.some(x => +x !== 0);
const pct = x => (100 * x).toFixed(x > 0 && x < 0.01 ? 2 : 1) + ' %';
// Потеря потока меньше 10⁻⁶ — шум остановки решателя (невязка 10⁻⁹ от начальной), показывается как 0.
const LOSS_NOISE = 1e-6;
function fmt(v, d = 3) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return String(+v.toFixed(d));
}
function V() { if (!S.viewer) S.viewer = new Viewer3D($('v3d'), { onPick: k => select(idxOf(k)) }); return S.viewer; }
function showErr(msg) {
  const e = $('g3-err'); if (e) { e.textContent = msg; e.style.display = msg ? 'block' : 'none'; }
  if (msg) setConv('c', '● ' + msg);
}
function download(href, name) { const a = document.createElement('a'); a.href = href; a.download = name; document.body.appendChild(a); a.click(); a.remove(); }

// ---------------------------------------------------------------- модель: тела
const dirOut = d => VEC[d] || d || 'axial';
function payload() {
  return S.objects.map(o => ({ name: (o.name || '').trim(), kind: o.kind, params: o.params, material: o.material,
    magnet_dir: dirOut(o.magnet_dir), center: o.center, rotation: o.rotation,
    // поворот намагниченности, ° — только если задан: у прежних моделей тело запроса (и ключ модели) не меняется
    magnet_rotation: (o.magnet_rotation || []).some(a => +a) ? o.magnet_rotation.map(Number) : undefined,
    mesh_size_mm: o.mesh_size_mm || undefined, priority: o.priority || 1 }));
}
const modelBody = () => ({ objects: payload(), default_mesh_mm: S.h, margin: S.margin, grading: S.grading });
function checkNames() {
  const nm = S.objects.map(o => (o.name || '').trim());
  if (nm.some(n => !n)) return 'У каждого тела должно быть имя.';
  const dup = [...new Set(nm.filter((n, i) => nm.indexOf(n) !== i))];
  if (dup.length) return 'Имена тел должны быть разными: ' + dup.join(', ') + '.';
  if (nm.includes('domain')) return 'Имя «domain» занято фоновой областью.';
  return null;
}
function newObject(kind) {
  const n = S.objects.filter(o => o.kind === kind).length + 1;
  let name = KIND[kind] + ' ' + n;
  while (S.objects.some(o => o.name === name)) name += '′';
  const p = DEF[kind], half = kind === 'box' ? p.lx / 2 : kind === 'prism' ? 5 : (p.r_out || p.r || 5);
  const c = [0, 0, 0];
  if (S.pvBBox && S.objects.length) c[0] = Math.round(S.pvBBox[1][0] + 8 + half);    // правее уже стоящих
  return { name, kind, params: JSON.parse(JSON.stringify(p)), material: 'steel', magnet_dir: 'axial', center: c,
    rotation: [0, 0, 0], mesh_size_mm: '', priority: 10, hidden: false };
}

// ---------------------------------------------------------------- тела из STEP (этап 3D-1б)
// Файл уходит на сервер телом запроса и хранится там под SHA-256; сами байты лежат и в файле расчёта
// (.mfz) — расчёт переносим целиком, а сервер после перезапуска получает файл заново (ensureStepFiles).
function bytesB64(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
const usedStepIds = () => [...new Set(S.objects.filter(o => o.kind === 'step').map(o => o.params.file_id))];
function pruneStepFiles() {
  const used = new Set(usedStepIds());
  for (const id of Object.keys(S.stepFiles)) if (!used.has(id)) delete S.stepFiles[id];
}
async function uploadStep(name, bytes) {
  const r = await fetch('/api/3d/step_upload?name=' + encodeURIComponent(name),
    { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: bytes });
  return r.json();
}
async function ensureStepFiles() {
  const ids = usedStepIds();
  if (!ids.length) return true;
  let d;
  try { d = await post('/api/3d/step_has', { file_ids: ids }); } catch (e) { showErr('Сервер недоступен: ' + e); return false; }
  for (const id of d.missing || []) {
    const f = S.stepFiles[id];
    if (!f || !f.b64) { showErr('В расчёте нет STEP-файла «' + ((f && f.name) || id.slice(0, 8)) + '» — импортируйте его заново.'); return false; }
    let r;
    try { r = await uploadStep(f.name, b64(f.b64, Uint8Array)); } catch (e) { showErr('Сервер недоступен: ' + e); return false; }
    if (r.error || r.file_id !== id) { showErr('STEP-файл «' + f.name + '» не принят сервером: ' + (r.error || 'содержимое не совпало')); return false; }
  }
  return true;
}
async function stepFileChosen(file) {
  if (!file) return;
  setConv('', '◐ Читаю STEP «' + file.name + '»…');
  let bytes, d;
  try { bytes = new Uint8Array(await file.arrayBuffer()); d = await uploadStep(file.name, bytes); }
  catch (e) { setConv('c', '● Сервер недоступен: ' + e); return; }
  if (d.error) { setConv('c', '● ' + d.error); toast(d.error, 'warn'); return; }
  setConv('', '● В файле «' + d.name + '» тел: ' + d.bodies.length);
  importStepBodies(d, bytesB64(bytes));
}
async function importStepBytes(name, bytes) {                 // то же без окна выбора файла (проверка, сценарии)
  const d = await uploadStep(name, bytes);
  if (d.error) return d;
  importStepBodies(d, bytesB64(bytes));
  return { file_id: d.file_id, bodies: d.bodies.length };
}
// Одинаковые тела — тот же объём (до 1e-9) и то же число граней: материал и намагничивание — на всю группу.
function stepGroups(bodies) {
  const groups = [];
  for (const b of bodies) {
    const g = groups.find(x => x.faces === b.n_faces && Math.abs(x.volume - b.volume_mm3) <= 1e-9 * Math.max(x.volume, b.volume_mm3));
    if (g) g.bodies.push(b); else groups.push({ faces: b.n_faces, volume: b.volume_mm3, bodies: [b] });
  }
  return groups;
}
// Ось двигателя — подсказка, её видно и можно сменить: у самой многочисленной группы одинаковых тел (≥ 3)
// ищем координату, вдоль которой их центры не меняются; точка на оси — центр окружности центров, если они
// на ней лежат, иначе начало координат CAD. Остаток округления среднего (порядка n·2e-16·r) обнуляем при
// |v| ≤ 1e-9·r: ось сдвигается меньше чем на 1e-9·r, а круг тел мы и так признаём с допуском 1e-6·r.
function guessAxis(groups) {
  const none = { key: 'z', origin: [0, 0, 0], dir: [0, 0, 1], guessed: false };
  const g = groups.filter(x => x.bodies.length >= 3).sort((a, b) => b.bodies.length - a.bodies.length)[0];
  if (!g) return none;
  const C = g.bodies.map(b => b.centroid_mm);
  const span = k => Math.max(...C.map(c => c[k])) - Math.min(...C.map(c => c[k]));
  const size = Math.max(span(0), span(1), span(2));
  const k = [0, 1, 2].find(j => span(j) <= 1e-6 * size);
  if (!(size > 0) || k === undefined) return none;
  const dir = [0, 0, 0]; dir[k] = 1;
  const mean = [0, 1, 2].map(j => C.reduce((s, c) => s + c[j], 0) / C.length);
  const dist = C.map(c => Math.hypot(...[0, 1, 2].filter(j => j !== k).map(j => c[j] - mean[j])));
  const r = dist.reduce((s, x) => s + x, 0) / dist.length;
  const onCircle = r > 0 && Math.max(...dist.map(x => Math.abs(x - r))) <= 1e-6 * r;
  const origin = onCircle ? mean.map((v, j) => (j === k || Math.abs(v) <= 1e-9 * r ? 0 : v)) : [0, 0, 0];
  return { key: 'xyz'[k], origin, dir, guessed: true };
}
function matOptions(cur) {
  const opt = (id, name) => '<option value="' + esc(id) + '"' + (id === cur ? ' selected' : '') + '>' + esc(name) + '</option>';
  return opt('air', 'Воздух') + '<optgroup label="Сталь">' + STEELS.map(s => opt(s.id, s.name)).join('') + '</optgroup>'
    + '<optgroup label="Магниты">' + MAGNETS.map(m => opt(m.id, m.name)).join('') + '</optgroup>';
}
function importStepBodies(d, data64) {
  const groups = stepGroups(d.bodies), ax = guessAxis(groups);
  const st = { axis: ax.key, origin: ax.origin.slice(), dir: ax.dir.slice(), rows: groups.map(() => ({ on: true, material: 'steel', dir: 'radial' })) };
  let ov = $('step3d-ov');
  if (!ov) { ov = document.createElement('div'); ov.className = 'ui-modal-ov'; ov.id = 'step3d-ov'; document.body.appendChild(ov); }
  const close = () => ov.classList.remove('on');
  const nums = g => { const n = g.bodies.map(b => b.index + 1); return n.length <= 4 ? n.join(', ') : n.slice(0, 3).join(', ') + ' … ' + n[n.length - 1]; };
  const size = b => b.bbox_mm[1].map((v, k) => fmt(v - b.bbox_mm[0][k], 1)).join(' × ');
  const render = () => {
    const axBtn = k => '<button data-stax="' + k + '"' + (st.axis === k ? ' class="on"' : '') + '>' + (k === 'custom' ? 'своя' : k.toUpperCase()) + '</button>';
    const vin = (attr, a) => '<div class="vec3">' + a.map((x, k) => '<input class="num" data-' + attr + '="' + k + '" value="' + esc(fmt(x, 4)) + '">').join('') + '</div>';
    let h = '<div class="ui-modal-msg" style="margin-bottom:6px">Тела из «' + esc(d.name) + '»: ' + d.bodies.length + '</div>'
      + '<p class="hint" style="margin:0 0 8px">Одинаковые тела (тот же объём и число граней) собраны в одну строку — материал и намагничивание назначаются всей строке. Размеры тел не меняются, положение — как в CAD.</p>'
      + '<div class="stp-scroll"><table class="stp-tab"><thead><tr><th></th><th>Тела №</th><th>Шт.</th><th>Объём, мм³</th><th>Габарит X × Y × Z, мм</th><th>Граней</th><th>Материал</th><th>Намагничивание</th></tr></thead><tbody>'
      + groups.map((g, i) => {
        const r = st.rows[i], mag = kindOf(r.material) === 'magnet';
        return '<tr><td><input type="checkbox" data-ston="' + i + '"' + (r.on ? ' checked' : '') + '></td><td class="n">' + nums(g) + '</td><td class="n">'
          + g.bodies.length + '</td><td class="n">' + fmt(g.volume, 2) + '</td><td class="n">' + size(g.bodies[0]) + '</td><td class="n">' + g.faces
          + '</td><td><select class="select" data-stmat="' + i + '">' + matOptions(r.material) + '</select></td><td>'
          + (mag ? '<select class="select" data-stdir="' + i + '">' + DIRS_STEP.map(x => '<option value="' + x[0] + '"' + (x[0] === r.dir ? ' selected' : '') + '>' + x[1] + '</option>').join('') + '</select>' : '—')
          + '</td></tr>';
      }).join('')
      + '</tbody></table></div>'
      + '<div class="stp-axis"><span>Ось двигателя — для осевого и радиального намагничивания:</span><div class="rseg">' + ['x', 'y', 'z', 'custom'].map(axBtn).join('') + '</div></div>'
      + (st.axis === 'custom'
        ? '<div class="stp-axis"><span>точка на оси, мм</span>' + vin('storg', st.origin) + '<span>направление</span>' + vin('stdv', st.dir) + '</div>'
        : '<p class="hint" style="margin:6px 0 0">Точка на оси: (' + st.origin.map(v => fmt(v, 3)).join('; ') + ') мм'
          + (ax.guessed && st.axis === ax.key ? ' — ось найдена по кругу одинаковых тел; проверьте.' : '.') + '</p>')
      + '<div class="ui-modal-btns" style="margin-top:14px"><button class="btn pri" id="stp-ok">Импортировать</button><button class="btn" id="stp-cancel">Отмена</button></div>';
    ov.innerHTML = '<div class="ui-modal stp-modal">' + h + '</div>';
    ov.querySelectorAll('[data-ston]').forEach(el => { el.onchange = () => { st.rows[+el.dataset.ston].on = el.checked; }; });
    ov.querySelectorAll('[data-stmat]').forEach(el => { el.onchange = () => { st.rows[+el.dataset.stmat].material = el.value; render(); }; });
    ov.querySelectorAll('[data-stdir]').forEach(el => { el.onchange = () => { st.rows[+el.dataset.stdir].dir = el.value; }; });
    ov.querySelectorAll('[data-stax]').forEach(el => {
      el.onclick = () => {
        st.axis = el.dataset.stax;
        if (st.axis !== 'custom') {
          st.dir = [0, 0, 0]; st.dir['xyz'.indexOf(st.axis)] = 1;
          st.origin = st.axis === ax.key ? ax.origin.slice() : [0, 0, 0];
        }
        render();
      };
    });
    const vecIn = (attr, arr) => ov.querySelectorAll('[data-' + attr + ']').forEach(el => {
      el.onchange = () => { const x = +el.value, k = +el.getAttribute('data-' + attr); if (el.value.trim() !== '' && Number.isFinite(x)) arr[k] = x; else el.value = arr[k]; };
    });
    vecIn('storg', st.origin); vecIn('stdv', st.dir);
    $('stp-cancel').onclick = () => { close(); pruneStepFiles(); };
    $('stp-ok').onclick = () => {
      if (st.dir.every(v => v === 0)) { toast('Направление оси не может быть нулевым.', 'warn'); return; }
      const stem = d.name.replace(/\.(stp|step)$/i, ''), made = [];
      groups.forEach((g, i) => {
        const r = st.rows[i];
        if (!r.on) return;
        for (const b of g.bodies) {
          let name = stem + ' · тело ' + (b.index + 1);
          while (S.objects.some(o => o.name === name) || made.some(o => o.name === name)) name += '′';
          made.push({ name, kind: 'step', material: r.material, magnet_dir: kindOf(r.material) === 'magnet' ? r.dir : 'axial',
            params: { file_id: d.file_id, body: b.index, volume_mm3: b.volume_mm3, centroid_mm: b.centroid_mm.slice(),
              axis_origin_mm: st.origin.slice(), axis_dir: st.dir.slice() },
            center: [0, 0, 0], rotation: [0, 0, 0], mesh_size_mm: '', priority: 10, hidden: false });
        }
      });
      close();
      if (!made.length) { pruneStepFiles(); return; }
      S.stepFiles[d.file_id] = { name: d.name, size: d.size, b64: data64, bodies: d.bodies };
      S.objects.push(...made); S.sel = S.objects.length - made.length;
      changed(); setStage('geom');
      toast('Импортировано тел: ' + made.length, 'ok');
    };
  };
  render();
  ov.classList.add('on');
}

// Любая правка геометрии или сетки обесценивает сетку и решение: назад к предпросмотру.
function dropModel() {
  const had = !!S.model || !!S.result;
  S.model = null; S.result = null; S.values = null; S.secData = null; S.cells = {};
  S.field3d = null; S.restoreNote = '';
  S.linesData = null; S.secLinesData = null;
  if (S.viewer) S.viewer.setSection(null);
  if (had) { $('st-cells').textContent = ''; $('st-iters').textContent = ''; renderResults(); updateLegend(); updateResults(); }
}
function changed() { dropModel(); markDirty(); renderList(); schedulePreview(); }

// ---------------------------------------------------------------- предпросмотр и сетка
function schedulePreview() { clearTimeout(S.pvTimer); S.pvTimer = setTimeout(preview, 250); }
async function preview() {
  if (!S.active || S.model) return;
  const v = V();
  if (!S.objects.length) {
    v.setObjects([]); S.pvBBox = null; S.pvAxes = null; S.pvArrows = null; updateAxes(); updateArrows(); showErr(''); return;
  }
  const seq = ++S.pvSeq;
  if (S.objects.some(o => o.kind === 'step') && !(await ensureStepFiles())) return;
  if (seq !== S.pvSeq || S.model || !S.active) return;
  const withArrows = S.showArrows;                   // стрелки считаются только по запросу (≈ 0,4 мс на точку проверки)
  let d;
  try { d = await post('/api/3d/preview', { objects: payload().map((o, i) => ({ ...o, name: 'o' + i })), arrows: withArrows }); }
  catch (e) { if (!S.model) showErr('Сервер недоступен: ' + e); return; }
  if (seq !== S.pvSeq || !S.active) return;                           // устарело
  if (d.error) { if (!S.model) showErr(d.error.replace(/^o(\d+):/, (m, i) => '«' + ((S.objects[+i] || {}).name || '?') + '»:')); return; }
  S.pvBBox = d.bbox;
  S.pvAxes = d.objects.map(o => o.axis || null);             // оси тел — от сервера, той же функцией, что намагничивание
  S.pvArrows = d.objects.map(o => o.arrows || null); S.pvArrowsOn = withArrows;
  if (S.model) { updateAxes(); return; }        // сетка уже есть (поле открыто из файла, пока шёл предпросмотр) — вид не трогаем
  showErr('');
  const refit = S.fitCount !== S.objects.length;               // тело добавили или удалили — вписать заново
  S.fitCount = S.objects.length;
  v.setObjects(d.objects.map((o, i) => ({ key: keyOf(i), tris: b64(o.tris, Float32Array), color: matColor(S.objects[i]) })), { keepView: !refit });
  afterObjects();
}
function afterObjects() {
  const v = V();
  S.objects.forEach((o, i) => v.setVisible(keyOf(i), !o.hidden));
  v.setSelected(S.sel >= 0 ? keyOf(S.sel) : null);
  v.setOpacity(S.opacity);
  updateSecRange(false); applyClip(); updateAxes(); updateArrows();
}
// Силовые линии (этап 3D-7): считает сервер по решению. Объёмные — каждая несёт одинаковый поток ΔΦ (где
// гуще, там больше индукция); линии на разрезе расставлены равномерно и верны как линии поля только там, где
// поле лежит в плоскости (доля выхода из плоскости — в строке состояния). Цвет — по |B| в точке, палитра и
// шкала те же, что у поля (от нуля до наибольшего |B| расчёта).
async function fetchLines(url, body, key) {
  try {
    const d = await post(url, body);
    if (d.error) { setConv('c', '● ' + d.error); toast(d.error, 'warn'); return null; }
    return { points: b64(d.points, Float32Array), offsets: b64(d.offsets, Uint32Array),
      values: b64(d.values, Float32Array), n: d.n_lines, dFlux: d.delta_flux_Wb, out: d.out_of_plane, key };
  } catch (e) { setConv('c', '● Сервер недоступен: ' + e); return null; }
}
async function updateLines() {
  const v = V();
  const wantVol = S.showLines && S.model && S.result;
  const wantSec = S.showSecLines && S.model && S.result && S.sec.axis !== 'off';
  if (!wantVol && !wantSec) { v.setFieldLines(null); return; }
  const secKey = S.sec.axis + ':' + S.sec.pos;
  if (wantVol && !S.linesData) {
    setConv('', '◐ Строю силовые линии…');
    S.linesData = await fetchLines('/api/3d/field_lines', { model_id: S.model.model_id, n_lines: S.linesN }, '');
    if (!S.linesData) { $('ln3d-show').checked = S.showLines = false; }
    else setConv('', '● Силовых линий: ' + S.linesData.n + ' · поток на линию ' + fmt(S.linesData.dFlux * 1e6, 3) + ' мкВб');
  }
  if (wantSec && (!S.secLinesData || S.secLinesData.key !== secKey)) {
    setConv('', '◐ Строю линии на разрезе…');
    const n = VEC[S.sec.axis];
    S.secLinesData = await fetchLines('/api/3d/section_lines',
      { model_id: S.model.model_id, n_lines: S.linesN, point_mm: n.map(c => c * S.sec.pos), normal: n }, secKey);
    if (!S.secLinesData) { $('ln3d-sec').checked = S.showSecLines = false; }
    else setConv('', '● Линий на разрезе: ' + S.secLinesData.n + ' · поле выходит из плоскости на '
      + pct(S.secLinesData.out) + (S.secLinesData.out < 0.05 ? ' — это линии поля' : ' — это проекция'));
  }
  const parts = [];
  if (wantVol && S.linesData) parts.push({ d: S.linesData, onTop: false });
  if (wantSec && S.secLinesData) parts.push({ d: S.secLinesData, onTop: true });   // поверх заливки разреза
  if (!parts.length) { v.setFieldLines(null); return; }
  const top = Math.max((S.result.ranges.B || [0, 1])[1], 1e-12), ramp = S.pal === 'viridis' ? viridis : rainbow;
  v.setFieldLines(parts.map(({ d, onTop }) => {
    const rgb = new Uint8Array(3 * d.values.length);
    for (let i = 0; i < d.values.length; i++) {
      // Объёмные линии — цветом по |B| в шкале поля. Линии на разрезе идут поверх заливки, которая сама
      // показывает |B|: их красим одним светлым цветом, иначе в воздухе линия сливается с заливкой.
      const c = onTop ? LINE_ON_SECTION : ramp(d.values[i] / top);
      rgb[3 * i] = c[0]; rgb[3 * i + 1] = c[1]; rgb[3 * i + 2] = c[2];
    }
    return { points: d.points, offsets: d.offsets, rgb, onTop };
  }));
}
// Стрелки намагничивания (этап 3D-7): до сетки — из предпросмотра (точки внутри тела, `magnet_axis_at`), после
// сетки — из ячеек (ровно то, что уйдёт в решатель). Не запрошенные при последнем предпросмотре — запросить.
function updateArrows() {
  const v = V(), list = [];
  if (S.showArrows && S.model) {
    for (const o of S.model.scene.objects) {
      const i = S.objects.findIndex(x => (x.name || '').trim() === o.name);
      if (i >= 0 && o.arrows && o.arrows.n) list.push({ key: keyOf(i), points: b64(o.arrows.points, Float32Array), dirs: b64(o.arrows.dirs, Float32Array), spacing: o.arrows.spacing });
    }
  } else if (S.showArrows && S.objects.length) {
    if (!S.pvArrowsOn) { schedulePreview(); return; }
    (S.pvArrows || []).forEach((a, i) => {
      if (a && a.n) list.push({ key: keyOf(i), points: b64(a.points, Float32Array), dirs: b64(a.dirs, Float32Array), spacing: a.spacing });
    });
  }
  v.setArrows(list.length ? list : null);
}
// Осевые линии (этап 3D-7): оси координат — на габарит тел; оси тел — отрезки с сервера (после сетки тела те же,
// что при последнем предпросмотре: любая правка тела сбрасывает сетку и заново строит предпросмотр).
function updateAxes() {
  const v = V(), box = S.model ? S.model.scene.bbox : S.pvBBox;
  v.setCoordAxes(S.showCoordAxes && box ? box : null);
  v.setBodyAxes(S.showBodyAxes && S.pvAxes
    ? S.pvAxes.map((a, i) => a && { key: keyOf(i), p0: a[0], p1: a[1] }).filter(Boolean) : null);
}

async function buildModel(keepResult = false) {
  const err = S.objects.length ? checkNames() : 'Добавьте хотя бы одно тело.';
  if (err) { setConv('c', '● ' + err); toast(err, 'warn'); return false; }
  if (!(await ensureStepFiles())) { setConv('c', '● Сервер не получил STEP-файл — подробности в форме тела.'); return false; }
  const body = modelBody(), key = JSON.stringify(body), keep = keepResult ? S.result : null;
  setConv('', '◐ Строю сетку…'); $('st-cells').textContent = '';
  let d;
  try { d = await post('/api/3d/model', body); } catch (e) { setConv('c', '● Сервер недоступен: ' + e); return false; }
  if (d.error) { setConv('c', '● ' + d.error); toast(d.error, 'warn'); return false; }
  S.model = d; S.modelKey = key; S.result = keep; S.values = null; S.secData = null;
  showModel();
  $('st-cells').textContent = 'Сетка ' + d.n_cells + ' эл.';
  setConv('', '● Сетка построена: ' + d.n_cells + ' ячеек' + (d.empty && d.empty.length ? ' · ⚠ без ячеек: ' + d.empty.join(', ') : ''));
  renderMeshInfo(); markDirty();
  return true;
}
function showModel() {
  const objs = [];
  S.cells = {}; S.regionKey = {}; S.regionRGB = { 0: AIR_RGB };
  for (const o of S.model.scene.objects) {
    const i = S.objects.findIndex(x => (x.name || '').trim() === o.name);
    if (i < 0) continue;
    const key = keyOf(i);
    S.cells[key] = b64(o.cells, Uint32Array); S.regionKey[o.id] = key; S.regionRGB[o.id] = hexRgb(matColor(S.objects[i]));
    objs.push({ key, tris: b64(o.tris, Float32Array), color: matColor(S.objects[i]) });
  }
  V().setObjects(objs, { keepView: true });
  afterObjects(); recolor();
  if (S.sec.axis !== 'off') updateSection();
}

// ---------------------------------------------------------------- расчёт
async function run() {
  if (!S.objects.length) { setConv('c', '● Нет тел — добавьте тело кнопками над видом.'); return; }
  if (S.result) { toast('Расчёт готов. Чтобы пересчитать с изменениями, нажмите «Редактировать» на строке проекта.', 'warn'); pulse('proj-edit'); return; }
  if (!S.model || S.modelKey !== JSON.stringify(modelBody())) { if (!(await buildModel())) return; }
  openPrecalc();
}
function openPrecalc() {
  const nameRow = '<div class="pc-row"><span>Название</span><input class="pc-name-inp" id="pc-name" value="' + esc(PROJECT.name || '') + '"></div>';
  let html = pcSec('Проект', nameRow + pcRow('Тип', 'Свободная геометрия 3D'), true);
  html += pcSec('Тела', S.objects.map(o => pcRow(esc(o.name), KIND[o.kind] + ' · ' + esc(matName(o.material)))).join(''));
  html += pcSec('Сетка', pcRow('Размер по умолчанию', S.h + ' мм') + pcRow('Запас воздуха', S.margin + ' ×')
    + S.objects.filter(o => o.mesh_size_mm).map(o => pcRow(esc(o.name), o.mesh_size_mm + ' мм')).join('')
    + pcRow('<b>Всего ячеек</b>', '<b>' + (S.model ? S.model.n_cells : '—') + '</b>'));
  html += pcSec('Параметры расчёта', pcRow('Температура T', S.T + ' °C')
    + pcRow('Граница области', S.bc === 'neumann' ? 'поток не выходит' : 'φ = 0')
    + pcRow('Внешнее поле', hasH0() ? S.H0.join(' / ') + ' кА/м' : 'нет'));
  $('precalc-body').innerHTML = html;
  const nm = $('pc-name');
  if (nm) nm.oninput = () => { PROJECT.name = nm.value.trim(); $('proj-tree-name').textContent = PROJECT.name || '—'; };
  setStage('precalc');
}
async function solve(quiet = false) {
  if (!S.model) { setConv('c', '● Сначала постройте сетку.'); return; }
  const t0 = Date.now(), label = (PROJECT.name || 'Проект') + ' · 3D';
  const snap = quiet ? null : buildCalcBundle(PROJECT.name || 'расчёт', 'done', null);   // снимок на момент запуска, как в 2D
  const body = { model_id: S.model.model_id, T: S.T, bc: S.bc, label };
  if (hasH0()) body.applied_field_kA = S.H0.map(Number);
  try {
    const { job_id } = await post('/api/3d/solve', body);
    pollJobs();
    for (;;) {
      await sleep(500);
      const j = await (await fetch('/api/jobs/' + job_id)).json();
      if (j.status === 'queued') { setConv('', '◷ В очереди…'); continue; }
      if (j.status === 'running') { setConv('', '◐ Расчёт 3D… ' + (((Date.now() - t0) / 1000) | 0) + ' с'); continue; }
      if (j.status !== 'done') { const msg = j.error || 'задача не найдена'; setConv('c', '● ' + msg); toast(msg, 'warn'); return; }
      let via = 'dir';
      S.field3d = await fetchField3d(body.model_id);             // сетка и φ — в файл расчёта (этап 3D-9)
      S.restoreNote = '';
      if (snap) {
        snap.field = j.result; snap.field3d = S.field3d; snap.saved = nowIso();
        via = (await saveCalcToDir(snap).catch(() => ({ via: 'err' }))).via;
        if (PROJECT.id !== snap.id) return;                            // пользователь уже в другом проекте
        PROJECT.status = 'done'; PROJECT.file = calcFilename(snap.name); DIRTY = (via !== 'dir');
      }
      S.result = j.result;
      S.linesData = S.secLinesData = null;                           // поле пересчитано — линии строить заново
      if (quiet && S.field3d) markDirty();                           // пересчёт открытого файла — сохранить с полем
      if (!availQ().includes(S.q)) S.q = 'B';
      await setQuantity(S.q);
      renderResults(); updateResults(); updateLines();
      const ok = j.result.converged;
      // «не сохранено» — заметно, поверх вида (как в 2D), а не мелко в строке состояния
      if (ok && via === 'none') setConv('c', '● Решено, но не сохранено — задайте рабочую папку в ⚙ · итераций ' + j.result.iters);
      else setConv(ok ? 'g' : 'c', (ok ? '● Решено' : '● Не сошлось') + ' · итераций ' + j.result.iters
        + (via === 'none' ? ' — не сохранено: задайте рабочую папку в ⚙' : ''));
      $('st-iters').textContent = 'итераций ' + j.result.iters;
      return;
    }
  } catch (e) { setConv('c', '● Ошибка: ' + e); }
}
async function restoreField() {
  const keep = S.result;
  if (!(await buildModel(true))) return;
  await solve(true);
  if (!S.result) S.result = keep;
}
async function fetchField3d(modelId) {
  try { const d = await post('/api/3d/solution', { model_id: modelId }); return d.error ? null : d; }
  catch (e) { return null; }
}
// Поле из файла расчёта (этап 3D-9): сервер собирает модель из сохранённых сетки и потенциала φ и проверяет φ
// уравнениями текущего кода (Л-80). Не годится или не вышло — остаётся «Пересчитать поле» с объяснением.
async function restoreSaved() {
  const fd = S.field3d, pid = PROJECT.id, key = JSON.stringify(modelBody());
  if (!fd || !S.result) return;
  const moved = () => S.field3d !== fd || PROJECT.id !== pid;          // пользователь уже в другом расчёте
  const fail = note => { if (moved()) return; S.restoring = false; S.restoreNote = note; setConv('c', '● ' + note); renderResults(); };
  S.restoring = true; renderResults();
  clearTimeout(S.pvTimer); preview();                // оси тел — из предпросмотра, параллельно с восстановлением
  if (!(await ensureStepFiles())) { fail('Поле не открыто: сервер не получил STEP-файл — подробности в форме тела.'); return; }
  setConv('', '◐ Открываю поле из файла расчёта…');
  let j;
  try {
    const { job_id } = await post('/api/3d/restore', { model: modelBody(), field: fd, label: (PROJECT.name || 'Проект') + ' · поле из файла' });
    pollJobs();
    do { await sleep(400); j = await (await fetch('/api/jobs/' + job_id)).json(); } while (j.status === 'queued' || j.status === 'running');
  } catch (e) { fail('Поле не открыто: сервер недоступен (' + e + ').'); return; }
  if (moved()) return;
  if (j.status !== 'done') { fail('Поле не открыто: ' + (j.error || 'задача не найдена') + '.'); return; }
  const r = j.result;
  if (!r.ok) {
    fail('Поле в файле не подходит к текущей версии решателя: уравнения с тех пор изменились (невязка при сохранённом потенциале '
      + fmt(r.residual) + ' против ' + fmt(r.stored_residual) + ' при расчёте). Пересчитайте поле, чтобы числа отвечали текущей версии.');
    return;
  }
  let d;
  try { d = await post('/api/3d/scene', { model_id: r.model_id }); } catch (e) { fail('Поле не открыто: сервер недоступен (' + e + ').'); return; }
  if (moved()) return;
  if (d.error) { fail('Поле не открыто: ' + d.error); return; }
  S.model = d; S.modelKey = key; S.values = null; S.secData = null; S.restoring = false; S.restoreNote = '';
  if (S.active) showModel();
  $('st-cells').textContent = 'Сетка ' + d.n_cells + ' эл.';
  if (!availQ().includes(S.q)) S.q = 'B';
  await setQuantity(S.q);
  renderResults(); updateResults(); updateLines();
  setConv('g', '● Поле открыто из файла расчёта — без пересчёта · ' + d.n_cells + ' ячеек');
}
function clearResult() {
  S.result = null; S.values = null; S.linesData = S.secLinesData = null; S.field3d = null;
  updateLines();
  recolor(); renderResults(); updateLegend(); updateResults();
  if (S.secData && S.viewer) S.viewer.setSectionColors(sectionColors());
}

// ---------------------------------------------------------------- поле: цвет и шкала
const availQ = () => QTY.map(q => q[0]).filter(q => !MAGNET_ONLY.has(q) || (S.result && S.result.demag.length));
const qLabel = q => (QTY.find(x => x[0] === q) || [q, q])[1];
async function setQuantity(q) {
  S.q = q;
  if (!S.result || !S.model) { updateLegend(); return; }
  const d = await post('/api/3d/quantity', { model_id: S.model.model_id, quantity: q });
  if (d.error) { toast(d.error, 'warn'); return; }
  S.values = b64(d.values, Float32Array); S.range = [d.min, d.max]; S.unit = d.unit;
  recolor(); updateLegend(); renderResults();
}
function rng() {
  let [a, b] = S.range || [0, 1];
  if (a == null || b == null) { a = 0; b = 1; }
  if (FROM_ZERO.has(S.q)) a = 0;
  if (SIGNED.has(S.q)) { const m = Math.max(Math.abs(a), Math.abs(b), 1e-12); a = -m; b = m; }
  if (!(b > a)) b = a + 1e-12;
  return [a, b];
}
function cmap() {
  const [a, b] = rng();
  if (S.q === 'margin') {                        // за коленом — красный, у колена — жёлтый, запас — зелёный
    const lo = Math.min(a, -1e-9), hi = Math.max(b, 1e-9);
    return v => riskColor(v >= 0 ? 0.5 + 0.5 * Math.min(v / hi, 1) : 0.5 - 0.5 * Math.min(v / lo, 1));
  }
  if (SIGNED.has(S.q)) return v => diverging(v / b);
  const ramp = S.pal === 'viridis' ? viridis : rainbow;
  return v => ramp((v - a) / (b - a));
}
function recolor() {
  const v = S.viewer;
  if (!v || !S.model) return;
  const f = (S.values && $('r3-onsurf').checked) ? cmap() : null;
  for (const key of Object.keys(S.cells)) {
    if (!f) { v.setColors(key, null); continue; }
    const cells = S.cells[key], rgb = new Uint8Array(cells.length * 3);
    let any = false;
    for (let t = 0; t < cells.length; t++) {
      const val = S.values[cells[t]];
      const c = Number.isFinite(val) ? (any = true, f(val)) : GRAY_RGB;
      rgb[3 * t] = c[0]; rgb[3 * t + 1] = c[1]; rgb[3 * t + 2] = c[2];
    }
    v.setColors(key, any ? rgb : null);                // вне магнита у величин магнита — цвет материала
  }
  if (S.secData) v.setSectionColors(sectionColors());
}
function sectionColors() {
  const { cells, regions } = S.secData, n = cells.length, rgb = new Uint8Array(n * 3);
  const f = S.values ? cmap() : null;
  for (let t = 0; t < n; t++) {
    const val = S.values ? S.values[cells[t]] : NaN;
    const c = (f && Number.isFinite(val)) ? f(val) : (S.regionRGB[regions[t]] || AIR_RGB);
    rgb[3 * t] = c[0]; rgb[3 * t + 1] = c[1]; rgb[3 * t + 2] = c[2];
  }
  return rgb;
}
function updateLegend() {
  const cb = $('cbar');
  if (!S.active || !S.values) { cb.classList.add('hide'); return; }
  let [a, b] = rng();
  const risk = S.q === 'margin';
  if (risk) { a = Math.min(a, 0); b = Math.max(b, 0); }
  // Градиент шкалы — той же функцией, что красит тела: шкала совпадает с картинкой при любом отображении.
  $('cscale').style.background = 'linear-gradient(90deg,' + stops(cmap(), a, b) + ')';
  $('cmin').textContent = fmt(a) + (risk ? (a < 0 ? ' (за коленом)' : ' (колено)') : '');
  $('cmax').textContent = fmt(b) + (risk ? ' (запас)' : '');
  $('cunit').textContent = S.unit || '';
  cb.classList.remove('hide');
}

// ---------------------------------------------------------------- разрез
function secBounds() {
  const bb = S.model ? S.model.scene.bbox : S.pvBBox, k = 'xyz'.indexOf(S.sec.axis);
  if (!bb || k < 0) return null;
  const lo = bb[0][k], hi = bb[1][k], pad = 0.15 * ((hi - lo) || 10);
  return [lo - pad, hi + pad];
}
function updateSecRange(center) {
  const b = secBounds();
  if (!b) return;
  S.sec.lo = b[0]; S.sec.hi = b[1];
  if (center) S.sec.pos = +((b[0] + b[1]) / 2).toFixed(2);
  $('sec3d-pos').value = Math.round(1000 * (S.sec.pos - b[0]) / (b[1] - b[0]));
  $('sec3d-val').value = fmt(S.sec.pos, 2);
}
function applyClip() { if (S.viewer) S.viewer.setClip(S.sec.axis === 'off' ? null : S.sec.axis, S.sec.pos, S.sec.flip); }
function scheduleSection() { clearTimeout(S.secTimer); S.secTimer = setTimeout(updateSection, 120); }
async function updateSection() {
  applyClip();
  const v = V();
  if (S.sec.axis === 'off' || !S.model) { S.secData = null; v.setSection(null); updateLines(); return; }
  const seq = ++S.secSeq, n = VEC[S.sec.axis];
  const body = { model_id: S.model.model_id, point_mm: n.map(c => c * S.sec.pos), normal: n };
  if (!$('sec3d-air').checked) body.objects = S.model.scene.objects.map(o => o.name);
  let d;
  try { d = await post('/api/3d/section', body); } catch (e) { return; }
  if (seq !== S.secSeq) return;
  if (d.error || !d.n) { S.secData = null; v.setSection(null); return; }
  S.secData = { cells: b64(d.cells, Uint32Array), regions: b64(d.regions, Uint32Array) };
  v.setSection(b64(d.tris, Float32Array), sectionColors());
  updateLines();                                            // линии на разрезе — под новую плоскость
}

// ---------------------------------------------------------------- дерево тел и формы
function renderList() {
  if (!S.active) return;
  $('objlist').innerHTML = S.objects.map((o, i) => '<div class="node i1' + (i === S.sel ? ' sel' : '') + '" data-o3="' + i + '">'
    + '<span class="sw3" style="background:' + matColor(o) + '"></span>'
    + '<span class="o3name"' + ((o.name || '').trim() ? '' : ' style="color:var(--crit)"') + '>' + esc((o.name || '').trim() || '(без имени)') + '</span>'
    + (o.kind === 'step' ? '<span class="tag">STEP</span>' : '') + '<span class="tag mat">' + esc(matTag(o.material)) + '</span>'
    + '<button class="o3btn' + (o.hidden ? ' off' : '') + '" data-eye3="' + i + '" title="Показать или скрыть в виде">◉</button>'
    + '<button class="o3btn" data-edit3="' + i + '" title="Изменить">✎</button>'
    + '<button class="o3btn" data-del3="' + i + '" title="Удалить">✕</button></div>').join('')
    || '<p class="hint" style="padding:6px 10px">Пусто — добавьте тело кнопками над видом.</p>';
  updateSteps();                                            // строка шагов в шапке: тела есть / нет
}
function select(i) {
  S.sel = (i >= 0 && i < S.objects.length) ? i : -1;
  renderList();
  if (S.viewer) S.viewer.setSelected(S.sel >= 0 ? keyOf(S.sel) : null);
  if (STAGE === 'geom') renderForm();
}
function renderForm() {
  const o = S.objects[S.sel], box = $('g3-form');
  $('g3-empty').style.display = o ? 'none' : '';
  if (!o) { box.innerHTML = ''; return; }
  const row = (label, f, v, u, ph = '') => '<div class="mrow"><label>' + label + '</label><input class="num g3" data-f="' + f + '" value="' + esc(v) + '"'
    + (ph ? ' placeholder="' + ph + '"' : '') + ' style="width:74px"><span class="u">' + u + '</span></div>';
  const vec = (label, f, a) => '<div class="field"><label>' + label + '</label><div class="vec3">'
    + a.map((x, k) => '<input class="num g3" data-f="' + f + ':' + k + '" value="' + esc(x) + '" title="' + 'XYZ'[k] + '">').join('') + '</div></div>';
  let h = '<div class="field"><label>Имя тела <span style="color:var(--crit)">*</span></label><input class="g3 g3text" data-f="name" value="' + esc(o.name) + '"></div>';
  h += '<div class="field"><label>Тип</label><div class="g3kind">' + KIND[o.kind] + '</div></div>';
  h += '<div class="field"><label>Материал</label><button class="matbtn" id="g3-mat"><span>' + esc(matName(o.material)) + '</span><span class="sub">'
    + esc(matFamOf(o.material)) + ' ▾</span></button></div>';
  const cad = o.kind === 'step';
  if (cad) {
    const f = S.stepFiles[o.params.file_id] || {}, b = (f.bodies || [])[o.params.body];
    const info = (label, text) => '<div class="field"><label>' + label + '</label><div class="g3kind">' + text + '</div></div>';
    h += '<div class="reshead">Тело из CAD</div>' + info('Файл', esc(f.name || '— нет в расчёте')) + info('Номер тела в файле', String(o.params.body + 1))
      + info('Объём', fmt(o.params.volume_mm3, 2) + ' мм³')
      + (b ? info('Габарит X × Y × Z', b.bbox_mm[1].map((v, k) => fmt(v - b.bbox_mm[0][k], 2)).join(' × ') + ' мм') : '');
    h += '<div class="reshead">Ось двигателя</div>' + vec('Точка на оси (X, Y, Z), мм', 'ax', o.params.axis_origin_mm)
      + vec('Направление оси (X, Y, Z)', 'ad', o.params.axis_dir);
    h += '<p class="hint" style="margin:-4px 0 10px">Относительно этой оси считаются «осевое» и «радиальное» намагничивание. Ось задана в координатах CAD и сдвигается вместе с телом.</p>';
  } else {
    h += '<div class="reshead">Размеры</div>';
    if (o.kind === 'prism') {
      h += '<div class="field"><label>Многоугольник в плоскости XY тела: «x y» в мм, строка на вершину</label><textarea class="g3 g3area" data-f="points" rows="5" spellcheck="false">'
        + esc(o.params.points.map(p => p[0] + ' ' + p[1]).join('\n')) + '</textarea></div>' + row('Высота вдоль оси Z тела', 'p:h', o.params.h, 'мм');
    } else for (const k of Object.keys(DEF[o.kind])) h += row(PAR[k] || k, 'p:' + k, o.params[k], (k === 'a1' || k === 'a2') ? '°' : 'мм');
  }
  h += '<div class="reshead">Положение</div>' + vec(cad ? 'Сдвиг (X, Y, Z), мм' : 'Центр (X, Y, Z), мм', 'c', o.center) + vec('Поворот вокруг X, Y, Z, °', 'rot', o.rotation);
  h += '<p class="hint" style="margin:-4px 0 10px">' + (cad
    ? 'Сдвиг и поворот — дополнительно к положению в CAD; поворот вокруг начала координат CAD по порядку: X, затем Y, затем Z.'
    : 'Поворот — вокруг глобальных осей по порядку: X, затем Y, затем Z. Ось цилиндра, трубы и призмы — локальная ось Z тела.') + '</p>';
  if (kindOf(o.material) === 'magnet') {
    h += '<div class="field"><label>Намагничивание</label><select class="select g3" data-f="magnet_dir">'
      + (cad ? DIRS_STEP : DIRS).map(d => '<option value="' + d[0] + '"' + (d[0] === o.magnet_dir ? ' selected' : '') + '>' + d[1] + '</option>').join('') + '</select></div>';
    h += vec('Поворот намагниченности вокруг X, Y, Z, °', 'mrot', o.magnet_rotation || [0, 0, 0]);
    h += '<p class="hint" id="g3-mdir" style="margin:-4px 0 10px">' + magnetDirText(o) + '</p>';
  }
  h += '<div class="reshead">Сетка и наложение</div>' + row('Размер сетки', 'mesh_size_mm', o.mesh_size_mm, 'мм', 'авто')
    + row('Приоритет слоя', 'priority', o.priority, '1–50');
  box.innerHTML = h;
  box.querySelectorAll('.g3').forEach(el => {
    const live = el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';
    el.addEventListener(live ? 'input' : 'change', () => onField(o, el));
  });
  $('g3-mat').onclick = () => openMatPicker(o.material, 'Материал тела «' + ((o.name || '').trim() || 'без имени') + '»',
    id => { o.material = id; changed(); renderForm(); });
}
function onField(o, el) {
  const f = el.dataset.f, v = el.value;
  if (f === 'name') { o.name = v; changed(); return; }                 // области сетки названы по именам
  if (f === 'magnet_dir') { o.magnet_dir = v; const md = $('g3-mdir'); if (md) md.textContent = magnetDirText(o); changed(); return; }
  if (f === 'points') {
    const pts = v.trim().split(/\n+/).map(l => l.trim().split(/[\s;]+/).map(Number))
      .filter(p => p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])).map(p => [p[0], p[1]]);
    if (pts.length < 3) { showErr('Многоугольник — не меньше трёх вершин.'); return; }
    o.params.points = pts; changed(); return;
  }
  if (f === 'mesh_size_mm') { const t = v.trim(); if (t !== '' && !(+t > 0)) return; o.mesh_size_mm = t === '' ? '' : +t; changed(); return; }
  const x = +v;
  if (v.trim() === '' || !Number.isFinite(x)) return;
  if (f === 'priority') { o.priority = Math.max(1, Math.min(50, Math.round(x))); changed(); return; }
  const [grp, k] = f.split(':');
  if (grp === 'ad') {
    const dir = o.params.axis_dir.slice(); dir[+k] = x;
    if (dir.every(v => v === 0)) { showErr('Направление оси не может быть нулевым.'); return; }
    o.params.axis_dir = dir;
  } else if (grp === 'ax') o.params.axis_origin_mm[+k] = x;
  else if (grp === 'p') o.params[k] = x; else if (grp === 'c') o.center[+k] = x; else if (grp === 'rot') o.rotation[+k] = x;
  else if (grp === 'mrot') { o.magnet_rotation = (o.magnet_rotation || [0, 0, 0]).slice(); o.magnet_rotation[+k] = x; }
  const md = $('g3-mdir'); if (md) md.textContent = magnetDirText(o);     // направление — сразу под полями
  changed();
}
// Направление намагниченности (этап 3D-10): направление из списка, затем поворот вокруг ГЛОБАЛЬНЫХ осей X → Y → Z —
// то же правило, что поворот тела (`objects.rotation_matrix`: R = Rz·Ry·Rx) и что `magnet_axis_at` на сервере.
function rotMat(deg) {
  const [x, y, z] = deg.map(a => (+a || 0) * Math.PI / 180);
  const cx = Math.cos(x), sx = Math.sin(x), cy = Math.cos(y), sy = Math.sin(y), cz = Math.cos(z), sz = Math.sin(z);
  return [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx]];
}
const mulv = (R, v) => R.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const unit3 = v => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return v.map(x => x / n); };
function magnetDirection(o) {                       // единичный вектор M; null — у радиального (в каждой точке свой)
  const d = o.magnet_dir;
  let v;
  if (VEC[d]) v = VEC[d];
  else if (!d || d === 'axial' || d === 'axial-in') {
    v = mulv(rotMat(o.rotation), o.kind === 'step' ? unit3(o.params.axis_dir) : [0, 0, 1]);   // ось тела
    if (d === 'axial-in') v = v.map(x => -x);
  } else return null;
  return mulv(rotMat(o.magnet_rotation || [0, 0, 0]), v);
}
function magnetDirText(o) {
  const v = magnetDirection(o), turned = (o.magnet_rotation || []).some(a => +a);
  if (!v) return turned ? 'Радиально, затем поворот вокруг глобальных осей X → Y → Z: у каждой точки своё направление — смотрите стрелки «намагничивание» над видом.'
    : 'Радиально: у каждой точки своё направление — смотрите стрелки «намагничивание» над видом.';
  const c = v.map(x => (Math.abs(x) < 5e-5 ? 0 : x).toFixed(2).replace('.', ','));
  return 'Направление M: (' + c.join('; ') + '). Поворот — вокруг глобальных осей по порядку: X, затем Y, затем Z, как у тела.';
}
function renderMesh() {
  $('m3-h').value = S.h; $('m3-margin').value = S.margin; $('m3-grading').value = S.grading;
  $('m3-objs').innerHTML = S.objects.map((o, i) => '<div class="mrow"><label>' + esc((o.name || '').trim() || '(без имени)')
    + '</label><input class="num m3o" data-i="' + i + '" value="' + esc(o.mesh_size_mm || '') + '" placeholder="авто" style="width:70px"><span class="u">мм</span></div>').join('')
    || '<p class="hint" style="padding:2px 0">Нет тел — добавьте на шаге «Геометрия».</p>';
  $('m3-objs').querySelectorAll('.m3o').forEach(inp => inp.onchange = () => {
    const t = inp.value.trim(); if (t !== '' && !(+t > 0)) return;
    S.objects[+inp.dataset.i].mesh_size_mm = t === '' ? '' : +t; changed();
  });
  renderMeshInfo();
}
function renderMeshInfo() {
  const el = $('m3-info'); if (!el) return;
  el.innerHTML = S.model ? '<span class="tot">' + S.model.n_cells + '</span> ячеек · ' + S.model.n_vertices + ' узлов'
    + (S.model.empty && S.model.empty.length ? '<br><span style="color:var(--warn)">⚠ без ячеек: ' + esc(S.model.empty.join(', ')) + '</span>' : '')
    : 'Сетка ещё не построена — «Построить сетку» ниже или «Рассчитать» в шапке.';
}
function renderCalc() {
  $('c3-T').value = S.T; $('c3-bc').value = S.bc;
  ['x', 'y', 'z'].forEach((a, k) => { $('c3-h' + a).value = S.H0[k]; });
}

// ---------------------------------------------------------------- правая панель результатов
function renderResults() {
  if (!S.active) return;
  const r = S.result;
  $('r3-empty').style.display = r ? 'none' : '';
  $('r3-body').style.display = r ? '' : 'none';
  $('vplbl').innerHTML = (r && S.values) ? '<b>' + esc(qLabel(S.q)) + '</b>' : '<b>Геометрия модели · 3D</b>';
  if (!r) return;
  const qs = availQ();
  $('r3-q').innerHTML = QTY.filter(q => qs.includes(q[0])).map(q => '<option value="' + q[0] + '"' + (q[0] === S.q ? ' selected' : '') + '>' + q[1] + '</option>').join('');
  $('r3-pal').value = S.pal;
  $('r3-restore').style.display = ((S.model && S.values) || S.restoring) ? 'none' : '';
  $('r3-restore-note').textContent = S.restoreNote || (S.field3d
    ? 'Поле этого расчёта не открыто — его можно пересчитать с теми же данными.'
    : 'В файле нет сохранённого поля (до 21.09.2026 файл хранил только сводку) — его можно пересчитать с теми же данными.');
  const rows = [['Сходимость', r.converged ? 'да' : 'нет', r.converged ? 'ok' : 'crit'], ['Итераций', r.iters],
    ['Невязка', fmt(r.residual)], ['Температура', r.T + ' °C'], ['Ячеек сетки', S.model ? S.model.n_cells : '—'],
    ['Коэнергия поля', fmt(r.coenergy_J * 1e3) + ' мДж']];
  $('r3-sum').innerHTML = rows.map(x => '<li class="' + (x[2] || '') + '"><span>' + x[0] + '</span><span class="v">' + x[1] + '</span></li>').join('');
  $('r3-objs').innerHTML = '<tr><th>Тело</th><th>V, см³</th><th>B ср, Тл</th><th>B макс</th></tr>'
    + r.objects.map(o => '<tr><td>' + esc(o.name) + '</td><td>' + fmt(o.volume_cm3) + '</td><td>' + fmt(o.B_mean) + '</td><td>' + fmt(o.B_max) + '</td></tr>').join('');
  $('r3-demag-wrap').style.display = r.demag.length ? '' : 'none';
  // Вердикт — потеря потока после расчёта (Л-104): замер при 20 °C без поля, новый магнит против этого.
  const lossRow = (label, v) => '<li class="' + (v == null ? 'crit' : (v > LOSS_NOISE ? 'warn' : 'ok')) + '"><span>'
    + label + '</span><span class="v">' + (v == null ? '—' : (Math.abs(v) < LOSS_NOISE ? '0 %' : pct(v))) + '</span></li>';
  const lossLabel = 'Потеря потока (замер при ' + r.flux_measure_T + ' °C)';
  let head = r.flux_loss_error ? '<p class="hint" style="color:var(--crit)">⚠ Потеря потока не посчитана: '
    + esc(r.flux_loss_error) + '</p>' : '';
  if (r.demag.length > 1) head += '<ul class="res">' + lossRow('Потеря потока — все магниты вместе', r.flux_loss_total) + '</ul>';
  $('r3-demag').innerHTML = head + r.demag.map(d => '<div class="reshead" style="color:var(--text)">' + esc(d.name) + '</div><ul class="res">'
    + lossRow(lossLabel, d.flux_loss)
    + '<li><span>Повреждено (доля объёма)</span><span class="v">' + pct(d.damaged) + '</span></li>'
    + '<li><span>За коленом сейчас (доля объёма)</span><span class="v">' + pct(d.past_knee) + '</span></li>'
    + (d.beyond_hcj > 0 ? '<li class="crit"><span>За −H_cJ (модель не определена)</span><span class="v">' + pct(d.beyond_hcj) + '</span></li>' : '')
    + '<li><span>Сохранено B_r (по объёму)</span><span class="v">' + pct(d.retained) + '</span></li></ul>').join('');
  const names = r.objects.map(o => o.name);
  for (const b of [...S.bodies]) if (!names.includes(b)) S.bodies.delete(b);
  $('r3-bodies').innerHTML = names.map(n => '<label class="chk"><input type="checkbox" data-body="' + esc(n) + '"' + (S.bodies.has(n) ? ' checked' : '') + '>' + esc(n) + '</label>').join('');
}
async function force() {
  const out = $('r3-force-out');
  if (!S.model || !S.values) { out.textContent = 'Сначала пересчитайте поле (кнопка выше).'; return; }
  if (!S.bodies.size) { out.textContent = 'Отметьте хотя бы одно тело.'; return; }
  const body = { model_id: S.model.model_id, bodies: [...S.bodies] };
  if ($('r3-tpoint').value === 'o') body.point_mm = [0, 0, 0];
  out.textContent = '…';
  const d = await post('/api/3d/force', body);
  if (d.error) { out.textContent = '⚠ ' + d.error; return; }
  const F = d.force_N, T = d.torque_Nm, P = d.point_mm;
  out.textContent = 'F = (' + F.map(x => fmt(x, 4)).join('; ') + ') Н\n|F| = ' + fmt(Math.hypot(...F), 4) + ' Н\n'
    + 'τ = (' + T.map(x => fmt(x, 5)).join('; ') + ') Н·м\nотносительно (' + P.map(x => fmt(x, 2)).join('; ') + ') мм';
}
async function flux() {
  const out = $('r3-flux-out');
  if (!S.model || !S.values) { out.textContent = 'Сначала пересчитайте поле (кнопка выше).'; return; }
  if (S.sec.axis === 'off') { out.textContent = 'Включите разрез (X, Y или Z) над видом.'; return; }
  const n = VEC[S.sec.axis], body = { model_id: S.model.model_id, point_mm: n.map(c => c * S.sec.pos), normal: n };
  if (S.bodies.size) body.objects = [...S.bodies];
  const d = await post('/api/3d/flux', body);
  if (d.error) { out.textContent = '⚠ ' + d.error; return; }
  const phi = d.flux_Wb, ax = S.sec.axis.toUpperCase();
  out.textContent = 'Φ = ' + (Math.abs(phi) < 1e-3 ? fmt(phi * 1e6, 4) + ' мкВб' : fmt(phi * 1e3, 4) + ' мВб')
    + '\nплоскость ' + ax + ' = ' + fmt(S.sec.pos, 2) + ' мм, нормаль +' + ax
    + (S.bodies.size ? '\nтела: ' + [...S.bodies].join(', ') : '\nвся область');
}

// ---------------------------------------------------------------- связь со страницей
function activate() {
  S.active = true;
  V();
  $('st-scn').textContent = 'Магнитостатика 3D'; $('st-mode').textContent = 'Вид: 3D';
  renderList(); renderResults(); updateLegend();
  if (S.model) showModel(); else schedulePreview();
  requestAnimationFrame(() => { if (S.viewer) S.viewer.resize(); });
}
function deactivate() { S.active = false; $('cbar').classList.add('hide'); }
function reset() {
  const v = S.viewer, act = S.active;
  Object.assign(S, fresh(), { viewer: v, active: act, fitCount: -1 });
  if (v) {
    v.setObjects([]); v.setSection(null); v.setClip(null);
    v.setCoordAxes(null); v.setBodyAxes(null); v.setArrows(null); v.setFieldLines(null);
  }
  document.querySelectorAll('#sec3d-axis button').forEach(b => b.classList.toggle('on', b.dataset.ax === 'off'));
  $('vt3-sec').hidden = true;
  $('op3d').value = 100; $('sec3d-flip').checked = false;
  if (act) { renderList(); renderResults(); updateLegend(); }
}
function geomDef() {
  pruneStepFiles();                                   // в файл расчёта — только файлы, из которых есть тела
  return { objects: S.objects.map(o => ({ ...o, params: JSON.parse(JSON.stringify(o.params)), center: [...o.center], rotation: [...o.rotation],
      ...(o.magnet_rotation ? { magnet_rotation: [...o.magnet_rotation] } : {}) })),
    stepFiles: Object.fromEntries(Object.entries(S.stepFiles).map(([id, f]) => [id, { ...f }])),
    h: S.h, margin: S.margin, grading: S.grading, T: S.T, bc: S.bc, H0: S.H0.slice() };
}
function applyBundle(m) {
  reset();
  const g = m.geom || {};
  S.objects = (g.objects || []).map(o => ({ ...o, params: JSON.parse(JSON.stringify(o.params)),
    center: [...(o.center || [0, 0, 0])], rotation: [...(o.rotation || [0, 0, 0])],
    ...(o.magnet_rotation ? { magnet_rotation: [...o.magnet_rotation] } : {}) }));
  S.stepFiles = Object.fromEntries(Object.entries(g.stepFiles || {}).map(([id, f]) => [id, { ...f }]));
  for (const k of ['h', 'margin', 'grading', 'T', 'bc']) if (g[k] !== undefined) S[k] = g[k];
  if (Array.isArray(g.H0)) S.H0 = g.H0.slice();
  S.result = m.field || null;                     // сводка
  S.field3d = (m.field && m.field3d) || null;     // сетка и φ решения (этап 3D-9); в старых файлах нет — «Пересчитать поле»
  if (S.active) { renderList(); renderResults(); updateLegend(); schedulePreview(); }
  if (S.field3d) restoreSaved();                  // поле — сразу, без пересчёта
}
function onStage(st) { if (st === 'geom') renderForm(); else if (st === 'mesh') renderMesh(); else if (st === 'calc') renderCalc(); }
async function onBuild(stage) {
  if (stage === 'mesh') { if (await buildModel()) setStage('result'); return; }
  setStage('result');                              // правки тела применяются сразу — окно просто закрыть
}
function nextView() { if (S.viewer) $('st-mode').textContent = 'Вид: 3D · ' + S.viewer.nextView(); }

function bindUI() {
  $('palette3d').addEventListener('click', e => {
    const b = e.target.closest('[data-add3]'); if (!b || editGuard()) return;
    S.objects.push(newObject(b.dataset.add3)); S.sel = S.objects.length - 1;
    changed(); setStage('geom');
  });
  $('step3d-btn').onclick = () => { if (editGuard()) return; const inp = $('step3d-file'); inp.value = ''; inp.click(); };
  $('step3d-file').onchange = e => stepFileChosen(e.target.files && e.target.files[0]);
  $('objlist').addEventListener('click', async e => {
    if (MODE !== 'objects3d') return;
    const t = e.target, ds = t.dataset;
    if (ds.eye3 !== undefined) { const o = S.objects[+ds.eye3]; o.hidden = !o.hidden; if (S.viewer) S.viewer.setVisible(keyOf(+ds.eye3), !o.hidden); renderList(); return; }
    if (ds.del3 !== undefined) {
      if (editGuard()) return;
      const i = +ds.del3;
      if (!(await uiConfirm('Удалить «' + S.objects[i].name + '»?'))) return;
      S.objects.splice(i, 1); if (S.sel >= S.objects.length) S.sel = S.objects.length - 1; pruneStepFiles();
      changed(); if (STAGE === 'geom') renderForm(); return;
    }
    if (ds.edit3 !== undefined) { if (editGuard()) return; select(+ds.edit3); setStage('geom'); return; }
    const row = t.closest('[data-o3]'); if (row) select(+row.dataset.o3);
  });
  document.querySelectorAll('#sec3d-axis button').forEach(b => b.onclick = () => {
    S.sec.axis = b.dataset.ax;
    document.querySelectorAll('#sec3d-axis button').forEach(x => x.classList.toggle('on', x === b));
    $('vt3-sec').hidden = S.sec.axis === 'off';               // положение разреза — только при включённом разрезе
    updateSecRange(true); updateSection();
  });
  $('sec3d-pos').oninput = () => {
    if (S.sec.axis === 'off') return;
    S.sec.pos = +(S.sec.lo + (S.sec.hi - S.sec.lo) * (+$('sec3d-pos').value) / 1000).toFixed(2);
    $('sec3d-val').value = fmt(S.sec.pos, 2); applyClip(); scheduleSection();
  };
  $('sec3d-val').onchange = () => { const x = +$('sec3d-val').value; if (!Number.isFinite(x)) return; S.sec.pos = x; updateSecRange(false); updateSection(); };
  $('sec3d-flip').onchange = e => { S.sec.flip = e.target.checked; applyClip(); };
  $('sec3d-air').onchange = () => updateSection();
  $('op3d').oninput = e => { S.opacity = +e.target.value / 100; if (S.viewer) S.viewer.setOpacity(S.opacity); };
  $('ax3d-coord').onchange = e => { S.showCoordAxes = e.target.checked; if (S.active) updateAxes(); };
  $('ax3d-body').onchange = e => { S.showBodyAxes = e.target.checked; if (S.active) updateAxes(); };
  $('mag3d-arrows').onchange = e => { S.showArrows = e.target.checked; if (S.active) updateArrows(); };
  $('ln3d-show').onchange = e => { S.showLines = e.target.checked; if (S.active) updateLines(); };
  $('ln3d-sec').onchange = e => { S.showSecLines = e.target.checked; if (S.active) updateLines(); };
  $('ln3d-n').onchange = () => {
    const x = Math.round(+$('ln3d-n').value);
    if (!(x >= 1 && x <= 2000)) { $('ln3d-n').value = S.linesN; return; }
    S.linesN = x; S.linesData = S.secLinesData = null; if (S.active) updateLines();
  };
  const num = (id, key, ok) => { $(id).onchange = () => { const x = +$(id).value; if (!ok(x)) { $(id).value = S[key]; return; } S[key] = x; changed(); renderMeshInfo(); }; };
  num('m3-h', 'h', x => x > 0); num('m3-margin', 'margin', x => x > 0); num('m3-grading', 'grading', x => x >= 0);
  $('c3-T').onchange = () => { const x = +$('c3-T').value; if (Number.isFinite(x)) { S.T = x; markDirty(); } else $('c3-T').value = S.T; };
  $('c3-bc').onchange = () => { S.bc = $('c3-bc').value; markDirty(); };
  ['x', 'y', 'z'].forEach((a, k) => { $('c3-h' + a).onchange = () => { const x = +$('c3-h' + a).value; if (Number.isFinite(x)) { S.H0[k] = x; markDirty(); } }; });
  $('r3-q').onchange = e => setQuantity(e.target.value);
  $('r3-pal').onchange = e => { S.pal = e.target.value; recolor(); updateLegend(); updateLines(); };
  $('r3-onsurf').onchange = () => recolor();
  $('r3-bodies').addEventListener('change', e => { const n = e.target.dataset.body; if (n === undefined) return; if (e.target.checked) S.bodies.add(n); else S.bodies.delete(n); });
  $('r3-force').onclick = force;
  $('r3-flux').onclick = flux;
  $('r3-restore-btn').onclick = restoreField;
  $('r3-vtu').onclick = () => {
    if (!S.model || !S.values) { toast('Сначала пересчитайте поле.', 'warn'); return; }
    const nm = (PROJECT.name || 'model3d').trim() || 'model3d';
    download('/api/3d/export_vtu?model_id=' + encodeURIComponent(S.model.model_id) + '&name=' + encodeURIComponent(nm), nm + '.vtu');
  };
  $('r3-png').onclick = () => { if (S.viewer) download(S.viewer.screenshot(), ((PROJECT.name || 'model3d').trim() || 'model3d') + '.png'); };
}

window.WS3D = { activate, deactivate, reset, applyBundle, geomDef, onStage, onBuild, run, clearResult, nextView, importStepBytes,
  solve: () => solve(false), materialIds: () => S.objects.map(o => o.material), hasObjects: () => S.objects.length > 0,
  hasModel: () => !!S.model, hasResult: () => !!S.result, result: () => S.result, field3d: () => S.field3d,
  temperature: () => S.T, cells: () => (S.model ? S.model.n_cells : null), screenshot: () => (S.viewer ? S.viewer.screenshot() : null),
  // смена темы оформления: вид перекрашивается, оси с подписями и стрелки намагничивания строятся заново
  applyTheme: () => { if (!S.viewer) return; S.viewer.applyTheme(); if (S.active) { updateAxes(); updateArrows(); } },
  zoom: f => { if (S.viewer) S.viewer.zoom(f); }, fit: () => { if (S.viewer) S.viewer.fit(); } };
bindUI();
if (MODE === 'objects3d') activate();
