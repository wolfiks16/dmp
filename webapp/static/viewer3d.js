// ОБЪЁМНЫЙ ВИД (этап 3D-5, план — docs/plan_3d_2026-09-11.md): обёртка над three.js. Рисует тела
// из треугольников (координаты в мм, по 9 чисел на треугольник), цвет — на треугольник, выбор мышью,
// секущую плоскость и заливку разреза, оси в углу, снимок. Геометрию и величины готовит сервер
// (magcore.fem3d.scene) — здесь только рисование. Цвета приходят в sRGB (0…255), как у шкал 2D.
import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { LineMaterial } from './vendor/three/lines/LineMaterial.js';
import { LineSegments2 } from './vendor/three/lines/LineSegments2.js';
import { LineSegmentsGeometry } from './vendor/three/lines/LineSegmentsGeometry.js';

// Нейтральные цвета вида — из переменных стиля страницы (тема оформления), без них — прежние тёмные.
// Перечитываются при смене темы (Viewer3D.applyTheme).
const CSSV = (k, d) => getComputedStyle(document.documentElement).getPropertyValue(k).trim() || d;
const hex = s => new THREE.Color(s).getHex();
const AXIS_COL = ['#e5544e', '#42c25a', '#4d8bff'];                // X, Y, Z — как у тройки осей в углу
const GHOST = 0.28;                          // яркость участков осей, скрытых телами (как невидимые линии в CAD)
let BG, EDGE, EDGE_SEL, BODY_AXIS, BODY_AXIS_SEL, ARROW, LABEL, LABEL_HALO;
function readTheme() {
  BG = hex(CSSV('--cv-bg', '#0a0f18'));
  EDGE = hex(CSSV('--v3-edge', '#0b111b')); EDGE_SEL = hex(CSSV('--cv-sel', '#2fd39c'));
  BODY_AXIS = hex(CSSV('--v3-body-axis', '#8a9cc0')); BODY_AXIS_SEL = EDGE_SEL;
  ARROW = hex(CSSV('--cv-arrow', '#f2f5fa'));   // стрелки намагничивания — один цвет: полюса цветом не подсвечиваются
  LABEL = CSSV('--cv-label', '#c8d4ea'); LABEL_HALO = CSSV('--v3-halo', 'rgba(10,15,24,0.92)');
}
readTheme();
const ARROW_BODY_ALPHA = 0.35;               // непрозрачность магнита, пока показаны его стрелки (иначе их не видно)
// Шаг делений: 1, 2 или 5 × 10ⁿ мм, не мельче raw.
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))); return [1, 2, 5, 10].map(m => m * p).find(s => s >= raw * (1 - 1e-9)); }
// sRGB 0…255 → линейная яркость (рабочее пространство three.js; при выводе вернётся в sRGB).
const LIN = new Float32Array(256).map((_, i) => { const v = i / 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
const VIEWS = [['изометрия', [0.55, -0.75, 0.45]], ['сверху (+Z)', [0.0001, -0.0001, 1]], ['спереди (−Y)', [0, -1, 0.0001]], ['справа (+X)', [1, 0, 0.0001]]];

export class Viewer3D {
  constructor(el, { onPick = null } = {}) {
    this.el = el; this.onPick = onPick;
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.setClearColor(BG, 1); r.autoClear = false; r.localClippingEnabled = true;
    Object.assign(r.domElement.style, { width: '100%', height: '100%', display: 'block' });
    el.appendChild(r.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1e6);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(55, -75, 45);
    this.scene.add(this.camera);
    this.scene.add(new THREE.HemisphereLight(0xe6eeff, 0x283040, 1.2));
    const head = new THREE.DirectionalLight(0xffffff, 1.5);          // «фонарь» у камеры
    head.position.set(0.4, 0.6, 1); this.camera.add(head);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.addEventListener('change', () => this.render());
    this.group = new THREE.Group(); this.scene.add(this.group);
    this.items = new Map(); this.section = null; this.selected = null; this.opacity = 1; this.viewIdx = 0;
    this.clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0); this.clipOn = false;
    this.axes = new THREE.Group(); this.bodyAxes = new THREE.Group(); this.arrows = new THREE.Group();
    this.lines = new THREE.Group();
    this.scene.add(this.axes, this.bodyAxes, this.arrows, this.lines);
    this.bodyAxisList = []; this.arrowList = []; this.lineMats = [];
    this._gizmo();
    this.ray = new THREE.Raycaster(); this._down = null;
    r.domElement.addEventListener('pointerdown', e => { this._down = [e.clientX, e.clientY]; });
    r.domElement.addEventListener('pointerup', e => {
      const d = this._down; this._down = null;
      if (d && Math.hypot(e.clientX - d[0], e.clientY - d[1]) < 5) this._pick(e);        // клик, не поворот
    });
    this._ro = new ResizeObserver(() => this.resize()); this._ro.observe(el);
    this.resize();
  }

  // ---------------------------------------------------------------- тела
  setObjects(list, { keepView = false } = {}) {
    for (const it of this.items.values()) this._drop(it);
    this.items.clear();
    for (const o of list) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(o.tris, 3));
      g.computeVertexNormals();                                        // без общих вершин — плоские грани
      const base = new THREE.Color(o.color || '#8391a6');
      const col = new Float32Array(o.tris.length);
      for (let i = 0; i < col.length; i += 3) { col[i] = base.r; col[i + 1] = base.g; col[i + 2] = base.b; }
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
      const mesh = new THREE.Mesh(g, mat); mesh.userData.key = o.key;
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g, 28), new THREE.LineBasicMaterial({ color: EDGE }));
      this.group.add(mesh, edges);
      this.items.set(o.key, { mesh, edges, base });
    }
    this._state();
    if (keepView && this._fitted) this.render(); else this.fit();
  }

  _drop(it) {
    this.group.remove(it.mesh, it.edges);
    it.mesh.geometry.dispose(); it.mesh.material.dispose(); it.edges.geometry.dispose(); it.edges.material.dispose();
  }

  // Цвет каждого треугольника тела: rgb — sRGB 0…255 по три числа на треугольник; null — цвет материала.
  setColors(key, rgb) {
    const it = this.items.get(key); if (!it) return;
    const attr = it.mesh.geometry.getAttribute('color'), a = attr.array, n = a.length / 9;
    if (!rgb) for (let i = 0; i < a.length; i += 3) { a[i] = it.base.r; a[i + 1] = it.base.g; a[i + 2] = it.base.b; }
    else for (let t = 0; t < n; t++) {
      const r = LIN[rgb[3 * t] | 0], g = LIN[rgb[3 * t + 1] | 0], b = LIN[rgb[3 * t + 2] | 0];
      for (let k = 0; k < 9; k += 3) { a[9 * t + k] = r; a[9 * t + k + 1] = g; a[9 * t + k + 2] = b; }
    }
    attr.needsUpdate = true; this.render();
  }

  setSelected(key) { this.selected = key; this._state(); this._buildBodyAxes(); this.render(); }
  setVisible(key, v) {
    const it = this.items.get(key);
    if (it) { it.mesh.visible = it.edges.visible = !!v; this._buildBodyAxes(); this._buildArrows(); this.render(); }
  }
  setOpacity(a) { this.opacity = Math.max(0.05, Math.min(1, a)); this._state(); this.render(); }

  // Секущая плоскость: axis 'x'|'y'|'z' или null; скрывается сторона «+» (flip — сторона «−»).
  setClip(axis, pos = 0, flip = false) {
    this.clipOn = !!axis;
    if (axis) {
      const e = new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
      this.clip.set(flip ? e : e.negate(), flip ? -pos : pos);        // видно там, где n·x + c ≥ 0
    }
    this._state(); this.render();
  }

  // Заливка разреза: треугольники (мм) и цвет на треугольник (sRGB 0…255); null — убрать.
  setSection(tris, rgb) {
    if (this.section) { this.scene.remove(this.section); this.section.geometry.dispose(); this.section.material.dispose(); this.section = null; }
    if (tris && tris.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(tris, 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(tris.length), 3));
      this.section = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      this.scene.add(this.section);
      this.setSectionColors(rgb);
    }
    this.render();
  }

  setSectionColors(rgb) {
    if (!this.section) return;
    const attr = this.section.geometry.getAttribute('color'), a = attr.array, n = a.length / 9;
    for (let t = 0; t < n; t++) {
      const r = rgb ? LIN[rgb[3 * t] | 0] : 0.3, g = rgb ? LIN[rgb[3 * t + 1] | 0] : 0.3, b = rgb ? LIN[rgb[3 * t + 2] | 0] : 0.3;
      for (let k = 0; k < 9; k += 3) { a[9 * t + k] = r; a[9 * t + k + 1] = g; a[9 * t + k + 2] = b; }
    }
    attr.needsUpdate = true; this.render();
  }

  // ---------------------------------------------------------------- осевые линии (этап 3D-7)
  // Оси координат через начало (0; 0; 0): на габарит box (мм, [[min],[max]]) вместе с началом и с запасом,
  // с делениями и подписями в мм через шаг 1, 2 или 5 × 10ⁿ. Разрез их не режет — это опорные линии.
  // null — убрать.
  setCoordAxes(box) {
    this._axesBox = box;                                                      // для перерисовки при смене темы
    this._clear(this.axes);
    if (box) {
      const lo = [0, 1, 2].map(k => Math.min(box[0][k], 0)), hi = [0, 1, 2].map(k => Math.max(box[1][k], 0));
      const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-6);
      const pad = 0.12 * size, step = niceStep(size / 6), tick = 0.012 * size;
      const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
      for (let k = 0; k < 3; k++) {
        const e = [0, 0, 0], u = [0, 0, 0]; e[k] = 1; u[k === 2 ? 0 : 2] = 1;       // деления — поперёк оси
        const at = (t, s = 0) => [0, 1, 2].map(j => e[j] * t + u[j] * s);
        const a = lo[k] - pad, b = hi[k] + pad, seg = [...at(a), ...at(b)];
        for (let i = Math.ceil(a / step); i * step <= b; i++) {
          if (i === 0) continue;
          seg.push(...at(i * step, -tick), ...at(i * step, tick));
          this.axes.add(this._label((i * step).toFixed(dec), LABEL, 12, at(i * step, 3.2 * tick), k));
        }
        this._lines(this.axes, seg, new THREE.Color(AXIS_COL[k]));
        this.axes.add(this._label('XYZ'[k], AXIS_COL[k], 15, at(b + 0.04 * size), k));
      }
    }
    this.render();
  }

  // Осевые линии тел: list — [{key, p0, p1}] в мм (сервер: ось, от которой считается осевое и радиальное
  // намагничивание); штрихпунктир, у выбранного тела — цвет выбора, у скрытого тела линии нет. null — убрать.
  setBodyAxes(list) { this.bodyAxisList = list || []; this._buildBodyAxes(); this.render(); }

  _buildBodyAxes() {
    this._clear(this.bodyAxes);
    if (!this.bodyAxisList.length) return;
    const box = new THREE.Box3();
    for (const it of this.items.values()) box.expandByObject(it.mesh);
    const size = box.isEmpty() ? 1 : box.getSize(new THREE.Vector3()).length();
    const dash = 0.03 * size, gap = 0.01 * size, dot = 0.004 * size;           // штрихпунктир: штрих, точка
    const seg = { sel: [], other: [] };
    for (const { key, p0, p1 } of this.bodyAxisList) {
      const it = this.items.get(key);
      if (!it || !it.mesh.visible) continue;
      const d = [0, 1, 2].map(j => p1[j] - p0[j]), L = Math.hypot(...d), out = key === this.selected ? seg.sel : seg.other;
      if (!(L > 0)) continue;
      const at = t => [0, 1, 2].map(j => p0[j] + d[j] * t / L);
      for (let t = 0; t < L; t += dash + dot + 2 * gap) {
        out.push(...at(t), ...at(Math.min(t + dash, L)));
        const s = t + dash + gap;
        if (s < L) out.push(...at(s), ...at(Math.min(s + dot, L)));
      }
    }
    if (seg.other.length) this._lines(this.bodyAxes, seg.other, new THREE.Color(BODY_AXIS));
    if (seg.sel.length) this._lines(this.bodyAxes, seg.sel, new THREE.Color(BODY_AXIS_SEL));
  }

  // ---------------------------------------------------------------- стрелки намагничивания (этап 3D-7)
  // list — [{key, points (мм, по 3 числа на стрелку), dirs (единичные), spacing (мм)}]: стрелка — вектор M от S к N
  // (хвост → остриё), центр — в точке, длина 0,7 шага. Магниты со стрелками полупрозрачны; разрез режет стрелки,
  // как тела. null — убрать.
  setArrows(list) {
    this.arrowList = list || [];
    this.arrowKeys = new Set(this.arrowList.map(a => a.key));
    this._buildArrows(); this._state(); this.render();
  }

  _buildArrows() {
    this._clear(this.arrows);
    const list = this.arrowList.filter(a => { const it = this.items.get(a.key); return it && it.mesh.visible; });
    const n = list.reduce((s, a) => s + a.points.length / 3, 0);
    if (!n) return;
    if (!this._arrowGeo) {                                   // единичная стрелка вдоль +Y от −0,5 до +0,5
      const shaft = new THREE.CylinderGeometry(0.05, 0.05, 0.62, 10); shaft.translate(0, -0.19, 0);
      const head = new THREE.ConeGeometry(0.14, 0.38, 16); head.translate(0, 0.31, 0);
      this._arrowGeo = [shaft, head];
    }
    const mat = new THREE.MeshLambertMaterial({ color: ARROW, clippingPlanes: this.clipOn ? [this.clip] : [] });
    const parts = this._arrowGeo.map(g => { const m = new THREE.InstancedMesh(g, mat, n); m.userData.shared = true; return m; });
    const M = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    const d = new THREE.Vector3(), p = new THREE.Vector3(), s = new THREE.Vector3();
    let j = 0;
    for (const a of list) {
      const L = 0.7 * a.spacing;
      for (let i = 0; i < a.points.length; i += 3, j++) {
        d.set(a.dirs[i], a.dirs[i + 1], a.dirs[i + 2]);
        if (d.lengthSq() < 1e-12) M.makeScale(0, 0, 0);                    // ось не определена — стрелки нет
        else { q.setFromUnitVectors(up, d.normalize()); M.compose(p.set(a.points[i], a.points[i + 1], a.points[i + 2]), q, s.set(L, L, L)); }
        for (const m of parts) m.setMatrixAt(j, M);
      }
    }
    this.arrows.add(...parts);
  }

  // ---------------------------------------------------------------- силовые линии (этап 3D-7)
  // list — наборы линий [{points (мм, по 3 числа), offsets (начало каждой линии), rgb (цвет точки, sRGB
  // 0…255), onTop}]: ломаные рисуются толстыми линиями (обычная линия WebGL всегда в один пиксель), по ходу
  // через равные промежутки ставятся конусы-стрелки — направление B. onTop — рисовать поверх тел и заливки
  // разреза (так показываются линии, лежащие в плоскости разреза, иначе заливка их прячет). null — убрать.
  setFieldLines(list) {
    this._clear(this.lines);
    this.lineMats = [];
    const box = new THREE.Box3();
    for (const it of this.items.values()) box.expandByObject(it.mesh);
    const size = box.isEmpty() ? 1 : box.getSize(new THREE.Vector3()).length();
    for (const data of (list || [])) {
      if (!data || data.offsets.length < 2) continue;
      const { points, offsets, rgb } = data;
      const gap = 0.09 * size, aLen = 0.022 * size, aRad = 0.007 * size;
      const seg = [], col = [], at = [], dir = [];
      const q = new THREE.Vector3();
      for (let l = 0; l + 1 < offsets.length; l++) {
        let acc = 0.5 * gap;
        for (let i = offsets[l]; i + 1 < offsets[l + 1]; i++) {
          const a = 3 * i, b = 3 * i + 3;
          seg.push(points[a], points[a + 1], points[a + 2], points[b], points[b + 1], points[b + 2]);
          col.push(LIN[rgb[a]], LIN[rgb[a + 1]], LIN[rgb[a + 2]], LIN[rgb[b]], LIN[rgb[b + 1]], LIN[rgb[b + 2]]);
          q.set(points[b] - points[a], points[b + 1] - points[a + 1], points[b + 2] - points[a + 2]);
          const len = q.length();
          acc += len;
          if (acc >= gap && len > 0) {
            acc = 0;
            at.push(0.5 * (points[a] + points[b]), 0.5 * (points[a + 1] + points[b + 1]), 0.5 * (points[a + 2] + points[b + 2]));
            dir.push(q.x / len, q.y / len, q.z / len);
          }
        }
      }
      const top = !!data.onTop;
      if (seg.length) {
        const g = new LineSegmentsGeometry();
        g.setPositions(seg); g.setColors(col);
        const mat = new LineMaterial({ vertexColors: true, linewidth: 2.2, worldUnits: false,
          depthTest: !top, depthWrite: !top, transparent: top });
        mat.resolution.set(this.w || 800, this.h || 600);
        const mesh = new LineSegments2(g, mat);
        mesh.renderOrder = top ? 4 : 0;
        this.lines.add(mesh);
        this.lineMats.push(mat);
      }
      if (at.length) {
        if (!this._coneGeo) this._coneGeo = new THREE.ConeGeometry(1, 1, 12);
        const mat = new THREE.MeshLambertMaterial({ color: ARROW, depthTest: !top, depthWrite: !top, transparent: top });
        const mesh = new THREE.InstancedMesh(this._coneGeo, mat, at.length / 3);
        mesh.userData.shared = true;
        mesh.renderOrder = top ? 5 : 0;
        const M = new THREE.Matrix4(), rot = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
        const p = new THREE.Vector3(), d = new THREE.Vector3(), s = new THREE.Vector3(aRad, aLen, aRad);
        for (let i = 0; i < at.length; i += 3) {
          d.set(dir[i], dir[i + 1], dir[i + 2]);
          rot.setFromUnitVectors(up, d);
          M.compose(p.set(at[i], at[i + 1], at[i + 2]), rot, s);
          mesh.setMatrixAt(i / 3, M);
        }
        this.lines.add(mesh);
      }
    }
    this._state(); this.render();
  }

  // Отрезки (по 6 чисел, мм) двумя проходами: видимые — с проверкой глубины, скрытые телами — бледно поверх.
  _lines(group, xyz, color) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(xyz), 3));
    const vis = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
    const ghost = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: GHOST,
      depthTest: false, depthWrite: false }));
    ghost.renderOrder = 5; vis.userData.shared = true;                        // геометрия общая — освобождать один раз
    group.add(vis, ghost);
  }

  // Подпись постоянного размера на экране (px — высота в пикселях); поверх тел. axis — номер оси координат:
  // подписи оси, смотрящей почти в камеру, скрываются (иначе все деления сходятся в одну точку).
  _label(text, color, px, pos, axis = null) {
    const fs = 48, pad = 10, c = document.createElement('canvas');
    let g = c.getContext('2d'); g.font = '600 ' + fs + 'px sans-serif';
    c.width = Math.ceil(g.measureText(text).width) + 2 * pad; c.height = fs + 2 * pad;
    g = c.getContext('2d'); g.font = '600 ' + fs + 'px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.lineWidth = 10; g.strokeStyle = LABEL_HALO; g.strokeText(text, c.width / 2, c.height / 2 + 2);
    g.fillStyle = color; g.fillText(text, c.width / 2, c.height / 2 + 2);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false,
      sizeAttenuation: false }));
    sp.position.set(...pos); sp.renderOrder = 6;
    sp.userData.label = { px, aspect: c.width / c.height, axis };
    this._labelScale(sp);
    return sp;
  }

  // Без затухания размера спрайт занимает scale·P₁₁ по высоте в нормированных координатах: P₁₁ = 1/tg(fov/2).
  _labelScale(sp) {
    const L = sp.userData.label, k = L.px * 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) / (this.h || 600);
    sp.scale.set(k * L.aspect, k, 1);
  }

  _clear(group) {
    for (const o of [...group.children]) {
      group.remove(o);
      if (o.isInstancedMesh) o.dispose();                                    // буферы положений стрелок
      if (o.geometry && !o.userData.shared) o.geometry.dispose();
      if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
    }
  }

  _state() {
    const planes = this.clipOn ? [this.clip] : [];
    for (const g of [this.arrows, this.lines]) {
      for (const o of g.children) { o.material.clippingPlanes = planes; o.material.needsUpdate = true; }
    }
    for (const [key, it] of this.items) {
      const m = it.mesh.material, sel = key === this.selected;
      const a = (this.arrowKeys && this.arrowKeys.has(key)) ? Math.min(this.opacity, ARROW_BODY_ALPHA) : this.opacity;
      m.transparent = a < 1; m.opacity = a; m.depthWrite = a >= 1;
      m.clippingPlanes = planes; m.emissive.setHex(sel ? 0x0e3a2c : 0x000000); m.needsUpdate = true;
      it.edges.material.color.setHex(sel ? EDGE_SEL : EDGE); it.edges.material.clippingPlanes = planes;
      it.edges.material.needsUpdate = true;
    }
  }

  // ---------------------------------------------------------------- камера
  fit() {
    const box = new THREE.Box3();
    for (const it of this.items.values()) if (it.mesh.visible) box.expandByObject(it.mesh);
    if (box.isEmpty()) { this.render(); return; }
    const c = box.getCenter(new THREE.Vector3());
    const R = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
    const dist = R / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.05;
    const dir = this.camera.position.clone().sub(this.controls.target);
    if (dir.lengthSq() < 1e-12) dir.set(...VIEWS[0][1]);
    dir.normalize();
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(dir, dist);
    this.camera.near = dist / 500; this.camera.far = dist * 500; this.camera.updateProjectionMatrix();
    this._fitted = true;
    this.controls.update(); this.render();
  }

  zoom(f) {
    const t = this.controls.target, d = this.camera.position.clone().sub(t).multiplyScalar(1 / f);
    this.camera.position.copy(t).add(d); this.controls.update(); this.render();
  }

  // Стандартный вид по номеру: 0 — изометрия, 1 — сверху, 2 — спереди, 3 — справа; возвращает его название.
  setView(i) {
    this.viewIdx = ((Math.trunc(i) % VIEWS.length) + VIEWS.length) % VIEWS.length;
    const [name, d] = VIEWS[this.viewIdx];
    const t = this.controls.target, dist = this.camera.position.distanceTo(t);
    this.camera.position.copy(t).addScaledVector(new THREE.Vector3(...d).normalize(), dist);
    this.controls.update(); this.fit();
    return name;
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (w < 2 || h < 2) return;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    for (const o of this.axes.children) if (o.userData.label) this._labelScale(o);   // подписи — постоянного размера
    for (const m of (this.lineMats || [])) m.resolution.set(w, h);   // толщина линий задана в пикселях
    this.render();
  }

  render() {
    if (!this.w) return;
    const r = this.renderer;
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    for (const o of this.axes.children) {                                        // ось вдоль взгляда — без подписей
      const L = o.userData.label;
      if (L && L.axis != null) o.visible = Math.abs(dir.getComponent(L.axis)) < 0.9;
    }
    r.setScissorTest(false); r.setViewport(0, 0, this.w, this.h); r.clear();
    r.render(this.scene, this.camera);
    const s = Math.min(92, Math.floor(Math.min(this.w, this.h) * 0.22)), m = 8;   // оси в углу
    this.gCam.position.copy(dir.multiplyScalar(6)); this.gCam.quaternion.copy(this.camera.quaternion);
    r.setScissorTest(true); r.setViewport(m, m, s, s); r.setScissor(m, m, s, s);
    r.clearDepth(); r.render(this.gScene, this.gCam); r.setScissorTest(false);
  }

  screenshot() { this.render(); return this.renderer.domElement.toDataURL('image/png'); }

  // Смена темы оформления: фон, рёбра тел, оси тел и координат с подписями, стрелки — новыми цветами.
  applyTheme() {
    readTheme();
    this.renderer.setClearColor(BG, 1);
    this._state();                                                            // рёбра тел
    this._buildBodyAxes();
    this._buildArrows();
    for (const o of this.lines.children) if (o.isInstancedMesh) o.material.color.setHex(ARROW);   // стрелки силовых линий
    if (this._axesBox !== undefined) this.setCoordAxes(this._axesBox);      // подписи делений — заново
    this.render();
  }

  _gizmo() {
    this.gScene = new THREE.Scene();
    this.gCam = new THREE.OrthographicCamera(-1.8, 1.8, 1.8, -1.8, 0.1, 20);
    this.gScene.add(new THREE.AxesHelper(1.15));                      // X красная, Y зелёная, Z синяя
    for (const [txt, pos, col] of [['X', [1.5, 0, 0], '#e5544e'], ['Y', [0, 1.5, 0], '#42c25a'], ['Z', [0, 0, 1.5], '#4d8bff']]) {
      const c = document.createElement('canvas'); c.width = c.height = 64;
      const g = c.getContext('2d');
      g.fillStyle = col; g.font = 'bold 44px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(txt, 32, 34);
      const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
      sp.position.set(...pos); sp.scale.set(0.6, 0.6, 1); this.gScene.add(sp);
    }
  }

  _pick(e) {
    if (!this.onPick) return;
    const rc = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((e.clientX - rc.left) / rc.width) * 2 - 1, -((e.clientY - rc.top) / rc.height) * 2 + 1);
    this.ray.setFromCamera(p, this.camera);
    const meshes = [...this.items.values()].filter(it => it.mesh.visible).map(it => it.mesh);
    for (const h of this.ray.intersectObjects(meshes, false)) {
      if (this.clipOn && this.clip.distanceToPoint(h.point) < 0) continue;  // срезанная часть не выбирается
      this.onPick(h.object.userData.key); return;
    }
    this.onPick(null);
  }
}
