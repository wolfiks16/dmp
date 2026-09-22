from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magcore.fem2d.model.materials import MagnetMaterial
from magcore.fem3d.mesh import _FACES, TetMesh3D, unique_rows
from magcore.fem3d.problem import Problem3D
from magcore.fem3d.scalar import ScalarField3D, p1_gradients

# СИЛОВЫЕ ЛИНИИ ПОЛЯ B (этап 3D-7, план — docs/plan_3d_2026-09-11.md). Только для рисунка: все числа
# (силы, потоки, размагничивание) считаются по полю в ячейках и здесь не участвуют.
#
# ГДЕ НАЧИНАТЬ. Токов нет, значит H = −∇φ и ∮H·dl = 0 по любому замкнутому пути. Вне магнитов H направлено
# вдоль B (μ > 0), поэтому вдоль линии B потенциал φ строго убывает: замкнуться, не заходя в магнит, линия
# не может, а внутри магнита H может быть противоположно B. Отсюда: начала линий достаточно ставить на
# гранях магнитов, через которые поток ВЫХОДИТ (B·n > 0), и ни одна линия не потеряется. Число начал на
# грани — по её потоку, так что каждая линия несёт одинаковый поток ΔΦ = Φ_выход / n_lines: где линии гуще,
# там больше индукция. Линия ведётся вперёд до выхода из СЛЕДУЮЩЕГО магнита — дальше её продолжают линии,
# начатые на его гранях, и каждый участок потока нарисован ровно один раз. При внешнем поле или границе
# φ = 0 часть линий входит снаружи — начала ставятся и на внешней границе, где поток входит в область.
#
# ПО КАКОМУ ПОЛЮ ВЕСТИ. Решатель даёт B ПОСТОЯННОЙ в ячейке. Вести линию прямо по нему нельзя: нормальная
# составляющая B на грани непрерывна лишь в среднем (слабая форма), и там, где поле почти касается грани,
# соседние ячейки гонят линию то наружу, то обратно. Проверено на намагниченном шаре: 8 линий из 20
# «залипали» у поверхности, делая тысячи шагов нулевой длины. Поэтому для рисунка поле восстанавливается
# В УЗЛАХ: среднее по объёму прилегающих ячеек ТОГО ЖЕ тела (на стыке тел у каждого тела своё значение,
# поэтому скачок касательной составляющей на границе материалов сохраняется). Внутри ячейки поле
# интерполируется барицентрически, и направление становится непрерывным. Это обычное восстановление
# (осреднение по патчу), оно линейно по решению и ничего не добавляет от себя; точность проверена
# оракулами: однородное поле — линии строго прямые, поле диполя и намагниченный шар — вдоль линии
# сохраняется r/sin²θ.
#
# ШАГ. Половина размера ячейки, по средней точке (метод Рунге — Кутты второго порядка): поле кусочно-линейно
# и не гладко, поэтому схемы выше второго порядка смысла не имеют. Ячейку следующей точки находим ходом по
# соседям (барицентрические координаты показывают, через какую грань выйти).

STOP_MAGNET = 0        # вошла в магнит и вышла из него — нормальный конец участка
STOP_BOUNDARY = 1      # ушла за внешнюю границу области
STOP_ZERO = 2          # поле почти ноль
STOP_LENGTH = 3        # исчерпан предел длины
STOP_STEPS = 4         # исчерпан предел числа шагов
STOP_LOST = 5          # не удалось найти ячейку точки (ход по соседям не сошёлся)

STEP_FRAC = 0.5        # шаг = STEP_FRAC · размер текущей ячейки
WALK_HOPS = 24         # столько переходов по соседям на поиск ячейки точки
_LAM_TOL = -1.0e-9     # настолько отрицательная барицентрическая координата ещё считается «внутри»
# Линии на разрезе: если поле почти перпендикулярно плоскости, его проекция — численный шум, и рисовать по
# ней линии нельзя (проверено: в экваториальной плоскости шара, где B перпендикулярна ей, получались линии
# длиной в три габарита из ничего). Линия ведётся только там, где в плоскости лежит хотя бы эта доля поля.
SECTION_MIN_IN_PLANE = 0.1


@dataclass(frozen=True)
class FieldLines3D:
    """
    Силовые линии: точки всех линий подряд (P,3) [м], начало каждой в `offsets` (L+1,), |B| в точке (P,) [Тл],
    поток на линию `delta_flux` [Вб], причина остановки каждой линии (L,) — коды STOP_*.
    """

    points: np.ndarray
    offsets: np.ndarray
    values: np.ndarray
    delta_flux: float
    stop: np.ndarray
    out_of_plane: float = 0.0      # у линий разреза — медиана |B·n|/|B| по точкам (0 на плоскости симметрии)

    @property
    def n_lines(self) -> int:
        return int(self.offsets.size - 1)

    def line(self, i: int) -> np.ndarray:
        return self.points[self.offsets[i]:self.offsets[i + 1]]

    def line_values(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]


@dataclass(frozen=True)
class _FaceData:
    """Грани ячеек (M,4): соседняя ячейка (−1 — внешняя граница), наружная нормаль, площадь, вершины."""

    neighbor: np.ndarray
    normal: np.ndarray
    area: np.ndarray
    tri: np.ndarray


_FACE_CACHE: dict[int, tuple[TetMesh3D, _FaceData]] = {}


def face_data(mesh: TetMesh3D) -> _FaceData:
    """Грани ячеек с соседями и наружными нормалями (кэш на последнюю сетку — считается за один проход)."""
    hit = _FACE_CACHE.get(id(mesh))
    if hit is not None and hit[0] is mesh:
        return hit[1]
    tri = mesh.vertices[mesh.cells[:, _FACES]]                      # (M,4,3,3): ячейка, грань, вершина, координата
    vec = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    area = 0.5 * np.linalg.norm(vec, axis=2)
    normal = np.divide(vec, (2.0 * area)[:, :, None], out=np.zeros_like(vec), where=area[:, :, None] > 0.0)
    flat = np.sort(mesh.cells[:, _FACES].reshape(-1, 3), axis=1)
    _, inv = unique_rows(flat, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    owner = np.repeat(np.arange(mesh.n_cells), 4)
    order = np.argsort(inv, kind="stable")
    same = inv[order[1:]] == inv[order[:-1]]
    i, j = order[:-1][same], order[1:][same]
    nb = np.full(flat.shape[0], -1, dtype=np.int64)
    nb[i], nb[j] = owner[j], owner[i]
    out = _FaceData(nb.reshape(-1, 4), normal, area, tri)
    _FACE_CACHE.clear()
    _FACE_CACHE[id(mesh)] = (mesh, out)
    return out


def magnet_cells(problem: Problem3D, names=None) -> np.ndarray:
    """Маска ячеек магнитов (по желанию — только перечисленных объектов)."""
    reg = np.asarray(problem.cell_region)
    keep = np.zeros(max(problem.regions) + 1, dtype=bool)
    for rid, r in problem.regions.items():
        if isinstance(r.material, MagnetMaterial) and (names is None or r.name in set(names)):
            keep[rid] = True
    return keep[reg]


def nodal_field(problem: Problem3D, cell_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Поячеечная величина → значения в узлах ПО ТЕЛАМ: среднее по объёму прилегающих ячеек того же тела.
    Возвращает (значения (K,3), номер значения для каждой вершины каждой ячейки (M,4)); на стыке тел у
    каждого тела свой узловой вектор, поэтому скачок поля на границе материалов не размазывается.
    """
    mesh = problem.mesh
    vol = mesh.cell_volumes()
    key = np.stack([np.repeat(np.asarray(problem.cell_region), 4), mesh.cells.ravel()], axis=1)
    _, inv = unique_rows(key, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    w = np.bincount(inv, weights=np.repeat(vol, 4))
    acc = np.stack([np.bincount(inv, weights=np.repeat(cell_values[:, k] * vol, 4)) for k in range(3)], axis=1)
    return acc / w[:, None], inv.reshape(-1, 4)


def _triangle_points(tri: np.ndarray, k: int) -> np.ndarray:
    """k различных точек внутри треугольника (k,3): центр при k = 1, дальше — правильная сетка по барицентру."""
    if k <= 1:
        return tri.mean(axis=0)[None, :]
    m = int(np.ceil(0.5 * (np.sqrt(8.0 * k + 1.0) - 1.0)))                     # строк сетки: m(m+1)/2 ≥ k
    a, b = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    sel = (a + b) < m
    u = (a[sel] + 1.0 / 3.0) / m
    v = (b[sel] + 1.0 / 3.0) / m
    return np.stack([1.0 - u - v, u, v], axis=1)[:k] @ tri


def _seed_faces(field: ScalarField3D, magnets: np.ndarray, fd: _FaceData):
    """
    Грани, через которые поток ВЫХОДИТ из магнитов в немагнитную ячейку, плюс грани внешней границы, через
    которые поток ВХОДИТ в область (внешнее поле, граница φ = 0). Возвращает (ячейка, номер грани, поток).
    """
    flux = np.einsum("mfk,mk->mf", fd.normal, field.B_cells) * fd.area          # (M,4) поток наружу из ячейки
    inner = fd.neighbor >= 0
    other_magnet = np.zeros_like(inner)
    other_magnet[inner] = magnets[fd.neighbor[inner]]
    out_of_magnet = magnets[:, None] & inner & ~other_magnet & (flux > 0.0)     # магнит → не магнит, наружу
    into_domain = (~inner) & (flux < 0.0)                                       # снаружи внутрь через границу
    cells, faces = np.where(out_of_magnet | into_domain)
    return cells, faces, np.abs(flux[cells, faces])


def _locate(points, start, grads, verts, neighbor):
    """Ячейка каждой точки ходом по соседям от start: (ячейка, ушла за границу, барицентрические координаты)."""
    cur = np.array(start, dtype=np.int64, copy=True)
    gone = np.zeros(points.shape[0], dtype=bool)
    lam = np.zeros((points.shape[0], 4))
    rows = np.arange(points.shape[0])
    for _ in range(WALK_HOPS):
        lam = 1.0 + np.einsum("nfk,nfk->nf", grads[cur], points[:, None, :] - verts[cur])
        f = np.argmin(lam, axis=1)
        bad = (lam[rows, f] < _LAM_TOL) & ~gone
        if not bad.any():
            break
        nb = neighbor[cur[bad], f[bad]]
        out = nb < 0
        idx = rows[bad]
        gone[idx[out]] = True
        cur[idx[~out]] = nb[~out]
    return cur, gone, lam


def _plane_basis(normal):
    """Единичная нормаль и два единичных вектора в плоскости."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return n, e1, np.cross(n, e1)


def _seed_cells(mesh: TetMesh3D, points, *, k: int = 32, tol: float = 1.0e-9):
    """Ячейка каждой точки (−1 — вне сетки) и барицентрические координаты; без перебора всех ячеек."""
    from scipy.spatial import cKDTree

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    kk = int(min(k, mesh.n_cells))
    _, cand = cKDTree(mesh.cell_centroids()).query(pts, k=kk)
    cand = np.asarray(cand).reshape(pts.shape[0], kk)
    grads, _ = p1_gradients(mesh)
    verts = mesh.vertices[mesh.cells]
    lam = 1.0 + np.einsum("ncfk,ncfk->ncf", grads[cand], pts[:, None, None, :] - verts[cand])
    ok = lam.min(axis=2) >= -tol
    hit = ok.any(axis=1)
    first = ok.argmax(axis=1)
    rows = np.arange(pts.shape[0])
    return np.where(hit, cand[rows, first], -1), lam[rows, first], hit


def trace_section_lines(field: ScalarField3D, point, normal, *, n_lines: int = 60, max_steps: int = 2000,
                        max_length: float | None = None) -> FieldLines3D:
    """
    Силовые линии в плоскости разреза (этап 3D-7) — линии ПРОЕКЦИИ B на плоскость. На плоскости симметрии
    модели (где B лежит в плоскости) это настоящие линии поля; на любой другой плоскости это проекция, и
    густота линий ничего не значит — в интерфейсе подписано. Линии проводятся РАВНОМЕРНО (не по потоку, в
    отличие от объёмных): кандидаты — узлы решётки на плоскости, отсортированные по убыванию |B| в плоскости;
    линия строится, если её начало дальше d_sep от уже построенных, и ведётся в обе стороны, пока не выйдет
    из области, не подойдёт ближе половины d_sep к другой линии, не потеряет поле или не упрётся в пределы.
    Поле берётся то же, что у объёмных линий (восстановленное в узлах по телам), шаг — половина ячейки.
    """
    if int(n_lines) < 1:
        raise ValueError("n_lines должен быть не меньше 1.")
    prob = field.problem
    mesh = prob.mesh
    fd = face_data(mesh)
    grads, _ = p1_gradients(mesh)
    verts = mesh.vertices[mesh.cells]
    nodal, node_of = nodal_field(prob, field.B_cells)
    size_cell = np.cbrt(6.0 * np.abs(mesh.cell_volumes()))
    nhat, e1, e2 = _plane_basis(normal)
    p0 = np.asarray(point, dtype=float)
    corners = mesh.vertices
    u = (corners - p0) @ e1
    v = (corners - p0) @ e2
    span = max(u.max() - u.min(), v.max() - v.min())
    d_sep = span / int(n_lines)            # шаг расстановки: у однородного поля выйдет ровно n_lines линий
    d_test = 0.5 * d_sep
    if max_length is None:
        max_length = span                                       # линия пересекает картинку один раз
    b_zero = 1.0e-9 * max(float(np.linalg.norm(field.B_cells, axis=1).max()), 1.0e-300)
    g = min(int(np.ceil(span / (0.5 * d_sep))) + 1, 240)        # решётка кандидатов (не крупнее 240 × 240)
    gu, gv = np.meshgrid(np.linspace(u.min(), u.max(), g), np.linspace(v.min(), v.max(), g), indexing="ij")
    cand = p0 + gu.ravel()[:, None] * e1 + gv.ravel()[:, None] * e2
    cells, lam, inside = _seed_cells(mesh, cand)

    def in_plane(c, l):
        """
        Поле в точке, единичное направление его проекции на плоскость (None — поле почти перпендикулярно
        плоскости, см. SECTION_MIN_IN_PLANE) и доля поля, выходящая из плоскости.
        """
        b = l @ nodal[node_of[c]]
        mag = float(np.linalg.norm(b))
        bp = b - (b @ nhat) * nhat
        n = float(np.linalg.norm(bp))
        if mag <= b_zero or n < SECTION_MIN_IN_PLANE * mag:
            return mag, None, 1.0
        return mag, bp / n, abs(float(b @ nhat)) / mag

    order = []
    for i in np.where(inside)[0]:
        mag, d, _ = in_plane(int(cells[i]), lam[i])
        if d is not None:
            order.append((mag, int(i)))
    order.sort(key=lambda t: -t[0])
    from scipy.spatial import cKDTree

    taken: list[np.ndarray] = []                                 # точки уже построенных линий (для расстояний)
    tree = None

    def too_close(q, dist):
        return tree is not None and float(tree.query(q)[0]) < dist

    def run(start_cell, start_lam, start_p, sign):
        """Одна ветвь линии: точки, |B| в них и доли поля, выходящие из плоскости."""
        pts, vals, outs = [start_p.copy()], [], []
        c, l, p = int(start_cell), start_lam, start_p.copy()
        mag, d, frac = in_plane(c, l)
        vals.append(mag)
        outs.append(frac)
        total = 0.0
        for _ in range(int(max_steps)):
            if d is None:
                break
            s = STEP_FRAC * size_cell[c]
            mid_c, mid_gone, mid_lam = _locate((p + 0.5 * s * sign * d)[None, :], [c], grads, verts, fd.neighbor)
            if mid_gone[0]:
                break
            dm = in_plane(int(mid_c[0]), mid_lam[0])[1]
            q = p + s * sign * (dm if dm is not None else d)
            nc, gone, nl = _locate(q[None, :], [c], grads, verts, fd.neighbor)
            if gone[0] or nl[0].min() < _LAM_TOL:
                break
            total += s
            p, c, l = q, int(nc[0]), nl[0]
            mag, d, frac = in_plane(c, l)
            pts.append(p.copy())
            vals.append(mag)
            outs.append(frac)
            if total > max_length or too_close(p, d_test):
                break
        return np.array(pts), np.array(vals), np.array(outs)

    lines, values, fracs = [], [], []
    for _, i in order:
        if len(lines) >= int(n_lines):
            break
        q = cand[i]
        if too_close(q, d_sep):
            continue
        fwd, vf, of = run(cells[i], lam[i], q, 1.0)
        bwd, vb, ob = run(cells[i], lam[i], q, -1.0)
        join = (lambda a, b: np.concatenate([a[::-1], b[1:]]) if b.shape[0] > 1 else a[::-1])
        pts, val, frac = join(bwd, fwd), join(vb, vf), join(ob, of)
        if pts.shape[0] < 3:
            continue
        lines.append(pts)
        values.append(val)
        fracs.append(frac)
        taken.append(pts)
        tree = cKDTree(np.concatenate(taken))
    if not lines:
        raise ValueError("в этой плоскости линий не получилось: поле почти перпендикулярно ей "
                         "(разрез не вдоль поля) или в плоскости нет модели.")
    offs = np.concatenate([[0], np.cumsum([q.shape[0] for q in lines])]).astype(np.int64)
    return FieldLines3D(np.concatenate(lines), offs, np.concatenate(values), 0.0,
                        np.zeros(len(lines), dtype=np.int64),
                        out_of_plane=float(np.median(np.concatenate(fracs))))


def trace_field_lines(field: ScalarField3D, *, n_lines: int = 200, objects=None, max_length: float | None = None,
                      max_steps: int = 4000) -> FieldLines3D:
    """
    Силовые линии поля B (этап 3D-7; устройство и обоснование — в шапке модуля). `n_lines` — сколько линий,
    каждая несёт одинаковый поток ΔΦ; `objects` — начинать только с этих магнитов; `max_length` — предел
    длины линии [м] (по умолчанию три диагонали области); `max_steps` — предел числа шагов.
    """
    if int(n_lines) < 1:
        raise ValueError("n_lines должен быть не меньше 1.")
    prob = field.problem
    mesh = prob.mesh
    fd = face_data(mesh)
    magnets = magnet_cells(prob, objects)
    if not magnets.any():
        raise ValueError("нет магнитов, от которых начинать линии"
                         + ("." if objects is None else f": {list(objects)}."))
    cells, faces, flux = _seed_faces(field, magnets, fd)
    if cells.size == 0 or flux.sum() <= 0.0:
        raise ValueError("поток из магнитов равен нулю — линии строить не от чего.")
    n = int(n_lines)
    d_flux = float(flux.sum()) / n
    # Начала — через равные доли потока: линия № i берётся с грани, на которую приходится (i + ½)·ΔΦ.
    pick = np.clip(np.searchsorted(np.cumsum(flux), (np.arange(n) + 0.5) * d_flux), 0, cells.size - 1)
    uniq, counts = np.unique(pick, return_counts=True)
    p0, c0 = [], []
    for f_idx, k in zip(uniq, counts):
        c, f = int(cells[f_idx]), int(faces[f_idx])
        nb = int(fd.neighbor[c, f])
        if magnets[c] and nb < 0:
            continue                                    # магнит выходит на внешнюю границу — линию не начинаем
        start = nb if magnets[c] else c                 # из магнита — в соседнюю ячейку, с границы — внутрь
        pts = _triangle_points(fd.tri[c, f], int(k))
        p0.append(pts)
        c0.append(np.full(pts.shape[0], start, dtype=np.int64))
    if not p0:
        raise ValueError("не нашлось граней, с которых можно начать линии.")
    p = np.concatenate(p0)
    cell = np.concatenate(c0)
    L = p.shape[0]
    grads, _ = p1_gradients(mesh)
    verts = mesh.vertices[mesh.cells]
    nodal, node_of = nodal_field(prob, field.B_cells)
    size = np.cbrt(6.0 * np.abs(mesh.cell_volumes()))                        # характерный размер ячейки
    scale = float(np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)))
    if max_length is None:
        max_length = 3.0 * scale
    b_zero = 1.0e-9 * max(float(np.linalg.norm(field.B_cells, axis=1).max()), 1.0e-300)

    def direction(rows, lam):
        """Поле в точке по узловым значениям её ячейки и единичное направление (нули там, где поля нет)."""
        b = np.einsum("nf,nfk->nk", lam, nodal[node_of[rows]])
        nb = np.linalg.norm(b, axis=1)
        d = np.divide(b, nb[:, None], out=np.zeros_like(b), where=nb[:, None] > b_zero)
        return b, nb, d

    alive = np.ones(L, dtype=bool)
    stop = np.full(L, STOP_STEPS, dtype=np.int64)
    was_in_magnet = magnets[cell].copy()
    length = np.zeros(L)
    n_pts = np.ones(L, dtype=np.int64)
    _, _, lam0 = _locate(p, cell, grads, verts, fd.neighbor)
    bval = np.zeros(L)
    bval[:] = direction(cell, lam0)[1]
    steps = [(p.copy(), bval.copy())]
    lam = lam0
    for _ in range(int(max_steps)):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        c = cell[idx]
        b, bn, d1 = direction(c, lam[idx])
        dead = bn <= b_zero
        if dead.any():
            alive[idx[dead]] = False
            stop[idx[dead]] = STOP_ZERO
            idx = idx[~dead]
            if idx.size == 0:
                break
            c, d1 = cell[idx], d1[~dead]
        s = STEP_FRAC * size[c]
        mid, gone_m, lam_m = _locate(p[idx] + 0.5 * s[:, None] * d1, c, grads, verts, fd.neighbor)
        d2 = np.where(gone_m[:, None], d1, direction(mid, lam_m)[2])         # вышла на середине шага — по d1
        d2 = np.where(np.linalg.norm(d2, axis=1)[:, None] > 0.0, d2, d1)
        new_p = p[idx] + s[:, None] * d2
        new_c, gone, lam_n = _locate(new_p, c, grads, verts, fd.neighbor)
        b_new, bn_new, _ = direction(new_c, lam_n)
        p[idx] = new_p
        lam[idx] = lam_n
        cell[idx] = np.where(gone, c, new_c)
        length[idx] += s
        n_pts[idx] += 1
        left_magnet = (~gone) & magnets[c] & ~magnets[new_c] & was_in_magnet[idx]
        was_in_magnet[idx] |= (~gone) & magnets[new_c]
        over = length[idx] > max_length
        lost = (~gone) & (lam_n.min(axis=1) < _LAM_TOL)                      # ход по соседям не нашёл ячейку
        done = gone | left_magnet | over | lost
        bval[idx] = np.where(gone, bval[idx], bn_new)          # за границей значения нет — оставляем прежнее
        if done.any():
            alive[idx[done]] = False
            stop[idx[over]] = STOP_LENGTH
            stop[idx[lost]] = STOP_LOST
            stop[idx[left_magnet]] = STOP_MAGNET
            stop[idx[gone]] = STOP_BOUNDARY
        steps.append((p.copy(), bval.copy()))
    P = np.stack([s[0] for s in steps])                                      # (S,L,3)
    V = np.stack([s[1] for s in steps])                                      # (S,L)
    offs = np.concatenate([[0], np.cumsum(n_pts)]).astype(np.int64)
    pts = np.concatenate([P[:n_pts[i], i] for i in range(L)])
    vals = np.concatenate([V[:n_pts[i], i] for i in range(L)])
    return FieldLines3D(pts, offs, vals, d_flux, stop)
