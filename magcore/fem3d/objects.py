from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from magcore.fem2d.model.materials import MagnetMaterial
from magcore.fem3d.mesh import TetMesh3D, orient_cells
from magcore.fem3d.problem import Problem3D, Region3D

# ОБЪЕКТНАЯ СВОБОДНАЯ 3D-ГЕОМЕТРИЯ (этап 3D-1, план — docs/plan_3d_2026-09-11.md). Модель =
# фон-домен + список объектов-примитивов. Каждый примитив задан в СВОЕЙ системе координат
# (центр в начале, главная ось — z) и размещается поворотом R = Rz·Ry·Rx (углы rx, ry, rz вокруг
# ГЛОБАЛЬНЫХ осей, по порядку X → Y → Z) и переносом: p = c + R·p_лок.
#
# ПРИНАДЛЕЖНОСТЬ ЯЧЕЙКИ ОБЪЕКТУ берётся из карты фрагментов gmsh, а не по «центроид ∈ объект»,
# как в 2D. У вогнутой кривой поверхности (внутренняя стенка трубы) плоская грань тетраэдра
# проходит внутри отверстия, и центроид может оказаться не в том объекте. Карта фрагментов
# точна: gmsh сам знает, из каких исходных тел состоит каждый кусок объёма. Наложение — по
# приоритету, как в 2D: больший priority сверху, при равном — позже добавленный.
#
# РАЗМЕР СЕТКИ — по расстоянию с ОТНОСИТЕЛЬНЫМ ростом: h(p) = min(h_домена, min_i h_i·(1 + g·d_i/L_i)),
# d_i — расстояние до объекта i (0 внутри), L_i — его наибольший габарит. Рост пропорционален
# самому h_i, поэтому уменьшение h измельчает ВСЮ сетку, включая ближнее поле снаружи, и
# проверка сходимости ведёт себя по теории (второй порядок для средних по объёму полей). При
# абсолютном росте h_i + g·d дальняя сетка от h не зависела, и сходимость упиралась в неё:
# у намагниченного шара при h = R/6 ошибка стояла на 2,6–4,9 % и падала лишь как h¹.
# Резкий скачок «мелко внутри — грубо снаружи» в 3D даёт вырожденные элементы и лишние ячейки.
# Кривые поверхности дополнительно разрешаются по кривизне: `curvature_elements` на оборот.
#
# Размеры примитивов (СИ, в своей системе координат):
#   box          lx, ly, lz               — полные размеры вдоль локальных осей
#   cylinder     r, h                     — ось z, центр посередине высоты
#   tube         r_in, r_out, h           — труба (кольцо × высота)
#   tube_sector  r_in, r_out, h, a1, a2   — сектор трубы от угла a1 до a2 против часовой [рад]
#   sphere       r
#   prism        points [(x, y), …], h    — простой многоугольник в плоскости xy, выдавленный на h
#
# ТЕЛО ИЗ CAD (этап 3D-1б):
#   step         path, body [, volume, centroid, axis_origin, axis_dir]
#                — тело № body (с нуля, в порядке файла) из STEP-файла. Геометрия — в координатах CAD;
#                единицы, объявленные в файле, gmsh/OpenCASCADE переводит в метры; center/rotation
#                дополнительно размещают тело тем же поворотом X → Y → Z вокруг начала координат CAD и
#                переносом (по умолчанию — как в CAD). Имён тел в STEP может не быть (T-FLEX выгружает
#                всё под одним изделием), поэтому тело узнаётся по номеру, а volume [м³] и centroid [м],
#                запомненные при выборе, — «отпечаток»: не совпал — файл изменился, ошибка, а не молча
#                другая геометрия. «Осевое» и «радиальное» намагничивание — вокруг прямой
#                axis_origin + t·axis_dir в координатах CAD (по умолчанию — ось z через начало). Объём и
#                габарит — от OpenCASCADE; знакового расстояния формулой нет, размер сетки вокруг тела
#                задают поля gmsh по точному расстоянию до его граней (`_cad_size_fields`).

PRIMITIVES_3D = ("box", "cylinder", "tube", "tube_sector", "sphere", "prism")
CAD_KIND = "step"
KINDS_3D = PRIMITIVES_3D + (CAD_KIND,)
MAGNET_DIRS_3D = ("axial", "axial-in", "radial", "radial-in")
_TWO_PI = 2.0 * math.pi
# Расстояние до граней тела CAD gmsh считает по точкам выборки: DISTANCE_SAMPLING на каждое
# параметрическое направление грани. Ошибка расстояния — не больше полушага выборки, у полной
# цилиндрической грани ≤ πL/(2·20) (L — наибольший габарит тела); размер сетки снаружи у поверхности
# тогда не крупнее h·(1 + g·π/40) ≈ 1,16·h при g = 2. Добавка пропорциональна h, поэтому измельчение
# сетки остаётся равномерным (Л-88). На самих гранях, рёбрах и внутри тела размер — ровно h.
DISTANCE_SAMPLING = 20
# «Отпечаток» тела: объём и центр масс при повторном чтении того же файла совпадают до округления
# интегрирования OpenCASCADE; правка в CAD на порядки больше (сдвиг грани магнита толщиной 2 мм на
# 0,01 мм меняет объём на 0,5 %).
_FINGERPRINT_TOL = 1.0e-6
# Предпросмотр (только показ, в расчёт не идёт): отклонение хорды от поверхности — в долях размера
# грани (относительный режим OpenCASCADE, как у STL из CAD). 1e-3 — значение gmsh по умолчанию: у
# тела Ø38 мм это сотые доли миллиметра, меньше пикселя на экране.
PREVIEW_LINEAR_DEFLECTION = 1.0e-3
# Стрелки намагничивания на виде (этап 3D-7): шаг решётки — наибольший габарит тела / ARROWS_ALONG, стрелок на
# тело не больше ARROWS_MAX (берутся равномерно по списку). До сетки точки — центры клеток решётки, которые
# OpenCASCADE признаёт лежащими внутри тела (около 0,4 мс на точку); если внутрь попало меньше ARROWS_MIN даже
# при шаге вдвое мельче (тонкое или наклонное тело), точки берутся от граней внутрь на глубину V/S — у
# пластины толщины t это ровно середина, t/2.
ARROWS_ALONG = 6
ARROWS_MAX = 60
ARROWS_MIN = 4
# Сетка у мелких особенностей формы (этап 3D-8 — сетка по CAD; `_feature_sizes`, `_feature_fields`). Обычные
# настройки сеточных программ, только записанные явно:
#   SIZE_GROWTH — соседние ячейки отличаются по размеру не больше чем в 1,3 раза (размер растёт от мелкой
#     грани линейно с наклоном 0,3); без этого у скругления в 0,12 мм через шаг стоит ячейка 1 мм, и на стыке
#     получаются почти плоские ячейки (сборка БПЛА32: худшее качество 0,0005, 2 419 ячеек с углом > 170°);
#   THIN_FACTOR — размер на гранях узкого места = толщина × 1: замерено на двух соосных цилиндрах с зазором
#     0,15 мм — 2–3 ячейки поперёк, медианное качество в зазоре 0,83 (при ячейке 0,47 мм было 0,21);
#   THIN_FLOOR — но не мельче 0,1·h: у клина, сходящегося в касание (скругление у соседнего тела), толщина
#     стремится к нулю, а сколь угодно мелкая сетка у самого касания всё равно не даст хороших ячеек;
#   CURVATURE_FLOOR — кривая грань: 2πR/n на оборот, но не мельче 0,25·h — скругление 0,3 мм при h = 1 мм
#     получает ячейку 0,25 мм (дуга в четверть оборота — два отрезка, отход хорды 0,02 мм);
#   _ACROSS_COS — точки двух граней считаются стоящими друг НАПРОТИВ, если нормали граней почти параллельны и
#     отрезок между точками отклоняется от обеих нормалей меньше чем на 45°. Без условия параллельности у
#     выпуклого ребра со скруглением две перпендикулярные грани «видели» друг друга под 45° и считали ребро
#     узким местом (брусок со скруглением 0,025 мм: ячеек вдвое больше, худшее качество 0,002).
SIZE_GROWTH = 1.3
THIN_FACTOR = 1.0
THIN_FLOOR = 0.1
CURVATURE_FLOOR = 0.25
_ACROSS_COS = 0.7


def rotation_matrix(angles) -> np.ndarray:
    """R = Rz(rz)·Ry(ry)·Rx(rx): повороты вокруг ГЛОБАЛЬНЫХ осей — сначала X, затем Y, затем Z."""
    rx, ry, rz = (float(a) for a in angles)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def sector_span(a1: float, a2: float) -> float:
    """Угловой размах сектора от a1 до a2 против часовой стрелки, в [0, 2π)."""
    return (float(a2) - float(a1)) % _TWO_PI


def polygon_area(points) -> float:
    """Ориентированная площадь многоугольника (формула шнурования); > 0 при обходе против часовой."""
    p = np.asarray(points, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _orient2(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def polygon_self_intersects(points) -> bool:
    """Есть ли собственное пересечение несоседних рёбер (касание концами не считается)."""
    p = np.asarray(points, dtype=float)
    n = p.shape[0]
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(i + 1, n):
            if (j + 1) % n == i or (i + 1) % n == j:
                continue                                          # соседние рёбра
            c, d = p[j], p[(j + 1) % n]
            if (_orient2(c, d, a) * _orient2(c, d, b) < 0.0
                    and _orient2(a, b, c) * _orient2(a, b, d) < 0.0):
                return True
    return False


@dataclass(frozen=True)
class GeoObject3D:
    """Объект: примитив (kind + params в СИ, в своей системе) + размещение + материал + опции."""

    name: str
    kind: str
    params: dict
    material: object                       # Air | LinearMaterial | SteelMaterial | MagnetMaterial
    center: tuple = (0.0, 0.0, 0.0)        # [м]
    rotation: tuple = (0.0, 0.0, 0.0)      # (rx, ry, rz) [рад], по порядку X → Y → Z
    magnet_dir: object = None              # 'axial' (по умолчанию) | 'axial-in' | 'radial' |
                                           # 'radial-in' | (dx, dy, dz) — глобальное направление
    mesh_size: float | None = None         # свой размер элемента [м], иначе общий
    priority: int = 1                      # приоритет наложения: больше = выше
    magnet_rotation: tuple = (0.0, 0.0, 0.0)   # (rx, ry, rz) [рад] — поворот направления из magnet_dir
                                           # вокруг ГЛОБАЛЬНЫХ осей X → Y → Z, как поворот тела

    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix(self.rotation)

    def to_local(self, points) -> np.ndarray:
        """Глобальные точки (N,3) → система объекта: Rᵀ·(p − c)."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return (p - np.asarray(self.center, dtype=float)) @ self.rotation_matrix()

    def to_global(self, points_local) -> np.ndarray:
        """Точки в системе объекта (N,3) → глобальные: c + R·p."""
        p = np.atleast_2d(np.asarray(points_local, dtype=float))
        return p @ self.rotation_matrix().T + np.asarray(self.center, dtype=float)

    def validate(self) -> None:
        k, q, nm = self.kind, self.params, self.name
        if k == CAD_KIND:
            _validate_step_params(nm, q)
        elif k not in PRIMITIVES_3D:
            raise ValueError(f"{nm}: неизвестный примитив {k!r}; допустимо: {KINDS_3D}.")

        def positive(*keys):
            for key in keys:
                if key not in q:
                    raise ValueError(f"{nm}: не задан размер {key!r}.")
                val = float(q[key])
                if not (math.isfinite(val) and val > 0.0):
                    raise ValueError(f"{nm}: {key} должен быть конечным и больше нуля.")

        if k == "box":
            positive("lx", "ly", "lz")
        elif k == "cylinder":
            positive("r", "h")
        elif k in ("tube", "tube_sector"):
            positive("r_in", "r_out", "h")
            if not float(q["r_in"]) < float(q["r_out"]):
                raise ValueError(f"{nm}: нужно r_in < r_out.")
            if k == "tube_sector":
                for key in ("a1", "a2"):
                    if key not in q or not math.isfinite(float(q[key])):
                        raise ValueError(f"{nm}: угол {key} должен быть задан конечным числом.")
                if sector_span(q["a1"], q["a2"]) == 0.0:
                    raise ValueError(f"{nm}: нулевой угол сектора (для полного оборота — tube).")
        elif k == "sphere":
            positive("r")
        elif k == "prism":
            positive("h")
            pts = np.asarray(q.get("points", ()), dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
                raise ValueError(f"{nm}: многоугольник требует не меньше трёх точек (x, y).")
            if not np.isfinite(pts).all():
                raise ValueError(f"{nm}: координаты многоугольника должны быть конечными.")
            if polygon_area(pts) == 0.0:
                raise ValueError(f"{nm}: многоугольник вырожден (нулевая площадь).")
            if polygon_self_intersects(pts):
                raise ValueError(f"{nm}: многоугольник самопересекается.")

        c = np.asarray(self.center, dtype=float)
        r = np.asarray(self.rotation, dtype=float)
        if c.shape != (3,) or not np.isfinite(c).all():
            raise ValueError(f"{nm}: center — три конечных числа.")
        if r.shape != (3,) or not np.isfinite(r).all():
            raise ValueError(f"{nm}: rotation — три конечных угла.")
        d = self.magnet_dir
        if isinstance(d, str):
            if d not in MAGNET_DIRS_3D:
                raise ValueError(f"{nm}: magnet_dir {d!r}; допустимо: {MAGNET_DIRS_3D} или вектор.")
        elif d is not None:
            v = np.asarray(d, dtype=float)
            if v.shape != (3,) or not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0.0:
                raise ValueError(f"{nm}: вектор magnet_dir — три конечных числа, не все нули.")
        mr = np.asarray(self.magnet_rotation, dtype=float)
        if mr.shape != (3,) or not np.isfinite(mr).all():
            raise ValueError(f"{nm}: magnet_rotation — три конечных угла.")
        if self.mesh_size is not None and not (float(self.mesh_size) > 0.0):
            raise ValueError(f"{nm}: mesh_size должен быть больше нуля.")
        if not (int(self.priority) >= 1):
            raise ValueError(f"{nm}: priority должен быть не меньше 1.")


# ----------------------------------------------------------------- тела из CAD (STEP, этап 3D-1б)
@dataclass(frozen=True)
class StepBody:
    """Тело STEP-файла: номер (с нуля, в порядке файла), метка, объём [м³], центр масс и габарит [м] в координатах CAD."""

    index: int
    name: str
    volume: float
    centroid: tuple
    bbox_min: tuple
    bbox_max: tuple
    n_faces: int


_STEP_CACHE: dict[tuple, tuple] = {}


def _step_file_key(path) -> tuple:
    """Ключ кэша: абсолютный путь, размер и время изменения файла (изменённый файл читается заново)."""
    p = os.path.abspath(os.fspath(path))
    if not os.path.isfile(p):
        raise ValueError(f"файл STEP не найден: {p}.")
    st = os.stat(p)
    return p, int(st.st_size), int(st.st_mtime_ns)


def _import_step_solids(occ, path: str) -> list[int]:
    """
    Прочитать STEP в текущую модель gmsh: единицы, объявленные в файле, OpenCASCADE переводит в метры.
    Возвращает теги тел файла в порядке файла. Файл без тел — ошибка; поверхности вне тел — тоже
    ошибка: в склейке они стали бы лишними гранями сетки, а отбрасывать геометрию молча нельзя.
    """
    import gmsh

    before = {d: {int(t) for _, t in occ.getEntities(d)} for d in (2, 3)}
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    occ.importShapes(path, highestDimOnly=False)
    solids = sorted(int(t) for _, t in occ.getEntities(3) if int(t) not in before[3])
    faces = {int(t) for _, t in occ.getEntities(2) if int(t) not in before[2]}
    if not solids:
        what = f"только поверхности ({len(faces)})" if faces else "нет и поверхностей"
        raise ValueError(f"в файле STEP нет тел — {what}: {path}.")
    bound: set[int] = set()
    for v in solids:
        for loop in occ.getSurfaceLoops(v)[1]:
            bound.update(int(s) for s in np.asarray(loop).ravel())
    free = faces - bound
    if free:
        raise ValueError(f"в файле STEP {len(free)} поверхностей вне тел — удалите их в CAD или выгрузите "
                         f"только тела: {path}.")
    return solids


def step_bodies(path) -> tuple:
    """
    Тела STEP-файла (этап 3D-1б): номер, метка, объём, центр масс, габарит, число граней — в метрах, в
    координатах CAD (`StepBody`). Файл читает gmsh/OpenCASCADE; результат кэшируется по пути, размеру и
    времени изменения файла. ⚠ gmsh требует главный поток; при открытой сессии gmsh вызывать нельзя —
    построители берут сведения о телах заранее, до своей сессии.
    """
    import gmsh

    key = _step_file_key(path)
    if key in _STEP_CACHE:
        return _STEP_CACHE[key]
    if gmsh.isInitialized():
        raise RuntimeError("сведения о телах STEP читаются вне открытой сессии gmsh.")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        solids = _import_step_solids(occ, key[0])
        occ.synchronize()
        bodies = []
        for i, tag in enumerate(solids):
            bb = occ.getBoundingBox(3, tag)
            bodies.append(StepBody(
                index=i, name=gmsh.model.getEntityName(3, tag), volume=float(occ.getMass(3, tag)),
                centroid=tuple(float(c) for c in occ.getCenterOfMass(3, tag)),
                bbox_min=tuple(float(v) for v in bb[:3]), bbox_max=tuple(float(v) for v in bb[3:]),
                n_faces=len(gmsh.model.getBoundary([(3, tag)], combined=False, oriented=False))))
    finally:
        gmsh.finalize()
    _STEP_CACHE[key] = tuple(bodies)
    return _STEP_CACHE[key]


def step_body(obj: GeoObject3D) -> StepBody:
    """Сведения о теле CAD объекта (`step_bodies` по его файлу и номеру)."""
    return step_bodies(obj.params["path"])[int(obj.params["body"])]


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / float(np.linalg.norm(a))


def _validate_step_params(nm: str, q: dict) -> None:
    path = q.get("path")
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"{nm}: не задан путь к файлу STEP.")
    bodies = step_bodies(path)
    i = q.get("body")
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < len(bodies):
        raise ValueError(f"{nm}: номер тела {i!r} вне 0…{len(bodies) - 1} (тел в файле {len(bodies)}).")
    b = bodies[int(i)]
    ext = float(np.max(np.subtract(b.bbox_max, b.bbox_min)))
    if "volume" in q:
        try:
            v = float(q["volume"])
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and v > 0.0) or abs(b.volume / v - 1.0) > _FINGERPRINT_TOL:
            raise ValueError(f"{nm}: у тела {int(i)} объём {b.volume:.9e} м³, запомнен {q['volume']!r} — "
                             f"файл STEP изменился.")
    if "centroid" in q:
        c = np.asarray(q["centroid"], dtype=float)
        if (c.shape != (3,) or not np.isfinite(c).all()
                or float(np.linalg.norm(c - np.asarray(b.centroid))) > _FINGERPRINT_TOL * ext):
            raise ValueError(f"{nm}: у тела {int(i)} центр масс не совпал с запомненным — файл STEP изменился.")
    o = np.asarray(q.get("axis_origin", (0.0, 0.0, 0.0)), dtype=float)
    if o.shape != (3,) or not np.isfinite(o).all():
        raise ValueError(f"{nm}: axis_origin — три конечных числа [м].")
    a = np.asarray(q.get("axis_dir", (0.0, 0.0, 1.0)), dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or float(np.linalg.norm(a)) == 0.0:
        raise ValueError(f"{nm}: axis_dir — три конечных числа, не все нули.")


# ----------------------------------------------------------------- знаковое расстояние
def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Расстояние от точек p (N,2) до отрезка ab."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _annular_sector_sd(p2, r_in, r_out, a1, span) -> np.ndarray:
    """Точное знаковое расстояние до кольцевого сектора на плоскости."""
    rho = np.hypot(p2[:, 0], p2[:, 1])
    t = np.mod(np.arctan2(p2[:, 1], p2[:, 0]) - a1, _TWO_PI)
    in_angle = t <= span
    e1 = np.array([math.cos(a1), math.sin(a1)])
    e2 = np.array([math.cos(a1 + span), math.sin(a1 + span)])
    d_edges = np.minimum(_segment_distance(p2, r_in * e1, r_out * e1),
                         _segment_distance(p2, r_in * e2, r_out * e2))
    band = np.maximum(r_in - rho, rho - r_out)                # ≤ 0 внутри кольца
    outside = np.where(in_angle, np.maximum(band, 0.0), d_edges)
    inside = in_angle & (band <= 0.0)
    return np.where(inside, -np.minimum(-band, d_edges), outside)


def _polygon_sd(p2, points) -> np.ndarray:
    """Точное знаковое расстояние до простого многоугольника на плоскости."""
    a = np.asarray(points, dtype=float)
    b = np.roll(a, -1, axis=0)
    ab = b - a                                                # (E,2)
    ap = p2[:, None, :] - a[None, :, :]                      # (N,E,2)
    t = np.clip((ap * ab[None]).sum(axis=2) / (ab * ab).sum(axis=1)[None], 0.0, 1.0)
    dist = np.linalg.norm(ap - t[..., None] * ab[None], axis=2).min(axis=1)
    x, y = p2[:, 0:1], p2[:, 1:2]
    xi, yi, xj, yj = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
    crosses = (yi > y) != (yj > y)
    dy = np.where(yj == yi, 1.0, yj - yi)
    x_cross = xi + (xj - xi) * (y - yi) / dy
    inside = (crosses & (x < x_cross)).sum(axis=1) % 2 == 1
    return np.where(inside, -dist, dist)


def _extrusion_sd(d2: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Знаковое расстояние до тела «плоская фигура × отрезок» по расстояниям до фигуры и до торцов."""
    w = np.stack([d2, dz], axis=1)
    return np.minimum(w.max(axis=1), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=1)


def local_signed_distance(kind: str, params: dict, p: np.ndarray) -> np.ndarray:
    """Знаковое расстояние в системе объекта: точки p (N,3) → (N,), < 0 внутри [м]."""
    if kind == CAD_KIND:
        raise ValueError("у тела CAD знакового расстояния формулой нет — его считает gmsh по граням тела.")
    q = params
    if kind == "box":
        d = np.abs(p) - 0.5 * np.array([q["lx"], q["ly"], q["lz"]], dtype=float)
        return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    if kind == "sphere":
        return np.linalg.norm(p, axis=1) - float(q["r"])
    rho = np.hypot(p[:, 0], p[:, 1])
    dz = np.abs(p[:, 2]) - 0.5 * float(q["h"])
    if kind == "cylinder":
        d2 = rho - float(q["r"])
    elif kind == "tube":
        d2 = np.maximum(float(q["r_in"]) - rho, rho - float(q["r_out"]))
    elif kind == "tube_sector":
        d2 = _annular_sector_sd(p[:, :2], float(q["r_in"]), float(q["r_out"]), float(q["a1"]),
                                sector_span(q["a1"], q["a2"]))
    elif kind == "prism":
        d2 = _polygon_sd(p[:, :2], q["points"])
    else:
        raise ValueError(f"неизвестный примитив {kind!r}.")
    return _extrusion_sd(d2, dz)


def signed_distance(obj: GeoObject3D, points) -> np.ndarray:
    """Знаковое расстояние от глобальных точек (N,3) до объекта: < 0 внутри, > 0 снаружи [м]."""
    return local_signed_distance(obj.kind, obj.params, obj.to_local(points))


def contains3d(obj: GeoObject3D, points, tol: float = 1.0e-12) -> np.ndarray:
    """Лежат ли глобальные точки (N,3) внутри объекта (граница — с допуском tol [м])."""
    return signed_distance(obj, points) <= tol


# ----------------------------------------------------------------- объём, габарит, домен
def object_volume(obj: GeoObject3D) -> float:
    """Объём объекта [м³]: у примитива — формула, у тела CAD — OpenCASCADE."""
    k, q = obj.kind, obj.params
    if k == CAD_KIND:
        return step_body(obj).volume
    if k == "box":
        return float(q["lx"]) * float(q["ly"]) * float(q["lz"])
    if k == "cylinder":
        return math.pi * float(q["r"]) ** 2 * float(q["h"])
    if k == "tube":
        return math.pi * (float(q["r_out"]) ** 2 - float(q["r_in"]) ** 2) * float(q["h"])
    if k == "tube_sector":
        return (0.5 * sector_span(q["a1"], q["a2"])
                * (float(q["r_out"]) ** 2 - float(q["r_in"]) ** 2) * float(q["h"]))
    if k == "sphere":
        return 4.0 / 3.0 * math.pi * float(q["r"]) ** 3
    if k == "prism":
        return abs(polygon_area(q["points"])) * float(q["h"])
    raise ValueError(f"неизвестный примитив {k!r}.")


def _local_bbox(obj: GeoObject3D) -> tuple[np.ndarray, np.ndarray]:
    k, q = obj.kind, obj.params
    if k == CAD_KIND:                                     # в координатах CAD, от OpenCASCADE
        b = step_body(obj)
        return np.asarray(b.bbox_min, dtype=float), np.asarray(b.bbox_max, dtype=float)
    if k == "box":
        h = 0.5 * np.array([q["lx"], q["ly"], q["lz"]], dtype=float)
        return -h, h
    if k == "sphere":
        r = float(q["r"])
        return -np.full(3, r), np.full(3, r)
    hz = 0.5 * float(q["h"])
    if k == "prism":
        p = np.asarray(q["points"], dtype=float)
        return (np.array([p[:, 0].min(), p[:, 1].min(), -hz]),
                np.array([p[:, 0].max(), p[:, 1].max(), hz]))
    r = float(q["r"] if k == "cylinder" else q["r_out"])
    return np.array([-r, -r, -hz]), np.array([r, r, hz])


def object_bbox(obj: GeoObject3D) -> tuple[np.ndarray, np.ndarray]:
    """Габарит объекта по глобальным осям (у повёрнутых кривых тел — с запасом): (min, max) [м]."""
    lo, hi = _local_bbox(obj)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    g = obj.to_global(corners)
    return g.min(axis=0), g.max(axis=0)


def auto_domain3d(objects, *, material, margin_frac: float = 2.0,
                  mesh_size: float | None = None) -> GeoObject3D:
    """
    Фон-домен — параллелепипед по общему габариту объектов плюс запас margin_frac·(наибольший
    габарит) СО ВСЕХ сторон одинаково: поле плоского магнита уходит во все стороны на масштаб
    его наибольшего размера. Граница домена — внешнее граничное условие, область обрезается.
    Запас по умолчанию 2 — по измерению на намагниченном шаре (этап 3D-2): разрыв между
    границами «поток не выходит» и «φ = 0» в поле магнита 6,7 % при запасе 1, 1,45 % при 2,
    0,25 % при 4 (∝ L⁻³ — дипольное поле); ошибка одной границы — около половины разрыва, при
    запасе 2 около 0,7 %, меньше ошибки сетки при обычном размере элемента.
    """
    objs = list(objects)
    if not objs:
        raise ValueError("нужен хотя бы один объект.")
    if not (margin_frac > 0.0):
        raise ValueError("margin_frac должен быть больше нуля.")
    boxes = [object_bbox(o) for o in objs]
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    ext = hi - lo
    pad = float(margin_frac) * float(ext.max())
    size = ext + 2.0 * pad
    if mesh_size is None:
        mesh_size = float(size.max()) / 10.0
    return GeoObject3D(
        "domain", "box", {"lx": float(size[0]), "ly": float(size[1]), "lz": float(size[2])},
        material, center=tuple(float(v) for v in 0.5 * (lo + hi)), mesh_size=float(mesh_size))


def object_axis(obj: GeoObject3D) -> tuple[np.ndarray, np.ndarray]:
    """
    Ось тела — прямая origin + t·direction в глобальных координатах (origin [м], direction единичный).
    От неё считаются «осевое» и «радиальное» намагничивание (`magnet_axis_at`), её же рисует вид
    (осевые линии, этап 3D-7) — одно правило в одном месте. У примитива — локальная ось z через центр,
    у тела CAD — axis_origin + t·axis_dir в координатах CAD, размещённая вместе с телом (по умолчанию —
    ось z через начало координат CAD).
    """
    R = obj.rotation_matrix()
    if obj.kind == CAD_KIND:
        a = R @ _unit(obj.params.get("axis_dir", (0.0, 0.0, 1.0)))
        o = obj.to_global(np.asarray(obj.params.get("axis_origin", (0.0, 0.0, 0.0)), dtype=float))[0]
        return o, a
    return np.asarray(obj.center, dtype=float), R[:, 2].copy()


def magnet_axis_at(obj: GeoObject3D, points) -> np.ndarray:
    """
    Единичная ось намагничивания объекта в глобальных точках (N,3): 'axial' — вдоль оси тела
    (`object_axis`, по умолчанию), 'axial-in' — против неё, 'radial' / 'radial-in' — от оси тела (у
    шара-примитива — от центра) наружу / внутрь, вектор — заданное глобальное направление. Затем —
    поворот `magnet_rotation` вокруг глобальных осей X → Y → Z (та же `rotation_matrix`, что у тела):
    «по +X» и 45° вокруг Z — под 45° в плоскости XY; «радиально» и 10° вокруг оси мотора — радиально
    со скосом 10° (каждая радиальная ось повёрнута на 10°).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    n = pts.shape[0]
    d = obj.magnet_dir
    if d is None or isinstance(d, str):
        o, a = object_axis(obj)
        if d is None or d == "axial":
            base = np.tile(a, (n, 1))
        elif d == "axial-in":
            base = np.tile(-a, (n, 1))
        else:
            v = pts - o
            if obj.kind != "sphere":
                v = v - (v @ a)[:, None] * a
            nrm = np.linalg.norm(v, axis=1)
            unit = np.divide(v, nrm[:, None], out=np.zeros_like(v), where=nrm[:, None] > 0.0)
            base = -unit if d == "radial-in" else unit
    else:
        v = np.asarray(d, dtype=float)
        base = np.tile(v / np.linalg.norm(v), (n, 1))
    if not any(float(r) != 0.0 for r in obj.magnet_rotation):
        return base
    return base @ rotation_matrix(obj.magnet_rotation).T


# ----------------------------------------------------------------- gmsh
def _polygon_face(occ, points, z: float) -> int:
    tags = [occ.addPoint(float(x), float(y), z) for x, y in points]
    n = len(tags)
    lines = [occ.addLine(tags[i], tags[(i + 1) % n]) for i in range(n)]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _sector_face(occ, r_in: float, r_out: float, a1: float, span: float, z: float) -> int:
    """Плоский кольцевой сектор на высоте z; дуги делятся на части ≤ π/2 (дуга gmsh строго < π)."""
    n_arc = max(1, math.ceil(span / (0.5 * math.pi)))
    angs = [a1 + span * i / n_arc for i in range(n_arc + 1)]
    o = occ.addPoint(0.0, 0.0, z)
    outer = [occ.addPoint(r_out * math.cos(a), r_out * math.sin(a), z) for a in angs]
    inner = [occ.addPoint(r_in * math.cos(a), r_in * math.sin(a), z) for a in angs]
    curves = [occ.addLine(inner[0], outer[0])]
    curves += [occ.addCircleArc(outer[i], o, outer[i + 1]) for i in range(n_arc)]
    curves.append(occ.addLine(outer[-1], inner[-1]))
    curves += [occ.addCircleArc(inner[i + 1], o, inner[i]) for i in reversed(range(n_arc))]
    return occ.addPlaneSurface([occ.addCurveLoop(curves)])


def _single_volume(dim_tags) -> int:
    vols = [t for d, t in dim_tags if d == 3]
    if len(vols) != 1:
        raise RuntimeError(f"построение тела дало {len(vols)} объёмов вместо одного.")
    return int(vols[0])


def _add_volume(occ, obj: GeoObject3D) -> int:
    """Построить тело объекта в своей системе, повернуть (X → Y → Z) и перенести в центр."""
    k, q = obj.kind, obj.params
    if k == "box":
        lx, ly, lz = (float(q[key]) for key in ("lx", "ly", "lz"))
        v = occ.addBox(-lx / 2, -ly / 2, -lz / 2, lx, ly, lz)
    elif k == "sphere":
        v = occ.addSphere(0.0, 0.0, 0.0, float(q["r"]))
    else:
        h = float(q["h"])
        if k == "cylinder":
            v = occ.addCylinder(0.0, 0.0, -h / 2, 0.0, 0.0, h, float(q["r"]))
        elif k == "tube":
            outer = occ.addCylinder(0.0, 0.0, -h / 2, 0.0, 0.0, h, float(q["r_out"]))
            inner = occ.addCylinder(0.0, 0.0, -h / 2, 0.0, 0.0, h, float(q["r_in"]))
            out, _ = occ.cut([(3, outer)], [(3, inner)])
            v = _single_volume(out)
        elif k == "tube_sector":
            s = _sector_face(occ, float(q["r_in"]), float(q["r_out"]), float(q["a1"]),
                             sector_span(q["a1"], q["a2"]), -h / 2)
            v = _single_volume(occ.extrude([(2, s)], 0.0, 0.0, h))
        elif k == "prism":
            s = _polygon_face(occ, q["points"], -h / 2)
            v = _single_volume(occ.extrude([(2, s)], 0.0, 0.0, h))
        else:
            raise ValueError(f"неизвестный примитив {k!r}.")
    _place(occ, v, obj)
    return v


def _place(occ, v: int, obj: GeoObject3D) -> None:
    """Повернуть тело (X → Y → Z вокруг начала координат) и перенести в center."""
    dim_tags = [(3, v)]
    for axis, angle in zip(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), obj.rotation):
        if float(angle) != 0.0:
            occ.rotate(dim_tags, 0.0, 0.0, 0.0, axis[0], axis[1], axis[2], float(angle))
    cx, cy, cz = (float(c) for c in obj.center)
    if cx or cy or cz:
        occ.translate(dim_tags, cx, cy, cz)


def _add_volumes(occ, objs) -> list[int]:
    """
    Тела всех объектов по порядку. Примитивы — `_add_volume`; STEP-файл читается один раз на все
    выбранные из него тела, невыбранные тела удаляются, повторно выбранное тело копируется, каждое тело
    размещается своими center/rotation.
    """
    vols = [0] * len(objs)
    by_file: dict[str, list[int]] = {}
    for i, o in enumerate(objs):
        if o.kind == CAD_KIND:
            by_file.setdefault(_step_file_key(o.params["path"])[0], []).append(i)
        else:
            vols[i] = _add_volume(occ, o)
    for path, idxs in by_file.items():
        solids = _import_step_solids(occ, path)
        taken: set[int] = set()
        for i in idxs:
            b = int(objs[i].params["body"])
            if b in taken:
                vols[i] = _single_volume(occ.copy([(3, solids[b])]))
            else:
                taken.add(b)
                vols[i] = solids[b]
        unused = [(3, t) for j, t in enumerate(solids) if j not in taken]
        if unused:
            occ.remove(unused, recursive=True)
        for i in idxs:
            if not occ.getSurfaceLoops(vols[i])[1]:
                raise RuntimeError(f"{objs[i].name}: после удаления лишних тел файла у тела не осталось граней.")
            _place(occ, vols[i], objs[i])
    return vols


def _cad_size_fields(gmsh, cad_idx, out_map, sizes, extents, grading) -> list[int]:
    """
    Размер сетки у тел CAD — тем же правилом, что у примитивов: ровно h внутри тела, на его гранях и
    рёбрах (поле Constant), снаружи h·(1 + g·d/L) по расстоянию d до граней тела, которое gmsh считает
    сам (поле Distance, точность — см. DISTANCE_SAMPLING). Возвращает поля; их минимум с полями особенностей
    формы (`_feature_fields`) — фоновое поле, а функция размера примитивов берёт минимум с ним (значение
    фонового поля приходит в её аргумент lc — проверено).
    """
    field = gmsh.model.mesh.field
    parts: list[int] = []
    for i in cad_idx:
        pieces = sorted({int(t) for d, t in out_map[i] if d == 3})
        faces = sorted({abs(int(t)) for _, t in gmsh.model.getBoundary([(3, t) for t in pieces],
                                                                     combined=True, oriented=False)})
        curves = sorted({abs(int(t)) for _, t in gmsh.model.getBoundary([(2, s) for s in faces],
                                                                      combined=False, oriented=False)})
        f_dist = field.add("Distance")
        field.setNumbers(f_dist, "SurfacesList", faces)
        field.setNumber(f_dist, "Sampling", DISTANCE_SAMPLING)
        f_grow = field.add("MathEval")
        field.setString(f_grow, "F", f"{sizes[i]:.17g}*(1+{float(grading):.17g}*F{f_dist}/{extents[i]:.17g})")
        f_in = field.add("Constant")
        field.setNumber(f_in, "VIn", sizes[i])
        field.setNumbers(f_in, "VolumesList", pieces)
        field.setNumbers(f_in, "SurfacesList", faces)
        field.setNumbers(f_in, "CurvesList", curves)
        parts += [f_grow, f_in]
    return parts


def _feature_sizes(gmsh, piece_region: dict, sizes, curvature_elements: int, *, thin_factor: float,
                   thin_floor: float, curvature_floor: float) -> dict[int, float]:
    """
    Размер ячейки на гранях с мелкими особенностями формы (этап 3D-8, сетка по CAD) — {грань: размер}.
    Точки граней берутся из триангуляции ядра CAD (как у предпросмотра; сама сетка потом стирается).
      · Кривизна: 2πR/`curvature_elements` по наименьшему радиусу грани, но не мельче curvature_floor·h —
        технологическое скругление в доли h не должно диктовать сетку всему телу.
      · Узкие места: толщина t у точки грани — расстояние до ближайшей точки ДРУГОЙ, не смежной грани,
        лежащей НАПРОТИВ (направление на неё отклоняется от нормалей обеих граней меньше чем на 45°); это
        и зазор между телами, и тонкая стенка, и щель паза. У грани берётся 10-й процентиль по её точкам
        (узким считается место, занимающее хотя бы десятую часть грани); размер — thin_factor·t, но не
        мельче thin_floor·h: у клина, сходящегося в касание, t → 0.
    h — размер сетки тела, к которому грань прилегает (меньший из соседних); грани только фона не трогаются.
    """
    from scipy.spatial import cKDTree

    faces = [int(t) for _, t in gmsh.model.getEntities(2)]
    h_face: dict[int, float] = {}
    curves_of: dict[int, set[int]] = {}
    for f in faces:
        up, down = gmsh.model.getAdjacencies(2, f)
        regs = [int(piece_region.get(int(p), 0)) for p in up]
        if not regs or all(r == 0 for r in regs):
            continue                                            # внешняя граница фона — не особенность
        h_face[f] = min(float(sizes[r]) for r in regs)
        curves_of[f] = {abs(int(c)) for c in down}
    if not h_face:
        return {}
    gmsh.option.setNumber("Mesh.StlLinearDeflectionRelative", 1)
    gmsh.option.setNumber("Mesh.StlLinearDeflection", PREVIEW_LINEAR_DEFLECTION)
    gmsh.option.setNumber("Mesh.StlAngularDeflection", _TWO_PI / 24.0)
    gmsh.model.mesh.importStl()
    pts, nrm, own, kappa = [], [], [], {}
    try:
        for f in h_face:
            _, xyz, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True, returnParametricCoord=False)
            xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
            if xyz.shape[0] == 0:
                lo, hi = gmsh.model.getParametrizationBounds(2, f)
                xyz = np.asarray(gmsh.model.getValue(2, f, [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]),
                                 dtype=float).reshape(-1, 3)
            uv = np.asarray(gmsh.model.getParametrization(2, f, xyz.ravel().tolist()), dtype=float)
            n = np.asarray(gmsh.model.getNormal(f, uv.tolist()), dtype=float).reshape(-1, 3)
            if gmsh.model.getType(2, f) != "Plane":
                kmax, kmin, _, _ = gmsh.model.getPrincipalCurvatures(f, uv.tolist())
                kappa[f] = float(np.max(np.abs(np.concatenate([np.ravel(kmax), np.ravel(kmin)]))))
            pts.append(xyz)
            nrm.append(n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None])
            own.append(np.full(xyz.shape[0], f, dtype=np.int64))
    finally:
        gmsh.model.mesh.clear()
    P, N, O = np.concatenate(pts), np.concatenate(nrm), np.concatenate(own)
    reach = max(h_face.values()) / thin_factor                  # толщина больше h/thin_factor не мельчит
    pairs = cKDTree(P).query_pairs(r=reach, output_type="ndarray")
    t_pt = np.full(P.shape[0], np.inf)
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        fi, fj = O[i], O[j]
        adj = np.zeros(i.size, dtype=bool)
        by_curve: dict[int, list[int]] = {}
        for f, cs in curves_of.items():
            for c in cs:
                by_curve.setdefault(c, []).append(f)
        neighbours = {(a, b) for fs in by_curve.values() for a in fs for b in fs}
        if neighbours:
            key = fi * (max(faces) + 1) + fj
            nb = np.array([a * (max(faces) + 1) + b for a, b in neighbours], dtype=np.int64)
            adj = np.isin(key, nb)
        d = P[j] - P[i]
        dist = np.linalg.norm(d, axis=1)
        ok = (fi != fj) & ~adj & (dist > 0.0)
        cos_i = np.abs(np.einsum("ij,ij->i", d, N[i])) / np.maximum(dist, 1e-300)
        cos_j = np.abs(np.einsum("ij,ij->i", d, N[j])) / np.maximum(dist, 1e-300)
        cos_n = np.abs(np.einsum("ij,ij->i", N[i], N[j]))           # грани почти параллельны: у зазора и стенки
        ok &= (cos_i >= _ACROSS_COS) & (cos_j >= _ACROSS_COS) & (cos_n >= _ACROSS_COS)   # да, у ребра — нет
        h_arr = np.zeros(max(faces) + 1)
        h_arr[list(h_face)] = list(h_face.values())
        ok &= dist < np.minimum(h_arr[fi], h_arr[fj]) / thin_factor
        np.minimum.at(t_pt, i[ok], dist[ok])
        np.minimum.at(t_pt, j[ok], dist[ok])
    out: dict[int, float] = {}
    for f, h in h_face.items():
        s = h
        if kappa.get(f, 0.0) > 0.0 and int(curvature_elements) > 0:
            s = min(s, max(_TWO_PI / (kappa[f] * int(curvature_elements)), curvature_floor * h))
        t = t_pt[O == f]
        t10 = float(np.percentile(t, 10, method="lower")) if t.size else np.inf     # без интерполяции через ∞
        if np.isfinite(t10):
            s = min(s, max(thin_factor * t10, thin_floor * h))
        if s < 0.95 * h:
            out[f] = s
    return out


def _feature_fields(gmsh, feature: dict[int, float], growth: float) -> list[int]:
    """
    Поля размера у граней с особенностями формы: на самой грани — её размер s (Constant), от неё размер
    растёт линейно s + (growth − 1)·d — соседние ячейки отличаются не больше чем в growth раз. Грани с
    близкими размерами (в пределах множителя 1,25) — в одном поле, чтобы полей было немного.
    """
    field = gmsh.model.mesh.field
    groups: dict[int, list[int]] = {}
    for f, s in feature.items():
        groups.setdefault(int(np.floor(np.log(s) / np.log(1.25))), []).append(int(f))
    parts: list[int] = []
    for k, fs in sorted(groups.items()):
        s = min(feature[f] for f in fs)
        c = field.add("Constant")
        field.setNumber(c, "VIn", s)
        field.setNumbers(c, "SurfacesList", fs)
        d = field.add("Distance")
        field.setNumbers(d, "SurfacesList", fs)
        field.setNumber(d, "Sampling", DISTANCE_SAMPLING)
        g = field.add("MathEval")
        field.setString(g, "F", f"{s:.17g}+{float(growth) - 1.0:.17g}*F{d}")
        parts += [c, g]
    return parts


def _extract_tets(gmsh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gmsh → (вершины, ячейки положительной ориентации, тег куска объёма на ячейку); без висячих узлов."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag2idx = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    tag2idx[node_tags] = np.arange(node_tags.size)
    cells_l, piece_l = [], []
    for _, tag in gmsh.model.getEntities(3):
        etypes, _, enodes = gmsh.model.mesh.getElements(3, tag)
        for et, en in zip(etypes, enodes):
            if int(et) != 4:
                raise RuntimeError(f"неожиданный тип объёмного элемента gmsh: {et}.")
            t = tag2idx[np.asarray(en, dtype=np.int64).reshape(-1, 4)]
            cells_l.append(t)
            piece_l.append(np.full(t.shape[0], int(tag), dtype=np.int64))
    if not cells_l:
        raise RuntimeError("gmsh не построил тетраэдров.")
    cells = np.concatenate(cells_l)
    piece = np.concatenate(piece_l)
    if np.any(cells < 0):
        raise RuntimeError("в ячейках есть узлы, которых нет в списке узлов gmsh.")
    used = np.unique(cells)
    remap = np.full(coords.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    verts = np.ascontiguousarray(coords[used])
    return verts, orient_cells(verts, remap[cells]), piece


def build_object_problem3d(objects, domain: GeoObject3D, *, default_mesh_size: float,
                           grading: float = 2.0, curvature_elements: int = 16,
                           T: float = 20.0, feature_sizing: bool = True, thin_factor: float = THIN_FACTOR,
                           size_growth: float = SIZE_GROWTH) -> Problem3D:
    """
    Собрать Problem3D из фон-домена и объектов. gmsh: тела → склейка (fragment) → конформная
    тетраэдральная сетка; регион ячейки — самый приоритетный объект, в состав которого входит её
    кусок объёма (карта фрагментов gmsh), иначе домен. Размер сетки — свой у объекта h_i, снаружи
    растёт как h_i·(1 + grading·d/L_i) (L_i — наибольший габарит объекта: при grading = 2 размер
    удваивается на удалении в полгабарита), не крупнее размера домена. Тела из STEP (kind='step')
    склеиваются вместе с примитивами; размер сетки у них — то же правило по расстоянию, которое считает
    gmsh (`_cad_size_fields`).
    Особенности формы (`feature_sizing`, по умолчанию включено; этап 3D-8): кривые грани — не меньше
    `curvature_elements` элементов на оборот, но не мельче CURVATURE_FLOOR·h; узкие места (зазоры, тонкие
    стенки) — размер `thin_factor`·толщина, но не мельче THIN_FLOOR·h; от таких граней размер растёт
    плавно, соседние ячейки отличаются не больше чем в `size_growth` раз (`_feature_sizes`,
    `_feature_fields`). Выключено — прежнее правило: кривизну учитывает сам gmsh, без плавного роста.
    ⚠ gmsh требует главный поток.
    """
    import gmsh

    if not (default_mesh_size > 0.0):
        raise ValueError("default_mesh_size должен быть больше нуля.")
    if not (grading >= 0.0):
        raise ValueError("grading не может быть отрицательным.")
    if int(curvature_elements) < 0:
        raise ValueError("curvature_elements не может быть отрицательным.")
    if not (thin_factor > 0.0):
        raise ValueError("thin_factor должен быть больше нуля.")
    if not (size_growth > 1.0):
        raise ValueError("size_growth должен быть больше 1.")
    domain.validate()
    if domain.kind != "box":
        raise ValueError("домен должен быть параллелепипедом (kind='box').")
    objs = list(objects)
    for o in objs:
        o.validate()
    all_objs = [domain] + objs
    # Порядок наложения: по УБЫВАНИЮ (priority, индекс); домен (0) — всегда запасной фон.
    order = sorted(range(1, len(all_objs)), key=lambda i: (all_objs[i].priority, i), reverse=True)
    sizes = [float(o.mesh_size or default_mesh_size) for o in all_objs]
    frames = [(o.rotation_matrix(), np.asarray(o.center, dtype=float)) for o in all_objs]
    extents = [float(np.max(np.subtract(*_local_bbox(o)[::-1]))) for o in all_objs]
    analytic = [i for i in range(1, len(all_objs)) if all_objs[i].kind != CAD_KIND]
    cad = [i for i in range(1, len(all_objs)) if all_objs[i].kind == CAD_KIND]

    def size_at(x: float, y: float, z: float) -> float:
        p = np.array([[x, y, z]], dtype=float)
        h = sizes[0]
        for i in analytic:
            R, c = frames[i]
            d = float(local_signed_distance(all_objs[i].kind, all_objs[i].params, (p - c) @ R)[0])
            h = min(h, sizes[i] * (1.0 + grading * max(d, 0.0) / extents[i]))
        return h

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        vols = _add_volumes(occ, all_objs)
        if len(vols) > 1:
            _, out_map = occ.fragment([(3, vols[0])], [(3, v) for v in vols[1:]])
        else:
            out_map = [[(3, vols[0])]]
        occ.synchronize()

        owners: dict[int, list[int]] = {}
        for i, pieces in enumerate(out_map):
            for dim, tag in pieces:
                if dim == 3:
                    owners.setdefault(int(tag), []).append(i)
        piece_region: dict[int, int] = {}
        for tag, own in owners.items():
            if 0 not in own:
                names = ", ".join(all_objs[i].name for i in own)
                raise ValueError(f"объект выходит за пределы домена: {names}.")
            piece_region[tag] = next((i for i in order if i in own), 0)
        if {int(t) for _, t in gmsh.model.getEntities(3)} != set(piece_region):
            raise RuntimeError("карта фрагментов gmsh не покрывает все тела модели.")

        fields = _cad_size_fields(gmsh, cad, out_map, sizes, extents, grading)
        if feature_sizing:
            feature = _feature_sizes(gmsh, piece_region, sizes, curvature_elements, thin_factor=thin_factor,
                                     thin_floor=THIN_FLOOR, curvature_floor=CURVATURE_FLOOR)
            fields += _feature_fields(gmsh, feature, size_growth)
        if fields:
            f_min = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, size_at(x, y, z)))
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        # кривизну при включённых особенностях учитывают поля (с нижним пределом и плавным ростом)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0 if feature_sizing else int(curvature_elements))
        gmsh.option.setNumber("Mesh.MeshSizeMax", sizes[0])
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        verts, cells, piece = _extract_tets(gmsh)
    finally:
        gmsh.finalize()

    mesh = TetMesh3D(verts, cells)
    lut = np.full(int(piece.max()) + 1, -1, dtype=np.int64)
    for tag, rid in piece_region.items():
        if tag < lut.size:
            lut[tag] = rid
    cell_region = lut[piece]
    if np.any(cell_region < 0):
        raise RuntimeError("часть ячеек не получила регион.")
    return _problem_on_mesh(all_objs, mesh, cell_region, T)


def object_problem3d_from_mesh(objects, domain: GeoObject3D, vertices, cells, cell_region, *,
                               T: float = 20.0) -> Problem3D:
    """
    Problem3D из тех же объектов и ГОТОВОЙ сетки — например, сохранённой в файле расчёта: регион i —
    объект i (0 — домен), материалы и оси намагничивания ячеек — ровно как в `build_object_problem3d`.
    Сетка проверяется так же, как построенная (форма, индексы, ориентация, без повторов).
    """
    domain.validate()
    if domain.kind != "box":
        raise ValueError("домен должен быть параллелепипедом (kind='box').")
    objs = list(objects)
    for o in objs:
        o.validate()
    all_objs = [domain] + objs
    mesh = TetMesh3D(vertices, cells)
    reg = np.asarray(cell_region)
    if reg.shape != (mesh.n_cells,) or not np.issubdtype(reg.dtype, np.integer):
        raise ValueError("cell_region — целое число на ячейку сетки.")
    if reg.min() < 0 or reg.max() >= len(all_objs):
        raise ValueError(f"cell_region вне номеров объектов 0…{len(all_objs) - 1}.")
    return _problem_on_mesh(all_objs, mesh, reg.astype(np.int64), T)


def _problem_on_mesh(all_objs, mesh: TetMesh3D, cell_region: np.ndarray, T: float) -> Problem3D:
    """Регионы (номер = место объекта в списке, 0 — домен) и оси намагничивания по ячейкам сетки."""
    regions = {i: Region3D(i, o.name, o.material) for i, o in enumerate(all_objs)}
    axis = None
    # Ось нужна ВСЕГДА, если в модели есть магнитный МАТЕРИАЛ (даже если объект перекрыт и не
    # получил ячеек) — как в 2D; вне ячеек магнита ось = 0.
    if any(isinstance(o.material, MagnetMaterial) for o in all_objs):
        axis = np.zeros((mesh.n_cells, 3), dtype=float)
        cen = mesh.cell_centroids()
        for i, o in enumerate(all_objs):
            sel = cell_region == i
            if isinstance(o.material, MagnetMaterial) and sel.any():
                axis[sel] = magnet_axis_at(o, cen[sel])
    return Problem3D(mesh=mesh, cell_region=cell_region, regions=regions, magnet_axis=axis,
                     T=float(T))


@dataclass(frozen=True)
class MagnetArrows:
    """Стрелки намагничивания тела на виде: точки (n,3) [м], единичные направления M (n,3), шаг решётки [м]."""

    points: np.ndarray
    directions: np.ndarray
    spacing: float


def _grid_centers(lo, hi, h: float) -> np.ndarray:
    """Центры клеток решётки с шагом не крупнее h на габарите [lo, hi] (N,3)."""
    n = np.maximum(1, np.ceil((hi - lo) / h - 1e-9)).astype(int)
    axes = [lo[k] + (np.arange(n[k]) + 0.5) * (hi[k] - lo[k]) / n[k] for k in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _evenly(n: int, cap: int) -> np.ndarray:
    """Номера не больше cap элементов из n, равномерно по списку."""
    return np.arange(n) if n <= cap else np.unique(np.round(np.linspace(0, n - 1, cap)).astype(np.int64))


def _inside(gmsh, tag: int, points) -> np.ndarray:
    """Какие точки (N,3) OpenCASCADE признаёт лежащими внутри тела tag."""
    return np.array([gmsh.model.isInside(3, tag, [float(x) for x in p]) > 0 for p in np.atleast_2d(points)],
                    dtype=bool)


def _arrows_in_volume(gmsh, obj: GeoObject3D, tag: int, tris: np.ndarray) -> MagnetArrows:
    """
    Стрелки до сетки: точки внутри тела (центры клеток решётки или, у тонкого тела, точки на глубине V/S от
    граней — см. ARROWS_MIN), направление — `magnet_axis_at`, как у ячеек сетки.
    """
    b = gmsh.model.getBoundingBox(3, tag)
    lo, hi = np.asarray(b[:3], dtype=float), np.asarray(b[3:], dtype=float)
    h = float(np.max(hi - lo)) / ARROWS_ALONG
    pts = np.zeros((0, 3))
    for _ in range(2):                                           # решётка и та же вдвое мельче
        cand = _grid_centers(lo, hi, h)
        pts = cand[_inside(gmsh, tag, cand)]
        if pts.shape[0] >= ARROWS_MIN:
            break
        h *= 0.5
    if pts.shape[0] < ARROWS_MIN and tris.shape[0]:
        e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
        nrm = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(nrm, axis=1)
        ok = area > 0.0
        depth = gmsh.model.occ.getMass(3, tag) / float(area.sum())
        cum = np.cumsum(area[ok])
        k = np.searchsorted(cum, (np.arange(ARROWS_MAX) + 0.5) / ARROWS_MAX * cum[-1])   # равномерно по площади
        c = tris[ok][k].mean(axis=1)
        n = nrm[ok][k] / (2.0 * area[ok][k])[:, None]
        plus, minus = c + depth * n, c - depth * n
        in_p, in_m = _inside(gmsh, tag, plus), _inside(gmsh, tag, minus)
        surf = np.concatenate([plus[in_p & ~in_m], minus[in_m & ~in_p]])
        if surf.shape[0] > pts.shape[0]:
            pts, h = surf, math.sqrt(0.5 * float(area.sum()) / surf.shape[0])
    pts = pts[_evenly(pts.shape[0], ARROWS_MAX)]
    return MagnetArrows(pts, magnet_axis_at(obj, pts) if pts.shape[0] else np.zeros((0, 3)), h)


@dataclass(frozen=True)
class PreviewGeometry:
    """Предпросмотр: треугольники поверхности каждого объекта (t,3,3) [м]; стрелки намагничивания (None у не магнита)."""

    surfaces: list
    arrows: list


def preview_geometry(objects, *, curvature_elements: int = 24, magnet_arrows: bool = False) -> PreviewGeometry:
    """
    Предпросмотр без объёмной сетки и без склейки: каждое тело строится тем же `_add_volumes`, что и
    для расчёта, а поверхности триангулирует OpenCASCADE так же, как CAD для показа (BRepMesh, через
    gmsh.model.mesh.importStl): вершины лежат на поверхностях, отклонение хорды — PREVIEW_LINEAR_DEFLECTION
    в долях размера грани, угловой шаг — 2π/`curvature_elements` (у мелких скруглений точек меньше: там шаг
    задаёт отклонение хорды). Сетчик gmsh для показа не годится: мелкие скругления тел из CAD дробят и
    плоские грани. Сборка БПЛА32: 25 тыс. треугольников за 0,6 с вместо 1,25 млн за 33 с; центры
    треугольников отстоят от граней не больше чем на 0,013 мм (10⁻³ габарита тела), площадь меньше истинной
    на ≤ 0,08 %. Перекрытия видны как есть. `magnet_arrows` — ещё и стрелки намагничивания каждого магнита
    (`_arrows_in_volume`) в том же сеансе gmsh.
    """
    import gmsh

    if int(curvature_elements) < 3:
        raise ValueError("curvature_elements должен быть не меньше 3.")
    objs = list(objects)
    if not objs:
        return PreviewGeometry([], [])
    for o in objs:
        o.validate()
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        vols = _add_volumes(occ, objs)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.StlLinearDeflectionRelative", 1)
        gmsh.option.setNumber("Mesh.StlLinearDeflection", PREVIEW_LINEAR_DEFLECTION)
        gmsh.option.setNumber("Mesh.StlAngularDeflection", 2.0 * math.pi / int(curvature_elements))
        gmsh.model.mesh.importStl()
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        lut = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
        lut[node_tags] = np.arange(node_tags.size)
        surfaces = []
        for v in vols:
            tris = []
            for _, tag in gmsh.model.getBoundary([(3, v)], combined=False, oriented=False):
                etypes, _, enodes = gmsh.model.mesh.getElements(2, abs(int(tag)))
                for et, en in zip(etypes, enodes):
                    if int(et) == 2:                                   # треугольник из трёх узлов
                        tris.append(coords[lut[np.asarray(en, dtype=np.int64).reshape(-1, 3)]])
            surfaces.append(np.concatenate(tris) if tris else np.zeros((0, 3, 3)))
        arrows = [(_arrows_in_volume(gmsh, o, v, t) if magnet_arrows and isinstance(o.material, MagnetMaterial)
                   else None) for o, v, t in zip(objs, vols, surfaces)]
        return PreviewGeometry(surfaces, arrows)
    finally:
        gmsh.finalize()


def preview_surfaces(objects, *, curvature_elements: int = 24) -> list[np.ndarray]:
    """Треугольники поверхности каждого объекта (t,3,3) [м] в порядке `objects` — см. `preview_geometry`."""
    return preview_geometry(objects, curvature_elements=curvature_elements).surfaces
