from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magcore.fem2d.mesh import TriangleMesh, signed_area2
from magcore.fem2d.model.materials import MagnetMaterial
from magcore.fem2d.model.problem import Problem2D, Region2D

# ОБЪЕКТНАЯ ПРОИЗВОЛЬНАЯ ГЕОМЕТРИЯ (носитель универсальности): модель = фон-домен + список
# объектов-примитивов, каждому свой материал/ток/направление намагничивания/размер сетки.
# gmsh склеивает их (fragment → конформная сетка), а РЕГИОН ячейки определяется аналитически
# по «центроид ∈ объект» с приоритетом (позже в списке = сверху) — тот же приём, что в
# генераторе PMSM, но для любого набора фигур. Результат — общий Problem2D (физядро уже
# геометронезависимо: PMSM становится частным ШАБЛОНОМ таких объектов).

PRIMITIVES = ("rect", "circle", "ring", "sector", "polygon")


@dataclass(frozen=True)
class GeoObject:
    """Один геометрический объект: фигура (kind+params, СИ м/рад) + материал + опции."""
    name: str
    kind: str
    params: dict
    material: object                     # Air | LinearMaterial | SteelMaterial | MagnetMaterial
    current_density: float = 0.0         # физ. J_z [А/м²] (0 — не проводник)
    magnet_dir: object = None            # 'radial' | 'radial-in' | (dx,dy) — ось намагничивания
    mesh_size: float | None = None       # свой размер элемента, иначе общий

    def center(self) -> tuple[float, float]:
        p = self.params
        if self.kind in ("circle", "ring", "sector", "rect"):
            return float(p["cx"]), float(p["cy"])
        pts = np.asarray(p["points"], dtype=float)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def validate(self) -> None:
        if self.kind not in PRIMITIVES:
            raise ValueError(f"неизвестный примитив {self.kind!r}; допустимо: {PRIMITIVES}.")
        p = self.params
        if self.kind == "rect" and not (p["w"] > 0 and p["h"] > 0):
            raise ValueError(f"{self.name}: w,h должны быть > 0.")
        if self.kind == "circle" and not (p["r"] > 0):
            raise ValueError(f"{self.name}: r должен быть > 0.")
        if self.kind in ("ring", "sector") and not (0 < p["r_in"] < p["r_out"]):
            raise ValueError(f"{self.name}: нужно 0 < r_in < r_out.")
        if self.kind == "polygon" and len(p["points"]) < 3:
            raise ValueError(f"{self.name}: полигон требует ≥3 точки.")
        if self.mesh_size is not None and not (self.mesh_size > 0):
            raise ValueError(f"{self.name}: mesh_size должен быть > 0.")


def _ang_between(th: float, a1: float, a2: float) -> bool:
    two = 2.0 * math.pi
    span = (a2 - a1) % two
    if span == 0.0:
        span = two
    return ((th - a1) % two) <= span + 1e-12


def contains(obj: GeoObject, x: float, y: float) -> bool:
    """Точка (x,y) внутри объекта? (аналитически, по kind)."""
    k, p = obj.kind, obj.params
    if k == "rect":
        ang = p.get("angle", 0.0)
        dx, dy = x - p["cx"], y - p["cy"]
        c, s = math.cos(-ang), math.sin(-ang)
        lx, ly = dx * c - dy * s, dx * s + dy * c
        return abs(lx) <= p["w"] / 2 + 1e-12 and abs(ly) <= p["h"] / 2 + 1e-12
    if k == "circle":
        return (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2 <= p["r"] ** 2 + 1e-15
    if k == "ring":
        d2 = (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2
        return p["r_in"] ** 2 - 1e-15 <= d2 <= p["r_out"] ** 2 + 1e-15
    if k == "sector":
        d2 = (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2
        if not (p["r_in"] ** 2 - 1e-15 <= d2 <= p["r_out"] ** 2 + 1e-15):
            return False
        return _ang_between(math.atan2(y - p["cy"], x - p["cx"]), p["a1"], p["a2"])
    if k == "polygon":
        pts = p["points"]
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside
    raise ValueError(f"неизвестный примитив {k!r}.")


def _magnet_axis(obj: GeoObject, x: float, y: float) -> tuple[float, float]:
    d = obj.magnet_dir
    if d is None or d == "radial" or d == "radial-in":
        ox, oy = obj.center()
        vx, vy = x - ox, y - oy
        n = math.hypot(vx, vy) or 1.0
        s = -1.0 if d == "radial-in" else 1.0
        return s * vx / n, s * vy / n
    dx, dy = float(d[0]), float(d[1])
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _add_surface(occ, obj: GeoObject) -> int:
    k, p = obj.kind, obj.params
    if k == "rect":
        ang = p.get("angle", 0.0)
        if abs(ang) < 1e-12:
            return occ.addRectangle(p["cx"] - p["w"] / 2, p["cy"] - p["h"] / 2, 0, p["w"], p["h"])
        w2, h2 = p["w"] / 2, p["h"] / 2
        c, s = math.cos(ang), math.sin(ang)
        corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
        pts = [occ.addPoint(p["cx"] + cx * c - cy * s, p["cy"] + cx * s + cy * c, 0) for cx, cy in corners]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])
    if k == "circle":
        return occ.addDisk(p["cx"], p["cy"], 0, p["r"], p["r"])
    if k == "ring":
        do = occ.addDisk(p["cx"], p["cy"], 0, p["r_out"], p["r_out"])
        di = occ.addDisk(p["cx"], p["cy"], 0, p["r_in"], p["r_in"])
        out, _ = occ.cut([(2, do)], [(2, di)])
        return out[0][1]
    if k == "sector":
        cx, cy, ra, rb, a1, a2 = p["cx"], p["cy"], p["r_in"], p["r_out"], p["a1"], p["a2"]
        o = occ.addPoint(cx, cy, 0)
        a = occ.addPoint(cx + ra * math.cos(a1), cy + ra * math.sin(a1), 0)
        b = occ.addPoint(cx + rb * math.cos(a1), cy + rb * math.sin(a1), 0)
        c = occ.addPoint(cx + rb * math.cos(a2), cy + rb * math.sin(a2), 0)
        d = occ.addPoint(cx + ra * math.cos(a2), cy + ra * math.sin(a2), 0)
        loop = occ.addCurveLoop([occ.addLine(a, b), occ.addCircleArc(b, o, c),
                                 occ.addLine(c, d), occ.addCircleArc(d, o, a)])
        return occ.addPlaneSurface([loop])
    if k == "polygon":
        pts = [occ.addPoint(x, y, 0) for x, y in p["points"]]
        n = len(pts)
        lines = [occ.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])
    raise ValueError(f"неизвестный примитив {k!r}.")


def _extract_mesh(gmsh) -> tuple[np.ndarray, np.ndarray]:
    """gmsh → (vertices, cells): CCW-ориентация + удаление orphan-узлов (как в PMSM-генераторе)."""
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
    tag2idx = {int(t): i for i, t in enumerate(node_tags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    tris = None
    for et, en in zip(etypes, enodes):
        if et == 2:  # 3-узловой треугольник
            tris = np.asarray(en, dtype=int).reshape(-1, 3)
            break
    if tris is None:
        raise RuntimeError("gmsh не вернул треугольных элементов.")
    cells = np.vectorize(tag2idx.get)(tris)
    verts = np.ascontiguousarray(coords)
    fixed = []
    for tri in cells:
        if signed_area2(verts[tri]) < 0.0:
            tri = tri[[0, 2, 1]]
        fixed.append(tri)
    cells = np.asarray(fixed, dtype=int)
    used = np.unique(cells.reshape(-1))
    if used.size != verts.shape[0]:
        remap = np.full(verts.shape[0], -1, dtype=int)
        remap[used] = np.arange(used.size)
        verts = np.ascontiguousarray(verts[used])
        cells = remap[cells]
    return verts, cells


def _bbox(obj: GeoObject) -> tuple[float, float, float, float]:
    k, p = obj.kind, obj.params
    if k == "circle":
        r = p["r"]
        return p["cx"] - r, p["cy"] - r, p["cx"] + r, p["cy"] + r
    if k in ("ring", "sector"):
        r = p["r_out"]
        return p["cx"] - r, p["cy"] - r, p["cx"] + r, p["cy"] + r
    if k == "rect":
        rad = math.hypot(p["w"] / 2, p["h"] / 2)  # с запасом на поворот
        return p["cx"] - rad, p["cy"] - rad, p["cx"] + rad, p["cy"] + rad
    pts = np.asarray(p["points"], dtype=float)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def auto_domain(objects, *, material, margin_frac: float = 0.4, mesh_size: float | None = None) -> GeoObject:
    """Фон-домен (прямоугольник) по общему bbox объектов + запас. Его граница = внешняя ГУ A_z=0."""
    boxes = np.array([_bbox(o) for o in objects], dtype=float)
    xmin, ymin = boxes[:, 0].min(), boxes[:, 1].min()
    xmax, ymax = boxes[:, 2].max(), boxes[:, 3].max()
    w, h = xmax - xmin, ymax - ymin
    mx, my = margin_frac * w, margin_frac * h
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    return GeoObject(name="domain", kind="rect",
                     params={"cx": cx, "cy": cy, "w": w + 2 * mx, "h": h + 2 * my},
                     material=material, mesh_size=mesh_size)


def build_object_problem(objects, domain, *, default_mesh_size: float, T: float = 20.0) -> Problem2D:
    """
    Собрать общий Problem2D из фон-домена + списка объектов. Приоритет = порядок (последний
    сверху). gmsh: все поверхности fragment → конформная сетка; регион ячейки = самый
    приоритетный объект, содержащий её центроид (иначе домен). Размер сетки — по объекту.
    ⚠ gmsh требует главный поток.
    """
    import gmsh

    domain.validate()
    for o in objects:
        o.validate()
    all_objs = [domain] + list(objects)           # индекс 0 = домен (низший приоритет)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        surfs = [_add_surface(occ, o) for o in all_objs]
        dt = [(2, s) for s in surfs]
        occ.fragment(dt, dt)
        occ.synchronize()

        def _size_cb(dim, tag, x, y, z, lc):
            for idx in range(len(all_objs) - 1, 0, -1):
                if contains(all_objs[idx], x, y):
                    return all_objs[idx].mesh_size or default_mesh_size
            return domain.mesh_size or default_mesh_size

        gmsh.model.mesh.setSizeCallback(_size_cb)
        sizes = [o.mesh_size or default_mesh_size for o in all_objs]
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(sizes) * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(sizes))
        gmsh.model.mesh.generate(2)
        verts, cells = _extract_mesh(gmsh)
    finally:
        gmsh.finalize()

    mesh = TriangleMesh(vertices=verts, cells=cells)
    nc = mesh.n_cells
    region = np.zeros(nc, dtype=int)
    j_cells = np.zeros(nc, dtype=float)
    axis = np.zeros((nc, 2), dtype=float)
    for c in range(nc):
        cx, cy = mesh.cell_centroid(c)
        owner, rid = domain, 0
        for idx in range(len(all_objs) - 1, 0, -1):   # объекты сверху вниз; домен — запас
            if contains(all_objs[idx], cx, cy):
                owner, rid = all_objs[idx], idx
                break
        region[c] = rid
        if owner.current_density:
            j_cells[c] = owner.current_density
        if isinstance(owner.material, MagnetMaterial):
            axis[c] = _magnet_axis(owner, cx, cy)

    regions = {i: Region2D(i, o.name, o.material) for i, o in enumerate(all_objs)}
    # Ось магнита нужна ВСЕГДА, если в модели есть магнитный МАТЕРИАЛ (даже если такой объект
    # перекрыт и не получил ячеек) — иначе Problem2D.validate() справедливо ругается. Для
    # неполученных магнитом ячеек ось = 0 (вклада нет).
    magnet_present = any(isinstance(o.material, MagnetMaterial) for o in all_objs)
    return Problem2D(
        mesh=mesh, cell_region=region, regions=regions,
        magnet_axis=(axis if magnet_present else None),
        j_cells=(j_cells if np.any(j_cells) else None), T=float(T),
    )
