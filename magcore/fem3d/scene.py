from __future__ import annotations

import numpy as np

from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d.mesh import _FACES, unique_rows
from magcore.fem3d.objects import ARROWS_ALONG, ARROWS_MAX, MagnetArrows, _evenly, object_axis
from magcore.fem3d.postprocess import _cells_of, plane_polygons, section_triangles
from magcore.fem3d.problem import Problem3D
from magcore.fem3d.scalar import ScalarField3D
from magcore.packing import pack, unpack  # noqa: F401 — общие с 2D (файл расчёта); прежние импорты из scene

# СЦЕНА ОБЪЁМНОГО ВИДА (этап 3D-5, план — docs/plan_3d_2026-09-11.md). Всё, что рисует браузер, —
# треугольники в глобальных координатах, у каждого — номер ячейки сетки; величины поля — по ячейкам.
# Поверхности объектов берутся из ТОЙ ЖЕ сетки, на которой решено, а сечение — тем же кодом, что
# поток через сечение (`plane_polygons`): картинка совпадает с расчётом. Цвет и шкалу строит браузер.

# Осевая линия тела на виде выходит за тело на эту долю длины с каждой стороны; длина — проекция тела
# на ось, но не меньше его наибольшего габарита (у тонкого диска ось иначе не видна за плоскостью).
AXIS_MARGIN = 0.15


def material_kind(material) -> str:
    """Род материала для вида: 'magnet' | 'steel' | 'linear' | 'air'."""
    if isinstance(material, MagnetMaterial):
        return "magnet"
    if isinstance(material, SteelMaterial):
        return "steel"
    if isinstance(material, LinearMaterial):
        return "linear"
    if isinstance(material, Air):
        return "air"
    raise TypeError(f"неизвестный материал: {type(material)}")


def region_surfaces(problem: Problem3D) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Поверхность каждого объекта (кроме фон-домена, регион 0): грани его ячеек, за которыми другой
    регион или внешняя граница, с НАРУЖНОЙ нормалью (локальный порядок граней положительно
    ориентированной ячейки). Стык двух объектов входит в поверхность обоих — с противоположными
    нормалями. Возвращает {регион: (треугольники (t,3,3) [м], номер ячейки объекта (t,))}.
    """
    mesh = problem.mesh
    reg = np.asarray(problem.cell_region)
    faces = mesh.cells[:, _FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(mesh.n_cells), 4)
    _, inv = unique_rows(np.sort(faces, axis=1), return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    order = np.argsort(inv, kind="stable")
    same = inv[order[1:]] == inv[order[:-1]]              # внутренняя грань — две ячейки подряд
    i, j = order[:-1][same], order[1:][same]
    other = np.full(faces.shape[0], -1, dtype=np.int64)
    other[i], other[j] = owner[j], owner[i]
    r_own = reg[owner]
    r_other = np.where(other >= 0, reg[np.maximum(other, 0)], -1)
    surf = (r_own != 0) & (r_own != r_other)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rid in np.unique(r_own[surf]):
        m = surf & (r_own == rid)
        out[int(rid)] = (mesh.vertices[faces[m]], owner[m])
    return out


def axis_segment(obj, points) -> np.ndarray:
    """
    Отрезок оси тела для вида (2,3) [м]: прямая `object_axis` на участке проекции точек тела `points`
    (например, вершин его поверхности) на ось, продолженный на AXIS_MARGIN с каждой стороны.
    """
    o, a = object_axis(obj)
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if p.shape[0] == 0:
        raise ValueError(f"{obj.name}: нет точек тела для осевой линии.")
    t = (p - o) @ a
    t0, t1 = float(t.min()), float(t.max())
    pad = AXIS_MARGIN * max(t1 - t0, float(np.max(p.max(axis=0) - p.min(axis=0))))
    return np.array([o + (t0 - pad) * a, o + (t1 + pad) * a])


def cell_magnet_arrows(problem: Problem3D) -> dict[int, MagnetArrows]:
    """
    Стрелки намагничивания по ячейкам сетки (этап 3D-7): ровно то, что уйдёт в решатель. У каждого магнита с
    ячейками — решётка с шагом (наибольший габарит центров его ячеек) / ARROWS_ALONG; в каждой занятой клетке —
    ячейка, чей центр ближе всего к центру клетки; стрелка — в её центре, направление — её ось
    `problem.magnet_axis`. Не больше ARROWS_MAX на магнит. Возвращает {регион: MagnetArrows}.
    """
    out: dict[int, MagnetArrows] = {}
    if problem.magnet_axis is None:
        return out
    reg = np.asarray(problem.cell_region)
    cen = problem.mesh.cell_centroids()
    for rid, region in sorted(problem.regions.items()):
        if not isinstance(region.material, MagnetMaterial):
            continue
        cells = np.where(reg == rid)[0]
        if cells.size == 0:
            continue
        c = cen[cells]
        lo = c.min(axis=0)
        ext = float(np.max(c.max(axis=0) - lo))
        h = ext / ARROWS_ALONG if ext > 0.0 else 1.0
        idx = np.floor((c - lo) / h).astype(np.int64)
        dist = np.linalg.norm(c - (lo + (idx + 0.5) * h), axis=1)
        _, box = np.unique(idx, axis=0, return_inverse=True)
        box = np.asarray(box).reshape(-1)
        order = np.lexsort((dist, box))                          # в каждой клетке — сначала ближайшая
        first = order[np.r_[True, box[order][1:] != box[order][:-1]]]
        sel = cells[first][_evenly(first.size, ARROWS_MAX)]
        out[int(rid)] = MagnetArrows(cen[sel], np.asarray(problem.magnet_axis)[sel].copy(), h)
    return out


def arrows_payload(arrows: MagnetArrows | None, *, scale: float = 1.0e3) -> dict | None:
    """Стрелки для браузера: точки × scale (мм) и направления — float32 base64, шаг решётки × scale."""
    if arrows is None:
        return None
    return {"n": int(arrows.points.shape[0]), "points": pack(np.asarray(arrows.points) * scale, np.float32),
            "dirs": pack(arrows.directions, np.float32), "spacing": float(arrows.spacing) * scale}


def cell_quantities(field: ScalarField3D) -> dict[str, tuple[np.ndarray, str]]:
    """
    Величины по ячейкам для вида: {имя: (значения (n_cells,), единица)}. У величин магнита вне
    магнитов — NaN.
      B, Bx, By, Bz — индукция [Тл];  H, Hx, Hy, Hz — напряжённость [кА/м];  mu — хордовая относительная проницаемость
      (след тензора / 3);  Hpar — поле вдоль оси магнита [кА/м];  margin — запас до колена [кА/м]
      (< 0 — за коленом);  loss — потеря ремнантности с учётом истории [Тл].
    """
    B, H = field.B_cells, field.H_cells
    q = {"B": (np.linalg.norm(B, axis=1), "Тл"),
         "Bx": (B[:, 0].copy(), "Тл"), "By": (B[:, 1].copy(), "Тл"), "Bz": (B[:, 2].copy(), "Тл"),
         "H": (np.linalg.norm(H, axis=1) / 1.0e3, "кА/м"),
         "Hx": (H[:, 0] / 1.0e3, "кА/м"), "Hy": (H[:, 1] / 1.0e3, "кА/м"), "Hz": (H[:, 2] / 1.0e3, "кА/м"),
         "mu": (np.trace(field.mu_cells, axis1=1, axis2=2) / 3.0, "")}
    risk = field.risk
    if risk is not None:
        nc = B.shape[0]
        for key, vals, unit in (("Hpar", risk.H_par / 1.0e3, "кА/м"),
                                ("margin", risk.margin / 1.0e3, "кА/м"),
                                ("loss", risk.loss, "Тл")):
            a = np.full(nc, np.nan)
            a[risk.cell_indices] = vals
            q[key] = (a, unit)
    return q


def section(problem: Problem3D, point, normal, *, objects=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Сечение плоскостью (точка `point` [м], нормаль `normal`) ячеек объектов `objects` (None — вся
    область): треугольники (t,3,3) [м] и номер ячейки на треугольник (t,).
    """
    sel = np.where(_cells_of(problem, objects))[0]
    cells, poly, k, _ = plane_polygons(problem.mesh, point, normal, sel)
    tris, owner = section_triangles(poly, k)
    return tris, cells[owner]


def scene_payload(problem: Problem3D, *, scale: float = 1.0e3) -> dict:
    """
    Сцена для браузера: поверхности объектов (координаты × scale, по умолчанию мм; float32 base64 —
    по 9 чисел на треугольник) с номерами ячеек (uint32 base64), стрелки намагничивания магнитов по
    ячейкам (`cell_magnet_arrows`), габарит объектов и домена.
    """
    objects = []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    arrows = cell_magnet_arrows(problem)
    for rid, (tris, cells) in sorted(region_surfaces(problem).items()):
        r = problem.regions[rid]
        xyz = tris.reshape(-1, 3) * scale
        lo, hi = np.minimum(lo, xyz.min(axis=0)), np.maximum(hi, xyz.max(axis=0))
        objects.append({"id": int(rid), "name": r.name, "material": material_kind(r.material),
                        "n": int(tris.shape[0]), "tris": pack(xyz, np.float32),
                        "cells": pack(cells, np.uint32), "arrows": arrows_payload(arrows.get(int(rid)), scale=scale)})
    v = problem.mesh.vertices * scale
    return {"objects": objects,
            "bbox": [lo.tolist(), hi.tolist()] if objects else None,
            "domain": [v.min(axis=0).tolist(), v.max(axis=0).tolist()]}
