import dataclasses
import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    contains3d,
    magnet_axis_at,
    object_axis,
    object_bbox,
    object_volume,
    rotation_matrix,
    signed_distance,
    solve_nonlinear3d,
)

# Этап 3D-1: объектная свободная 3D-геометрия. Оракулы проверяют НОВЫЙ код — сборку геометрии:
#  (1) предикаты и расстояния на РУЧНЫХ точках;
#  (2) поворот — порядок X → Y → Z, ортонормированность, обратимость;
#  (3) объёмы: плоскогранные тела — точно, выпуклые кривые — сходимость второго порядка;
#  (4) вершины ячеек каждого тела лежат в его аналитической форме — ловит расхождение
#      соглашений о повороте между нашим кодом и gmsh;
#  (5) наложение по приоритету — точные объёмы;
#  (6) граница вогнутой кривой поверхности (труба ↔ вставка) разделена точно;
#  (7) ось намагничивания по ячейкам; (8) негативные случаи.

MM = 1.0e-3
AIR = Air()
STEEL = SteelMaterial(m270_35a_bh_curve())
MAGNET = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))


def obj(name, kind, params_mm, material=AIR, center=(0, 0, 0), rotation=(0, 0, 0), **kw):
    """Объект в миллиметрах (углы сектора — в градусах, повороты — в радианах) → СИ."""
    p = {}
    for key, val in params_mm.items():
        if key in ("a1", "a2"):
            p[key] = math.radians(val)
        elif key == "points":
            p[key] = [(x * MM, y * MM) for x, y in val]
        else:
            p[key] = val * MM
    if kw.get("mesh_size") is not None:
        kw["mesh_size"] = kw["mesh_size"] * MM
    return GeoObject3D(name, kind, p, material, center=tuple(c * MM for c in center),
                       rotation=tuple(float(r) for r in rotation), **kw)


def pts_mm(*points):
    return np.array(points, dtype=float) * MM


# ----------------------------------------------------------------------- предикаты и расстояния
def test_contains_predicates_on_hand_points():
    box = obj("b", "box", {"lx": 10, "ly": 6, "lz": 4})
    assert contains3d(box, pts_mm((4.9, 2.9, 1.9), (5.1, 0, 0), (0, 0, 2.1))).tolist() == [True, False, False]
    cyl = obj("c", "cylinder", {"r": 3, "h": 10})                                   # ось z
    assert contains3d(cyl, pts_mm((0, 0, 4.9), (4.9, 0, 0))).tolist() == [True, False]
    cyl_x = obj("cx", "cylinder", {"r": 3, "h": 10}, rotation=(0, math.pi / 2, 0))   # ось → x
    assert contains3d(cyl_x, pts_mm((4.9, 0, 0), (0, 0, 4.9))).tolist() == [True, False]
    tube = obj("t", "tube", {"r_in": 2, "r_out": 4, "h": 6})
    assert contains3d(tube, pts_mm((3, 0, 0), (1, 0, 0), (5, 0, 0), (3, 0, 3.1))).tolist() == [
        True, False, False, False]
    sec = obj("s", "tube_sector", {"r_in": 2, "r_out": 4, "h": 6, "a1": 0, "a2": 90})
    assert contains3d(sec, pts_mm((2.1, 2.1, 0), (-3, 0, 0), (3, -0.1, 0))).tolist() == [True, False, False]
    sph = obj("p", "sphere", {"r": 5})
    assert contains3d(sph, pts_mm((2.8, 2.8, 2.8), (3, 3, 3))).tolist() == [True, False]
    tri = obj("tr", "prism", {"points": [(0, 0), (10, 0), (0, 10)], "h": 4})
    assert contains3d(tri, pts_mm((2, 2, 1.9), (6, 6, 0), (2, 2, 2.1))).tolist() == [True, False, False]
    ell = obj("L", "prism", {"points": [(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10)], "h": 2})
    assert contains3d(ell, pts_mm((1, 8, 0), (8, 8, 0), (8, 1, 0))).tolist() == [True, False, True]


def test_signed_distance_exact_values():
    box = obj("b", "box", {"lx": 10, "ly": 10, "lz": 10})
    assert np.allclose(signed_distance(box, pts_mm((8, 0, 0), (0, 0, 0), (8, 8, 0))) / MM,
                       [3.0, -5.0, math.hypot(3, 3)])
    cyl = obj("c", "cylinder", {"r": 5, "h": 10})
    assert np.allclose(signed_distance(cyl, pts_mm((0, 0, 8), (8, 0, 0), (8, 0, 8))) / MM,
                       [3.0, 3.0, math.hypot(3, 3)])
    tube = obj("t", "tube", {"r_in": 2, "r_out": 4, "h": 6})
    assert np.allclose(signed_distance(tube, pts_mm((0, 0, 0), (3, 0, 0))) / MM, [2.0, -1.0])
    sec = obj("s", "tube_sector", {"r_in": 2, "r_out": 4, "h": 6, "a1": 0, "a2": 90})
    assert np.allclose(signed_distance(sec, pts_mm((-3, 0, 0))) / MM, [math.hypot(3, 2)])
    sph = obj("p", "sphere", {"r": 5})
    assert np.allclose(signed_distance(sph, pts_mm((0, 0, 7), (1, 0, 0))) / MM, [2.0, -4.0])
    tri = obj("tr", "prism", {"points": [(0, 0), (10, 0), (0, 10)], "h": 4})
    assert np.allclose(signed_distance(tri, pts_mm((-1, 5, 0), (1, 5, 0))) / MM, [1.0, -1.0])


def test_rotation_order_and_orthonormality():
    ex, ez = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotation_matrix((0, 0, math.pi / 2)) @ ex, [0, 1, 0])
    assert np.allclose(rotation_matrix((0, math.pi / 2, 0)) @ ez, [1, 0, 0])
    # порядок X → Y → Z: при rz = 0 R = Ry·Rx ⇒ ez → (0, −1, 0); обратный порядок дал бы (1, 0, 0)
    assert np.allclose(rotation_matrix((math.pi / 2, math.pi / 2, 0)) @ ez, [0, -1, 0])
    R = rotation_matrix((0.3, -1.1, 2.0))
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1.0) < 1e-12
    o = obj("b", "box", {"lx": 1, "ly": 2, "lz": 3}, center=(1, 2, 3), rotation=(0.3, -1.1, 2.0))
    p = np.random.default_rng(0).normal(size=(5, 3)) * MM
    assert np.allclose(o.to_global(o.to_local(p)), p)


# ----------------------------------------------------------------------- негативные случаи
@pytest.mark.parametrize("kind, params", [
    ("cube", {"lx": 1}),
    ("box", {"lx": 0, "ly": 1, "lz": 1}),
    ("box", {"lx": 1, "ly": 1}),
    ("cylinder", {"r": -1, "h": 1}),
    ("tube", {"r_in": 3, "r_out": 2, "h": 1}),
    ("tube_sector", {"r_in": 1, "r_out": 2, "h": 1, "a1": 30, "a2": 30}),
    ("sphere", {"r": 0}),
    ("prism", {"points": [(0, 0), (1, 0)], "h": 1}),
    ("prism", {"points": [(0, 0), (1, 1), (2, 2)], "h": 1}),                # нулевая площадь
    ("prism", {"points": [(0, 0), (2, 2), (2, 0), (0, 1)], "h": 1}),        # самопересечение
])
def test_invalid_shapes_rejected(kind, params):
    with pytest.raises(ValueError):
        obj("x", kind, params).validate()


def test_invalid_options_rejected():
    p = {"lx": 1 * MM, "ly": 1 * MM, "lz": 1 * MM}
    bad = [dict(magnet_dir="up"), dict(magnet_dir=(0.0, 0.0, 0.0)), dict(mesh_size=0.0),
           dict(priority=0), dict(center=(0.0, float("nan"), 0.0)), dict(rotation=(0.0, 0.0))]
    for kw in bad:
        with pytest.raises(ValueError):
            GeoObject3D("x", "box", p, AIR, **kw).validate()


def test_auto_domain_encloses_rotated_objects_with_margin():
    o = obj("b", "box", {"lx": 20, "ly": 2, "lz": 2}, STEEL, center=(3, 4, 5), rotation=(0, 0, math.pi / 4))
    dom = auto_domain3d([o], material=AIR, margin_frac=1.0)
    lo, hi = object_bbox(o)
    dlo, dhi = object_bbox(dom)
    ext = float((hi - lo).max())
    assert np.allclose(lo - dlo, ext) and np.allclose(dhi - hi, ext)   # запас одинаковый со всех сторон
    assert dom.kind == "box" and dom.material is AIR


# ----------------------------------------------------------------------- сборка сетки (gmsh)
def _build(objects, *, margin=0.5, dom_mesh=6.0, default=3.0, **kw):
    pytest.importorskip("gmsh")
    dom = auto_domain3d(objects, material=AIR, margin_frac=margin, mesh_size=dom_mesh * MM)
    return dom, build_object_problem3d(objects, dom, default_mesh_size=default * MM, **kw)


def _cells_of(prob, name):
    rid = next(i for i, r in prob.regions.items() if r.name == name)
    return np.where(np.asarray(prob.cell_region) == rid)[0]


def _vertex_sd(prob, o, cells):
    v = np.unique(prob.mesh.cells[cells].ravel())
    return signed_distance(o, prob.mesh.vertices[v])


def test_planar_bodies_exact_volume_and_inside_their_shape():
    box = obj("box", "box", {"lx": 10, "ly": 6, "lz": 4}, STEEL, center=(-9, 0, 0),
              rotation=(0.3, 0.5, 0.7), mesh_size=2)
    ell = obj("L", "prism", {"points": [(0, 0), (8, 0), (8, 3), (3, 3), (3, 8), (0, 8)], "h": 3},
              STEEL, center=(4, -3, 1), rotation=(0.0, 0.4, -0.2), mesh_size=2)
    dom, prob = _build([box, ell])
    assert prob.validate() == []
    vols = prob.region_volumes()
    for rid, o in ((1, box), (2, ell)):
        assert abs(vols[rid] - object_volume(o)) / object_volume(o) < 1e-9     # плоские грани ⇒ точно
        assert _vertex_sd(prob, o, _cells_of(prob, o.name)).max() < 1e-10    # поворот как у gmsh
    assert abs(prob.mesh.cell_volumes().sum() - object_volume(dom)) / object_volume(dom) < 1e-9
    assert prob.mesh.quality().min() > 0.01                                   # без вырожденных ячеек


@pytest.mark.parametrize("kind, params", [
    ("cylinder", {"r": 5, "h": 8}),
    ("sphere", {"r": 5}),
])
def test_convex_curved_volume_converges_second_order(kind, params):
    # Вписанная ломаная: ошибка объёма ~ (h/r)² ⇒ при h/2 должна упасть примерно вчетверо.
    errs = []
    for h in (1.2, 0.6):
        o = obj("o", kind, params, STEEL, rotation=(0.2, -0.3, 0.4), mesh_size=h)
        _, prob = _build([o], margin=0.3, dom_mesh=4.0, default=h, curvature_elements=0)
        errs.append(abs(prob.region_volumes()[1] - object_volume(o)) / object_volume(o))
        assert _vertex_sd(prob, o, _cells_of(prob, "o")).max() < 1e-9
    assert errs[1] < errs[0] / 2.5
    assert errs[1] < 0.01


@pytest.mark.parametrize("kind, params", [
    ("tube", {"r_in": 3, "r_out": 5, "h": 6}),
    ("tube_sector", {"r_in": 3, "r_out": 5, "h": 6, "a1": 20, "a2": 290}),   # > π: дуги делятся
])
def test_concave_bodies_volume_and_shape(kind, params):
    # У трубы внешняя (выпуклая) и внутренняя (вогнутая) ошибки ломаной почти гасят друг друга,
    # поэтому здесь проверяется точность и форма, а порядок — на выпуклых телах и на вставке.
    o = obj("o", kind, params, STEEL, rotation=(0.1, 0.2, -0.3), mesh_size=0.6)
    _, prob = _build([o], margin=0.3, dom_mesh=4.0, default=0.6, curvature_elements=0)
    assert abs(prob.region_volumes()[1] - object_volume(o)) / object_volume(o) < 0.005
    assert _vertex_sd(prob, o, _cells_of(prob, "o")).max() < 1e-9


def test_concave_interface_is_split_exactly():
    # Труба и вставка в её отверстие: ни одна ячейка трубы не заходит в отверстие, ни одна
    # ячейка вставки не выходит за него — разделение по карте фрагментов, а не по центроиду.
    r_in = 3.0
    tube = obj("tube", "tube", {"r_in": r_in, "r_out": 5, "h": 6}, STEEL, mesh_size=1.5)
    core = obj("core", "cylinder", {"r": r_in, "h": 6}, MAGNET, magnet_dir="axial", mesh_size=1.5)
    _, prob = _build([tube, core], curvature_elements=0)
    V = prob.mesh.vertices
    rho_t = np.hypot(*V[np.unique(prob.mesh.cells[_cells_of(prob, "tube")])][:, :2].T)
    rho_c = np.hypot(*V[np.unique(prob.mesh.cells[_cells_of(prob, "core")])][:, :2].T)
    assert rho_t.min() >= r_in * MM * (1.0 - 1e-9)
    assert rho_c.max() <= r_in * MM * (1.0 + 1e-9)
    # Объём вставки — в пределах ошибки вписанного многогранника: у хорды с углом θ = h/r недобор
    # площади круга 1 − sin θ / θ (при h = 1,5 мм, r = 3 мм — около 4 %); запас 1,5 — на
    # неравномерность хорд. Разделение точное (проверки выше), здесь — только разумный объём.
    theta = 1.5 / r_in
    bound = 1.5 * (1.0 - math.sin(theta) / theta)
    assert abs(prob.region_volumes()[2] - object_volume(core)) / object_volume(core) < bound


def test_priority_overlap_exact_volumes():
    a = obj("a", "box", {"lx": 10, "ly": 10, "lz": 10}, STEEL, mesh_size=2.5)
    b = obj("b", "box", {"lx": 10, "ly": 10, "lz": 10}, STEEL, center=(5, 0, 0), mesh_size=2.5)
    _, prob = _build([a, b])                              # равный приоритет ⇒ позже добавленный сверху
    v = prob.region_volumes() / MM ** 3
    assert np.isclose(v[2], 1000.0, rtol=1e-9) and np.isclose(v[1], 500.0, rtol=1e-9)
    a_high = obj("a", "box", {"lx": 10, "ly": 10, "lz": 10}, STEEL, mesh_size=2.5, priority=5)
    _, prob = _build([a_high, b])                         # явный приоритет перекрывает порядок
    v = prob.region_volumes() / MM ** 3
    assert np.isclose(v[1], 1000.0, rtol=1e-9) and np.isclose(v[2], 500.0, rtol=1e-9)


def test_magnet_axis_per_cell():
    ax = obj("ax", "cylinder", {"r": 3, "h": 6}, MAGNET, center=(-10, 0, 0),
             rotation=(0, math.pi / 2, 0), magnet_dir="axial", mesh_size=2)
    rad = obj("rad", "tube", {"r_in": 3, "r_out": 5, "h": 4}, MAGNET, center=(6, 0, 0),
              magnet_dir="radial", mesh_size=2)
    vec = obj("vec", "box", {"lx": 4, "ly": 4, "lz": 4}, MAGNET, center=(0, 12, 0),
              magnet_dir=(1.0, 1.0, 0.0), mesh_size=2)
    fe = obj("fe", "box", {"lx": 4, "ly": 4, "lz": 4}, STEEL, center=(0, -12, 0), mesh_size=2)
    _, prob = _build([ax, rad, vec, fe])
    assert prob.validate() == []
    A = prob.magnet_axis
    c_ax, c_rad, c_vec = _cells_of(prob, "ax"), _cells_of(prob, "rad"), _cells_of(prob, "vec")
    assert np.allclose(A[c_ax], [1.0, 0.0, 0.0])                         # локальная z → глобальная x
    xy = prob.mesh.cell_centroids()[c_rad][:, :2] - np.array([6 * MM, 0.0])
    assert np.allclose(np.linalg.norm(A[c_rad], axis=1), 1.0)
    assert np.allclose(A[c_rad][:, 2], 0.0)
    assert np.all((A[c_rad][:, :2] * xy).sum(axis=1) > 0.0)              # радиально наружу
    assert np.allclose(A[c_vec], [2 ** -0.5, 2 ** -0.5, 0.0])
    others = np.setdiff1d(np.arange(prob.mesh.n_cells), np.concatenate([c_ax, c_rad, c_vec]))
    assert np.allclose(A[others], 0.0)
    assert np.array_equal(np.sort(np.where(prob.magnet_mask())[0]),
                          np.sort(np.concatenate([c_ax, c_rad, c_vec])))


def test_covered_magnet_keeps_axis_and_is_reported():
    mag = obj("mag", "sphere", {"r": 2}, MAGNET, magnet_dir="axial")
    fe = obj("fe", "box", {"lx": 8, "ly": 8, "lz": 8}, STEEL)          # позже ⇒ перекрывает целиком
    _, prob = _build([mag, fe])
    assert _cells_of(prob, "mag").size == 0
    assert prob.magnet_axis is not None and prob.magnet_axis.shape == (prob.mesh.n_cells, 3)
    assert prob.validate() == []
    assert prob.empty_regions() == ["mag"]


def test_object_axis_is_the_line_of_axial_and_radial_magnetization():
    # Этап 3D-7: осевая линия тела на виде и ось отсчёта намагничивания — одна функция `object_axis`.
    # Оракулы ручные: поворот на 90° вокруг X переводит локальную z в глобальную −y; радиальное
    # направление в точке с локальными полярными координатами (ρ, θ, z) — это R·(cos θ, sin θ, 0);
    # у шара — от центра; у тела CAD ось задана в координатах CAD и размещается вместе с телом.
    cyl = obj("cyl", "cylinder", {"r": 3, "h": 6}, MAGNET, center=(1, 2, 3), rotation=(math.pi / 2, 0, 0))
    o, a = object_axis(cyl)
    assert np.allclose(o, pts_mm((1, 2, 3))[0], rtol=0.0, atol=1e-15)
    assert np.allclose(a, [0.0, -1.0, 0.0], rtol=0.0, atol=1e-15)
    tube = obj("tube", "tube", {"r_in": 3, "r_out": 5, "h": 4}, MAGNET, center=(6, -2, 1), rotation=(0.3, -0.2, 0.5))
    th = np.linspace(0.1, 6.0, 7)
    loc = np.stack([4 * MM * np.cos(th), 4 * MM * np.sin(th), np.linspace(-1.5, 1.5, 7) * MM], axis=1)
    p = tube.to_global(loc)
    expected = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1) @ rotation_matrix(tube.rotation).T
    for d, sign in (("radial", 1.0), ("radial-in", -1.0)):
        assert np.allclose(magnet_axis_at(dataclasses.replace(tube, magnet_dir=d), p), sign * expected,
                           rtol=0.0, atol=1e-12)
    assert np.array_equal(magnet_axis_at(dataclasses.replace(tube, magnet_dir="axial"), p),
                          np.tile(object_axis(tube)[1], (7, 1)))
    ball = obj("ball", "sphere", {"r": 2}, MAGNET, center=(0, 0, 5), magnet_dir="radial")
    assert np.allclose(magnet_axis_at(ball, pts_mm((1, 1, 6))), [[3 ** -0.5] * 3], rtol=0.0, atol=1e-15)
    # ось CAD (1, 2, 3) мм + t·(0, 0, 2); тело повёрнуто на 90° вокруг Z и сдвинуто на (5, 0, 0) мм:
    # точка оси → Rz·(1, 2, 3) + (5, 0, 0) = (3, 1, 3) мм, направление +z
    cad = GeoObject3D("cad", "step", {"path": "нет.step", "body": 0, "axis_origin": tuple(pts_mm((1, 2, 3))[0]),
                                      "axis_dir": (0.0, 0.0, 2.0)}, MAGNET,
                      center=tuple(pts_mm((5, 0, 0))[0]), rotation=(0.0, 0.0, math.pi / 2), magnet_dir="radial")
    o, a = object_axis(cad)
    assert np.allclose(o, pts_mm((3, 1, 3))[0], rtol=0.0, atol=1e-15)
    assert np.allclose(a, [0.0, 0.0, 1.0], rtol=0.0, atol=1e-15)
    assert np.allclose(magnet_axis_at(cad, pts_mm((7, 1, 7))), [[1.0, 0.0, 0.0]], rtol=0.0, atol=1e-12)


def test_magnet_rotation_turns_the_chosen_direction_about_global_axes():
    # Этап 3D-10 (просьба Sergey): направление намагниченности — из списка, затем поворот вокруг ГЛОБАЛЬНЫХ осей
    # X → Y → Z, как поворот тела. Оракулы ручные: «по +X» и 45° вокруг Z — (cos 45°, sin 45°, 0); ось тела +z и
    # 5° вокруг X — (0, −sin 5°, cos 5°); порядок: +z, 90° вокруг X, затем 90° вокруг Y — (0, −1, 0) (в обратном
    # порядке вышло бы (1, 0, 0)); «радиально» у трубы с осью z и 10° вокруг Z — радиальная ось каждой точки
    # повёрнута на 10° против часовой. Без поворота — ровно прежний результат.
    box = obj("m", "box", {"lx": 4, "ly": 4, "lz": 2}, MAGNET)
    p = pts_mm((0.5, -1.0, 0.2), (-1.5, 1.0, -0.8))
    deg = np.radians
    c45 = math.cos(math.pi / 4)
    for md, rot, expected in (((1.0, 0.0, 0.0), (0, 0, 45), (c45, c45, 0.0)),
                              ("axial", (5, 0, 0), (0.0, -math.sin(deg(5)), math.cos(deg(5)))),
                              ((0.0, 0.0, 1.0), (90, 90, 0), (0.0, -1.0, 0.0)),
                              ("axial-in", (0, 0, 30), (0.0, 0.0, -1.0))):
        turned = dataclasses.replace(box, magnet_dir=md, magnet_rotation=tuple(deg(rot)))
        turned.validate()
        assert np.allclose(magnet_axis_at(turned, p), [expected] * 2, rtol=0.0, atol=1e-15), (md, rot)
    tube = obj("tube", "tube", {"r_in": 3, "r_out": 5, "h": 4}, MAGNET, magnet_dir="radial",
               magnet_rotation=(0.0, 0.0, deg(10)))
    th = np.linspace(0.1, 6.0, 7)
    ring = np.stack([4 * MM * np.cos(th), 4 * MM * np.sin(th), np.zeros_like(th)], axis=1)
    got = magnet_axis_at(tube, ring)
    assert np.allclose(got, np.stack([np.cos(th + deg(10)), np.sin(th + deg(10)), 0 * th], axis=1), rtol=0.0, atol=1e-15)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, rtol=0.0, atol=1e-15)
    plain = dataclasses.replace(tube, magnet_rotation=(0.0, 0.0, 0.0))
    default = obj("t", "tube", {"r_in": 3, "r_out": 5, "h": 4}, MAGNET, magnet_dir="radial")    # поле не задано
    assert default.magnet_rotation == (0.0, 0.0, 0.0)
    assert np.array_equal(magnet_axis_at(default, ring), magnet_axis_at(plain, ring))
    assert np.allclose(magnet_axis_at(plain, ring), np.stack([np.cos(th), np.sin(th), 0 * th], axis=1),
                       rtol=0.0, atol=1e-15)
    for bad in ((0.0, 0.0), (0.0, math.nan, 0.0), (math.inf, 0.0, 0.0)):
        with pytest.raises(ValueError, match="magnet_rotation"):
            dataclasses.replace(box, magnet_rotation=bad).validate()


def test_rotated_magnetization_solves_as_the_same_vector():
    # Углы — лишь другой способ задать то же направление: «по +X» с поворотом (20°, −35°, 50°) и тот же вектор,
    # заданный напрямую, дают одну и ту же ось по ячейкам (до округления) и то же поле. Шар в воздухе: внутри
    # поле однородно вдоль оси намагничивания — средняя индукция в шаре идёт вдоль повёрнутого направления, а не
    # вдоль исходного +X: отклонение от повёрнутого во много раз меньше угла поворота (грубая сетка R/2,5).
    pytest.importorskip("gmsh")
    rot = tuple(np.radians((20.0, -35.0, 50.0)))
    v = rotation_matrix(rot) @ np.array([1.0, 0.0, 0.0])
    a = obj("m", "sphere", {"r": 5}, MAGNET, magnet_dir=(1.0, 0.0, 0.0), magnet_rotation=rot, mesh_size=2.0)
    b = dataclasses.replace(a, magnet_dir=tuple(v), magnet_rotation=(0.0, 0.0, 0.0))
    probs = []
    for o in (a, b):
        dom = auto_domain3d([o], material=AIR, margin_frac=2.0)
        probs.append(build_object_problem3d([o], dom, default_mesh_size=2.0 * MM))
    pa, pb = probs
    assert np.array_equal(pa.mesh.cells, pb.mesh.cells) and np.array_equal(pa.cell_region, pb.cell_region)
    assert np.allclose(pa.magnet_axis, pb.magnet_axis, rtol=0.0, atol=1e-15)
    fa, fb = solve_nonlinear3d(pa), solve_nonlinear3d(pb)
    assert np.allclose(fa.B_cells, fb.B_cells, rtol=0.0, atol=1e-12 * np.abs(fb.B_cells).max())
    m = np.asarray(pa.cell_region) == 1
    b_mean = fa.average(fa.B_cells, m)
    angle = lambda x, y: math.acos(min(1.0, abs(float(x @ y)) / (np.linalg.norm(x) * np.linalg.norm(y))))  # noqa: E731
    assert angle(b_mean, v) < 0.05 * angle(v, np.array([1.0, 0.0, 0.0]))


def test_thin_gap_is_resolved_with_well_shaped_cells():
    # Этап 3D-8 (сетка по CAD), правило узких мест: цилиндр в трубе с зазором 0,2 мм при h = 1 мм. Без правила
    # поперёк зазора стоит одна-две плоские ячейки (замерено: медианное качество в зазоре 0,37); с правилом
    # размер на гранях зазора = его толщина, и ячейки в зазоре не хуже, чем в среднем по сетке. Объём зазора
    # на сетке — точный до ошибки вписанных хорд (16 на оборот: у круга 0,64 % площади).
    core = obj("core", "cylinder", {"r": 5, "h": 4}, STEEL)
    shell = obj("shell", "tube", {"r_in": 5.2, "r_out": 7, "h": 4}, STEEL)
    pytest.importorskip("gmsh")
    dom = auto_domain3d([core, shell], material=AIR, margin_frac=1.0, mesh_size=4 * MM)
    med = {}
    for on in (False, True):
        prob = build_object_problem3d([core, shell], dom, default_mesh_size=1 * MM, feature_sizing=on)
        q = prob.mesh.quality()
        cen = prob.mesh.cell_centroids()
        rr = np.hypot(cen[:, 0], cen[:, 1])
        gap = (np.asarray(prob.cell_region) == 0) & (rr > 5 * MM) & (rr < 5.2 * MM) & (np.abs(cen[:, 2]) < 2 * MM)
        med[on] = float(np.median(q[gap]))
        if on:
            assert med[on] >= 0.9 * float(np.median(q))                      # в зазоре не хуже, чем в среднем
            assert q.min() > 0.1
            v_gap = float(prob.mesh.cell_volumes()[gap].sum())
            assert v_gap == pytest.approx(math.pi * (5.2 ** 2 - 5.0 ** 2) * 4 * MM ** 3, rel=0.01)
    assert med[True] > 1.5 * med[False]


def test_mesh_rule_arguments_are_checked():
    box = obj("b", "box", {"lx": 4, "ly": 4, "lz": 4}, STEEL)
    dom = auto_domain3d([box], material=AIR, margin_frac=1.0, mesh_size=4 * MM)
    for kw in ({"thin_factor": 0.0}, {"thin_factor": -1.0}, {"size_growth": 1.0}, {"size_growth": 0.5}):
        with pytest.raises(ValueError):
            build_object_problem3d([box], dom, default_mesh_size=1 * MM, **kw)


def test_object_outside_domain_rejected():
    pytest.importorskip("gmsh")
    inside = obj("in", "box", {"lx": 4, "ly": 4, "lz": 4}, STEEL)
    dom = auto_domain3d([inside], material=AIR, margin_frac=0.5, mesh_size=4 * MM)
    far = obj("far", "box", {"lx": 4, "ly": 4, "lz": 4}, STEEL, center=(50, 0, 0))
    with pytest.raises(ValueError):
        build_object_problem3d([inside, far], dom, default_mesh_size=2 * MM)


def test_per_object_mesh_size():
    fine = obj("fine", "box", {"lx": 6, "ly": 6, "lz": 6}, STEEL, center=(-6, 0, 0), mesh_size=0.8)
    coarse = obj("coarse", "box", {"lx": 6, "ly": 6, "lz": 6}, STEEL, center=(6, 0, 0), mesh_size=2.5)
    _, prob = _build([fine, coarse])
    L = prob.mesh.edge_lengths().mean(axis=1)
    lf = L[_cells_of(prob, "fine")].mean() / (0.8 * MM)
    lc = L[_cells_of(prob, "coarse")].mean() / (2.5 * MM)
    assert 0.5 < lf < 1.6 and 0.5 < lc < 1.6


def test_small_built_mesh_converts_to_old_core_mesh():
    b = obj("b", "box", {"lx": 4, "ly": 4, "lz": 4}, STEEL, mesh_size=2)
    _, prob = _build([b], margin=0.5, dom_mesh=3.0, default=2.0)
    assert prob.mesh.to_tetra_mesh().n_cells == prob.mesh.n_cells     # проверка старого ядра проходит
