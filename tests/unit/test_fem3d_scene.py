import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    flux_through_plane,
    signed_distance,
    solve_linear3d,
    solve_nonlinear3d,
)
from magcore.fem3d.objects import (
    ARROWS_ALONG,
    ARROWS_MAX,
    ARROWS_MIN,
    object_axis,
    preview_geometry,
    preview_surfaces,
    rotation_matrix,
)
from magcore.fem3d.scene import (
    AXIS_MARGIN,
    arrows_payload,
    axis_segment,
    cell_magnet_arrows,
    cell_quantities,
    pack,
    region_surfaces,
    scene_payload,
    section,
    unpack,
)

# Этап 3D-5: сцена объёмного вида. Оракулы:
#  (1) поверхность каждого объекта замкнута (Σ векторов площади = 0) и ориентирована наружу:
#      объём по теореме Гаусса Σ(x·n)S/3 совпадает с объёмом ячеек объекта ТОЧНО; у плоских тел
#      площадь поверхности — по формуле; стык двух тел входит в поверхности обоих;
#  (2) сечение — тот же код, что поток: площадь сечения плоского тела точная, поток по
#      треугольникам сечения = flux_through_plane;
#  (3) величины вида — те же числа, что в решении (|B|, |H|, запас до колена, потеря);
#  (4) предпросмотр строится тем же построением тел, что и сетка: вершины лежат на поверхности
#      объекта (знаковое расстояние — на уровне округления), площадь плоских тел — точная;
#  (5) упаковка для браузера обратима.

MM = 1.0e-3
AIR = Air()


def _problem(objects, *, margin=1.0, h=1.5 * MM, T=20.0):
    pytest.importorskip("gmsh")
    return build_object_problem3d(objects, auto_domain3d(objects, material=AIR, margin_frac=margin),
                                  default_mesh_size=h, T=T)


def _area_vectors(tris):
    return 0.5 * np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])


def _rid(p, name):
    return next(i for i, r in p.regions.items() if r.name == name)


# ----------------------------------------------------------------------- (1) поверхности
def test_object_surfaces_are_closed_outward_and_exact():
    box = GeoObject3D("b", "box", {"lx": 8 * MM, "ly": 6 * MM, "lz": 4 * MM}, LinearMaterial(2.0),
                      mesh_size=1.5 * MM)
    plate = GeoObject3D("p", "box", {"lx": 8 * MM, "ly": 6 * MM, "lz": 2 * MM},
                        SteelMaterial(m270_35a_bh_curve()), center=(0.0, 0.0, 3 * MM), mesh_size=1.5 * MM)
    ball = GeoObject3D("s", "sphere", {"r": 3 * MM}, LinearMaterial(3.0), center=(12 * MM, 0.0, 0.0),
                       rotation=(0.3, 0.2, 0.1), mesh_size=1.5 * MM)            # b и p — вплотную
    p = _problem([box, plate, ball])
    surf = region_surfaces(p)
    assert set(surf) == {_rid(p, "b"), _rid(p, "p"), _rid(p, "s")}             # фон не рисуется
    vols = p.region_volumes()
    for rid, (tris, cells) in surf.items():
        A = _area_vectors(tris)
        assert np.linalg.norm(A.sum(axis=0)) < 1e-12 * np.abs(A).sum()           # замкнута
        assert (tris.mean(axis=1) * A).sum() / 3.0 == pytest.approx(vols[rid], rel=1e-10)  # наружу
        assert np.all(np.asarray(p.cell_region)[cells] == rid)
    area = {nm: float(np.linalg.norm(_area_vectors(surf[_rid(p, nm)][0]), axis=1).sum()) for nm in "bp"}
    assert area["b"] == pytest.approx(2 * (8 * 6 + 8 * 4 + 6 * 4) * MM ** 2, rel=1e-12)
    assert area["p"] == pytest.approx(2 * (8 * 6 + 8 * 2 + 6 * 2) * MM ** 2, rel=1e-12)
    payload = scene_payload(p)                                                   # для браузера: мм, float32
    ob = {o["name"]: o for o in payload["objects"]}
    tris_b = unpack(ob["b"]["tris"], np.float32).reshape(-1, 3, 3).astype(float) / 1e3
    assert ob["b"]["n"] == tris_b.shape[0] == surf[_rid(p, "b")][0].shape[0]
    assert np.abs(tris_b - surf[_rid(p, "b")][0]).max() < 1e-9
    assert np.array_equal(unpack(ob["b"]["cells"], np.uint32), surf[_rid(p, "b")][1])
    assert (ob["b"]["material"], ob["p"]["material"]) == ("linear", "steel")


# ----------------------------------------------------------------------- (2) сечение
def test_section_is_the_flux_code_and_exact_for_flat_bodies():
    H0 = np.array([100.0, -50.0, 30.0])
    b = GeoObject3D("b", "box", {"lx": 4 * MM, "ly": 3 * MM, "lz": 2 * MM}, LinearMaterial(1.0), mesh_size=1.0 * MM)
    p = _problem([b], margin=1.0, h=1.0 * MM)
    f = solve_linear3d(p, bc="dirichlet", applied_field=H0)
    n = np.array([0.1, 0.1, 1.0])
    nh = n / np.linalg.norm(n)
    tris, cells = section(p, (0.0, 0.0, 0.0), n, objects="b")
    A = _area_vectors(tris) @ nh                                                 # обход — навстречу нормали
    assert A.min() > -1e-12 * A.max()
    assert A.sum() == pytest.approx(12 * MM ** 2 / nh[2], rel=1e-12)          # только вертикальные рёбра
    assert np.all(np.asarray(p.cell_region)[cells] == _rid(p, "b"))
    flux = float((f.B_cells[cells] @ nh * A).sum())
    assert flux == pytest.approx(flux_through_plane(f, (0.0, 0.0, 0.0), n, objects="b"), rel=1e-13)
    tris_all, _ = section(p, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))                  # вся область
    ext = p.mesh.vertices.max(axis=0) - p.mesh.vertices.min(axis=0)
    assert float(_area_vectors(tris_all)[:, 2].sum()) == pytest.approx(ext[0] * ext[1], rel=1e-12)


# ----------------------------------------------------------------------- (3) величины
def test_view_quantities_are_the_solution_numbers():
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    cube = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, mag, magnet_dir="axial",
                       mesh_size=2 * MM)
    p = _problem([cube], margin=2.0, h=2 * MM, T=155.0)                         # часть за коленом
    f = solve_nonlinear3d(p)
    q = cell_quantities(f)
    assert np.array_equal(q["B"][0], np.linalg.norm(f.B_cells, axis=1))
    assert np.array_equal(q["H"][0], np.linalg.norm(f.H_cells, axis=1) / 1.0e3)
    assert np.array_equal(q["Bz"][0], f.B_cells[:, 2])
    idx = f.risk.cell_indices
    other = np.setdiff1d(np.arange(p.mesh.n_cells), idx)
    for key, ref in (("margin", f.risk.margin / 1.0e3), ("loss", f.risk.loss), ("Hpar", f.risk.H_par / 1.0e3)):
        assert np.array_equal(q[key][0][idx], ref), key
        assert np.isnan(q[key][0][other]).all(), key
    assert q["loss"][0][idx].max() > 0.0
    assert {k: u for k, (_, u) in q.items()}["margin"] == "кА/м"


# ----------------------------------------------------------------------- (4) предпросмотр
def test_preview_is_built_by_the_same_bodies_as_the_mesh():
    L_shape = [(0.0, 0.0), (6 * MM, 0.0), (6 * MM, 2 * MM), (2 * MM, 2 * MM), (2 * MM, 5 * MM), (0.0, 5 * MM)]
    objs = [
        GeoObject3D("box", "box", {"lx": 8 * MM, "ly": 6 * MM, "lz": 4 * MM}, AIR,
                    center=(1 * MM, 2 * MM, 3 * MM), rotation=(0.3, -0.2, 0.5)),
        GeoObject3D("tube", "tube", {"r_in": 2 * MM, "r_out": 4 * MM, "h": 5 * MM}, AIR,
                    center=(20 * MM, 0.0, 0.0), rotation=(math.pi / 2, 0.0, 0.0)),
        GeoObject3D("sec", "tube_sector", {"r_in": 2 * MM, "r_out": 5 * MM, "h": 3 * MM, "a1": 0.2, "a2": 4.5},
                    AIR, center=(0.0, 20 * MM, 0.0)),                             # 246° — дуги делятся
        GeoObject3D("ball", "sphere", {"r": 3 * MM}, AIR, center=(-15 * MM, 0.0, 0.0)),
        GeoObject3D("pr", "prism", {"points": L_shape, "h": 3 * MM}, AIR, center=(0.0, -20 * MM, 0.0),
                    rotation=(0.0, 0.0, 0.7)),
    ]
    pytest.importorskip("gmsh")
    surfs = preview_surfaces(objs)
    for o, tris in zip(objs, surfs):
        assert tris.shape[0] > 0, o.name
        assert np.abs(signed_distance(o, tris.reshape(-1, 3))).max() < 1e-10, o.name
    area = [float(np.linalg.norm(_area_vectors(t), axis=1).sum()) for t in surfs]
    assert area[0] == pytest.approx(2 * (8 * 6 + 8 * 4 + 6 * 4) * MM ** 2, rel=1e-12)
    assert area[4] == pytest.approx((2 * 18 + 22 * 3) * MM ** 2, rel=1e-12)   # Г: площадь 18, периметр 22


def test_axis_segment_spans_the_body_along_its_axis():
    # Этап 3D-7: осевая линия тела на виде — прямая `object_axis` на длине проекции тела плюс запас
    # AXIS_MARGIN от большего из «длина проекции, наибольший габарит». Ручные ответы:
    #  цилиндр R 5 × 10 мм вдоль z с центром (1, 2, 3) мм — проекция ±5 мм, габарит 10 мм (торцы — точно)
    #  → ±(5 + 1,5) мм;
    #  пластина 20 × 20 × 1 мм, повёрнутая осью вдоль x, — проекция ±0,5 мм, габарит 20 мм (углы — точно)
    #  → ±(0,5 + 3) мм: у тонкого тела ось видна за его плоскостью.
    cyl = GeoObject3D("cyl", "cylinder", {"r": 5 * MM, "h": 10 * MM}, AIR, center=(1 * MM, 2 * MM, 3 * MM))
    plate = GeoObject3D("plate", "box", {"lx": 20 * MM, "ly": 20 * MM, "lz": 1 * MM}, AIR,
                        rotation=(0.0, math.pi / 2, 0.0))
    pytest.importorskip("gmsh")
    s_cyl, s_plate = (axis_segment(o, t) for o, t in zip((cyl, plate), preview_surfaces([cyl, plate])))
    half = 5 * MM + AXIS_MARGIN * 10 * MM
    assert np.allclose(s_cyl, [[1 * MM, 2 * MM, 3 * MM - half], [1 * MM, 2 * MM, 3 * MM + half]], rtol=0.0, atol=1e-15)
    half = 0.5 * MM + AXIS_MARGIN * 20 * MM
    assert np.allclose(object_axis(plate)[1], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-15)
    assert np.allclose(s_plate, [[-half, 0.0, 0.0], [half, 0.0, 0.0]], rtol=0.0, atol=1e-15)
    with pytest.raises(ValueError):
        axis_segment(cyl, np.zeros((0, 3)))


def _radial_expected(obj, points):
    """Радиальное направление по локальным координатам точки: R·(x, y, 0)/ρ — независимо от `magnet_axis_at`."""
    loc = obj.to_local(points)
    rho = np.hypot(loc[:, 0], loc[:, 1])
    return np.stack([loc[:, 0] / rho, loc[:, 1] / rho, np.zeros_like(rho)], axis=1) @ rotation_matrix(obj.rotation).T


def test_magnet_arrows_before_mesh_are_inside_and_follow_the_magnetization():
    # Этап 3D-7, стрелки до сетки. Оракулы ручные: у радиальной трубы направление — R·(x, y, 0)/ρ по локальным
    # координатам точки; у бруска «вдоль оси», повёрнутого на 90° вокруг Y, — (1, 0, 0); у вектора — он сам.
    # Точки внутри тела — по точному знаковому расстоянию. У стали и без запроса стрелок нет. Тонкая наклонная
    # пластина 20 × 20 × 0,2 мм — стрелки есть. Тонкостенная труба R 9,8…10 мм × 10 мм: центры клеток решётки
    # (шаг 20/6 и 20/12 мм) лежат на радиусах 9,72 и 9,50/10,07 мм — ни один не в стенке (проверено перебором
    # пар), поэтому точки берутся от граней на глубину V/S ≈ 0,1 мм — середину стенки.
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    tube = GeoObject3D("tube", "tube", {"r_in": 3 * MM, "r_out": 5 * MM, "h": 4 * MM}, mag, center=(1 * MM, 2 * MM, 3 * MM),
                       rotation=(0.3, -0.2, 0.5), magnet_dir="radial")
    bar = GeoObject3D("bar", "box", {"lx": 4 * MM, "ly": 3 * MM, "lz": 8 * MM}, mag, center=(15 * MM, 0.0, 0.0),
                      rotation=(0.0, math.pi / 2, 0.0), magnet_dir="axial")
    vec = GeoObject3D("vec", "cylinder", {"r": 2 * MM, "h": 3 * MM}, mag, center=(0.0, 15 * MM, 0.0),
                      magnet_dir=(1.0, 1.0, 0.0))
    plate = GeoObject3D("plate", "box", {"lx": 20 * MM, "ly": 20 * MM, "lz": 0.2 * MM}, mag, center=(0.0, -25 * MM, 0.0),
                        rotation=(0.4, 0.3, 0.0), magnet_dir="axial")
    shell = GeoObject3D("shell", "tube", {"r_in": 9.8 * MM, "r_out": 10 * MM, "h": 10 * MM}, mag,
                        center=(40 * MM, 0.0, 0.0), magnet_dir="radial")
    fe = GeoObject3D("fe", "box", {"lx": 4 * MM, "ly": 4 * MM, "lz": 4 * MM}, SteelMaterial(m270_35a_bh_curve()),
                     center=(-15 * MM, 0.0, 0.0))
    pytest.importorskip("gmsh")
    objs = [tube, bar, vec, plate, shell, fe]
    pv = preview_geometry(objs, magnet_arrows=True)
    assert pv.arrows[5] is None and preview_geometry(objs).arrows == [None] * 6       # без запроса — не считаются
    for o, a in zip(objs[:5], pv.arrows[:5]):
        assert ARROWS_MIN <= a.points.shape[0] <= ARROWS_MAX, o.name
        assert np.all(signed_distance(o, a.points) < 0.0), o.name
    assert np.allclose(pv.arrows[0].directions, _radial_expected(tube, pv.arrows[0].points), rtol=0.0, atol=1e-12)
    assert np.allclose(pv.arrows[1].directions, [1.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
    assert np.allclose(pv.arrows[2].directions, [2 ** -0.5, 2 ** -0.5, 0.0], rtol=0.0, atol=1e-15)
    assert np.allclose(pv.arrows[3].directions, rotation_matrix(plate.rotation)[:, 2], rtol=0.0, atol=1e-12)
    assert np.allclose(pv.arrows[4].directions, _radial_expected(shell, pv.arrows[4].points), rtol=0.0, atol=1e-12)
    s_tri = float(np.linalg.norm(_area_vectors(pv.surfaces[4]), axis=1).sum())
    n_shell = pv.arrows[4].points.shape[0]                           # шаг — «по граням»: √(S/2/n), не шаг решётки
    assert pv.arrows[4].spacing == pytest.approx(math.sqrt(0.5 * s_tri / n_shell), rel=1e-12)


def test_magnet_arrows_on_the_mesh_are_the_solver_axes():
    # Этап 3D-7, стрелки по ячейкам: у каждой стрелки направление — ровно ось её ячейки `magnet_axis` (то, что
    # уйдёт в решатель), точка — центр этой ячейки; перекрытая сталью часть магнита стрелок не получает; у
    # полностью перекрытого магнита стрелок нет; не больше ARROWS_MAX; упаковка обратима.
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    tube = GeoObject3D("tube", "tube", {"r_in": 3 * MM, "r_out": 5 * MM, "h": 4 * MM}, mag, magnet_dir="radial",
                       mesh_size=1.0 * MM)
    cover = GeoObject3D("cover", "box", {"lx": 12 * MM, "ly": 6 * MM, "lz": 6 * MM}, SteelMaterial(m270_35a_bh_curve()),
                        center=(0.0, 3 * MM, 0.0), priority=5)                           # закрывает половину трубы (y > 0)
    hidden = GeoObject3D("hidden", "sphere", {"r": 1 * MM}, mag, center=(0.0, 4 * MM, 0.0))
    prob = _problem([tube, cover, hidden])
    arrows = cell_magnet_arrows(prob)
    rid = {r.name: i for i, r in prob.regions.items()}
    assert set(arrows) == {rid["tube"]}                                                  # у закрытого шара — нет
    a = arrows[rid["tube"]]
    assert ARROWS_MIN <= a.points.shape[0] <= ARROWS_MAX
    cen = prob.mesh.cell_centroids()
    reg = np.asarray(prob.cell_region)
    cells = np.array([int(np.argmin(np.linalg.norm(cen - p, axis=1))) for p in a.points])
    assert np.array_equal(cen[cells], a.points) and np.all(reg[cells] == rid["tube"])
    assert np.array_equal(a.directions, prob.magnet_axis[cells])
    assert np.all(a.points[:, 1] < 0.0)                                                 # только открытая половина
    assert np.allclose(a.directions, _radial_expected(tube, a.points), rtol=0.0, atol=1e-12)
    pl = arrows_payload(a)
    assert pl["n"] == a.points.shape[0] and np.allclose(unpack(pl["points"], np.float32), a.points.ravel() * 1e3, rtol=1e-6)
    assert arrows_payload(None) is None


# ----------------------------------------------------------------------- (5) упаковка
def test_pack_roundtrip():
    a = np.array([1.5, -2.25, 3.0e-7, 1.0e30], dtype=np.float32)
    c = np.array([0, 7, 2 ** 31 + 5], dtype=np.uint32)
    assert np.array_equal(unpack(pack(a, np.float32), np.float32), a)
    assert np.array_equal(unpack(pack(c, np.uint32), np.uint32), c)
