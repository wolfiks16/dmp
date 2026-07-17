import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model import (
    Air,
    GeoObject,
    SteelMaterial,
    MagnetMaterial,
    auto_domain,
    build_object_problem,
    contains,
    magnetic_energy,
    solve_problem2d,
)

# Объектная произвольная геометрия. Оракулы (проверяют НОВЫЙ код — сборку геометрии, не
# уже-верифицированную физику): (1) предикаты вхождения на РУЧНЫХ точках; (2) классификация
# ячеек в известных точках → нужный объект + приоритет наложения; (3) свой размер сетки на
# объект; (4) физ. вменяемость — равномерно намагниченный диск даёт ~однородное внутр. поле
# вдоль оси намагничивания (свойство цилиндра, независимое от реализации).


def test_contains_predicates():
    rect = GeoObject("r", "rect", {"cx": 0, "cy": 0, "w": 4, "h": 2}, Air())
    assert contains(rect, 1.9, 0.9) and not contains(rect, 2.1, 0.0) and not contains(rect, 0.0, 1.1)
    circ = GeoObject("c", "circle", {"cx": 1, "cy": 0, "r": 2}, Air())
    assert contains(circ, 2.9, 0.0) and not contains(circ, 3.1, 0.0)
    ring = GeoObject("g", "ring", {"cx": 0, "cy": 0, "r_in": 1, "r_out": 2}, Air())
    assert contains(ring, 1.5, 0.0) and not contains(ring, 0.5, 0.0) and not contains(ring, 2.5, 0.0)
    sec = GeoObject("s", "sector", {"cx": 0, "cy": 0, "r_in": 1, "r_out": 2, "a1": 0.0, "a2": math.pi / 2}, Air())
    assert contains(sec, 1.5, 0.1) and not contains(sec, 0.1, -1.5)          # только 1-й квадрант
    tri = GeoObject("t", "polygon", {"points": [(0, 0), (4, 0), (0, 4)]}, Air())
    assert contains(tri, 1.0, 1.0) and not contains(tri, 3.0, 3.0)

    rot = GeoObject("rr", "rect", {"cx": 0, "cy": 0, "w": 4, "h": 1, "angle": math.pi / 2}, Air())
    assert contains(rot, 0.0, 1.9) and not contains(rot, 1.9, 0.0)          # повёрнут на 90°


def test_rotated_rect_predicate_matches_axis_aligned():
    # Поворот на 90° меняет местами роль w и h — проверка формулы вращения.
    r = GeoObject("r", "rect", {"cx": 2, "cy": 3, "w": 6, "h": 2, "angle": math.pi / 2}, Air())
    assert contains(r, 2.0, 3.0 + 2.9) and not contains(r, 2.0 + 1.1, 3.0)


@pytest.fixture(scope="module")
def _model():
    pytest.importorskip("gmsh")
    steel = GeoObject("steel", "rect", {"cx": -0.008, "cy": 0.0, "w": 0.010, "h": 0.020},
                      SteelMaterial(m270_35a_bh_curve()))
    mag = GeoObject("mag", "circle", {"cx": 0.008, "cy": 0.0, "r": 0.005},
                    MagnetMaterial(n42sh_magnet((1, 0, 0))), magnet_dir=(1.0, 0.0), mesh_size=0.0009)
    dom = auto_domain([steel, mag], material=Air(), margin_frac=0.6, mesh_size=0.0025)
    prob = build_object_problem([steel, mag], dom, default_mesh_size=0.0025, T=20.0)
    return prob, dom, steel, mag


def test_regions_and_known_point_classification(_model):
    prob, dom, steel, mag = _model
    # три региона: домен(0) + сталь(1) + магнит(2); все непустые.
    assert set(prob.regions.keys()) == {0, 1, 2}
    reg = np.asarray(prob.cell_region)
    assert reg.min() == 0 and reg.max() == 2
    for rid in (0, 1, 2):
        assert np.count_nonzero(reg == rid) > 0

    def region_at(x, y):
        i = min(range(prob.mesh.n_cells),
                key=lambda c: (prob.mesh.cell_centroid(c) - np.array([x, y]))[0] ** 2
                + (prob.mesh.cell_centroid(c) - np.array([x, y]))[1] ** 2)
        return int(reg[i]), prob.regions[int(reg[i])].name
    # РУЧНЫЕ точки с известным владельцем (не через contains — независимо):
    assert region_at(0.008, 0.0)[1] == "mag"      # центр магнита
    assert region_at(-0.008, 0.0)[1] == "steel"   # центр стальной пластины
    assert region_at(0.0, 0.010)[1] == "domain"   # заведомо воздух между/вокруг


def test_per_object_mesh_size(_model):
    prob, dom, steel, mag = _model
    reg = np.asarray(prob.cell_region)
    area = np.array([prob.mesh.cell_area(c) for c in range(prob.mesh.n_cells)])
    # магнит задан мельче (0.9мм) домена (2.5мм) ⇒ средняя площадь его ячеек заметно меньше.
    assert area[reg == 2].mean() < area[reg == 0].mean() / 3.0


def test_priority_overlap():
    pytest.importorskip("gmsh")
    a = GeoObject("a", "circle", {"cx": 0.0, "cy": 0.0, "r": 0.010}, SteelMaterial(m270_35a_bh_curve()))
    b = GeoObject("b", "circle", {"cx": 0.005, "cy": 0.0, "r": 0.010},
                  MagnetMaterial(n42sh_magnet((1, 0, 0))), magnet_dir=(1.0, 0.0))  # позже ⇒ сверху
    dom = auto_domain([a, b], material=Air(), mesh_size=0.0025)
    prob = build_object_problem([a, b], dom, default_mesh_size=0.0025)
    reg = np.asarray(prob.cell_region)
    # точка (0.008,0) лежит в ОБОИХ кругах → должен победить более поздний b (id=2, магнит).
    i = min(range(prob.mesh.n_cells),
            key=lambda c: (prob.mesh.cell_centroid(c)[0] - 0.008) ** 2 + prob.mesh.cell_centroid(c)[1] ** 2)
    assert prob.regions[int(reg[i])].name == "b"


def test_covered_magnet_has_axis_and_solves():
    # Магнит полностью перекрыт объектом с бОльшим приоритетом → 0 ячеек магнита. Раньше это
    # давало magnet_axis=None и падение validate(). Теперь ось есть всегда при наличии
    # магнитного МАТЕРИАЛА (нули там, где ячеек нет) ⇒ постановка валидна и решается.
    pytest.importorskip("gmsh")
    mag = GeoObject("mag", "circle", {"cx": 0.0, "cy": 0.0, "r": 0.004},
                    MagnetMaterial(n42sh_magnet((1, 0, 0))), magnet_dir=(1.0, 0.0))
    steel = GeoObject("steel", "circle", {"cx": 0.0, "cy": 0.0, "r": 0.008},
                      SteelMaterial(m270_35a_bh_curve()))       # позже ⇒ перекрывает магнит
    dom = auto_domain([mag, steel], material=Air(), mesh_size=0.0025)
    prob = build_object_problem([mag, steel], dom, default_mesh_size=0.0025)
    reg = np.asarray(prob.cell_region)
    assert int((reg == 1).sum()) == 0                          # у магнита нет ячеек
    assert prob.magnet_axis is not None                        # но ось задана (валидно)
    assert np.asarray(prob.magnet_axis).shape == (prob.mesh.n_cells, 2)
    sol = solve_problem2d(prob, max_iter=60)                    # не падает
    assert sol.converged


def test_magnet_disk_uniform_interior(_model):
    # Равномерно намагниченный (вдоль +x) диск в воздухе → внутри поле ~однородно и вдоль x.
    prob, dom, steel, mag = _model
    # отдельная задача только с магнитным диском (без стали рядом), чтобы не искажать симметрию.
    md = GeoObject("mag", "circle", {"cx": 0.0, "cy": 0.0, "r": 0.006},
                   MagnetMaterial(n42sh_magnet((1, 0, 0))), magnet_dir=(1.0, 0.0), mesh_size=0.0008)
    d = auto_domain([md], material=Air(), margin_frac=1.5, mesh_size=0.003)
    p = build_object_problem([md], d, default_mesh_size=0.003)
    sol = solve_problem2d(p, max_iter=60)
    assert sol.converged
    reg = np.asarray(p.cell_region)
    cen = np.array([p.mesh.cell_centroid(c) for c in range(p.mesh.n_cells)])
    inner = (reg == 1) & (np.hypot(cen[:, 0], cen[:, 1]) < 0.0035)   # ядро диска
    B = sol.B_cells[inner]
    assert B.shape[0] > 10
    bx, by = B[:, 0], B[:, 1]
    assert np.abs(bx.mean()) > 0.05                                   # ненулевое поле
    assert np.abs(by.mean()) < 0.15 * np.abs(bx.mean())              # вдоль оси намагничивания x
    assert bx.std() < 0.2 * abs(bx.mean())                           # ~однородно в ядре
    assert magnetic_energy(sol, axial_length=0.03) > 0.0
