import dataclasses
import math
import os

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    CAD_KIND,
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    object_volume,
    signed_distance,
    step_bodies,
)
from magcore.fem3d.objects import _add_volume, preview_surfaces

# Этап 3D-1б: тела из STEP-файлов CAD. Оракулы:
#  (1) круг «примитивы → STEP в миллиметрах → импорт в метрах»: объёмы тел = формулам до 1e-9, центр масс
#      симметричного тела — его центр; единицы, объявленные в файле, переводятся в метры;
#  (2) сетка тел из STEP — по тем же критериям, что у примитивов (этап 3D-1): плоскогранные тела — точно
#      и вершины в форме исходного примитива, выпуклые кривые — второй порядок и < 1 %, вогнутые — < 0,5 %,
#      средний размер ячейки внутри тела — в пределах (0,5; 1,6)·h; наложение с примитивом — по
#      приоритету, точные объёмы;
#  (3) ось намагничивания тела CAD — вокруг заданной прямой, вместе с размещением тела;
#  (4) настоящие файлы: электродвигатель БПЛА32 (T-FLEX) — объём кольца ротора по площадям его
#      цилиндрических граней = объём тела, 14 магнитов одинаковы, вся сборка строится, объёмы тел на
#      сетке — в пределах ошибки вписанных хорд, магниты радиальны вокруг оси мотора (ось x CAD);
#      сборка «Хурмы» — 6 тел, 4 одинаковых магнита; IM-8008 (SolidWorks) — только поверхности, ошибка;
#  (5) негативные случаи: нет файла, номер тела, «отпечаток» не совпал, ось, поверхности вне тел.

MM = 1.0e-3
AIR = Air()
STEEL = SteelMaterial(m270_35a_bh_curve())
MAGNET = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
MOTORS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "motors"))
UAV32 = os.path.join(MOTORS, "БПЛА32.00.00 СБ _ Электродвигатель.stp")
HURMA = os.path.join(MOTORS, "СБОРКА ХУРМЫ.stp")
IM8008 = os.path.join(MOTORS, "scorpion_im8008", "IM-8008-Public.STEP")
L_SHAPE = [(0, 0), (8, 0), (8, 3), (3, 3), (3, 8), (0, 8)]


def obj(name, kind, params_mm, material=AIR, center=(0, 0, 0), rotation=(0, 0, 0), **kw):
    """Примитив в миллиметрах (углы сектора — в градусах, повороты — в радианах) → СИ."""
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


def cad(name, path, body, material=STEEL, params=None, **kw):
    return GeoObject3D(name, CAD_KIND, {"path": path, "body": body, **(params or {})}, material, **kw)


def in_mm(o):
    """Тот же примитив числами в миллиметрах: размеры и центр × 1000, углы без изменений."""
    p = {}
    for key, val in o.params.items():
        if key in ("a1", "a2"):
            p[key] = val
        elif key == "points":
            p[key] = [(x * 1.0e3, y * 1.0e3) for x, y in val]
        else:
            p[key] = val * 1.0e3
    return dataclasses.replace(o, params=p, center=tuple(c * 1.0e3 for c in o.center))


def write_step(objects, path):
    """
    Записать тела примитивов в STEP В МИЛЛИМЕТРАХ — так выгружают КОМПАС и T-FLEX. Тела строятся сразу
    числами в миллиметрах: масштаб через `occ.dilate` (общее преобразование OpenCASCADE) переводит все
    грани в B-сплайны и искажает объём — у повёрнутого цилиндра +0,86 % (проверено), а построенный сразу
    в мм возвращается из STEP с ошибкой 2·10⁻¹⁴ и аналитическими гранями.
    """
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        for o in objects:
            _add_volume(occ, in_mm(o))
        occ.synchronize()
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


def build(objects, *, margin=1.0, dom_mesh=6.0, default=2.0, curvature_elements=16):
    pytest.importorskip("gmsh")
    dom = auto_domain3d(objects, material=AIR, margin_frac=margin, mesh_size=dom_mesh * MM)
    return dom, build_object_problem3d(objects, dom, default_mesh_size=default * MM,
                                       curvature_elements=curvature_elements)


def cells_of(prob, name):
    rid = next(i for i, r in prob.regions.items() if r.name == name)
    return np.where(np.asarray(prob.cell_region) == rid)[0]


def vertex_sd(prob, o, name):
    v = np.unique(prob.mesh.cells[cells_of(prob, name)].ravel())
    return signed_distance(o, prob.mesh.vertices[v])


# ----------------------------------------------------------------------- (1) круг через STEP
def test_round_trip_volumes_centroids_and_units(tmp_path):
    objs = [
        obj("box", "box", {"lx": 10, "ly": 6, "lz": 4}, center=(-9, 0, 0), rotation=(0.3, 0.5, 0.7)),
        obj("cyl", "cylinder", {"r": 5, "h": 8}, center=(12, 0, 0), rotation=(0.2, -0.3, 0.4)),
        obj("tube", "tube", {"r_in": 3, "r_out": 5, "h": 6}, center=(0, 14, 0), rotation=(0.1, 0.2, -0.3)),
        obj("sec", "tube_sector", {"r_in": 3, "r_out": 5, "h": 6, "a1": 20, "a2": 290}, center=(0, -14, 0)),
        obj("sph", "sphere", {"r": 5}, center=(0, 0, 15)),
        obj("L", "prism", {"points": L_SHAPE, "h": 3}, center=(4, -3, -15), rotation=(0.0, 0.4, -0.2)),
    ]
    path = write_step(objs, tmp_path / "shapes.step")
    bodies = step_bodies(path)
    assert [b.index for b in bodies] == list(range(len(objs)))           # порядок файла сохранён
    for b, o in zip(bodies, objs):
        assert abs(b.volume / object_volume(o) - 1.0) < 1e-9, o.name     # мм файла → м, объём точно
        if o.kind in ("box", "cylinder", "tube", "sphere"):
            assert np.allclose(b.centroid, o.center, rtol=0.0, atol=1e-9), o.name
    assert step_bodies(path) is bodies                                   # кэш по пути, размеру и времени


# ----------------------------------------------------------------------- (2) сетка тел из STEP
def test_planar_step_bodies_mesh_exactly(tmp_path):
    box = obj("box", "box", {"lx": 10, "ly": 6, "lz": 4}, center=(-9, 0, 0), rotation=(0.3, 0.5, 0.7))
    ell = obj("L", "prism", {"points": L_SHAPE, "h": 3}, center=(4, -3, 1), rotation=(0.0, 0.4, -0.2))
    path = write_step([box, ell], tmp_path / "planar.step")
    dom, prob = build([cad("box", path, 0, mesh_size=2 * MM), cad("L", path, 1, mesh_size=2 * MM)])
    assert prob.validate() == []
    vols = prob.region_volumes()
    for rid, o in ((1, box), (2, ell)):
        assert abs(vols[rid] / object_volume(o) - 1.0) < 1e-9, o.name
        assert vertex_sd(prob, o, o.name).max() < 1e-10, o.name          # тело там же, где исходный примитив
    assert abs(prob.mesh.cell_volumes().sum() / object_volume(dom) - 1.0) < 1e-9
    assert prob.mesh.quality().min() > 0.01


@pytest.mark.parametrize("kind, params", [("cylinder", {"r": 5, "h": 8}), ("sphere", {"r": 5})])
def test_convex_curved_step_body_converges_second_order(tmp_path, kind, params):
    # Вписанная ломаная: ошибка объёма ~ (h/r)² ⇒ при h/2 должна упасть примерно вчетверо (как у примитива).
    o = obj("o", kind, params, rotation=(0.2, -0.3, 0.4))
    path = write_step([o], tmp_path / f"{kind}.step")
    errs = []
    for h in (1.2, 0.6):
        _, prob = build([cad("o", path, 0, mesh_size=h * MM)], margin=0.3, dom_mesh=4.0, default=h,
                        curvature_elements=0)
        errs.append(abs(prob.region_volumes()[1] / object_volume(o) - 1.0))
        assert vertex_sd(prob, o, "o").max() < 1e-9
    assert errs[1] < errs[0] / 2.5 and errs[1] < 0.01


@pytest.mark.parametrize("kind, params", [
    ("tube", {"r_in": 3, "r_out": 5, "h": 6}),
    ("tube_sector", {"r_in": 3, "r_out": 5, "h": 6, "a1": 20, "a2": 290}),
])
def test_concave_step_bodies_volume(tmp_path, kind, params):
    o = obj("o", kind, params, rotation=(0.1, 0.2, -0.3))
    path = write_step([o], tmp_path / f"{kind}.step")
    _, prob = build([cad("o", path, 0, mesh_size=0.6 * MM)], margin=0.3, dom_mesh=4.0, default=0.6,
                    curvature_elements=0)
    assert abs(prob.region_volumes()[1] / object_volume(o) - 1.0) < 0.005
    assert vertex_sd(prob, o, "o").max() < 1e-9


def test_mesh_size_inside_step_bodies_and_overlap_priority(tmp_path):
    fine = obj("fine", "box", {"lx": 6, "ly": 6, "lz": 6}, center=(-6, 0, 0))
    coarse = obj("coarse", "box", {"lx": 6, "ly": 6, "lz": 6}, center=(6, 0, 0))
    path = write_step([fine, coarse], tmp_path / "boxes.step")
    _, prob = build([cad("fine", path, 0, mesh_size=0.8 * MM), cad("coarse", path, 1, mesh_size=2.5 * MM)])
    L = prob.mesh.edge_lengths().mean(axis=1)
    lf = L[cells_of(prob, "fine")].mean() / (0.8 * MM)
    lc = L[cells_of(prob, "coarse")].mean() / (2.5 * MM)
    assert 0.5 < lf < 1.6 and 0.5 < lc < 1.6                              # тот же критерий, что у примитивов
    # тело из STEP и примитив перекрываются: равный приоритет — позже добавленный сверху, явный — сильнее
    a = obj("a", "box", {"lx": 10, "ly": 10, "lz": 10})
    pa = write_step([a], tmp_path / "a.step")
    b = obj("b", "box", {"lx": 10, "ly": 10, "lz": 10}, STEEL, center=(5, 0, 0), mesh_size=2.5)
    _, prob = build([cad("a", pa, 0, mesh_size=2.5 * MM), b])
    v = prob.region_volumes() / MM ** 3
    assert np.isclose(v[2], 1000.0, rtol=1e-9) and np.isclose(v[1], 500.0, rtol=1e-9)
    _, prob = build([cad("a", pa, 0, mesh_size=2.5 * MM, priority=5), b])
    v = prob.region_volumes() / MM ** 3
    assert np.isclose(v[1], 1000.0, rtol=1e-9) and np.isclose(v[2], 500.0, rtol=1e-9)


# ----------------------------------------------------------------------- (3) размещение и ось
def test_repeated_body_placement_and_magnet_axes(tmp_path):
    # Труба с осью вдоль x CAD (как у двигателя БПЛА32): радиальное намагничивание — вокруг заданной оси.
    tube = obj("t", "tube", {"r_in": 3, "r_out": 5, "h": 4}, center=(2, 0, 0), rotation=(0.0, math.pi / 2, 0.0))
    path = write_step([tube], tmp_path / "tube.step")
    axis = {"axis_origin": (2 * MM, 0.0, 0.0), "axis_dir": (1.0, 0.0, 0.0)}
    rad = cad("rad", path, 0, MAGNET, params=axis, magnet_dir="radial", mesh_size=1.0 * MM)
    # то же тело второй раз: повёрнуто на 90° вокруг z (вокруг начала CAD) и сдвинуто — ось вместе с телом
    ax = cad("ax", path, 0, MAGNET, params=axis, magnet_dir="axial", center=(0.0, 20 * MM, 0.0),
             rotation=(0.0, 0.0, math.pi / 2), mesh_size=1.0 * MM)
    _, prob = build([rad, ax], default=1.0)
    assert prob.validate() == []
    moved = obj("t2", "tube", {"r_in": 3, "r_out": 5, "h": 4}, center=(0, 22, 0), rotation=(math.pi / 2, 0.0, 0.0))
    assert vertex_sd(prob, tube, "rad").max() < 1e-9
    assert vertex_sd(prob, moved, "ax").max() < 1e-9                       # вторая копия — там, где должна
    A, cen = prob.magnet_axis, prob.mesh.cell_centroids()
    c_rad, c_ax = cells_of(prob, "rad"), cells_of(prob, "ax")
    r = cen[c_rad] - np.array([2 * MM, 0.0, 0.0])
    r[:, 0] = 0.0
    assert np.allclose(A[c_rad], r / np.linalg.norm(r, axis=1)[:, None], rtol=0.0, atol=1e-12)
    assert np.allclose(A[c_ax], [0.0, 1.0, 0.0], rtol=0.0, atol=1e-12)     # ось x CAD, повёрнутая вокруг z → y


# ----------------------------------------------------------------------- (4) настоящие файлы
@pytest.mark.skipif(not os.path.isfile(UAV32), reason="нет файла двигателя БПЛА32")
def test_uav_motor_ring_volume_from_faces_and_identical_magnets():
    gmsh = pytest.importorskip("gmsh")
    bodies = step_bodies(UAV32)
    assert len(bodies) == 16
    vm = np.array([b.volume for b in bodies[2:]])
    assert np.allclose(vm, vm[0], rtol=1e-9, atol=0.0)                     # 14 одинаковых магнитов
    # Кольцо ротора (тело 1, 4 грани): радиусы — из площадей цилиндрических граней A = 2πrL, длина — из
    # положения торцевых плоскостей; объём π(r² − r²)L должен совпасть с объёмом тела — две независимые
    # величины OpenCASCADE (интеграл по граням и по объёму).
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(UAV32)
        gmsh.model.occ.synchronize()
        ring = sorted(t for _, t in gmsh.model.getEntities(3))[1]
        faces = [abs(t) for _, t in gmsh.model.getBoundary([(3, ring)], combined=False, oriented=False)]
        kinds = {f: gmsh.model.getType(2, f) for f in faces}
        cyl = sorted(gmsh.model.occ.getMass(2, f) for f in faces if "Cylinder" in kinds[f])
        ends = sorted(gmsh.model.occ.getCenterOfMass(2, f)[0] for f in faces if "Plane" in kinds[f])
    finally:
        gmsh.finalize()
    assert len(cyl) == 2 and len(ends) == 2
    length = ends[1] - ends[0]
    r_in, r_out = (a / (2.0 * math.pi * length) for a in cyl)
    assert math.pi * (r_out ** 2 - r_in ** 2) * length == pytest.approx(bodies[1].volume, rel=1e-9)
    assert (r_out - r_in) == pytest.approx(1.0 * MM, rel=1e-6)            # стенка кольца 1 мм, Ø38,4 / Ø36,4


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(UAV32), reason="нет файла двигателя БПЛА32")
def test_uav_motor_assembly_builds_with_radial_magnets():
    bodies = step_bodies(UAV32)
    axis = {"axis_origin": (0.0, 0.0, 0.0), "axis_dir": (1.0, 0.0, 0.0)}      # ось мотора — x CAD
    objs = [cad("stator", UAV32, 0, STEEL, params={"volume": bodies[0].volume, "centroid": bodies[0].centroid}),
            cad("ring", UAV32, 1, STEEL)]
    objs += [cad(f"m{k}", UAV32, 2 + k, MAGNET, params=axis, magnet_dir="radial") for k in range(14)]
    _, prob = build(objs, margin=0.5, dom_mesh=6.0, default=1.0)
    assert prob.validate() == [] and prob.empty_regions() == []
    # Объём тела на сетке — в пределах ошибки вписанных хорд: не меньше 16 элементов на оборот
    # (curvature_elements) ⇒ угол хорды θ ≤ π/8, недобор 1 − sin θ/θ; запас 1,5 — на неравномерность хорд.
    theta = math.pi / 8
    bound = 1.5 * (1.0 - math.sin(theta) / theta)
    vols = prob.region_volumes()
    for i, o in enumerate(objs, start=1):
        assert abs(vols[i] / bodies[o.params["body"]].volume - 1.0) < bound, o.name
    A, cen = prob.magnet_axis, prob.mesh.cell_centroids()
    for k in range(14):
        c = cells_of(prob, f"m{k}")
        r = cen[c].copy()
        r[:, 0] = 0.0
        assert np.allclose(A[c], r / np.linalg.norm(r, axis=1)[:, None], rtol=0.0, atol=1e-12)
    steel_cells = np.concatenate([cells_of(prob, "stator"), cells_of(prob, "ring")])
    assert np.allclose(A[steel_cells], 0.0)


@pytest.mark.skipif(not os.path.isfile(HURMA), reason="нет файла сборки Хурмы")
def test_hurma_assembly_bodies():
    bodies = step_bodies(HURMA)
    assert len(bodies) == 6
    vm = np.array([b.volume for b in bodies[:4]])
    assert np.allclose(vm, vm[0], rtol=1e-9, atol=0.0)                     # 4 одинаковых магнита
    assert all(b.volume > 0.0 for b in bodies)


# ----------------------------------------------------------------------- (5) негативные случаи
def test_invalid_step_inputs(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    path = write_step([obj("box", "box", {"lx": 4, "ly": 4, "lz": 4})], tmp_path / "box.step")
    b = step_bodies(path)[0]
    cad("ok", path, 0, params={"volume": b.volume, "centroid": b.centroid}).validate()
    bad = [
        GeoObject3D("x", CAD_KIND, {"path": str(tmp_path / "нет.step"), "body": 0}, STEEL),
        GeoObject3D("x", CAD_KIND, {"body": 0}, STEEL),
        cad("x", path, 1), cad("x", path, -1), cad("x", path, True), cad("x", path, "0"),
        cad("x", path, 0, params={"volume": b.volume * 1.001}),
        cad("x", path, 0, params={"volume": "много"}),
        cad("x", path, 0, params={"centroid": (0.0, 0.0, 1 * MM)}),
        cad("x", path, 0, params={"axis_dir": (0.0, 0.0, 0.0)}),
        cad("x", path, 0, params={"axis_origin": (0.0, 0.0)}),
    ]
    for o in bad:
        with pytest.raises(ValueError):
            o.validate()
    with pytest.raises(ValueError):
        signed_distance(cad("ok", path, 0), np.zeros((1, 3)))              # знакового расстояния формулой нет
    # файл с телом и свободной поверхностью: молча отбрасывать геометрию нельзя
    free = str(tmp_path / "free.step")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 4, 4, 4)
        gmsh.model.occ.addRectangle(10, 0, 0, 4, 4)
        gmsh.model.occ.synchronize()
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.write(free)
    finally:
        gmsh.finalize()
    with pytest.raises(ValueError, match="вне тел"):
        step_bodies(free)
    empty = tmp_path / "empty.step"                                         # верный STEP без геометрии
    empty.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="ascii")
    with pytest.raises(ValueError, match="нет тел — нет и поверхностей"):
        step_bodies(str(empty))
    if os.path.isfile(IM8008):                                              # настоящий файл без тел
        with pytest.raises(ValueError, match="нет тел — только поверхности"):
            step_bodies(IM8008)


def test_preview_surfaces_of_step_body_lie_on_the_shape(tmp_path):
    o = obj("box", "box", {"lx": 10, "ly": 6, "lz": 4}, center=(-9, 0, 0), rotation=(0.3, 0.5, 0.7))
    path = write_step([o], tmp_path / "box.step")
    tris = preview_surfaces([cad("box", path, 0)])[0]
    assert tris.shape[0] > 0 and np.abs(signed_distance(o, tris.reshape(-1, 3))).max() < 1e-10


def _neighbour_size_ratio(mesh):
    """Отношение средних длин рёбер у ячеек с общей гранью (у каждой пары — большее к меньшему)."""
    from magcore.fem3d.mesh import _FACES

    L = mesh.edge_lengths().mean(axis=1)
    faces = np.sort(mesh.cells[:, _FACES].reshape(-1, 3), axis=1)
    _, inv = np.unique(faces, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    owner = np.repeat(np.arange(mesh.n_cells), 4)
    order = np.argsort(inv, kind="stable")
    same = inv[order[1:]] == inv[order[:-1]]
    a, b = owner[order[:-1][same]], owner[order[1:][same]]
    return np.maximum(L[a], L[b]) / np.minimum(L[a], L[b])


@pytest.mark.slow
def test_small_fillets_do_not_drive_the_mesh_and_sizes_grow_smoothly(tmp_path):
    # Этап 3D-8, правила кривизны и плавного роста на теле CAD: брусок 12 × 6 × 3 мм со скруглёнными рёбрами
    # R = 0,1 и 0,025 мм при h = 1 мм. Замерено без правил: скругление в 4 раза меньше — ячеек втрое больше
    # (85 → 252 тыс.), перепад размеров соседних ячеек до 32 раз. С правилами размер на скруглении не мельче
    # 0,25·h, поэтому число ячеек от радиуса почти не зависит, а размер растёт плавно.
    gmsh = pytest.importorskip("gmsh")
    counts, jumps = {}, {}
    for r_mm in (0.1, 0.025):
        path = str(tmp_path / f"fillet_{r_mm}.step")
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            occ = gmsh.model.occ
            box = occ.addBox(-6.0, -3.0, -1.5, 12.0, 6.0, 3.0)
            occ.synchronize()
            occ.fillet([box], [t for _, t in gmsh.model.getEntities(1)], [r_mm])
            occ.synchronize()
            gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            gmsh.write(path)
        finally:
            gmsh.finalize()
        body = cad("b", path, 0, STEEL)
        dom = auto_domain3d([body], material=AIR, margin_frac=0.5, mesh_size=3 * MM)
        for on in (False, True):
            prob = build_object_problem3d([body], dom, default_mesh_size=1 * MM, feature_sizing=on)
            counts[(r_mm, on)] = prob.mesh.n_cells
            jumps[(r_mm, on)] = float(np.percentile(_neighbour_size_ratio(prob.mesh), 99.9))
    assert counts[(0.025, True)] < 1.25 * counts[(0.1, True)]              # мелкое скругление не диктует сетку
    assert counts[(0.025, False)] > 2.0 * counts[(0.1, False)]             # а без правила — диктует
    for r_mm in (0.1, 0.025):
        assert jumps[(r_mm, True)] < 3.0 < jumps[(r_mm, False)]            # плавный рост против скачков


def test_preview_of_filleted_body_is_on_the_shape_and_does_not_grow_with_small_fillets(tmp_path):
    # Показ тела CAD триангулирует OpenCASCADE по отклонению хорды. Сетчик gmsh дробил плоские грани до
    # размера скругления (число треугольников ∝ 1/R²): сборка БПЛА32 — 1,25 млн треугольников и 33 с на показ,
    # на этих брусках — 51 тыс. при R = 0,4 мм и 840 тыс. при R = 0,1 мм (последняя проверка ниже тогда падает).
    # Оракул — брусок магнита 12 × 6 × 3 мм со скруглёнными рёбрами R: это сумма Минковского бруска
    # (12 − 2R) × (6 − 2R) × (3 − 2R) и шара R, знаковое расстояние до него — точная формула.
    gmsh = pytest.importorskip("gmsh")
    half = np.array([6.0, 3.0, 1.5]) * MM
    counts = []
    for r_mm in (0.4, 0.1, 0.025):
        path = str(tmp_path / f"fillet_{r_mm}.step")
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            occ = gmsh.model.occ
            box = occ.addBox(-6.0, -3.0, -1.5, 12.0, 6.0, 3.0)
            occ.synchronize()
            occ.fillet([box], [t for _, t in gmsh.model.getEntities(1)], [r_mm])
            occ.synchronize()
            gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            gmsh.write(path)
        finally:
            gmsh.finalize()
        r = r_mm * MM
        tris = preview_surfaces([cad("magnet", path, 0)])[0]
        q = np.abs(tris) - (half - r)                                        # (t, 3, 3)
        sdf = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0) - r
        assert np.abs(sdf).max() < 1e-10, r_mm                              # вершины — на поверхности
        c = tris.mean(axis=1)
        qc = np.abs(c) - (half - r)
        dev = np.abs(np.linalg.norm(np.maximum(qc, 0.0), axis=-1) + np.minimum(qc.max(axis=-1), 0.0) - r)
        # хорды отходят от поверхности меньше чем на 1/500 габарита: тело во весь вид — меньше пикселя
        assert dev.max() < 12.0 * MM / 500.0, r_mm
        counts.append(tris.shape[0])
    assert counts[2] <= counts[1] <= counts[0]                  # радиус меньше в 16 раз — треугольников не больше
