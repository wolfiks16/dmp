import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import magnet_from_datasheet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, MagnetMaterial, SteelMaterial
from magcore.fem3d import GeoObject3D, auto_domain3d, build_object_problem3d, solve_linear3d, solve_nonlinear3d
from magcore.fem3d.probes import body_mean, circle, circle_basis, line_points, sample, steel_saturation

# ИЗМЕРЕНИЯ ПО РЕШЕНИЮ 3D (точка, линия, окружность, среднее по телу, насыщение стали). Оракулы:
#  (1) однородное поле решается ТОЧНО (линейный потенциал лежит в пространстве элементов) — значит, в любой точке,
#      на линии и на окружности B = μ₀H₀ до машинной точности; на окружности с нормалью Z и полем вдоль X
#      B_r = B₀ cos θ, B_θ = −B₀ sin θ, B_n = 0;
#  (2) точка вне сетки — NaN и пустое имя тела, а не ближайшая ячейка;
#  (3) среднее по телу — по определению (среднее с весом объёма тех же ячеек); у жёсткого магнита (μ_rec = μ⊥ = 1)
#      закон B = μ₀(H + M) линеен, поэтому держится и в среднем: B_d = μ₀(H_d + M) — средние B и H взяты из одних
#      ячеек и вдоль одной оси. Шар: точно P_c = 2 (размагничивающий фактор 1/3). Граница φ = 0 и сетка обе
#      ОСЛАБЛЯЮТ размагничивание (оракул (2) в test_fem3d_scalar.py) — поэтому с границей φ = 0 P_c ≥ 2 строго, а при
#      измельчении сетки вдвое ошибка убывает (измерено: 0,293 → 0,090 при h = R/4 → R/8, порядок 1,7);
#  (4) насыщение стали — по определению; порог 0 — вся сталь, порог выше наибольшей |B| — ни одной ячейки.

MM = 1.0e-3
AIR = Air()
RIGID = magnet_from_datasheet("rigid", "rigid", (0.0, 0.0, 1.0), Br=1.2, Hcb=1.2 / MU0, Hk=1.1e6, Hcj=1.6e6)
M_RIGID = 1.2 / MU0
H0 = np.array([30.0e3, -10.0e3, 5.0e3])                    # приложенное поле [А/м]


def _problem(objects, *, margin=1.0, h=2.0 * MM):
    pytest.importorskip("gmsh")
    return build_object_problem3d(objects, auto_domain3d(objects, material=AIR, margin_frac=margin), default_mesh_size=h)


def _uniform():
    p = _problem([GeoObject3D("куб", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, AIR)])
    return solve_linear3d(p, bc="neumann", applied_field=H0)


def test_point_line_and_circle_reproduce_a_uniform_field_exactly():
    f = _uniform()
    B0 = MU0 * H0
    rng = np.random.default_rng(1)
    pts = rng.uniform(-6 * MM, 6 * MM, size=(40, 3))
    s = sample(f, pts)
    assert (s.cells >= 0).all()
    assert np.abs(s.B - B0).max() < 1e-12 * np.linalg.norm(B0)
    assert np.abs(s.H - H0).max() < 1e-12 * np.linalg.norm(H0)
    # сетка повторяет грани куба точно — точка внутри куба лежит в его ячейке, снаружи — в области «domain»
    inside = (np.abs(pts) < 5 * MM).all(axis=1)
    assert inside.any() and (~inside).any()
    assert [b for b, i in zip(s.body, inside)] == ["куб" if i else "domain" for i in inside]

    lp, dist = line_points((-5 * MM, 0.0, 0.0), (5 * MM, 3 * MM, -2 * MM), 25)
    assert dist[0] == 0.0 and dist[-1] == pytest.approx(np.linalg.norm([10 * MM, 3 * MM, -2 * MM]), rel=1e-15)
    assert np.abs(sample(f, lp).B - B0).max() < 1e-12 * np.linalg.norm(B0)


def test_circle_components_of_a_uniform_field_are_cos_and_sin():
    p = _problem([GeoObject3D("куб", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, AIR)])
    Hx = np.array([40.0e3, 0.0, 0.0])
    f = solve_linear3d(p, bc="neumann", applied_field=Hx)
    B0 = MU0 * Hx[0]
    c = circle(f, (0.0, 0.0, 1 * MM), (0.0, 0.0, 1.0), 4 * MM, 72)
    assert np.abs(c.Br - B0 * np.cos(c.theta)).max() < 1e-12 * B0
    assert np.abs(c.Bt + B0 * np.sin(c.theta)).max() < 1e-12 * B0
    assert np.abs(c.Bn).max() < 1e-12 * B0
    # базис плоскости — правый, для осей — как в 2D (нормаль Z: θ от +X к +Y)
    for nrm in ((0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 2, 2)):
        n, e1, e2 = circle_basis(nrm)
        assert np.allclose(np.cross(e1, e2), n) and abs(e1 @ n) < 1e-15 and abs(e1 @ e2) < 1e-15
    assert np.allclose(circle_basis((0, 0, 1))[1], (1, 0, 0)) and np.allclose(circle_basis((0, 0, 1))[2], (0, 1, 0))


def test_points_outside_the_mesh_are_nan_not_the_nearest_cell():
    f = _uniform()
    far = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])        # первая — в метре от модели
    s = sample(f, far)
    assert s.cells[0] == -1 and np.isnan(s.B[0]).all() and s.body[0] == ""
    assert s.cells[1] >= 0 and np.isfinite(s.B[1]).all()
    with pytest.raises(ValueError):
        sample(f, [[0.0, 0.0]])


def test_body_mean_is_the_volume_mean_of_the_same_cells():
    f = _uniform()
    m = body_mean(f, "куб")
    rid = next(i for i, r in f.problem.regions.items() if r.name == "куб")
    assert m.volume == pytest.approx(float(f.problem.region_volumes()[rid]), rel=1e-12)
    assert np.allclose(m.B_mean, MU0 * H0, rtol=1e-12, atol=0) and np.allclose(m.H_mean, H0, rtol=1e-12, atol=0)
    assert m.magnet is False and m.B_par is None and m.permeance is None
    with pytest.raises(ValueError):
        body_mean(f, "нет такого")


def _ball_pc(h):
    ball = GeoObject3D("шар", "sphere", {"r": 5 * MM}, MagnetMaterial(RIGID), magnet_dir="axial", mesh_size=h)
    f = solve_linear3d(_problem([ball], margin=2.0, h=2 * h), bc="dirichlet")
    return f, body_mean(f, "шар")


def test_rigid_magnet_operating_point_follows_its_linear_law_on_average():
    f, m = _ball_pc(1.25 * MM)
    sel = np.asarray(f.problem.cell_region) == next(i for i, r in f.problem.regions.items() if r.name == "шар")
    e = np.asarray(f.problem.magnet_axis)[sel]
    v = f.volumes[sel]
    assert m.B_par == pytest.approx(float((np.einsum("ij,ij->i", f.B_cells[sel], e) * v).sum() / v.sum()), rel=1e-12)
    assert m.B_par == pytest.approx(MU0 * (m.H_par + M_RIGID), rel=1e-9)        # закон в среднем
    assert m.B_par > 0.0 > m.H_par and m.permeance == pytest.approx(m.B_par / (MU0 * abs(m.H_par)), rel=1e-12)
    # шар: P_c = 2 точно; с границей φ = 0 ошибка только сверху и убывает при измельчении сетки вдвое
    _, fine = _ball_pc(0.625 * MM)
    e_coarse, e_fine = m.permeance - 2.0, fine.permeance - 2.0
    assert e_coarse > 0.0 and e_fine > 0.0
    assert np.log2(e_coarse / e_fine) > 1.0


def test_steel_saturation_by_definition_and_at_the_extreme_thresholds():
    plate = GeoObject3D("пластина", "box", {"lx": 12 * MM, "ly": 12 * MM, "lz": 2 * MM}, SteelMaterial(m270_35a_bh_curve()))
    f = solve_nonlinear3d(_problem([plate], h=2.0 * MM), bc="neumann", applied_field=(0.0, 0.0, 200.0e3))
    (s,) = steel_saturation(f, 1.0)
    sel = np.asarray(f.problem.cell_region) == next(i for i, r in f.problem.regions.items() if r.name == "пластина")
    b, v = np.linalg.norm(f.B_cells[sel], axis=1), f.volumes[sel]
    assert s.name == "пластина" and s.B_max == pytest.approx(float(b.max()), rel=1e-15)
    assert s.fraction_above == pytest.approx(float(v[b > 1.0].sum() / v.sum()), rel=1e-12)
    assert steel_saturation(f, 0.0)[0].fraction_above == 1.0
    assert steel_saturation(f, s.B_max + 0.01)[0].fraction_above == 0.0
    with pytest.raises(ValueError):
        steel_saturation(f, -1.0)
