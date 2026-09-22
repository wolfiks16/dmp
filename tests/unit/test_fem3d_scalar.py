import math

import numpy as np
import pytest
from scipy.integrate import dblquad

from magcore.constants import MU0
from magcore.domain.magnet_model import magnet_from_datasheet, n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    evaluate_phi,
    solve_linear3d,
)

# Этап 3D-2: линейная магнитостатика на полном скалярном потенциале. Оракулы:
#  (1) однородное поле воспроизводится ТОЧНО (линейный потенциал лежит в пространстве элементов);
#  (2) обрезка домена: разрыв между границами «поток не выходит» и «φ = 0» убывает как (R/L)³ —
#      дипольное поле; граница φ = 0 и дискретизация обе ослабляют размагничивание ⇒ результат
#      с ней — нижняя граница |H| в магните;
#  (3) второй порядок сходимости средних полей и разностей потенциалов при измельчении всей
#      сетки (намагниченный шар H = −B_r/(μ₀(2+μ_rec)), проницаемый шар 3H₀/(μ+2), жёсткие
#      цилиндр и параллелепипед — потенциал поверхностных зарядов: формула дисков / численный
#      интеграл, сам интеграл сверен с формулой поля на оси);
#  (4) тензор проницаемости магнита: поперечный отклик = отклик изотропного шара μ⊥;
#  (5) две независимые формулировки (скалярный потенциал и векторный из старого ядра) на одной
#      сетке берут решение в вилку — вариационная двусторонняя оценка — и вилка сужается.

MM = 1.0e-3
AIR = Air()
R = 5.0 * MM
MAG = n42sh_magnet((0.0, 0.0, 1.0))
RIGID = magnet_from_datasheet("rigid", "rigid", (0.0, 0.0, 1.0), Br=1.2, Hcb=1.2 / MU0,
                              Hk=1.1e6, Hcj=1.6e6)                      # μ_rec = μ⊥ = 1
M_RIGID = 1.2 / MU0


def _problem(objects, *, margin, h):
    pytest.importorskip("gmsh")
    dom = auto_domain3d(objects, material=AIR, margin_frac=margin)
    return build_object_problem3d(objects, dom, default_mesh_size=h)


def _mask(prob, name):
    rid = next(i for i, r in prob.regions.items() if r.name == name)
    return np.asarray(prob.cell_region) == rid


def _both(prob, **kw):
    return {bc: solve_linear3d(prob, bc=bc, **kw) for bc in ("neumann", "dirichlet")}


def _order(e_coarse, e_fine, ratio=2.0):
    return math.log(abs(e_coarse) / abs(e_fine)) / math.log(ratio)


def _sphere_magnet(h, magnet=MAG, direction="axial"):
    return GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(magnet), magnet_dir=direction, mesh_size=h)


# ----------------------------------------------------------------------- (1) точность на линейном
def test_uniform_field_is_reproduced_exactly():
    box = GeoObject3D("b", "box", {"lx": 4 * MM, "ly": 3 * MM, "lz": 2 * MM}, LinearMaterial(1.0),
                      rotation=(0.3, 0.2, 0.1), mesh_size=1.2 * MM)
    prob = _problem([box], margin=1.0, h=1.2 * MM)
    H0 = np.array([100.0, -50.0, 30.0])
    for bc, f in _both(prob, applied_field=H0).items():
        assert f.residual < 1e-10
        assert np.abs(f.H_cells - H0).max() / np.linalg.norm(H0) < 1e-10, bc
        rng = np.random.default_rng(1)
        pts = rng.uniform(-3, 3, size=(20, 3)) * MM
        assert np.allclose(evaluate_phi(f, pts), -pts @ H0, rtol=0.0, atol=1e-12 * np.linalg.norm(H0))
    with pytest.raises(ValueError):
        evaluate_phi(f, np.array([[1.0, 1.0, 1.0]]))                     # точка вне сетки


def test_cg_matches_direct_and_sign_symmetry():
    prob = _problem([_sphere_magnet(R / 3)], margin=1.0, h=R / 3)
    d = solve_linear3d(prob, bc="neumann", solver="direct")
    c = solve_linear3d(prob, bc="neumann", solver="cg", rtol=1e-12)
    assert np.linalg.norm(c.H_cells - d.H_cells) / np.linalg.norm(d.H_cells) < 1e-6
    rev = _problem([_sphere_magnet(R / 3, direction="axial-in")], margin=1.0, h=R / 3)
    r = solve_linear3d(rev, bc="neumann")
    assert np.allclose(r.H_cells, -d.H_cells, rtol=0.0, atol=1e-9 * np.abs(d.H_cells).max())
    air = _problem([GeoObject3D("a", "box", {"lx": 2 * MM, "ly": 2 * MM, "lz": 2 * MM}, AIR)],
                   margin=1.0, h=1.0 * MM)
    assert np.abs(solve_linear3d(air).H_cells).max() == 0.0             # нет источников — нет поля


def test_invalid_inputs_rejected():
    fe = GeoObject3D("fe", "box", {"lx": 2 * MM, "ly": 2 * MM, "lz": 2 * MM}, SteelMaterial(m270_35a_bh_curve()))
    prob = _problem([fe], margin=1.0, h=1.0 * MM)
    with pytest.raises(NotImplementedError):
        solve_linear3d(prob)                                            # нелинейная сталь — этап 3D-3
    air = _problem([GeoObject3D("a", "box", {"lx": 2 * MM, "ly": 2 * MM, "lz": 2 * MM}, AIR)],
                   margin=1.0, h=1.0 * MM)
    for kw in (dict(bc="robin"), dict(solver="lu"), dict(applied_field=(1.0, 2.0))):
        with pytest.raises(ValueError):
            solve_linear3d(air, **kw)


# ----------------------------------------------------------------------- (2) обрезка домена
def test_truncation_gap_decays_as_dipole_and_dirichlet_is_lower_bound():
    exact = -(MAG.Br(20.0) / MU0) / (2.0 + MAG.mu_rec)
    gaps = {}
    for margin in (1.0, 2.0):
        prob = _problem([_sphere_magnet(R / 4)], margin=margin, h=R / 4)
        f = _both(prob)
        hz = {bc: f[bc].average(f[bc].H_cells, _mask(prob, "m"))[2] for bc in f}
        gaps[margin] = (hz["neumann"] - hz["dirichlet"]) / exact
        assert gaps[margin] > 0.0                       # «поток не выходит» размагничивает сильнее
        assert hz["dirichlet"] / exact < 1.0            # φ = 0 и дискретизация — обе ослабляют
    # половина ребра домена L = R·(1 + 2·запас): разрыв ∝ L⁻³ (показатель 3 ± 10 % на
    # предасимптотику и различие сеток)
    p = math.log(gaps[1.0] / gaps[2.0]) / math.log(5.0 / 3.0)
    assert 2.7 < p < 3.3


# ----------------------------------------------------------------------- (3) сходимость
def test_magnetized_sphere_converges_second_order():
    exact = -(MAG.Br(20.0) / MU0) / (2.0 + MAG.mu_rec)
    errs = []
    for k in (3, 6):
        prob = _problem([_sphere_magnet(R / k)], margin=2.0, h=R / k)
        f = _both(prob)
        mid = 0.5 * sum(f[bc].average(f[bc].H_cells, _mask(prob, "m"))[2] for bc in f)
        errs.append((mid - exact) / exact)
        assert errs[-1] < 0.0                            # скалярный потенциал недооценивает |H|
    assert _order(*errs) > 1.6                           # теория: 2 (средние по объёму)
    assert abs(errs[1]) < 0.05                           # грубая проверка вменяемости


def test_permeable_sphere_in_uniform_field_converges_second_order():
    H0 = np.array([1000.0, 0.0, 0.0])
    exact = 3.0 * H0[0] / (10.0 + 2.0)
    errs = []
    for k in (3, 6):
        s = GeoObject3D("s", "sphere", {"r": R}, LinearMaterial(10.0), mesh_size=R / k)
        prob = _problem([s], margin=4.0, h=R / k)
        f = solve_linear3d(prob, bc="dirichlet", applied_field=H0)
        errs.append(f.average(f.H_cells, _mask(prob, "s"))[0] / exact - 1.0)
        assert errs[-1] > 0.0          # дискретная сфера «пропускает» поле хуже точной ⇒ H_in выше
    assert _order(*errs) > 1.6
    assert abs(errs[1]) < 0.05


def test_rigid_cylinder_axis_potential_converges():
    # Потенциал поверхностных зарядов ±M на торцах — формула диска на оси.
    Lc = 10.0 * MM

    def phi_axis(z):
        c = Lc / 2
        return 0.5 * M_RIGID * ((math.hypot(R, z - c) - abs(z - c)) - (math.hypot(R, z + c) - abs(z + c)))

    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 7.5 * MM]])
    dphi_exact = phi_axis(0.0) - phi_axis(7.5 * MM)
    hz_exact = -M_RIGID * (1.0 - (Lc / 2) / math.hypot(R, Lc / 2))
    e_phi, e_hz = [], []
    for k in (3, 6):
        cyl = GeoObject3D("cyl", "cylinder", {"r": R, "h": Lc}, MagnetMaterial(RIGID),
                          magnet_dir="axial", mesh_size=R / k)
        prob = _problem([cyl], margin=2.0, h=R / k)
        core = _mask(prob, "cyl") & (np.linalg.norm(prob.mesh.cell_centroids(), axis=1) < 1.5 * MM)
        f = _both(prob)
        dphi = 0.5 * sum(np.subtract(*evaluate_phi(f[bc], pts)) for bc in f)
        hz = 0.5 * sum(f[bc].average(f[bc].H_cells, core)[2] for bc in f)
        e_phi.append(dphi / dphi_exact - 1.0)
        e_hz.append(hz / hz_exact - 1.0)
    assert _order(*e_phi) > 1.5 and _order(*e_hz) > 1.5
    assert abs(e_phi[1]) < 0.05 and abs(e_hz[1]) < 0.05


def test_rigid_cuboid_matches_surface_charge_integral():
    a, b, c = 5.0 * MM, 3.0 * MM, 2.0 * MM                           # полуразмеры

    def phi_exact(p):
        x, y, z = p

        def face(zf):
            return dblquad(lambda yy, xx: 1.0 / math.sqrt((x - xx) ** 2 + (y - yy) ** 2 + (z - zf) ** 2),
                           -a, a, lambda xx: -b, lambda xx: b, epsabs=1e-14, epsrel=1e-11)[0]
        return M_RIGID / (4.0 * math.pi) * (face(c) - face(-c))

    # сам эталон: производная интеграла на оси = известная формула поля прямоугольного магнита
    zt, dz = 3.0 * c, 1.0e-6
    bz_int = -MU0 * (phi_exact((0, 0, zt + dz)) - phi_exact((0, 0, zt - dz))) / (2 * dz)
    bz_formula = 1.2 / math.pi * (
        math.atan(a * b / ((zt - c) * math.sqrt((zt - c) ** 2 + a * a + b * b)))
        - math.atan(a * b / ((zt + c) * math.sqrt((zt + c) ** 2 + a * a + b * b))))
    assert abs(bz_int / bz_formula - 1.0) < 1e-6

    P = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3 * c], [1.2 * a, 0.5 * b, 2.5 * c]])
    ex = np.array([phi_exact(p) for p in P])
    errs = {"ось": [], "вбок": []}
    for k in (3, 6):
        cub = GeoObject3D("cub", "box", {"lx": 2 * a, "ly": 2 * b, "lz": 2 * c}, MagnetMaterial(RIGID),
                          magnet_dir="axial", mesh_size=2 * c / k)
        prob = _problem([cub], margin=2.0, h=2 * c / k)
        f = _both(prob)
        ph = 0.5 * sum(evaluate_phi(f[bc], P) for bc in f)
        errs["ось"].append((ph[0] - ph[1]) / (ex[0] - ex[1]) - 1.0)
        errs["вбок"].append((ph[0] - ph[2]) / (ex[0] - ex[2]) - 1.0)
    for key, e in errs.items():
        assert _order(*e) > 1.5, key
        assert abs(e[1]) < 0.05, key


# ----------------------------------------------------------------------- (4) тензор
def test_anisotropic_magnet_uses_perpendicular_permeability():
    # Поперечный отклик (решение с полем минус без поля — линейность) обязан совпасть с откликом
    # изотропного шара μ = μ⊥ на той же сетке. Ошибка «μ_rec вместо μ⊥» дала бы 3/3,05 против
    # 3/3,5 — 15 %; допуск 0,5 % в 30 раз меньше и с запасом выше численной связи осей.
    aniso = magnet_from_datasheet("an", "an", (0, 0, 1), Br=1.2, Hcb=1.2 / MU0 / 1.05,
                                  Hk=1.1e6, Hcj=1.6e6, mu_perp=1.5)
    iso_axial = magnet_from_datasheet("ia", "ia", (0, 0, 1), Br=1.2, Hcb=1.2 / MU0 / 1.05,
                                      Hk=1.1e6, Hcj=1.6e6)                       # μ⊥ = μ_rec
    H0 = np.array([1000.0, 0.0, 0.0])
    pa = _problem([_sphere_magnet(R / 4, magnet=aniso)], margin=2.0, h=R / 4)
    pi_ = _problem([GeoObject3D("m", "sphere", {"r": R}, LinearMaterial(1.5), mesh_size=R / 4)],
                   margin=2.0, h=R / 4)
    pia = _problem([_sphere_magnet(R / 4, magnet=iso_axial)], margin=2.0, h=R / 4)
    assert pa.mesh.n_cells == pi_.mesh.n_cells == pia.mesh.n_cells              # одна и та же сетка
    m = _mask(pa, "m")
    f0 = solve_linear3d(pa)
    resp = solve_linear3d(pa, applied_field=H0)
    dH = resp.average(resp.H_cells - f0.H_cells, m)
    fi = solve_linear3d(pi_, applied_field=H0)
    iso_x = fi.average(fi.H_cells, _mask(pi_, "m"))[0]
    assert abs(dH[0] / iso_x - 1.0) < 5e-3
    assert abs(dH[2]) < 1e-3 * abs(dH[0])                                        # оси не смешиваются
    fa = solve_linear3d(pia)
    hz_iso = fa.average(fa.H_cells, _mask(pia, "m"))[2]
    assert abs(f0.average(f0.H_cells, m)[2] / hz_iso - 1.0) < 5e-3              # осевой — только μ∥


# ----------------------------------------------------------------------- (5) вилка двух формулировок
@pytest.mark.slow
def test_scalar_and_vector_potential_bracket_the_solution():
    """
    Одна и та же обрезанная задача (шар-магнит в коробке, B·n = 0 на границе) на одной и той же
    сетке двумя формулировками: узловой скалярный потенциал (точно ∇×H = 0) и рёберный векторный
    потенциал старого ядра (точно ∇·B = 0). Вариационно первая недооценивает |H| в магните,
    вторая переоценивает ⇒ точное решение между ними; при измельчении вилка сужается.
    """
    from magcore.femcore.assembly import assemble_magnetization_rhs, assemble_mixed_coulomb_system
    from magcore.femcore.boundary_conditions import apply_zero_mixed_dirichlet_bc, find_mixed_boundary_dofs
    from magcore.femcore.post import evaluate_curl_on_cell
    from magcore.femcore.scalar_spaces import LagrangeP1Space
    from magcore.femcore.solver import solve_mixed_coulomb_problem, split_mixed_solution
    from magcore.femcore.spaces import NedelecP1Space

    br, mu = MAG.Br(20.0), MAG.mu_rec
    ratios = []
    for h, dom_h in ((R / 2.0, R / 1.2), (R / 2.6, R / 1.5)):
        pytest.importorskip("gmsh")
        sph = _sphere_magnet(h)
        dom = auto_domain3d([sph], material=AIR, margin_frac=0.6, mesh_size=dom_h)
        prob = build_object_problem3d([sph], dom, default_mesh_size=h)
        m = _mask(prob, "m")
        vol = prob.mesh.cell_volumes()
        fs = solve_linear3d(prob, bc="neumann")
        hs = MU0 * fs.average(fs.H_cells, m)[2]
        old = prob.mesh.to_tetra_mesh()
        vs, ss = NedelecP1Space.from_mesh(old), LagrangeP1Space(old)
        nu = np.where(m, 1.0 / mu, 1.0)
        nu_br = np.zeros((old.n_cells, 3))
        nu_br[m] = (br / mu) * np.asarray(prob.magnet_axis)[m]
        A, rhs = assemble_mixed_coulomb_system(old, vs, ss, nu, lambda x: np.zeros(3),
                                               extra_vector_rhs=assemble_magnetization_rhs(old, vs, nu_br))
        vb, sb = find_mixed_boundary_dofs(vs, ss)
        A_bc, b_bc = apply_zero_mixed_dirichlet_bc(A, rhs, vb, sb, vs.ndofs)
        a, _ = split_mixed_solution(solve_mixed_coulomb_problem(A_bc, b_bc), vs.ndofs)
        Bc = np.array([evaluate_curl_on_cell(vs, a, c) for c in range(old.n_cells)])
        hv = ((nu[m] * Bc[m, 2] - nu_br[m, 2]) * vol[m]).sum() / vol[m].sum()   # μ₀H = νB − νB_r
        assert abs(hs) < abs(hv)                                  # вилка
        ratios.append(abs(hs) / abs(hv))
    assert ratios[1] > ratios[0]                                  # и она сужается
