import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import quad

from magcore.constants import MU0
from magcore.domain.magnet_model import magnet_from_datasheet, n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    GeoObject3D,
    Problem3D,
    TetMesh3D,
    auto_domain3d,
    build_object_problem3d,
    coenergy,
    demag_summary,
    field_energy,
    flux_loss,
    flux_through_plane,
    force_weight,
    magnet_axial_coenergy,
    magnet_axial_flux,
    magnetic_force_torque,
    solve_linear3d,
    solve_nonlinear3d,
    steel_B_of_H,
    steel_coenergy,
)

# Этап 3D-4: сила, момент, энергия, поток, сводка риска. Оракулы:
#  (1) плотность коэнергии — точный интеграл ∫₀^H B dh тех же законов, что в решателе (сталь,
#      магнит с памятью, включая участки за коленом и за −H_cJ) — сверка с квадратурой;
#  (2) однородное поле: сила и момент на любое тело — ноль ТОЧНО (тензор Максвелла постоянен,
#      а ∮ n dS = 0 по замкнутой поверхности тела), поток сечения = μ₀H₀·n·S, энергия = ½μ₀H₀²V;
#  (3) виртуальная работа: сила и момент = производная коэнергии по сдвигу и повороту тела —
#      тождество дискретной задачи (сталь в насыщении, магнит за коленом), с точностью до O(δ²);
#  (4) жёсткий шар: поле внутри однородно, B = (2/3)μ₀M — поток через экваториальное сечение;
#      коэнергия системы −(2/9)πμ₀M²R³, и с границей φ = 0 дискретная коэнергия не меньше точной;
#  (5) сводка риска и признак «за H_cJ»: магнит, загнанный внешним полем за −H_cJ, — потеря
#      полная, помечены все ячейки; после снятия умеренного поля за коленом ничего, потеря осталась;
#  (6) два жёстких шара: поле снаружи однородно намагниченного шара — точно диполь, а сила на шар
#      в гармоническом поле — точно m·∇B в центре (теорема о среднем) ⇒ сила и момент между
#      шарами = формулы диполей ТОЧНО; сходимость к ним при измельчении сетки.
#  (7) брусок-магнит (этап 3D-6): у однородно намагниченного тела с μ_rec = μ⊥ = 1 поле в
#      точности равно полю поверхностных зарядов σ = M·n, а нормальная составляющая поля
#      равномерно заряженной прямоугольной грани = σ·Ω/(4π), где Ω — телесный угол грани ⇒
#      поток через среднее сечение, двумерный предел (L → ∞) и поправка на торцы известны точно.
#  (8) потеря потока — вердикт о размагничивании (Л-104): после события все ячейки на линиях возврата,
#      при 20 °C без поля задача линейна, и по взаимности (оператор поля симметричен) поток одного
#      магнита после события = Σ V·r·B∥_нов — потеря = средняя по магниту потерянная доля 1 − r с весом
#      индукции нового магнита B∥_нов. Отсюда: однородная r — потеря ровно 1 − r; одна ячейка весит
#      свою долю потока и не больше (V_c/V_м)·B_r/⟨B∥⟩; шар, выведенный полем за колено, — 1 − r_w точно.

MM = 1.0e-3
AIR = Air()
R = 5.0 * MM
CURVE = m270_35a_bh_curve()
RIGID = MagnetMaterial(magnet_from_datasheet("rigid", "rigid", (0.0, 0.0, 1.0), Br=1.2, Hcb=1.2 / MU0,
                                             Hk=1.1e6, Hcj=1.6e6))                  # μ_rec = μ⊥ = 1
M_RIGID = 1.2 / MU0


def _problem(objects, *, margin, h, T=20.0):
    pytest.importorskip("gmsh")
    dom = auto_domain3d(objects, material=AIR, margin_frac=margin)
    return build_object_problem3d(objects, dom, default_mesh_size=h, T=T)


def _mask(prob, name):
    rid = next(i for i, r in prob.regions.items() if r.name == name)
    return np.asarray(prob.cell_region) == rid


def _order(e_coarse, e_fine, ratio=2.0):
    return math.log(abs(e_coarse) / abs(e_fine)) / math.log(ratio)


def _integral(f, a, b, breaks):
    """∫_a^b f с изломами закона в точках `breaks` (квадратура по кускам)."""
    lo, hi = min(a, b), max(a, b)
    pts = sorted(x for x in breaks if lo < x < hi)
    val = quad(f, lo, hi, points=pts or None, limit=2000, epsabs=0.0, epsrel=1e-12)[0] if hi > lo else 0.0
    return val if b >= a else -val


# ----------------------------------------------------------------------- (1) плотность коэнергии
def test_coenergy_densities_are_exact_integrals_of_the_solver_laws():
    for H in (0.0, 30.0, 700.0, 5.0e4, 3.0e5, 2.0e6):                # до и за H_max таблицы
        ref = _integral(lambda h: float(steel_B_of_H(CURVE, h)), 0.0, H, CURVE.H_values)
        assert float(steel_coenergy(CURVE, H)) == pytest.approx(ref, rel=1e-9, abs=1e-12)
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    T = 100.0                                                            # колено −758, H_cJ −891 кА/м
    table = list(mag.curve_at(T).H_values)
    mr = MU0 * mag.mu_rec
    br = float(mag.Br(T))
    lo = table[0]                                                        # −H_cJ
    r_low = 0.5 * float(mag.retention_now(lo, T))    # ниже порога таблицы: потеря получена при другой T (Л-100)
    assert float(mag.switch_field(r_low, T)) == lo                       # предпосылка ветви
    # новый магнит; повреждённый полем −800 кА/м; полная потеря (ниже −H_cJ); доля ниже порога таблицы
    for r in (1.0, float(mag.retention_now(-800.0e3, T)), 0.0, r_low):
        hs = float(mag.switch_field(r, T))

        def law(x, r=r):
            return float(mag.B_major_parallel(x, T)) if float(mag.retention_now(x, T)) < r else r * br + mr * x

        for h in (1.0e5, -2.0e5, -5.0e5, -8.0e5, -8.5e5, -1.0e6):
            ref = _integral(law, 0.0, h, table + [hs])
            assert float(magnet_axial_coenergy(mag, T, h, r)) == pytest.approx(ref, rel=1e-9, abs=1e-9)


# ----------------------------------------------------------------------- (2) однородное поле
def test_uniform_field_force_torque_vanish_and_flux_energy_are_exact():
    H0 = np.array([100.0, -50.0, 30.0])
    b = GeoObject3D("b", "box", {"lx": 4 * MM, "ly": 3 * MM, "lz": 2 * MM}, LinearMaterial(1.0), mesh_size=1.0 * MM)
    p = _problem([b], margin=1.0, h=1.0 * MM)
    f = solve_linear3d(p, bc="dirichlet", applied_field=H0)       # поле однородно точно (3D-2)
    scale = MU0 * float(H0 @ H0) * 52 * MM ** 2                   # μ₀H₀² × площадь поверхности тела
    for kind in ("laplace", "layer"):
        ft = magnetic_force_torque(f, "b", weight=kind)
        assert np.linalg.norm(ft.force) < 1e-12 * scale
        assert np.linalg.norm(ft.torque) < 1e-12 * scale * 4 * MM
    # весовая функция итерационным решателем (для больших сеток) = прямым
    assert np.abs(force_weight(p, "b", solver="cg") - force_weight(p, "b")).max() < 1e-9
    with pytest.raises(ValueError):
        force_weight(p, "b", solver="lu")
    w_ex = 0.5 * MU0 * float(H0 @ H0) * 24 * MM ** 3
    assert field_energy(f, "b") == pytest.approx(w_ex, rel=1e-12)
    assert coenergy(f, "b") == pytest.approx(w_ex, rel=1e-12)
    n = np.array([0.1, 0.1, 1.0])
    nh = n / np.linalg.norm(n)
    area = 12 * MM ** 2 / nh[2]                  # косое сечение через центр пересекает только
    assert flux_through_plane(f, (0, 0, 0), n, objects="b") == pytest.approx(   # вертикальные рёбра
        MU0 * float(H0 @ nh) * area, rel=1e-12)
    # плоскость по грани тела (узлы на плоскости): грань считается один раз
    assert flux_through_plane(f, (0, 0, 1 * MM), (0, 0, 1), objects="b") == pytest.approx(
        MU0 * H0[2] * 12 * MM ** 2, rel=1e-12)


# ----------------------------------------------------------------------- (3) виртуальная работа
def test_force_and_torque_are_derivatives_of_the_discrete_coenergy():
    # Тело сдвигается на ±δ (узлы воздуха — на ±δθ) и поворачивается на ±α, задача решается заново;
    # центральная разность коэнергии обязана совпасть с формулой виртуальной работы. Сталь насыщена,
    # у магнита при 155 °C часть ячеек за коленом — в разность входят коэнергии всех законов.
    # Ошибка центральной разности ~ (δ/h)² = 10⁻⁶ — допуск 10⁻⁵.
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    h = 1.5 * MM
    cube = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, mag, magnet_dir="axial",
                       mesh_size=h)
    fe = GeoObject3D("fe", "box", {"lx": 8 * MM, "ly": 6 * MM, "lz": 3 * MM}, SteelMaterial(CURVE),
                     center=(1 * MM, 0.5 * MM, 7 * MM), rotation=(0.1, 0.0, 0.2), mesh_size=h)
    p = _problem([cube, fe], margin=2.0, h=h, T=155.0)
    f0 = solve_nonlinear3d(p)
    assert f0.converged and f0.risk.n_demagnetized > 0                               # предпосылки:
    assert np.linalg.norm(f0.B_cells[_mask(p, "fe")], axis=1).max() > 1.8             # колено, насыщение
    d = np.array([0.3, -0.4, 0.866])
    d /= np.linalg.norm(d)
    a = np.array([0.2, 0.9, -0.3])
    a /= np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    delta, alpha = 1.0e-3 * h, 1.0e-4

    def coenergy_moved(disp):
        mesh = TetMesh3D(p.mesh.vertices + disp, p.mesh.cells)
        f = solve_nonlinear3d(Problem3D(mesh, p.cell_region, p.regions, p.magnet_axis, p.T))
        assert f.converged
        return coenergy(f)

    def rotation(s):
        return np.eye(3) + math.sin(s) * K + (1.0 - math.cos(s)) * K @ K

    for kind in ("laplace", "layer"):
        theta = force_weight(p, "fe", kind=kind)
        ft = magnetic_force_torque(f0, "fe", weight=theta)
        r = p.mesh.vertices - ft.point
        dW = (coenergy_moved(delta * theta[:, None] * d)
              - coenergy_moved(-delta * theta[:, None] * d)) / (2.0 * delta)
        dWa = (coenergy_moved(theta[:, None] * (r @ rotation(alpha).T - r))
               - coenergy_moved(theta[:, None] * (r @ rotation(-alpha).T - r))) / (2.0 * alpha)
        assert abs(float(ft.force @ d) - dW) < 1e-5 * abs(dW), kind
        assert abs(float(ft.torque @ a) - dWa) < 1e-5 * abs(dWa), kind


# ----------------------------------------------------------------------- (4) жёсткий шар
def test_rigid_sphere_flux_and_coenergy():
    # Коэнергия: минимум функционала; с границей φ = 0 решение, продолженное нулём, — допустимая
    # функция задачи в свободном пространстве, поэтому дискретная коэнергия не меньше точной.
    flux_ex = 2.0 / 3.0 * MU0 * M_RIGID * math.pi * R ** 2
    w_ex = -2.0 / 9.0 * math.pi * MU0 * M_RIGID ** 2 * R ** 3
    e_flux, e_w = [], []
    for k in (3, 6):
        s = GeoObject3D("s", "sphere", {"r": R}, RIGID, magnet_dir="axial", mesh_size=R / k)
        p = _problem([s], margin=2.0, h=R / k)
        fl, w = [], []
        for bc in ("neumann", "dirichlet"):
            f = solve_linear3d(p, bc=bc)
            fl.append(flux_through_plane(f, (0, 0, 0), (0, 0, 1), objects="s"))
            w.append(coenergy(f))
        assert w[1] >= w_ex                                   # вариационная граница
        e_flux.append(0.5 * sum(fl) / flux_ex - 1.0)
        e_w.append(0.5 * sum(w) / w_ex - 1.0)
    assert _order(*e_flux) > 1.0      # сечение кусочно-постоянного поля: не хуже первого порядка
    assert _order(*e_w) > 1.5         # коэнергия — квадрат ошибки поля: второй порядок


# ----------------------------------------------------------------------- (5) сводка риска
def test_demag_summary_and_beyond_hcj_flag():
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    T = 100.0                                       # колено −758, H_cJ −891 кА/м
    s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / 3)
    p = _problem([s], margin=2.0, h=R / 3, T=T)
    # (а) внешнее поле −1500 кА/м: рабочая точка на продолжении за −H_cJ (−1446 кА/м) — модель там
    #     не определена, потеря принята полной, помечены все ячейки
    f = solve_nonlinear3d(p, bc="dirichlet", applied_field=(0.0, 0.0, -1500.0e3))
    n_mag = f.risk.cell_indices.size
    assert f.converged and f.risk.n_beyond_hcj == n_mag
    assert np.allclose(f.risk.loss, mag.Br(T), rtol=0.0, atol=1e-12)
    sm = demag_summary(f)["m"]
    assert sm.beyond_hcj_fraction == 1.0 and sm.damaged_fraction == 1.0 and sm.past_knee_fraction == 1.0
    assert sm.retained == pytest.approx(0.0, abs=1e-12)
    # (б) поле −550 кА/м приложили и сняли: сейчас за коленом ничего, потеря осталась и при снятии
    #     не выросла (ячейки идут по линии возврата)
    loaded = solve_nonlinear3d(p, bc="dirichlet", applied_field=(0.0, 0.0, -550.0e3))
    removed = solve_nonlinear3d(p, bc="dirichlet", retention=loaded.retention)
    a, b = demag_summary(loaded)["m"], demag_summary(removed)["m"]
    assert a.worst_margin < 0.0 < b.worst_margin and b.past_knee_fraction == 0.0
    assert b.damaged_fraction == pytest.approx(a.damaged_fraction, rel=1e-12) and a.damaged_fraction > 0.5
    assert b.retained == pytest.approx(a.retained, rel=1e-12) and 1.0 - b.retained > 1e-6
    assert a.beyond_hcj_fraction == b.beyond_hcj_fraction == 0.0 and removed.risk.n_damaged > 0
    # (в) сводка = прямой подсчёт по карте риска
    vol = removed.volumes[removed.risk.cell_indices]
    assert b.volume == pytest.approx(float(vol.sum()), rel=1e-12)
    assert b.retained == pytest.approx(float(np.average(removed.risk.Br_eff, weights=vol))
                                       / removed.risk.Br_nominal, rel=1e-12)
    assert b.max_loss == pytest.approx(float(removed.risk.loss.max()), rel=1e-12)


# ----------------------------------------------------------------------- (6) два жёстких шара
@pytest.mark.slow
def test_rigid_spheres_force_and_torque_match_point_dipoles():
    # Шары R = 5 мм, центры на оси z в ±d/2, d = 15 мм. Соосно (оба вдоль z) — притяжение
    # F = 3μ₀m²/(2πd⁴); поперёк (второй вдоль x): F₂ = 3μ₀m²/(4πd⁴)·x̂, моменты относительно центров
    # τ₂ = m₂×B₁ = −μ₀m²/(2πd³)·ŷ, τ₁ = −μ₀m²/(4πd³)·ŷ.
    # Запас домена 4: однородная часть поля «отражения» от стенок даёт момент ∝ (d/L)³, а силу лишь
    # ∝ (d/L)⁴ — при запасе 2 разрыв двух границ по моменту 1,6 %, при 4 — 0,25 % (Л-94). Среднее двух
    # границ. Порядок: гладкий (гармонический) вес — второй, один слой (градиент веса ~1/h) — не хуже
    # первого. Оценка Ричардсона (p = 2) при степенной ошибке лежит в пределах доли c последней
    # поправки: c = 1/3 при порядке ≥ 1,5, c = 2/3 при порядке ≥ 1. Баланс пары (действие —
    # противодействие, суммарный момент) — в пределах погрешности сетки.
    m = M_RIGID * 4.0 / 3.0 * math.pi * R ** 3
    d = 15.0 * MM
    exact = {"F2z": -3 * MU0 * m * m / (2 * math.pi * d ** 4),
             "F2x": 3 * MU0 * m * m / (4 * math.pi * d ** 4),
             "tau2y": -MU0 * m * m / (2 * math.pi * d ** 3),
             "tau1y": -MU0 * m * m / (4 * math.pi * d ** 3)}
    c1, c2 = np.array([0.0, 0.0, -d / 2]), np.array([0.0, 0.0, d / 2])
    res = {}
    for k in (3, 6):
        s1 = GeoObject3D("s1", "sphere", {"r": R}, RIGID, center=tuple(c1), magnet_dir="axial", mesh_size=R / k)
        s2 = GeoObject3D("s2", "sphere", {"r": R}, RIGID, center=tuple(c2), magnet_dir="axial", mesh_size=R / k)
        p = _problem([s1, s2], margin=4.0, h=R / k)
        axis = np.array(p.magnet_axis, dtype=float)
        axis[_mask(p, "s2")] = (1.0, 0.0, 0.0)                     # та же сетка, второй шар поперёк
        pp = dataclasses.replace(p, magnet_axis=axis)
        fields = [(solve_linear3d(p, bc=bc), solve_linear3d(pp, bc=bc)) for bc in ("neumann", "dirichlet")]
        for kind in ("laplace", "layer"):
            th1, th2 = force_weight(p, "s1", kind=kind), force_weight(p, "s2", kind=kind)
            r = {"F1z": 0.0, "F2z": 0.0, "F1": 0.0, "F2": 0.0, "tau1": 0.0, "tau2": 0.0}
            for fc, fp in fields:                                  # среднее двух границ
                r["F1z"] += 0.5 * magnetic_force_torque(fc, "s1", weight=th1).force[2]
                r["F2z"] += 0.5 * magnetic_force_torque(fc, "s2", weight=th2).force[2]
                pa = magnetic_force_torque(fp, "s1", weight=th1, point=c1)
                pb = magnetic_force_torque(fp, "s2", weight=th2, point=c2)
                r["F1"] += 0.5 * pa.force
                r["F2"] += 0.5 * pb.force
                r["tau1"] += 0.5 * pa.torque
                r["tau2"] += 0.5 * pb.torque
            res[(k, kind)] = r
    for kind, p_min, c_rich in (("laplace", 1.5, 1.0 / 3.0), ("layer", 1.0, 2.0 / 3.0)):
        crs, fin = res[(3, kind)], res[(6, kind)]
        pairs = {"F2z": (crs["F2z"], fin["F2z"]), "F2x": (crs["F2"][0], fin["F2"][0]),
                 "tau2y": (crs["tau2"][1], fin["tau2"][1]), "tau1y": (crs["tau1"][1], fin["tau1"][1])}
        for key, (vc, vf) in pairs.items():
            ex = exact[key]
            assert _order(vc / ex - 1.0, vf / ex - 1.0) > p_min, (kind, key)
            assert abs(vf + (vf - vc) / 3.0 - ex) <= c_rich * abs(vf - vc), (kind, key)
        assert abs(fin["F1z"] + fin["F2z"]) < abs(pairs["F2z"][1] - pairs["F2z"][0]), kind
        assert np.linalg.norm(fin["F1"] + fin["F2"]) < abs(pairs["F2x"][1] - pairs["F2x"][0]), kind
        total = fin["tau1"] + np.cross(c1, fin["F1"]) + fin["tau2"] + np.cross(c2, fin["F2"])
        assert np.linalg.norm(total) < abs(pairs["tau2y"][1] - pairs["tau2y"][0]), kind


# ----------------------------------------------------------------------- неверные входы
def _bar_h_y_exact(px, py, pz, *, width, thick, length, m_s):
    """
    H_y [А/м] от однородно намагниченного бруска width×thick×length (намагничен по +y,
    μ_rec = μ⊥ = 1) — ТОЧНО, полем поверхностных зарядов σ = ±M на гранях y = ±thick/2.
    Нормальная составляющая поля равномерно заряженного прямоугольника равна σ·Ω/(4π), телесный
    угол Ω прямоугольника — сумма арктангенсов. Арктангенс именно ГЛАВНОЙ ветви: с atan2 при
    отрицательном расстоянии до грани ветвь другая и знак поля переворачивается.
    """
    px, py, pz = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (px, py, pz))

    def solid_angle(y0):
        d = py - y0
        a = (-width / 2 - px, width / 2 - px)
        b = (-length / 2 - pz, length / 2 - pz)
        s = np.zeros(np.broadcast(px, py, pz).shape)
        for i in range(2):
            for j in range(2):
                den = d * np.sqrt(a[i] ** 2 + b[j] ** 2 + d ** 2)
                den = np.where(den == 0.0, np.finfo(float).tiny, den)   # точка в плоскости грани → ±π/2
                s = s + ((-1.0) ** (i + j)) * np.arctan(a[i] * b[j] / den)
        return s

    return m_s * (solid_angle(thick / 2) - solid_angle(-thick / 2)) / (4.0 * math.pi)


def _bar_flux_exact(*, width, thick, length, m_s, nx=48, nz=64):
    """
    Точный поток через среднее сечение бруска y = 0 [Вб]: ∫∫ μ₀(H_y + M) dx dz по сечению магнита.
    По z — составная квадратура Гаусса: у каждого торца свой отрезок длиной до пяти ширин (там
    поле и меняется), середина — своим, иначе у длинного бруска узлы проскакивают торцы.
    """
    def gauss(n, lo, hi):
        x, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * (hi - lo) * x + 0.5 * (lo + hi), 0.5 * (hi - lo) * w

    xs, wx = gauss(nx, -width / 2, width / 2)
    edge = min(5.0 * width, length / 2)
    bounds = [(-length / 2, -length / 2 + edge), (-length / 2 + edge, length / 2 - edge),
              (length / 2 - edge, length / 2)]
    parts = [gauss(nz, lo, hi) for lo, hi in bounds if hi - lo > 0.0]
    zs = np.concatenate([p[0] for p in parts])
    wz = np.concatenate([p[1] for p in parts])
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    by = MU0 * (_bar_h_y_exact(X.ravel(), np.zeros(X.size), Z.ravel(),
                               width=width, thick=thick, length=length, m_s=m_s) + m_s)
    return float(np.einsum("i,j,ij->", wx, wz, by.reshape(X.shape)))


def test_bar_magnet_end_correction_matches_the_exact_charge_model():
    """
    Оракул (7), этап 3D-6. Сначала точная формула проверяет сама себя на известных пределах
    (центр куба — размагничивающий множитель 1/3; очень широкая тонкая пластина — H = −M), затем
    поток через среднее сечение бруска сверяется с ней, а измельчение сетки обязано уменьшить
    ошибку не хуже второго порядка. Допуски — из измеренной сходимости (прогон 2026-09-16:
    +4,24 / +1,50 % при h = 0,70 / 0,50 мм), а не на глаз.
    """
    w, t, m_s = 10 * MM, 3 * MM, M_RIGID
    zero = np.zeros(1)
    n_cube = -_bar_h_y_exact(zero, zero, zero, width=w, thick=w, length=w, m_s=m_s)[0] / m_s
    plate = -_bar_h_y_exact(zero, zero, zero, width=1e4, thick=t, length=1e4, m_s=m_s)[0] / m_s
    assert abs(n_cube - 1.0 / 3.0) < 1e-12, "центр куба обязан дать размагничивающий множитель 1/3"
    assert abs(plate - 1.0) < 1e-6, "широкая тонкая пластина обязана дать H = −M"

    def fem_flux(length, h):
        """Поток через среднее сечение бруска нашим решателем; область — 3 ширины во все стороны."""
        pytest.importorskip("gmsh")
        bar = GeoObject3D("bar", "box", {"lx": w, "ly": t, "lz": length}, RIGID,
                          magnet_dir=(0.0, 1.0, 0.0), mesh_size=h)
        dom = GeoObject3D("domain", "box",
                          {"lx": w + 6 * w, "ly": t + 6 * w, "lz": length + 6 * w}, AIR, mesh_size=4 * MM)
        prob = build_object_problem3d([bar], dom, default_mesh_size=h)
        f = solve_linear3d(prob, bc="neumann", solver="cg", rtol=1e-12)
        return flux_through_plane(f, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), objects="bar")

    # Двумерный предел той же формулы: поток на единицу длины через среднюю линию (L → ∞).
    phi_2d = _bar_flux_exact(width=w, thick=t, length=1e4, m_s=m_s) / 1e4

    errors = {}
    for length in (5 * MM, 10 * MM):
        exact = _bar_flux_exact(width=w, thick=t, length=length, m_s=m_s)
        err = fem_flux(length, 0.7 * MM) / exact - 1.0
        errors[length] = err
        assert 0.0 < err < 0.06, (length, err)     # знак: скалярный потенциал завышает B (Л-91)
        k_exact = exact / (phi_2d * length)        # поправка на торцы: точные 1,892 (5 мм) и 1,515 (10 мм)
        assert abs(k_exact - (1.89219 if length == 5 * MM else 1.51527)) < 1e-4
    # «лишняя длина» (k − 1)·L растёт с длиной и стремится к постоянной (точно 5,98 мм при L → ∞)
    assert 4.4 * MM < (_bar_flux_exact(width=w, thick=t, length=5 * MM, m_s=m_s) / phi_2d - 5 * MM) < 4.5 * MM

    fine = fem_flux(5 * MM, 0.5 * MM) / _bar_flux_exact(width=w, thick=t, length=5 * MM, m_s=m_s) - 1.0
    # второй порядок при измельчении 0,70 → 0,50 мм дал бы 0,51 от прежней ошибки; измерено 0,35
    assert 0.0 < fine < 0.6 * errors[5 * MM]


# ----------------------------------------------------------------------- (8) потеря потока (Л-104)
def _magnet_under_iron(T):
    """Магнит 10×10×4 мм под линейным «железом» (μ_r = 1000) через зазор 1 мм: цепь линейна."""
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    box = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 4 * MM}, mag, magnet_dir="axial",
                      mesh_size=1.5 * MM)
    fe = GeoObject3D("fe", "box", {"lx": 12 * MM, "ly": 12 * MM, "lz": 2 * MM}, LinearMaterial(1000.0),
                     center=(0, 0, 4 * MM), mesh_size=1.5 * MM)
    return _problem([box, fe], margin=2.0, h=1.5 * MM, T=T)


def test_flux_loss_is_the_flux_weighted_mean_of_the_lost_fraction():
    # Тождество дискретной задачи. Доли r случайны (30 % ячеек, от 0 до 1); при 20 °C без поля ни одна
    # ячейка не уходит с линии возврата — задача линейна, и потеря = Σ V(1 − r)B∥_нов / Σ V·B∥_нов.
    # Допуск — от остановки решателя (невязка 10⁻⁹ от начальной); фактически совпадение ~10⁻¹⁶.
    p = _magnet_under_iron(T=150.0)                    # температура события в замер не входит
    m = p.magnet_mask()
    rng = np.random.default_rng(1)
    r = np.ones(p.mesh.n_cells)
    hit = m & (rng.random(p.mesh.n_cells) < 0.3)
    r[hit] = rng.uniform(0.0, 1.0, int(hit.sum()))
    cold = dataclasses.replace(p, T=20.0)
    for bc in ("neumann", "dirichlet"):
        new = solve_nonlinear3d(cold, bc=bc, solver="cg")
        after = solve_nonlinear3d(cold, bc=bc, solver="cg", retention=r)
        assert new.converged and after.converged and np.array_equal(after.retention, r)   # новой потери нет
        w = np.einsum("ck,ck->c", new.B_cells, p.magnet_axis)[m] * new.volumes[m]
        assert magnet_axial_flux(new) == {"m": pytest.approx(float(w.sum()), rel=1e-12)}
        expected = float(((1.0 - r[m]) * w).sum() / w.sum())
        assert flux_loss(p, r, bc=bc, new_flux=magnet_axial_flux(new))["m"] == pytest.approx(expected, abs=1e-8)
    # следствие: одна доля на весь магнит — поток ровно в r раз меньше нового
    assert flux_loss(p, np.where(m, 0.9, 1.0))["m"] == pytest.approx(0.1, abs=1e-8)


def test_single_damaged_cell_weighs_only_its_share_of_the_flux():
    # Л-104: одна ячейка вердикт не сдвигает. Ячейка c, потерявшая ремнантность целиком (r = 0), по
    # тождеству выше отнимает V_c·B∥_c / Σ V·B∥ потока. Поле нового магнита в ней против намагниченности
    # (H∥ < 0 ⇒ B∥_c < B_r), поэтому потеря меньше её доли объёма × B_r/⟨B∥⟩ и при измельчении сетки
    # уходит в ноль вместе с ячейкой. Ячейка — угловая у полюсной грани: у краёв поле меняется резче
    # всего, там и сидит «худшая ячейка».
    p = _magnet_under_iron(T=20.0)
    m = p.magnet_mask()
    new = solve_nonlinear3d(p, solver="cg")
    idx = np.where(m)[0]
    c = idx[np.argmax(np.abs(p.mesh.cell_centroids()[idx] / [5 * MM, 5 * MM, 2 * MM]).sum(axis=1))]
    b_par = np.einsum("ck,ck->c", new.B_cells, p.magnet_axis)
    assert float(new.H_cells[c] @ p.magnet_axis[c]) < 0.0                          # предпосылка
    r = np.ones(p.mesh.n_cells)
    r[c] = 0.0
    loss = flux_loss(p, r, new_flux=magnet_axial_flux(new))["m"]
    vol = new.volumes
    assert loss == pytest.approx(float(vol[c] * b_par[c] / (vol[m] * b_par[m]).sum()), abs=1e-8)
    share = float(vol[c] / vol[m].sum())
    mean_b = float(np.average(b_par[m], weights=vol[m]))
    assert 0.0 < loss < share * p.magnet().Br(20.0) / mean_b


def test_flux_loss_without_damage_is_zero_without_solving(monkeypatch):
    p = _magnet_under_iron(T=150.0)

    def no_solve(*args, **kwargs):
        raise AssertionError("без повреждения решать не нужно")

    monkeypatch.setattr("magcore.fem3d.postprocess.solve_nonlinear3d", no_solve)
    assert flux_loss(p, np.ones(p.mesh.n_cells)) == {"m": 0.0}
    air_only = _problem([GeoObject3D("b", "box", {"lx": 4 * MM, "ly": 4 * MM, "lz": 4 * MM}, LinearMaterial(1.0),
                                     mesh_size=2 * MM)], margin=1.0, h=2 * MM)
    assert flux_loss(air_only, np.ones(air_only.mesh.n_cells)) == {}                 # магнитов нет


@pytest.mark.slow
def test_flux_loss_of_a_sphere_driven_past_the_knee_matches_exact():
    # Шар N42SH при 100 °C во внешнем поле: внутреннее поле однородно, рабочая точка на главной кривой,
    # B_главн(H) + 2μ₀H = 3μ₀H₀. Поле H₀ — из модели, чтобы точка легла посередине между коленом и −H_cJ
    # (−824 кА/м): одна доля r_w = 0,777 на весь шар. Поле сняли, шар остыл до 20 °C: задача линейна,
    # источник ∝ r_w — поток ровно в r_w раз меньше нового, потеря 1 − r_w = 22,3 %. Второй порядок
    # сходимости (среднее по границам Неймана и Дирихле, как в других оракулах шара). У колена потеря
    # круто зависит от поля (главная кривая падает от колена до −H_cJ на 133 кА/м), поэтому на грубой
    # сетке ошибка велика: измерено −30 % и −8 % при R/3 и R/6.
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    T = 100.0
    h_in = 0.5 * (mag.knee_field(T) - mag.Hcj(T))
    H0 = (float(mag.B_major_parallel(h_in, T)) + 2.0 * MU0 * h_in) / (3.0 * MU0)
    loss_ex = 1.0 - float(mag.effective_Br(h_in, T)) / mag.Br(T)
    assert 0.1 < loss_ex < 0.5                           # предпосылка: потеря заметная, не полная
    errs = []
    for k in (3, 6):
        s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / k)
        p = _problem([s], margin=2.0, h=R / k, T=T)
        losses = []
        for bc in ("neumann", "dirichlet"):
            f = solve_nonlinear3d(p, bc=bc, applied_field=(0.0, 0.0, H0))
            assert f.converged and f.risk.n_beyond_hcj == 0
            losses.append(flux_loss(p, f.retention, bc=bc)["m"])
        errs.append(0.5 * sum(losses) / loss_ex - 1.0)
    assert _order(*errs) > 1.5
    assert abs(errs[1]) < 0.10                           # грубая проверка; точность задаёт порядок


def test_invalid_inputs_rejected():
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    cube = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 4 * MM}, mag, magnet_dir="axial",
                       mesh_size=2 * MM)
    plate = GeoObject3D("fe", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 2 * MM}, SteelMaterial(CURVE),
                        center=(0, 0, 3 * MM), mesh_size=2 * MM)                    # вплотную к магниту
    p = _problem([cube, plate], margin=1.0, h=2 * MM)
    f = solve_nonlinear3d(p)
    with pytest.raises(ValueError, match="касается другого тела"):
        magnetic_force_torque(f, "m")
    with pytest.raises(ValueError, match="нет объектов"):
        magnetic_force_torque(f, "nope")
    both = ["m", "fe"]                                                              # вместе — можно
    assert np.isfinite(magnetic_force_torque(f, both, weight="layer").force).all()
    for kw in (dict(weight="shell"), dict(weight=np.zeros(3)), dict(point=(1.0, 2.0)),
               dict(weight=0.5 * force_weight(p, both))):                          # θ ≠ 1 на теле
        with pytest.raises(ValueError):
            magnetic_force_torque(f, both, **kw)
    with pytest.raises(ValueError):
        flux_through_plane(f, (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        field_energy(f, "m")                                                        # энергия в магните
    with pytest.raises(ValueError, match="сохранённой доли"):
        flux_loss(p, None)
    with pytest.raises(ValueError):
        flux_loss(p, np.ones(3))
    with pytest.raises(ValueError, match="new_flux"):
        flux_loss(p, np.where(p.magnet_mask(), 0.5, 1.0), new_flux={"другой": 1.0})
    box = GeoObject3D("b", "box", {"lx": 4 * MM, "ly": 4 * MM, "lz": 4 * MM}, LinearMaterial(1.0),
                      center=(3 * MM, 0.0, 0.0), mesh_size=2 * MM)                  # грань на стенке
    dom = GeoObject3D("domain", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, AIR, mesh_size=2 * MM)
    fb = solve_linear3d(build_object_problem3d([box], dom, default_mesh_size=2 * MM))
    with pytest.raises(ValueError, match="внешней границы"):
        magnetic_force_torque(fb, "b")
