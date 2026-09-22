import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    solve_linear3d,
    solve_nonlinear3d,
    steel_B_of_H,
    steel_mu_rel,
)

# Этап 3D-3: нелинейная 3D-магнитостатика (сталь с насыщением, магнит с коленом). Оракулы:
#  (1) материал стали в 3D тот же, что в 2D: обратная ломаная B(H) совпадает с кривой до
#      машинной точности;
#  (2) линейные случаи решаются ТОЧНО: сталь на первом (линейном) отрезке кривой = линейная
#      задача с начальной проницаемостью; задача без стали и выше колена = линейный решатель;
#  (3) насыщенный стальной шар в однородном поле: внутреннее поле однородно и подчиняется
#      B(H_in) + 2μ₀H_in = 3μ₀H₀ — корень одномерного уравнения; второй порядок сходимости;
#      Ньютон: число итераций от сетки не зависит;
#  (4) магнитный шар за коленом: рабочая точка — пересечение главной кривой размагничивания
#      с нагрузочной прямой шара B = −2μ₀H; второй порядок сходимости, потеря ремнантности есть;
#  (5) замыкатель: стальная пластина на полюсе уменьшает размагничивающее поле (добавление
#      проницаемого материала не может уменьшить проводимость пути потока);
#  (6) история нагружения: шар во внешнем поле уходит за колено (рабочая точка — корень
#      B_главн(H) + 2μ₀H = 3μ₀H₀), поле сняли — магнит возвращается по линии возврата из
#      наихудшей точки до нагрузочной прямой шара: H = −B_r,eff(H_w)/(μ₀(2 + μ_rec)); второй
#      порядок сходимости. На одной сетке — дискретные тождества: после снятия поля новой потери
#      нет, то же поле снова даёт то же состояние;
#  (7) частичное размагничивание куба: размагничивающее поле жёсткого куба (телесные углы
#      заряженных полюсных граней) сильнее всего под полюсными гранями, в центре −M/3 — при
#      нагреве за колено уходит слой под обеими гранями, центр цел, слой растёт с температурой.

MM = 1.0e-3
AIR = Air()
R = 5.0 * MM
CURVE = m270_35a_bh_curve()


def _problem(objects, *, margin, h, T=20.0):
    pytest.importorskip("gmsh")
    dom = auto_domain3d(objects, material=AIR, margin_frac=margin)
    return build_object_problem3d(objects, dom, default_mesh_size=h, T=T)


def _mask(prob, name):
    rid = next(i for i, r in prob.regions.items() if r.name == name)
    return np.asarray(prob.cell_region) == rid


def _order(e_coarse, e_fine, ratio=2.0):
    return math.log(abs(e_coarse) / abs(e_fine)) / math.log(ratio)


# ----------------------------------------------------------------------- (1) материал стали
def test_vectorized_steel_curve_matches_the_curve():
    B = np.random.default_rng(0).uniform(1e-3, 2.6, 300)          # включая насыщение за B_max
    H = np.array([CURVE.H_of_B(b) for b in B])
    assert np.allclose(steel_B_of_H(CURVE, H), B, rtol=1e-12, atol=0.0)
    mu_c, mu_d = steel_mu_rel(CURVE, H)
    assert np.allclose(mu_c * MU0 * np.array([CURVE.nu_chord(b) for b in B]), 1.0, rtol=1e-12)
    assert np.allclose(mu_d * MU0 * np.array([CURVE.nu_differential(b) for b in B]), 1.0, rtol=1e-12)
    mu_c0, mu_d0 = steel_mu_rel(CURVE, np.array([0.0]))           # предел в нуле — первый отрезок
    assert np.isclose(mu_c0[0], mu_d0[0]) and np.isclose(mu_c0[0] * MU0, 1.0 / CURVE.nu_initial)


# ----------------------------------------------------------------------- (2) линейные случаи
def test_weak_field_steel_is_exactly_linear():
    mu_init = CURVE.B_values[1] / (MU0 * CURVE.H_values[1])
    box = {"lx": 10 * MM, "ly": 6 * MM, "lz": 4 * MM}
    ps = _problem([GeoObject3D("fe", "box", box, SteelMaterial(CURVE), mesh_size=2 * MM)], margin=1.0, h=2 * MM)
    pl = _problem([GeoObject3D("fe", "box", box, LinearMaterial(mu_init), mesh_size=2 * MM)], margin=1.0, h=2 * MM)
    H0 = np.array([5.0, 0.0, 0.0])
    fs = solve_nonlinear3d(ps, bc="dirichlet", applied_field=H0)
    fl = solve_linear3d(pl, bc="dirichlet", applied_field=H0)
    assert np.linalg.norm(fs.H_cells[_mask(ps, "fe")], axis=1).max() < CURVE.H_values[1]   # предпосылка
    assert fs.converged
    assert np.linalg.norm(fs.H_cells - fl.H_cells) / np.linalg.norm(fl.H_cells) < 1e-10


def test_problem_without_steel_above_knee_equals_linear_solver():
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / 4)
    p = _problem([s], margin=2.0, h=R / 4)
    fn, fl = solve_nonlinear3d(p), solve_linear3d(p)
    assert fn.converged and fn.risk.n_demagnetized == 0
    assert np.linalg.norm(fn.H_cells - fl.H_cells) / np.linalg.norm(fl.H_cells) < 1e-10


# ----------------------------------------------------------------------- (3) насыщенная сталь
def test_saturated_steel_sphere_converges_second_order():
    H0 = 4.0e5
    h_in = brentq(lambda h: float(steel_B_of_H(CURVE, h)) + 2.0 * MU0 * h - 3.0 * MU0 * H0, 0.0, H0)
    B_in = float(steel_B_of_H(CURVE, h_in))
    assert B_in > 1.4                                     # предпосылка: колено кривой, насыщение
    errs, iters = [], []
    for k in (3, 6):
        s = GeoObject3D("s", "sphere", {"r": R}, SteelMaterial(CURVE), mesh_size=R / k)
        p = _problem([s], margin=4.0, h=R / k)
        f = solve_nonlinear3d(p, bc="dirichlet", applied_field=np.array([H0, 0.0, 0.0]))
        assert f.converged and f.residual < 1e-9
        errs.append(f.average(f.B_cells, _mask(p, "s"))[0] / B_in - 1.0)
        iters.append(f.n_iterations)
    assert _order(*errs) > 1.5                            # теория: 2
    assert abs(errs[1]) < 0.10                            # грубая проверка; точность задаёт порядок
    assert abs(iters[1] - iters[0]) <= 3 and max(iters) <= 20      # Ньютон: от сетки не зависит


# ----------------------------------------------------------------------- (4) магнит за коленом
@pytest.mark.slow
def test_magnet_sphere_beyond_knee_matches_load_line():
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    T = 175.0
    lo = float(mag.curve_at(T).H_values[0])
    h_ex = brentq(lambda h: float(mag.B_major_parallel(h, T)) + 2.0 * MU0 * h, lo, 0.0)
    assert float(mag.risk_margin(h_ex, T)) < 0.0          # предпосылка: точка за коленом
    loss_ex = mag.Br(T) - float(mag.effective_Br(h_ex, T))
    assert loss_ex > 0.01 * mag.Br(T)                     # и потеря заметная
    errs = []
    for k in (3, 6):
        s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / k)
        p = _problem([s], margin=2.0, h=R / k, T=T)
        hz = []
        for bc in ("neumann", "dirichlet"):
            f = solve_nonlinear3d(p, bc=bc, max_iter=200)
            assert f.converged
            hz.append(f.average(f.H_cells, _mask(p, "m"))[2])
            assert f.risk.n_demagnetized > 0 and float(f.risk.loss.max()) > 0.0
        errs.append(0.5 * (hz[0] + hz[1]) / h_ex - 1.0)
    assert _order(*errs) > 1.5
    assert abs(errs[1]) < 0.10


# ----------------------------------------------------------------------- (5) замыкатель
def test_keeper_reduces_demagnetizing_field():
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    cube = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 4 * MM}, mag,
                       magnet_dir="axial", mesh_size=1.5 * MM)
    plate = GeoObject3D("fe", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 2 * MM}, SteelMaterial(CURVE),
                        center=(0, 0, 3 * MM), mesh_size=1.5 * MM)
    res = {}
    for key, objs in (("alone", [cube]), ("keeper", [cube, plate])):
        p = _problem(objs, margin=2.0, h=1.5 * MM)
        f = solve_nonlinear3d(p)
        assert f.converged
        m = _mask(p, "m")
        res[key] = (f.average(f.H_cells, m)[2], f.average(f.B_cells, m)[2])
    assert res["keeper"][0] < 0.0 and abs(res["keeper"][0]) < abs(res["alone"][0])
    assert res["keeper"][1] > res["alone"][1]


# ----------------------------------------------------------------------- (6) история нагружения
# Шар N42SH при 100 °C: новый шар без поля стоит на −300 кА/м — до колена (−758) далеко. Внешнее
# поле −550 кА/м вдоль оси уводит его за колено (−801 кА/м, на 90 кА/м выше H_cJ), потеря
# 9,5 % B_r; после снятия поля — линия возврата, −271 кА/м, снова с запасом выше колена.
HIST_T = 100.0
HIST_H0 = np.array([0.0, 0.0, -550.0e3])


def test_removed_field_keeps_the_loss_and_reapplied_field_returns_the_same_state():
    # Тождества на одной сетке (погрешность сетки в них не входит):
    #  • поле сняли: каждая ячейка идёт вверх по своей линии возврата — сохранённая доля r и потеря
    #    те же, что под полем; сейчас за коленом нет ни одной ячейки, а магнит слабее нового;
    #  • то же поле снова: решение первого нагружения удовлетворяет закону с памятью (каждая ячейка
    #    ровно в точке переключения, где ветви сходятся), закон монотонный — решение единственное.
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / 3)
    p = _problem([s], margin=2.0, h=R / 3, T=HIST_T)
    m = _mask(p, "m")
    fresh = solve_nonlinear3d(p, bc="dirichlet")
    loaded = solve_nonlinear3d(p, bc="dirichlet", applied_field=HIST_H0)
    removed = solve_nonlinear3d(p, bc="dirichlet", retention=loaded.retention)
    again = solve_nonlinear3d(p, bc="dirichlet", applied_field=HIST_H0, retention=removed.retention)
    assert all(f.converged for f in (fresh, loaded, removed, again))
    assert fresh.risk.n_demagnetized == 0 and float(fresh.risk.loss.max()) < 1e-9     # предпосылки
    assert loaded.risk.n_demagnetized > 0 and float(loaded.risk.loss.max()) > 0.05 * mag.Br(HIST_T)
    assert removed.risk.n_demagnetized == 0
    assert np.array_equal(removed.retention, loaded.retention)
    assert np.allclose(removed.risk.loss, loaded.risk.loss, rtol=0.0, atol=1e-12)
    assert removed.average(removed.B_cells, m)[2] < fresh.average(fresh.B_cells, m)[2]
    assert np.linalg.norm(again.H_cells - loaded.H_cells) / np.linalg.norm(loaded.H_cells) < 1e-8
    # Доля r — липшицева функция поля: |Δr| ≤ L·|ΔH∥|, L = max (dB/dH − μ₀μ_rec)/B_r по таблице кривой.
    # Допуск берётся из фактической разницы полей, а не назначается (Л-86).
    curve = mag.curve_at(HIST_T)
    L_r = float((np.diff(curve.B_values) / np.diff(curve.H_values) - MU0 * mag.mu_rec).max()) / mag.Br(HIST_T)
    idx = loaded.risk.cell_indices
    dh = float(np.abs(again.risk.H_par - loaded.risk.H_par).max())
    assert float(np.abs(again.retention[idx] - loaded.retention[idx]).max()) <= L_r * dh + 1e-15


@pytest.mark.slow
def test_magnet_sphere_recoil_after_field_removal_matches_exact():
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    T, H0 = HIST_T, float(HIST_H0[2])
    lo = float(mag.curve_at(T).H_values[0])
    h_w = brentq(lambda h: float(mag.B_major_parallel(h, T)) + 2.0 * MU0 * h - 3.0 * MU0 * H0, lo, 0.0)
    br_w = float(mag.effective_Br(h_w, T))
    h_back = -br_w / (MU0 * (2.0 + mag.mu_rec))
    h_fresh = -mag.Br(T) / (MU0 * (2.0 + mag.mu_rec))
    assert float(mag.risk_margin(h_w, T)) < 0.0 and h_w > lo          # предпосылки: под полем за
    assert br_w < 0.95 * mag.Br(T)                                    # коленом, выше H_cJ, потеря > 5 %,
    assert float(mag.risk_margin(h_back, T)) > 0.0                    # после снятия — снова до колена
    errs_w, errs_back = [], []
    for k in (3, 6):
        s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / k)
        p = _problem([s], margin=2.0, h=R / k, T=T)
        m = _mask(p, "m")
        hw, hb = [], []
        for bc in ("neumann", "dirichlet"):
            loaded = solve_nonlinear3d(p, bc=bc, applied_field=HIST_H0)
            removed = solve_nonlinear3d(p, bc=bc, retention=loaded.retention)
            assert loaded.converged and removed.converged
            hw.append(loaded.average(loaded.H_cells, m)[2])
            hb.append(removed.average(removed.H_cells, m)[2])
        errs_w.append(0.5 * (hw[0] + hw[1]) / h_w - 1.0)
        errs_back.append(0.5 * (hb[0] + hb[1]) / h_back - 1.0)
    assert _order(*errs_w) > 1.5 and _order(*errs_back) > 1.5
    assert abs(errs_w[1]) < 0.10 and abs(errs_back[1]) < 0.10
    # потеря различима на фоне ошибки сетки: без неё поле после снятия было бы −300, а не −271 кА/м
    assert abs(errs_back[1]) < 0.5 * abs(h_fresh / h_back - 1.0)


def test_loss_survives_cooling_and_reheating_without_double_counting():
    # Л-100: история хранится долей r, а не наихудшим полем. Шар N42SH повреждают внешним полем при
    # 150 °C, поле снимают, шар остывает до 20 °C и снова нагревается до 150 °C. Тождества на одной сетке:
    #  • остывание не лечит и нагрев без поля не добавляет: доля r не меняется ни там, ни там;
    #  • после снятия поля все ячейки на своих линиях возврата — задача линейна, источник ∝ B_r(T),
    #    поэтому решение при 20 °C = решение при 150 °C × B_r(20)/B_r(150). Память «по худшему полю» при
    #    остывании вернула бы ячейкам полную ремнантность (колено уходит глубже) и это тождество сломала бы;
    #  • потеря при 20 °C — та же доля от своего B_r.
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    T_hot, T_cold = 150.0, 20.0
    # Поле — из модели, не на глаз: по линии возврата шара H_in = (3H₀ − B_r/μ₀)/(2 + μ_rec); берём H₀,
    # при котором H_in — середина между коленом и −H_cJ при 150 °C (за коленом, выше −H_cJ).
    h_in = 0.5 * (mag.knee_field(T_hot) - mag.Hcj(T_hot))
    h0 = (h_in * (2.0 + mag.mu_rec) + mag.Br(T_hot) / MU0) / 3.0
    s = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(mag), magnet_dir="axial", mesh_size=R / 3)
    hot = _problem([s], margin=2.0, h=R / 3, T=T_hot)
    cold = dataclasses.replace(hot, T=T_cold)                                  # та же сетка
    loaded = solve_nonlinear3d(hot, bc="dirichlet", applied_field=(0.0, 0.0, h0))
    removed = solve_nonlinear3d(hot, bc="dirichlet", retention=loaded.retention)
    cooled = solve_nonlinear3d(cold, bc="dirichlet", retention=removed.retention)
    reheated = solve_nonlinear3d(hot, bc="dirichlet", retention=cooled.retention)
    assert all(f.converged for f in (loaded, removed, cooled, reheated))
    assert loaded.risk.n_damaged > 0 and loaded.risk.n_beyond_hcj == 0        # предпосылки: потеря есть,
    assert removed.risk.n_demagnetized == 0                                    # ниже −H_cJ нет, после снятия — до колена
    assert np.array_equal(cooled.retention, removed.retention)
    assert np.array_equal(reheated.retention, removed.retention)
    assert cooled.risk.n_damaged == removed.risk.n_damaged
    idx = cooled.risk.cell_indices
    assert np.allclose(cooled.risk.loss, mag.Br(T_cold) * (1.0 - removed.retention[idx]), rtol=0.0, atol=1e-12)
    # после снятия поля задача линейна: метод Ньютона кончает за шаг, невязка 1e-9 от начальной — допуск с запасом
    ratio = mag.Br(T_cold) / mag.Br(T_hot)
    assert np.linalg.norm(cooled.B_cells - ratio * removed.B_cells) / np.linalg.norm(cooled.B_cells) < 1e-8
    assert np.linalg.norm(reheated.B_cells - removed.B_cells) / np.linalg.norm(removed.B_cells) < 1e-8


# ----------------------------------------------------------------------- (7) частичное размагничивание
def _rigid_cube_hz_on_axis(z):
    """H_z/M жёсткого куба (заряды ±M на гранях z = ±c) на оси, z в долях c: −(Ω_верх + Ω_низ)/(4π)."""
    def omega(d):                      # телесный угол квадрата 2c×2c с его оси на расстоянии d·c
        return 2.0 * math.pi if d == 0.0 else 4.0 * math.atan(1.0 / (d * math.sqrt(2.0 + d * d)))
    return -(omega(1.0 - z) + omega(1.0 + z)) / (4.0 * math.pi)


def test_cube_partial_demagnetization_starts_under_pole_faces():
    # Жёсткий куб: под полюсной гранью H_z ≈ −M/2 (своя грань) минус вклад дальней грани — в центре
    # грани −0,564·M; в центре куба −M/3; к рёбрам и углам граней поле слабеет (угол — −M/6).
    # Колено при нагреве поднимается быстрее, чем падает M (γ_Hc = 0,55 > α_Br = 0,115 %/°C):
    # |H_k|/M = 0,67 при 120 °C — за коленом ничего; 0,44 при 150 °C и 0,40 при 155 °C — за коленом
    # слой под полюсными гранями, на оси глубиной 0,25c и 0,4c (оценка по жёсткому кубу).
    mag = n42sh_magnet((0.0, 0.0, 1.0))
    temps = (120.0, 150.0, 155.0)
    kappa = {T: -mag.knee_field(T) * MU0 / mag.Br(T) for T in temps}             # |H_k|/M
    face, centre = -_rigid_cube_hz_on_axis(1.0), -_rigid_cube_hz_on_axis(0.0)
    assert kappa[120.0] > face and all(face > kappa[T] > centre for T in temps[1:])   # предпосылки
    a = 10 * MM
    cube = GeoObject3D("m", "box", {"lx": a, "ly": a, "lz": a}, MagnetMaterial(mag), magnet_dir="axial",
                       mesh_size=a / 10)
    p = _problem([cube], margin=2.0, h=a / 10)
    m = _mask(p, "m")
    x = p.mesh.cell_centroids()[m] / (a / 2)                                      # в долях c
    vol = p.mesh.cell_volumes()[m]
    rho2, z2 = (x[:, :2] ** 2).sum(axis=1), x[:, 2] ** 2
    risk = {}
    for T in temps:
        f = solve_nonlinear3d(dataclasses.replace(p, T=T))                       # та же сетка
        assert f.converged
        risk[T] = f.risk
    assert risk[120.0].n_demagnetized == 0 and float(risk[120.0].loss.max()) < 1e-9
    dvol = {T: float(vol[risk[T].demagnetized].sum()) for T in temps[1:]}
    lost = {T: float((risk[T].loss * vol).sum()) for T in temps[1:]}
    assert 0.0 < dvol[150.0] < dvol[155.0] < 0.5 * vol.sum()                     # частичное, растёт с T
    assert 0.0 < lost[150.0] < lost[155.0]
    for T in temps[1:]:
        d, w = risk[T].demagnetized, risk[T].loss * vol
        assert d[x[:, 2] > 0.0].any() and d[x[:, 2] < 0.0].any()                 # под обеими гранями
        # у граней: размагничивание, равномерное по объёму, дало бы 1; слой |z| > 0,6c — больше 2
        assert np.average(z2[d], weights=vol[d]) > 1.5 * np.average(z2, weights=vol)
        # к оси: слой постоянной глубины под всей гранью дал бы 1; к рёбрам поле слабее, слой тоньше
        assert np.average(rho2, weights=w) < 0.9 * np.average(rho2, weights=vol)
        assert not d[np.linalg.norm(x, axis=1) < 0.3].any()                      # центр цел


# ----------------------------------------------------------------------- (8) сопряжённые градиенты
def test_conjugate_gradient_newton_reaches_the_same_solution():
    # Касательная симметрична и положительно определена (законы монотонны), поэтому линейную задачу шага
    # Ньютона можно решать сопряжёнными градиентами (3D-6). Решение дискретной задачи единственно, оба пути
    # останавливаются по одной нелинейной невязке (10⁻⁹). Задача трудная для итерационного решателя:
    # сталь в насыщении (начальная μ_r ≈ 4·10³ против воздуха) и магнит за коленом при 155 °C.
    # Допуск по полю 10⁻⁶ — на три порядка грубее точности остановки, с запасом на обусловленность;
    # фактически совпадение около 10⁻¹³, как и ячейки за коленом — поштучно.
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    h = 1.5 * MM
    cube = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, mag, magnet_dir="axial",
                       mesh_size=h)
    fe = GeoObject3D("fe", "box", {"lx": 8 * MM, "ly": 6 * MM, "lz": 3 * MM}, SteelMaterial(CURVE),
                     center=(1 * MM, 0.5 * MM, 7 * MM), rotation=(0.1, 0.0, 0.2), mesh_size=h)
    p = _problem([cube, fe], margin=2.0, h=h, T=155.0)
    fd, fc = solve_nonlinear3d(p), solve_nonlinear3d(p, solver="cg")
    assert fd.converged and fc.converged and fd.risk.n_demagnetized > 0           # предпосылка: колено
    assert np.array_equal(fd.risk.demagnetized, fc.risk.demagnetized)
    assert np.linalg.norm(fc.H_cells - fd.H_cells) / np.linalg.norm(fd.H_cells) < 1e-6


def test_invalid_inputs_rejected():
    p = _problem([GeoObject3D("a", "box", {"lx": 2 * MM, "ly": 2 * MM, "lz": 2 * MM}, AIR)], margin=1.0, h=1 * MM)
    nc = p.mesh.n_cells
    bad_nan = np.ones(nc)
    bad_nan[0] = np.nan
    for kw in (dict(bc="robin"), dict(applied_field=(1.0, 2.0)), dict(retention=np.ones(nc + 1)),
               dict(retention=bad_nan), dict(retention=np.full(nc, -0.1)), dict(retention=np.full(nc, 1.1)),
               dict(retention=np.ones(nc), demag=False), dict(solver="lu")):
        with pytest.raises(ValueError):
            solve_nonlinear3d(p, **kw)
