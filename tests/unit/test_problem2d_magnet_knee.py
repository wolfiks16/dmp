import numpy as np
import pytest

pytest.importorskip("gmsh")

from magcore.constants import MU0  # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet  # noqa: E402
from magcore.fem2d.assembly import p1_cell_geometry  # noqa: E402
from magcore.fem2d.model.materials import Air, MagnetMaterial, SteelMaterial  # noqa: E402
from magcore.fem2d.model.object_geometry import GeoObject, build_object_problem  # noqa: E402
from magcore.fem2d.model.problem import _reluctivity_newton, solve_problem2d  # noqa: E402
from magcore.fem2d.newton import solve_nonlinear_2d_newton  # noqa: E402
from magcore.fem2d.spaces import LagrangeP1Space2D  # noqa: E402
from magcore.domain.steel_curves import m270_35a_bh_curve  # noqa: E402
from magcore.hybrid.magnet_demag import MagnetDemagPolicy  # noqa: E402

# МАГНИТ ЗА КОЛЕНОМ в статическом 2D. Раньше колено учитывалось внешним циклом по источнику с
# постоянной релаксацией (Л-93): за коленом главная кривая в 7–30 раз круче линии возврата, множитель
# цикла λ = −(μ_d − μ_rec)/(μ_rec + P) выходит за границу устойчивости, и расчёт не сходился —
# магнит 10×4 мм под стальной пластиной при 140–160 °C не сошёлся за 60 итераций, доля площади за
# коленом получалась 0,77 вместо 1,00. Теперь магнит — в касательной Ньютона (MagnetLaw2D).
# Оракул: круглый магнит радиуса R, намагниченный поперёк, в круглой области радиуса L с A_z = 0 на
# границе. Поле внутри однородно при любом законе вдоль оси, а из непрерывности A и H_θ на r = R и
# A(L) = 0 следует нагрузочная прямая B = −μ0·P·H, P = (L² − R²)/(L² + R²) (при L → ∞ P = 1).

MAGNET = n42sh_magnet((1.0, 0.0, 0.0))


def _circle_problem(h, *, R=5.0e-3, L=25.0e-3, T=170.0):
    magnet = GeoObject(name="магнит", kind="circle", params={"cx": 0.0, "cy": 0.0, "r": R},
                       material=MagnetMaterial(MAGNET), magnet_dir=(0.0, 1.0), mesh_size=h, priority=10)
    domain = GeoObject(name="domain", kind="circle", params={"cx": 0.0, "cy": 0.0, "r": L},
                       material=Air(), mesh_size=4.0 * h)
    return build_object_problem([magnet], domain, default_mesh_size=h, T=T)


def _exact_operating_point(T, P):
    """Пересечение нагрузочной прямой B = −μ0·P·H с главной кривой: f(H) монотонна ⇒ корень один."""
    lo, hi = -MAGNET.Hcj(T), 0.0
    f = lambda H: float(MAGNET.B_major_parallel(H, T) + MU0 * P * H)  # noqa: E731
    assert f(lo) < 0.0 < f(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0.0 else (lo, mid)
    H = 0.5 * (lo + hi)
    return H, float(MAGNET.B_major_parallel(H, T))


def _magnet_mean(solution, values):
    idx = solution.risk.cell_indices
    S = p1_cell_geometry(solution.problem.mesh)[2][idx]
    return float(np.sum(S * values) / S.sum())


def test_round_magnet_past_the_knee_matches_the_exact_load_line_and_converges():
    """Оракул: рабочая точка совпадает с точной, ошибка убывает при сгущении, итераций единицы."""
    R, L, T = 5.0e-3, 25.0e-3, 170.0
    P = (L ** 2 - R ** 2) / (L ** 2 + R ** 2)
    H_exact, B_exact = _exact_operating_point(T, P)
    assert H_exact < MAGNET.knee_field(T)                      # случай именно за коленом
    err_H, err_B = [], []
    for h in (1.2e-3, 0.6e-3):
        sol = solve_problem2d(_circle_problem(h, R=R, L=L, T=T), max_iter=40)
        hist = np.array(sol.field.rel_change_history)
        assert sol.converged and hist[-1] / hist[0] < 1e-6
        assert sol.field.n_iterations <= 6                      # число итераций не зависит от сетки
        err_H.append(abs(_magnet_mean(sol, sol.risk.H_par) / H_exact - 1.0))
        err_B.append(abs(_magnet_mean(sol, sol.field.B_cells[sol.risk.cell_indices, 1]) / B_exact - 1.0))
    # Сгущение вдвое уменьшает обе ошибки. По индукции ошибка БОЛЬШЕ, чем по полю, и это физика,
    # а не дефект: за коленом кривая крутая (dB = μ0·μ_d·dH, μ_d ≈ 25), поэтому малая ошибка поля
    # даёт заметную ошибку индукции. Проверяются и убывание, и уровень на мелкой сетке.
    assert err_H[0] < 3e-3 and err_H[1] < err_H[0] / 2.0
    assert err_B[0] < 3e-2 and err_B[1] < err_B[0] / 1.8
    assert err_B[1] < 1.2e-2


def test_solution_lies_on_the_branch_law_in_every_magnet_cell():
    """В сошедшемся решении B∥ каждой ячейки лежит на той же кривой, по которой строилась касательная."""
    sol = solve_problem2d(_circle_problem(1.2e-3), max_iter=40)
    idx = sol.risk.cell_indices
    b_par = sol.field.B_cells[idx, 1]                           # ось магнита — +Y
    b_law, _ = MAGNET.branch_parallel(sol.risk.H_par, sol.problem.T, np.ones(idx.size))
    assert np.abs(b_law - b_par).max() < 1e-12


def _plate_problem(T, mesh_mm=3.0):
    """Случай из интерфейса: магнит 10×4 мм, намагниченный по +Y, под стальной пластиной 14×2 мм."""
    mm = 1.0e-3
    magnet = GeoObject(name="магнит", kind="rect",
                       params={"cx": 0.0, "cy": 0.0, "w": 10 * mm, "h": 4 * mm, "angle": 0.0},
                       material=MagnetMaterial(MAGNET), magnet_dir=(0.0, 1.0), mesh_size=mesh_mm * mm, priority=10)
    plate = GeoObject(name="пластина", kind="rect",
                      params={"cx": 0.0, "cy": 4 * mm, "w": 14 * mm, "h": 2 * mm, "angle": 0.0},
                      material=SteelMaterial(m270_35a_bh_curve()), mesh_size=mesh_mm * mm, priority=10)
    domain = GeoObject(name="domain", kind="rect", params={"cx": 0.0, "cy": 2 * mm, "w": 90 * mm, "h": 90 * mm},
                       material=Air(), mesh_size=8.0 * mesh_mm * mm)
    return build_object_problem([magnet, plate], domain, default_mesh_size=mesh_mm * mm, T=T)


@pytest.mark.parametrize("T", [140.0, 160.0])
def test_plate_case_beyond_the_knee_converges(T):
    """Тот самый случай, который раньше не сходился за 60 итераций; весь магнит за коленом."""
    sol = solve_problem2d(_plate_problem(T), max_iter=60)
    hist = np.array(sol.field.rel_change_history)
    assert sol.converged and hist[-1] / hist[0] < 1e-6
    assert sol.field.n_iterations <= 8
    assert sol.risk.n_demagnetized == sol.risk.cell_indices.size


def test_same_solution_as_the_previous_scheme_where_that_one_converged():
    """Физика не изменилась: там, где прежний путь (источник с релаксацией) сходился, решение то же."""
    T = 120.0
    prob = _plate_problem(T)
    new = solve_problem2d(prob, max_iter=40)
    space = LagrangeP1Space2D(prob.mesh)
    nu_and_dnu, nu_init = _reluctivity_newton(prob)
    policy = MagnetDemagPolicy(prob.magnet(), prob.magnet_mask(), T=prob.T, n_cells=prob.mesh.n_cells,
                               axis=prob.magnet_axis, relaxation=0.1)
    old = solve_nonlinear_2d_newton(space, nu_and_dnu, nu_init=nu_init, magnetization=policy,
                                    dirichlet_dofs=prob.dirichlet_dofs, dirichlet_values=prob.dirichlet_values,
                                    max_iter=300, tol=1e-9)
    assert old.converged and new.converged
    assert new.field.n_iterations < old.n_iterations / 4        # прежний путь тратит десятки итераций
    # Допуск не машинный: прежняя схема доходит до своего порога по невязке (1e-9), а не до округления.
    assert np.abs(old.B_cells - new.field.B_cells).max() < 1e-6


def test_convergence_flag_is_set_only_by_the_residual():
    """Л-108: оборванный расчёт честно помечается «не сошлось», а невязка относится к выданному решению."""
    sol = solve_problem2d(_plate_problem(150.0), max_iter=1)
    hist = np.array(sol.field.rel_change_history)
    assert not sol.converged
    assert hist[-1] / hist[0] > 1e-6                            # невязка большая — флага сходимости нет
    full = solve_problem2d(_plate_problem(150.0), max_iter=40)
    assert full.converged and np.array(full.field.rel_change_history)[-1] / hist[0] < 1e-6


def test_loading_history_changes_both_the_field_and_the_risk_map():
    """История нагружения (доля r) входит и в решение, и в карту: прежняя потеря не исчезает (Л-100)."""
    prob = _plate_problem(120.0)
    idx = np.where(prob.magnet_mask())[0]
    r = np.ones(prob.mesh.n_cells)
    r[idx[: idx.size // 2]] = 0.8                               # половина магнита уже потеряла 20 %
    fresh = solve_problem2d(prob, max_iter=40)
    aged = solve_problem2d(prob, max_iter=40, retention=r)
    assert fresh.converged and aged.converged
    assert np.abs(aged.field.B_cells - fresh.field.B_cells).max() > 1e-2      # поле слабее — потеря учтена
    assert aged.risk.n_damaged > fresh.risk.n_damaged
    assert float(np.mean(aged.risk.retention)) < float(np.mean(fresh.risk.retention))


def test_loading_history_of_the_previous_scheme_is_refused_in_newton():
    """`track_worst_point` работает только в прежней схеме: в Ньютоне — понятная ошибка, а не тишина."""
    with pytest.raises(ValueError, match="track_worst_point"):
        solve_problem2d(_plate_problem(120.0), max_iter=5, track_worst_point=True)

