import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

from magcore.fem2d.losses import copper_loss_sensitivity
from magcore.fem2d.mesh_generators import build_disk_tri_mesh
from magcore.fem2d.runaway import (
    couple_thermal_loss_picard,
    runaway_margin,
    runaway_threshold,
    solve_thermal_steady_with_feedback,
)
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import assemble_robin_multi


def _setup():
    disk = build_disk_tri_mesh(1.0, 16, 64)
    space = LagrangeP1Space2D(disk.mesh)
    k, h, T_amb = 1.0, 3.0, 20.0
    q0 = np.full(disk.mesh.n_cells, 5.0)
    return space, k, h, T_amb, q0


def test_runaway_threshold_governs_stability():
    # Ядро К6′ (линеаризованно): порог s_crit разделяет устойчивый тепловой баланс и разгон.
    space, k, h, T_amb, q0 = _setup()
    s_crit = runaway_threshold(space, k, h)
    assert s_crit > 0.0

    safe = couple_thermal_loss_picard(
        space, k, q0_cells=q0, loss_sensitivity=0.7 * s_crit, h=h, T_amb=T_amb, max_iter=500
    )
    runaway = couple_thermal_loss_picard(
        space, k, q0_cells=q0, loss_sensitivity=1.3 * s_crit, h=h, T_amb=T_amb, max_iter=500
    )

    assert safe.converged
    assert np.isfinite(safe.T).all()
    assert safe.T.max() > T_amb                       # тепловыделение греет выше окружающей

    assert not runaway.converged                      # разгон: охлаждение не догоняет потери
    assert runaway.norm_history[-1] > 1e4 * safe.norm_history[-1]


def test_safe_branch_matches_direct_solve():
    # Неподвижная точка Picard = прямое решение (A−s·M)T=f (та же консистентная масса).
    space, k, h, T_amb, q0 = _setup()
    s = 0.7 * runaway_threshold(space, k, h)
    picard = couple_thermal_loss_picard(
        space, k, q0_cells=q0, loss_sensitivity=s, h=h, T_amb=T_amb, max_iter=500
    )
    direct = solve_thermal_steady_with_feedback(
        space, k, q0_cells=q0, loss_sensitivity=s, h=h, T_amb=T_amb
    )
    assert picard.converged
    assert np.linalg.norm(picard.T - direct) < 1e-3


def test_temperature_amplification_grows_toward_threshold():
    # По мере s→s_crit устойчивая T усиливается (усиление обратной связью), но конечна.
    space, k, h, T_amb, q0 = _setup()
    s_crit = runaway_threshold(space, k, h)
    tmax = []
    for frac in (0.3, 0.6, 0.9):
        T = solve_thermal_steady_with_feedback(
            space, k, q0_cells=q0, loss_sensitivity=frac * s_crit, h=h, T_amb=T_amb
        )
        assert np.isfinite(T).all()
        tmax.append(T.max())
    assert tmax[0] < tmax[1] < tmax[2]                 # монотонное усиление к порогу
    assert tmax[0] > T_amb


# --------------------------------------------- аналитический оракул порога: диск с условием Робена
#
# Задача на собственные значения k·(−Δv) = λ·s(x)·v в круге радиуса R, −k·∂v/∂r = h·v при r = R (Bi = hR/k).
#  (а) s ≡ s₀ во всём круге: первая мода v = J₀(μr), k·μ² = λ·s₀; условие Робена k·μ·J₁(μR) = h·J₀(μR) ⇒
#      для j = μR трансцендентное  Bi·J₀(j) = j·J₁(j),  j ∈ (0, j₀,₁),  и порог однородной связи
#      s_crit = k·j²/R² (λ = 1);
#  (б) s = s₀ только в КОЛЬЦЕ a < r < R (медь у края), в круге r < a источника нет: там v гармонична и
#      регулярна ⇒ v = const и k·v'(a) = 0; в кольце v = P·J₀(μr) + Q·Y₀(μr) с v'(a) = 0 ⇒ P = Y₁(μa),
#      Q = −J₁(μa); μ — первый корень −k·v'(R) = h·v(R); λ = k·μ²/s₀.
# Сетка `build_disk_tri_mesh` ставит узлы на окружностях r_j = R·j/n, поэтому граница кольца a = R·m/n
# совпадает с рёбрами сетки; ошибка P1 — O(h²) (и вписанного многоугольника — тоже).

R_DISK, K_DISK, H_DISK = 0.05, 2.0, 60.0                  # Bi = 1,5


def _disk_root(bi):
    return brentq(lambda j: bi * j0(j) - j * j1(j), 1e-9, 2.404825557695773)


def _ring_mu(a, R=R_DISK, k=K_DISK, h=H_DISK):
    def f(mu):
        P, Q = y1(mu * a), -j1(mu * a)
        v = P * j0(mu * R) + Q * y0(mu * R)
        dv = -mu * (P * j1(mu * R) + Q * y1(mu * R))
        return -k * dv - h * v
    grid = np.linspace(1e-3, 200.0, 4001)
    vals = np.array([f(m) for m in grid])
    i = int(np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0][0])   # первый корень = наименьшее λ
    return brentq(f, grid[i], grid[i + 1])


def _ring_disk(n_rings, n_theta, a_frac=0.6):
    disk = build_disk_tri_mesh(R_DISK, n_rings, n_theta)
    space = LagrangeP1Space2D(disk.mesh)
    cent = disk.mesh.vertices[disk.mesh.cells].mean(axis=1)
    ring = np.hypot(cent[:, 0], cent[:, 1]) > a_frac * R_DISK
    return disk, space, ring


def test_uniform_threshold_matches_bessel_robin_disk():
    # (а): s_crit однородной обратной связи против k·j²/R², Bi·J₀(j) = j·J₁(j); порядок сходимости 2.
    j = _disk_root(H_DISK * R_DISK / K_DISK)
    exact = K_DISK * j * j / R_DISK ** 2
    errs = []
    for n in (10, 20, 40):
        disk = build_disk_tri_mesh(R_DISK, n, 4 * n)
        errs.append(abs(runaway_threshold(LagrangeP1Space2D(disk.mesh), K_DISK, H_DISK) / exact - 1.0))
    assert errs[-1] < 5e-4
    assert np.log2(errs[0] / errs[1]) == pytest.approx(2.0, abs=0.2)
    assert np.log2(errs[1] / errs[2]) == pytest.approx(2.0, abs=0.2)


def test_uniform_margin_is_threshold_over_s():
    # Частный случай: однородное s ⇒ λ_min = s_crit/s (та же задача, переписанная через множитель).
    disk, space, _ = _ring_disk(12, 48)
    s0 = 1.0e3
    lam = runaway_margin(space, K_DISK, np.full(disk.mesh.n_cells, s0), h=H_DISK)
    assert lam == pytest.approx(runaway_threshold(space, K_DISK, H_DISK) / s0, rel=1e-10)
    assert runaway_margin(space, K_DISK, np.zeros(disk.mesh.n_cells), h=H_DISK) == np.inf   # нет связи — нет разгона


def test_ring_localized_margin_matches_bessel_and_differs_from_whole_domain_mass():
    # (б): источник только в кольце a < r < R. Это то различие, ради которого порог переписан: прежняя формула
    # (масса по всей области) сравнивала s меди с s_crit всего сечения и здесь объявила бы РАЗГОН (s_crit/s ≈ 0,57),
    # а на деле запас есть (λ_min ≈ 1,04).
    a_frac, s0 = 0.6, 3.0e3
    exact = K_DISK * _ring_mu(a_frac * R_DISK) ** 2 / s0
    errs = []
    for n in (10, 20, 40):
        disk, space, ring = _ring_disk(n, 4 * n, a_frac)
        lam = runaway_margin(space, K_DISK, np.where(ring, s0, 0.0), h=H_DISK)
        errs.append(abs(lam / exact - 1.0))
    assert errs[-1] < 1e-3
    assert np.log2(errs[0] / errs[1]) == pytest.approx(2.0, abs=0.2)
    assert np.log2(errs[1] / errs[2]) == pytest.approx(2.0, abs=0.2)
    assert exact > 1.0 > runaway_threshold(space, K_DISK, H_DISK) / s0     # старая формула ошиблась бы в знаке вывода


@pytest.mark.parametrize("factor,expect_runaway", [(0.9, False), (1.1, True)])
def test_localized_margin_predicts_the_coupled_transient(factor, expect_runaway):
    # Порог по меди с ДИФФЕРЕНЦИРОВАННЫМИ ГУ Робена (полуокружности с разными h и T_ref, как расточка/ротор)
    # предсказывает исход самой связки `solve_coupled_magneto_thermal_transient` с теми же ГУ: ток меди
    # подбирается под долю порога, а не наоборот. Шаг неявного Эйлера с явным источником устойчив ⟺ λ_min > 1
    # при любом dt; dt большой, чтобы мода у порога менялась в 1/λ раз за шаг.
    from magcore.fem2d.coupled_transient import solve_coupled_magneto_thermal_transient

    disk, space, ring = _ring_disk(16, 64)
    mesh = disk.mesh
    edges = np.asarray(mesh.boundary_edges(), dtype=int)
    mid_y = mesh.vertices[edges].mean(axis=1)[:, 1]
    robin = assemble_robin_multi(space, [(edges[mid_y >= 0.0], 200.0, 40.0), (edges[mid_y < 0.0], 15.0, 20.0)])
    J_ref = 1.0e7
    lam_ref = runaway_margin(space, K_DISK, copper_loss_sensitivity(np.where(ring, J_ref, 0.0)), robin=robin)
    lam_uniform_h = runaway_margin(space, K_DISK, copper_loss_sensitivity(np.where(ring, J_ref, 0.0)), h=15.0)
    assert abs(lam_ref / lam_uniform_h - 1.0) > 0.1           # ГУ влияют на порог — подставлять «одно h» нельзя
    J = J_ref * np.sqrt(factor * lam_ref)                     # s ∝ J²: λ_min(J) = 1/factor
    assert runaway_margin(space, K_DISK, copper_loss_sensitivity(np.where(ring, J, 0.0)), robin=robin) == \
        pytest.approx(1.0 / factor, rel=1e-9)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=K_DISK, capacity_cells=3.0e6, h=0.0, robin=robin, T_amb=20.0, dt=1.0e6, n_steps=400,
        j_cells=np.where(ring, J, 0.0), T_cap=20.0 + 1.0e4,
    )
    assert res.runaway is expect_runaway
