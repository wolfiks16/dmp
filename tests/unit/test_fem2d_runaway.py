import numpy as np

from magcore.fem2d.mesh_generators import build_disk_tri_mesh
from magcore.fem2d.runaway import (
    couple_thermal_loss_picard,
    runaway_threshold,
    solve_thermal_steady_with_feedback,
)
from magcore.fem2d.spaces import LagrangeP1Space2D


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
