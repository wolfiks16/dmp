import math

import numpy as np
import pytest

from magcore.fem2d.manufactured import manufactured_nonlinear_sine
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.newton import solve_nonlinear_2d_newton
from magcore.fem2d.post import l2_error
from magcore.fem2d.spaces import LagrangeP1Space2D

# MMS-оракул метода Ньютона (закон ν=ν₀(1+c|B|²)): восстанавливает точное решение, порядок
# сходимости 2, и — ГЛАВНОЕ — число итераций мало и НЕ растёт с измельчением сетки (нет
# зависимости от релаксации, в отличие от хордового Пикара).


def _nu_and_dnu(nu0, c):
    def f(B):
        s = B[:, 0] ** 2 + B[:, 1] ** 2
        return nu0 * (1.0 + c * s), np.full(s.shape[0], nu0 * c)   # ν, dν/d|B|²
    return f


def _solve(n, nu0=1.2, c=0.4):
    mesh = build_structured_rectangle_tri_mesh(n, n)
    space = LagrangeP1Space2D(mesh)
    mms = manufactured_nonlinear_sine(nu0, c)
    res = solve_nonlinear_2d_newton(
        space, _nu_and_dnu(nu0, c), nu_init=np.full(mesh.n_cells, nu0),
        j_fn=mms.J_fn, max_iter=30, tol=1e-10,
    )
    return space, mms, res


def test_newton_recovers_exact_in_few_iterations():
    space, mms, res = _solve(24)
    assert res.converged
    assert res.n_iterations <= 8                       # квадратично, без подбора релаксации
    assert l2_error(space, res.a, mms.A_exact, quadrature_order=5) < 5e-3


def test_newton_convergence_order_two():
    ns, errs = [8, 16, 32], []
    for n in ns:
        space, mms, res = _solve(n)
        assert res.converged
        errs.append(l2_error(space, res.a, mms.A_exact, quadrature_order=5))
    assert errs[0] > errs[1] > errs[2] > 0.0
    assert math.log2(errs[1] / errs[2]) == pytest.approx(2.0, abs=0.35)


def test_newton_iterations_mesh_independent():
    # Суть фикса: измельчение сетки НЕ раздувает число итераций (у Пикара — раздувает и
    # требует уменьшать ω). Ньютон сходится за единицы итераций на всех сетках.
    its = [_solve(n)[2].n_iterations for n in (8, 16, 32, 48)]
    assert all(i <= 8 for i in its), its


def test_newton_residual_quadratic_drop():
    # Квадратичная сходимость: невязка падает резко (r_{k+1} ≪ r_k) у хвоста.
    _, _, res = _solve(20)
    h = [x for x in res.rel_change_history if x > 0]
    assert res.converged and len(h) >= 3
    assert h[-1] < 1e-6 * h[0]                          # многопорядковое падение невязки


# ------------------------------------------------- тёплый старт: критерий от начала не зависит

def _warm(space, mms, a_init, nu0=1.2, c=0.4, **kw):
    return solve_nonlinear_2d_newton(
        space, _nu_and_dnu(nu0, c), nu_init=np.full(space.mesh.n_cells, nu0),
        j_fn=mms.J_fn, max_iter=30, tol=1e-10, a_init=a_init, **kw,
    )


def test_warm_start_measures_the_residual_against_the_cold_start_scale():
    # Шаг по времени стартует с решения прошлого шага. Отнеси решатель невязку к невязке СВОЕГО старта,
    # почти готовый старт ужесточал бы допуск в разы (и «сошлось» значило бы разное). Поэтому R₀ — всегда
    # невязка холодного старта; она идёт первой в истории, и «последнее/первое» сравнимо с холодным решением
    # и с `evaluate_nonlinear_2d_newton` (та же дискретная задача, Л-80).
    from magcore.fem2d.newton import evaluate_nonlinear_2d_newton

    space, mms, cold = _solve(16)
    again = _warm(space, mms, cold.a)
    assert again.rel_change_history[0] == cold.rel_change_history[0]           # тот же масштаб, бит в бит
    assert again.converged and again.n_iterations == 1                       # готовое решение — сразу
    bnd = np.asarray(space.boundary_dofs(), dtype=int)
    inner = np.setdiff1d(np.arange(space.ndofs), bnd)
    a_pert = cold.a.copy()
    a_pert[inner] += 0.05 * np.abs(cold.a).max() * np.random.default_rng(3).standard_normal(inner.size)
    a_pert[bnd] = 7.0                                                        # мусор в узлах Дирихле
    warm = _warm(space, mms, a_pert)
    assert warm.converged and warm.rel_change_history[0] == cold.rel_change_history[0]
    assert np.all(warm.a[bnd] == 0.0)                                        # граничные значения — заданные
    ev = evaluate_nonlinear_2d_newton(space, _nu_and_dnu(1.2, 0.4), warm.a, j_fn=mms.J_fn, tol=1e-10)
    assert ev.converged
    assert ev.rel_change_history[1] / ev.rel_change_history[0] == warm.rel_change_history[-1] / warm.rel_change_history[0]


def test_warm_start_is_refused_with_a_stateful_magnet_source_or_a_wrong_shape():
    space, mms, cold = _solve(8)
    with pytest.raises(ValueError, match="a_init"):
        _warm(space, mms, cold.a, magnetization=lambda B, H, nu: np.zeros((space.mesh.n_cells, 2)))
    with pytest.raises(ValueError, match="a_init"):
        _warm(space, mms, cold.a[:-1])
    with pytest.raises(ValueError, match="a_init"):
        _warm(space, mms, np.full(space.ndofs, np.nan))
