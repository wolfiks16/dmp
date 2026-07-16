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
