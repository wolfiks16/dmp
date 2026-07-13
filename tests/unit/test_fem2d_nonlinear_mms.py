import math

import numpy as np
import pytest

from magcore.fem2d.manufactured import manufactured_nonlinear_sine
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.post import l2_error
from magcore.fem2d.spaces import LagrangeP1Space2D


def _solve_nonlinear(n: int, nu0: float = 1.2, c: float = 0.4):
    mesh = build_structured_rectangle_tri_mesh(n, n)
    space = LagrangeP1Space2D(mesh)
    mms = manufactured_nonlinear_sine(nu0, c)
    nu_init = np.full(mesh.n_cells, nu0)          # старт с линейного приближения (c=0)
    # Под-релаксация ω=0.5: закон ν↑(|B|) даёт ОТРИЦАТЕЛЬНУЮ производную карты Picard
    # (ν↑→|B|↓→ν↓) ⇒ ω=1 осциллирует (та же находка, что для demag/стали в 3D).
    res = solve_nonlinear_2d_picard(
        space, mms.nu_of_B, nu_init=nu_init, j_fn=mms.J_fn,
        max_iter=100, tol=1e-9, relaxation=0.5, quadrature_order=5,
    )
    return space, mms, res


def test_nonlinear_picard_converges_and_recovers_exact():
    # Через ОБЩЕЕ ядро run_picard_fixed_point: 2D-backend восстанавливает точное A_z.
    space, mms, res = _solve_nonlinear(24)
    assert res.converged
    assert res.n_iterations >= 2                  # закон реально нелинеен (не одна итерация)
    err = l2_error(space, res.a, mms.A_exact, quadrature_order=5)
    assert err < 5e-3


def test_nonlinear_solution_is_self_consistent_residual_oracle():
    # Невязочный оракул (не тавтология): на сошедшемся решении ν(|B|)-система имеет
    # МАЛУЮ невязку, а система с ЗАМОРОЖЕННОЙ линейной ν₀ — на порядки большую.
    from magcore.fem2d.assembly import assemble_current_rhs, assemble_stiffness
    from magcore.fem2d.solver import apply_dirichlet

    space, mms, res = _solve_nonlinear(16)
    K_nl = assemble_stiffness(space, res.nu_cells)          # ν(|B|) сошедшегося решения
    f = assemble_current_rhs(space, mms.J_fn, quadrature_order=5)
    K_bc, f_bc = apply_dirichlet(K_nl, f, space.boundary_dofs(), 0.0)
    resid_nl = float(np.linalg.norm(K_bc @ res.a - f_bc))

    nu0 = res.nu_cells.min()                                # линейное приближение
    K_lin = assemble_stiffness(space, np.full(space.mesh.n_cells, nu0))
    K_lin_bc, _ = apply_dirichlet(K_lin, f, space.boundary_dofs(), 0.0)
    resid_lin = float(np.linalg.norm(K_lin_bc @ res.a - f_bc))

    assert resid_nl < 1e-8               # решение удовлетворяет НЕЛИНЕЙНОЙ системе
    assert resid_lin > 1e-2              # замороженная линейная ν₀ — крупная невязка (не тавтология)


def test_nonlinear_convergence_order():
    ns = [8, 16, 32]
    errs = []
    for n in ns:
        space, mms, res = _solve_nonlinear(n)
        assert res.converged
        errs.append(l2_error(space, res.a, mms.A_exact, quadrature_order=5))
    assert errs[0] > errs[1] > errs[2] > 0.0
    rate = math.log2(errs[1] / errs[2])
    assert rate == pytest.approx(2.0, abs=0.35)
