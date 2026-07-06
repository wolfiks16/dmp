from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_mixed_coulomb_system
from magcore.femcore.boundary_conditions import (
    apply_zero_mixed_dirichlet_bc,
    find_mixed_boundary_dofs,
)
from magcore.femcore.mixed_problem import MixedCoulombProblem
from magcore.femcore.nonlinear import solve_nonlinear_mixed_picard
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _J_rot(scale: float = 1.0):
    # вихревой источник => B=curl A неоднороден по ячейкам (нелинейность «активна»)
    def J(x):
        return scale * np.array([-(x[1] - 0.5), (x[0] - 0.5), 0.0])
    return J


def _nonlinear_residual(mesh, vs, ss, nu_cells, J_fn, a, p) -> float:
    """
    Относительная невязка ‖A(ν)·x − b‖/‖b‖ системы, СОБРАННОЙ при заданной ν_cells,
    для x=(a,p). НЕЗАВИСИМЫЙ оракул фиксированной точки: если (a,p) действительно
    решает нелинейную систему при ν(|B|), невязка мала; иначе — велика.
    """
    A, b = assemble_mixed_coulomb_system(mesh, vs, ss, nu=nu_cells, J_fn=J_fn)
    vbnd, sbnd = find_mixed_boundary_dofs(vector_space=vs, scalar_space=ss)
    A_bc, b_bc = apply_zero_mixed_dirichlet_bc(
        A=A, b=b, vector_dofs=vbnd, scalar_dofs=sbnd, n_vector_dofs=vs.ndofs
    )
    x = np.concatenate([np.asarray(a), np.asarray(p)])
    return float(np.linalg.norm(A_bc @ x - b_bc)) / max(float(np.linalg.norm(b_bc)), 1e-30)


def test_per_cell_nu_matches_scalar() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    A1, b1 = assemble_mixed_coulomb_system(mesh, vs, ss, nu=2.5, J_fn=_J_rot())
    A2, b2 = assemble_mixed_coulomb_system(
        mesh, vs, ss, nu=np.full(mesh.n_cells, 2.5), J_fn=_J_rot()
    )
    assert np.allclose(A1, A2)
    assert np.allclose(b1, b2)


def test_picard_constant_nu_equals_linear_solve() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    nu0 = 3.0

    res = solve_nonlinear_mixed_picard(
        mesh, vs, ss,
        nu_of_B=lambda B: np.full(mesh.n_cells, nu0),
        J_fn=_J_rot(),
        nu_init=np.full(mesh.n_cells, nu0),
        tol=1e-10,
    )
    assert res.converged
    assert res.n_iterations <= 3

    lin = MixedCoulombProblem.from_mesh(mesh, nu=nu0, J_fn=_J_rot()).solve(
        compute_gauge_projection=False
    )
    assert np.allclose(res.a, lin.a, atol=1e-9)


def test_picard_nonlinear_converges_and_self_consistent() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    nu0, c = 2.0, 5.0

    def nu_of_B(B):
        s2 = np.sum(np.asarray(B) ** 2, axis=1)
        return nu0 * (1.0 + c * s2)  # монотонно растёт с |B| (saturation-like)

    res = solve_nonlinear_mixed_picard(
        mesh, vs, ss,
        nu_of_B=nu_of_B,
        J_fn=_J_rot(scale=2.0),
        nu_init=np.full(mesh.n_cells, nu0),
        tol=1e-8,
        max_iter=100,
    )
    assert res.converged
    # САМОСОГЛАСОВАННОСТЬ (невязочный оракул, НЕ тавтология): сошедшееся (a,p) РЕШАЕТ
    # систему, собранную при ν(|B_фин|) — невязка фиксированной точки мала.
    # (Прежняя проверка allclose(nu_cells, nu_of_B(B)) была истинна ПО ПОСТРОЕНИЮ:
    #  res.nu_cells = nu_of_B(B_cells) возвращается из решателя ⇒ ничего не проверяла.)
    resid = _nonlinear_residual(
        mesh, vs, ss, nu_of_B(res.B_cells), _J_rot(scale=2.0), res.a, res.p
    )
    assert resid < 1e-6
    # нелинейность реально активна: ν варьируется по ячейкам и выше стартовой
    assert res.nu_cells.max() > res.nu_cells.min() + 1e-9
    assert res.nu_cells.max() > nu0


def test_picard_nonlinear_differs_from_linear_guess() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    nu0, c = 2.0, 5.0

    def nu_of_B(B):
        s2 = np.sum(np.asarray(B) ** 2, axis=1)
        return nu0 * (1.0 + c * s2)

    res = solve_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=nu_of_B, J_fn=_J_rot(scale=2.0),
        nu_init=np.full(mesh.n_cells, nu0), tol=1e-8, max_iter=100,
    )
    lin = MixedCoulombProblem.from_mesh(mesh, nu=nu0, J_fn=_J_rot(scale=2.0)).solve(
        compute_gauge_projection=False
    )
    # нелинейное решение отличается от линейного (ν0)
    assert not np.allclose(res.a, lin.a, atol=1e-6)


def test_picard_with_steel_curve_runs_and_converges() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    curve = m270_35a_bh_curve()

    def nu_of_B(B):
        mag = np.linalg.norm(np.asarray(B), axis=1)
        return np.array([curve.nu_chord(float(b)) for b in mag])

    nu0 = curve.nu_initial
    res = solve_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=nu_of_B, J_fn=_J_rot(scale=1.0),
        nu_init=np.full(mesh.n_cells, nu0), tol=1e-7, max_iter=100, relaxation=1.0,
    )
    assert res.converged
    assert np.all(res.nu_cells > 0.0)
    # невязочный оракул (не тавтология): решение удовлетворяет системе при ν(|B|).
    resid = _nonlinear_residual(mesh, vs, ss, nu_of_B(res.B_cells), _J_rot(scale=1.0), res.a, res.p)
    assert resid < 1e-6


def test_picard_residual_discriminates_against_frozen_nu() -> None:
    """
    Дискриминирующий тест: невязочный оракул РЕАЛЬНО ловит неверную физику. Замороженное
    линейное решение (ν≡ν0) НЕ удовлетворяет нелинейной системе ⇒ его невязка при ν(|B|)
    ВЕЛИКА; сошедшееся нелинейное — мала. (Если бы оракул был «подогнан», он проходил бы
    и для неверного решения.)
    """
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    nu0, c = 2.0, 5.0

    def nu_of_B(B):
        s2 = np.sum(np.asarray(B) ** 2, axis=1)
        return nu0 * (1.0 + c * s2)

    J = _J_rot(scale=2.0)
    nl = solve_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=nu_of_B, J_fn=J,
        nu_init=np.full(mesh.n_cells, nu0), tol=1e-9, max_iter=100,
    )
    # замороженное линейное решение при ν0 (через тот же решатель, max_iter=2 — const ν).
    frozen = solve_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=lambda B: np.full(mesh.n_cells, nu0), J_fn=J,
        nu_init=np.full(mesh.n_cells, nu0), tol=1e-12, max_iter=3,
    )

    r_nl = _nonlinear_residual(mesh, vs, ss, nu_of_B(nl.B_cells), J, nl.a, nl.p)
    r_frozen = _nonlinear_residual(mesh, vs, ss, nu_of_B(frozen.B_cells), J, frozen.a, frozen.p)

    assert r_nl < 1e-6                  # верное решение — мала
    assert r_frozen > 1e-3              # неверное (замороженное) — заметно ненулевая
    assert r_nl < 1e-3 * r_frozen       # разделение ≥3 порядков (главный критерий)
